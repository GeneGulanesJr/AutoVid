"""Organization stage: tags, quality flags, filename suggestions, search.

Tags come from the temporal map the VLM already produced (activity, objects,
environment, camera motion) — no extra model calls. Every segment gets a
deterministic, human-readable suggested filename that keeps the source
provenance; renames of extracted clips go through a journal table so any
filesystem change can be undone.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .config import Settings
from .errors import AutoVidError
from .ingest import get_asset

logger = logging.getLogger(__name__)

MAX_SLUG_WORDS = 6


@dataclass
class OrganizeResult:
    asset_id: str
    segments_tagged: int = 0
    renames_applied: int = 0
    notes: list[str] = field(default_factory=list)


def slugify(text: str, max_words: int = MAX_SLUG_WORDS) -> str:
    """Description → lowercase hyphenated filename fragment."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words[:max_words]) if words else "clip"


def build_tags(analysis: dict) -> list[str]:
    """Merged, normalized tags from a temporal-map entry."""
    tags: list[str] = []
    for key in ("activity", "environment"):
        value = analysis.get(key)
        if value:
            tags.append(str(value).strip().lower())
    for obj in analysis.get("objects") or []:
        tag = str(obj).strip().lower()
        if tag:
            tags.append(tag)
    motion = analysis.get("camera_motion")
    if motion and motion != "static":
        tags.append(f"cam-{motion}")
    seen: set[str] = set()
    ordered = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def build_quality_flags(verdict: str, decision_confidence: float | None,
                        quality_level: int | None, uncertainty: float | None) -> dict:
    flags: dict[str, bool | float] = {}
    if verdict != "accept":
        flags["needs_review"] = True
    if decision_confidence is not None and decision_confidence < 0.5:
        flags["low_decision_confidence"] = True
    if quality_level is not None and quality_level <= 1:
        flags["weak_quality"] = True
    if uncertainty is not None and uncertainty >= 0.6:
        flags["uncertain_understanding"] = True
    if quality_level is not None:
        flags["quality_score"] = quality_level / 3
    return flags


def suggested_name(asset_id: str, start_s: float, end_s: float, description: str,
                   tags: list[str]) -> str:
    """``<slug>__<asset8>__<start>-<end>.mp4`` — readable + provenance-preserving."""
    lead = slugify(description) or (tags[0] if tags else "clip")
    return f"{lead}__{asset_id[2:10]}__{start_s:.3f}-{end_s:.3f}.mp4"


def organize_asset(conn: sqlite3.Connection, settings: Settings, asset_id: str) -> OrganizeResult:
    """Tag every segment, suggest names, prepare the searchable catalog."""
    asset = get_asset(conn, asset_id)
    if asset is None:
        raise AutoVidError(f"Unknown asset {asset_id}")
    result = OrganizeResult(asset_id=asset_id)

    db.job_start(conn, asset_id, "organize")
    try:
        windows = [
            (float(r["window_start"]), float(r["window_end"]), json.loads(r["response_json"]))
            for r in conn.execute(
                "SELECT window_start, window_end, response_json FROM analyses "
                "WHERE asset_id = ?",
                (asset_id,),
            ).fetchall()
        ]
        segments = conn.execute(
            "SELECT * FROM segments WHERE asset_id = ?", (asset_id,)
        ).fetchall()

        for seg in segments:
            # a segment (logical scene) contains every window it spans
            members = [
                data for (ws, we, data) in windows
                if ws >= seg["start_s"] - 0.01 and we <= seg["end_s"] + 0.01
            ]
            tags: list[str] = []
            uncertainty = 0.0
            for data in members:
                for tag in build_tags(data):
                    if tag not in tags:
                        tags.append(tag)
                u = data.get("uncertainty")
                if isinstance(u, (int, float)):
                    uncertainty = max(uncertainty, float(u))
            try:
                quality_level = json.loads(seg["quality_json"] or "{}").get("quality_level")
            except json.JSONDecodeError:
                quality_level = None
            quality = build_quality_flags(
                seg["verdict"] or "accept", seg["decision_confidence"],
                quality_level, uncertainty or None,
            )
            name = suggested_name(
                asset_id, seg["start_s"], seg["end_s"], seg["description"] or "", tags
            )
            conn.execute(
                """UPDATE segments SET tags_json = ?, quality_json = ?, suggested_name = ?
                   WHERE id = ?""",
                (json.dumps(tags), json.dumps(quality), name, seg["id"]),
            )
            result.segments_tagged += 1
        conn.commit()

        conn.execute(
            "UPDATE assets SET status = 'organized', updated_at = datetime('now') WHERE id = ?",
            (asset_id,),
        )
        conn.commit()
        db.job_finish(conn, asset_id, "organize")
    except Exception as e:  # noqa: BLE001
        db.job_finish(conn, asset_id, "organize", error=str(e))
        conn.execute(
            "UPDATE assets SET status = 'error', error = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (f"organize: {e}", asset_id),
        )
        conn.commit()
        raise
    return result


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(conn: sqlite3.Connection, query: str, *, limit: int = 50) -> list[sqlite3.Row]:
    """Segments matching the query across description, tags, asset name.

    Simple LIKE-based matching over a bounded catalog; an FTS index is the
    upgrade path once a catalog outgrows it.
    """
    pattern = f"%{query.strip().lower()}%"
    return conn.execute(
        """SELECT s.*, a.original_name FROM segments s
           JOIN assets a ON a.id = s.asset_id
           WHERE lower(s.description) LIKE ?
              OR lower(s.tags_json) LIKE ?
              OR lower(a.original_name) LIKE ?
           ORDER BY s.highlight_score DESC
           LIMIT ?""",
        (pattern, pattern, pattern, limit),
    ).fetchall()


# ---------------------------------------------------------------------------
# Recoverable renames
# ---------------------------------------------------------------------------


def apply_rename(conn: sqlite3.Connection, segment_id: int) -> int:
    """Rename an extracted clip to its suggested name; journal the mapping."""
    seg = conn.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()
    if seg is None:
        raise AutoVidError(f"Unknown segment {segment_id}")
    if not seg["clip_path"]:
        raise AutoVidError(f"Segment {segment_id} has no extracted clip to rename")
    if seg["suggested_name"] is None:
        raise AutoVidError(f"Segment {segment_id} has no suggested name (run organize first)")
    old = Path(seg["clip_path"])
    if not old.exists():
        raise AutoVidError(f"Clip file missing: {old}")
    new = old.with_name(seg["suggested_name"])
    if new.exists():
        raise AutoVidError(f"Target name already exists: {new}")
    old.rename(new)
    conn.execute("UPDATE segments SET clip_path = ? WHERE id = ?", (str(new), segment_id))
    cur = conn.execute(
        "INSERT INTO renames (asset_id, old_path, new_path) VALUES (?, ?, ?)",
        (seg["asset_id"], str(old), str(new)),
    )
    conn.commit()
    return int(cur.lastrowid)


def undo_rename(conn: sqlite3.Connection, rename_id: int) -> None:
    """Restore a journaled rename (only if it has not been reverted yet)."""
    row = conn.execute("SELECT * FROM renames WHERE id = ?", (rename_id,)).fetchone()
    if row is None:
        raise AutoVidError(f"Unknown rename {rename_id}")
    if row["reverted_at"] is not None:
        raise AutoVidError(f"Rename {rename_id} was already reverted")
    new = Path(row["new_path"])
    old = Path(row["old_path"])
    if new.exists():
        new.rename(old)
    elif not old.exists():
        raise AutoVidError(f"Neither {new} nor {old} exists; cannot undo rename {rename_id}")
    conn.execute(
        "UPDATE renames SET reverted_at = datetime('now') WHERE id = ?", (rename_id,)
    )
    conn.execute(
        "UPDATE segments SET clip_path = ? WHERE clip_path = ?", (str(old), str(new))
    )
    conn.commit()
