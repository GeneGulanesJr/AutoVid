"""Folder ingestion: discovery, dedupe, metadata, thumbnails.

Originals are never modified — every derived artifact lives under the catalog
data dir keyed by the asset id (``a_`` + first 16 hex of the content sha256).
Re-running ingest skips unchanged files (same path + size) and never re-hashes
known content; identical content found at a second path is recorded as a
duplicate on the original asset row.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .config import Settings
from .errors import AutoVidError
from .ffmpeg_io import probe as ffprobe
from .ffmpeg_io.frames import extract_periodic_frames

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
    ".mpg", ".mpeg", ".ts", ".wmv", ".flv", ".3gp",
}


def asset_id_for(sha256: str) -> str:
    return f"a_{sha256[:16]}"


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan_folder(folder: str | Path, *, recursive: bool = True) -> list[Path]:
    """Video files only, sorted for deterministic ingest order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise AutoVidError(f"Not a folder: {folder}")
    it: Iterator[Path] = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def get_asset(conn: sqlite3.Connection, asset_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()


def ensure_thumbs(
    conn: sqlite3.Connection, settings: Settings, asset_id: str, source: Path
) -> int:
    """Contact-sheet thumbnails at ``thumb_interval_s``; returns frame count."""
    asset = get_asset(conn, asset_id)
    if asset is None:
        raise AutoVidError(f"Unknown asset {asset_id}")
    if not float(asset["duration_s"]) > 0:
        return 0

    thumbs_dir = settings.assets_dir / asset_id / "thumbs"
    thumbs = extract_periodic_frames(
        source, thumbs_dir, interval_s=settings.thumb_interval_s,
        width=settings.thumb_width,
    )

    conn.execute("DELETE FROM frames WHERE asset_id = ? AND kind = 'thumb'", (asset_id,))
    conn.executemany(
        """INSERT INTO frames (asset_id, ts, kind, path) VALUES (?, ?, 'thumb', ?)""",
        [(asset_id, round(ts, 3), str(path)) for ts, path in thumbs],
    )
    if thumbs:
        conn.execute(
            "UPDATE assets SET poster_path = ? WHERE id = ?",
            (str(thumbs[0][1]), asset_id),
        )
    conn.commit()
    return len(thumbs)


def ingest_file(conn: sqlite3.Connection, settings: Settings, path: Path) -> tuple[str, str]:
    """Ingest one file. Returns (action, asset_id): added | duplicate | skipped."""
    path = path.resolve()
    size = path.stat().st_size

    existing = conn.execute(
        "SELECT id, size_bytes FROM assets WHERE path = ?", (str(path),)
    ).fetchone()
    if existing is not None and existing["size_bytes"] == size:
        return "skipped", existing["id"]

    sha = sha256_of(path)
    asset_id = asset_id_for(sha)

    dupe = get_asset(conn, asset_id)
    if dupe is not None:
        dupes = json.loads(dupe["dup_paths_json"])
        dupes.append(str(path))
        conn.execute(
            "UPDATE assets SET dup_paths_json = ?, updated_at = datetime('now') WHERE id = ?",
            (json.dumps(dupes), asset_id),
        )
        conn.commit()
        return "duplicate", asset_id

    info = ffprobe.probe(path)
    conn.execute(
        """INSERT INTO assets (id, sha256, path, original_path, original_name,
                               duration_s, width, height, fps, has_audio,
                               size_bytes, ffprobe_json, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ingested')""",
        (
            asset_id, sha, str(path), str(path), path.name,
            info.duration_s, info.width, info.height, info.fps, int(info.has_audio),
            info.size_bytes, json.dumps(info.raw),
        ),
    )
    conn.commit()
    return "added", asset_id


def ingest_folder(
    conn: sqlite3.Connection,
    settings: Settings,
    folder: str | Path,
    *,
    recursive: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> IngestReport:
    """Ingest every supported video under ``folder`` (idempotent, resumable)."""
    report = IngestReport()
    files = scan_folder(folder, recursive=recursive)
    total = len(files)
    for i, path in enumerate(files, 1):
        if progress is not None:
            progress(str(path), i, total)
        try:
            action, asset_id = ingest_file(conn, settings, path)
        except Exception as e:  # noqa: BLE001 — one bad file must not stop the batch
            logger.exception("Ingest failed for %s", path)
            report.failed.append(f"{path}: {e}")
            continue
        field_name = {"duplicate": "duplicates"}.get(action, action)
        getattr(report, field_name).append(str(path))
        if action != "added":
            continue

        db.job_start(conn, asset_id, "thumbs")
        try:
            ensure_thumbs(conn, settings, asset_id, path)
            db.job_finish(conn, asset_id, "thumbs")
        except Exception as e:  # noqa: BLE001
            db.job_finish(conn, asset_id, "thumbs", error=str(e))
            logger.exception("Thumbnail generation failed for %s", path)
            conn.execute(
                "UPDATE assets SET status = 'error', error = ? WHERE id = ?",
                (f"thumbs: {e}", asset_id),
            )
            conn.commit()
    conn.commit()
    return report
