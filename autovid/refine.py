"""Refinement stage — hierarchical temporal scene segmentation, phases 2–5.

Phase 1 (coarse fixed-time windows, one VLM description each) lives in
:mod:`autovid.analyze`. This stage turns that window stream into a scene
catalog:

- **Phase 2 — continuity**: the decision layer judges every adjacent window
  pair (descriptions plus end-of-left/start-of-right frames) → merge, split,
  or verify.
- **Phase 3 — refinement**: ``needs_verification`` pairs with a nearby
  scene-change detection cut get frame-level boundary verification around
  that cut; a confirmed cut both splits the pair and refines the shared edge
  to the cut timestamp.
- **Phase 4 — merging**: adjacent windows judged the same continuous scene
  merge into one logical scene. Distinct scenes stay separate even when they
  look similar. Logical scenes are written to the ``scenes`` table
  (``source='vlm-merge'``, with real spans) and the analysis windows map to
  them by timestamp containment.
- **Phase 5 — catalog**: every logical scene whose windows reach the
  highlight threshold becomes a candidate segment (description aggregated
  from member windows, highlight = max); segment acceptance and clip
  extraction follow as before.

All timestamps remain on the original source timeline.
"""
from __future__ import annotations

import bisect
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .config import Settings
from .decisions import DecisionClient
from .errors import AutoVidError
from .ffmpeg_io.clips import clip_filename, extract_clip
from .ffmpeg_io.frames import frame_at
from .ingest import get_asset
from .media import b64_jpeg

logger = logging.getLogger(__name__)

SHIFT_STEP_S = 0.25
BOUNDARY_CONTEXT_S = 0.3   # frame offset on each side of a candidate cut
REFINE_WINDOW_S = 1.5      # a detector cut within this of a shared window edge is "near"


@dataclass
class RefineResult:
    asset_id: str
    windows: int = 0
    scenes: int = 0
    boundaries_verified: int = 0
    boundaries_refined: int = 0
    segments_accepted: int = 0
    segments_flagged: int = 0
    segments_rejected: int = 0
    clips_extracted: int = 0
    notes: list[str] = field(default_factory=list)


def _boundary_frames(
    conn: sqlite3.Connection,
    settings: Settings,
    asset_id: str,
    source: Path,
    ts: float,
    duration_s: float,
    nearby_frame=None,
) -> tuple[list[str], list[str]]:
    """(before_b64, after_b64) around a candidate cut.

    ``nearby_frame(t) -> path | None`` lets the caller serve a probe from an
    already-extracted grid/dense frame within ~±0.3 s instead of spawning one
    ffmpeg process per probe frame on Windows.
    """
    before_ts = [max(ts - BOUNDARY_CONTEXT_S, 0.0), max(ts - BOUNDARY_CONTEXT_S / 2, 0.0)]
    after_ts = [min(ts + BOUNDARY_CONTEXT_S / 2, duration_s - 0.05),
                min(ts + BOUNDARY_CONTEXT_S, duration_s - 0.05)]
    frame_dir = settings.assets_dir / asset_id / "probe"
    before, after = [], []
    for tag, times in (("b", before_ts), ("a", after_ts)):
        out = before if tag == "b" else after
        for t in dict.fromkeys(times):
            path = nearby_frame(t) if nearby_frame is not None else None
            if path is None:
                path = frame_at(source, t, frame_dir / f"{tag}_{ts:.3f}_{t:.3f}.jpg", width=512)[1]
            out.append(b64_jpeg(path))
    return before, after


def _analysis_windows(conn: sqlite3.Connection, asset_id: str) -> list[dict]:
    """Temporal-map windows with parsed VLM data, in time order."""
    wins = []
    for r in conn.execute(
        """SELECT window_start, window_end, response_json FROM analyses
           WHERE asset_id = ? ORDER BY window_start""",
        (asset_id,),
    ).fetchall():
        wins.append({
            "start": float(r["window_start"]),
            "end": float(r["window_end"]),
            "data": json.loads(r["response_json"]),
        })
    return wins


def _scene_evidence(
    conn: sqlite3.Connection, asset_id: str, start: float, end: float, count: int = 4
) -> list[str]:
    """Up to ``count`` evidence frames (base64) spanning [start, end)."""
    frames = conn.execute(
        """SELECT ts, path FROM frames
           WHERE asset_id = ? AND kind IN ('grid', 'dense')
             AND ts >= ? AND ts < ? ORDER BY ts""",
        (asset_id, start, end),
    ).fetchall()
    if not frames:
        return []
    step = max(1, len(frames) // count)
    return [b64_jpeg(f["path"]) for f in frames[::step][:count]]


def segment_scenes(
    conn: sqlite3.Connection,
    settings: Settings,
    asset_id: str,
    source: Path,
    duration_s: float,
    decisions: DecisionClient,
) -> tuple[list[dict], dict]:
    """Phases 2–4: continuity → refinement → merging. Returns logical scenes.

    Each scene dict carries ``start``, ``end``, aggregated ``description``,
    ``tags``, and ``highlight`` (max over member windows). Scenes are also
    written to the ``scenes`` table with ``source='vlm-merge'``.
    """
    wins = _analysis_windows(conn, asset_id)
    stats = {"windows": len(wins), "verified": 0, "refined": 0}
    if not wins:
        return [], stats

    # One pass over the asset's frames (sorted by ts); the continuity loop
    # bisects this instead of issuing two SQL lookups per window pair.
    frame_ts: list[float] = []
    frame_path: dict[float, str] = {}
    for r in conn.execute(
        "SELECT ts, path FROM frames WHERE asset_id = ? AND kind IN ('grid', 'dense') "
        "ORDER BY ts",
        (asset_id,),
    ).fetchall():
        ts = float(r["ts"])
        frame_ts.append(ts)
        frame_path[ts] = r["path"]

    def frame_before(t: float) -> str | None:
        i = bisect.bisect_left(frame_ts, t)
        return frame_path[frame_ts[i - 1]] if i else None

    def frame_at_or_after(t: float) -> str | None:
        i = bisect.bisect_left(frame_ts, t)
        return frame_path[frame_ts[i]] if i < len(frame_ts) else None

    def nearest_frame(t: float, tol: float = BOUNDARY_CONTEXT_S) -> str | None:
        """Closest extracted frame within ``tol`` of ``t`` (boundary probes)."""
        i = bisect.bisect_left(frame_ts, t)
        best: int | None = None
        for j in (i - 1, i):
            if 0 <= j < len(frame_ts) and abs(frame_ts[j] - t) <= tol:
                if best is None or abs(frame_ts[j] - t) < abs(frame_ts[best] - t):
                    best = j
        return frame_path[frame_ts[best]] if best is not None else None

    cuts = sorted(
        float(r["start_s"])
        for r in conn.execute(
            "SELECT start_s FROM scenes WHERE asset_id = ? AND source LIKE 'scdet%'",
            (asset_id,),
        ).fetchall()
    )

    def near_cut(boundary: float) -> tuple[bool, float | None]:
        if not cuts:
            return False, None
        i = bisect.bisect_left(cuts, boundary)
        nearest = min(cuts[max(0, i - 1):i + 1], key=lambda c: abs(boundary - c))
        return abs(boundary - nearest) <= REFINE_WINDOW_S, nearest

    def summarize(w: dict) -> dict:
        d = w["data"]
        return {
            "description": (d.get("description") or "")[:200],
            "activity": d.get("activity"),
            "environment": d.get("environment"),
            "tags": d.get("objects", []),
        }

    groups: list[list[dict]] = [[wins[0]]]
    refined_edges: dict[float, float] = {}  # window boundary → refined cut ts
    for i in range(1, len(wins)):
        left, right = wins[i - 1], wins[i]
        boundary = right["start"]
        is_near, cut_ts = near_cut(boundary)
        context = json.dumps({
            "boundary_ts": boundary,
            "left_window": summarize(left),
            "right_window": summarize(right),
            "detector_cut_near_boundary": is_near,
        })
        images = None
        left_path = frame_before(left["end"])
        right_path = frame_at_or_after(right["start"])
        if left_path and right_path:
            images = [b64_jpeg(left_path), b64_jpeg(right_path)]

        verdict = decisions.decide_continuity(
            asset_id, context, images, near_cut=is_near
        )
        if verdict.relation == "same_scene":
            groups[-1].append(right)
            continue
        if verdict.relation == "different_scene":
            groups.append([right])
            continue

        # needs_verification → phase 3: inspect around the boundary / nearby cut
        stats["verified"] += 1
        if is_near:
            before, after = _boundary_frames(
                conn, settings, asset_id, source, cut_ts, duration_s,
                nearby_frame=nearest_frame,
            )
            bv = decisions.decide_boundary(
                asset_id,
                json.dumps({"candidate_ts": round(cut_ts, 3),
                            "detector": "ffmpeg scdet",
                            "between_windows": True}),
                before, after,
            )
            if bv.verdict in ("boundary", "shift"):
                delta = (SHIFT_STEP_S if bv.position_hint == "after"
                         else -SHIFT_STEP_S if bv.position_hint == "before" else 0.0)
                refined = min(max(cut_ts + delta, 0.05), duration_s - 0.05)
                refined_edges[boundary] = round(refined, 3)
                groups.append([right])
                stats["refined"] += 1
            else:  # not_boundary → the scene continues across this edge
                groups[-1].append(right)
        elif verdict.change_level >= 2:
            # no detector evidence, but the descriptions clearly changed
            groups.append([right])
        else:
            groups[-1].append(right)

    # Phase 4 — materialize logical scenes with refined edges where verified.
    scenes: list[dict] = []
    for gi, group in enumerate(groups):
        start = group[0]["start"]
        end = group[-1]["end"]
        if gi > 0:  # edge between group gi-1 and gi
            edge = group[0]["start"]
            if edge in refined_edges:
                start = refined_edges[edge]
        if gi < len(groups) - 1:
            edge = group[-1]["end"]
            if edge in refined_edges:
                end = refined_edges[edge]
        descriptions = [
            w["data"]["description"] for w in group
            if w["data"].get("description") and not w["data"].get("error")
        ]
        tags: list[str] = []
        seen: set[str] = set()
        for w in group:
            for o in w["data"].get("objects") or []:
                t = str(o).strip().lower()
                if t and t not in seen:
                    seen.add(t)
                    tags.append(t)
        highlight = max(
            (float(w["data"].get("highlight_score") or 0.0) for w in group
             if not w["data"].get("error")),
            default=0.0,
        )
        scenes.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "description": descriptions[0] if descriptions else "",
            "extra_windows": max(0, len(descriptions) - 1),
            "tags": tags,
            "highlight": highlight,
            "windows": [w for w in group],
        })

    conn.execute("DELETE FROM scenes WHERE asset_id = ? AND source = 'vlm-merge'", (asset_id,))
    for s in scenes:
        conn.execute(
            """INSERT INTO scenes (asset_id, start_s, end_s, score, source)
               VALUES (?, ?, ?, ?, 'vlm-merge')""",
            (asset_id, s["start"], s["end"], s["highlight"]),
        )
    conn.commit()
    return scenes, stats


def catalog_scenes(
    conn: sqlite3.Connection,
    settings: Settings,
    asset_id: str,
    source: Path,
    scenes: list[dict],
    decisions: DecisionClient,
) -> tuple[list[int], int, int, int]:
    """Phase 5 — one candidate segment per highlight scene; returns
    (segment ids, accepted, flagged, rejected)."""
    ids: list[int] = []
    accepted = flagged = rejected = 0
    for scene in scenes:
        start, end = scene["start"], scene["end"]
        duration = end - start
        description = scene["description"]
        if scene["extra_windows"]:
            description += f" (+{scene['extra_windows']} more windows, same scene)"
        all_unanalyzed = all(w["data"].get("error") for w in scene["windows"])
        if all_unanalyzed:
            # Cold start (VLM unavailable/absent): the scene still becomes a
            # review-queue entry so a human can rescue it.
            cur = conn.execute(
                """INSERT INTO segments (asset_id, start_s, end_s, description,
                                         verdict, review_status)
                   VALUES (?, ?, ?, ?, 'needs_evidence', 'pending')""",
                (asset_id, start, end,
                 "(unanalyzed scene — VLM unavailable; review manually)"),
            )
            ids.append(cur.lastrowid)
            flagged += 1
            continue
        # Windows/scenes below this VLM highlight score never become candidate
        # clips (Settings.highlight_threshold; lower it for sparse/abstract footage).
        if scene["highlight"] < settings.highlight_threshold:
            continue

        summary = json.dumps({
            "description": description,
            "tags": scene["tags"],
            "duration_s": round(duration, 3),
            "window_count": len(scene["windows"]),
        })
        evidence = _scene_evidence(conn, asset_id, start, end)
        if not evidence:
            continue
        verdict = decisions.decide_segment(
            asset_id, summary, frames=evidence, duration_s=duration
        )
        review = "auto"
        if verdict.verdict in ("needs_evidence", "flag_review"):
            review = "pending"
        elif verdict.verdict == "reject":
            review = "rejected"
        cur = conn.execute(
            """INSERT INTO segments (asset_id, start_s, end_s, description,
                                     highlight_score, verdict, decision_confidence,
                                     review_status, quality_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                asset_id, start, end, description,
                scene["highlight"], verdict.verdict, verdict.confidence, review,
                json.dumps({"quality_level": verdict.quality_level}),
            ),
        )
        ids.append(cur.lastrowid)
        if verdict.verdict == "accept":
            accepted += 1
        elif verdict.verdict == "reject":
            rejected += 1
        else:
            flagged += 1
    conn.commit()
    return ids, accepted, flagged, rejected


def extract_accepted(
    conn: sqlite3.Connection, settings: Settings, asset_id: str, source: Path
) -> int:
    """Extract clips for accepted and human-approved segments; provenance kept."""
    rows = conn.execute(
        "SELECT id, start_s, end_s FROM segments "
        "WHERE asset_id = ? AND clip_path IS NULL "
        "AND (verdict = 'accept' OR review_status = 'approved')",
        (asset_id,),
    ).fetchall()
    clips_dir = settings.assets_dir / asset_id / "clips"
    count = 0
    for row in rows:
        out = clips_dir / clip_filename(asset_id, row["start_s"], row["end_s"])
        extract_clip(source, out, start_s=row["start_s"], end_s=row["end_s"])
        conn.execute("UPDATE segments SET clip_path = ? WHERE id = ?", (str(out), row["id"]))
        count += 1
    conn.commit()
    return count


def cut_asset(
    conn: sqlite3.Connection,
    settings: Settings,
    asset_id: str,
    *,
    decisions: DecisionClient,
) -> RefineResult:
    """Full cut stage for one analyzed asset: phases 2–5 + extraction."""
    asset = get_asset(conn, asset_id)
    if asset is None:
        raise AutoVidError(f"Unknown asset {asset_id}")
    result = RefineResult(asset_id=asset_id)

    db.job_start(conn, asset_id, "cut")
    try:
        source = Path(asset["path"])
        if not source.exists():
            raise AutoVidError(f"Source file missing: {source}")
        duration = float(asset["duration_s"])

        scenes, stats = segment_scenes(
            conn, settings, asset_id, source, duration, decisions
        )
        result.windows = stats["windows"]
        result.scenes = len(scenes)
        result.boundaries_verified = stats["verified"]
        result.boundaries_refined = stats["refined"]

        ids, accepted, flagged, rejected = catalog_scenes(
            conn, settings, asset_id, source, scenes, decisions
        )
        result.segments_accepted = accepted
        result.segments_flagged = flagged
        result.segments_rejected = rejected

        result.clips_extracted = extract_accepted(conn, settings, asset_id, source)

        conn.execute(
            "UPDATE assets SET status = 'cut', updated_at = datetime('now') WHERE id = ?",
            (asset_id,),
        )
        conn.commit()
        db.job_finish(conn, asset_id, "cut")
    except Exception as e:  # noqa: BLE001
        db.job_finish(conn, asset_id, "cut", error=str(e))
        conn.execute(
            "UPDATE assets SET status = 'error', error = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (f"cut: {e}", asset_id),
        )
        conn.commit()
        raise
    return result
