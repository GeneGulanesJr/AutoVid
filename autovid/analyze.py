"""Analysis stage: windows → D1 region decisions → VLM descriptions.

The temporal map of an asset is its set of ``analyses`` rows: per window
(between scene-change boundaries), what the cloud VLM understood, how
confident it was, and whether the decision layer flagged it for review.

Adaptive sampling happens here: each window's stats go to the decision layer
(decision_video_region); ``densify`` extracts extra frames at a tighter
interval and ``inspect_more`` adds two frames around the highest-motion spot
before the (single) VLM call, so expensive cloud calls see exactly the frames
the cheap local model asked for.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .config import Settings
from .decisions import DecisionClient, RegionStats
from .errors import AutoVidError
from .ffmpeg_io.frames import extract_periodic_frames, frame_at
from .ingest import get_asset
from .media import b64_jpeg
from .vlm import VLMClient, VLMUnavailable

logger = logging.getLogger(__name__)

MAX_FRAMES_PER_CALL = 10


@dataclass
class AnalyzeResult:
    asset_id: str
    windows: int = 0
    dense_frames: int = 0
    flagged: int = 0
    unparsed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    notes: list[str] = field(default_factory=list)


def build_windows(duration_s: float, *, coarse_window_s: float = 60.0) -> list[tuple[float, float]]:
    """Phase 1 — coarse segmentation: fixed-duration windows over the file.

    Structure is time-based, not scene-cut-based: scene-change hints are
    recorded separately (sampling) and used later, in the cut stage, when the
    decision layer verifies continuity between adjacent windows. The final
    window absorbs the remainder (it may be shorter than ``coarse_window_s``).
    """
    if coarse_window_s <= 0:
        coarse_window_s = 60.0
    edges: list[float] = [0.0]
    t = coarse_window_s
    while t < duration_s - 0.01:
        edges.append(round(t, 3))
        t += coarse_window_s
    edges.append(round(duration_s, 3))
    return list(zip(edges, edges[1:]))


def _frames_in(
    conn: sqlite3.Connection, asset_id: str, start: float, end: float, kind: str
) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT ts, path, motion_score FROM frames
           WHERE asset_id = ? AND kind = ? AND ts >= ? AND ts < ?
           ORDER BY ts""",
        (asset_id, kind, start, end),
    ).fetchall()


def _window_motion(frames: list[sqlite3.Row]) -> tuple[float, float]:
    motions = [f["motion_score"] or 0.0 for f in frames[1:]] or [0.0]
    mean = sum(motions) / len(motions)
    return mean, max(motions)


def _add_dense_frames(
    conn: sqlite3.Connection,
    settings: Settings,
    asset_id: str,
    source: Path,
    start: float,
    end: float,
    interval_s: float,
) -> int:
    """Extract kind='dense' frames for the span; returns count actually added."""
    span = max(end - start, 0.0)
    if span < interval_s / 2:
        return 0
    dense_dir = settings.assets_dir / asset_id / "dense" / f"{start:.3f}"
    existing = {round(r["ts"], 3) for r in _frames_in(conn, asset_id, start, end, "dense")}
    added = 0
    # Seek-scoped extraction: decode only this window's span, not the whole
    # source (which previously cost one full decode per densified window).
    for ts, path in extract_periodic_frames(
        source, dense_dir, interval_s=interval_s, width=512, prefix=f"d{start:.3f}",
        start_s=start, end_s=end,
    ):
        if start <= ts < end and round(ts, 3) not in existing:
            conn.execute(
                """INSERT OR IGNORE INTO frames (asset_id, ts, kind, path)
                   VALUES (?, ?, 'dense', ?)""",
                (asset_id, round(ts, 3), str(path)),
            )
            added += 1
    conn.commit()
    return added


def _select_for_call(rows: list[sqlite3.Row], cap: int = MAX_FRAMES_PER_CALL) -> list[sqlite3.Row]:
    """Evenly downselect frames so a VLM call stays bounded."""
    if len(rows) <= cap:
        return rows
    idx = [round(i * (len(rows) - 1) / (cap - 1)) for i in range(cap)]
    return [rows[i] for i in dict.fromkeys(idx)]


def analyze_asset(
    conn: sqlite3.Connection,
    settings: Settings,
    asset_id: str,
    *,
    vlm: VLMClient,
    decisions: DecisionClient,
) -> AnalyzeResult:
    """Analyze one sampled asset: windows, adaptive frames, VLM map. Resumable."""
    asset = get_asset(conn, asset_id)
    if asset is None:
        raise AutoVidError(f"Unknown asset {asset_id}")
    result = AnalyzeResult(asset_id=asset_id)

    db.job_start(conn, asset_id, "analyze")
    try:
        source = Path(asset["path"])
        if not source.exists():
            raise AutoVidError(f"Source file missing: {source}")
        windows = build_windows(
            float(asset["duration_s"]), coarse_window_s=settings.coarse_window_s
        )

        previous_description: str | None = None
        for start, end in windows:
            result.windows += 1
            frames = list(_frames_in(conn, asset_id, start, end, "grid"))
            if not frames:
                # Short/odd window with no grid frame: pin one at its start.
                frame_dir = settings.assets_dir / asset_id / "grid"
                ts, path = frame_at(source, start, frame_dir / f"grid_{start:.3f}.jpg", width=512)
                conn.execute(
                    "INSERT OR IGNORE INTO frames (asset_id, ts, kind, path) VALUES (?, ?, 'grid', ?)",
                    (asset_id, round(ts, 3), str(path)),
                )
                conn.commit()
                frames = _frames_in(conn, asset_id, start, end, "grid")

            scene_changes = conn.execute(
                "SELECT COUNT(*) AS n FROM scenes WHERE asset_id = ? AND start_s > ? "
                "AND start_s < ? AND source = 'scdet'",
                (asset_id, start, end),
            ).fetchone()["n"]
            mean_motion, max_motion = _window_motion(frames)

            action = decisions.decide_region(
                RegionStats(
                    asset_id=asset_id,
                    region_id=f"{start:.3f}-{end:.3f}",
                    start_s=start,
                    end_s=end,
                    scene_changes=scene_changes,
                    mean_motion=round(mean_motion, 4),
                    max_motion=round(max_motion, 4),
                    current_interval_s=settings.grid_interval_s,
                    vlm_uncertainty=None,
                )
            )

            if action.action == "densify" and action.suggested_interval_s:
                result.dense_frames += _add_dense_frames(
                    conn, settings, asset_id, source, start, end, action.suggested_interval_s
                )
                frames = list(_frames_in(conn, asset_id, start, end, "dense")) or frames
            elif action.action == "inspect_more" and len(frames) > 1:
                hottest = max(frames[1:], key=lambda r: r["motion_score"] or 0.0)
                half = settings.grid_interval_s / 2
                for offset in (-half, half):
                    ts = min(max(hottest["ts"] + offset, start), end - 0.05)
                    path = (
                        settings.assets_dir / asset_id / "dense"
                        / f"insp_{start:.3f}_{ts:.3f}.jpg"
                    )
                    p_path = frame_at(source, ts, path, width=512)
                    conn.execute(
                        "INSERT OR IGNORE INTO frames (asset_id, ts, kind, path) "
                        "VALUES (?, ?, 'dense', ?)",
                        (asset_id, round(ts, 3), str(p_path)),
                    )
                    result.dense_frames += 1
                conn.commit()
                frames = list(_frames_in(conn, asset_id, start, end, "dense")) or frames

            analysis = None
            error: str | None = None
            try:
                call_frames = _select_for_call(frames)
                analysis = vlm.describe_window(
                    frames=[(r["ts"], b64_jpeg(r["path"])) for r in call_frames],
                    previous_description=previous_description,
                )
                previous_description = analysis.description
            except VLMUnavailable as e:
                error = str(e)
                result.unparsed += 1
                result.notes.append(f"[{start:.1f}-{end:.1f}s] VLM unavailable: {e}")

            for entry in vlm.usage:
                result.prompt_tokens += entry["prompt_tokens"]
                result.completion_tokens += entry["completion_tokens"]
            vlm.usage.clear()

            uncertainty = analysis.uncertainty if analysis else 1.0
            flagged = (
                action.action == "flag_review"
                or analysis is None
                or uncertainty >= 0.6
            )
            if flagged:
                result.flagged += 1

            response = analysis.model_dump() if analysis else {"error": error}
            conn.execute(
                """INSERT INTO analyses (asset_id, window_start, window_end, provider,
                                         model, response_json, prompt_tokens,
                                         completion_tokens, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(asset_id, window_start) DO UPDATE SET
                       window_end = excluded.window_end,
                       provider = excluded.provider,
                       model = excluded.model,
                       response_json = excluded.response_json,
                       prompt_tokens = excluded.prompt_tokens,
                       completion_tokens = excluded.completion_tokens,
                       confidence = excluded.confidence""",
                (
                    asset_id, start, end,
                    settings.vlm_model if analysis else "none",
                    settings.vlm_model,
                    json.dumps({
                        **response,
                        "decision_action": action.action,
                        "flagged": flagged,
                        "engine": action.engine,
                    }),
                    0, 0,
                    round(1.0 - uncertainty, 3),
                ),
            )
            conn.commit()

        conn.execute(
            "UPDATE assets SET status = 'analyzed', error = NULL, updated_at = datetime('now') "
            "WHERE id = ?",
            (asset_id,),
        )
        conn.commit()
        db.job_finish(conn, asset_id, "analyze")
    except Exception as e:  # noqa: BLE001
        db.job_finish(conn, asset_id, "analyze", error=str(e))
        conn.execute(
            "UPDATE assets SET status = 'error', error = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (f"analyze: {e}", asset_id),
        )
        conn.commit()
        raise
    return result
