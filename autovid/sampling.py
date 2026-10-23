"""Coarse sampling: scene boundaries + fixed-interval grid frames.

The baseline policy is: ffmpeg scene-change detection for boundaries plus a
fixed grid of frames at ``grid_interval_s``. The decision layer (M3+) decides
which regions deserve denser sampling on top of this baseline.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import db
from .config import Settings
from .errors import AutoVidError
from .ffmpeg_io.frames import extract_grid_and_scene_changes
from .ingest import get_asset

logger = logging.getLogger(__name__)

GRID_WIDTH = 512  # analysis grid frames feed the VLM later — keep them legible
_MOTION_SIZE = (64, 64)


@dataclass
class SamplingResult:
    asset_id: str
    scene_changes: int = 0
    grid_frames: int = 0
    warnings: list[str] = field(default_factory=list)


def _motion_score(prev: np.ndarray, cur: np.ndarray) -> float:
    """Mean absolute grayscale difference, 0..1, on a small common raster."""
    return float(np.mean(np.abs(cur - prev)))


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize(_MOTION_SIZE)
    return np.asarray(img, dtype=np.float32) / 255.0


def sample_asset(
    conn: sqlite3.Connection, settings: Settings, asset_id: str
) -> SamplingResult:
    """Run baseline sampling for one asset (idempotent: replaces prior rows)."""
    asset = get_asset(conn, asset_id)
    if asset is None:
        raise AutoVidError(f"Unknown asset {asset_id}")
    result = SamplingResult(asset_id=asset_id)

    db.job_start(conn, asset_id, "sampling")
    try:
        source = Path(asset["path"])
        if not source.exists():
            raise AutoVidError(f"Source file missing: {source}")

        # One decode serves both the grid frames and the scene-change
        # detections (previously two separate full-file decode passes).
        grid_dir = settings.assets_dir / asset_id / "grid"
        grid, changes = extract_grid_and_scene_changes(
            source, grid_dir,
            interval_s=settings.grid_interval_s, width=GRID_WIDTH, prefix="grid",
            scene_threshold=settings.scene_threshold,
        )

        conn.execute("DELETE FROM scenes WHERE asset_id = ?", (asset_id,))
        conn.executemany(
            """INSERT INTO scenes (asset_id, start_s, score, source)
               VALUES (?, ?, ?, 'scdet')""",
            [(asset_id, round(c.ts, 3), c.score) for c in changes],
        )
        result.scene_changes = len(changes)

        conn.execute("DELETE FROM frames WHERE asset_id = ? AND kind = 'grid'", (asset_id,))
        prev_gray: np.ndarray | None = None
        rows = []
        for ts, path in grid:
            gray = _load_gray(path)
            motion = 0.0 if prev_gray is None else _motion_score(prev_gray, gray)
            prev_gray = gray
            rows.append((asset_id, round(ts, 3), "grid", str(path), None, round(motion, 4)))
        conn.executemany(
            """INSERT INTO frames (asset_id, ts, kind, path, scene_score, motion_score)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        result.grid_frames = len(rows)

        conn.execute(
            """UPDATE assets SET status = 'sampled', updated_at = datetime('now')
               WHERE id = ?""",
            (asset_id,),
        )
        conn.commit()
        db.job_finish(conn, asset_id, "sampling")
    except Exception as e:  # noqa: BLE001 — record and re-raise for the caller
        db.job_finish(conn, asset_id, "sampling", error=str(e))
        conn.execute(
            "UPDATE assets SET status = 'error', error = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (f"sampling: {e}", asset_id),
        )
        conn.commit()
        raise
    return result


def pending_assets(conn: sqlite3.Connection, status: str = "ingested") -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM assets WHERE status = ? ORDER BY created_at", (status,)
    ).fetchall()
