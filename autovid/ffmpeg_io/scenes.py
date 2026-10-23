"""Scene-change detection via ffmpeg's ``select=gt(scene,T)`` + metadata print.

This is the conventional baseline boundary detector; the decision layer (M3+)
verifies/refines these boundaries rather than replacing the mechanism.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..errors import AutoVidError
from .bins import ffmpeg_path

# metadata=print prefixes every line with "[Parsed_metadata_N @ addr] ", so the
# scene_score line never directly follows pts_time — allow bounded junk between.
_PTS_TIME_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
_SCORE_RE = re.compile(r"lavfi\.scene_score=(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class SceneChange:
    ts: float
    score: float


def parse_scene_pairs(stderr: str) -> list[tuple[float, float]]:
    """``(pts_time, scene_score)`` pairs from ``metadata=print`` log output.

    Pairs are matched line-by-line: a ``pts_time`` line arms a pending
    timestamp that the next ``lavfi.scene_score`` line completes (both layouts
    — same line or adjacent lines — are handled). Lines from other filters
    sharing the log (e.g. a showinfo grid branch) never arm or pair, so this
    is safe when scene detection shares a decode with frame extraction.
    """
    pairs: list[tuple[float, float]] = []
    pending_ts: float | None = None
    for line in stderr.splitlines():
        if "Parsed_showinfo" in line:
            continue
        ts_match = _PTS_TIME_RE.search(line)
        score_match = _SCORE_RE.search(line)
        if ts_match:
            pending_ts = float(ts_match.group(1))
            if score_match:
                pairs.append((pending_ts, float(score_match.group(1))))
                pending_ts = None
            continue
        if score_match and pending_ts is not None:
            pairs.append((pending_ts, float(score_match.group(1))))
            pending_ts = None
    return pairs


def merge_detections(
    found: list[SceneChange], min_gap_s: float = 1.0
) -> list[SceneChange]:
    """Detections closer than ``min_gap_s`` merged, keeping the highest score."""
    ordered = sorted(found, key=lambda c: c.ts)
    merged: list[SceneChange] = []
    for change in ordered:
        if merged and change.ts - merged[-1].ts < min_gap_s:
            if change.score > merged[-1].score:
                merged[-1] = change
            continue
        merged.append(change)
    return merged


def detect_scene_changes(
    source: str | Path, *, threshold: float = 0.3, min_gap_s: float = 1.0
) -> list[SceneChange]:
    """Timestamps (source pts) where the scene score exceeds ``threshold``.

    Detections closer than ``min_gap_s`` are merged, keeping the highest score.
    For sampling, prefer :func:`autovid.ffmpeg_io.frames.extract_grid_and_scene_changes`
    which produces the same detections from a shared decode.
    """
    cmd = [
        ffmpeg_path(), "-nostdin", "-v", "info",
        "-i", str(source),
        "-vf", f"select='gt(scene,{threshold})',metadata=print",
        "-fps_mode", "vfr",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise AutoVidError(
            f"scene detection failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}"
        )

    found = [SceneChange(ts=t, score=s) for t, s in parse_scene_pairs(proc.stderr)]
    return merge_detections(found, min_gap_s)
