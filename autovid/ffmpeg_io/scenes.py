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
_SCENE_PAIR_RE = re.compile(
    r"pts_time:(\d+(?:\.\d+)?)[\s\S]{0,120}?lavfi\.scene_score=(\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class SceneChange:
    ts: float
    score: float


def detect_scene_changes(
    source: str | Path, *, threshold: float = 0.3, min_gap_s: float = 1.0
) -> list[SceneChange]:
    """Timestamps (source pts) where the scene score exceeds ``threshold``.

    Detections closer than ``min_gap_s`` are merged, keeping the highest score.
    """
    cmd = [
        ffmpeg_path(), "-v", "info",
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

    found = [
        SceneChange(ts=float(t), score=float(s))
        for t, s in _SCENE_PAIR_RE.findall(proc.stderr)
    ]
    found.sort(key=lambda c: c.ts)

    merged: list[SceneChange] = []
    for change in found:
        if merged and change.ts - merged[-1].ts < min_gap_s:
            if change.score > merged[-1].score:
                merged[-1] = change
            continue
        merged.append(change)
    return merged
