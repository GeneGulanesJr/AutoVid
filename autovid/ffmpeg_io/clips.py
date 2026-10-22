"""Clip extraction: source-accurate segments re-encoded for edit stability.

Every clip keeps its provenance in the filename
(``<asset_id>__<start_s>-<end_s>.mp4``) and in the segments table; the
original files are only ever read.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..errors import AutoVidError
from .bins import ffmpeg_path


def clip_filename(asset_id: str, start_s: float, end_s: float) -> str:
    return f"{asset_id}__{start_s:.3f}-{end_s:.3f}.mp4"


def extract_clip(
    source: str | Path,
    out_path: Path,
    *,
    start_s: float,
    end_s: float,
    reencode: bool = True,
    crf: int = 20,
    preset: str = "veryfast",
) -> Path:
    """Extract ``[start_s, end_s)`` from ``source`` into ``out_path``.

    Output-seek + re-encode gives frame-accurate cuts independent of keyframe
    placement; stream-copy is only offered for callers that have already
    verified both edges land on keyframes.
    """
    if end_s <= start_s:
        raise AutoVidError(f"invalid clip span [{start_s:.3f}, {end_s:.3f})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(source)]
    if reencode:
        cmd += ["-t", f"{end_s - start_s:.3f}",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-c:a", "aac", "-movflags", "+faststart"]
    else:
        cmd += ["-t", f"{end_s - start_s:.3f}", "-c", "copy"]
    cmd += [str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not out_path.exists():
        raise AutoVidError(
            f"clip extraction failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}"
        )
    return out_path
