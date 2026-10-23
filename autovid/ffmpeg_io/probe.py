"""ffprobe wrapper: typed media metadata for ingested files.

All timestamps anywhere in AutoVid are seconds (float) against the source
timeline as reported here — everything stays traceable to the original media.
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..errors import AutoVidError
from .bins import ffprobe_path


class MediaInfo(BaseModel):
    """The subset of ffprobe output AutoVid reasons about."""

    duration_s: float = 0.0
    width: int | None = None
    height: int | None = None
    fps: float = 0.0
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool = False
    format_name: str = ""
    size_bytes: int = 0
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)


def _parse_fps(rate: str | None) -> float:
    """'30000/1001' → 29.97; None/0/invalid → 0.0."""
    if not rate:
        return 0.0
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            num_f, den_f = float(num), float(den)
            return round(num_f / den_f, 3) if den_f else 0.0
        return float(rate)
    except ValueError:
        return 0.0


@lru_cache(maxsize=512)
def _probe_cached(path: str, mtime_ns: int, size: int) -> MediaInfo:
    return _probe_uncached(path)


def probe(path: str | Path) -> MediaInfo:
    """Run ``ffprobe -show_format -show_streams`` and reduce to MediaInfo.

    Results are cached per (path, mtime, size): renders re-probe the same
    immutable derived clips across stages/items, and a re-written file gets a
    new mtime, so entries stay correct.
    """
    p = Path(path)
    st = p.stat()  # missing file → caller sees the OSError, cache untouched
    return _probe_cached(str(p), st.st_mtime_ns, st.st_size)


def _probe_uncached(path: str) -> MediaInfo:
    """Run ``ffprobe -show_format -show_streams`` and reduce to MediaInfo."""
    cmd = [
        ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        # AutoVidError, not RuntimeError: callers (web job targets) catch the
        # app's error type to degrade gracefully instead of failing the job.
        raise AutoVidError(f"ffprobe failed for {path}: {proc.stderr.strip()[:400]}")
    data = json.loads(proc.stdout or "{}")

    info = MediaInfo(
        format_name=data.get("format", {}).get("format_name", ""),
        size_bytes=int(data.get("format", {}).get("size") or 0),
        raw=data,
    )
    try:
        info.duration_s = round(float(data.get("format", {}).get("duration") or 0.0), 3)
    except ValueError:
        info.duration_s = 0.0

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and info.width is None:
            info.width = stream.get("width")
            info.height = stream.get("height")
            info.video_codec = stream.get("codec_name")
            info.fps = _parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
            # format duration can be missing for some containers; fall back.
            if not info.duration_s:
                try:
                    info.duration_s = round(float(stream.get("duration") or 0.0), 3)
                except ValueError:
                    pass
        elif stream.get("codec_type") == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = stream.get("codec_name")
    return info
