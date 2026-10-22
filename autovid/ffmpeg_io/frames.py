"""Frame extraction helpers.

Exact timestamps come from ffmpeg's ``showinfo`` log (``pts_time``), so every
extracted frame is recorded against the true source timeline — never assumed
from filter arithmetic. (This build's ``fps`` filter drops all frames; the
``select``-based recipe is the reliable path.)
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..errors import AutoVidError
from .bins import ffmpeg_path

_PTS_TIME_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise AutoVidError(f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
    return proc.stderr


def extract_periodic_frames(
    source: str | Path,
    out_dir: Path,
    *,
    interval_s: float,
    width: int | None = None,
    prefix: str = "ts",
) -> list[tuple[float, Path]]:
    """Extract one frame every ``interval_s``; returns ``[(ts, path)]``.

    Timestamps are the frames' true ``pts_time`` values on the source timeline.
    Output files are ``<prefix>_%06d.jpg`` in ``out_dir`` (created if needed).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%06d.jpg"
    scale = f",scale={width}:-2" if width else ""
    vf = (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval_s})'"
        f",showinfo{scale}"
    )
    try:
        stderr = _run([
            ffmpeg_path(), "-y", "-v", "info",
            "-i", str(source),
            "-vf", vf,
            "-fps_mode", "vfr",
            str(pattern),
        ])
    except AutoVidError:
        for leftover in out_dir.glob(f"{prefix}_*.jpg"):
            leftover.unlink()
        raise

    pts = [float(m) for m in _PTS_TIME_RE.findall(stderr)]
    files = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    if len(pts) != len(files):
        for leftover in files:
            leftover.unlink()
        raise AutoVidError(
            f"frame extraction mismatch: {len(pts)} pts_time entries vs {len(files)} files"
        )
    return list(zip(pts, files))


def frame_at(
    source: str | Path, ts: float, out_path: Path, *, width: int | None = None
) -> Path:
    """Extract the single frame at ``ts`` seconds (output-seek, exact)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-ss", f"{ts:.3f}", "-i", str(source)]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]
    cmd += ["-frames:v", "1", str(out_path)]
    _run(cmd)
    if not out_path.exists():
        raise AutoVidError(f"no frame at t={ts:.3f}s in {source}")
    return out_path
