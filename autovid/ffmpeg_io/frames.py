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
from .scenes import SceneChange, merge_detections, parse_scene_pairs

_PTS_TIME_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
# showinfo logs one line per kept frame; scoping to those lines keeps the
# grid timestamps unambiguous when other branches also log pts_time.
_SHOWINFO_PTS_RE = re.compile(r"Parsed_showinfo[^\n]*?pts_time:(\d+(?:\.\d+)?)")


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise AutoVidError(f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
    return proc.stderr


def _seek_args(start_s: float | None, end_s: float | None) -> tuple[list[str], float]:
    """Input-seek options for an optional ``[start_s, end_s)`` span.

    Input seeking resets ffmpeg's timestamps to ~0, so callers rebase parsed
    ``pts_time`` values by the returned offset to stay on the source timeline.
    A span that starts at 0 still gets ``-t`` — without it the first window
    would decode the entire file instead of just its span.
    """
    start = float(start_s) if start_s is not None else 0.0
    if end_s is not None and end_s <= start:
        return [], 0.0  # empty/invalid span
    if start > 0:
        args = ["-ss", f"{start:.3f}"]
        if end_s is not None:
            args += ["-t", f"{end_s - start:.3f}"]
        return args, start
    if end_s is not None:
        return ["-t", f"{end_s:.3f}"], 0.0
    return [], 0.0


def extract_periodic_frames(
    source: str | Path,
    out_dir: Path,
    *,
    interval_s: float,
    width: int | None = None,
    prefix: str = "ts",
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[tuple[float, Path]]:
    """Extract one frame every ``interval_s``; returns ``[(ts, path)]``.

    Timestamps are the frames' true ``pts_time`` values on the source timeline.
    Output files are ``<prefix>_%06d.jpg`` in ``out_dir`` (created if needed).
    When ``start_s``/``end_s`` are given, only that span is decoded instead of
    the whole file — essential for per-window extraction, which otherwise
    re-decodes the full source once per window.
    """
    seek, offset = _seek_args(start_s, end_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%06d.jpg"
    scale = f",scale={width}:-2" if width else ""
    vf = (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval_s})'"
        f",showinfo{scale}"
    )
    try:
        stderr = _run([
            ffmpeg_path(), "-nostdin", "-y", "-v", "info",
            *seek,
            "-i", str(source),
            "-vf", vf,
            "-fps_mode", "vfr",
            str(pattern),
        ])
    except AutoVidError:
        for leftover in out_dir.glob(f"{prefix}_*.jpg"):
            leftover.unlink()
        raise

    pts = [float(m) + offset for m in _PTS_TIME_RE.findall(stderr)]
    files = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    if len(pts) != len(files):
        for leftover in files:
            leftover.unlink()
        raise AutoVidError(
            f"frame extraction mismatch: {len(pts)} pts_time entries vs {len(files)} files"
        )
    return list(zip(pts, files))


def extract_grid_and_scene_changes(
    source: str | Path,
    out_dir: Path,
    *,
    interval_s: float,
    width: int | None = None,
    prefix: str = "grid",
    scene_threshold: float = 0.3,
    min_gap_s: float = 1.0,
) -> tuple[list[tuple[float, Path]], list[SceneChange]]:
    """Grid frames AND scene-change detections from a single decode.

    Both branches split off one decoded stream — the grid branch keeps a frame
    every ``interval_s`` (each branch has its own ``prev_selected_t`` state, so
    the grid phase is identical to running ``extract_periodic_frames`` alone),
    while the scene branch evaluates ``gt(scene,T)`` per frame. Merging the two
    previously separate full-file decode passes halves sampling decode time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%06d.jpg"
    scale = f",scale={width}:-2" if width else ""
    filter_complex = (
        f"[0:v]split=2[grid][scen];"
        f"[grid]select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval_s})'"
        f",showinfo{scale}[gout];"
        # nullsink (not -f null): a zero-detection scene branch must not fail
        # the run the way an empty muxer output would.
        f"[scen]select='gt(scene,{scene_threshold})',metadata=print,nullsink"
    )
    try:
        stderr = _run([
            ffmpeg_path(), "-nostdin", "-y", "-v", "info",
            "-i", str(source),
            "-filter_complex", filter_complex,
            "-map", "[gout]", "-fps_mode", "vfr", str(pattern),
        ])
    except AutoVidError:
        for leftover in out_dir.glob(f"{prefix}_*.jpg"):
            leftover.unlink()
        raise

    pts = [float(m) for m in _SHOWINFO_PTS_RE.findall(stderr)]
    files = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    if len(pts) != len(files):
        for leftover in files:
            leftover.unlink()
        raise AutoVidError(
            f"frame extraction mismatch: {len(pts)} pts_time entries vs {len(files)} files"
        )
    changes = merge_detections(
        [SceneChange(ts=t, score=s) for t, s in parse_scene_pairs(stderr)], min_gap_s
    )
    return list(zip(pts, files)), changes


def frame_at(
    source: str | Path, ts: float, out_path: Path, *, width: int | None = None
) -> tuple[float, Path]:
    """Extract the single frame at ``ts`` seconds; returns ``(ts, out_path)``.

    Input seeking resets ffmpeg's timestamps, so the frame's true source
    timestamp is the requested ``ts`` itself.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_path(), "-nostdin", "-y", "-v", "error", "-ss", f"{ts:.3f}", "-i", str(source)]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]
    cmd += ["-frames:v", "1", str(out_path)]
    _run(cmd)
    if not out_path.exists():
        raise AutoVidError(f"no frame at t={ts:.3f}s in {source}")
    return ts, out_path
