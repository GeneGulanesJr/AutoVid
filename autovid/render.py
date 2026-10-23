"""Rough-cut preview rendering: an EDL → one MP4 via the concat demuxer.

All clips come from :mod:`autovid.ffmpeg_io.clips` (same libx264/aac
parameters), so a lossless concat copy is safe and fast. Transitions marked
``fade`` become fade-to/from-black on the affected clip (re-encode of that
clip only). The render never touches originals; output lands in
``catalog/renders/<edl_id>/``.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from .config import Settings
from .edl import EDL
from .errors import AutoVidError
from .ffmpeg_io.bins import ffmpeg_path

logger = logging.getLogger(__name__)


def _fade_clip(src: Path, dst: Path, *, fade_in: bool, fade_out: bool) -> Path:
    """Re-encode one clip with fade-to/from-black (0.4 s) at its edges."""
    from .ffmpeg_io.probe import probe

    duration = max(probe(src).duration_s, 0.2)
    fade_len = min(0.4, duration / 3)
    vf = []
    if fade_in:
        vf.append(f"fade=t=in:st=0:d={fade_len:.3f}")
    if fade_out:
        vf.append(f"fade=t=out:st={max(duration - fade_len, 0):.3f}:d={fade_len:.3f}")
    cmd = [
        ffmpeg_path(), "-y", "-v", "error", "-i", str(src),
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not dst.exists():
        raise AutoVidError(f"fade render failed: {proc.stderr.strip()[-300:]}")
    return dst


def _conform(src: Path, dst: Path, ref, has_audio: bool) -> Path:
    """Re-encode ``src`` to ``ref``'s frame size (letterboxed) and frame rate, with an audio track."""
    w, h = ref.width, ref.height
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-i", str(src)]
    if not has_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={ref.fps:.6f}",
            "-map", "0:v", "-map", "0:a" if has_audio else "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-ac", "2"]
    if not has_audio:
        cmd += ["-shortest"]
    proc = subprocess.run(cmd + [str(dst)], capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not dst.exists():
        raise AutoVidError(f"conform failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
    return dst


def _materialize(conn, settings: Settings, board: EDL) -> list[tuple[Any, Path]]:
    """(item, video file) pairs, extracting whatever the items need.

    - A segment's pre-extracted clip is reused only when the item covers the
      whole scene span (the classic storyboard case).
    - Music-driven items trim sub-spans of a scene, so each such item gets an
      accurate per-item clip cut from the source (staged in the render dir —
      reproducible, never written back over the scene clip).
    """
    from .edl import EDLItem
    from .ffmpeg_io.clips import extract_clip
    from .ffmpeg_io.probe import probe

    pairs: list[tuple[EDLItem, Path]] = []
    for i, item in enumerate(board.items):
        row = conn.execute(
            """SELECT s.clip_path, s.start_s, s.end_s, s.asset_id,
                      a.path AS source_path
               FROM segments s JOIN assets a ON a.id = s.asset_id
               WHERE s.id = ?""",
            (item.clip_id,),
        ).fetchone()
        if row is None:
            raise AutoVidError(f"EDL item references unknown segment {item.clip_id}")

        item_span = item.source_end - item.source_start
        clip_path = Path(row["clip_path"]) if row["clip_path"] else None
        if (
            clip_path is not None
            and clip_path.exists()
            and item.source_start == row["start_s"]
            and abs(probe(clip_path).duration_s - item_span) < 0.25
        ):
            pairs.append((item, clip_path))
            continue

        source = Path(row["source_path"])
        if not source.exists():
            raise AutoVidError(f"source file missing for segment {item.clip_id}: {source}")
        out = settings.data_dir / "renders" / board.id / f"item_{i:04d}.mp4"
        extract_clip(source, out, start_s=item.source_start, end_s=item.source_end)
        pairs.append((item, out))
    return pairs


def render_edl(conn, settings: Settings, board: EDL) -> Path:
    """Render the rough-cut preview; returns the output path.

    With ``board.music_path`` set (music-driven editing), the music is muxed
    over the stitched video: video length always wins, the audio is padded if
    short and faded out over the last 2 s.
    """
    from .ffmpeg_io.probe import probe

    # ffmpeg resolves concat-list entries against the list's folder, so a relative --data-dir breaks.
    settings = settings.model_copy(update={"data_dir": settings.data_dir.resolve()})
    pairs = _materialize(conn, settings, board)
    if not pairs:
        raise AutoVidError("EDL has no items")

    out_dir = settings.data_dir / "renders" / board.id
    out_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    concat_file = out_dir / "concat.txt"
    concat_out = out_dir / "_video_only.mp4"
    out_path = out_dir / f"{board.name}.mp4"
    try:
        for i, (item, path) in enumerate(pairs):
            fade_in = item.transition_in == "fade"
            fade_out = item.transition_out == "fade"
            if fade_in or fade_out:
                staged.append(_fade_clip(
                    path, out_dir / f"faded_{i:04d}.mp4",
                    fade_in=fade_in, fade_out=fade_out,
                ))
            else:
                staged.append(path)

        # Stream-copy concat needs identical streams. Clips from different cameras (size, frame rate,
        # no audio track) otherwise leave timestamp gaps, so conform them all to the first clip.
        infos = [probe(p) for p in staged]
        if len({(i.width, i.height, round(i.fps, 2), i.has_audio) for i in infos}) > 1:
            # the format most clips (by length) already have, so the majority is not letterboxed
            weight: dict[tuple, float] = {}
            for i in infos:
                weight[(i.width, i.height, round(i.fps, 2))] = weight.get((i.width, i.height, round(i.fps, 2)), 0) + i.duration_s
            ref = next(i for i in infos if (i.width, i.height, round(i.fps, 2)) == max(weight, key=weight.get))
            staged = [_conform(p, out_dir / f"conformed_{k:04d}.mp4", ref, i.has_audio)
                      for k, (p, i) in enumerate(zip(staged, infos))]

        concat_file.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in staged), encoding="utf-8"
        )
        cmd = [
            ffmpeg_path(), "-y", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(concat_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0 or not concat_out.exists():
            raise AutoVidError(
                f"render failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}"
            )

        if board.music_path:
            _mux_music(concat_out, Path(board.music_path), out_path)
        else:
            concat_out.replace(out_path)
        if not out_path.exists():
            raise AutoVidError("render produced no output")
        return out_path
    finally:
        # keep only the final render; staged fades and per-item clips are
        # reproducible from the EDL
        for p in staged:
            if p.name.startswith(("faded_", "conformed_")):
                p.unlink(missing_ok=True)
        for p in out_dir.glob("item_*.mp4"):
            p.unlink(missing_ok=True)
        concat_file.unlink(missing_ok=True)
        concat_out.unlink(missing_ok=True)


def _mux_music(video: Path, music: Path, out_path: Path) -> None:
    """Lay the music over the stitched video: pad if short, fade out 2 s.

    ``-shortest`` plus infinite audio pad (``apad``) means the video track
    always defines the output length — a short BGM never truncates the cut.
    """
    from .ffmpeg_io.probe import probe

    duration = max(probe(video).duration_s, 0.5)
    fade_st = max(duration - 2.0, 0.0)
    cmd = [
        ffmpeg_path(), "-y", "-v", "error",
        "-i", str(video), "-i", str(music),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-af", f"apad,afade=t=out:st={fade_st:.3f}:d=2",
        "-shortest",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not out_path.exists():
        raise AutoVidError(f"music mux failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
