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
from .organize import slugify

logger = logging.getLogger(__name__)


def _concat_line(path: Path) -> str:
    """One concat-demuxer ``file`` directive, safe for ``'`` in the path."""
    return "file '{}'".format(path.as_posix().replace("'", "'\\''"))


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
        ffmpeg_path(), "-nostdin", "-y", "-v", "error", "-i", str(src),
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy", str(dst),  # fade is video-only; clips are already AAC
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not dst.exists():
        raise AutoVidError(f"fade render failed: {proc.stderr.strip()[-300:]}")
    return dst


def _materialize(conn, settings: Settings, board: EDL) -> list[tuple[Any, Path]]:
    """(item, video file) pairs, extracting whatever the items need.

    - A segment's pre-extracted clip is reused only when the item covers the
      whole scene span (the classic storyboard case).
    - Music-driven items trim sub-spans of a scene, so each such item gets an
      accurate per-item clip cut from the source (staged in the render dir —
      reproducible, never written back over the scene clip). Staged clips are
      named by provenance ``(segment, start, end)``, so re-renders reuse them
      instead of re-extracting the whole timeline after a one-item edit.
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
        staged = (
            settings.data_dir / "renders" / board.id
            / f"item_{item.clip_id}_{item.source_start:.3f}-{item.source_end:.3f}.mp4"
        )
        if staged.exists():
            try:
                if abs(probe(staged).duration_s - item_span) < 0.25:
                    pairs.append((item, staged))
                    continue
            except Exception:  # noqa: BLE001 — partial/corrupt staged clip
                pass  # fall through: re-extract over it (a killed render must
                # never poison every future render of this EDL)
        # .part.mp4, not .part: ffmpeg picks its muxer from the extension
        tmp = staged.with_name(staged.stem + ".part.mp4")
        extract_clip(source, tmp, start_s=item.source_start, end_s=item.source_end)
        tmp.replace(staged)
        pairs.append((item, staged))
    return pairs


def render_edl(conn, settings: Settings, board: EDL) -> Path:
    """Render the rough-cut preview; returns the output path.

    With ``board.music_path`` set (music-driven editing), the music is muxed
    over the stitched video: video length always wins, the audio is padded if
    short and faded out over the last 2 s.

    The EDL is validated first (unknown segments, bad ranges, timeline
    overlaps, mixed resolutions) — rendering an invalid cut wastes minutes of
    ffmpeg only to produce a broken file.
    """
    from .edl import validate_edl

    validation = validate_edl(conn, board)
    if not validation.ok:
        details = "; ".join(e.message for e in validation.errors)
        raise AutoVidError(f"EDL failed validation: {details}")

    pairs = _materialize(conn, settings, board)
    if not pairs:
        raise AutoVidError("EDL has no items")

    out_dir = settings.data_dir / "renders" / board.id
    out_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    concat_file = out_dir / "concat.txt"
    concat_out = out_dir / "_video_only.mp4"
    # The name is client- and LLM-settable free text — slugify it so it can
    # never escape out_dir or carry Windows-illegal characters. The EDL keeps
    # its display name; only the file on disk is sanitized.
    out_path = out_dir / f"{slugify(board.name)}.mp4"
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

        concat_file.write_text(
            "\n".join(_concat_line(p) for p in staged), encoding="utf-8"
        )
        if board.music_path:
            # Single pass: the concat demuxer feeds the muxer directly, so the
            # stitched video is never written to disk just to be re-read.
            _render_with_music(concat_file, Path(board.music_path), staged, out_path)
        else:
            cmd = [
                ffmpeg_path(), "-nostdin", "-y", "-v", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-c", "copy", "-movflags", "+faststart", str(concat_out),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0 or not concat_out.exists():
                raise AutoVidError(
                    f"render failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}"
                )
            concat_out.replace(out_path)
        if not out_path.exists():
            raise AutoVidError("render produced no output")
        return out_path
    finally:
        # faded intermediates and staged clips no longer referenced by this
        # render are reproducible from the EDL; referenced item clips stay so
        # the next re-render skips their extraction
        for p in staged:
            if p.name.startswith("faded_"):
                p.unlink(missing_ok=True)
        keep = {p.resolve() for _, p in pairs}
        for p in out_dir.glob("item_*.mp4"):
            if p.resolve() not in keep:
                p.unlink(missing_ok=True)  # includes *.part.mp4 leftovers
        concat_file.unlink(missing_ok=True)
        concat_out.unlink(missing_ok=True)


def _render_with_music(
    concat_file: Path, music: Path, staged: list[Path], out_path: Path
) -> None:
    """Concat + music mux in one ffmpeg pass: video stream-copied from the
    concat demuxer, music padded if short and faded out over the last 2 s.

    ``-shortest`` plus infinite audio pad (``apad``) means the video track
    always defines the output length — a short BGM never truncates the cut.
    """
    from .ffmpeg_io.probe import probe

    duration = sum(max(probe(p).duration_s, 0.0) for p in staged)
    fade_st = max(duration - 2.0, 0.0)
    cmd = [
        ffmpeg_path(), "-nostdin", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", str(music),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-af", f"apad,afade=t=out:st={fade_st:.3f}:d=2",
        "-shortest",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not out_path.exists():
        raise AutoVidError(
            f"render failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}"
        )
