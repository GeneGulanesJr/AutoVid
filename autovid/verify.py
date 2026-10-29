"""Render verification: eyes on the finished MP4, for people and agents who
cannot watch video.

:func:`autovid.check.check_edl` lints the plan before a render;
:func:`verify_render` inspects the render after it — does the file match the
EDL (duration, cut placement), does the mix clip or go dead silent, are there
frozen stretches or hard flashes mid-shot. It also writes the artifacts a
reviewer reads instead of watching: a contact sheet per EDL item, an overview
grid of the whole cut, and a waveform PNG with the cut marks drawn on it.

Levels follow check.py: ``fail`` = the render diverges from the EDL (wrong
length, missing file), ``warn`` = worth a look (pop, freeze, clipping,
silence), ``info`` / ``ok`` = facts. Exit-code semantics live in the CLI
(``autovid verify``): any ``fail`` → exit 1, like ``check``.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .check import Finding
from .config import Settings
from .edl import EDL
from .errors import AutoVidError
from .ffmpeg_io.bins import ffmpeg_path
from .ffmpeg_io.probe import probe

# A boundary whose visible change is below this counts as "no visible cut" —
# normal when two items are adjacent stretches of one continuous take.
CUT_THRESHOLD = 0.35
BOUNDARY_TOL_S = 0.4       # the actual cut may sit a few frames off the plan
DURATION_WARN_S = 0.5
DURATION_FAIL_S = 1.5      # …or 5% off, whichever is larger
DURATION_FAIL_FRAC = 0.05
FREEZE_N = 0.004           # <0.4% of pixels changing = frozen (handheld grain stays above this)
FREEZE_D = 3.0             # …held for 3s (static b-roll shots are shorter in the cut)
CLIP_PEAK_DBFS = -0.5      # true peak at/above this is clipping
LUFS_MIN, LUFS_MAX = -30.0, -9.0   # web target is ≈ -14 LUFS
SILENCE_DB = -50
SILENCE_MIN_S = 1.0
TIMELINE_TOL_S = 0.3

_I_RE = re.compile(r"I:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*LUFS")
_PEAK_RE = re.compile(r"Peak:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dBFS")
_SIL_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
_SIL_END_RE = re.compile(r"silence_end:\s*([\d.]+)")
_PENDING_TS_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
_FREEZE_KEY_RE = re.compile(r"freeze_duration[=:]\s*(\d+(?:\.\d+)?)")


def render_path(settings: Settings, board: EDL) -> Path:
    """Where ``render_edl`` puts this EDL's preview (same convention)."""
    from .organize import slugify
    return settings.data_dir.resolve() / "renders" / board.id / f"{slugify(board.name)}.mp4"


def expected_windows(board: EDL) -> list[tuple[int, float, float]]:
    """Output-timeline ``[start, end)`` per item, as the renderer builds it.

    Items are extracted at 1.0 speed and concatenated back-to-back, so a
    boundary is the cumulative span sum — an EDL's ``timeline_start``/``end``
    placement is checked against this, not trusted.
    """
    t = 0.0
    out: list[tuple[int, float, float]] = []
    for i, item in enumerate(board.items):
        span = item.source_end - item.source_start
        out.append((i, t, t + span))
        t += span
    return out


def verify_render(
    conn, settings: Settings, board: EDL, render: str | Path | None = None, *,
    sheets: bool = True, waveform: bool = True, audio: bool = True,
) -> dict[str, Any]:
    """One pass over a finished render; returns the report dict the CLI prints
    or dumps as JSON. Findings reuse check.py's Finding levels."""
    path = Path(render) if render is not None else render_path(settings, board)
    windows = expected_windows(board)
    expected_dur = windows[-1][2] if windows else 0.0
    report: dict[str, Any] = {
        "edl": {"id": board.id, "name": board.name},
        "render": str(path),
        "expected_duration_s": round(expected_dur, 3),
        "actual_duration_s": None,
        "items": [{"item": i + 1, "clip_id": item.clip_id,
                   "output_start_s": round(s, 3), "output_end_s": round(e, 3),
                   "sheet": None}
                  for (i, s, e), item in zip(windows, board.items)],
        "overview_sheet": None,
        "waveform": None,
        "findings": [],
    }
    findings = report["findings"]

    def add(level: str, name: str, detail: str, *, item_index: int | None = None) -> None:
        findings.append(Finding(level, name if item_index is None else f"item {item_index + 1} {name}",
                                detail))

    if not board.items:
        add("fail", "edl", "the EDL is empty — nothing to verify")
        return report

    # Plan-side facts first: the renderer ignores the v2 placement/speed fields.
    for i, item in enumerate(board.items):
        if item.speed != 1.0:
            add("warn", "speed", f"the renderer plays at 1.0×; the EDL's {item.speed}× is not applied",
                item_index=i)
        win = windows[i]
        if item.timeline_start is not None and abs(item.timeline_start - win[1]) > TIMELINE_TOL_S:
            add("warn", "timeline placement",
                f"the EDL places this item at {item.timeline_start:.2f}s but the render stitches it "
                f"at {win[1]:.2f}s (back-to-back) — the file follows the stitch",
                item_index=i)

    if not path.exists():
        add("fail", "render", f"missing: {path} — render the EDL first (`autovid run --objective …`)")
        return report

    info = probe(path)
    report["actual_duration_s"] = info.duration_s
    if info.width is None:
        add("fail", "streams", "no video stream in the render")
        return report
    add("ok", "streams", f"{info.width}x{info.height} @ {info.fps:.2f} fps, "
                         f"{info.duration_s:.2f}s, audio: {info.audio_codec or 'none'}")

    drift = abs(info.duration_s - expected_dur)
    if drift > max(DURATION_FAIL_S, expected_dur * DURATION_FAIL_FRAC):
        add("fail", "duration", f"render is {info.duration_s:.2f}s, the EDL plans {expected_dur:.2f}s")
    elif drift > DURATION_WARN_S:
        add("warn", "duration", f"render is {info.duration_s:.2f}s, the EDL plans {expected_dur:.2f}s")

    spikes, freezes = _scene_spikes_and_freezes(path)
    cuts = [e for _, _, e in windows[:-1]]  # interior boundaries
    for n, boundary in enumerate(cuts):
        near = next((s for s in spikes if abs(s[0] - boundary) <= BOUNDARY_TOL_S), None)
        if near:
            add("ok", "cut", f"boundary at {boundary:.2f}s confirmed (score {near[1]:.2f})",
                item_index=n)
        else:
            add("warn", "cut", f"no visible cut at {boundary:.2f}s — adjacent shots of one take "
                               f"blend; fine when intended", item_index=n)
    for ts, score in spikes:
        if any(abs(ts - b) <= BOUNDARY_TOL_S for b in cuts):
            continue  # this spike *is* a planned boundary
        owner = next((i for i, _, e in windows if ts < e - 0.05), None)
        if owner is not None:
            add("warn", "pop", f"hard change mid-shot at {ts:.2f}s (score {score:.2f}) — a flash "
                               f"frame or an unplanned cut inside the clip", item_index=owner)
    for ts, dur in freezes:
        owner = next((i for i, _, e in windows if ts - dur / 2 < e - 0.05), None)
        if owner is not None:
            add("warn", "freeze", f"~{dur:.1f}s frozen around {ts - dur:.2f}s — a stalled source "
                                  f"or a genuinely static shot", item_index=owner)

    verify_dir = path.parent / "verify"
    if sheets:
        verify_dir.mkdir(parents=True, exist_ok=True)
        overview = _sheet(path, 0.0, info.duration_s, verify_dir / "sheet_overview.png",
                          frames=24, cols=6, width=200)
        report["overview_sheet"] = str(overview) if overview else None
        for i, start, end in windows:
            sheet = _sheet(path, start, end, verify_dir / f"sheet_item_{i + 1:02d}.png")
            report["items"][i]["sheet"] = str(sheet) if sheet else None
        if overview:
            add("ok", "sheets", f"{len(report['items'])} item sheets + overview in {verify_dir}")

    if audio and info.has_audio:
        for i, start, end in windows:
            a = _item_audio(path, start, end)
            report["items"][i]["audio"] = {
                "lufs": a["lufs"], "peak_dbfs": a["peak"],
                "silences_s": [[round(s, 2), round(e, 2)] for s, e in a["silences"]],
            }
            span = end - start
            if a["peak"] is not None and a["peak"] >= CLIP_PEAK_DBFS:
                add("warn", "clipping", f"true peak {a['peak']:.1f} dBFS", item_index=i)
            if a["lufs"] is not None and not LUFS_MIN <= a["lufs"] <= LUFS_MAX:
                add("warn", "loudness", f"{a['lufs']:.1f} LUFS (web target ≈ -14)", item_index=i)
            if a["lufs"] is None and not a["silences"]:
                add("info", "silence", "digital silence throughout", item_index=i)
            else:
                dead = sum(e - s for s, e in a["silences"])
                if dead > span / 2:
                    add("warn", "silence", f"{dead:.1f}s of {span:.1f}s is silent", item_index=i)
                elif a["silences"]:
                    first = a["silences"][0]
                    add("info", "silence", f"quiet stretch {first[0]:.1f}–{first[1]:.1f}s",
                        item_index=i)
    elif audio:
        add("info", "audio", "the render has no audio track — loudness and waveform checks skipped")

    if waveform and info.has_audio:
        verify_dir.mkdir(parents=True, exist_ok=True)
        wave = _waveform(path, verify_dir / "waveform.png", cuts, info.duration_s)
        if wave:
            report["waveform"] = str(wave)
            add("ok", "waveform", f"waveform with {len(cuts)} cut mark(s): {wave.name}")

    return report


def report_to_json(report: dict[str, Any]) -> str:
    """Findings are dataclasses; everything else is already JSON-native."""
    import dataclasses
    out = dict(report)
    out["findings"] = [dataclasses.asdict(f) for f in report["findings"]]
    return json.dumps(out, indent=2)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _scene_spikes_and_freezes(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """``(spikes, freezes)`` — scene-score detections and freeze intervals.

    Two decodes, boringly: the scene side reuses the battle-tested scenes.py
    parser; the freeze side parses ``freezedetect + metadata=print`` output the
    same way (a pts_time line arms, then the freeze_duration key completes it —
    both the start and the end event carry a duration, so duplicates are
    dropped).
    """
    from .ffmpeg_io.scenes import detect_scene_changes

    spikes = [(c.ts, c.score) for c in
              detect_scene_changes(path, threshold=CUT_THRESHOLD, min_gap_s=0.25)]

    cmd = [
        ffmpeg_path(), "-nostdin", "-v", "info", "-i", str(path),
        "-vf", f"freezedetect=n={FREEZE_N}:d={FREEZE_D},metadata=print",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise AutoVidError(
            f"freeze detection failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}"
        )
    freezes: list[tuple[float, float]] = []
    for ts, dur in _paired_freezes(proc.stderr):
        if all(abs(ts - t) > 0.01 for t, _ in freezes):
            freezes.append((ts, dur))
    return spikes, freezes


def _paired_freezes(stderr: str) -> list[tuple[float, float]]:
    """``(pts_time, freeze_duration)`` pairs from freezedetect's metadata print."""
    out: list[tuple[float, float]] = []
    pending: float | None = None
    for line in stderr.splitlines():
        if "Parsed_metadata" not in line:
            continue
        ts_match = _PENDING_TS_RE.search(line)
        dur_match = _FREEZE_KEY_RE.search(line)
        if ts_match and not dur_match:
            pending = float(ts_match.group(1))
        elif dur_match:
            ts = float(ts_match.group(1)) if ts_match else pending
            if ts is not None:
                out.append((ts, float(dur_match.group(1))))
            pending = None
    return out


def _item_audio(path: Path, start: float, end: float) -> dict[str, Any]:
    """Loudness (LUFS), true peak and silences of one output window.

    Input-seek resets pts, which is exactly what we want here: reported
    silences are relative to the item's own start.
    """
    cmd = [
        ffmpeg_path(), "-nostdin", "-v", "info",
        "-ss", f"{start:.3f}", "-t", f"{max(end - start, 0.1):.3f}", "-i", str(path),
        "-map", "0:a:0",
        "-af", f"ebur128=peak=true,silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise AutoVidError(
            f"audio analysis failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}"
        )
    err = proc.stderr
    starts = [float(m) for m in _SIL_START_RE.findall(err)]
    ends = [float(m) for m in _SIL_END_RE.findall(err)]
    return {
        "lufs": _last_match(_I_RE, err),
        "peak": _last_match(_PEAK_RE, err),
        "silences": list(zip(starts, ends + [end - start])),
    }


def _last_match(rx: re.Pattern[str], text: str) -> float | None:
    """Last regex capture as float; '-inf' (digital silence) → None."""
    found = rx.findall(text)
    if not found:
        return None
    v = found[-1]
    return None if "inf" in v else float(v)


def _sheet(path: Path, start: float, end: float, out: Path, *, frames: int = 8,
           cols: int = 4, width: int = 320) -> Path | None:
    """A contact sheet of one output window (the filename carries the label)."""
    span = max(end - start, 0.04)
    cmd = [
        ffmpeg_path(), "-nostdin", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", str(path),
        "-vf", f"fps={frames / span:.4f},scale={width}:-2,tile={cols}x{max(frames // cols, 1)}",
        "-frames:v", "1", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return out if proc.returncode == 0 and out.exists() else None


def _waveform(path: Path, out: Path, cuts: list[float], total: float) -> Path | None:
    """Waveform of the whole render with the EDL cut marks drawn on it."""
    cmd = [
        ffmpeg_path(), "-nostdin", "-y", "-v", "error", "-i", str(path),
        "-filter_complex", "showwavespic=s=1400x300:colors=#4aa3ff",
        "-frames:v", "1", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not out.exists():
        return None
    from PIL import Image, ImageDraw

    img = Image.open(out).convert("RGB")
    draw = ImageDraw.Draw(img)
    for c in cuts:
        x = min(int(c / total * img.width), img.width - 1)
        draw.line([(x, 0), (x, img.height)], fill=(230, 60, 60), width=2)
    img.save(out)
    return out
