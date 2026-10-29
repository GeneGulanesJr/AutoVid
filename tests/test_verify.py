"""Render verification against real tiny renders: does the file match the EDL,
and do the agent-readable artifacts (sheets, waveform) actually appear?"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

import autovid.cli as cli
from autovid import db
from autovid.config import Settings
from autovid.edl import EDL, EDLItem, save as save_edl
from autovid.ffmpeg_io.bins import ffmpeg_path
from autovid.render import render_edl
from autovid.verify import report_to_json, verify_render


def make_multiscene_video(path: Path, *, scene_s: float = 2.0, scenes: int = 4,
                          width: int = 160, height: int = 120, fps: int = 12) -> Path:
    """Concatenated distinct sources → hard cuts at scene_s, 2*scene_s, …"""
    spec = f"duration={{d}}:size={width}x{height}:rate={fps}"
    sources = ["smptebars=" + spec, "testsrc2=" + spec, "color=red:" + spec, "gradients=" + spec]
    cmd = [ffmpeg_path(), "-y", "-v", "error"]
    for i in range(scenes):
        cmd += ["-f", "lavfi", "-i", sources[i % len(sources)].format(d=scene_s)]
    inputs = [f"[{i}:v]" for i in range(scenes)]
    cmd += ["-filter_complex", f"{''.join(inputs)}concat=n={scenes}:v=1[v]", "-map", "[v]",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


def make_video(path: Path, *, duration: int = 4, audio: bool = False) -> Path:
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-f", "lavfi",
           "-i", f"testsrc2=duration={duration}:size=160x120:rate=12"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


def _color(path: Path, colour: str, duration: int = 4) -> Path:
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-f", "lavfi",
           "-i", f"color=c={colour}:size=160x120:rate=12:duration={duration}",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


def _seed(conn, asset_id: str, clip: Path, duration: float = 8.0) -> int:
    conn.execute(
        "INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s)"
        " VALUES (?,?,?,?,?,?)", (asset_id, f"s-{asset_id}", str(clip), str(clip), clip.name, duration))
    return conn.execute(
        "INSERT INTO segments (asset_id, start_s, end_s) VALUES (?,?,?)",
        (asset_id, 0.0, duration)).lastrowid


def _catalog(tmp_path: Path) -> tuple[object, Settings]:
    conn = db.connect(tmp_path / "m.db")
    settings = Settings(data_dir=tmp_path / "data", _env_file=None, vlm_api_key="")
    return conn, settings


def test_verify_confirms_cuts_flags_pops_and_writes_sheets(tmp_path: Path) -> None:
    conn, settings = _catalog(tmp_path)
    src = make_multiscene_video(tmp_path / "multiscene.mp4", scene_s=2, scenes=4)  # cuts at 2, 4, 6
    seg = _seed(conn, "av1", src, 8.0)
    board = EDL(name="verifytest", items=[
        EDLItem(clip_id=seg, asset_id="av1", source_start=0.0, source_end=2.0),
        EDLItem(clip_id=seg, asset_id="av1", source_start=2.0, source_end=6.0),  # covers the cut at 4s
    ])
    out = render_edl(conn, settings, board)
    assert out == settings.data_dir.resolve() / "renders" / board.id / "verifytest.mp4"

    report = verify_render(conn, settings, board)
    findings = report["findings"]
    assert not [f for f in findings if f.level == "fail"]
    assert any(f.level == "ok" and "confirmed" in f.detail for f in findings if f.name.endswith("cut"))
    # item 2 plays over the source's own cut at 4s → a mid-shot pop is expected and reported
    assert any("pop" in f.name and f.level == "warn" for f in findings)
    assert abs(report["actual_duration_s"] - 6.0) < 0.5

    # the render has no audio (video-only sources) → graceful skip, not a crash
    assert report["waveform"] is None
    assert any(f.name == "audio" and f.level == "info" for f in findings)

    sheets = [Path(i["sheet"]) for i in report["items"]]
    assert all(s and s.exists() and s.stat().st_size > 0 for s in sheets)
    overview = Path(report["overview_sheet"])
    assert overview.exists() and overview.stat().st_size > 0

    parsed = json.loads(report_to_json(report))
    assert parsed["edl"]["id"] == board.id and len(parsed["findings"]) == len(findings)


def test_verify_reports_audio_levels_and_waveform(tmp_path: Path) -> None:
    conn, settings = _catalog(tmp_path)
    src = make_video(tmp_path / "tone.mp4", duration=4, audio=True)
    seg = _seed(conn, "av2", src, 4.0)
    board = EDL(name="tonetest", items=[
        EDLItem(clip_id=seg, asset_id="av2", source_start=0.0, source_end=4.0)])
    render_edl(conn, settings, board)

    report = verify_render(conn, settings, board)
    findings = report["findings"]
    assert not [f for f in findings if f.level == "fail"]
    audio = report["items"][0]["audio"]
    assert audio["lufs"] is not None and audio["lufs"] > -40.0   # a loud sine, parsed
    assert audio["silences_s"] == []                             # 4s of 440 Hz never drops out
    assert report["waveform"] and Path(report["waveform"]).exists()


def test_verify_confirms_boundary_and_warns_on_freezes(tmp_path: Path) -> None:
    conn, settings = _catalog(tmp_path)
    red = _color(tmp_path / "red.mp4", "red")
    blue = _color(tmp_path / "blue.mp4", "blue")
    s1 = _seed(conn, "av3", red, 4.0)
    s2 = _seed(conn, "av4", blue, 4.0)
    board = EDL(name="freezetest", items=[
        EDLItem(clip_id=s1, asset_id="av3", source_start=0.0, source_end=4.0),
        EDLItem(clip_id=s2, asset_id="av4", source_start=0.0, source_end=4.0)])
    render_edl(conn, settings, board)

    report = verify_render(conn, settings, board, sheets=False, waveform=False, audio=False)
    findings = report["findings"]
    assert not [f for f in findings if f.level == "fail"]
    assert any(f.level == "ok" and "confirmed" in f.detail for f in findings)   # red→blue is a hard cut
    assert any("freeze" in f.name and f.level == "warn" for f in findings)      # static colour = frozen


def test_verify_fails_when_duration_diverges(tmp_path: Path) -> None:
    conn, settings = _catalog(tmp_path)
    src = make_multiscene_video(tmp_path / "multiscene.mp4", scene_s=2, scenes=4)
    seg = _seed(conn, "av5", src, 8.0)
    rendered_board = EDL(name="short", items=[
        EDLItem(clip_id=seg, asset_id="av5", source_start=0.0, source_end=6.0)])
    out = render_edl(conn, settings, rendered_board)              # a 6s file

    planned_board = EDL(name="short", items=[
        EDLItem(clip_id=seg, asset_id="av5", source_start=0.0, source_end=4.0),
        EDLItem(clip_id=seg, asset_id="av5", source_start=4.0, source_end=8.0)])  # plans 8s
    report = verify_render(conn, settings, planned_board, render=out,
                           sheets=False, waveform=False, audio=False)
    fails = [f for f in report["findings"] if f.level == "fail"]
    assert any("duration" in f.name for f in fails)


def test_verify_missing_render_fails_cleanly(tmp_path: Path) -> None:
    conn, settings = _catalog(tmp_path)
    board = EDL(name="ghost", items=[])
    report = verify_render(conn, settings, board, render=tmp_path / "nope.mp4")
    fails = [f for f in report["findings"] if f.level == "fail"]
    assert fails and report["actual_duration_s"] is None


def test_verify_cli_json(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None, vlm_api_key="")
    conn = db.connect(settings.db_path)  # the CLI opens the catalog DB inside data_dir
    src = make_multiscene_video(tmp_path / "multiscene.mp4", scene_s=2, scenes=2)
    seg = _seed(conn, "av6", src, 4.0)
    board = EDL(name="clitest", items=[
        EDLItem(clip_id=seg, asset_id="av6", source_start=0.0, source_end=4.0)])
    save_edl(conn, board)
    render_edl(conn, settings, board)

    runner = CliRunner()
    result = runner.invoke(cli.app, [
        "verify", board.id, "--json", "--no-sheets", "--no-waveform", "--no-audio",
        "--data-dir", str(settings.data_dir),
    ])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["edl"]["id"] == board.id
    assert abs(parsed["actual_duration_s"] - 4.0) < 0.5
