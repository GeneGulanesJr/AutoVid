"""EDL lint."""
from __future__ import annotations

import pytest

from autovid import db
from autovid.check import check_edl
from autovid.edl import EDL, EDLItem


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "c.db")
    for n in range(2):
        c.execute("INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s, fps, width,"
                  " height, has_audio, ffprobe_json) VALUES (?,?,?,?,?,60,24,1920,1080,1,'{}')",
                  (f"a{n}", f"s{n}", "p", "p", f"c{n}.mp4"))
    c.commit()
    return c


def _item(asset, start, end, note=""):
    return EDLItem(clip_id=1, asset_id=asset, source_start=start, source_end=end, note=note, audio="clip")


def _by_name(findings):
    return {f.name: f for f in findings}


def test_clean_cut_passes(conn):
    board = EDL(items=[_item("a0", 0, 5), _item("a0", 10, 15), _item("a1", 0, 5, "x: [food] pancakes")])
    f = _by_name(check_edl(conn, board, fingerprint=lambda p, t: 0))
    assert f["replay"].level == "ok" and f["look-alikes"].level == "ok"
    assert f["food share"].detail.startswith("1 food shots = 5.0s (33%")
    assert "fail" not in {x.level for x in f.values()}


def test_replayed_footage_fails(conn):
    board = EDL(items=[_item("a0", 0, 5), _item("a0", 3, 8)])          # 2 s of a0 plays twice
    f = _by_name(check_edl(conn, board))
    assert f["replay"].level == "fail" and f["replay"].detail.startswith("2.00s")


def test_lookalike_food_shots_fail_and_a_speech_heavy_cut_warns(conn):
    board = EDL(items=[_item("a0", 0, 30), _item("a1", 0, 4, "b: [food] rice"), _item("a1", 20, 24, "b: [food] fish")])
    f = _by_name(check_edl(conn, board, fingerprint=lambda p, t: 0))   # identical frames
    assert f["look-alikes"].level == "fail" and "items [2, 3]" in f["look-alikes"].detail
    assert f["food share"].level == "warn"                             # 8 of 38 s = 21%


def test_tiny_clips_empty_edl_and_unknown_asset(conn):
    assert _by_name(check_edl(conn, EDL(items=[_item("a0", 0, 5), _item("a1", 0, 0.2)])))["tiny clips"].level == "warn"
    assert check_edl(conn, EDL(items=[]))[0].level == "fail"
    assert check_edl(conn, EDL(items=[_item("nope", 0, 5)]))[-1].level == "fail"


def test_timelapse_overuse_warns(conn, monkeypatch):
    import autovid.check as check_mod

    monkeypatch.setattr(check_mod, "classify", lambda c: {"a0": {"timelapse"}, "a1": set()})
    f = _by_name(check_edl(conn, EDL(items=[_item("a0", 0, 40)])))     # 40 of 60 s of a timelapse
    assert f["timelapse"].level == "warn"


def test_check_cli_exits_nonzero_on_a_defect(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from autovid import edl as edl_mod
    from autovid.cli import app
    from autovid.config import get_settings

    monkeypatch.setenv("AUTOVID_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    c = db.connect(tmp_path / "autovid.db")
    c.execute("INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s, fps, width, height,"
              " has_audio, ffprobe_json) VALUES ('a0','s','p','p','c.mp4',60,24,1920,1080,1,'{}')")
    edl_mod.save(c, EDL(name="bad", items=[_item("a0", 0, 5), _item("a0", 2, 7)]))
    result = CliRunner().invoke(app, ["check", "bad"])
    get_settings.cache_clear()
    assert result.exit_code == 1 and "replay" in result.output and "FAIL" in result.output
