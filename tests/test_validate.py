"""EDL validation: item-level issues, errors block render, warnings don't."""
from __future__ import annotations

import pytest

from autovid.db import connect
from autovid.edl import EDL, EDLItem, validate_edl
from tests.conftest import seed_asset, seed_segment


@pytest.fixture
def conn(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")  # the validator only checks existence
    c = connect(tmp_path / "catalog" / "autovid.db")
    seed_asset(c, "a1", duration=30.0, path=str(src))
    yield c
    c.close()


def _item(clip_id: int, *, start: float = 0.0, end: float = 5.0,
          asset_id: str = "a1", **kw) -> EDLItem:
    return EDLItem(clip_id=clip_id, asset_id=asset_id,
                   source_start=start, source_end=end, **kw)


def test_valid_board_passes(conn) -> None:
    seed_segment(conn, "a1", 1, start=0.0, end=10.0)
    board = EDL(target_duration_s=5.0, items=[_item(1, end=5.0)])
    v = validate_edl(conn, board)
    assert v.ok, [e.message for e in v.errors]


def test_empty_and_unknown_segment_are_errors(conn) -> None:
    v = validate_edl(conn, EDL(items=[]))
    assert not v.ok and v.errors[0].code == "empty"

    v = validate_edl(conn, EDL(items=[_item(99)]))
    codes = [e.code for e in v.errors]
    assert "missing_segment" in codes


def test_bad_range_and_beyond_asset_block(conn, tmp_path) -> None:
    short = tmp_path / "short.mp4"
    short.write_bytes(b"\x00")
    seed_asset(conn, "a_short", duration=10.0, path=str(short))
    seed_segment(conn, "a_short", 1, start=0.0, end=10.0)
    v = validate_edl(conn, EDL(items=[_item(1, start=5.0, end=2.0, asset_id="a_short")]))
    assert any(e.code == "bad_range" for e in v.errors)

    v = validate_edl(conn, EDL(items=[_item(1, end=20.0, asset_id="a_short")]))
    assert any(e.code == "beyond_asset" for e in v.errors)


def test_beyond_segment_warns_but_renders(conn) -> None:
    """Items may cut outside the scene span (the source file is what gets
    cut) — provenance drift is a warning, not a blocker."""
    seed_segment(conn, "a1", 1, start=10.0, end=20.0)
    v = validate_edl(conn, EDL(items=[_item(1, start=8.0, end=12.0)]))
    assert v.ok
    assert any(w.code == "beyond_segment" for w in v.warnings)


def test_timeline_overlap_errors_gap_warns(conn) -> None:
    seed_segment(conn, "a1", 1, start=0.0, end=10.0)
    seed_segment(conn, "a1", 2, start=0.0, end=10.0)
    items = [
        _item(1, end=5.0, timeline_start=0.0, timeline_end=5.0),
        _item(2, end=5.0, timeline_start=4.0, timeline_end=9.0),  # overlap
    ]
    v = validate_edl(conn, EDL(items=items))
    assert any(e.code == "overlap" for e in v.errors)

    items[1].timeline_start = 6.5
    v = validate_edl(conn, EDL(items=items))
    assert v.ok
    assert any(w.code == "gap" for w in v.warnings)


def test_mixed_resolution_errors_mixed_fps_warns(conn, tmp_path) -> None:
    src = tmp_path / "src2.mp4"
    src.write_bytes(b"\x00")
    seed_asset(conn, "a1", duration=30.0, width=160, height=120, fps=12.0)
    seed_asset(conn, "a2", duration=30.0, width=320, height=240, fps=12.0, path=str(src))
    seed_segment(conn, "a1", 1)
    seed_segment(conn, "a2", 2)
    v = validate_edl(conn, EDL(items=[_item(1, asset_id="a1"), _item(2, asset_id="a2")]))
    assert any(e.code == "mixed_resolution" for e in v.errors)

    conn.execute("UPDATE assets SET width=160, height=120, fps=30 WHERE id='a2'")
    conn.commit()
    v = validate_edl(conn, EDL(items=[_item(1, asset_id="a1"), _item(2, asset_id="a2")]))
    assert v.ok
    assert any(w.code == "mixed_fps" for w in v.warnings)


def test_mixed_audio_blocks(conn, tmp_path) -> None:
    """Audio and silent sources cannot stream-copy concat — invalid EDL."""
    silent_src = tmp_path / "silent.mp4"
    silent_src.write_bytes(b"\x00")
    seed_asset(conn, "a_silent", duration=30.0, path=str(silent_src))
    conn.execute("UPDATE assets SET has_audio = 1 WHERE id = 'a1'")
    seed_segment(conn, "a1", 1)
    seed_segment(conn, "a_silent", 2)
    v = validate_edl(conn, EDL(items=[_item(1), _item(2, asset_id="a_silent")]))
    assert any(e.code == "mixed_audio" for e in v.errors)

    conn.execute("UPDATE assets SET has_audio = 1 WHERE id = 'a_silent'")
    v = validate_edl(conn, EDL(items=[_item(1), _item(2, asset_id="a_silent")]))
    assert v.ok, [e.message for e in v.errors]


def test_target_overshoot_warns(conn) -> None:
    seed_segment(conn, "a1", 1, start=0.0, end=30.0)
    v = validate_edl(conn, EDL(target_duration_s=10.0, items=[_item(1, end=15.0)]))
    assert v.ok
    assert any(w.code == "over_target" for w in v.warnings)
