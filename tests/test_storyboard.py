"""Storyboard + EDL + render tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autovid import db, sampling
from autovid.analyze import analyze_asset
from autovid.config import Settings
from autovid.decisions import (
    BoundaryVerdict,
    DecisionClient,
    SegmentVerdict,
)
from autovid.edl import EDL, EDLItem, list_edls, load, save
from autovid.errors import AutoVidError
from autovid.ffmpeg_io.probe import probe
from autovid.refine import cut_asset
from autovid.render import render_edl
from autovid.storyboard import load_candidates, select_clips
from autovid.vlm import VLMClient, WindowAnalysis
from tests.conftest import make_multiscene_video


def _analysis(**overrides) -> WindowAnalysis:
    base = dict(
        description="market scene", activity="walking", objects=["stalls"],
        environment="market", camera_motion="static",
        highlight_score=0.8, uncertainty=0.1, continues_previous=False,
    )
    base.update(overrides)
    return WindowAnalysis(**base)


@pytest.fixture
def catalog(settings: Settings, tmp_path: Path, monkeypatch):
    """Cut+organized catalog: 3 accepted segments (3 s each, one per scene)."""
    conn = db.connect(settings.db_path)
    clip = make_multiscene_video(tmp_path / "ms.mp4", scene_s=3.0, scenes=3)
    from autovid.ingest import ingest_file

    _, asset_id = ingest_file(conn, settings, clip)
    sampling.sample_asset(conn, settings, asset_id)
    coarse = settings.model_copy(update={"coarse_window_s": 3.0})
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)
    monkeypatch.setattr(vlm, "describe_window", lambda **kw: _analysis())
    analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    monkeypatch.setattr(
        decisions, "decide_boundary",
        lambda a, c, b, f: BoundaryVerdict("boundary", "at", 0.9, 0.9, "test", 0.0),
    )
    monkeypatch.setattr(
        decisions, "decide_segment",
        lambda a, s, frames, duration_s=None, min_duration_s=0.5: SegmentVerdict(
            "accept", 0.9, 0.9, 2, 0.9, "test", 0.0
        ),
    )
    cut_asset(conn, coarse, asset_id, decisions=decisions)
    return conn, coarse, asset_id


# ---------------------------------------------------------------------------
# EDL persistence
# ---------------------------------------------------------------------------


def test_edl_roundtrip(settings: Settings) -> None:
    conn = db.connect(settings.db_path)
    board = EDL(
        name="intro", objective="recap", style="fast_paced", target_duration_s=10.0,
        items=[EDLItem(clip_id=1, asset_id="a_x", source_start=0.0, source_end=3.0,
                       transition_in="fade")],
    )
    save(conn, board)
    again = load(conn, board.id)
    assert again.name == "intro"
    assert again.items[0].transition_in == "fade"
    assert again.duration_s == pytest.approx(3.0)

    board.objective = "changed"
    save(conn, board)
    assert load(conn, board.id).objective == "changed"
    assert len(list_edls(conn)) == 1

    with pytest.raises(AutoVidError, match="Unknown EDL"):
        load(conn, "nope")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_selection_respects_target_duration(settings: Settings, catalog) -> None:
    conn, settings, asset_id = catalog
    board = select_clips(conn, objective="short teaser", target_duration_s=6.0,
                         style="cinematic", name="teaser")
    assert board.duration_s <= 6.0 * 1.2
    assert 1 <= len(board.items) <= 2
    # persisted
    assert load(conn, board.id).id == board.id


def test_selection_uses_everything_when_pool_is_small(settings: Settings, catalog) -> None:
    conn, settings, asset_id = catalog
    board = select_clips(conn, objective="long recap", target_duration_s=120.0,
                         name="recap")
    assert len(board.items) == 3  # pool exhausted (3×3 s), within tolerance rules


def test_selection_shape_open_earliest_close_strongest(settings: Settings, catalog) -> None:
    conn, settings, asset_id = catalog
    # Give each segment a distinct highlight; the last one becomes the closer.
    rows = conn.execute(
        "SELECT id, start_s FROM segments ORDER BY start_s"
    ).fetchall()
    scores = {rows[0]["id"]: 0.5, rows[1]["id"]: 0.3, rows[2]["id"]: 0.9}
    for seg_id, score in scores.items():
        conn.execute(
            "UPDATE segments SET highlight_score = ? WHERE id = ?", (score, seg_id)
        )
    conn.commit()

    board = select_clips(conn, objective="shaped", target_duration_s=9.0, name="shaped")
    by_clip = {item.clip_id: item for item in board.items}
    order = [item.clip_id for item in board.items]
    # earliest source opens…
    assert by_clip[order[0]].source_start == min(i.source_start for i in board.items)
    # …and the strongest highlight closes
    strongest = max(scores, key=lambda sid: scores[sid])
    assert order[-1] == strongest


def test_selection_requires_accepted_segments(settings: Settings, tmp_path: Path) -> None:
    conn = db.connect(settings.db_path)  # empty catalog
    with pytest.raises(ValueError, match="no accepted segments"):
        select_clips(conn, objective="x", target_duration_s=10.0)


def test_candidates_exclude_pending_review(settings: Settings, catalog) -> None:
    conn, settings, asset_id = catalog
    conn.execute("UPDATE segments SET review_status = 'pending' WHERE start_s = 0.0")
    conn.commit()
    cands = load_candidates(conn)
    assert all(c.start_s > 0.0 for c in cands)


def test_human_approval_makes_flagged_segment_eligible(settings: Settings, catalog) -> None:
    """The fallback path: needs_evidence + human approval → storyboard-eligible."""
    conn, settings, asset_id = catalog
    conn.execute(
        "UPDATE segments SET verdict = 'needs_evidence', review_status = 'pending' "
        "WHERE start_s = 0.0"
    )
    conn.commit()
    assert all(c.start_s > 0.0 for c in load_candidates(conn))

    conn.execute(
        "UPDATE segments SET review_status = 'approved' WHERE start_s = 0.0"
    )
    conn.commit()
    starts = sorted(c.start_s for c in load_candidates(conn))
    assert starts == [0.0, 3.0, 6.0]


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_concatenates_clips(settings: Settings, catalog) -> None:
    conn, settings, asset_id = catalog
    board = select_clips(conn, objective="full", target_duration_s=9.0, name="full")
    out = render_edl(conn, settings, board)
    assert out.exists()
    info = probe(out)
    assert abs(info.duration_s - board.duration_s) < 1.0
    assert info.video_codec == "h264"
    # staged artifacts cleaned up
    assert list(out.parent.glob("faded_*")) == []
    assert not (out.parent / "concat.txt").exists()


def test_render_with_fade_transition(settings: Settings, catalog) -> None:
    conn, settings, asset_id = catalog
    board = select_clips(conn, objective="fade", target_duration_s=9.0, name="fades")
    board.items[0].transition_out = "fade"
    board.items[1].transition_in = "fade"
    save(conn, board)
    out = render_edl(conn, settings, board)
    assert out.exists()
    # ~9 s expected: fades re-encode but keep durations
    assert abs(probe(out).duration_s - 9.0) < 1.5


def test_render_requires_clips(settings: Settings) -> None:
    conn = db.connect(settings.db_path)
    board = EDL(name="ghost", items=[EDLItem(clip_id=999, asset_id="a_x",
                                             source_start=0, source_end=1)])
    with pytest.raises(AutoVidError, match="unknown segment 999"):
        render_edl(conn, settings, board)
