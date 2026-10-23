"""Cut-stage tests: hierarchical phases 2–5 (continuity, refinement, catalog)."""
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
from autovid.ffmpeg_io.clips import clip_filename, extract_clip
from autovid.ffmpeg_io.probe import probe
from autovid.refine import REFINE_WINDOW_S, cut_asset
from autovid.vlm import VLMClient, WindowAnalysis
from tests.conftest import make_multiscene_video, make_video


def _analysis(**overrides) -> WindowAnalysis:
    base = dict(
        description="synthetic scene", activity="none", objects=[],
        environment="synthetic", camera_motion="static",
        highlight_score=0.8, uncertainty=0.1, continues_previous=False,
    )
    base.update(overrides)
    return WindowAnalysis(**base)


def _continuity(relation: str, change_level: int = 0):
    from autovid.decisions import ContinuityVerdict

    return ContinuityVerdict(
        relation=relation, shared_subjects=0.5, change_level=change_level,
        confidence=0.9, engine="test", elapsed_ms=0.0,
    )


def _boundary(verdict: str, hint: str = "at"):
    return BoundaryVerdict(verdict, hint, 0.9 if verdict == "boundary" else 0.4,
                           0.9, "test", 0.0)


def _segment(verdict: str):
    return SegmentVerdict(verdict, 0.9, 0.9, 2, 0.9, "test", 0.0)


@pytest.fixture
def analyzed(settings: Settings, tmp_path: Path):
    """Analyzed 3-scene clip (9 s, cuts at 3 s and 6 s) in 3 coarse windows."""
    conn = db.connect(settings.db_path)
    clip = make_multiscene_video(tmp_path / "ms.mp4", scene_s=3.0, scenes=3)
    from autovid.ingest import ingest_file

    _, asset_id = ingest_file(conn, settings, clip)
    sampling.sample_asset(conn, settings, asset_id)
    coarse = settings.model_copy(update={"coarse_window_s": 3.0})
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vlm, "describe_window", lambda **kw: _analysis())
    analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    monkeypatch.undo()
    return conn, coarse, asset_id


# ---------------------------------------------------------------------------
# clips.py (unchanged behaviour, kept as a contract)
# ---------------------------------------------------------------------------


def test_extract_clip_produces_accurate_span(tmp_path: Path) -> None:
    src = make_multiscene_video(tmp_path / "src.mp4", scene_s=3.0, scenes=3)
    out = tmp_path / "clip.mp4"
    extract_clip(src, out, start_s=3.0, end_s=6.0)
    info = probe(out)
    assert out.exists()
    assert abs(info.duration_s - 3.0) < 0.5
    assert info.video_codec == "h264"


def test_extract_clip_rejects_bad_span(tmp_path: Path) -> None:
    from autovid.errors import AutoVidError

    with pytest.raises(AutoVidError, match="invalid clip span"):
        extract_clip(tmp_path / "x.mp4", tmp_path / "o.mp4", start_s=5.0, end_s=5.0)


def test_clip_filename_carries_provenance() -> None:
    assert clip_filename("a_abc123", 3.0, 6.0) == "a_abc123__3.000-6.000.mp4"


# ---------------------------------------------------------------------------
# phases 2–4: continuity → refinement → merging
# ---------------------------------------------------------------------------


def test_same_scene_continuity_merges_into_one_scene(settings: Settings, analyzed,
                                                     monkeypatch) -> None:
    conn, coarse, asset_id = analyzed
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    monkeypatch.setattr(decisions, "decide_continuity",
                        lambda a, c, i, near_cut=False: _continuity("same_scene"))
    monkeypatch.setattr(decisions, "decide_segment",
                        lambda a, s, frames=None, duration_s=None,
                        min_duration_s=0.5: _segment("accept"))

    result = cut_asset(conn, coarse, asset_id, decisions=decisions)
    assert result.windows == 3
    assert result.scenes == 1

    scene = conn.execute(
        "SELECT start_s, end_s FROM scenes WHERE source = 'vlm-merge'"
    ).fetchone()
    assert (scene["start_s"], scene["end_s"]) == (0.0, 9.0)

    seg = conn.execute("SELECT description FROM segments").fetchone()
    assert "(+2 more windows, same scene)" in seg["description"]
    assert result.clips_extracted == 1


def test_different_scene_continuity_splits(settings: Settings, analyzed, monkeypatch) -> None:
    conn, coarse, asset_id = analyzed
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    monkeypatch.setattr(decisions, "decide_continuity",
                        lambda a, c, i, near_cut=False: _continuity("different_scene"))
    monkeypatch.setattr(decisions, "decide_segment",
                        lambda a, s, frames=None, duration_s=None, min_duration_s=0.5: _segment("accept"))

    result = cut_asset(conn, coarse, asset_id, decisions=decisions)
    assert result.scenes == 3
    assert result.segments_accepted == 3

    spans = [(r["start_s"], r["end_s"]) for r in conn.execute(
        "SELECT start_s, end_s FROM scenes WHERE source = 'vlm-merge' ORDER BY start_s"
    ).fetchall()]
    assert spans == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]


def test_needs_verification_with_near_cut_confirms_and_refines(
    settings: Settings, analyzed, monkeypatch
) -> None:
    conn, coarse, asset_id = analyzed
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)

    relations = iter([_continuity("needs_verification"), _continuity("needs_verification")])
    monkeypatch.setattr(decisions, "decide_continuity",
                        lambda a, c, i, near_cut=False: next(relations))

    # boundary verification CONFIRMS the first cut and SHIFTS the second
    boundary_calls: list[float] = []

    def decide_boundary(asset_id, context, before, after):
        ctx = json.loads(context)
        boundary_calls.append(ctx["candidate_ts"])
        if len(boundary_calls) == 1:
            return _boundary("boundary")
        return _boundary("shift", "after")

    monkeypatch.setattr(decisions, "decide_boundary", decide_boundary)
    monkeypatch.setattr(decisions, "decide_segment",
                        lambda a, s, frames=None, duration_s=None, min_duration_s=0.5: _segment("accept"))

    from autovid.refine import SHIFT_STEP_S

    result = cut_asset(conn, coarse, asset_id, decisions=decisions)
    assert result.boundaries_verified == 2
    assert result.boundaries_refined == 2
    assert boundary_calls == pytest.approx([3.0, 6.0])

    spans = [(r["start_s"], r["end_s"]) for r in conn.execute(
        "SELECT start_s, end_s FROM scenes WHERE source = 'vlm-merge' ORDER BY start_s"
    ).fetchall()]
    # edges refined to the verified cut positions (3.0 stays, 6.0 shifts later)
    assert spans == [(0.0, 3.0), (3.0, 6.0 + SHIFT_STEP_S), (6.0 + SHIFT_STEP_S, 9.0)]


def test_needs_verification_not_boundary_merges(settings: Settings, analyzed,
                                                monkeypatch) -> None:
    conn, coarse, asset_id = analyzed
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    monkeypatch.setattr(decisions, "decide_continuity",
                        lambda a, c, i, near_cut=False: _continuity("needs_verification"))
    monkeypatch.setattr(decisions, "decide_boundary",
                        lambda a, c, b, f: _boundary("not_boundary"))

    result = cut_asset(conn, coarse, asset_id, decisions=decisions)
    assert result.boundaries_verified == 2
    assert result.scenes == 1  # both boundaries refuted → one merged scene


def test_verification_without_near_cut_uses_description_change(
    settings: Settings, analyzed, monkeypatch
) -> None:
    conn, coarse, asset_id = analyzed
    # suppress the detector hints: no scdet rows → nothing is "near"
    conn.execute("DELETE FROM scenes WHERE source LIKE 'scdet%'")
    conn.commit()
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)

    rels = iter([_continuity("needs_verification", change_level=3),
                 _continuity("needs_verification", change_level=0)])
    monkeypatch.setattr(decisions, "decide_continuity",
                        lambda a, c, i, near_cut=False: next(rels))

    def no_boundary_calls(a, c, b, f):  # must never be reached without a near cut
        raise AssertionError("boundary verification requires a nearby detector cut")

    monkeypatch.setattr(decisions, "decide_boundary", no_boundary_calls)
    monkeypatch.setattr(decisions, "decide_segment",
                        lambda a, s, frames=None, duration_s=None, min_duration_s=0.5: _segment("accept"))

    result = cut_asset(conn, coarse, asset_id, decisions=decisions)
    # clear description change → split; subtle change → merge
    assert result.scenes == 2


# ---------------------------------------------------------------------------
# phase 5: acceptance + extraction
# ---------------------------------------------------------------------------


def test_cut_accepts_highlights_and_extracts_clips(settings: Settings, analyzed,
                                                   monkeypatch) -> None:
    conn, coarse, asset_id = analyzed
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    monkeypatch.setattr(decisions, "decide_continuity",
                        lambda a, c, i, near_cut=False: _continuity("different_scene"))
    monkeypatch.setattr(decisions, "decide_segment",
                        lambda a, s, frames=None, duration_s=None, min_duration_s=0.5: _segment("accept"))

    result = cut_asset(conn, coarse, asset_id, decisions=decisions)
    assert result.segments_accepted == 3
    assert result.clips_extracted == 3

    clips = conn.execute(
        "SELECT clip_path, start_s, end_s FROM segments WHERE asset_id = ? AND verdict = 'accept'",
        (asset_id,),
    ).fetchall()
    assert len(clips) == 3
    for row in clips:
        assert Path(row["clip_path"]).exists()
        info = probe(row["clip_path"])
        assert abs(info.duration_s - (row["end_s"] - row["start_s"])) < 0.5
    assert conn.execute(
        "SELECT status FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()["status"] == "cut"


def test_cut_reject_and_flag_go_to_review(settings: Settings, analyzed, monkeypatch) -> None:
    conn, coarse, asset_id = analyzed
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    monkeypatch.setattr(decisions, "decide_continuity",
                        lambda a, c, i, near_cut=False: _continuity("different_scene"))
    verdicts = iter([
        SegmentVerdict("reject", 0.2, 0.3, 0, 0.8, "test", 0.0),
        SegmentVerdict("needs_evidence", 0.5, 0.5, 1, 0.4, "test", 0.0),
        SegmentVerdict("flag_review", 0.9, 0.2, 2, 0.5, "test", 0.0),
    ])
    monkeypatch.setattr(
        decisions, "decide_segment",
        lambda a, s, frames=None, duration_s=None, min_duration_s=0.5: next(verdicts)
    )
    result = cut_asset(conn, coarse, asset_id, decisions=decisions)

    assert result.segments_rejected == 1
    assert result.segments_flagged == 2
    assert result.clips_extracted == 0
    rows = conn.execute(
        "SELECT review_status, verdict FROM segments WHERE asset_id = ? ORDER BY start_s",
        (asset_id,),
    ).fetchall()
    assert [r["verdict"] for r in rows] == ["reject", "needs_evidence", "flag_review"]
    assert [r["review_status"] for r in rows] == ["rejected", "pending", "pending"]


def test_low_highlights_are_not_proposed(settings: Settings, tmp_path: Path,
                                         monkeypatch) -> None:
    """Scenes below the highlight threshold never become segments."""
    conn = db.connect(settings.db_path)
    clip = make_multiscene_video(tmp_path / "boring.mp4", scene_s=3.0, scenes=2)
    from autovid.ingest import ingest_file

    _, asset_id = ingest_file(conn, settings, clip)
    sampling.sample_asset(conn, settings, asset_id)
    coarse = settings.model_copy(update={"coarse_window_s": 3.0})
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)
    monkeypatch.setattr(
        vlm, "describe_window",
        lambda **kw: _analysis(highlight_score=0.1, description="dead time"),
    )
    analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)

    result = cut_asset(conn, coarse, asset_id, decisions=decisions)
    assert result.segments_accepted == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"] == 0


def test_cold_start_without_vlm_rescues_via_review(settings: Settings, tmp_path: Path,
                                                   monkeypatch) -> None:
    """No VLM key: scenes become pending placeholders; approval + render still work."""
    conn = db.connect(settings.db_path)
    clip = make_multiscene_video(tmp_path / "cold.mp4", scene_s=3.0, scenes=2)
    from autovid.ingest import ingest_file

    _, asset_id = ingest_file(conn, settings, clip)
    sampling.sample_asset(conn, settings, asset_id)
    coarse = settings.model_copy(update={"coarse_window_s": 3.0})
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)  # no key

    analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    result = cut_asset(conn, coarse, asset_id, decisions=decisions)
    assert result.segments_flagged == 2  # two unanalyzed scenes awaiting review

    segs = conn.execute(
        "SELECT id, review_status FROM segments WHERE asset_id = ? ORDER BY start_s",
        (asset_id,),
    ).fetchall()
    assert all(s["review_status"] == "pending" for s in segs)

    from autovid.organize import organize_asset
    from autovid.storyboard import select_clips
    from autovid.render import render_edl
    from autovid.ffmpeg_io.probe import probe

    for s in segs:
        conn.execute(
            "UPDATE segments SET review_status = 'approved' WHERE id = ?", (s["id"],)
        )
    conn.commit()
    organize_asset(conn, coarse, asset_id)

    board = select_clips(conn, objective="cold start", target_duration_s=6.0, name="cold")
    assert len(board.items) >= 1
    out = render_edl(conn, coarse, board)
    assert out.exists()
    assert probe(out).duration_s > 1.0
    # lazily extracted per-item clips are staged for the render, then cleaned up
    assert list(out.parent.glob("item_*.mp4")) == []


def test_speech_driven_assets_skip_the_visual_highlight_gate(settings, tmp_path, monkeypatch):
    """Talking footage scores low visually; its low-highlight scenes must still reach the decision layer."""
    import autovid.refine as refine
    from types import SimpleNamespace

    conn = db.connect(tmp_path / "g.db")
    conn.execute("INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s)"
                 " VALUES ('a', 's', 'p', 'p', 'a.mp4', 10)")
    scene = {"start": 0.0, "end": 5.0, "description": "two friends chatting", "extra_windows": 0,
             "windows": [{"data": {}, "summary": {}}], "highlight": 0.1, "tags": []}
    monkeypatch.setattr(refine, "_scene_evidence", lambda *a, **k: ["frame"])
    decisions = SimpleNamespace(decide_segment=lambda *a, **k: SimpleNamespace(
        verdict="accept", confidence=0.8, quality_level="ok"))

    refine.catalog_scenes(conn, settings, "a", Path("p"), [scene], decisions)
    assert conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0] == 0          # dropped by the gate
    refine.catalog_scenes(conn, settings, "a", Path("p"), [scene], decisions, ignore_highlight=True)
    assert conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0] == 1          # kept for speech assets
