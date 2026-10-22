"""Analysis-stage tests: coarse windows, adaptive frames, temporal map (mocked VLM)."""
from __future__ import annotations

from pathlib import Path

import pytest

from autovid import db, sampling
from autovid.analyze import analyze_asset, build_windows
from autovid.config import Settings
from autovid.decisions import DecisionClient
from autovid.vlm import VLMClient, VLMUnavailable, WindowAnalysis
from tests.conftest import make_multiscene_video


def _analysis(**overrides) -> WindowAnalysis:
    base = dict(
        description="colored test pattern",
        activity="none",
        objects=["bars"],
        environment="synthetic",
        camera_motion="static",
        highlight_score=0.3,
        uncertainty=0.1,
        continues_previous=False,
    )
    base.update(overrides)
    return WindowAnalysis(**base)


@pytest.fixture
def sampled(settings: Settings, tmp_path: Path):
    """A sampled 3-scene clip (cuts at 3 s and 6 s, 9 s total) + coarse 3 s windows."""
    conn = db.connect(settings.db_path)
    clip = make_multiscene_video(tmp_path / "ms.mp4", scene_s=3.0, scenes=3)
    from autovid.ingest import ingest_file

    _, asset_id = ingest_file(conn, settings, clip)
    sampling.sample_asset(conn, settings, asset_id)
    coarse = settings.model_copy(update={"coarse_window_s": 3.0})
    return conn, coarse, asset_id


def test_build_windows_is_fixed_time_grid() -> None:
    # pure time windows, independent of any scene cuts
    assert build_windows(130.0, coarse_window_s=60.0) == [
        (0.0, 60.0), (60.0, 120.0), (120.0, 130.0)
    ]
    assert build_windows(9.0, coarse_window_s=3.0) == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]


def test_build_windows_single_window_when_short() -> None:
    assert build_windows(9.0, coarse_window_s=60.0) == [(0.0, 9.0)]
    assert build_windows(130.0, coarse_window_s=0) == [(0.0, 60.0), (60.0, 120.0), (120.0, 130.0)]


def test_analyze_builds_temporal_map(settings: Settings, sampled, monkeypatch) -> None:
    conn, coarse, asset_id = sampled
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)

    seen: list = []

    def fake_describe_window(*, frames, previous_description=None, window_hint=""):
        seen.append((len(frames), previous_description))
        return _analysis(description=f"window #{len(seen)}")

    monkeypatch.setattr(vlm, "describe_window", fake_describe_window)

    result = analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    assert result.windows == 3
    assert result.unparsed == 0
    assert result.flagged == 0

    rows = conn.execute(
        "SELECT * FROM analyses WHERE asset_id = ? ORDER BY window_start", (asset_id,)
    ).fetchall()
    assert [(r["window_start"], r["window_end"]) for r in rows] == [
        (0.0, 3.0), (3.0, 6.0), (6.0, 9.0)
    ]
    assert all(r["provider"] == coarse.vlm_model for r in rows)
    assert all(r["confidence"] == pytest.approx(0.9) for r in rows)

    # Continuity: each call after the first saw the previous window's description.
    assert seen[0][1] is None
    assert seen[1][1] == "window #1"
    assert seen[2][1] == "window #2"

    assert conn.execute(
        "SELECT status FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()["status"] == "analyzed"
    assert db.job_done(conn, asset_id, "analyze")


def test_analyze_densifies_on_decision(settings: Settings, sampled, monkeypatch) -> None:
    conn, coarse, asset_id = sampled
    from autovid.decisions import RegionAction

    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    monkeypatch.setattr(
        decisions,
        "decide_region",
        lambda stats: RegionAction(
            action="densify", suggested_interval_s=1.0, has_content=0.9,
            ambiguity_level=1, confidence=0.9, engine="heuristic", elapsed_ms=0.0,
        ),
    )
    vlm = VLMClient(settings)
    monkeypatch.setattr(vlm, "describe_window", lambda **kw: _analysis())

    result = analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    assert result.dense_frames >= 3  # 3 windows × ~2 dense frames at 1 s interval

    dense = conn.execute(
        "SELECT COUNT(*) AS n FROM frames WHERE asset_id = ? AND kind = 'dense'",
        (asset_id,),
    ).fetchone()["n"]
    assert dense == result.dense_frames

    row = conn.execute(
        "SELECT response_json FROM analyses LIMIT 1"
    ).fetchone()
    import json

    assert json.loads(row["response_json"])["decision_action"] == "densify"


def test_analyze_survives_vlm_outage(settings: Settings, sampled, monkeypatch) -> None:
    conn, coarse, asset_id = sampled
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)

    def down(*, frames, previous_description=None, window_hint=""):
        raise VLMUnavailable("no key configured")

    monkeypatch.setattr(vlm, "describe_window", down)

    result = analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    assert result.windows == 3
    assert result.unparsed == 3
    assert result.flagged == 3  # unanalyzed windows must surface for review

    rows = conn.execute("SELECT * FROM analyses WHERE asset_id = ?", (asset_id,)).fetchall()
    assert all(r["provider"] == "none" for r in rows)
    assert all(r["confidence"] == 0.0 for r in rows)
    # Still 'analyzed': the map exists, its gaps are flagged — not a hard failure.
    assert conn.execute(
        "SELECT status FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()["status"] == "analyzed"


def test_analyze_flags_high_uncertainty(settings: Settings, sampled, monkeypatch) -> None:
    conn, coarse, asset_id = sampled
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)
    monkeypatch.setattr(
        vlm, "describe_window", lambda **kw: _analysis(uncertainty=0.9)
    )
    result = analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    assert result.flagged == 3
