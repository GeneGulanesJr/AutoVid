"""Decision-client tests: tool translation, logging, fallbacks (no server needed)."""
from __future__ import annotations

import json

import pytest

from autovid import db
from autovid.config import Settings
from autovid.decisions import (
    DecisionClient,
    DecisionProtocolError,
    DecisionUnavailable,
    RegionStats,
)


@pytest.fixture
def conn(settings: Settings):
    conn = db.connect(settings.db_path)
    # decisions.asset_id → assets(id) is FK-enforced; log a minimal asset.
    conn.execute(
        """INSERT INTO assets (id, sha256, path, original_path, original_name, size_bytes)
           VALUES ('a_test', 'deadbeef', 'C:/x.mp4', 'C:/x.mp4', 'x.mp4', 1)"""
    )
    conn.commit()
    yield conn
    conn.close()


def _stats(**overrides) -> RegionStats:
    base = dict(
        asset_id="a_test", region_id="r0", start_s=0.0, end_s=10.0,
        scene_changes=0, mean_motion=0.02, max_motion=0.05,
        current_interval_s=2.0, vlm_uncertainty=None,
    )
    base.update(overrides)
    return RegionStats(**base)


def _region_tool_raw(action: str, suggested: float | None, model: str = "d1/test") -> dict:
    return {
        "action": action,
        "suggested_interval_s": suggested,
        "has_content": 0.8,
        "ambiguity_level": 1,
        "ambiguity_text": "mostly clear",
        "confidence": 0.9,
        "details": {"answers": {}},
        "model": model,
    }


def test_region_translates_tool_output_and_logs(settings: Settings, conn, monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_call(name, args):
        calls.append((name, args))
        return _region_tool_raw("densify", 1.0)

    client = DecisionClient(settings, conn)
    monkeypatch.setattr(client, "_call_tool", fake_call)

    out = client.decide_region(_stats())
    assert out.action == "densify"
    assert out.suggested_interval_s == pytest.approx(1.0)
    assert out.confidence == pytest.approx(0.9)
    assert out.engine == "d1/test"
    assert calls and calls[0][0] == "decision_video_region"
    assert json.loads(calls[0][1]["summary"])["asset_id"] == "a_test"

    row = conn.execute("SELECT * FROM decisions").fetchone()
    assert row["tool"] == "decision_video_region"
    assert row["action"] == "densify"
    assert row["engine_version"] == "d1/test"
    assert row["confidence"] == pytest.approx(0.9)


def test_region_falls_back_to_heuristic_when_unreachable(settings: Settings, conn,
                                                         monkeypatch) -> None:
    def unreachable(name, args):
        raise DecisionUnavailable("down")

    settings_auto = settings.model_copy(update={"decision_mode": "auto"})
    client = DecisionClient(settings_auto, conn)
    monkeypatch.setattr(client, "_call_tool", unreachable)

    out = client.decide_region(_stats(mean_motion=0.4))
    assert out.action == "densify"
    assert out.suggested_interval_s == pytest.approx(1.0)
    assert out.engine == "heuristic"
    assert out.confidence == 0.0
    row = conn.execute("SELECT engine_version FROM decisions").fetchone()
    assert row["engine_version"] == "heuristic"


def test_region_d1_mode_requires_the_server(settings: Settings, conn, monkeypatch) -> None:
    settings_d1 = settings.model_copy(update={"decision_mode": "d1"})
    client = DecisionClient(settings_d1, conn)
    monkeypatch.setattr(client, "_call_tool", lambda n, a: (_ for _ in ()).throw(
        DecisionUnavailable("down")))

    with pytest.raises(DecisionUnavailable):
        client.decide_region(_stats())


def test_region_heuristic_mode_never_calls_the_server(settings: Settings, conn,
                                                      monkeypatch) -> None:
    def boom(name, args):
        raise AssertionError("server must not be called in heuristic mode")

    settings_h = settings.model_copy(update={"decision_mode": "heuristic"})
    client = DecisionClient(settings_h, conn)
    monkeypatch.setattr(client, "_call_tool", boom)

    assert client.decide_region(_stats(mean_motion=0.05)).action == "sufficient"
    assert client.decide_region(_stats(mean_motion=0.12)).action == "inspect_more"
    assert client.decide_region(_stats(scene_changes=2)).action == "densify"
    assert client.decide_region(_stats(vlm_uncertainty=0.7)).action == "flag_review"


def test_boundary_translates_and_gates(settings: Settings, conn, monkeypatch) -> None:
    client = DecisionClient(settings, conn)

    monkeypatch.setattr(client, "_call_tool", lambda n, a: {
        "verdict": "boundary", "position_hint": "at", "cut_probability": 0.9,
        "confidence": 0.85, "details": {}, "model": "d1/test",
    })
    out = client.decide_boundary("a_test", "t=3.0", ["<a>"], ["<b>"])
    assert out.verdict == "boundary" and out.engine == "d1/test"

    monkeypatch.setattr(client, "_call_tool", lambda n, a: (_ for _ in ()).throw(
        DecisionUnavailable("down")))
    out = client.decide_boundary("a_test", "t=3.0", ["<a>"], ["<b>"])
    assert out.verdict == "shift" and out.confidence == 0.0  # never claims unseen cuts
    assert conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"] == 2


def test_segment_fallback_needs_evidence(settings: Settings, conn, monkeypatch) -> None:
    client = DecisionClient(settings, conn)
    monkeypatch.setattr(client, "_call_tool", lambda n, a: (_ for _ in ()).throw(
        DecisionUnavailable("down")))

    out = client.decide_segment("a_test", "a clip", ["<f>"], duration_s=6.0)
    assert out.verdict == "needs_evidence"
    assert out.engine == "heuristic"


def test_tool_shape_mismatch_is_decision_unavailable(settings: Settings, conn,
                                                     monkeypatch) -> None:
    client = DecisionClient(settings, conn)
    monkeypatch.setattr(client, "_call_tool", lambda n, a: {"unexpected": 1})

    # Protocol drift is loud: no silent heuristic fallback.
    with pytest.raises(DecisionProtocolError, match="unexpected region output"):
        client.decide_region(_stats())
