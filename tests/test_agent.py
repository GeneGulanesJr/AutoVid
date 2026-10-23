"""Storyboard Agent: hallucination gate, repair round, planner fallback,
scoped regeneration with locked-scene preservation. All LLM calls faked —
the suite stays hermetic (no network, no key)."""
from __future__ import annotations

import json

import pytest

from autovid import storyboard_doc as doc_mod
from autovid.agent import (
    AgentUnavailable,
    StoryboardAgent,
    generate_with_fallback,
    merge_locked,
    validate_document,
)
from autovid.config import Settings
from autovid.db import connect
from autovid.storyboard_doc import (
    SceneShot,
    ScriptBeat,
    ScriptDocument,
    StoryboardDocument,
    StoryboardScene,
    create_board,
    load_document,
    load_script,
)
from tests.conftest import seed_asset, seed_segment


@pytest.fixture
def conn(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")  # compile-time validation checks source existence
    c = connect(tmp_path / "catalog" / "autovid.db")
    seed_asset(c, "a1", duration=30.0, path=str(src))
    seed_segment(c, "a1", 1, start=0.0, end=10.0, description="a campfire",
                 tags=["campfire", "night"])
    seed_segment(c, "a1", 2, start=10.0, end=20.0, description="a hike",
                 tags=["hike", "day"])
    yield c
    c.close()


def _full_payload() -> dict:
    return {
        "title": "Trip Recap",
        "synopsis": "Fire then trail.",
        "beats": [
            {"id": "b1", "role": "hook", "narration": "Night one.",
             "on_screen": "DAY 1", "dialogue": ""},
            {"id": "b2", "role": "resolution", "narration": "Then we walked.",
             "on_screen": "", "dialogue": ""},
        ],
        "scenes": [
            {"id": "s1", "title": "Fire", "description": "campfire close-up",
             "narrative_role": "hook", "beat_id": "b1",
             "shots": [{"segment_id": 1, "source_start": 0.0, "source_end": 4.0,
                        "transition_out": "cut"}],
             "pacing": "slow", "transition": "cut", "open_issues": [],
             "confidence": 0.9},
            {"id": "s2", "title": "Trail", "description": "hiking ridge",
             "narrative_role": "resolution", "beat_id": "b2",
             "shots": [{"segment_id": 2, "source_start": 10.0, "source_end": 14.0,
                        "transition_out": "fade"}],
             "pacing": "", "transition": "fade", "open_issues": [],
             "confidence": 0.8},
        ],
        "alternates": [{"scene_id": "s1", "segment_ids": [2]}],
        "missing_footage": ["a drone establishing shot"],
    }


class FakeClient:
    """VLMClient stand-in: replies canned answers, records prompts."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[str] = []

    def complete_text(self, *, system: str, user: str, model=None,
                      temperature: float = 0.3) -> str:
        self.calls.append(user)
        if not self.answers:
            raise AssertionError("FakeClient ran out of canned answers")
        return self.answers.pop(0)


def test_validate_document_catches_hallucinations(conn, settings: Settings) -> None:
    agent = StoryboardAgent(settings, FakeClient([]))
    document, script = agent._parse_full(json.dumps(_full_payload()), objective="x")
    assert validate_document(conn, document, script) == []

    document.scenes[0].shots[0].segment_id = 999   # invented segment
    document.scenes[1].beat_id = "bX"              # invented beat
    problems = validate_document(conn, document, script)
    assert any("unknown segment 999" in p for p in problems)
    assert any("unknown beat" in p for p in problems)


def test_generate_persists_agent_storyboard(conn, settings: Settings) -> None:
    client = FakeClient([json.dumps(_full_payload())])
    board_id = StoryboardAgent(settings, client).generate(
        conn, objective="recap", target_duration_s=8.0, style="cinematic", name="sb",
    )
    assert len(client.calls) == 1
    row = doc_mod.get_board(conn, board_id)
    assert row["name"] == "Trip Recap"
    versions = doc_mod.list_versions(conn, board_id)
    assert versions[0]["origin"] == "agent"
    doc, _ = load_document(conn, board_id)
    assert doc.missing_footage == ["a drone establishing shot"]
    assert doc.scenes[0].alternates == [2]
    # compile works even though the agent left asset_id blank: the catalog resolves it
    board = doc_mod.compile_to_edl(conn, board_id)
    assert board.items[0].asset_id == "a1"
    script = load_script(conn, board_id)
    assert script.title == "Trip Recap"


def test_invalid_draft_gets_one_repair_round(conn, settings: Settings) -> None:
    bad = _full_payload()
    bad["scenes"][0]["shots"][0]["segment_id"] = 999
    client = FakeClient([json.dumps(bad), json.dumps(_full_payload())])
    board_id = StoryboardAgent(settings, client).generate(
        conn, objective="recap", target_duration_s=8.0, style="cinematic", name="sb",
    )
    assert len(client.calls) == 2, "the violations go back for exactly one repair round"
    assert "999" in client.calls[1]
    assert doc_mod.get_board(conn, board_id)["current_version"] == 1


def test_persistently_invalid_draft_falls_back_to_planner(conn, settings: Settings) -> None:
    bad = _full_payload()
    bad["scenes"][0]["shots"][0]["segment_id"] = 999
    client = FakeClient([json.dumps(bad), json.dumps(bad)])
    with pytest.raises(AgentUnavailable):
        StoryboardAgent(settings, client).generate(
            conn, objective="recap", target_duration_s=8.0, style="cinematic", name="sb")

    # the documented fallback path produces a planner storyboard
    board_id, origin = generate_with_fallback(
        conn, settings, objective="recap", target_duration_s=8.0,
        style="cinematic", name="sb2",
    )
    assert origin == "planner"
    assert doc_mod.list_versions(conn, board_id)[0]["origin"] == "planner"


def test_generate_with_fallback_without_key_uses_planner(conn, settings: Settings) -> None:
    assert not settings.vlm_api_key
    board_id, origin = generate_with_fallback(
        conn, settings, objective="recap", target_duration_s=8.0,
        style="cinematic", name="sb3",
    )
    assert origin == "planner"
    doc, _ = load_document(conn, board_id)
    assert doc.scenes


def test_merge_locked_preserves_positions() -> None:
    old = StoryboardDocument(scenes=[
        StoryboardScene(id="s1", locked=True),
        StoryboardScene(id="s2"),
        StoryboardScene(id="s3", locked=True),
    ])
    new = StoryboardDocument(scenes=[
        StoryboardScene(id="n1"), StoryboardScene(id="n2"), StoryboardScene(id="n3"),
    ])
    merged = merge_locked(old, new)
    assert [s.id for s in merged.scenes] == ["s1", "n1", "s3", "n2", "n3"]


def test_regenerate_single_scene(conn, settings: Settings) -> None:
    board_id = create_board(
        conn, name="sb", objective="recap", style="cinematic",
        target_duration_s=8.0,
        document=StoryboardDocument(scenes=[
            StoryboardScene(id="s1", title="Fire", beat_id="b1",
                            shots=[SceneShot(segment_id=1, asset_id="a1",
                                             source_start=0.0, source_end=4.0)]),
        ]),
        script=ScriptDocument(storyboard_id="", beats=[
            ScriptBeat(id="b1", role="hook", narration="Night one."),
        ]),
    )
    replacement = {
        "id": "s1", "title": "Fire wide", "description": "wider shot",
        "narrative_role": "hook", "beat_id": "b1",
        "shots": [{"segment_id": 2, "source_start": 10.0, "source_end": 13.0,
                   "transition_out": "cut"}],
        "pacing": "", "transition": "cut", "open_issues": [], "confidence": 0.7,
    }
    agent = StoryboardAgent(settings, FakeClient([json.dumps(replacement)]))
    version = agent.regenerate(conn, board_id, scene_id="s1")
    assert version == 2
    doc, _ = load_document(conn, board_id)
    assert doc.scenes[0].title == "Fire wide"
    assert doc.scenes[0].shots[0].segment_id == 2

    # a failing scene regeneration keeps the current scene and records why
    agent = StoryboardAgent(settings, FakeClient([json.dumps({"shots": []})]))
    version = agent.regenerate(conn, board_id, scene_id="s1")
    doc, _ = load_document(conn, board_id)
    assert doc.scenes[0].title == "Fire wide"  # unchanged
    assert any("regeneration failed" in i for i in doc.scenes[0].open_issues)
    assert version == 3


def test_regenerate_all_invalid_raises_board_untouched(conn, settings: Settings) -> None:
    board_id = create_board(
        conn, name="sb", objective="", style="", target_duration_s=0,
        document=StoryboardDocument(scenes=[StoryboardScene(id="s1")]),
        script=ScriptDocument(storyboard_id=""),
    )
    bad = _full_payload()
    bad["scenes"][0]["shots"][0]["segment_id"] = 999
    agent = StoryboardAgent(settings, FakeClient([json.dumps(bad)]))
    with pytest.raises(AgentUnavailable):
        agent.regenerate(conn, board_id)
    doc, v = load_document(conn, board_id)
    assert v == 1 and doc.scenes[0].id == "s1"


def test_regenerate_all_keeps_locked_scenes(conn, settings: Settings) -> None:
    board_id = create_board(
        conn, name="sb", objective="recap", style="cinematic",
        target_duration_s=8.0,
        document=StoryboardDocument(scenes=[
            StoryboardScene(id="s1", locked=True, title="Locked open",
                            shots=[SceneShot(segment_id=1, asset_id="a1",
                                             source_start=0.0, source_end=3.0)]),
        ]),
        script=ScriptDocument(storyboard_id=""),
    )
    agent = StoryboardAgent(settings, FakeClient([json.dumps(_full_payload())]))
    agent.regenerate(conn, board_id)
    doc, _ = load_document(conn, board_id)
    # the locked scene keeps its position and content; the fresh scenes fill
    # in behind it with collision-free ids
    assert [s.id for s in doc.scenes] == ["s1", "s2", "s3"]
    assert [s.title for s in doc.scenes] == ["Locked open", "Fire", "Trail"]
