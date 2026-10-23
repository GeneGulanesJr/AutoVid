"""Storyboard document: versioned persistence, script-stale propagation,
compile → EDL, deterministic fallback."""
from __future__ import annotations

import pytest

from autovid.db import connect
from autovid.errors import AutoVidError
from autovid.storyboard_doc import (
    SceneShot,
    ScriptBeat,
    ScriptDocument,
    StoryboardDocument,
    StoryboardScene,
    board_from_edl,
    build_deterministic,
    compile_to_edl,
    create_board,
    get_board,
    list_versions,
    load_document,
    load_script,
    save_document,
    save_script,
)
from tests.conftest import seed_asset, seed_segment


@pytest.fixture
def conn(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")  # the validator only checks existence
    c = connect(tmp_path / "catalog" / "autovid.db")
    seed_asset(c, "a1", duration=30.0, path=str(src))
    seed_segment(c, "a1", 1, start=0.0, end=10.0)
    seed_segment(c, "a1", 2, start=10.0, end=20.0)
    yield c
    c.close()


def _doc(*, beat2: str | None = "b2") -> StoryboardDocument:
    return StoryboardDocument(scenes=[
        StoryboardScene(
            id="s1", title="Open", narrative_role="hook", beat_id="b1",
            shots=[SceneShot(segment_id=1, asset_id="a1",
                             source_start=0.0, source_end=4.0)],
            est_duration_s=4.0,
        ),
        StoryboardScene(
            id="s2", title="Close", narrative_role="resolution", beat_id=beat2,
            shots=[SceneShot(segment_id=2, asset_id="a1",
                             source_start=10.0, source_end=15.0,
                             transition_out="fade")],
            est_duration_s=5.0,
        ),
    ])


def _script(board_id: str) -> ScriptDocument:
    return ScriptDocument(
        storyboard_id=board_id, title="Test", synopsis="A test.",
        beats=[ScriptBeat(id="b1", role="hook", narration="hello"),
               ScriptBeat(id="b2", role="resolution", narration="bye")],
    )


def test_create_load_and_version_history(conn) -> None:
    board_id = create_board(
        conn, name="test", objective="obj", style="cinematic",
        target_duration_s=9.0, document=_doc(), script=_script(""),
        origin="agent",
    )
    row = get_board(conn, board_id)
    assert row["status"] == "draft" and row["current_version"] == 1

    doc, v1 = load_document(conn, board_id)
    assert v1 == 1 and len(doc.scenes) == 2
    script = load_script(conn, board_id)
    assert script is not None and [b.id for b in script.beats] == ["b1", "b2"]

    doc.scenes.reverse()
    v2 = save_document(conn, board_id, doc, origin="user_edit", note="reorder")
    assert v2 == 2
    assert [s.id for s in load_document(conn, board_id)[0].scenes] == ["s2", "s1"]
    # history intact: v1 still loads with the original order
    old, v = load_document(conn, board_id, 1)
    assert v == 1 and [s.id for s in old.scenes] == ["s1", "s2"]
    assert [(r["version"], r["origin"]) for r in list_versions(conn, board_id)] == \
        [(1, "agent"), (2, "user_edit")]


def test_script_edit_marks_dependent_scenes_stale(conn) -> None:
    board_id = create_board(
        conn, name="test", objective="", style="", target_duration_s=0,
        document=_doc(), script=_script(""),
    )
    script = load_script(conn, board_id)
    script.beats[0].narration = "changed narration"
    save_script(conn, script)

    doc, _ = load_document(conn, board_id)
    by_id = {s.id: s for s in doc.scenes}
    assert by_id["s1"].stale is True    # illustrates the changed beat
    assert by_id["s2"].stale is False   # untouched beat
    versions = list_versions(conn, board_id)
    assert versions[-1]["origin"] == "script_edit"


def test_compile_to_edl_and_recompile_keeps_id(conn) -> None:
    board_id = create_board(
        conn, name="my-cut", objective="obj", style="cinematic",
        target_duration_s=9.0, document=_doc(), script=_script(""),
    )
    board = compile_to_edl(conn, board_id)
    assert board.storyboard_id == board_id
    assert [i.clip_id for i in board.items] == [1, 2]
    assert board.items[0].timeline_start == 0.0
    assert board.items[1].timeline_start == 4.0
    assert board.items[1].timeline_end == 9.0
    assert board.items[1].transition_out == "fade"  # the scene's exit transition
    assert board.items[1].note == "Close"

    again = compile_to_edl(conn, board_id)
    assert again.id == board.id  # re-compile updates, not duplicates
    assert len(conn.execute("SELECT * FROM edls").fetchall()) == 1


def test_compile_rejects_out_of_asset_ranges(conn) -> None:
    board_id = create_board(
        conn, name="bad", objective="", style="", target_duration_s=0,
        document=StoryboardDocument(scenes=[StoryboardScene(
            id="s1", beat_id=None,
            shots=[SceneShot(segment_id=1, asset_id="a1",
                             source_start=0.0, source_end=99.0)],
            est_duration_s=99.0,
        )]),
        script=ScriptDocument(storyboard_id=""),
    )
    with pytest.raises(AutoVidError, match="does not compile"):
        compile_to_edl(conn, board_id)


def test_build_deterministic_wraps_planner_items(conn) -> None:
    board_id = build_deterministic(
        conn, objective="recap", target_duration_s=8.0, style="cinematic",
        name="det",
    )
    assert get_board(conn, board_id)["name"] == "det"
    doc, _ = load_document(conn, board_id)
    assert doc.scenes, "planner fallback should produce scenes from accepted segments"
    for scene in doc.scenes:
        assert scene.shots, "every scene carries real footage"
    script = load_script(conn, board_id)
    assert script is not None and len(script.beats) == len(doc.scenes)
    # compiling the fallback board yields the same footage the planner chose
    board = compile_to_edl(conn, board_id)
    assert board.duration_s == pytest.approx(doc.duration_s, abs=0.01)


def test_board_from_edl_maps_items() -> None:
    from autovid.edl import EDL, EDLItem

    board = EDL(items=[
        EDLItem(clip_id=7, asset_id="a1", source_start=1.0, source_end=3.5),
        EDLItem(clip_id=8, asset_id="a1", source_start=5.0, source_end=9.0),
    ])
    doc, script = board_from_edl(board, name="n", objective="o", style="s")
    assert [s.shots[0].segment_id for s in doc.scenes] == [7, 8]
    assert doc.scenes[0].est_duration_s == 2.5
    assert [b.id for b in script.beats] == ["b1", "b2"]
