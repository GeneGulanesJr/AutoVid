"""Storyboard document: Script · Storyboard · EDL as separate artifacts.

The storyboard is the central planning artifact (plan §4.3/§5):

- **Script** — narrative only: ordered story beats with narration, dialogue
  and on-screen text. No timeline data. Editing a beat marks the storyboard
  scenes that illustrate it ``stale`` (deliberate propagation, never silent).
- **Storyboard** — ordered scenes referencing catalog segments (by id, with
  source ranges), plus alternates, open issues and confidence. Versioned:
  every save appends an immutable version; restore = save an old version as
  a new one.
- **EDL** — compiled from an approved storyboard: exact ranges and timeline
  positions. Re-compiling a storyboard updates the same EDL id, so render
  history and the staged-clip cache survive.

Scenes must reference real segments; :func:`autovid.agent.validate_document`
enforces that before anything is stored.
"""
from __future__ import annotations

import json
import sqlite3
import uuid

from pydantic import BaseModel, Field

from . import edl as edl_mod
from .edl import EDL, EDLItem
from .errors import AutoVidError

BOARD_STATUSES = ("draft", "approved")
VERSION_ORIGINS = ("agent", "planner", "user_edit", "script_edit", "restore")


class ScriptBeat(BaseModel):
    id: str                                   # stable within the script ("b1")
    role: str = ""                            # hook | setup | development | climax | resolution | …
    narration: str = ""
    dialogue: str = ""
    on_screen: str = ""


class ScriptDocument(BaseModel):
    storyboard_id: str
    title: str = ""
    synopsis: str = ""
    beats: list[ScriptBeat] = Field(default_factory=list)
    origin: str = "planner"


class SceneShot(BaseModel):
    segment_id: int
    asset_id: str
    source_start: float
    source_end: float
    transition_out: str = "cut"


class StoryboardScene(BaseModel):
    id: str                                   # stable within the board ("s1")
    title: str = ""
    description: str = ""
    narrative_role: str = ""
    beat_id: str | None = None                # script beat this scene illustrates
    shots: list[SceneShot] = Field(default_factory=list)
    est_duration_s: float = 0.0
    pacing: str = ""                          # e.g. "slow establisher", "beat-snapped"
    transition: str = "cut"                   # how the scene hands over to the next
    alternates: list[int] = Field(default_factory=list)   # alternative segment ids
    open_issues: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    locked: bool = False                      # user lock: survives regeneration
    stale: bool = False                       # its script beat changed after generation


class StoryboardDocument(BaseModel):
    version: int = 1
    scenes: list[StoryboardScene] = Field(default_factory=list)
    missing_footage: list[str] = Field(default_factory=list)  # requested-but-absent shots
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return sum(s.est_duration_s for s in self.scenes)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def create_board(
    conn: sqlite3.Connection, *,
    name: str, objective: str, style: str, target_duration_s: float,
    document: StoryboardDocument, script: ScriptDocument,
    music_path: str | None = None, mode: str | None = None,
    origin: str = "planner", note: str | None = None,
) -> str:
    board_id = uuid.uuid4().hex[:12]
    document.version = 1
    script.storyboard_id = board_id
    conn.execute(
        """INSERT INTO storyboards (id, name, objective, style, target_duration_s,
                                    status, current_version, music_path, mode)
           VALUES (?, ?, ?, ?, ?, 'draft', 1, ?, ?)""",
        (board_id, name, objective, style, target_duration_s, music_path, mode),
    )
    conn.execute(
        """INSERT INTO storyboard_versions (storyboard_id, version, origin, note, document_json)
           VALUES (?, 1, ?, ?, ?)""",
        (board_id, origin, note, document.model_dump_json()),
    )
    save_script(conn, script, bump_version=False)
    conn.commit()
    return board_id


def get_board(conn: sqlite3.Connection, storyboard_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM storyboards WHERE id = ?", (storyboard_id,)
    ).fetchone()
    if row is None:
        raise AutoVidError(f"Unknown storyboard {storyboard_id!r}")
    return row


def list_boards(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT b.*, (SELECT COUNT(*) FROM storyboard_versions v
                        WHERE v.storyboard_id = b.id) AS versions
           FROM storyboards b ORDER BY b.updated_at DESC"""
    ).fetchall()


def save_document(
    conn: sqlite3.Connection, storyboard_id: str, document: StoryboardDocument, *,
    origin: str, note: str | None = None,
) -> int:
    """Append a new version and make it current. Never overwrites history."""
    get_board(conn, storyboard_id)
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) AS v FROM storyboard_versions WHERE storyboard_id = ?",
        (storyboard_id,),
    ).fetchone()
    version = int(row["v"]) + 1
    document.version = version
    conn.execute(
        """INSERT INTO storyboard_versions (storyboard_id, version, origin, note, document_json)
           VALUES (?, ?, ?, ?, ?)""",
        (storyboard_id, version, origin, note, document.model_dump_json()),
    )
    conn.execute(
        "UPDATE storyboards SET current_version = ?, updated_at = datetime('now') WHERE id = ?",
        (version, storyboard_id),
    )
    conn.commit()
    return version


def load_document(
    conn: sqlite3.Connection, storyboard_id: str, version: int | None = None,
) -> tuple[StoryboardDocument, int]:
    get_board(conn, storyboard_id)
    if version is None:
        row = conn.execute(
            "SELECT document_json, version FROM storyboard_versions "
            "WHERE storyboard_id = ? ORDER BY version DESC LIMIT 1",
            (storyboard_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT document_json, version FROM storyboard_versions "
            "WHERE storyboard_id = ? AND version = ?",
            (storyboard_id, version),
        ).fetchone()
    if row is None:
        raise AutoVidError(f"Storyboard {storyboard_id!r} has no version {version!r}")
    return StoryboardDocument.model_validate(json.loads(row["document_json"])), int(row["version"])


def list_versions(conn: sqlite3.Connection, storyboard_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT version, origin, note, created_at FROM storyboard_versions "
        "WHERE storyboard_id = ? ORDER BY version",
        (storyboard_id,),
    ).fetchall()


def set_status(conn: sqlite3.Connection, storyboard_id: str, status: str) -> None:
    if status not in BOARD_STATUSES:
        raise AutoVidError(f"status must be one of {BOARD_STATUSES}")
    cur = conn.execute(
        "UPDATE storyboards SET status = ?, updated_at = datetime('now') WHERE id = ?",
        (status, storyboard_id),
    )
    conn.commit()
    if cur.rowcount == 0:
        raise AutoVidError(f"Unknown storyboard {storyboard_id!r}")


def save_script(conn: sqlite3.Connection, script: ScriptDocument, *,
                bump_version: bool = True) -> None:
    """Store the script and mark scenes whose beat changed ``stale``.

    Deliberate propagation (plan §4.3): an edit never silently rewrites the
    storyboard — scenes referencing a changed beat are flagged for review,
    and regeneration clears the flag.
    """
    old = conn.execute(
        "SELECT beats_json FROM scripts WHERE storyboard_id = ?", (script.storyboard_id,)
    ).fetchone()
    changed_beats: set[str] = set()
    if old is not None:
        try:
            old_beats = {
                b["id"]: b for b in json.loads(old["beats_json"] or "[]")
            }
        except ValueError:
            old_beats = {}
        for beat in script.beats:
            prev = old_beats.get(beat.id)
            if prev is not None and (
                prev.get("narration") != beat.narration
                or prev.get("dialogue") != beat.dialogue
                or prev.get("on_screen") != beat.on_screen
            ):
                changed_beats.add(beat.id)

    conn.execute(
        """INSERT INTO scripts (id, storyboard_id, title, synopsis, beats_json, origin,
                                created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
           ON CONFLICT(storyboard_id) DO UPDATE SET
               id = excluded.id, title = excluded.title, synopsis = excluded.synopsis,
               beats_json = excluded.beats_json, origin = excluded.origin,
               updated_at = datetime('now')""",
        (
            uuid.uuid4().hex[:12], script.storyboard_id, script.title, script.synopsis,
            json.dumps([b.model_dump() for b in script.beats]), script.origin,
        ),
    )
    for scene_id in {s.id for s in
                     load_document(conn, script.storyboard_id)[0].scenes
                     if s.beat_id in changed_beats}:
        _mark_scene(conn, script.storyboard_id, scene_id, stale=True)
    if bump_version and changed_beats:
        doc, _ = load_document(conn, script.storyboard_id)
        save_document(conn, script.storyboard_id, doc,
                      origin="script_edit",
                      note=f"script changed for beats {sorted(changed_beats)}")
    conn.commit()


def load_script(conn: sqlite3.Connection, storyboard_id: str) -> ScriptDocument | None:
    row = conn.execute(
        "SELECT * FROM scripts WHERE storyboard_id = ?", (storyboard_id,)
    ).fetchone()
    if row is None:
        return None
    return ScriptDocument(
        storyboard_id=storyboard_id,
        title=row["title"], synopsis=row["synopsis"],
        beats=[ScriptBeat.model_validate(b) for b in json.loads(row["beats_json"] or "[]")],
        origin=row["origin"],
    )


def _mark_scene(conn: sqlite3.Connection, storyboard_id: str, scene_id: str, *,
                stale: bool) -> None:
    """Flip a flag inside the current version's document (history untouched —
    the flag change itself is recorded by the caller's version bump)."""
    doc, version = load_document(conn, storyboard_id)
    for scene in doc.scenes:
        if scene.id == scene_id:
            scene.stale = stale
    conn.execute(
        "UPDATE storyboard_versions SET document_json = ? "
        "WHERE storyboard_id = ? AND version = ?",
        (doc.model_dump_json(), storyboard_id, version),
    )


# ---------------------------------------------------------------------------
# Compile: storyboard → EDL
# ---------------------------------------------------------------------------


def compile_to_edl(conn: sqlite3.Connection, storyboard_id: str) -> EDL:
    """Turn the current storyboard version into an editable EDL.

    Scenes compile in order; a scene's ``transition`` becomes the outgoing
    fade of its last shot. Re-compiling updates the EDL previously compiled
    from this storyboard (same id) so re-renders reuse staged clips.
    """
    row = get_board(conn, storyboard_id)
    document, version = load_document(conn, storyboard_id)
    if not document.scenes:
        raise AutoVidError("storyboard has no scenes to compile")

    # Resolve each referenced segment's real asset (agent-produced shots carry
    # no asset_id — the catalog is the source of truth, not the model).
    seg_ids = sorted({shot.segment_id for s in document.scenes for shot in s.shots})
    placeholders = ",".join("?" * len(seg_ids))
    asset_rows = conn.execute(
        f"SELECT id, asset_id FROM segments WHERE id IN ({placeholders})", seg_ids,
    ).fetchall()
    asset_of = {int(r["id"]): r["asset_id"] for r in asset_rows}

    items: list[EDLItem] = []
    timeline = 0.0
    for scene in document.scenes:
        if not scene.shots:
            raise AutoVidError(f"scene {scene.id!r} has no shots — every scene needs footage")
        for j, shot in enumerate(scene.shots):
            span = shot.source_end - shot.source_start
            last_of_scene = j == len(scene.shots) - 1
            items.append(EDLItem(
                clip_id=shot.segment_id,
                asset_id=asset_of.get(shot.segment_id, shot.asset_id),
                source_start=round(shot.source_start, 3),
                source_end=round(shot.source_end, 3),
                transition_out=shot.transition_out if last_of_scene else "cut",
                note=scene.title,
                timeline_start=round(timeline, 3),
                timeline_end=round(timeline + span, 3),
                audio="music" if row["music_path"] else "clip",
            ))
            timeline += span

    existing = edl_mod.find_for_storyboard(conn, storyboard_id)
    board = existing or EDL()
    board.name = row["name"]
    board.objective = row["objective"]
    board.style = row["style"]
    board.target_duration_s = float(row["target_duration_s"])
    board.music_path = row["music_path"]
    board.mode = row["mode"]
    board.storyboard_id = storyboard_id
    board.items = items

    validation = edl_mod.validate_edl(conn, board)
    if not validation.ok:
        details = "; ".join(e.message for e in validation.errors)
        raise AutoVidError(f"storyboard does not compile: {details}")
    edl_mod.save(conn, board)
    return board


# ---------------------------------------------------------------------------
# Deterministic fallback: catalog selection wrapped into the document model.
# Same contract as the agent (real segments only) without any LLM call.
# ---------------------------------------------------------------------------


def board_from_edl(
    board: EDL, *, name: str, objective: str, style: str,
) -> tuple[StoryboardDocument, ScriptDocument]:
    """Wrap planner-selected items into scenes + beats (origin ``planner``)."""
    scenes: list[StoryboardScene] = []
    beats: list[ScriptBeat] = []
    for i, item in enumerate(board.items, 1):
        scene_id, beat_id = f"s{i}", f"b{i}"
        scenes.append(StoryboardScene(
            id=scene_id,
            title=item.note or f"Scene {i}",
            description=f"Catalog segment {item.clip_id} "
                        f"({item.source_start:.1f}–{item.source_end:.1f}s).",
            narrative_role="footage",
            beat_id=beat_id,
            shots=[SceneShot(
                segment_id=item.clip_id, asset_id=item.asset_id,
                source_start=item.source_start, source_end=item.source_end,
                transition_out=item.transition_out,
            )],
            est_duration_s=round(item.source_end - item.source_start, 3),
            pacing="beat-snapped" if item.beat_ref else "",
            transition=item.transition_out,
            confidence=0.5,
        ))
        beats.append(ScriptBeat(id=beat_id, role="footage"))
    document = StoryboardDocument(version=1, scenes=scenes)
    script = ScriptDocument(
        storyboard_id="", title=name, synopsis=objective,
        beats=beats, origin="planner",
    )
    return document, script


def build_deterministic(
    conn: sqlite3.Connection, *,
    objective: str, target_duration_s: float, style: str, name: str,
    music_path: str | None = None, mode: str | None = None,
) -> str:
    """Planner-path storyboard: selection without any model call."""
    from .storyboard import select_clips

    board = select_clips(
        conn, objective=objective, target_duration_s=target_duration_s,
        style=style, name=name, persist=False,
    )
    document, script = board_from_edl(board, name=name, objective=objective, style=style)
    return create_board(
        conn, name=name, objective=objective, style=style,
        target_duration_s=target_duration_s, document=document, script=script,
        music_path=music_path, mode=mode, origin="planner",
        note="deterministic planner (no agent)",
    )
