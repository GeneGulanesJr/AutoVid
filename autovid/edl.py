"""Edit Decision List: typed, editable, persisted.

The EDL is the artifact a human reviews and revises: an ordered list of source
segments with exact timestamps. Items reference segments (and through them,
assets and original files), so every frame in a render stays traceable.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .errors import AutoVidError

TRANSITIONS = ("cut", "fade")
AUDIO_MODES = ("music", "clip", "duck")


class EDLItem(BaseModel):
    clip_id: int  # segments.id
    asset_id: str
    source_start: float
    source_end: float
    transition_in: str = "cut"
    transition_out: str = "cut"
    note: str = ""
    # v2 — music-driven editing: timeline placement and musical alignment.
    timeline_start: float | None = None   # position in the output timeline
    timeline_end: float | None = None
    beat_ref: str | None = None           # musical event this cut targets, e.g. "beat 24"
    speed: float = 1.0                    # playback rate (1.0 = untouched footage)
    audio: str = "music"                  # music | clip | duck


class EDL(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "rough-cut"
    objective: str = ""
    style: str = ""
    target_duration_s: float = 0.0
    items: list[EDLItem] = Field(default_factory=list)
    created_at: str | None = None
    # v2 — music-driven editing
    music_path: str | None = None
    mode: str | None = None               # beat | cinematic | hybrid

    @property
    def duration_s(self) -> float:
        if self.items and self.items[-1].timeline_end is not None:
            return max(i.timeline_end for i in self.items)
        return sum(i.source_end - i.source_start for i in self.items)

    def touch(self) -> None:
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(conn: sqlite3.Connection, edl: EDL) -> None:
    if edl.created_at is None:
        edl.touch()
    conn.execute(
        """INSERT INTO edls (id, name, objective, style, target_duration_s, items_json,
                             created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT(id) DO UPDATE SET
               name = excluded.name, objective = excluded.objective,
               style = excluded.style, target_duration_s = excluded.target_duration_s,
               items_json = excluded.items_json, updated_at = datetime('now')""",
        (
            edl.id, edl.name, edl.objective, edl.style, edl.target_duration_s,
            edl.model_dump_json(), edl.created_at,
        ),
    )
    conn.commit()


def load(conn: sqlite3.Connection, edl_id: str) -> EDL:
    row = conn.execute("SELECT items_json FROM edls WHERE id = ?", (edl_id,)).fetchone()
    if row is None:
        raise AutoVidError(f"Unknown EDL {edl_id!r}")
    return EDL.model_validate(json.loads(row["items_json"]))


def list_edls(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, name, objective, style, target_duration_s, updated_at "
        "FROM edls ORDER BY updated_at DESC"
    ).fetchall()
