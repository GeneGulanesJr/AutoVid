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

from pydantic import BaseModel, Field, computed_field

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
    # v4 — provenance: EDL compiled from a storyboard
    storyboard_id: str | None = None

    @property
    def duration_s(self) -> float:
        # Timeline placement wins as soon as any item has it; max over the
        # items that carry it (never compare None with float).
        ends = [i.timeline_end for i in self.items if i.timeline_end is not None]
        if ends:
            return max(ends)
        return sum(i.source_end - i.source_start for i in self.items)

    def touch(self) -> None:
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(conn: sqlite3.Connection, edl: EDL) -> None:
    if edl.created_at is None:
        edl.touch()
    conn.execute(
        """INSERT INTO edls (id, name, objective, style, target_duration_s, items_json,
                             storyboard_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT(id) DO UPDATE SET
               name = excluded.name, objective = excluded.objective,
               style = excluded.style, target_duration_s = excluded.target_duration_s,
               items_json = excluded.items_json, storyboard_id = excluded.storyboard_id,
               updated_at = datetime('now')""",
        (
            edl.id, edl.name, edl.objective, edl.style, edl.target_duration_s,
            edl.model_dump_json(), edl.storyboard_id, edl.created_at,
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


def find_for_storyboard(conn: sqlite3.Connection, storyboard_id: str) -> EDL | None:
    """The EDL previously compiled from a storyboard, if any (re-compile
    keeps its id so render history and the staged-clip cache survive)."""
    row = conn.execute(
        "SELECT items_json FROM edls WHERE storyboard_id = ? ORDER BY updated_at DESC",
        (storyboard_id,),
    ).fetchone()
    return EDL.model_validate(json.loads(row["items_json"])) if row else None


# ---------------------------------------------------------------------------
# Validation — an EDL must be renderable before anyone spends minutes on it.
# Errors block rendering; warnings surface to the reviewer without blocking.
# ---------------------------------------------------------------------------

# An item may leave the segment's span (the source file is what gets cut) but
# never the asset's duration; concat demuxer output corrupts on mixed sizes.
ASSET_TOLERANCE_S = 0.25
SEGMENT_TOLERANCE_S = 0.05
GAP_TOLERANCE_S = 0.5
TARGET_TOLERANCE = 1.2  # mirror storyboard.OVERSHOOT_TOLERANCE (no import cycle)


class ValidationIssue(BaseModel):
    severity: str          # error | warning
    code: str
    item_index: int | None = None
    clip_id: int | None = None
    message: str


class EDLValidation(BaseModel):
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        return not self.errors


def validate_edl(conn: sqlite3.Connection, board: EDL) -> EDLValidation:
    v = EDLValidation()

    def issue(severity: str, code: str, message: str, *,
              item_index: int | None = None, clip_id: int | None = None) -> None:
        bucket = v.errors if severity == "error" else v.warnings
        bucket.append(ValidationIssue(
            severity=severity, code=code, message=message,
            item_index=item_index, clip_id=clip_id,
        ))

    if not board.items:
        issue("error", "empty", "EDL has no items")
        return v

    # One lookup per distinct segment: span, source path, media properties.
    placeholders = ",".join("?" * len({i.clip_id for i in board.items}))
    rows = conn.execute(
        f"""SELECT s.id, s.asset_id, s.start_s, s.end_s, a.path AS source_path,
                   a.duration_s, a.width, a.height, a.fps, a.has_audio
            FROM segments s JOIN assets a ON a.id = s.asset_id
            WHERE s.id IN ({placeholders})""",
        list({i.clip_id for i in board.items}),
    ).fetchall()
    segs = {r["id"]: r for r in rows}

    sizes: set[tuple[int, int]] = set()
    rates: list[float] = []
    audio_flags: set[bool] = set()
    for idx, item in enumerate(board.items):
        seg = segs.get(item.clip_id)
        if seg is None:
            issue("error", "missing_segment", f"item {idx + 1} references unknown segment {item.clip_id}",
                  item_index=idx, clip_id=item.clip_id)
            continue
        if item.source_start < 0 or item.source_end <= item.source_start:
            issue("error", "bad_range",
                  f"item {idx + 1}: invalid source range {item.source_start:.2f}–{item.source_end:.2f}s",
                  item_index=idx, clip_id=item.clip_id)
        if item.speed <= 0:
            issue("error", "bad_speed", f"item {idx + 1}: speed must be positive",
                  item_index=idx, clip_id=item.clip_id)
        if item.source_end > float(seg["duration_s"]) + ASSET_TOLERANCE_S:
            issue("error", "beyond_asset",
                  f"item {idx + 1}: source end {item.source_end:.2f}s is past the asset's "
                  f"{float(seg['duration_s']):.2f}s",
                  item_index=idx, clip_id=item.clip_id)
        elif item.source_start < float(seg["start_s"]) - SEGMENT_TOLERANCE_S or \
                item.source_end > float(seg["end_s"]) + SEGMENT_TOLERANCE_S:
            issue("warning", "beyond_segment",
                  f"item {idx + 1}: source range leaves the scene span "
                  f"{float(seg['start_s']):.2f}–{float(seg['end_s']):.2f}s (still cut from the original)",
                  item_index=idx, clip_id=item.clip_id)
        if item.asset_id and str(seg["asset_id"]) != item.asset_id:
            issue("warning", "asset_mismatch",
                  f"item {idx + 1}: asset_id {item.asset_id!r} differs from segment {item.clip_id}'s asset",
                  item_index=idx, clip_id=item.clip_id)
        source = Path(seg["source_path"])
        if not source.exists():
            issue("error", "missing_source",
                  f"item {idx + 1}: source file missing: {source}",
                  item_index=idx, clip_id=item.clip_id)
        if seg["width"] and seg["height"]:
            sizes.add((int(seg["width"]), int(seg["height"])))
        if seg["fps"]:
            rates.append(float(seg["fps"]))
        audio_flags.add(bool(seg["has_audio"]))

        # Timeline placement (v2 items): overlaps corrupt the cut; gaps mean
        # the timeline silently diverges from the plan.
        if item.timeline_start is not None and item.timeline_end is not None:
            if item.timeline_end <= item.timeline_start:
                issue("error", "bad_timeline",
                      f"item {idx + 1}: timeline end {item.timeline_end:.2f} ≤ start {item.timeline_start:.2f}",
                      item_index=idx, clip_id=item.clip_id)
            prev = board.items[idx - 1] if idx > 0 else None
            if prev is not None and prev.timeline_end is not None \
                    and item.timeline_start < prev.timeline_end - 1e-6:
                issue("error", "overlap",
                      f"item {idx + 1} starts at {item.timeline_start:.2f}s before item {idx} "
                      f"ends at {prev.timeline_end:.2f}s",
                      item_index=idx, clip_id=item.clip_id)
            if prev is not None and prev.timeline_end is not None \
                    and item.timeline_start > prev.timeline_end + GAP_TOLERANCE_S:
                issue("warning", "gap",
                      f"{item.timeline_start - prev.timeline_end:.2f}s gap between items {idx} and {idx + 1}",
                      item_index=idx, clip_id=item.clip_id)

    if len(sizes) > 1:
        issue("warning", "mixed_resolution",
              "items span multiple resolutions "
              f"({', '.join(f'{w}x{h}' for w, h in sorted(sizes))}) — the renderer conforms them to the "
              "majority size (letterboxed re-encode, so rendering is slower)")
    if len(audio_flags) > 1:
        issue("warning", "mixed_audio",
              "items mix audio and silent sources — the renderer adds silence to the silent ones "
              "so the audio track stays aligned")
    if rates and max(rates) - min(rates) > max(rates) * 0.01:
        issue("warning", "mixed_fps",
              f"items span frame rates {min(rates):.2f}–{max(rates):.2f} fps — the renderer converts "
              "them to one rate when it conforms the clips")

    if board.target_duration_s > 0:
        dur = board.duration_s
        if dur > board.target_duration_s * TARGET_TOLERANCE:
            issue("warning", "over_target",
                  f"duration {dur:.1f}s exceeds the {board.target_duration_s:.1f}s target "
                  f"beyond the {TARGET_TOLERANCE:.1f}× tolerance")
    return v
