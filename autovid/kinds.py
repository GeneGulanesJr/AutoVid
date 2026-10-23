"""Footage kinds: what each asset *is*, from metadata and transcripts (no model calls).

Tags, and what the pipeline does with them:

- ``speech``       talking footage: the cut stage ignores its (low) visual highlight score,
                   because the value is in what is said (see ``refine.catalog_scenes``).
- ``timelapse``    time-compressed footage: a long run of near-identical shots; the script stage
                   keeps at most one food shot from it.
- ``other_camera`` frame size / rate differs from most of the footage: rendering conforms it.
- ``no_audio``     no audio track: rendering adds silence so clips can be joined.

shortcut: timelapse is inferred from recording times. A clip counts when the next clip was
recorded TIMELAPSE_RATIO times its length later *and* it is in a different format (the usual
timelapse mode), or when the gap is extreme on its own. A long pause between ordinary clips
(waiting for food) has the ratio but not the format, so it is not flagged. Upgrade: read the
camera's own timelapse flag when the container carries one.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

TIMELAPSE_RATIO = 4.0         # gap to the next clip / this clip's length, with a different format
TIMELAPSE_RATIO_ALONE = 20.0  # ... or this much on its own
SPEECH_DENSITY = 0.25         # spoken seconds / clip seconds


def _recorded_at(row: sqlite3.Row) -> datetime | None:
    """The camera's creation_time (container or stream tag), else None."""
    try:
        info = json.loads(row["ffprobe_json"] or "{}")
    except ValueError:
        return None
    tags = [info.get("format", {}).get("tags") or {}] + [s.get("tags") or {} for s in info.get("streams", [])]
    for t in tags:
        if t.get("creation_time"):
            try:
                return datetime.fromisoformat(t["creation_time"].replace("Z", "+00:00"))
            except ValueError:
                return None
    return None


def classify(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Kind tags per asset id (an asset with no special kind maps to an empty set)."""
    rows = conn.execute("SELECT id, duration_s, fps, width, height, has_audio, ffprobe_json, created_at "
                        "FROM assets").fetchall()
    fmt = lambda r: (r["width"], r["height"], round(r["fps"] or 0, 2))  # noqa: E731
    weight: dict[tuple, float] = {}
    for r in rows:
        weight[fmt(r)] = weight.get(fmt(r), 0.0) + (r["duration_s"] or 0.0)
    majority = max(weight, key=weight.get) if weight else None

    when = {r["id"]: _recorded_at(r) for r in rows}
    timed = sorted((r for r in rows if when[r["id"]]), key=lambda r: when[r["id"]])
    next_start = {a["id"]: when[b["id"]] for a, b in zip(timed, timed[1:])}

    speech: dict[str, float] = {r["asset_id"]: r["s"] for r in conn.execute(
        "SELECT asset_id, SUM(end_s - start_s) AS s FROM transcripts GROUP BY asset_id")}

    kinds: dict[str, set[str]] = {}
    for r in rows:
        tags: set[str] = set()
        dur = r["duration_s"] or 0.0
        if not r["has_audio"]:
            tags.add("no_audio")
        if majority and fmt(r) != majority:
            tags.add("other_camera")
        if dur > 0 and r["id"] in next_start:
            ratio = (next_start[r["id"]] - when[r["id"]]).total_seconds() / dur
            if ratio >= TIMELAPSE_RATIO_ALONE or (ratio >= TIMELAPSE_RATIO and "other_camera" in tags):
                tags.add("timelapse")
        if dur > 0 and speech.get(r["id"], 0.0) / dur >= SPEECH_DENSITY:
            tags.add("speech")
        kinds[r["id"]] = tags
    return kinds
