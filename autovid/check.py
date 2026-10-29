"""EDL lint: measure what made earlier cuts "bad", so it shows up after every edit.

Levels: ``fail`` (the cut has a real defect: footage that plays twice, look-alike
shots, a missing source), ``warn`` (worth a look), ``info`` / ``ok`` (facts).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .edl import EDL
from .kinds import classify
from .script import SAME_SCENE_BITS, Fingerprint, _fingerprint

REPLAY_FAIL_S = 0.5      # more than this much footage played twice is a defect
REPLAY_WARN_S = 0.1
TINY_ITEM_S = 0.5
BACKWARD_JUMP_S = 0.3    # a source start this far before the take's furthest point is a backward jump
FOOD_SHARE_MIN = 0.25    # of the runtime, when the cut has food shots at all
TIMELAPSE_MAX_SHARE = 0.5  # of a timelapse clip's own length


@dataclass(frozen=True)
class Finding:
    level: str   # ok | info | warn | fail
    name: str
    detail: str


def _is_food(item) -> bool:
    return "[food]" in item.note


def check_edl(conn: sqlite3.Connection, board: EDL, *, fingerprint: Fingerprint = _fingerprint) -> list[Finding]:
    out: list[Finding] = []
    items = board.items
    total = sum(i.source_end - i.source_start for i in items)
    if not items:
        return [Finding("fail", "items", "the EDL is empty")]
    out.append(Finding("ok", "length", f"{len(items)} items, {total:.1f}s"))

    assets = {r["id"]: r for r in conn.execute(
        "SELECT id, path, duration_s, width, height, fps, has_audio FROM assets")}
    missing = sorted({i.asset_id for i in items if i.asset_id not in assets})
    if missing:
        return out + [Finding("fail", "sources", f"unknown assets: {', '.join(missing)}")]

    # footage that plays twice (the same stretch of one source used by two items)
    replay = 0.0
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            x, y = items[a], items[b]
            if x.asset_id == y.asset_id:
                replay += max(0.0, min(x.source_end, y.source_end) - max(x.source_start, y.source_start))
    level = "fail" if replay > REPLAY_FAIL_S else "warn" if replay > REPLAY_WARN_S else "ok"
    out.append(Finding(level, "replay", f"{replay:.2f}s of footage plays twice"))

    # the story going back to earlier footage of one take (the LLM may list a take's items out of
    # capture order); overlaps here are fine too — replay above judges footage that plays twice
    high_water: dict[str, float] = {}
    jumps: list[str] = []
    for n, i in enumerate(items, 1):
        mark = high_water.get(i.asset_id)
        if mark is not None and i.source_start < mark - BACKWARD_JUMP_S:
            jumps.append(f"item {n} goes back {mark - i.source_start:.1f}s")
        high_water[i.asset_id] = i.source_end if mark is None else max(mark, i.source_end)
    if jumps:
        out.append(Finding("warn", "story order",
                           "the story jumps back within a take: " + ", ".join(jumps)))

    tiny = [n for n, i in enumerate(items, 1) if i.source_end - i.source_start < TINY_ITEM_S]
    if tiny:
        out.append(Finding("warn", "tiny clips", f"items {tiny} are under {TINY_ITEM_S}s"))

    # food: share of the runtime, and shots that look the same as another shot of the same take
    food = [(n, i) for n, i in enumerate(items, 1) if _is_food(i)]
    if food:
        food_s = sum(i.source_end - i.source_start for _, i in food)
        share = food_s / total
        out.append(Finding("ok" if share >= FOOD_SHARE_MIN else "warn", "food share",
                           f"{len(food)} food shots = {food_s:.1f}s ({100 * share:.0f}% of the runtime)"))
        prints = [(n, i.asset_id, fingerprint(assets[i.asset_id]["path"], (i.source_start + i.source_end) / 2))
                  for n, i in food]
        alike = [(a, b) for k, (a, ta, fa) in enumerate(prints) for b, tb, fb in prints[k + 1:]
                 if ta == tb and fa is not None and fb is not None and bin(fa ^ fb).count("1") <= SAME_SCENE_BITS]
        out.append(Finding("fail" if alike else "ok", "look-alikes",
                           (f"{len(alike)} pairs of food shots look the same, items "
                            f"{sorted({n for pair in alike for n in pair})}") if alike else "no look-alike food shots"))

    # footage kinds in use
    kinds = classify(conn)
    for aid, tags in kinds.items():
        used = sum(i.source_end - i.source_start for i in items if i.asset_id == aid)
        if "timelapse" in tags and used > TIMELAPSE_MAX_SHARE * assets[aid]["duration_s"]:
            out.append(Finding("warn", "timelapse", f"{aid[:10]} is a timelapse and {used:.0f}s of "
                                                    f"its {assets[aid]['duration_s']:.0f}s is used"))
    formats = {(assets[i.asset_id]["width"], assets[i.asset_id]["height"], round(assets[i.asset_id]["fps"] or 0, 2),
                bool(assets[i.asset_id]["has_audio"])) for i in items}
    if len(formats) > 1:
        out.append(Finding("info", "formats", f"{len(formats)} different formats in use; rendering conforms them "
                                              "(re-encode, a bit slower)"))
    return out
