"""Storyboard selection: objective + duration + style → ordered EDL.

Selection runs over the accepted-segment catalog only (``verdict='accept'``,
review status not pending), scored by highlight score weighted by decision
confidence. Constraints:

- **Pacing**: each style defines a preferred clip-length range; longer clips
  are trimmed at their tail (source provenance keeps the original end).
- **Diversity**: adjacent items from the same asset, or sharing most tags,
  are avoided when an alternative exists (duplicate-content guard).
- **Shape**: when the pool allows, the earliest clip opens (a natural
  beginning) and the highest-scoring clip closes (a strong ending); the
  middle fills by score.

Deterministic: same catalog + same inputs → same EDL.
"""
from __future__ import annotations

import bisect
import json
import logging
import sqlite3
from dataclasses import dataclass

from . import edl as edl_mod
from .edl import EDL, EDLItem

logger = logging.getLogger(__name__)

# style → (min_clip_s, max_clip_s)
STYLE_CLIP_LENGTHS = {
    "fast_paced": (1.5, 5.0),
    "social": (2.0, 8.0),
    "cinematic": (4.0, 12.0),
    "informative": (3.0, 20.0),
    "food": (2.5, 5.0),
}
DEFAULT_STYLE = "cinematic"
OVERSHOOT_TOLERANCE = 1.2  # never exceed target × this unless the pool is smaller
TAG_OVERLAP_MAX = 0.5  # adjacent items sharing more than this tag fraction are "similar"


@dataclass(frozen=True)
class Candidate:
    segment_id: int
    asset_id: str
    start_s: float
    end_s: float
    highlight: float
    confidence: float
    tags: frozenset[str]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def score(self) -> float:
        return self.highlight * (0.5 + 0.5 * self.confidence)


def load_candidates(conn: sqlite3.Connection) -> list[Candidate]:
    """Accepted segments, plus anything a human approved in review.

    The decision layer's conservative fallbacks (``needs_evidence``,
    ``flag_review``) go to the review queue; an approval there makes a segment
    storyboard-eligible even though the model never accepted it.
    """
    rows = conn.execute(
        """SELECT id, asset_id, start_s, end_s, highlight_score, decision_confidence,
                  tags_json FROM segments
           WHERE ((verdict = 'accept' AND review_status != 'pending')
                  OR review_status = 'approved')"""
    ).fetchall()
    candidates = []
    for r in rows:
        try:
            tags = frozenset(json.loads(r["tags_json"] or "[]"))
        except ValueError:
            tags = frozenset()
        candidates.append(Candidate(
            segment_id=int(r["id"]), asset_id=r["asset_id"],
            start_s=float(r["start_s"]), end_s=float(r["end_s"]),
            highlight=float(r["highlight_score"] or 0.0),
            confidence=float(r["decision_confidence"] or 0.0),
            tags=tags,
        ))
    return candidates


def _trim(candidate: Candidate, style: str) -> tuple[float, float]:
    lo, hi = STYLE_CLIP_LENGTHS.get(style, STYLE_CLIP_LENGTHS["cinematic"])
    end = candidate.end_s
    if candidate.duration_s > hi:
        end = candidate.start_s + hi
    return candidate.start_s, end


def _similar(a: Candidate, b: Candidate) -> bool:
    if a.asset_id == b.asset_id:
        return True
    if not a.tags or not b.tags:
        return False
    overlap = len(a.tags & b.tags) / max(len(a.tags | b.tags), 1)
    return overlap > TAG_OVERLAP_MAX


def select_clips(
    conn: sqlite3.Connection,
    *,
    objective: str,
    target_duration_s: float,
    style: str = DEFAULT_STYLE,
    name: str = "rough-cut",
    persist: bool = True,
) -> EDL:
    """Build an EDL from the accepted-segment catalog.

    ``persist=False`` builds the selection without storing it — the storyboard
    document flow wraps the items into scenes and persists those instead.
    """
    if style == "food":
        return _select_food(conn, objective=objective, target_duration_s=target_duration_s,
                            name=name, persist=persist)
    pool = load_candidates(conn)
    if not pool:
        raise ValueError("no accepted segments in the catalog — run the cut stage first")

    ranked = sorted(pool, key=lambda c: (-c.score, c.start_s))
    chosen: list[Candidate] = []
    total = 0.0
    budget = target_duration_s * OVERSHOOT_TOLERANCE
    for cand in ranked:
        if total >= target_duration_s:
            break
        length = _trim(cand, style)[1] - cand.start_s  # budget the trimmed length we'll actually emit
        if total + length > budget and chosen:
            continue
        chosen.append(cand)
        total += length
    if not chosen:  # pool smaller than one budget-safe pick
        chosen = [ranked[0]]
        total = _trim(chosen[0], style)[1] - chosen[0].start_s

    # Ordering: earliest-clip opening, strongest closing, score-ordered middle.
    chosen.sort(key=lambda c: (-c.score, c.start_s))
    opener = min(chosen, key=lambda c: c.start_s)
    closer = max(chosen, key=lambda c: c.score)
    middle = [c for c in chosen if c is not opener and c is not closer]
    # Diversity pass: avoid adjacent similar clips when a swap exists.
    ordered = [opener]
    remaining = sorted(middle, key=lambda c: (-c.score, c.start_s))
    while remaining:
        prev = ordered[-1]
        idx = next((i for i, c in enumerate(remaining) if not _similar(prev, c)), None)
        pick_i = idx if idx is not None else 0
        ordered.append(remaining[pick_i])
        del remaining[pick_i]
    if closer is not opener:
        ordered.append(closer)

    items = []
    for cand in ordered:
        start, end = _trim(cand, style)
        items.append(EDLItem(
            clip_id=cand.segment_id,
            asset_id=cand.asset_id,
            source_start=round(start, 3),
            source_end=round(end, 3),
        ))
    board = EDL(
        name=name, objective=objective, style=style,
        target_duration_s=target_duration_s, items=items,
    )
    if persist:
        edl_mod.save(conn, board)
    return board


def _dish_key(name: str) -> str:
    n = name.lower().strip()
    return n[:-1] if n.endswith("s") and not n.endswith("ss") else n


def _canonical_dishes(names: set[str]) -> dict[str, str]:
    """Fold "pork siomai" / "steamed siomai" into "siomai": the shortest listed name it ends with."""
    toks = {n: n.split() for n in names}
    return {
        n: min((m for m, u in toks.items() if t[len(t) - len(u):] == u), key=lambda m: len(toks[m]))
        for n, t in toks.items()
    }


def _select_food(conn: sqlite3.Connection, *, objective: str, target_duration_s: float,
                 name: str, persist: bool = True) -> EDL:
    """Food-vlog edit: one shot per distinct dish, in the order it was eaten.

    shortcut: variants fold into the shortest dish name they end with ("shrimp siomai" ->
    "siomai"), and the shot is the first seconds of the dish's best window because the VLM
    says which dishes a window shows, not when. Upgrade: per-frame dish timing, finer variants.
    """
    lo, hi = STYLE_CLIP_LENGTHS["food"]
    pool = load_candidates(conn)
    if not pool:
        raise ValueError("no accepted segments in the catalog — run the cut stage first")
    # candidates grouped per asset, sorted by start — window lookup bisects
    # instead of scanning the whole pool per analysis row
    by_asset: dict[str, list] = {}
    for c in pool:
        by_asset.setdefault(c.asset_id, []).append(c)
    starts_by_asset: dict[str, list[float]] = {}
    for aid, cands in by_asset.items():
        cands.sort(key=lambda c: c.start_s)
        starts_by_asset[aid] = [c.start_s for c in cands]
    rows = conn.execute(
        """SELECT an.asset_id, an.window_start, an.window_end, an.response_json, a.original_name
           FROM analyses an JOIN assets a ON a.id = an.asset_id
           ORDER BY a.original_name, an.window_start"""
    ).fetchall()
    hits = []  # (quality, dish, shot) for every food seen in a usable window
    for r in rows:
        try:
            a = json.loads(r["response_json"])
        except ValueError:
            continue
        if "menu" in (a.get("description") or "").lower():
            continue  # shortcut: the VLM lists dishes printed on a menu board; skip those windows
        quality = float(a.get("highlight_score") or 0) * (1 - float(a.get("uncertainty") or 0))
        seg = None
        cands = by_asset.get(r["asset_id"])
        if cands:
            i = bisect.bisect_right(starts_by_asset[r["asset_id"]], r["window_start"]) - 1
            if i >= 0 and cands[i].start_s <= r["window_start"] < cands[i].end_s:
                seg = cands[i]
        end = min(r["window_start"] + hi, r["window_end"], seg.end_s) if seg else 0.0
        if seg is None or end - r["window_start"] < lo:
            continue
        shot = (r["original_name"], r["window_start"], end, seg.segment_id, r["asset_id"])
        hits += [(quality, food.strip(), shot) for food in a.get("foods") or [] if food.strip()]
    if not hits:
        raise ValueError("no foods found — re-run analyze (with a short AUTOVID_COARSE_WINDOW_S) "
                         "after the foods update")
    canon = _canonical_dishes({_dish_key(h[1]) for h in hits})
    best: dict[str, tuple[float, str, tuple]] = {}  # dish -> its best hit
    for q, food, shot in hits:
        key = canon[_dish_key(food)]
        if key not in best or q > best[key][0]:
            best[key] = (q, food, shot)
    windows: dict[tuple, dict] = {}  # one shot per window, labelled with every dish it was picked for
    for q, food, shot in best.values():
        w = windows.setdefault(shot, {"quality": 0.0, "dishes": []})
        w["quality"] = max(w["quality"], q)
        w["dishes"].append(food)
    shots = sorted(windows.items(), key=lambda kv: -kv[1]["quality"])
    budget = target_duration_s * OVERSHOOT_TOLERANCE
    total = sum(sh[2] - sh[1] for sh, _ in shots)
    while total > budget and len(shots) > 1:
        total -= shots[-1][2] - shots[-1][1]
        shots.pop()  # drop the weakest shot
    shots.sort(key=lambda kv: (kv[0][0], kv[0][1]))  # meal order: file name is capture time
    items = [EDLItem(clip_id=sh[3], asset_id=sh[4], source_start=round(sh[1], 3),
                     source_end=round(sh[2], 3), note=" + ".join(w["dishes"]), audio="clip")
             for sh, w in shots]
    board = EDL(name=name, objective=objective, style="food",
                target_duration_s=target_duration_s, items=items)
    if persist:
        edl_mod.save(conn, board)
    return board
