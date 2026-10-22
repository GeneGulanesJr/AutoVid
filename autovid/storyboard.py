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
) -> EDL:
    """Build (and persist) an EDL from the accepted-segment catalog."""
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
        if total + cand.duration_s > budget and chosen:
            continue
        chosen.append(cand)
        total += cand.duration_s
    if not chosen:  # pool smaller than one budget-safe pick
        chosen = [ranked[0]]
        total = chosen[0].duration_s

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
        alt = next((c for c in remaining if not _similar(prev, c)), None)
        pick = alt if alt is not None else remaining[0]
        ordered.append(pick)
        remaining.remove(pick)
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
    edl_mod.save(conn, board)
    return board
