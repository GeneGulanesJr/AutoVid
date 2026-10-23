"""Music-driven edit decision planner.

Takes the scene catalog (accepted/approved segments) plus a
:class:`autovid.music.MusicAnalysis` and produces a stored, editable EDL v2:
every item carries timeline positions, the musical event it cuts on, its
transition, and audio handling. Rendering the EDL never re-asks the models.

Three modes share the same music analysis:

- ``beat`` — cuts snap to a beat grid (every ``beats_per_cut`` beats). Rhythm
  first.
- ``cinematic`` — each music section is divided at natural proportions and
  pacing follows the section's role (long establishing shots in the intro,
  quick cuts in the climax); cuts may fall between beats.
- ``hybrid`` (default) — beat-snapped inside sections whose energy is above
  the track's median, structure-paced elsewhere.

Section roles are derived from the energy curve: first section = intro, last
= outro, the highest-energy middle section = climax, the rest = build. The
D1 decision layer reviews each section's plan (bounded: approve / replace a
clip / slow down / speed up) — it never estimates timing itself.
"""
from __future__ import annotations

import bisect
import json
import logging
import sqlite3

from . import edl as edl_mod
from .config import Settings
from .decisions import DecisionClient
from .errors import AutoVidError
from .music import MusicAnalysis, section_audio_b64
from .storyboard import Candidate, load_candidates

logger = logging.getLogger(__name__)

MIN_SLOT_S = 1.5       # a shorter cut makes footage incomprehensible
MAX_SLOT_S = 20.0
TAG_OVERLAP_MAX = 0.5

# beats per cut, per section role, per style
STYLE_PACING = {
    "fast_paced": {"intro": 8, "build": 4, "climax": 2, "outro": 6},
    "cinematic": {"intro": 16, "build": 8, "climax": 4, "outro": 12},
    "social": {"intro": 6, "build": 3, "climax": 2, "outro": 4},
}
DEFAULT_STYLE = "cinematic"


def derive_structure(music: MusicAnalysis) -> list[tuple[str, object]]:
    """Label music sections intro / build / climax / outro from the energy curve."""
    secs = list(music.sections)
    if not secs:
        raise AutoVidError("music analysis produced no sections")
    roles = ["build"] * len(secs)
    roles[0] = "intro"
    roles[-1] = "outro"
    if len(secs) >= 3:
        climax = max(range(len(secs)), key=lambda i: secs[i].energy_mean)
        if 0 < climax < len(secs) - 1:
            roles[climax] = "climax"
    return list(zip(roles, secs))


def _beats_between(music: MusicAnalysis, start: float, end: float) -> list[float]:
    beats = music.beats  # ascending
    return beats[bisect.bisect_left(beats, start - 1e-6):bisect.bisect_left(beats, end)]


def _slots(
    music: MusicAnalysis,
    role: str,
    section_start: float,
    section_end: float,
    beats_per_cut: int,
    mode: str,
    median_energy: float,
) -> list[tuple[float, float]]:
    """Cut boundaries inside one section; returns consecutive (start, end) pairs."""
    length = section_end - section_start
    beat_len = 60.0 / music.tempo_bpm if music.tempo_bpm > 0 else 0.5
    ideal = max(min(beats_per_cut * beat_len, MAX_SLOT_S), MIN_SLOT_S)

    if mode == "cinematic" or (
        mode == "hybrid" and _section_energy(music, section_start, section_end) < median_energy
    ):
        n = max(1, round(length / ideal))
        if n == 1:
            return [(section_start, section_end)]
        inner = [section_start + i * length / n for i in range(1, n)]
    else:
        beats = _beats_between(music, section_start, section_end)
        inner = [b for i, b in enumerate(beats) if (i + 1) % beats_per_cut == 0]
        inner = [b for b in inner if section_start + MIN_SLOT_S <= b <= section_end - MIN_SLOT_S]
        if not inner:  # too few beats in the section — fall back to uniform
            n = max(1, round(length / ideal))
            inner = [section_start + i * length / n for i in range(1, n)] or []

    edges = [section_start] + inner + [section_end]
    return list(zip(edges, edges[1:]))


def _section_energy(music: MusicAnalysis, start: float, end: float) -> float:
    energies = [s.energy_mean for s in music.sections if s.start < end and s.end > start]
    return sum(energies) / len(energies) if energies else 0.0


def _pick_clip(
    candidates: list[Candidate],
    used: set[int],
    slot_len: float,
    prev_tags: frozenset[str],
    *,
    prefer: str = "score",
) -> Candidate | None:
    """Best unused candidate that can actually fill the slot (no speed tricks).

    ``prefer="duration"`` (calm sections) ranks by clip length so calm music
    gets the longest, most establishing shots; ``prefer="score"`` (default)
    ranks by highlight. Relaxation order: fresh + fresh-tags → fresh (any
    tags) → reuse, and only then a too-short candidate trimmed to fit.
    ``None`` only when the pool is empty.
    """
    eligible = sorted(
        (c for c in candidates if c.duration_s >= slot_len),
        key=(lambda c: (-c.duration_s, -c.score, c.start_s)) if prefer == "duration"
        else (lambda c: (-c.score, c.start_s)),
    )
    fresh = [c for c in eligible if c.segment_id not in used]

    def pick(ranked: list[Candidate], *, distinct: bool) -> Candidate | None:
        if distinct and prev_tags:
            for c in ranked:
                overlap = len(c.tags & prev_tags) / max(len(c.tags | prev_tags), 1)
                if overlap <= TAG_OVERLAP_MAX:
                    return c
        return ranked[0] if ranked else None

    # Relaxation order: fresh + fresh-tags → fresh (any tags) → reuse.
    for pool, distinct in ((fresh, True), (fresh, False),
                           (eligible, True), (eligible, False)):
        pick_result = pick(pool, distinct=distinct)
        if pick_result is not None:
            return pick_result
    # last resort: the longest anything, trimmed to the slot
    return max(candidates, key=lambda c: c.duration_s, default=None)


def plan_music_video(
    conn: sqlite3.Connection,
    settings: Settings,
    music: MusicAnalysis,
    *,
    objective: str,
    mode: str = "hybrid",
    style: str = DEFAULT_STYLE,
    name: str = "music-cut",
    decisions: DecisionClient | None = None,
) -> edl_mod.EDL:
    """Build (and persist) a music-aligned EDL from the scene catalog."""
    if mode not in ("beat", "cinematic", "hybrid"):
        raise AutoVidError(f"unknown mode {mode!r} (beat | cinematic | hybrid)")
    candidates = load_candidates(conn)
    if not candidates:
        raise AutoVidError("no accepted segments in the catalog — run the cut stage first")

    used: set[int] = set()
    items: list[edl_mod.EDLItem] = []
    timeline = 0.0
    pacing = STYLE_PACING.get(style, STYLE_PACING[DEFAULT_STYLE])
    median_energy = (
        sorted(s.energy_mean for s in music.sections)[len(music.sections) // 2]
        if music.sections else 0.0
    )

    section_plans: list[dict] = []
    for role, sec in derive_structure(music):
        k = pacing.get(role, 4)
        mood_label = "unmeasured"
        if decisions is not None:
            # Phase 2 of the music flow: d1-omni listens to the section and
            # classifies its mood. Slow sections get slower, driving sections
            # get faster; clip preference follows (calm → longer shots).
            # Best-effort: an audio decode or server failure keeps the
            # measured-pacing plan rather than aborting the edit.
            fallback = "energetic" if sec.energy_mean >= median_energy else "calm"
            try:
                audio_b64 = section_audio_b64(
                    music.path, sec.start, sec.end, waveform=music.waveform
                )
                mood = decisions.decide_music_mood(
                    json.dumps({
                        "role": role,
                        "section_start": sec.start,
                        "section_end": sec.end,
                        "measured_energy": sec.energy_mean,
                    }),
                    audio_b64,
                    fallback_mood=fallback,
                )
                mood_label = mood.mood
                if mood.mood in ("energetic", "driving"):
                    k = max(2, k // 2)
                elif mood.mood == "calm":
                    k = min(16, k * 2)
            except Exception as e:  # noqa: BLE001 — mood enrichment is optional
                logger.warning("Mood enrichment skipped for %s section: %s", role, e)

        slots = _slots(music, role, sec.start, sec.end, k, mode, median_energy)
        section_plans.append({"role": role, "start": sec.start, "end": sec.end,
                              "slots": slots, "beats_per_cut": k,
                              "mood": mood_label})

    for plan in section_plans:
        prev_tags: frozenset[str] = frozenset()
        prefer = "duration" if plan["mood"] == "calm" else "score"
        for slot_start, slot_end in plan["slots"]:
            slot_len = slot_end - slot_start
            clip = _pick_clip(candidates, used, slot_len, prev_tags, prefer=prefer)
            if clip is None:
                continue
            used.add(clip.segment_id)
            src_start = clip.start_s
            src_end = min(clip.start_s + slot_len, clip.end_s)
            slot_len = src_end - src_start  # a trimmed last resort may be shorter
            # first beat within half a beat of the timeline (beats ascending)
            half = 60.0 / music.tempo_bpm / 2 + 1e-6
            i = bisect.bisect_left(music.beats, timeline)
            beat_idx = next(
                (j for j in (i - 1, i)
                 if 0 <= j < len(music.beats)
                 and abs(music.beats[j] - timeline) <= half),
                None,
            )
            items.append(edl_mod.EDLItem(
                clip_id=clip.segment_id,
                asset_id=clip.asset_id,
                source_start=round(src_start, 3),
                source_end=round(src_end, 3),
                timeline_start=round(timeline, 3),
                timeline_end=round(timeline + slot_len, 3),
                beat_ref=f"beat {beat_idx + 1}" if beat_idx is not None else None,
                audio="music",
            ))
            prev_tags = clip.tags
            timeline += slot_len

        if decisions is not None and items:
            review = decisions.decide_sequence_review(
                context=json.dumps({
                    "role": plan["role"],
                    "mood": plan["mood"],
                    "section_start": plan["start"],
                    "section_end": plan["end"],
                    "beats_per_cut": plan["beats_per_cut"],
                    "clips": [
                        {"clip_id": i.clip_id, "tags": [], "slot_s":
                         round((i.timeline_end or 0) - (i.timeline_start or 0), 2)}
                        for i in items if i.timeline_start is not None
                        and i.timeline_start >= plan["start"] - 0.01
                    ],
                }),
            )
            if review.action == "replace_clip" and len(items) >= 2:
                # swap the section's middle item for the best unused candidate
                mid = len(items) - max(1, len(items) // 2)
                old = items[mid]
                replacement = _pick_clip(
                    candidates, used | {old.clip_id},
                    (old.timeline_end or 1.0) - (old.timeline_start or 1.0), frozenset(),
                )
                if replacement is not None and replacement.segment_id != old.clip_id:
                    used.discard(old.clip_id)
                    used.add(replacement.segment_id)
                    items[mid] = edl_mod.EDLItem(
                        clip_id=replacement.segment_id,
                        asset_id=replacement.asset_id,
                        source_start=replacement.start_s,
                        source_end=replacement.start_s + (old.source_end - old.source_start),
                        timeline_start=old.timeline_start,
                        timeline_end=old.timeline_end,
                        beat_ref=old.beat_ref,
                        audio="music",
                        note="replaced by decision layer",
                    )

    board = edl_mod.EDL(
        name=name,
        objective=objective,
        style=style,
        target_duration_s=round(music.duration_s, 3),
        items=items,
        music_path=music.path,
        mode=mode,
    )
    edl_mod.save(conn, board)
    return board
