"""Storyboard Agent: narrative planning over the structured footage catalog.

Consumes a *catalog digest* (eligible segments + music sections) — never raw
video — and produces a Script + Storyboard document. Hard rules:

- The agent must reference **real** segment ids and stay inside their spans;
  :func:`validate_document` is the hallucination gate. Invalid output gets
  exactly one bounded repair round, then the caller falls back to the
  deterministic planner (:func:`autovid.storyboard_doc.build_deterministic`).
- The agent plans; the application executes. It can request missing shots
  (``missing_footage``), never invent them.
- Regeneration is scoped: a single scene, or the whole board with user-locked
  scenes preserved verbatim at their positions (:func:`merge_locked`).

No API key configured → the caller should skip straight to the fallback;
:class:`AgentUnavailable` signals any LLM/validation failure.
"""
from __future__ import annotations

import json
import logging
import sqlite3

from pydantic import ValidationError

from . import storyboard_doc as doc_mod
from .config import Settings
from .errors import AutoVidError
from .storyboard_doc import (
    ScriptBeat,
    ScriptDocument,
    SceneShot,
    StoryboardDocument,
    StoryboardScene,
)
from .vlm import VLMClient, _extract_json

logger = logging.getLogger(__name__)

SPAN_TOLERANCE_S = 0.05

_SYSTEM_PROMPT = """You are the AutoVid Storyboard Agent: a narrative planner. You receive \
a catalog digest of REAL footage segments (ids, source spans, descriptions, tags, scores) \
plus the user's objective, target duration and style. You select and order existing footage \
into a storyboard. You must never invent segment ids or timestamps. Answer with STRICT JSON \
only (no markdown, no prose) with exactly these keys:
"title": short title for the edit;
"synopsis": one or two sentences describing the story arc;
"beats": list of {"id": "b1", "role": "hook|setup|development|climax|resolution", \
"narration": "voice-over line", "on_screen": "on-screen text or empty", "dialogue": "empty"};
"scenes": list of {"id": "s1", "title": "...", "description": "what this scene shows", \
"narrative_role": "hook|setup|development|climax|resolution", "beat_id": "b1", \
"shots": [{"segment_id": 123, "source_start": 0.5, "source_end": 4.0, \
"transition_out": "cut|fade"}], "pacing": "e.g. slow establisher", "transition": "cut|fade", \
"open_issues": [], "confidence": 0.0-1.0};
"alternates": list of {"scene_id": "s1", "segment_ids": [456]} — alternative EXISTING \
segments, one entry per scene that has one;
"missing_footage": list of strings — shots the story needs but the catalog lacks.
Rules: use ONLY segment ids from the digest; source ranges must stay within the segment's \
span; summed shot durations should approach the target duration; every scene needs at least \
one shot and a beat_id defined in "beats"; do not reuse one segment in two scenes unless \
nothing else fits."""


class AgentUnavailable(AutoVidError):
    """The agent could not produce a catalog-valid storyboard."""


def build_digest(conn: sqlite3.Connection, *, limit: int = 60) -> dict:
    """Compact, score-ranked digest of storyboard-eligible segments."""
    from .storyboard import load_candidates

    candidates = load_candidates(conn)
    if not candidates:
        raise AutoVidError("no accepted segments in the catalog — run the cut stage first")
    ranked = sorted(candidates, key=lambda c: (-c.score, c.start_s))[:limit]
    by_id = {c.segment_id: c for c in ranked}
    placeholders = ",".join("?" * len(by_id))
    desc_rows = conn.execute(
        f"SELECT id, description, tags_json FROM segments WHERE id IN ({placeholders})",
        list(by_id),
    ).fetchall()
    descriptions = {
        int(r["id"]): (r["description"] or "",
                       json.loads(r["tags_json"] or "[]") if _is_json(r["tags_json"]) else [])
        for r in desc_rows
    }
    segments = []
    for c in ranked:
        description, tags = descriptions.get(c.segment_id, ("", []))
        segments.append({
            "id": c.segment_id,
            "start": round(c.start_s, 2),
            "end": round(c.end_s, 2),
            "duration": round(c.duration_s, 2),
            "score": round(c.score, 3),
            "description": description[:220],
            "tags": tags,
        })
    return {
        "segments": segments,
        "note": (f"{len(candidates)} eligible segments total, showing the top {len(segments)} by "
                 "score. 'start'/'end' are the segment's source span in seconds.") if
                len(segments) < len(candidates) else
                "All eligible segments shown. 'start'/'end' are the segment's source span.",
    }


def _is_json(text: str | None) -> bool:
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def validate_document(
    conn: sqlite3.Connection, document: StoryboardDocument, script: ScriptDocument,
) -> list[str]:
    """Hallucination gate: every reference must resolve in the catalog.

    Returns a list of error strings (empty = valid). Beyond-span trims are
    warnings by contract here (compile-time EDL validation still blocks
    beyond-asset ranges); unknown ids and impossible ranges are errors.
    """
    errors: list[str] = []
    beat_ids = {b.id for b in script.beats}
    if len(beat_ids) != len(script.beats):
        errors.append("duplicate beat ids in script")

    seg_ids: set[int] = set()
    for scene in document.scenes:
        for shot in scene.shots:
            seg_ids.add(shot.segment_id)
        seg_ids.update(scene.alternates)
    spans: dict[int, tuple[float, float, float]] = {}
    if seg_ids:
        placeholders = ",".join("?" * len(seg_ids))
        rows = conn.execute(
            f"""SELECT s.id, s.start_s, s.end_s, a.duration_s
                FROM segments s JOIN assets a ON a.id = s.asset_id
                WHERE s.id IN ({placeholders})""",
            list(seg_ids),
        ).fetchall()
        spans = {int(r["id"]): (float(r["start_s"]), float(r["end_s"]),
                                float(r["duration_s"])) for r in rows}

    scene_ids = set()
    for scene in document.scenes:
        if scene.id in scene_ids:
            errors.append(f"duplicate scene id {scene.id!r}")
        scene_ids.add(scene.id)
        if scene.beat_id is not None and scene.beat_id not in beat_ids:
            errors.append(f"scene {scene.id!r} references unknown beat {scene.beat_id!r}")
        if not scene.shots:
            errors.append(f"scene {scene.id!r} has no shots")
        for shot in scene.shots:
            span = spans.get(shot.segment_id)
            if span is None:
                errors.append(f"scene {scene.id!r} references unknown segment {shot.segment_id}")
                continue
            start, end, duration = span
            if shot.source_end <= shot.source_start:
                errors.append(
                    f"scene {scene.id!r} segment {shot.segment_id}: empty/reversed range")
            elif shot.source_start < start - SPAN_TOLERANCE_S \
                    or shot.source_end > end + SPAN_TOLERANCE_S:
                if shot.source_end > duration + 0.25 or shot.source_start < 0:
                    errors.append(
                        f"scene {scene.id!r} segment {shot.segment_id}: range leaves the asset "
                        f"(0–{duration:.2f}s)")
                else:
                    logger.info(
                        "scene %s trims outside segment %d's span (allowed, warned at compile)",
                        scene.id, shot.segment_id)
        for alt in scene.alternates:
            if alt not in spans:
                errors.append(f"scene {scene.id!r} lists unknown alternate segment {alt}")
    return errors


def merge_locked(old_doc: StoryboardDocument, new_doc: StoryboardDocument) -> StoryboardDocument:
    """Keep user-locked scenes at their relative positions; fill the rest
    from the regenerated document, in its order. Deterministic — the agent
    never has to cooperate for locked scenes to survive. Ids stay unique:
    a fresh scene colliding with a locked id is renamed."""
    if not any(s.locked for s in old_doc.scenes):
        return new_doc
    fresh = iter(new_doc.scenes)
    merged: list[StoryboardScene] = []
    for old_scene in old_doc.scenes:
        if old_scene.locked:
            merged.append(old_scene)
        else:
            nxt = next(fresh, None)
            if nxt is not None:
                merged.append(nxt)
    merged.extend(fresh)  # a longer regeneration appends the remainder
    placed: set[str] = set()
    counter = 0
    out: list[StoryboardScene] = []
    for scene in merged:
        if scene.id in placed:
            counter += 1
            while f"s{counter}" in placed:
                counter += 1
            scene = scene.model_copy(update={"id": f"s{counter}"})
        placed.add(scene.id)
        out.append(scene)
    return new_doc.model_copy(update={"scenes": out})


class StoryboardAgent:
    """LLM-backed planning; every stored document passes the catalog gate."""

    def __init__(self, settings: Settings, client: VLMClient | None = None) -> None:
        self.settings = settings
        self.client = client or VLMClient(settings)

    # -- generation ---------------------------------------------------------

    def generate(
        self, conn: sqlite3.Connection, *,
        objective: str, target_duration_s: float, style: str, name: str,
        music_path: str | None = None, mode: str | None = None,
    ) -> str:
        """Plan a full storyboard; returns the new storyboard id."""
        digest = build_digest(conn)
        user = (
            f"Objective: {objective}\nTarget duration: {target_duration_s:.0f} seconds\n"
            f"Style: {style}\n\nCatalog digest:\n{json.dumps(digest, indent=1)}\n\n"
            "Return the JSON object now."
        )
        answer = self._ask(user)
        document, script = self._parse_full(answer, objective=objective)
        problems = validate_document(conn, document, script)
        if problems:
            # One bounded repair round: the errors go back with the digest.
            logger.warning("agent draft invalid, requesting repair: %s", problems)
            user2 = (
                f"Your previous storyboard had these contract violations:\n"
                f"{json.dumps(problems, indent=1)}\n\nOriginal request:\n{user}\n\n"
                "Return a corrected JSON object now."
            )
            document, script = self._parse_full(self._ask(user2), objective=objective)
            problems = validate_document(conn, document, script)
            if problems:
                raise AgentUnavailable(
                    f"agent storyboard still invalid after repair: {problems[:4]}")

        title = script.title or name
        return doc_mod.create_board(
            conn, name=title, objective=objective, style=style,
            target_duration_s=target_duration_s, document=document, script=script,
            music_path=music_path, mode=mode, origin="agent",
            note="storyboard agent",
        )

    def regenerate(
        self, conn: sqlite3.Connection, storyboard_id: str, *, scene_id: str | None = None,
    ) -> int:
        """Regenerate the whole board (locked scenes preserved) or one scene.

        Returns the new current version. A single-scene failure leaves the
        scene untouched and records an open issue instead of aborting.
        """
        row = doc_mod.get_board(conn, storyboard_id)
        document, _ = doc_mod.load_document(conn, storyboard_id)
        script = doc_mod.load_script(conn, storyboard_id) or ScriptDocument(
            storyboard_id=storyboard_id)

        if scene_id is not None:
            scene = next((s for s in document.scenes if s.id == scene_id), None)
            if scene is None:
                raise AutoVidError(f"storyboard has no scene {scene_id!r}")
            try:
                replacement = self._regen_one_scene(conn, row, document, script, scene)
            except (AgentUnavailable, ValidationError, ValueError) as e:
                logger.warning("scene regeneration failed, keeping current scene: %s", e)
                scene.open_issues.append(f"regeneration failed: {str(e)[:120]}")
                return doc_mod.save_document(
                    conn, storyboard_id, document, origin="agent",
                    note=f"scene {scene_id} regeneration failed (scene kept)")
            idx = document.scenes.index(scene)
            document.scenes[idx] = replacement
            document.scenes[idx].stale = False
            return doc_mod.save_document(
                conn, storyboard_id, document, origin="agent",
                note=f"scene {scene_id} regenerated")

        digest = build_digest(conn)
        user = (
            f"Objective: {row['objective']}\nTarget duration: "
            f"{float(row['target_duration_s']):.0f} seconds\nStyle: {row['style']}\n\n"
            f"The current storyboard (to improve upon — do not just copy it):\n"
            f"{json.dumps(_doc_brief(document), indent=1)}\n\n"
            f"Catalog digest:\n{json.dumps(digest, indent=1)}\n\n"
            "Return the JSON object now."
        )
        new_doc, new_script = self._parse_full(self._ask(user), objective=row["objective"])
        problems = validate_document(conn, new_doc, new_script)
        if problems:
            logger.warning("agent regeneration invalid after one draft: %s", problems)
            raise AgentUnavailable(f"regenerated storyboard invalid: {problems[:4]}")
        merged = merge_locked(document, new_doc)
        return doc_mod.save_document(
            conn, storyboard_id, merged, origin="agent", note="full regeneration")

    # -- internals ----------------------------------------------------------

    def _ask(self, user: str) -> str:
        return self.client.complete_text(
            system=_SYSTEM_PROMPT, user=user, model=self.settings.agent_model or None,
        )

    def _parse_full(self, answer: str, *, objective: str) \
            -> tuple[StoryboardDocument, ScriptDocument]:
        data = _extract_json(answer)
        scenes_raw = data.get("scenes") or []
        alternates = {a.get("scene_id"): a.get("segment_ids") or []
                      for a in data.get("alternates") or []}
        scenes = []
        for s in scenes_raw:
            scene = StoryboardScene(
                id=str(s.get("id") or ""),
                title=s.get("title") or "",
                description=s.get("description") or "",
                narrative_role=s.get("narrative_role") or "",
                beat_id=s.get("beat_id"),
                shots=[SceneShot(
                    segment_id=int(sh["segment_id"]),
                    asset_id="",  # resolved from the segment at validation/compile
                    source_start=float(sh.get("source_start") or 0.0),
                    source_end=float(sh.get("source_end") or 0.0),
                    transition_out="fade" if sh.get("transition_out") == "fade" else "cut",
                ) for sh in (s.get("shots") or [])],
                est_duration_s=sum(
                    float(sh.get("source_end") or 0) - float(sh.get("source_start") or 0)
                    for sh in (s.get("shots") or [])),
                pacing=s.get("pacing") or "",
                transition="fade" if s.get("transition") == "fade" else "cut",
                alternates=[int(x) for x in alternates.get(s.get("id"), [])],
                open_issues=[str(x) for x in (s.get("open_issues") or [])],
                confidence=float(s.get("confidence") or 0.5),
            )
            scenes.append(scene)
        document = StoryboardDocument(
            scenes=scenes,
            missing_footage=[str(x) for x in (data.get("missing_footage") or [])],
            notes=str(data.get("notes") or ""),
        )
        script = ScriptDocument(
            storyboard_id="",
            title=data.get("title") or "",
            synopsis=data.get("synopsis") or objective,
            beats=[ScriptBeat(
                id=str(b.get("id") or ""),
                role=b.get("role") or "",
                narration=b.get("narration") or "",
                dialogue=b.get("dialogue") or "",
                on_screen=b.get("on_screen") or "",
            ) for b in (data.get("beats") or [])],
            origin="agent",
        )
        return document, script

    def _regen_one_scene(
        self, conn: sqlite3.Connection, row: sqlite3.Row,
        document: StoryboardDocument, script: ScriptDocument, scene: StoryboardScene,
    ) -> StoryboardScene:
        digest = build_digest(conn)
        i = document.scenes.index(scene)
        neighbors = [document.scenes[j] for j in (i - 1, i + 1) if 0 <= j < len(document.scenes)]
        user = (
            f"Objective: {row['objective']}\nStyle: {row['style']}\n\n"
            f"Rewrite ONLY this storyboard scene:\n{scene.model_dump_json(indent=1)}\n\n"
            f"Neighbouring scenes for continuity:\n"
            f"{json.dumps([_scene_brief(n) for n in neighbors], indent=1)}\n\n"
            f"Script beats:\n{json.dumps([b.model_dump() for b in script.beats], indent=1)}\n\n"
            f"Catalog digest:\n{json.dumps(digest, indent=1)}\n\n"
            'Answer with STRICT JSON only: a single scene object '
            '{"id": "...", "title": "...", "description": "...", "narrative_role": "...", '
            '"beat_id": "...", "shots": [{"segment_id": 0, "source_start": 0.0, '
            '"source_end": 0.0, "transition_out": "cut|fade"}], "pacing": "...", '
            '"transition": "cut|fade", "open_issues": [], "confidence": 0.0-1.0} '
            "using only real segment ids from the digest. Keep the scene's id."
        )
        data = _extract_json(self._ask(user))
        if "scenes" in data and isinstance(data["scenes"], list):
            data = data["scenes"][0]
        data["id"] = scene.id  # the scene keeps its identity
        replacement = StoryboardScene(
            id=scene.id,
            title=data.get("title") or scene.title,
            description=data.get("description") or scene.description,
            narrative_role=data.get("narrative_role") or scene.narrative_role,
            beat_id=data.get("beat_id") or scene.beat_id,
            shots=[SceneShot(
                segment_id=int(sh["segment_id"]),
                asset_id="",
                source_start=float(sh.get("source_start") or 0.0),
                source_end=float(sh.get("source_end") or 0.0),
                transition_out="fade" if sh.get("transition_out") == "fade" else "cut",
            ) for sh in (data.get("shots") or [])],
            est_duration_s=sum(
                float(sh.get("source_end") or 0) - float(sh.get("source_start") or 0)
                for sh in (data.get("shots") or [])),
            pacing=data.get("pacing") or "",
            transition="fade" if data.get("transition") == "fade" else "cut",
            alternates=scene.alternates,
            open_issues=[str(x) for x in (data.get("open_issues") or [])],
            confidence=float(data.get("confidence") or 0.5),
            locked=scene.locked,
        )
        probe_doc = StoryboardDocument(scenes=[replacement])
        problems = validate_document(conn, probe_doc, script)
        if problems:
            raise AgentUnavailable(f"regenerated scene invalid: {problems[:3]}")
        return replacement


def _scene_brief(scene: StoryboardScene) -> dict:
    return {
        "id": scene.id, "title": scene.title, "description": scene.description,
        "narrative_role": scene.narrative_role, "beat_id": scene.beat_id,
        "shots": [{"segment_id": sh.segment_id,
                   "source_start": sh.source_start, "source_end": sh.source_end}
                  for sh in scene.shots],
    }


def _doc_brief(document: StoryboardDocument) -> dict:
    return {
        "scenes": [_scene_brief(s) for s in document.scenes],
        "missing_footage": document.missing_footage,
    }


def generate_with_fallback(
    conn: sqlite3.Connection, settings: Settings, *,
    objective: str, target_duration_s: float, style: str, name: str,
) -> tuple[str, str]:
    """Agent-first generation with the deterministic planner as the documented
    fallback. Returns ``(storyboard_id, origin)``."""
    client = VLMClient(settings)
    try:
        if not client.configured:
            logger.info("no VLM key — storyboard falls back to the deterministic planner")
        else:
            try:
                board_id = StoryboardAgent(settings, client).generate(
                    conn, objective=objective, target_duration_s=target_duration_s,
                    style=style, name=name,
                )
                return board_id, "agent"
            except (AgentUnavailable, ValidationError, ValueError) as e:
                logger.warning("agent generation failed (%s) — falling back to planner", e)
    finally:
        client.close()
    board_id = doc_mod.build_deterministic(
        conn, objective=objective, target_duration_s=target_duration_s,
        style=style, name=name,
    )
    return board_id, "planner"
