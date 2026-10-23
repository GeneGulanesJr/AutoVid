"""Decision-layer client: DecisionMCP over MCP streamable HTTP.

Every analyzer decision (sampling policy, boundary verification, segment
acceptance) goes through here. Two modes:

- ``auto`` (default): call the DecisionMCP server; if it is unreachable, fall
  back to documented heuristics so the pipeline keeps running.
- ``d1``: the server is required; unreachability raises.
- ``heuristic``: never call the server.

Every decision — heuristic or model — is logged to the ``decisions`` table
with the action, confidence, engine and elapsed time, so decision-guided
processing stays measurable against the fixed-interval baseline.

Note: ``DecisionClient`` is sync (the pipeline is sync); MCP calls run on a
private background event loop over ONE cached streamable-HTTP session, so the
connect+initialize handshake is paid once per client instead of once per call
(≈100+ calls per hour of footage). Any transport failure drops the cached
session, so the next call reconnects — server restarts between calls are
survived exactly as before.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .config import Settings

logger = logging.getLogger(__name__)


class DecisionUnavailable(RuntimeError):
    """The DecisionMCP server could not be reached or the call failed."""


class DecisionProtocolError(RuntimeError):
    """The server answered, but not in the expected shape.

    This is a code/protocol drift between DecisionMCP and AutoVid, not an
    outage — it propagates instead of triggering the heuristic fallback.
    """


# ---------------------------------------------------------------------------
# Inputs / results
# ---------------------------------------------------------------------------


class RegionStats(BaseModel):
    """What the sampler knows about a region — composed into the model state."""

    asset_id: str
    region_id: str
    start_s: float
    end_s: float
    scene_changes: int = 0
    mean_motion: float = 0.0
    max_motion: float = 0.0
    current_interval_s: float
    vlm_uncertainty: float | None = None


@dataclass(frozen=True)
class RegionAction:
    action: str  # sufficient | densify | inspect_more | flag_review
    suggested_interval_s: float | None
    has_content: float
    ambiguity_level: int
    confidence: float
    engine: str  # "heuristic" or the engine identity from the server
    elapsed_ms: float


@dataclass(frozen=True)
class BoundaryVerdict:
    verdict: str  # boundary | shift | not_boundary
    position_hint: str  # before | at | after
    cut_probability: float
    confidence: float
    engine: str
    elapsed_ms: float


@dataclass(frozen=True)
class ContinuityVerdict:
    relation: str  # same_scene | different_scene | needs_verification
    shared_subjects: float
    change_level: int
    confidence: float
    engine: str
    elapsed_ms: float


@dataclass(frozen=True)
class SequenceReview:
    action: str  # approve | replace_clip | slow_pace | fast_pace
    concern: str  # none | pacing | repetition | mood_fit
    confidence: float
    engine: str
    elapsed_ms: float


@dataclass(frozen=True)
class MoodVerdict:
    mood: str  # calm | neutral | energetic | driving
    energy_level: int
    builds: float
    confidence: float
    engine: str
    elapsed_ms: float


@dataclass(frozen=True)
class SegmentVerdict:
    verdict: str  # accept | needs_evidence | reject | flag_review
    is_highlight: float
    self_contained: float
    quality_level: int
    confidence: float
    engine: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DecisionClient:
    """Typed wrapper around the three DecisionMCP video tools + fallbacks."""

    def __init__(self, settings: Settings, conn: sqlite3.Connection | None = None) -> None:
        self.settings = settings
        self.conn = conn
        self.mode = settings.decision_mode
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_lock = threading.Lock()
        self._session: Any = None  # mcp ClientSession (lazy import keeps startup light)
        self._session_stack: Any = None  # AsyncExitStack owning the session's contexts
        self._session_lock: asyncio.Lock | None = None

    def close(self) -> None:
        """Release the cached MCP session and the background loop (idempotent)."""
        with self._loop_lock:
            loop, self._loop = self._loop, None
        if loop is None or loop.is_closed():
            return
        if loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._reset_session(), loop).result(timeout=10)
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.debug("session teardown failed", exc_info=True)
        loop.call_soon_threadsafe(loop.stop)

    # -- public API ----------------------------------------------------------

    def decide_region(self, stats: RegionStats) -> RegionAction:
        """Next bounded sampling action for a region."""
        started = time.perf_counter()
        args = {
            "summary": json.dumps(stats.model_dump(), separators=(",", ":")),
            "current_interval_s": stats.current_interval_s,
            "min_interval_s": 0.5,
            "max_interval_s": 30.0,
        }
        engine = "heuristic"
        if self.mode != "heuristic":
            try:
                raw = self._call_tool("decision_video_region", args)
                out = self._region_from_tool(raw)
                self._log(stats.asset_id, "decision_video_region", out.action, raw,
                          out.confidence, out.engine, out.elapsed_ms)
                return out
            except DecisionUnavailable as e:
                if self.mode == "d1":
                    raise
                logger.warning("Decision server unavailable, heuristic fallback: %s", e)
        out = self._region_heuristic(stats, started)
        self._log(stats.asset_id, "decision_video_region", out.action,
                  {"stats": stats.model_dump()}, out.confidence, "heuristic", out.elapsed_ms)
        return out

    def decide_boundary(
        self, asset_id: str, context: str, before: list[str], after: list[str]
    ) -> BoundaryVerdict:
        """Verify a candidate cut from frames before/after it."""
        started = time.perf_counter()
        args = {"context": context, "before": before, "after": after}
        if self.mode != "heuristic":
            try:
                raw = self._call_tool("decision_video_boundary", args)
                out = self._boundary_from_tool(raw)
                self._log(asset_id, "decision_video_boundary", out.verdict, raw,
                          out.confidence, out.engine, out.elapsed_ms)
                return out
            except DecisionUnavailable as e:
                if self.mode == "d1":
                    raise
                logger.warning("Decision server unavailable, heuristic fallback: %s", e)
        # No vision in fallback mode: never claim a boundary we cannot see.
        out = BoundaryVerdict(
            verdict="shift", position_hint="at", cut_probability=0.5,
            confidence=0.0, engine="heuristic",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self._log(asset_id, "decision_video_boundary", out.verdict, {"reason": "fallback"},
                  out.confidence, "heuristic", out.elapsed_ms)
        return out

    def decide_continuity(
        self,
        asset_id: str,
        context: str,
        images: list[str] | None = None,
        *,
        near_cut: bool = False,
    ) -> ContinuityVerdict:
        """Phase 2 — do two adjacent windows belong to one continuous scene?

        ``images`` (end-of-left, start-of-right frames) are forwarded when the
        server's engine is vision-capable. Fallback (no server): trust the
        scene-change detector — a nearby cut means different scenes, no cut
        means the scene continues.
        """
        started = time.perf_counter()
        args = {"context": context, "images": images or []}
        if self.mode != "heuristic":
            try:
                raw = self._call_tool("decision_video_continuity", args)
                out = self._continuity_from_tool(raw)
                self._log(asset_id, "decision_video_continuity", out.relation, raw,
                          out.confidence, out.engine, out.elapsed_ms)
                return out
            except DecisionUnavailable as e:
                if self.mode == "d1":
                    raise
                logger.warning("Decision server unavailable, heuristic fallback: %s", e)
        relation = "different_scene" if near_cut else "same_scene"
        out = ContinuityVerdict(
            relation=relation,
            shared_subjects=0.5,
            change_level=2 if near_cut else 0,
            confidence=0.0,
            engine="heuristic",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self._log(asset_id, "decision_video_continuity", out.relation,
                  {"near_cut": near_cut, "reason": "fallback"},
                  out.confidence, "heuristic", out.elapsed_ms)
        return out

    def decide_sequence_review(self, context: str) -> SequenceReview:
        """Review a drafted music-section cut plan (bounded planner adjustments).

        Music plans span multiple assets, so the decision is logged without an
        asset reference. Fallback (no server): approve — the deterministic
        planner stands.
        """
        started = time.perf_counter()
        args = {"plan": context}
        if self.mode != "heuristic":
            try:
                raw = self._call_tool("decision_music_sequence", args)
                out = self._sequence_from_tool(raw)
                self._log(None, "decision_music_sequence", out.action, raw,
                          out.confidence, out.engine, out.elapsed_ms)
                return out
            except DecisionUnavailable as e:
                if self.mode == "d1":
                    raise
                logger.warning("Decision server unavailable, heuristic fallback: %s", e)
        out = SequenceReview(
            action="approve", concern="none", confidence=0.0, engine="heuristic",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self._log(None, "decision_music_sequence", out.action,
                  {"reason": "fallback"}, out.confidence, "heuristic", out.elapsed_ms)
        return out

    def decide_music_mood(
        self, context: str, audio_b64: str, *, fallback_mood: str = "neutral"
    ) -> MoodVerdict:
        """Phase 2 of music-driven editing — classify a section by listening.

        The audio-capable engine (d1-omni) hears the section audio; precise
        timing stays in librosa. Fallback (no server / no audio engine):
        ``fallback_mood`` — the caller's energy-derived guess.
        """
        started = time.perf_counter()
        args = {"context": context, "audio_pcm16_b64": audio_b64}
        if self.mode != "heuristic":
            try:
                raw = self._call_tool("decision_music_mood", args)
                out = self._mood_from_tool(raw)
                self._log(None, "decision_music_mood", out.mood, raw,
                          out.confidence, out.engine, out.elapsed_ms)
                return out
            except DecisionUnavailable as e:
                if self.mode == "d1":
                    raise
                logger.warning("Decision server unavailable, energy fallback: %s", e)
        out = MoodVerdict(
            mood=fallback_mood, energy_level=1, builds=0.0,
            confidence=0.0, engine="heuristic",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self._log(None, "decision_music_mood", out.mood,
                  {"fallback_mood": fallback_mood}, out.confidence, "heuristic",
                  out.elapsed_ms)
        return out

    def decide_segment(
        self,
        asset_id: str,
        summary: str,
        frames: list[str],
        duration_s: float | None = None,
        min_duration_s: float = 0.5,
    ) -> SegmentVerdict:
        """Accept/reject a candidate clip from evidence frames + summary."""
        started = time.perf_counter()
        args = {
            "summary": summary, "frames": frames,
            "duration_s": duration_s, "min_duration_s": min_duration_s,
        }
        if self.mode != "heuristic":
            try:
                raw = self._call_tool("decision_video_segment", args)
                out = self._segment_from_tool(raw)
                self._log(asset_id, "decision_video_segment", out.verdict, raw,
                          out.confidence, out.engine, out.elapsed_ms)
                return out
            except DecisionUnavailable as e:
                if self.mode == "d1":
                    raise
                logger.warning("Decision server unavailable, heuristic fallback: %s", e)
        out = SegmentVerdict(
            verdict="needs_evidence", is_highlight=0.5, self_contained=0.5,
            quality_level=1, confidence=0.0, engine="heuristic",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self._log(asset_id, "decision_video_segment", out.verdict, {"reason": "fallback"},
                  out.confidence, "heuristic", out.elapsed_ms)
        return out

    # -- transport -----------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Private background loop, so sessions outlive individual sync calls."""
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._loop.run_forever, daemon=True, name="decisionmcp-client"
                ).start()
            return self._loop

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """One MCP call over the cached session; reconnects after any failure."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool_async(name, arguments), self._ensure_loop()
            )
            return future.result()
        except DecisionUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 — any transport failure is "unavailable"
            raise DecisionUnavailable(f"{name}: {e}") from e

    async def _get_session(self) -> Any:
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        if self._session_lock is None:
            self._session_lock = asyncio.Lock()
        async with self._session_lock:
            if self._session is not None:
                return self._session
            stack = AsyncExitStack()
            try:
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(self.settings.decisionmcp_url)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), timeout=10.0)
            except BaseException:
                await stack.aclose()
                raise
            self._session_stack = stack
            self._session = session
            return session

    async def _reset_session(self) -> None:
        stack, self._session_stack, self._session = self._session_stack, None, None
        self._session_lock = None  # a Lock binds to its loop; never reuse one
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:  # noqa: BLE001 — closing must never mask the real error
                logger.debug("MCP session close failed", exc_info=True)

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict:
        try:
            session = await self._get_session()
            result = await asyncio.wait_for(
                session.call_tool(name, arguments=arguments), timeout=60.0
            )
        except Exception:  # noqa: BLE001 — broken/hung session → fresh one next call
            await self._reset_session()
            raise
        if getattr(result, "isError", False):
            text = result.content[0].text if result.content else "unknown error"
            raise DecisionUnavailable(f"{name} error: {text}")
        if not result.content:
            raise DecisionUnavailable(f"{name} returned no content")
        return json.loads(result.content[0].text)

    # -- tool output → results ------------------------------------------------

    @staticmethod
    def _engine_of(raw: dict) -> str:
        model = raw.get("model")
        if not model:
            details = raw.get("details")
            if isinstance(details, dict):
                model = details.get("model")
        return str(model) if model else "decisionmcp"

    def _region_from_tool(self, raw: dict) -> RegionAction:
        from pydantic import ValidationError

        from ._tool_shapes import VideoRegionOutput

        try:
            out = VideoRegionOutput.model_validate(raw)
        except ValidationError as e:
            raise DecisionProtocolError(f"unexpected region output: {e}") from e
        return RegionAction(
            action=out.action,
            suggested_interval_s=out.suggested_interval_s,
            has_content=out.has_content,
            ambiguity_level=out.ambiguity_level,
            confidence=out.confidence,
            engine=self._engine_of(raw),
            elapsed_ms=out.details.get("elapsed_ms", 0.0) if isinstance(out.details, dict) else 0.0,
        )

    def _boundary_from_tool(self, raw: dict) -> BoundaryVerdict:
        from pydantic import ValidationError

        from ._tool_shapes import VideoBoundaryOutput

        try:
            out = VideoBoundaryOutput.model_validate(raw)
        except ValidationError as e:
            raise DecisionProtocolError(f"unexpected boundary output: {e}") from e
        return BoundaryVerdict(
            verdict=out.verdict,
            position_hint=out.position_hint,
            cut_probability=out.cut_probability,
            confidence=out.confidence,
            engine=self._engine_of(raw),
            elapsed_ms=0.0,
        )

    def _continuity_from_tool(self, raw: dict) -> ContinuityVerdict:
        from pydantic import ValidationError

        from ._tool_shapes import VideoContinuityOutput

        try:
            out = VideoContinuityOutput.model_validate(raw)
        except ValidationError as e:
            raise DecisionProtocolError(f"unexpected continuity output: {e}") from e
        return ContinuityVerdict(
            relation=out.relation,
            shared_subjects=out.shared_subjects,
            change_level=out.change_level,
            confidence=out.confidence,
            engine=self._engine_of(raw),
            elapsed_ms=0.0,
        )

    def _sequence_from_tool(self, raw: dict) -> SequenceReview:
        from pydantic import ValidationError

        from ._tool_shapes import MusicSequenceOutput

        try:
            out = MusicSequenceOutput.model_validate(raw)
        except ValidationError as e:
            raise DecisionProtocolError(f"unexpected sequence review output: {e}") from e
        return SequenceReview(
            action=out.action,
            concern=out.concern,
            confidence=out.confidence,
            engine=self._engine_of(raw),
            elapsed_ms=0.0,
        )

    def _mood_from_tool(self, raw: dict) -> MoodVerdict:
        from pydantic import ValidationError

        from ._tool_shapes import MusicMoodOutput

        try:
            out = MusicMoodOutput.model_validate(raw)
        except ValidationError as e:
            raise DecisionProtocolError(f"unexpected mood output: {e}") from e
        return MoodVerdict(
            mood=out.mood,
            energy_level=out.energy_level,
            builds=out.builds,
            confidence=out.confidence,
            engine=self._engine_of(raw),
            elapsed_ms=0.0,
        )

    def _segment_from_tool(self, raw: dict) -> SegmentVerdict:
        from pydantic import ValidationError

        from ._tool_shapes import VideoSegmentOutput

        try:
            out = VideoSegmentOutput.model_validate(raw)
        except ValidationError as e:
            raise DecisionProtocolError(f"unexpected segment output: {e}") from e
        return SegmentVerdict(
            verdict=out.verdict,
            is_highlight=out.is_highlight,
            self_contained=out.self_contained,
            quality_level=out.quality_level,
            confidence=out.confidence,
            engine=self._engine_of(raw),
            elapsed_ms=0.0,
        )

    # -- heuristics (documented fallbacks, low confidence by construction) ----

    @staticmethod
    def _region_heuristic(stats: RegionStats, started: float) -> RegionAction:
        """Motion/scene-change rules; conservative stand-in for the model."""
        elapsed = (time.perf_counter() - started) * 1000
        if stats.vlm_uncertainty is not None and stats.vlm_uncertainty >= 0.6:
            action = "flag_review"
        elif stats.mean_motion >= 0.25 or stats.scene_changes >= 2:
            action = "densify"
        elif stats.mean_motion >= 0.10 or stats.scene_changes == 1:
            action = "inspect_more"
        else:
            action = "sufficient"
        suggested = stats.current_interval_s / 2 if action == "densify" else None
        return RegionAction(
            action=action,
            suggested_interval_s=suggested,
            has_content=min(1.0, stats.mean_motion * 4),
            ambiguity_level=3 if action == "flag_review" else 1,
            confidence=0.0,
            engine="heuristic",
            elapsed_ms=elapsed,
        )

    # -- logging ---------------------------------------------------------------

    def _log(
        self,
        asset_id: str,
        tool: str,
        action: str | None,
        payload: Any,
        confidence: float,
        engine: str,
        elapsed_ms: float,
    ) -> None:
        if self.conn is None:
            return
        try:
            payload_json = json.dumps(payload, default=str)
            if len(payload_json) > 4096:  # audit trail, not a data store — cap dumps
                payload_json = payload_json[:4096]
            self.conn.execute(
                """INSERT INTO decisions (asset_id, tool, action, payload_json,
                                          confidence, engine_version, elapsed_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (asset_id, tool, action, payload_json,
                 confidence, engine, elapsed_ms),
            )
            self.conn.commit()
        except Exception:  # noqa: BLE001 — logging must never fail a decision
            logger.exception("Failed to log decision (%s/%s)", asset_id, tool)
