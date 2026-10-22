"""Mirrors of the DecisionMCP video tool output schemas.

Validating the server's response against these keeps a protocol drift between
DecisionMCP and AutoVid a loud, typed error instead of silently wrong data.
The definitions must match decision_mcp/tools/video_*.py on the server side.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class VideoRegionOutput(BaseModel):
    action: str
    suggested_interval_s: float | None = Field(default=None, gt=0)
    has_content: float = Field(..., ge=0.0, le=1.0)
    ambiguity_level: int = Field(..., ge=0)
    ambiguity_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class VideoBoundaryOutput(BaseModel):
    verdict: str
    position_hint: str
    cut_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class VideoContinuityOutput(BaseModel):
    relation: str  # same_scene | different_scene | needs_verification
    shared_subjects: float = Field(..., ge=0.0, le=1.0)
    change_level: int = Field(..., ge=0, le=3)
    change_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class MusicSequenceOutput(BaseModel):
    action: str  # approve | replace_clip | slow_pace | fast_pace
    concern: str  # none | pacing | repetition | mood_fit
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class MusicMoodOutput(BaseModel):
    mood: str  # calm | neutral | energetic | driving
    energy_level: int = Field(..., ge=0, le=3)
    builds: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class VideoSegmentOutput(BaseModel):
    verdict: str
    is_highlight: float = Field(..., ge=0.0, le=1.0)
    self_contained: float = Field(..., ge=0.0, le=1.0)
    quality_level: int = Field(..., ge=0, le=3)
    quality_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)
