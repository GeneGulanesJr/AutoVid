"""AutoVid settings.

Every field maps to an ``AUTOVID_*`` environment variable (or a ``.env`` file
in the working directory), mirroring DecisionMCP's pydantic-settings config.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUTOVID_", env_file=".env", extra="ignore")

    # Catalog root: SQLite DB, thumbnails, clips, renders. Originals stay put.
    data_dir: Path = REPO_ROOT / "catalog"

    # ffmpeg/ffprobe location. Empty → PATH → vendored .tools build.
    ffmpeg_dir: Path | None = None

    # Decision layer (DecisionMCP, streamable HTTP MCP).
    decisionmcp_url: str = "http://127.0.0.1:8765/mcp"
    # auto = call DecisionMCP, fall back to heuristics when unreachable;
    # d1 = require the server; heuristic = never call it.
    decision_mode: str = "auto"

    # Primary video-understanding VLM (OpenAI-compatible chat API).
    vlm_base_url: str = "https://api.z.ai/api/paas/v4"
    vlm_api_key: str = ""
    vlm_model: str = "glm-4.6v"
    vlm_timeout_s: float = 120.0
    vlm_max_retries: int = 2
    # glm-4.6v-class models are reasoners: they spend tokens thinking before the
    # JSON answer, so the budget must cover both or the answer truncates.
    vlm_max_tokens: int = 4096

    # Speech-to-text (faster-whisper, local CPU). d1-omni only classifies audio, it can't transcribe.
    # `small` garbles Tagalog/Taglish; full large-v3 beats large-v3-turbo on it (turbo is the faster fallback).
    whisper_model: str = "large-v3"
    # whispercpp = whisper.cpp + Vulkan (AMD/Intel/NVIDIA GPUs); faster-whisper = CPU or NVIDIA CUDA only;
    # auto = whispercpp when its executable exists under whispercpp_dir, else faster-whisper.
    whisper_backend: str = "auto"
    whispercpp_dir: Path = Path.home() / ".whisper-vulkan"  # bin/whisper-cli.exe, models/ggml-<model>.bin
    # faster-whisper only: auto = CUDA when ctranslate2 sees an NVIDIA GPU, else CPU.
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    # Baseline sampling policy (the decision layer may override per region).
    grid_interval_s: float = 2.0
    thumb_interval_s: float = 30.0
    thumb_width: int = 320
    scene_threshold: float = 0.3

    # Hierarchical segmentation, phase 1: coarse fixed-duration analysis
    # windows. A 1-hour file becomes 60 ~1-minute windows, each described by
    # one VLM call — the model never reasons across more than this much
    # footage at once. The decision layer then merges/splits these windows
    # into logical scenes (phases 2–4 in refine.py).
    coarse_window_s: float = 60.0
    # Clips up to fine_window_max_duration_s are analyzed in fine_window_s windows instead (when that is
    # shorter): a dish or a reaction can then be placed to ~8 s, not to a minute. Long footage keeps coarse.
    fine_window_s: float = 8.0
    fine_window_max_duration_s: float = 600.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "autovid.db"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"


@lru_cache
def get_settings() -> Settings:
    return Settings()
