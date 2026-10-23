"""AutoVid settings.

Every field maps to an ``AUTOVID_*`` environment variable (or the ``.env``
file in the repo root), mirroring DecisionMCP's pydantic-settings config.
The ``.env`` path is absolute so the desktop launcher works from any
working directory.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUTOVID_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    # Catalog root: SQLite DB, thumbnails, clips, renders. Originals stay put.
    data_dir: Path = REPO_ROOT / "catalog"

    # ffmpeg/ffprobe location. Empty → PATH → vendored .tools build. The
    # ffmpeg_io layer reads the raw AUTOVID_FFMPEG_DIR environment variable
    # (get_settings re-exports it below, so a .env-only value works too).
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

    # Storyboard Agent (text-only planner over the catalog digest). Empty →
    # same model as the VLM; point it at a cheaper/fast text model if desired.
    agent_model: str = ""

    # Baseline sampling policy (the decision layer may override per region).
    grid_interval_s: float = 2.0
    thumb_interval_s: float = 30.0
    thumb_width: int = 320
    scene_threshold: float = 0.3

    # Windows/scenes below this VLM highlight score never become candidate
    # clips (lower it for sparse/abstract footage).
    highlight_threshold: float = 0.4

    # Hierarchical segmentation, phase 1: coarse fixed-duration analysis
    # windows. A 1-hour file becomes 60 ~1-minute windows, each described by
    # one VLM call — the model never reasons across more than this much
    # footage at once. The decision layer then merges/splits these windows
    # into logical scenes (phases 2–4 in refine.py).
    coarse_window_s: float = 60.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "autovid.db"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # ffmpeg_io resolves binaries from the process environment; re-export a
    # value that only came from .env so it takes effect. Runs before the
    # first ffmpeg call (every entry point builds settings first), and tests
    # construct Settings directly, so this never fires under pytest.
    if settings.ffmpeg_dir is not None and "AUTOVID_FFMPEG_DIR" not in os.environ:
        os.environ["AUTOVID_FFMPEG_DIR"] = str(settings.ffmpeg_dir)
    return settings
