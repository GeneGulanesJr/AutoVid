"""Settings persistence: read and update the ``.env`` file from the app.

The ``.env`` stays the single source of truth (pydantic-settings reads it on
startup). Updates rewrite the file **in place**, preserving comments and
unrelated lines; keys not yet present are appended. Secret values never
round-trip to the client — ``mask()`` shows only a hint.
"""
from __future__ import annotations

from pathlib import Path

from .config import REPO_ROOT, Settings

ENV_PATH = REPO_ROOT / ".env"

# Keys the settings page may edit, grouped for the UI. Secrets render masked.
SETTING_GROUPS: list[dict] = [
    {
        "group": "Cloud VLM (video understanding)",
        "fields": [
            {"key": "AUTOVID_VLM_API_KEY", "label": "API key", "secret": True,
             "help": "OpenAI-compatible vision API key (Z.ai, OpenRouter, OpenAI…)."},
            {"key": "AUTOVID_VLM_BASE_URL", "label": "Base URL", "secret": False},
            {"key": "AUTOVID_VLM_MODEL", "label": "Model", "secret": False,
             "help": "e.g. glm-4.6v, qwen-vl-max, gpt-4o"},
        ],
    },
    {
        "group": "Decision layer (local d1-omni)",
        "fields": [
            {"key": "AUTOVID_DECISIONMCP_URL", "label": "DecisionMCP URL", "secret": False,
             "help": "Streamable-HTTP MCP endpoint of your DecisionMCP server."},
            {"key": "AUTOVID_DECISION_MODE", "label": "Mode", "secret": False,
             "help": "auto = use it, fall back to heuristics; d1 = require it; "
                     "heuristic = never call it."},
        ],
    },
    {
        "group": "Pipeline",
        "fields": [
            {"key": "AUTOVID_COARSE_WINDOW_S", "label": "Coarse window (s)", "secret": False,
             "help": "Analysis window length — a 1-hour file becomes 60 of these."},
            {"key": "AUTOVID_GRID_INTERVAL_S", "label": "Grid interval (s)", "secret": False},
            {"key": "AUTOVID_HIGHLIGHT_THRESHOLD", "label": "Highlight threshold", "secret": False,
             "help": "Windows below this VLM score never become clips (0–1)."},
        ],
    },
]

EDITABLE_KEYS = {f["key"] for g in SETTING_GROUPS for f in g["fields"]}


def read_env_lines(env_path: Path | None = None) -> list[str]:
    p = env_path or ENV_PATH
    if p.exists():
        # utf-8-sig: a BOM from "UTF-8 with BOM" editors must not corrupt the
        # first key's name.
        return p.read_text(encoding="utf-8-sig").splitlines()
    return []


def upsert_env_values(values: dict[str, str], env_path: Path | None = None) -> None:
    """Set ``values`` in the .env file, preserving everything else.

    Creates the file (from the shipped example, if present) when missing.
    Empty-string values keep the key present with an empty value.
    """
    p = env_path or ENV_PATH
    if not p.exists():
        example = REPO_ROOT / ".env.example"
        lines = (example.read_text(encoding="utf-8-sig").splitlines()
                 if example.exists() else [])
    else:
        lines = p.read_text(encoding="utf-8-sig").splitlines()

    remaining = dict(values)
    out: list[str] = []
    for line in lines:
        name = line.split("=", 1)[0].strip() if "=" in line and not line.strip().startswith("#") else None
        if name in remaining:
            out.append(f"{name}={remaining.pop(name)}")
        else:
            out.append(line)
    for name, value in remaining.items():
        out.append(f"{name}={value}")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def mask_secret(value: str) -> str:
    """Set/unset only — the masked view must never leak key material."""
    return "" if not value else "••••"


def current_settings_view(settings: Settings) -> dict:
    """Settings grouped for the UI, showing the **effective** values the
    pipeline would use (the passed Settings). Secret values are masked."""
    groups = []
    for group in SETTING_GROUPS:
        fields = []
        for f in group["fields"]:
            key = f["key"]
            is_secret = f.get("secret", False)
            current = getattr(settings, key.removeprefix("AUTOVID_").lower(), None)
            value = "" if current is None else str(current)
            fields.append({
                **f,
                "value": mask_secret(value) if is_secret else value,
                "set": bool(value),
            })
        groups.append({"group": group["group"], "fields": fields})
    return {"groups": groups, "env_path": str(ENV_PATH)}
