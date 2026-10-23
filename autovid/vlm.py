"""Cloud VLM adapter — the primary video-understanding model.

Speaks the OpenAI-compatible chat-completions API (works with Z.ai GLM-V,
OpenAI, and any compatible gateway) with image parts as data URIs. Responses
must be strict JSON matching :class:`WindowAnalysis`; malformed answers are
retried up to ``vlm_max_retries`` and then surface as :class:`VLMUnavailable`.

Structured output beats free-form text here: the temporal map built from these
answers feeds selection and editing, so every field is typed and every call is
logged (tokens, cost inputs) to the ``analyses`` table by the caller.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .errors import AutoVidError
from .media import data_uri

logger = logging.getLogger(__name__)

CAMERA_MOTIONS = ("static", "pan", "tilt", "zoom", "handheld", "dolly", "other")


class VLMUnavailable(AutoVidError):
    """The VLM call failed or the answer could not be parsed after retries."""


class WindowAnalysis(BaseModel):
    """One analyzed window of one asset — a row of the temporal map."""

    description: str = Field(..., min_length=1)
    activity: str | None = None
    objects: list[str] = Field(default_factory=list)
    # Dishes/drinks that are the subject of the shot (not tableware); empty if not food footage.
    foods: list[str] = Field(default_factory=list)
    environment: str | None = None
    camera_motion: Literal["static", "pan", "tilt", "zoom", "handheld", "dolly", "other"] | None = (
        None
    )
    highlight_score: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    continues_previous: bool = False


_SYSTEM_PROMPT = """You analyze sampled frames from one window of a raw video. \
Answer with STRICT JSON only (no markdown, no prose) with exactly these keys:
"description": one or two factual sentences about what happens in this window;
"activity": the main activity or null;
"objects": notable visible objects (list of short strings);
"foods": specific dishes or drinks that are the subject of these frames, named \
as a menu would (e.g. "pork siomai", "iced milk tea"); only real food or drink physically \
present, not pictures on menus, signs or posters, and not tableware; [] if none;
"environment": the setting/location type or null;
"camera_motion": one of static|pan|tilt|zoom|handheld|dolly|other or null;
"highlight_score": 0.0-1.0, how much a rough cut should want this window;
"uncertainty": 0.0-1.0, how unsure you are given only these sampled frames;
"continues_previous": whether this window appears to continue the scene \
described in the preceding context (true) or starts a new one (false)."""


def _extract_json(text: str) -> dict[str, Any]:
    """First balanced JSON object in the answer, tolerating markdown fences."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in answer")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in answer")


def _detail(e: Exception) -> str:
    """The error plus the server's own explanation (a bare "400 Bad Request" hid `temperature` once)."""
    if isinstance(e, httpx.HTTPStatusError):
        return f"{e} — {e.response.text.strip()[:300]}"
    return str(e)


def _permanent(e: Exception) -> bool:
    """A client error (bad key, bad parameter, unknown model) cannot succeed on retry; 429 can."""
    return (isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500
            and e.response.status_code not in (408, 429))


def list_models(
    base_url: str, api_key: str, transport: httpx.BaseTransport | None = None
) -> list[str]:
    """Model ids from an OpenAI-style ``GET {base_url}/models``.

    Sends both auth styles so the same call works for Anthropic (x-api-key +
    anthropic-version) and OpenAI-compatible providers (Bearer).
    """
    if not api_key:
        raise VLMUnavailable("no API key configured")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        with httpx.Client(timeout=15.0, transport=transport) as client:
            resp = client.get(base_url.rstrip("/") + "/models", headers=headers)
        resp.raise_for_status()
        return sorted(m["id"] for m in resp.json()["data"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
        raise VLMUnavailable(f"could not list models from {base_url}: {e}") from e


class VLMClient:
    """Thin sync client; one HTTP call per window."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport  # injectable for tests
        self.usage: list[dict[str, int]] = []

    @property
    def configured(self) -> bool:
        return bool(self.settings.vlm_api_key)

    def ping(self) -> None:
        """One tiny call proving the key, base URL and model work; raises VLMUnavailable with the reason."""
        if not self.configured:
            raise VLMUnavailable("no AUTOVID_VLM_API_KEY configured")
        body = {"model": self.settings.vlm_model, "max_tokens": 16,
                "messages": [{"role": "user", "content": "Reply with OK."}]}
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"
        try:
            with httpx.Client(timeout=min(self.settings.vlm_timeout_s, 30.0), transport=self._transport) as client:
                client.post(url, json=body, headers={"Authorization": f"Bearer {self.settings.vlm_api_key}"}
                            ).raise_for_status()
        except httpx.HTTPError as e:
            raise VLMUnavailable(f"VLM check failed ({self.settings.vlm_model} @ {self.settings.vlm_base_url}): "
                                 f"{_detail(e)}") from e

    def ask_json(self, system: str, user: str | list[dict[str, Any]]) -> dict[str, Any]:
        """Chat call that must answer with one JSON object (retried like a window).

        ``user`` is text, or a list of content parts (text and image_url) for image questions.
        """
        if not self.configured:
            raise VLMUnavailable("no AUTOVID_VLM_API_KEY configured")
        body = {
            "model": self.settings.vlm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.settings.vlm_max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.settings.vlm_api_key}"}
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"
        last_error: str | None = None
        for attempt in range(self.settings.vlm_max_retries + 1):
            try:
                with httpx.Client(timeout=self.settings.vlm_timeout_s, transport=self._transport) as client:
                    resp = client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                return _extract_json(resp.json()["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                last_error = _detail(e)
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, last_error)
                if _permanent(e):
                    break
        raise VLMUnavailable(f"LLM call failed after retries: {last_error}")

    def describe_window(
        self,
        *,
        frames: list[tuple[float, str]],
        previous_description: str | None = None,
        window_hint: str = "",
    ) -> WindowAnalysis:
        """Analyze one window: ``frames`` is ``[(ts, b64_jpeg), ...]`` in order."""
        if not self.configured:
            raise VLMUnavailable("no AUTOVID_VLM_API_KEY configured")

        ts_line = ", ".join(f"frame {i + 1} at t={ts:.2f}s" for i, (ts, _) in enumerate(frames))
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": f"Frames in this window, in order: {ts_line}."}
        ]
        for _, b64 in frames:
            parts.append({"type": "image_url", "image_url": {"url": data_uri(b64)}})
        if previous_description:
            parts.append({
                "type": "text",
                "text": f"Previous window (for continuity): {previous_description}",
            })
        if window_hint:
            parts.append({"type": "text", "text": window_hint})
        parts.append({"type": "text", "text": "Return the JSON object now."})

        body: dict[str, Any] = {
            "model": self.settings.vlm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": parts},
            ],
            "max_tokens": self.settings.vlm_max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.settings.vlm_api_key}"}
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"

        last_error: str | None = None
        for attempt in range(self.settings.vlm_max_retries + 1):
            try:
                with httpx.Client(
                    timeout=self.settings.vlm_timeout_s, transport=self._transport
                ) as client:
                    resp = client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                usage = payload.get("usage") or {}
                self.usage.append({
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                })
                text = payload["choices"][0]["message"]["content"]
                analysis = WindowAnalysis.model_validate(_extract_json(text))
                if analysis.uncertainty >= 0.99 and attempt < self.settings.vlm_max_retries:
                    # An answer that admits total uncertainty is worth one re-ask.
                    last_error = "model returned maximal uncertainty"
                    continue
                return analysis
            except (httpx.HTTPError, KeyError, IndexError, ValueError, ValidationError) as e:
                last_error = _detail(e)
                logger.warning("VLM call attempt %d failed: %s", attempt + 1, last_error)
                if _permanent(e):
                    break
        raise VLMUnavailable(f"VLM call failed after retries: {last_error}")
