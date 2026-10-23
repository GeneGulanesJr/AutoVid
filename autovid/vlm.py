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
import time
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


class VLMClient:
    """Thin sync client; one HTTP call per window."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport  # injectable for tests
        self._client: httpx.Client | None = None
        self.usage: list[dict[str, int]] = []

    def _http(self) -> httpx.Client:
        """Pooled client for the VLMClient's lifetime.

        One client (one connection pool) means TCP+TLS setup is paid once per
        run instead of once per window per attempt.
        """
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.settings.vlm_timeout_s, transport=self._transport
            )
        return self._client

    def close(self) -> None:
        """Release the pooled connection (idempotent)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.settings.vlm_api_key)

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
            "temperature": 0.2,
            "max_tokens": self.settings.vlm_max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.settings.vlm_api_key}"}
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.settings.vlm_max_retries + 1):
            try:
                resp = self._http().post(url, json=body, headers=headers)
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
                    last_error = VLMUnavailable("model returned maximal uncertainty")
                    continue
                return analysis
            except (httpx.HTTPError, KeyError, IndexError, ValueError, ValidationError) as e:
                last_error = e
                logger.warning("VLM call attempt %d failed: %s", attempt + 1, e)
                if isinstance(e, httpx.HTTPError) and attempt < self.settings.vlm_max_retries:
                    # 429/5xx/network: back off so retries don't hammer the
                    # endpoint (parse errors retry at once — a delay can't fix them).
                    time.sleep(min(2.0 * (2**attempt), 8.0))
        raise VLMUnavailable(f"VLM call failed after retries: {last_error}")
