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
    if not text:
        raise ValueError("empty answer")
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in answer")
    # Prefer parsing everything from the first '{' — brace counting alone
    # truncates objects whose string values contain '}'. The balanced scan
    # only handles what's left: trailing prose after a valid object.
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        pass
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

    def _chat(self, body: dict[str, Any], what: str) -> str:
        """POST one chat completion and return the answer text.

        Transient errors (429 / 5xx / network) are retried with backoff; client errors (bad key,
        parameter or model) are not, since they cannot succeed. Token usage is counted.
        """
        headers = {"Authorization": f"Bearer {self.settings.vlm_api_key}"}
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"
        last_error: str | None = None
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
                return payload["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
                last_error = _detail(e)
                logger.warning("%s attempt %d failed: %s", what, attempt + 1, last_error)
                if _permanent(e):
                    break
                if isinstance(e, httpx.HTTPError) and attempt < self.settings.vlm_max_retries:
                    time.sleep(min(2.0 * (2**attempt), 8.0))
        raise VLMUnavailable(f"{what} failed after retries: {last_error}")

    def ping(self) -> None:
        """One tiny call proving the key, base URL and model work; raises VLMUnavailable with the reason."""
        if not self.configured:
            raise VLMUnavailable("no AUTOVID_VLM_API_KEY configured")
        body = {"model": self.settings.vlm_model, "max_tokens": 16,
                "messages": [{"role": "user", "content": "Reply with OK."}]}
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"
        try:
            self._http().post(url, json=body, headers={"Authorization": f"Bearer {self.settings.vlm_api_key}"},
                              timeout=min(self.settings.vlm_timeout_s, 30.0)).raise_for_status()
        except httpx.HTTPError as e:
            raise VLMUnavailable(f"VLM check failed ({self.settings.vlm_model} @ {self.settings.vlm_base_url}): "
                                 f"{_detail(e)}") from e

    def complete_text(self, *, system: str, user: str, model: str | None = None,
                      temperature: float | None = None) -> str:
        """Text-only completion (no images) — the Storyboard Agent's channel.

        Same endpoint, key, retry/backoff and usage accounting as
        :meth:`describe_window`; the answer is raw text (the caller parses
        and validates the JSON). ``temperature`` is only sent when given: newer
        models (Claude Haiku 5.5) reject the parameter outright.
        """
        if not self.configured:
            raise VLMUnavailable("no AUTOVID_VLM_API_KEY configured")
        body: dict[str, Any] = {
            "model": model or self.settings.vlm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.settings.vlm_max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        return self._chat(body, "text completion")

    def ask_json(self, system: str, user: str | list[dict[str, Any]]) -> dict[str, Any]:
        """Chat call that must answer with one JSON object (one re-ask for a malformed answer).

        ``user`` is text, or a list of content parts (text and image_url) for image questions.
        """
        if not self.configured:
            raise VLMUnavailable("no AUTOVID_VLM_API_KEY configured")
        body: dict[str, Any] = {
            "model": self.settings.vlm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.settings.vlm_max_tokens,
        }
        last_error = ""
        for _ in range(2):
            try:
                return _extract_json(self._chat(body, "LLM call"))
            except ValueError as e:
                last_error = str(e)
        raise VLMUnavailable(f"LLM answer was not JSON: {last_error}")

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
                    last_error = "model returned maximal uncertainty"
                    continue
                return analysis
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, ValidationError) as e:
                last_error = _detail(e)
                logger.warning("VLM call attempt %d failed: %s", attempt + 1, last_error)
                if _permanent(e):
                    break
                if isinstance(e, httpx.HTTPError) and attempt < self.settings.vlm_max_retries:
                    # 429/5xx/network: back off so retries don't hammer the
                    # endpoint (parse errors retry at once — a delay can't fix them).
                    time.sleep(min(2.0 * (2**attempt), 8.0))
        raise VLMUnavailable(f"VLM call failed after retries: {last_error}")
