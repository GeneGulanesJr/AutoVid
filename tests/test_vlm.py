"""VLM adapter tests: request shape, JSON robustness, retries (mock transport)."""
from __future__ import annotations

import json

import httpx
import pytest

from autovid.config import Settings
from autovid.vlm import VLMClient, VLMUnavailable

GOOD_ANALYSIS = {
    "description": "A person walks through a market.",
    "activity": "walking",
    "objects": ["stalls", "umbrellas"],
    "environment": "outdoor market",
    "camera_motion": "handheld",
    "highlight_score": 0.7,
    "uncertainty": 0.2,
    "continues_previous": False,
}


def _completion(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 60},
    }


def _transport(responder) -> httpx.MockTransport:
    return httpx.MockTransport(responder)


def _client(settings: Settings, transport) -> VLMClient:
    settings = settings.model_copy(update={
        "vlm_api_key": "test-key",
        "vlm_base_url": "https://vlm.test/api",
        "vlm_model": "vlm-test-1",
        "vlm_max_retries": 1,
    })
    return VLMClient(settings, transport=transport)


def _frames() -> list[tuple[float, str]]:
    return [(0.0, "AAAA"), (2.0, "BBBB")]


def test_describe_window_posts_images_and_parses_json(settings: Settings) -> None:
    seen: dict = {}

    def responder(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        body = json.loads(request.content)
        seen["model"] = body["model"]
        content = body["messages"][1]["content"]
        seen["image_parts"] = [p for p in content if p.get("type") == "image_url"]
        seen["texts"] = [p["text"] for p in content if p.get("type") == "text"]
        return httpx.Response(200, json=_completion(json.dumps(GOOD_ANALYSIS)))

    client = _client(settings, _transport(responder))
    out = client.describe_window(frames=_frames(), previous_description="prior scene")

    assert out.description == GOOD_ANALYSIS["description"]
    assert out.activity == "walking"
    assert out.highlight_score == pytest.approx(0.7)
    assert out.camera_motion == "handheld"
    assert seen["url"] == "https://vlm.test/api/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["model"] == "vlm-test-1"
    assert len(seen["image_parts"]) == 2
    assert seen["image_parts"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,AAAA")
    assert any("t=2.00s" in t for t in seen["texts"])
    assert any("prior scene" in t for t in seen["texts"])
    assert client.usage == [{"prompt_tokens": 120, "completion_tokens": 60}]


def test_describe_window_tolerates_markdown_fences(settings: Settings) -> None:
    fenced = "Here you go:\n```json\n" + json.dumps(GOOD_ANALYSIS) + "\n```"
    client = _client(settings, _transport(lambda req: httpx.Response(200, json=_completion(fenced))))
    out = client.describe_window(frames=_frames())
    assert out.activity == "walking"


def test_describe_window_retries_on_bad_json(settings: Settings) -> None:
    calls = {"n": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        content = json.dumps(GOOD_ANALYSIS) if calls["n"] > 1 else "not json at all"
        return httpx.Response(200, json=_completion(content))

    client = _client(settings, _transport(responder))
    out = client.describe_window(frames=_frames())
    assert calls["n"] == 2
    assert out.description == GOOD_ANALYSIS["description"]


def test_describe_window_fails_after_retries(settings: Settings) -> None:
    client = _client(
        settings, _transport(lambda req: httpx.Response(200, json=_completion("garbage")))
    )
    with pytest.raises(VLMUnavailable):
        client.describe_window(frames=_frames())
    assert len(client.usage) == 2  # both attempts' usage recorded


def test_describe_window_requires_api_key(settings: Settings) -> None:
    client = VLMClient(settings)  # no key
    with pytest.raises(VLMUnavailable, match="API_KEY"):
        client.describe_window(frames=_frames())
