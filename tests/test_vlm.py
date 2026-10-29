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


def test_list_models_sends_both_auth_styles_and_sorts():
    seen = {}

    def responder(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["bearer"] = request.headers["authorization"]
        seen["x_api_key"] = request.headers["x-api-key"]
        return httpx.Response(200, json={"data": [{"id": "b-model"}, {"id": "a-model"}]})

    from autovid.vlm import list_models

    ids = list_models("https://vlm.test/v1/", "k", _transport(responder))
    assert ids == ["a-model", "b-model"]
    assert seen == {"url": "https://vlm.test/v1/models", "bearer": "Bearer k", "x_api_key": "k"}


def test_list_models_errors_become_vlm_unavailable():
    from autovid.vlm import list_models

    with pytest.raises(VLMUnavailable):
        list_models("https://vlm.test/v1", "k", _transport(lambda r: httpx.Response(401)))
    with pytest.raises(VLMUnavailable):
        list_models("https://vlm.test/v1", "")


def test_ping_ok_and_failure_carries_the_servers_reason(settings):
    ok = _client(settings, _transport(lambda r: httpx.Response(200, json=_completion("OK"))))
    ok.ping()  # no exception

    bad = _client(settings, _transport(lambda r: httpx.Response(
        400, json={"error": {"message": "`temperature` is deprecated for this model."}})))
    with pytest.raises(VLMUnavailable, match="temperature.*deprecated"):
        bad.ping()


def test_client_errors_are_not_retried_but_server_errors_are(settings):
    calls = []

    def responder(status):
        def _r(request):
            calls.append(status)
            return httpx.Response(status, json={"error": {"message": "nope"}})
        return _r

    client = _client(settings, _transport(responder(400)))  # vlm_max_retries = 1 in the fixture helper
    with pytest.raises(VLMUnavailable, match="nope"):
        client.describe_window(frames=[(0.0, "AAAA")])
    assert calls == [400]                                         # one attempt: a 400 cannot succeed later

    calls.clear()
    with pytest.raises(VLMUnavailable):
        _client(settings, _transport(responder(503))).describe_window(frames=[(0.0, "AAAA")])
    assert calls == [503, 503]                                    # transient: retried


def test_non_ascii_key_fails_with_the_real_cause(settings):
    """A pasted-in mask (••••) in a header surfaces as a bare 'ascii' codec error — name it."""
    from autovid.vlm import list_models

    with pytest.raises(VLMUnavailable, match="non-ASCII"):
        list_models("https://vlm.test/api", "••••")
    client = _client(settings, _transport(lambda r: httpx.Response(200, json=_completion("OK"))))
    client.settings = client.settings.model_copy(update={"vlm_api_key": "••••"})
    with pytest.raises(VLMUnavailable, match="non-ASCII"):
        client.ping()
