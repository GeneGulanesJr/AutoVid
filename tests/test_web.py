"""Web review-UI tests (TestClient over a seeded catalog)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autovid import db, sampling
from autovid.analyze import analyze_asset
from autovid.config import Settings
from autovid.decisions import (
    BoundaryVerdict,
    DecisionClient,
    SegmentVerdict,
)
from autovid.edl import EDL, EDLItem, save as save_edl
from autovid.refine import cut_asset
from autovid.vlm import VLMClient, WindowAnalysis
from tests.conftest import make_multiscene_video


def _analysis(**overrides) -> WindowAnalysis:
    base = dict(
        description="market scene", activity="walking", objects=["stalls"],
        environment="market", camera_motion="static",
        highlight_score=0.8, uncertainty=0.1, continues_previous=False,
    )
    base.update(overrides)
    return WindowAnalysis(**base)


@pytest.fixture
def client(settings: Settings, tmp_path: Path, monkeypatch) -> TestClient:
    from autovid.web import create_app

    conn = db.connect(settings.db_path)
    clip = make_multiscene_video(tmp_path / "ms.mp4", scene_s=3.0, scenes=3)
    from autovid.ingest import ingest_file

    _, asset_id = ingest_file(conn, settings, clip)
    sampling.sample_asset(conn, settings, asset_id)
    coarse = settings.model_copy(update={"coarse_window_s": 3.0})
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)
    monkeypatch.setattr(vlm, "describe_window", lambda **kw: _analysis())
    analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    monkeypatch.setattr(
        decisions, "decide_boundary",
        lambda a, c, b, f: BoundaryVerdict("boundary", "at", 0.9, 0.9, "test", 0.0),
    )
    monkeypatch.setattr(
        decisions, "decide_segment",
        lambda a, s, frames, duration_s=None, min_duration_s=0.5: SegmentVerdict(
            "accept", 0.9, 0.9, 2, 0.9, "test", 0.0
        ),
    )
    cut_asset(conn, coarse, asset_id, decisions=decisions)
    # one pending segment for the review queue
    conn.execute("UPDATE segments SET review_status = 'pending' WHERE start_s = 0.0")
    conn.commit()
    conn.close()

    return TestClient(create_app(settings))


def test_index_serves_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "AutoVid" in r.text


def test_assets_endpoint(client: TestClient) -> None:
    rows = client.get("/api/assets").json()
    assert len(rows) == 1
    assert rows[0]["status"] == "cut"
    assert rows[0]["segments"] == 3


def test_segments_and_search(client: TestClient) -> None:
    rows = client.get("/api/segments").json()
    assert len(rows) == 3
    hits = client.get("/api/segments", params={"q": "market"}).json()
    assert len(hits) == 3


def test_review_status_update(client: TestClient) -> None:
    seg_id = next(
        s["id"] for s in client.get("/api/segments").json() if s["review_status"] == "pending"
    )
    r = client.post(f"/api/segments/{seg_id}/review?status=approved")
    assert r.json()["review_status"] == "approved"
    assert client.post(f"/api/segments/{seg_id}/review?status=bogus").status_code == 400


def test_edl_create_get_put(client: TestClient) -> None:
    seg_ids = [s["id"] for s in client.get("/api/segments").json()]
    board = EDL(
        name="web", objective="test", style="cinematic", target_duration_s=9.0,
        items=[
            EDLItem(clip_id=sid, asset_id="a_x", source_start=0.0, source_end=3.0)
            for sid in seg_ids
        ],
    )
    made = client.post("/api/edls", json=board.model_dump()).json()
    edl_id = made["id"]

    fetched = client.get(f"/api/edls/{edl_id}").json()
    assert [i["clip_id"] for i in fetched["items"]] == seg_ids

    # reorder via PUT (what the UI's move buttons do, then save)
    fetched["items"].reverse()
    updated = client.put(f"/api/edls/{edl_id}", json={
        "name": "web", "objective": "test", "style": "cinematic",
        "items": fetched["items"],
    }).json()
    assert [i["clip_id"] for i in updated["items"]] == list(reversed(seg_ids))


def _await_job(client: TestClient, job: dict, timeout_s: float = 120.0) -> dict:
    """Poll a background job to completion (render/music run as jobs now)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["state"] != "running":
            return j
        time.sleep(0.2)
    raise AssertionError(f"job {job['id']} did not finish in {timeout_s}s")


def test_render_via_api(client: TestClient) -> None:
    seg_ids = [s["id"] for s in client.get("/api/segments").json()]
    board = EDL(
        name="webrender", objective="test", style="cinematic", target_duration_s=9.0,
        items=[
            EDLItem(clip_id=sid, asset_id="a_x", source_start=0.0, source_end=3.0)
            for sid in seg_ids
        ],
    )
    edl_id = client.post("/api/edls", json=board.model_dump()).json()["id"]
    job = _await_job(client, client.post(f"/api/edls/{edl_id}/render").json())
    assert job["state"] == "done", job["error"]
    out = job["result"]
    assert Path(out["path"]).exists()
    assert abs(out["duration_s"] - 9.0) < 1.5

    video = client.get(f"/renders/{edl_id}/webrender.mp4")
    assert video.status_code == 200
    assert video.headers["content-type"].startswith("video/mp4")


def test_clip_serving(client: TestClient) -> None:
    seg = next(
        s for s in client.get("/api/segments").json() if s["clip_path"]
    )
    r = client.get(f"/clips/{seg['id']}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/mp4")


# --- storyboards: Script · Storyboard · EDL ---------------------------------

def _agent_payload(seg_ids: list[int]) -> dict:
    """A catalog-valid agent answer over the fixture's three segments."""
    a, b, c = seg_ids
    return {
        "title": "Market Morning",
        "synopsis": "Three looks at the market.",
        "beats": [
            {"id": "b1", "role": "hook", "narration": "The market opens.",
             "on_screen": "6 AM", "dialogue": ""},
            {"id": "b2", "role": "development", "narration": "Stalls fill up.",
             "on_screen": "", "dialogue": ""},
        ],
        "scenes": [
            {"id": "s1", "title": "Open", "description": "market wide",
             "narrative_role": "hook", "beat_id": "b1",
             "shots": [{"segment_id": a, "source_start": 0.0, "source_end": 2.5,
                        "transition_out": "cut"}],
             "pacing": "", "transition": "cut", "open_issues": [], "confidence": 0.9},
            {"id": "s2", "title": "Fill", "description": "crowds arrive",
             "narrative_role": "development", "beat_id": "b2",
             "shots": [{"segment_id": b, "source_start": 3.0, "source_end": 5.5,
                        "transition_out": "fade"}],
             "pacing": "", "transition": "fade", "open_issues": [], "confidence": 0.8},
        ],
        "alternates": [{"scene_id": "s1", "segment_ids": [c]}],
        "missing_footage": ["an aerial of the square"],
    }


def test_storyboard_generate_fallback_and_full_flow(client: TestClient) -> None:
    """No VLM key in the fixture → documented planner fallback, then the
    whole edit cycle: PUT scenes, script edit → stale, compile → EDL."""
    job = _await_job(client, client.post("/api/storyboards/generate", json={
        "objective": "market recap", "target_duration_s": 6.0,
        "style": "cinematic", "name": "sb-fallback"}).json())
    assert job["state"] == "done", job["error"]
    assert job["result"]["origin"] == "planner"
    sb_id = job["result"]["storyboard_id"]

    board = client.get(f"/api/storyboards/{sb_id}").json()
    assert board["status"] == "draft"
    assert board["script"] is not None
    scenes = board["document"]["scenes"]
    assert scenes and all(s["shots"] for s in scenes)

    # full-document PUT reorders and bumps a user_edit version
    reordered = list(reversed(scenes))
    r = client.put(f"/api/storyboards/{sb_id}", json={"scenes": reordered})
    assert r.status_code == 200 and r.json()["version"] == 2

    # editing a script beat marks its scene stale (deliberate propagation)
    script = board["script"]
    first_scene_beat = reordered[0]["beat_id"]
    beats = [
        {**b, "narration": b["narration"] + " (edited)"}
        if b["id"] == first_scene_beat else b
        for b in script["beats"]
    ]
    r = client.put(f"/api/storyboards/{sb_id}/script", json={"beats": beats})
    assert r.status_code == 200
    assert reordered[0]["id"] in r.json()["stale_scenes"]

    # restore v1 (history: v1 planner, v2 user_edit, v3 script_edit → v4)
    r = client.post(f"/api/storyboards/{sb_id}/restore", json={"version": 1})
    assert r.json()["restored_from"] == 1 and r.json()["version"] == 4

    # compile → EDL linked to the storyboard
    compiled = client.post(f"/api/storyboards/{sb_id}/compile").json()
    assert "edl_id" in compiled
    edl = client.get(f"/api/edls/{compiled['edl_id']}").json()
    assert edl["storyboard_id"] == sb_id
    assert edl["items"], "compiled EDL carries the scenes as items"

    # approve toggles status
    r = client.post(f"/api/storyboards/{sb_id}/approve?approve=true")
    assert r.json()["status"] == "approved"


def test_storyboard_generate_via_agent(client: TestClient, monkeypatch) -> None:
    seg_ids = [s["id"] for s in client.get("/api/segments").json()]
    payload = _agent_payload(seg_ids)
    calls = []

    def fake_complete_text(self, *, system, user, model=None, temperature=0.3):
        calls.append(user)
        return json.dumps(payload)

    # the fixture has no key: force the configured check so the agent path runs
    monkeypatch.setattr("autovid.vlm.VLMClient.configured",
                        property(lambda self: True))
    monkeypatch.setattr("autovid.vlm.VLMClient.complete_text", fake_complete_text)
    job = _await_job(client, client.post("/api/storyboards/generate", json={
        "objective": "market recap", "target_duration_s": 5.0,
        "style": "cinematic", "name": "sb-agent"}).json())
    assert job["state"] == "done", job["error"]
    assert job["result"]["origin"] == "agent"
    assert len(calls) == 1
    board = client.get(f"/api/storyboards/{job['result']['storyboard_id']}").json()
    assert board["name"] == "Market Morning"
    assert board["document"]["missing_footage"] == ["an aerial of the square"]

    # hallucinated segment ids are rejected at the PUT gate too
    bad = board["document"]["scenes"]
    bad[0]["shots"][0]["segment_id"] = 424242
    r = client.put(f"/api/storyboards/{board['id']}", json={"scenes": bad})
    assert r.status_code == 400 and "unknown segment" in r.json()["detail"]


def test_storyboard_regenerate_requires_key(client: TestClient) -> None:
    job = _await_job(client, client.post("/api/storyboards/generate", json={
        "objective": "recap", "target_duration_s": 6.0}).json())
    sb_id = job["result"]["storyboard_id"]
    scene_id = client.get(f"/api/storyboards/{sb_id}").json()["document"]["scenes"][0]["id"]
    r = client.post(f"/api/storyboards/{sb_id}/regenerate", json={"scene_id": scene_id})
    assert r.status_code == 409, "regeneration needs the agent; the fixture has no key"


def test_edl_validate_endpoint(client: TestClient) -> None:
    board = EDL(
        name="broken", objective="t", style="cinematic", target_duration_s=9.0,
        items=[EDLItem(clip_id=9999, asset_id="a_x", source_start=0.0, source_end=3.0)],
    )
    edl_id = client.post("/api/edls", json=board.model_dump()).json()["id"]
    v = client.get(f"/api/edls/{edl_id}/validate").json()
    assert not v["ok"] and any(e["code"] == "missing_segment" for e in v["errors"])


def test_render_blocked_by_validation(client: TestClient) -> None:
    board = EDL(
        name="norender", objective="t", style="cinematic", target_duration_s=9.0,
        items=[EDLItem(clip_id=9999, asset_id="a_x", source_start=0.0, source_end=3.0)],
    )
    edl_id = client.post("/api/edls", json=board.model_dump()).json()["id"]
    r = client.post(f"/api/edls/{edl_id}/render")
    assert r.status_code == 400 and "unknown segment" in r.json()["detail"]
