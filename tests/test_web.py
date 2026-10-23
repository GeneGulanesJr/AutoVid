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
