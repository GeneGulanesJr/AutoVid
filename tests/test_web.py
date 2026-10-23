"""Web review-UI tests (TestClient over a seeded catalog)."""
from __future__ import annotations

import json
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
    out = client.post(f"/api/edls/{edl_id}/render").json()
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


def test_vlm_test_endpoint_reports_the_servers_reason(client: TestClient, monkeypatch) -> None:
    from autovid.vlm import VLMUnavailable

    monkeypatch.setattr(VLMClient, "ping", lambda self: None)
    ok = client.post("/api/vlm/test", json={"api_key": "k", "model": "m"})
    assert ok.status_code == 200 and ok.json() == {"ok": True, "model": "m"}

    def refuse(self):
        raise VLMUnavailable("VLM check failed: 400 — `temperature` is deprecated for this model.")

    monkeypatch.setattr(VLMClient, "ping", refuse)
    bad = client.post("/api/vlm/test", json={})
    assert bad.status_code == 502 and "temperature" in bad.json()["detail"]


def test_script_endpoint_needs_transcripts_then_returns_story_and_findings(
        client: TestClient, settings: Settings, monkeypatch) -> None:
    r = client.post("/api/script", json={"objective": "a day out"})
    assert r.status_code == 409 and "transcribe" in r.json()["detail"]

    conn = db.connect(settings.db_path)
    aid = conn.execute("SELECT id FROM assets").fetchone()["id"]
    line = conn.execute("INSERT INTO transcripts (asset_id, start_s, end_s, text) VALUES (?, 1, 3, 'Tara na!')",
                        (aid,)).lastrowid
    conn.commit()
    conn.close()
    monkeypatch.setattr(VLMClient, "ask_json", lambda self, system, user: {
        "title": "Day out", "beats": [{"label": "intro", "order": [f"L{line}"]}]})
    r = client.post("/api/script", json={"objective": "a day out", "name": "t"})
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Day out" and body["items"] == 1
    assert {f["name"] for f in body["findings"]} >= {"length", "replay"}


def test_transcribe_stage_runs_for_assets_with_audio(client: TestClient, settings: Settings, monkeypatch) -> None:
    import time

    import autovid.transcribe as tr

    conn = db.connect(settings.db_path)
    conn.execute("UPDATE assets SET has_audio = 1")
    conn.commit()
    conn.close()
    monkeypatch.setattr(tr, "load_model", lambda s: None)
    monkeypatch.setattr(tr, "transcribe_asset", lambda c, s, aid, model=None: 3)
    job = client.post("/api/pipeline/transcribe", json={}).json()
    for _ in range(50):
        view = client.get(f"/api/jobs/{job['id']}").json()
        if view["state"] != "running":
            break
        time.sleep(0.1)
    assert view["state"] == "done" and any("3 utterances" in line for line in view["log"])


def test_settings_page_offers_the_whisper_and_window_options(client: TestClient) -> None:
    groups = {g["group"]: [f["key"] for f in g["fields"]] for g in client.get("/api/settings").json()["groups"]}
    assert "AUTOVID_WHISPER_BACKEND" in groups["Speech-to-text (transcribe + story)"]
    assert "AUTOVID_FINE_WINDOW_S" in groups["Pipeline"]
