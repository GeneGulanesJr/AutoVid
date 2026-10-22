"""App-layer tests: settings store, setup checks, background pipeline jobs."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autovid import db
from autovid.config import Settings
from autovid.settings_store import (
    EDITABLE_KEYS,
    current_settings_view,
    mask_secret,
    upsert_env_values,
)


# ---------------------------------------------------------------------------
# settings store
# ---------------------------------------------------------------------------


def test_mask_secret() -> None:
    assert mask_secret("") == ""
    assert mask_secret("short") == "••••"
    masked = mask_secret("abcdefghijklmnop")
    assert masked.startswith("abcd")
    assert "bcdefghijklmn" not in masked


def test_upsert_preserves_comments_and_updates_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# my comments stay\n"
        "AUTOVID_VLM_API_KEY=old\n"
        "AUTOVID_VLM_MODEL=glm-4.6v\n",
        encoding="utf-8",
    )
    upsert_env_values({"AUTOVID_VLM_API_KEY": "new", "AUTOVID_VLM_MODEL": "gpt-4o"},
                      env_path=env)
    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# my comments stay"
    assert "AUTOVID_VLM_API_KEY=new" in lines
    assert "AUTOVID_VLM_MODEL=gpt-4o" in lines
    assert len(lines) == 3  # nothing duplicated


def test_upsert_creates_file_appending_new_keys(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    upsert_env_values({"AUTOVID_VLM_API_KEY": "k"}, env_path=env)
    assert "AUTOVID_VLM_API_KEY=k" in env.read_text(encoding="utf-8").splitlines()


def test_current_settings_view_masks_secrets(settings: Settings) -> None:
    settings = settings.model_copy(update={"vlm_api_key": "supersecretkey123"})
    view = current_settings_view(settings)
    flat = {f["key"]: f for g in view["groups"] for f in g["fields"]}
    assert flat["AUTOVID_VLM_API_KEY"]["value"].startswith("supe")
    assert "supersecretkey123" not in flat["AUTOVID_VLM_API_KEY"]["value"]
    assert set(flat) == EDITABLE_KEYS


# ---------------------------------------------------------------------------
# app API: settings roundtrip, setup checks, background pipeline job
# ---------------------------------------------------------------------------


@pytest.fixture
def client(settings: Settings) -> TestClient:
    from autovid.web import create_app

    return TestClient(create_app(settings))


def test_setup_checks_report(settings: Settings, client: TestClient) -> None:
    checks = {c["name"]: c for c in client.get("/api/setup").json()["checks"]}
    assert checks["ffmpeg"]["ok"] is True           # vendored build found
    assert checks["VLM key"]["ok"] is False         # hermetic settings: no key
    assert "DecisionMCP" in checks


def test_settings_roundtrip_updates_env(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(data_dir=tmp_path / "catalog", _env_file=None, vlm_api_key="")
    monkeypatch.setattr("autovid.settings_store.ENV_PATH", tmp_path / ".env")
    monkeypatch.chdir(tmp_path)  # get_settings() reads ./.env, like the real app
    from autovid.config import get_settings
    from autovid.web import create_app

    client = TestClient(create_app(settings))

    before = client.get("/api/settings").json()
    key_field = next(f for g in before["groups"] for f in g["fields"]
                     if f["key"] == "AUTOVID_VLM_MODEL")
    assert key_field["value"] == settings.vlm_model

    r = client.put("/api/settings", json={"values": {"AUTOVID_VLM_MODEL": "gpt-4o"}})
    assert r.json()["saved"] == ["AUTOVID_VLM_MODEL"]

    after = client.get("/api/settings").json()
    model_field = next(f for g in after["groups"] for f in g["fields"]
                       if f["key"] == "AUTOVID_VLM_MODEL")
    assert model_field["value"] == "gpt-4o"          # UI sees the new value
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "AUTOVID_VLM_MODEL=gpt-4o" in env_text.splitlines()

    bad = client.put("/api/settings", json={"values": {"AUTOVID_DATA_DIR": "x"}})
    assert bad.status_code == 400                    # paths stay out of the UI

    get_settings.cache_clear()  # don't leak the tmp .env into other tests


def test_pipeline_ingest_job_runs_to_done(settings: Settings, tmp_path: Path) -> None:
    from tests.conftest import make_video

    folder = tmp_path / "footage"
    folder.mkdir()
    make_video(folder / "clip.mp4", duration=2)

    settings = settings.model_copy(update={"_env_file": None})
    from autovid.web import create_app

    client = TestClient(create_app(settings))
    job = client.post("/api/pipeline/ingest",
                      json={"folder": str(folder), "recursive": True}).json()
    assert job["stage"] == "ingest"

    import time

    detail = job
    for _ in range(200):
        detail = client.get(f"/api/jobs/{job['id']}").json()
        if detail["state"] != "running":
            break
        time.sleep(0.1)
    assert detail["state"] == "done"
    assert detail["total"] == 1

    conn = db.connect(settings.db_path)
    assert conn.execute("SELECT COUNT(*) AS n FROM assets").fetchone()["n"] == 1
    conn.close()

    # a second ingest of the same folder re-runs cleanly (files unchanged)
    again = client.post("/api/pipeline/ingest",
                        json={"folder": str(folder), "recursive": True})
    assert again.status_code == 200


def test_stage_endpoint_rejects_unknown_stage(settings: Settings, client: TestClient) -> None:
    r = client.post("/api/pipeline/wat", json={})
    assert r.status_code == 400
