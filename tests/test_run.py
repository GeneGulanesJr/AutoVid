"""`autovid run`: one command, resumable, fails fast on a bad VLM."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import autovid.analyze as analyze_mod
import autovid.cli as cli
import autovid.organize as organize_mod
import autovid.refine as refine_mod
import autovid.sampling as sampling
import autovid.transcribe as transcribe
from autovid import db
from autovid.config import get_settings
from autovid.vlm import VLMClient, VLMUnavailable


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A catalog with two ingested assets (one with audio) and every heavy stage faked and recorded."""
    monkeypatch.setenv("AUTOVID_VLM_API_KEY", "k")
    get_settings.cache_clear()
    folder = tmp_path / "My Footage"
    folder.mkdir()
    cat = tmp_path / "cat"
    conn = db.connect(cat / "autovid.db")
    for n, audio in enumerate((1, 0)):
        conn.execute("INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s, has_audio,"
                     " status) VALUES (?,?,?,?,?,5,?, 'ingested')", (f"a{n}", f"s{n}", "p", "p", f"c{n}.mp4", audio))
    conn.commit()
    calls: dict[str, list[str]] = {"sample": [], "transcribe": [], "analyze": [], "ping": [],
                                   "cut": [], "organize": []}

    def set_status(aid, status):
        conn.execute("UPDATE assets SET status = ? WHERE id = ?", (status, aid))
        conn.commit()

    monkeypatch.setattr(cli, "ingest_folder", lambda *a, **k: SimpleNamespace(
        added=[], skipped=[], duplicates=[], failed=[]))
    monkeypatch.setattr(sampling, "sample_asset", lambda c, s, aid: (calls["sample"].append(aid), set_status(aid, "sampled")))
    monkeypatch.setattr(transcribe, "load_model", lambda s: None)

    def fake_transcribe(c, s, aid, model=None):
        calls["transcribe"].append(aid)
        db.job_start(c, aid, "transcribe")
        db.job_finish(c, aid, "transcribe")
        return 1

    monkeypatch.setattr(transcribe, "transcribe_asset", fake_transcribe)
    monkeypatch.setattr(analyze_mod, "analyze_asset", lambda c, s, aid, **k: (calls["analyze"].append(aid), set_status(aid, "analyzed")))

    def fake_cut(c, s, aid, *, decisions=None):
        calls["cut"].append(aid)
        set_status(aid, "cut")

    def fake_organize(c, s, aid):
        calls["organize"].append(aid)
        set_status(aid, "organized")

    monkeypatch.setattr(refine_mod, "cut_asset", fake_cut)
    monkeypatch.setattr(organize_mod, "organize_asset", fake_organize)
    monkeypatch.setattr(VLMClient, "ping", lambda self: calls["ping"].append("ping"))
    yield folder, cat, calls
    get_settings.cache_clear()


def test_run_does_every_stage_once_and_a_second_run_resumes_with_nothing_to_do(env):
    folder, cat, calls = env
    first = CliRunner().invoke(cli.app, ["run", str(folder), "--data-dir", str(cat)])
    assert first.exit_code == 0, first.output
    assert calls["sample"] == ["a0", "a1"] and calls["transcribe"] == ["a0"]        # only a0 has audio
    assert calls["analyze"] == ["a0", "a1"] and calls["ping"] == ["ping"]
    assert "Add --objective" in first.output

    for k in ("sample", "transcribe", "analyze"):
        calls[k].clear()
    second = CliRunner().invoke(cli.app, ["run", str(folder), "--data-dir", str(cat)])
    assert second.exit_code == 0 and not (calls["sample"] or calls["transcribe"] or calls["analyze"])

    CliRunner().invoke(cli.app, ["run", str(folder), "--data-dir", str(cat), "--force"])
    assert calls["transcribe"] == ["a0"] and calls["analyze"] == ["a0", "a1"]       # --force redoes both


def test_run_catalog_also_cuts_and_organizes_then_resumes_with_nothing_to_do(env):
    folder, cat, calls = env
    result = CliRunner().invoke(cli.app, ["run", str(folder), "--data-dir", str(cat), "--catalog"])
    assert result.exit_code == 0, result.output
    assert calls["cut"] == ["a0", "a1"] and calls["organize"] == ["a0", "a1"]       # whole catalog, in order
    assert "cut: 2 asset(s)" in result.output and "organize: 2 asset(s)" in result.output
    assert "Add --objective" in result.output                                       # --catalog runs regardless of --objective

    for k in ("cut", "organize"):
        calls[k].clear()
    second = CliRunner().invoke(cli.app, ["run", str(folder), "--data-dir", str(cat), "--catalog"])
    assert second.exit_code == 0 and not (calls["cut"] or calls["organize"])        # resumed: already organized


def test_run_without_catalog_never_cuts_or_organizes(env):
    folder, cat, calls = env
    result = CliRunner().invoke(cli.app, ["run", str(folder), "--data-dir", str(cat)])
    assert result.exit_code == 0, result.output
    assert not (calls["cut"] or calls["organize"])


def test_run_stops_before_any_slow_stage_when_the_vlm_check_fails(env, monkeypatch):
    folder, cat, calls = env

    def refuse(self):
        raise VLMUnavailable("VLM check failed: 400 — `temperature` is deprecated for this model.")

    monkeypatch.setattr(VLMClient, "ping", refuse)
    result = CliRunner().invoke(cli.app, ["run", str(folder), "--data-dir", str(cat)])
    assert result.exit_code == 1 and "temperature" in result.output
    assert not (calls["sample"] or calls["transcribe"] or calls["analyze"])


def test_run_defaults_the_catalog_to_a_folder_named_subdirectory(env, monkeypatch, tmp_path):
    folder, _, _ = env
    monkeypatch.setenv("AUTOVID_DATA_DIR", str(tmp_path / "root"))
    get_settings.cache_clear()
    result = CliRunner().invoke(cli.app, ["run", str(folder)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "root" / "my-footage" / "autovid.db").exists()
