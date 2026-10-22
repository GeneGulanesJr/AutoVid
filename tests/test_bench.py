"""Benchmark harness tests: both modes run isolated and report P/R/F1."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autovid import db, sampling
from autovid.analyze import analyze_asset
from autovid.bench import boundary_prf, run_benchmark
from autovid.config import Settings
from autovid.decisions import DecisionClient
from autovid.ingest import ingest_file
from autovid.vlm import VLMClient, WindowAnalysis
from tests.conftest import make_multiscene_video


def _analysis(**overrides) -> WindowAnalysis:
    base = dict(
        description="scene", activity="none", objects=[], environment="synthetic",
        camera_motion="static", highlight_score=0.8, uncertainty=0.1,
        continues_previous=False,
    )
    base.update(overrides)
    return WindowAnalysis(**base)


@pytest.fixture
def bench_settings(settings: Settings, tmp_path: Path, monkeypatch):
    """Sampled 3-scene asset + heuristic decisions + canned VLM."""
    conn = db.connect(settings.db_path)
    clip = make_multiscene_video(tmp_path / "ms.mp4", scene_s=3.0, scenes=3)
    _, asset_id = ingest_file(conn, settings, clip)
    sampling.sample_asset(conn, settings, asset_id)
    coarse = settings.model_copy(update={"coarse_window_s": 3.0})

    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)
    monkeypatch.setattr(vlm, "describe_window", lambda **kw: _analysis())
    return conn, coarse, asset_id, decisions, vlm


def test_boundary_prf_counts() -> None:
    assert boundary_prf([3.0, 6.0], [3.1, 6.2])["tp"] == 2
    out = boundary_prf([3.0, 5.0], [3.1, 6.0])  # 5.0 matches nothing (tol 0.5)
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 1
    assert out["precision"] == 0.5 and out["recall"] == 0.5 and out["f1"] == 0.5
    assert boundary_prf([], [3.0])["recall"] == 0.0
    assert boundary_prf([], [])["f1"] == 0.0


def test_benchmark_runs_both_modes_and_reports(settings: Settings, bench_settings) -> None:
    conn, settings, asset_id, decisions, vlm = bench_settings
    report = run_benchmark(
        conn, settings, vlm=vlm, decisions=decisions,
        ground_truth={asset_id: [3.0, 6.0]}, name="exp1",
    )

    assert report["baseline"]["mode"] == "baseline"
    assert report["guided"]["mode"] == "guided"
    # baseline never densifies; guided (heuristic) may
    assert report["baseline"]["dense_frames"] == 0
    assert report["guided"]["dense_frames"] >= 0

    for mode in ("baseline", "guided"):
        prf = report[mode]["boundaries"]["prf"]
        assert prf is not None and prf["tp"] >= 1

    # live catalog untouched by the benchmark (both modes ran in the fork)
    assert conn.execute("SELECT COUNT(*) AS n FROM analyses").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"] == 0
    assert conn.execute(
        "SELECT status FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()["status"] == "sampled"

    md = settings.data_dir / "bench" / "exp1" / "report.md"
    report_md = md.read_text(encoding="utf-8")
    assert "baseline" in report_md and "guided" in report_md and "F1" in report_md
    assert (md.with_suffix(".json")).exists()

    fork_dirs = list(settings.data_dir.parent.glob("autovid-bench-*"))
    assert fork_dirs == []  # sandbox cleaned up


def test_benchmark_ground_truth_by_filename(settings: Settings, bench_settings) -> None:
    conn, settings, asset_id, decisions, vlm = bench_settings
    report = run_benchmark(
        conn, settings, vlm=vlm, decisions=decisions,
        ground_truth={"ms.mp4": [3.0, 6.0]}, name="by-name",
    )
    assert report["guided"]["boundaries"]["prf"]["tp"] >= 1


def test_benchmark_requires_sampled_assets(settings: Settings) -> None:
    conn = db.connect(settings.db_path)
    with pytest.raises(ValueError, match="nothing to benchmark"):
        run_benchmark(conn, settings, vlm=VLMClient(settings), decisions=None,
                      ground_truth={})
