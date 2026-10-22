"""Scene detection + baseline sampling tests (real ffmpeg, synthetic clips)."""
from __future__ import annotations

from pathlib import Path

import pytest

from autovid import db, sampling
from autovid.config import Settings
from autovid.ffmpeg_io.scenes import detect_scene_changes
from tests.conftest import make_multiscene_video, make_video


@pytest.fixture
def multiscene(tmp_path: Path) -> Path:
    return make_multiscene_video(tmp_path / "multiscene.mp4", scene_s=3.0, scenes=3)


def test_detect_scene_changes_finds_hard_cuts(multiscene: Path) -> None:
    changes = detect_scene_changes(multiscene, threshold=0.3)
    ts = [c.ts for c in changes]
    assert len(ts) == 2
    assert abs(ts[0] - 3.0) < 0.3
    assert abs(ts[1] - 6.0) < 0.3
    assert all(0.3 <= c.score <= 1.0 for c in changes)


def test_detect_scene_changes_higher_threshold_filters(multiscene: Path) -> None:
    # threshold above every observed score → no detections
    assert detect_scene_changes(multiscene, threshold=0.99) == []


def test_detect_scene_changes_missing_file(tmp_path: Path) -> None:
    from autovid.errors import AutoVidError

    with pytest.raises(AutoVidError):
        detect_scene_changes(tmp_path / "nope.mp4")


def _ingested_asset(settings: Settings, source: Path) -> str:
    from autovid.ingest import ingest_file

    conn = db.connect(settings.db_path)
    _, asset_id = ingest_file(conn, settings, source)
    return asset_id


def test_sample_asset_populates_frames_and_scenes(settings: Settings, multiscene: Path) -> None:
    conn = db.connect(settings.db_path)
    asset_id = _ingested_asset(settings, multiscene)

    grid_settings = settings.model_copy(update={"grid_interval_s": 3.0})
    result = sampling.sample_asset(conn, grid_settings, asset_id)
    assert result.scene_changes == 2
    # 9 s clip, 3 s grid interval → frames at t = 0, 3, 6 (fps-equivalent grid)
    assert result.grid_frames == 3

    scene_ts = [r["start_s"] for r in conn.execute(
        "SELECT start_s FROM scenes WHERE asset_id = ? ORDER BY start_s", (asset_id,)
    ).fetchall()]
    assert abs(scene_ts[0] - 3.0) < 0.3 and abs(scene_ts[1] - 6.0) < 0.3

    frames = conn.execute(
        "SELECT * FROM frames WHERE asset_id = ? AND kind = 'grid' ORDER BY ts",
        (asset_id,),
    ).fetchall()
    assert [f["ts"] for f in frames] == [0.0, 3.0, 6.0]
    assert frames[0]["motion_score"] == 0.0
    # the frame right after the t=3 hard cut must show high motion
    assert frames[1]["motion_score"] > 0.2
    for f in frames:
        assert Path(f["path"]).exists()

    row = conn.execute("SELECT status FROM assets WHERE id = ?", (asset_id,)).fetchone()
    assert row["status"] == "sampled"
    assert db.job_done(conn, asset_id, "sampling")


def test_sample_asset_resampling_replaces_rows(settings: Settings, multiscene: Path) -> None:
    conn = db.connect(settings.db_path)
    asset_id = _ingested_asset(settings, multiscene)
    sampling.sample_asset(conn, settings, asset_id)
    tight = settings.model_copy(update={"grid_interval_s": 1.5})
    result = sampling.sample_asset(conn, tight, asset_id)

    assert result.grid_frames == 6  # 0, 1.5, 3, 4.5, 6, 7.5
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM frames WHERE asset_id = ? AND kind = 'grid'",
        (asset_id,),
    ).fetchone()
    assert count["n"] == 6


def test_sample_asset_missing_source_is_recorded(settings: Settings, tmp_path: Path) -> None:
    conn = db.connect(settings.db_path)
    clip = make_video(tmp_path / "gone.mp4", duration=2)
    asset_id = _ingested_asset(settings, clip)
    clip.unlink()

    with pytest.raises(Exception, match="missing"):
        sampling.sample_asset(conn, settings, asset_id)
    row = conn.execute("SELECT status, error FROM assets WHERE id = ?", (asset_id,)).fetchone()
    assert row["status"] == "error"
    assert "sampling" in row["error"]
    assert not db.job_done(conn, asset_id, "sampling")


def test_pending_assets_orders_by_created(settings: Settings, video_folder: Path) -> None:
    from autovid.ingest import ingest_folder

    conn = db.connect(settings.db_path)
    ingest_folder(conn, settings, video_folder)
    pending = sampling.pending_assets(conn)
    assert len(pending) == 3
    assert all(row["status"] == "ingested" for row in pending)

    sampling.sample_asset(conn, settings, pending[0]["id"])
    remaining = sampling.pending_assets(conn)
    assert len(remaining) == 2
