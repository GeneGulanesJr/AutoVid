"""Ingest pipeline tests: discovery, dedupe, metadata, thumbnails, resume."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from autovid import db, ingest
from autovid.config import Settings


def test_scan_finds_only_videos_recursively(video_folder: Path) -> None:
    found = ingest.scan_folder(video_folder)
    names = [p.name for p in found]
    assert names == ["clip_a.mp4", "clip_b.mp4", "clip_c.mov"]


def test_scan_non_recursive_skips_subfolders(video_folder: Path) -> None:
    found = ingest.scan_folder(video_folder, recursive=False)
    assert [p.name for p in found] == ["clip_a.mp4", "clip_b.mp4"]


def test_scan_rejects_missing_folder(tmp_path: Path) -> None:
    import pytest

    from autovid.errors import AutoVidError

    with pytest.raises(AutoVidError):
        ingest.scan_folder(tmp_path / "nope")


def test_ingest_folder_populates_catalog(settings: Settings, video_folder: Path) -> None:
    conn = db.connect(settings.db_path)
    report = ingest.ingest_folder(conn, settings, video_folder)

    assert len(report.added) == 3
    assert report.failed == []
    rows = conn.execute("SELECT * FROM assets ORDER BY duration_s").fetchall()
    assert len(rows) == 3

    for row in rows:
        assert row["id"].startswith("a_")
        assert row["duration_s"] > 0
        assert row["width"] == 160
        assert row["height"] == 120
        assert row["fps"] == 12.0
        assert json.loads(row["ffprobe_json"])["format"]["format_name"]
        assert Path(row["path"]).exists()

    audio_row = conn.execute(
        "SELECT has_audio FROM assets WHERE original_name = 'clip_b.mp4'"
    ).fetchone()
    assert audio_row["has_audio"] == 1


def test_ingest_generates_thumbnails(settings: Settings, video_folder: Path) -> None:
    conn = db.connect(settings.db_path)
    ingest.ingest_folder(conn, settings, video_folder)

    row = conn.execute("SELECT * FROM assets WHERE original_name = 'clip_b.mp4'").fetchone()
    thumbs = conn.execute(
        "SELECT * FROM frames WHERE asset_id = ? AND kind = 'thumb' ORDER BY ts", (row["id"],)
    ).fetchall()
    # 6-second clip at default 30s thumb interval → 1 thumb at t=0.
    assert len(thumbs) == 1
    assert thumbs[0]["ts"] == 0.0
    assert Path(thumbs[0]["path"]).exists()
    assert row["poster_path"] is not None

    settings_tight = settings.model_copy(update={"thumb_interval_s": 2.0})
    ingest.ensure_thumbs(conn, settings_tight, row["id"], Path(row["path"]))
    tight_thumbs = conn.execute(
        "SELECT COUNT(*) AS n FROM frames WHERE asset_id = ? AND kind = 'thumb'", (row["id"],)
    ).fetchone()
    assert tight_thumbs["n"] == 3  # t = 0, 2, 4


def test_ingest_detects_duplicates_by_content(settings: Settings, video_folder: Path) -> None:
    conn = db.connect(settings.db_path)
    ingest.ingest_folder(conn, settings, video_folder)

    copy_path = video_folder.parent / "same_content_elsewhere.mp4"
    shutil.copy2(video_folder / "clip_a.mp4", copy_path)
    report = ingest.ingest_folder(conn, settings, video_folder.parent, recursive=False)

    assert len(report.duplicates) == 1
    asset = conn.execute(
        "SELECT * FROM assets WHERE original_name = 'clip_a.mp4'"
    ).fetchone()
    dupes = json.loads(asset["dup_paths_json"])
    assert str(copy_path) in dupes


def test_ingest_is_idempotent(settings: Settings, video_folder: Path) -> None:
    conn = db.connect(settings.db_path)
    first = ingest.ingest_folder(conn, settings, video_folder)
    second = ingest.ingest_folder(conn, settings, video_folder)

    assert len(first.added) == 3
    assert len(second.added) == 0
    assert len(second.skipped) == 3
    assert conn.execute("SELECT COUNT(*) AS n FROM assets").fetchone()["n"] == 3


def test_job_ledger_tracks_stage_state(settings: Settings, video_folder: Path) -> None:
    conn = db.connect(settings.db_path)
    ingest.ingest_folder(conn, settings, video_folder)
    rows = conn.execute("SELECT * FROM jobs WHERE stage = 'thumbs'").fetchall()
    assert len(rows) == 3
    assert all(r["status"] == "done" for r in rows)

    asset_id = rows[0]["asset_id"]
    assert db.job_done(conn, asset_id, "thumbs")
    assert not db.job_done(conn, asset_id, "nonexistent-stage")

    db.job_finish(conn, asset_id, "thumbs", error="boom")
    assert not db.job_done(conn, asset_id, "thumbs")
    row = conn.execute(
        "SELECT status, detail FROM jobs WHERE asset_id = ? AND stage = 'thumbs'",
        (asset_id,),
    ).fetchone()
    assert row["status"] == "error" and "boom" in row["detail"]
