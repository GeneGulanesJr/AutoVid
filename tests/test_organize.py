"""Organization tests: tagging, suggested names, search, recoverable renames."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autovid import db, sampling
from autovid.analyze import analyze_asset
from autovid.config import Settings
from autovid.decisions import (
    BoundaryVerdict,
    DecisionClient,
    SegmentVerdict,
)
from autovid.organize import (
    apply_rename,
    build_quality_flags,
    build_tags,
    organize_asset,
    search,
    slugify,
    suggested_name,
    undo_rename,
)
from autovid.refine import cut_asset
from autovid.vlm import VLMClient, WindowAnalysis
from tests.conftest import make_multiscene_video


def _analysis(**overrides) -> WindowAnalysis:
    base = dict(
        description="A person walks through an outdoor market",
        activity="walking",
        objects=["stalls", "umbrellas"],
        environment="market",
        camera_motion="handheld",
        highlight_score=0.8,
        uncertainty=0.1,
        continues_previous=False,
    )
    base.update(overrides)
    return WindowAnalysis(**base)


@pytest.fixture
def cut(settings: Settings, tmp_path: Path, monkeypatch):
    """Cut 3-scene clip: 3 accepted segments with extracted clips."""
    conn = db.connect(settings.db_path)
    clip = make_multiscene_video(tmp_path / "ms.mp4", scene_s=3.0, scenes=3)
    from autovid.ingest import ingest_file

    _, asset_id = ingest_file(conn, settings, clip)
    sampling.sample_asset(conn, settings, asset_id)
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "heuristic"}), conn)
    vlm = VLMClient(settings)
    monkeypatch.setattr(vlm, "describe_window", lambda **kw: _analysis())
    coarse = settings.model_copy(update={"coarse_window_s": 3.0})
    analyze_asset(conn, coarse, asset_id, vlm=vlm, decisions=decisions)
    monkeypatch.setattr(
        decisions,
        "decide_boundary",
        lambda a, c, b, f: BoundaryVerdict("boundary", "at", 0.9, 0.9, "test", 0.0),
    )
    monkeypatch.setattr(
        decisions,
        "decide_segment",
        lambda a, s, frames, duration_s=None, min_duration_s=0.5: SegmentVerdict(
            "accept", 0.9, 0.9, 2, 0.9, "test", 0.0
        ),
    )
    cut_asset(conn, coarse, asset_id, decisions=decisions)
    return conn, coarse, asset_id


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_slugify_strips_and_limits() -> None:
    assert slugify("A Person walks, talks! Ünïcode…") == "a-person-walks-talks-unicode"
    assert slugify("one two three four five six seven eight") == (
        "one-two-three-four-five-six"
    )
    assert slugify("???") == "clip"


def test_build_tags_merges_analysis_fields() -> None:
    tags = build_tags({
        "activity": "Walking",
        "environment": "Market",
        "objects": ["stalls", " Umbrellas ", "stalls"],
        "camera_motion": "handheld",
    })
    assert tags == ["walking", "market", "stalls", "umbrellas", "cam-handheld"]


def test_build_quality_flags() -> None:
    flags = build_quality_flags("accept", 0.9, 2, 0.1)
    assert flags == {"quality_score": pytest.approx(2 / 3)}
    flagged = build_quality_flags("needs_evidence", 0.3, 1, 0.7)
    assert flagged["needs_review"] and flagged["low_decision_confidence"]
    assert flagged["weak_quality"] and flagged["uncertain_understanding"]


def test_suggested_name_keeps_provenance() -> None:
    name = suggested_name("a_abcdef1234567890", 3.0, 6.0, "Market walk", ["market"])
    assert name == "market-walk__abcdef12__3.000-6.000.mp4"


# ---------------------------------------------------------------------------
# organize stage
# ---------------------------------------------------------------------------


def test_organize_tags_segments_and_updates_status(settings: Settings, cut) -> None:
    conn, settings, asset_id = cut
    result = organize_asset(conn, settings, asset_id)
    assert result.segments_tagged == 3

    rows = conn.execute(
        "SELECT * FROM segments WHERE asset_id = ?", (asset_id,)
    ).fetchall()
    for r in rows:
        tags = json.loads(r["tags_json"])
        assert "walking" in tags and "market" in tags
        quality = json.loads(r["quality_json"])
        assert quality["quality_score"] == pytest.approx(2 / 3)
        assert "needs_review" not in quality
        assert r["suggested_name"].startswith("a-person-walks-through-an-outdoor__")
        assert f"{r['start_s']:.3f}-{r['end_s']:.3f}.mp4" in r["suggested_name"]

    assert conn.execute(
        "SELECT status FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()["status"] == "organized"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_finds_by_description_and_tags(settings: Settings, cut) -> None:
    conn, settings, asset_id = cut
    organize_asset(conn, settings, asset_id)

    hits = search(conn, "market")
    assert len(hits) == 3
    assert all("market" in json.loads(h["tags_json"]) for h in hits)

    assert search(conn, "nonexistent-thing") == []


# ---------------------------------------------------------------------------
# renames
# ---------------------------------------------------------------------------


def test_rename_and_undo_roundtrip(settings: Settings, cut) -> None:
    conn, settings, asset_id = cut
    organize_asset(conn, settings, asset_id)
    seg = conn.execute(
        "SELECT id, clip_path, start_s, end_s FROM segments WHERE asset_id = ? "
        "ORDER BY start_s LIMIT 1",
        (asset_id,),
    ).fetchone()
    old = Path(seg["clip_path"])
    assert old.exists()

    rename_id = apply_rename(conn, seg["id"])
    new = Path(conn.execute(
        "SELECT clip_path FROM segments WHERE id = ?", (seg["id"],)
    ).fetchone()["clip_path"])
    assert new.exists() and not old.exists()
    # provenance stays embedded in the suggested name
    assert f"{seg['start_s']:.3f}-{seg['end_s']:.3f}.mp4" in new.name

    undo_rename(conn, rename_id)
    restored = Path(conn.execute(
        "SELECT clip_path FROM segments WHERE id = ?", (seg["id"],)
    ).fetchone()["clip_path"])
    assert restored == old and restored.exists()

    with pytest.raises(Exception, match="already reverted"):
        undo_rename(conn, rename_id)


def test_rename_requires_organized_segment(settings: Settings, cut) -> None:
    conn, settings, asset_id = cut
    seg = conn.execute(
        "SELECT id FROM segments WHERE asset_id = ? LIMIT 1", (asset_id,)
    ).fetchone()
    with pytest.raises(Exception, match="no suggested name"):
        apply_rename(conn, seg["id"])
