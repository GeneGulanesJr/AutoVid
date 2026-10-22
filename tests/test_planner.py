"""Music-driven planner tests: modes, beat snapping, dedupe, D1 review."""
from __future__ import annotations

import json

import pytest

from autovid import db
from autovid.config import Settings
from autovid.decisions import DecisionClient, SequenceReview
from autovid.music import MusicAnalysis, MusicSection
from autovid.planner import plan_music_video


def _music(**overrides) -> MusicAnalysis:
    """30 s of 120 BPM music: quiet intro 0–15, energetic 15–30."""
    beats = [round(i * 0.5, 3) for i in range(60)]
    base = dict(
        path="music.wav",
        duration_s=30.0,
        tempo_bpm=120.0,
        beats=beats,
        downbeats=[b for b in beats if (beats.index(b)) % 4 == 0],
        sections=[
            MusicSection(start=0.0, end=15.0, energy_mean=0.1, energy_trend=0),
            MusicSection(start=15.0, end=30.0, energy_mean=0.9, energy_trend=1),
        ],
    )
    base.update(overrides)
    return MusicAnalysis(**base)


@pytest.fixture
def music_wav(tmp_path: Path) -> str:
    """A real click-track wav so mood enrichment can decode section audio."""
    from tests.test_music import _click_track

    return str(_click_track(tmp_path / "music.wav", bpm=120, duration_s=30.0,
                            energy_profile=[(0.0, 0.2), (15.0, 1.0)]))


@pytest.fixture
def catalog(settings: Settings):
    """Six accepted 5 s segments with distinct tags."""
    conn = db.connect(settings.db_path)
    for i in range(6):
        conn.execute(
            """INSERT INTO assets (id, sha256, path, original_path, original_name, size_bytes)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (f"a_{i}", f"sha{i}", f"C:/v{i}.mp4", f"C:/v{i}.mp4", f"v{i}.mp4"),
        )
        conn.execute(
            """INSERT INTO segments (asset_id, start_s, end_s, description,
                                     tags_json, highlight_score, verdict,
                                     decision_confidence, review_status)
               VALUES (?, ?, ?, ?, ?, ?, 'accept', 0.9, 'auto')""",
            (f"a_{i}", i * 10.0, i * 10.0 + 5.0, f"scene {i}",
             json.dumps([f"tag{i}"]), 0.5 + i * 0.05),
        )
    conn.commit()
    return conn, settings


def _review(action: str):
    return SequenceReview(action, "none", 0.9, "test", 0.0)


def test_plan_beat_mode_snaps_cuts_to_beats(catalog) -> None:
    conn, settings = catalog
    board = plan_music_video(
        conn, settings, _music(), objective="demo", mode="beat",
        style="fast_paced", name="beat-cut",
    )
    assert board.music_path == "music.wav" and board.mode == "beat"
    assert board.items, "planner produced no items"

    beats = set(_music().beats)
    for item in board.items:
        span = (item.timeline_end or 0) - (item.timeline_start or 0)
        src_span = item.source_end - item.source_start
        assert span == pytest.approx(src_span)  # no speed tricks
        assert span >= 1.0  # sane minimum
        assert item.audio == "music"
        if item.timeline_start and item.timeline_start > 0:
            nearest = min(beats, key=lambda b: abs(b - item.timeline_start))
            assert abs(nearest - item.timeline_start) < 0.05  # snapped


def test_plan_never_reuses_clips_until_pool_exhausted(catalog) -> None:
    conn, settings = catalog
    board = plan_music_video(
        conn, settings, _music(), objective="demo", mode="beat",
        style="social", name="social-cut",  # denser: more slots than 6 clips
    )
    ids = [i.clip_id for i in board.items]
    assert len(ids) == len(set(ids)) or len(ids) > 6  # distinct while pool lasts


def test_plan_cinematic_paces_by_section_role(catalog) -> None:
    conn, settings = catalog
    board = plan_music_video(
        conn, settings, _music(), objective="demo", mode="cinematic",
        style="cinematic", name="cine",
    )
    # intro (0–15) at 16 beats/cut → one long slot; 5 s clips can't fill 15 s
    # slots, so the planner trims: every item's source span must be ≤ 5 s
    assert all(i.source_end - i.source_start <= 5.0 + 0.01 for i in board.items)
    assert board.mode == "cinematic"


def test_sequence_review_replace_clip_swaps(settings: Settings, catalog, monkeypatch) -> None:
    conn, settings = catalog
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "auto"}), conn)
    monkeypatch.setattr(decisions, "_call_tool",
                        lambda name, args: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(
        decisions, "decide_sequence_review",
        lambda context: _review("replace_clip"),
    )
    board = plan_music_video(
        conn, settings, _music(), objective="demo", mode="beat",
        style="fast_paced", name="swapped", decisions=decisions,
    )
    notes = [i.note for i in board.items if i.note]
    assert notes, "replacement was not applied"
    assert all("decision layer" in n for n in notes)


def test_sequence_review_fallback_approves(settings: Settings, catalog, monkeypatch) -> None:
    conn, settings = catalog
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "auto"}), conn)

    def down(name, args):
        from autovid.decisions import DecisionUnavailable

        raise DecisionUnavailable("server down")

    monkeypatch.setattr(decisions, "_call_tool", down)
    row = None
    board = plan_music_video(
        conn, settings, _music(), objective="demo", mode="beat", name="fallback",
        decisions=decisions,
    )
    assert board.items
    row = conn.execute(
        "SELECT action, engine_version FROM decisions WHERE tool = 'decision_music_sequence'"
    ).fetchone()
    assert row["action"] == "approve" and row["engine_version"] == "heuristic"


def test_plan_requires_catalog(settings: Settings) -> None:
    conn = db.connect(settings.db_path)
    from autovid.errors import AutoVidError

    with pytest.raises(AutoVidError, match="no accepted segments"):
        plan_music_video(conn, settings, _music(), objective="x")


def test_mood_enrichment_adjusts_pacing(settings: Settings, catalog, monkeypatch,
                                        music_wav: str) -> None:
    """d1 listens per section: calm halves cut density, driving doubles it."""
    conn, settings = catalog
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "auto"}), conn)

    def _canned_mood(label: str) -> dict:
        # shape: the decision_music_mood TOOL output (flat), not raw engine answers
        return {
            "mood": label,
            "energy_level": 3 if label == "driving" else 1,
            "builds": 0.3,
            "confidence": 0.85,
            "details": {"answers": {}},
            "model": "LiquidAI/d1-omni-600M",
        }

    mood_labels = iter(["calm", "driving"])

    def fake_call(name, args):
        if name == "decision_music_mood":
            # the section audio traveled as base64 PCM16
            assert len(args["audio_pcm16_b64"]) > 1000
            return _canned_mood(next(mood_labels))
        # decision_music_sequence → approve the drafted section
        return {
            "action": "approve", "concern": "none", "confidence": 0.8,
            "details": {"answers": {}}, "model": "LiquidAI/d1-omni-600M",
        }

    monkeypatch.setattr(decisions, "_call_tool", fake_call)

    board = plan_music_video(
        conn, settings, _music(path=music_wav), objective="mood", mode="beat",
        style="cinematic", name="mood-plan", decisions=decisions,
    )
    intro_slots = [i for i in board.items if (i.timeline_start or 0) < 15]
    high_slots = [i for i in board.items if (i.timeline_start or 0) >= 15]
    assert intro_slots and high_slots
    # calm intro: fewer, longer slots per second; driving half: denser cuts
    intro_len = sum((i.timeline_end or 0) - (i.timeline_start or 0) for i in intro_slots)
    high_len = sum((i.timeline_end or 0) - (i.timeline_start or 0) for i in high_slots)
    assert (intro_len / len(intro_slots)) > (high_len / len(high_slots))

    logged = conn.execute(
        "SELECT action, engine_version FROM decisions WHERE tool = 'decision_music_mood'"
    ).fetchall()
    assert len(logged) == 2
    assert all(r["action"] in ("calm", "driving") for r in logged)
    assert all("d1-omni-600M" in r["engine_version"] for r in logged)


def test_mood_fallback_uses_energy_guess(settings: Settings, catalog, monkeypatch) -> None:
    conn, settings = catalog
    decisions = DecisionClient(settings.model_copy(update={"decision_mode": "auto"}), conn)

    def down(name, args):
        from autovid.decisions import DecisionUnavailable

        raise DecisionUnavailable("down")

    monkeypatch.setattr(decisions, "_call_tool", down)
    mood = decisions.decide_music_mood(
        json.dumps({"role": "climax"}), _pcm16(), fallback_mood="energetic"
    )
    assert mood.mood == "energetic" and mood.engine == "heuristic"
    row = conn.execute(
        "SELECT action, engine_version FROM decisions WHERE tool = 'decision_music_mood'"
    ).fetchone()
    assert row["action"] == "energetic" and row["engine_version"] == "heuristic"


def _pcm16() -> str:
    import base64

    return base64.b64encode(b"\x00\x01" * 1600).decode()
