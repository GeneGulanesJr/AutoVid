"""Music analysis tests against synthetic audio with known timing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autovid.music import analyze_music

SR = 22050


def _click_track(path: Path, *, bpm: float, duration_s: float,
                 energy_profile: list[tuple[float, float]] | None = None) -> Path:
    """A metronome at ``bpm`` with optional (start, level) energy changes."""
    total = int(duration_s * SR)
    t = np.arange(total) / SR
    audio = np.zeros(total, dtype=np.float32)
    beat_len = 60.0 / bpm
    n_beats = int(duration_s / beat_len)
    for i in range(n_beats):
        start = int(i * beat_len * SR)
        width = int(0.03 * SR)  # 30 ms click
        level = 0.5
        if energy_profile:
            elapsed = i * beat_len
            for seg_start, seg_level in energy_profile:
                if elapsed >= seg_start:
                    level = seg_level
        audio[start : start + width] = (np.sin(
            2 * np.pi * 1000 * t[:width]) * level).astype(np.float32)
    # fill the level into the gaps too so the RMS envelope follows the profile
    if energy_profile:
        env = np.full(total, 0.02, dtype=np.float32)
        for seg_start, seg_level in energy_profile:
            env[int(seg_start * SR):] = np.maximum(env[int(seg_start * SR):], seg_level * 0.3)
        audio = audio + env * np.sin(2 * np.pi * 220 * t)
    sf.write(path, audio, SR, subtype="FLOAT")
    return path


def test_tempo_and_beats_detected_on_click_track(tmp_path: Path) -> None:
    clip = _click_track(tmp_path / "click120.wav", bpm=120, duration_s=20.0)
    out = analyze_music(clip)

    assert out.tempo_bpm == pytest.approx(120, abs=3)
    assert len(out.beats) >= 35  # ~40 beats in 20 s at 120 BPM
    gaps = np.diff(out.beats)
    assert np.mean(gaps) == pytest.approx(0.5, abs=0.05)
    # beats stay in mono direction
    assert out.beats == sorted(out.beats)


def test_downbeats_approximate_bar_grid(tmp_path: Path) -> None:
    clip = _click_track(tmp_path / "click90.wav", bpm=90, duration_s=24.0)
    out = analyze_music(clip)
    assert len(out.downbeats) >= 5
    gaps = np.diff(out.downbeats)
    # ~4 beats per bar at 90 BPM → ~2.67 s per bar
    assert np.mean(gaps) == pytest.approx(4 * 60 / 90, abs=0.3)


def test_energy_sections_detect_level_change(tmp_path: Path) -> None:
    clip = _click_track(tmp_path / "dynamics.wav", bpm=120, duration_s=30.0,
                        energy_profile=[(0.0, 0.2), (15.0, 1.0)])
    out = analyze_music(clip)
    assert len(out.sections) >= 2
    # some section boundary lands near the 15 s dynamics change
    starts = [s.start for s in out.sections[1:]]
    assert any(abs(s - 15.0) < 3.0 for s in starts), starts
    # the louder later section carries more energy
    assert out.sections[-1].energy_mean > out.sections[0].energy_mean


def test_analyze_missing_file_raises(tmp_path: Path) -> None:
    from autovid.errors import AutoVidError

    with pytest.raises(AutoVidError, match="missing"):
        analyze_music(tmp_path / "nope.mp3")


def test_analyze_reports_analyzer_and_duration(tmp_path: Path) -> None:
    clip = _click_track(tmp_path / "short.wav", bpm=120, duration_s=12.0)
    out = analyze_music(clip)
    assert "librosa" in out.analyzer
    assert out.duration_s == pytest.approx(12.0, abs=0.2)
    assert Path(out.path).exists()
