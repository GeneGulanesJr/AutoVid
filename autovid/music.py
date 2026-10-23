"""Music analysis: measurable timing for music-driven editing.

Extracts what the planner needs from a BGM track — beat timestamps, tempo,
an approximate downbeat grid, an RMS energy envelope, and energy-based
sections — using librosa on an ffmpeg-decoded mono waveform. Editorial labels
(intro/build/climax/outro) are inferred later by the planner from the energy
curve; this module stays measurable.

Downbeats are an approximation assuming 4/4: the bar phase is chosen as the
beat offset whose beats carry the most onset strength. Good enough for
placing emphasis; not a substitute for a dedicated downbeat tracker.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import AutoVidError
from .ffmpeg_io.bins import ffmpeg_path

SR = 22050
ENERGY_SECTION_MIN_S = 8.0   # shorter energy sections carry no editing value
ENERGY_CHANGE_SENSITIVITY = 0.35  # fraction of median level change that counts


@dataclass(frozen=True)
class MusicSection:
    start: float
    end: float
    energy_mean: float
    energy_trend: int  # -1 falling, 0 flat, +1 rising vs the previous section


@dataclass(frozen=True)
class MusicAnalysis:
    path: str
    duration_s: float
    tempo_bpm: float
    beats: list[float]
    downbeats: list[float]
    sections: list[MusicSection]
    analyzer: str = field(default="librosa")
    # Decoded mono waveform at SR, kept so per-section mood calls slice memory
    # instead of re-decoding the file with a fresh ffmpeg process each time.
    waveform: np.ndarray | None = field(default=None, compare=False, repr=False)


def _decode_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Any audio/video container → mono float32 waveform at SR via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        cmd = [
            ffmpeg_path(), "-nostdin", "-y", "-v", "error",
            "-i", str(path),
            "-ac", "1", "-ar", str(SR),
            "-f", "wav", str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise AutoVidError(f"audio decode failed: {proc.stderr.strip()[-300:]}")
        import soundfile as sf

        data, sr = sf.read(wav_path, dtype="float32")
        return np.asarray(data, dtype=np.float32), int(sr)
    finally:
        wav_path.unlink(missing_ok=True)


def _energy_sections(
    times: np.ndarray, rms: np.ndarray
) -> list[tuple[float, float, float]]:
    """Split the RMS envelope into sustained-energy sections (start, end, mean).

    Compares the mean level of adjacent ~2–4 s windows; a relative jump of
    ``ENERGY_CHANGE_SENSITIVITY`` × the median level marks a section cut.
    """
    if len(rms) < 8:
        return [(float(times[0]), float(times[-1]), float(np.mean(rms)))]
    win = max(8, len(rms) // 32)
    step = max(1, win // 2)
    levels, centers = [], []
    for i in range(0, len(rms) - win + 1, step):
        levels.append(float(np.mean(rms[i : i + win])))
        centers.append(float(times[i + win // 2]))
    scale = float(np.median(levels)) or 1.0

    cut_times: list[float] = []
    for i in range(1, len(levels)):
        if abs(levels[i] - levels[i - 1]) > ENERGY_CHANGE_SENSITIVITY * scale:
            t = (centers[i - 1] + centers[i]) / 2
            if not cut_times or t - cut_times[-1] > ENERGY_SECTION_MIN_S:
                cut_times.append(t)

    edges = [float(times[0])] + cut_times + [float(times[-1])]
    sections: list[tuple[float, float, float]] = []
    for a, b in zip(edges, edges[1:]):
        mask = (times >= a) & (times < b)
        if not np.any(mask):
            continue
        mean = float(np.mean(rms[mask]))
        if sections and b - a < ENERGY_SECTION_MIN_S:
            # too short to edit against — fold into the previous section
            ps, _, pm = sections[-1]
            sections[-1] = (ps, b, (pm + mean) / 2)
            continue
        sections.append((a, b, mean))
    return sections


def section_audio_b64(
    path: str | Path,
    start: float,
    end: float,
    *,
    max_s: float = 28.0,
    sample_rate: int = 16000,
    waveform: np.ndarray | None = None,
) -> str:
    """[start, end) of an audio file as base64 16 kHz mono int16 PCM.

    Sections longer than ``max_s`` contribute their middle ``max_s`` window
    (the checkpoint cuts audio at 30 s; 28 s leaves transport headroom).
    Feeds the decision layer's audio mood tool.

    ``waveform`` (mono float32 at SR, as carried by :class:`MusicAnalysis`)
    serves the section from memory; without it the span is decoded from the
    file with a fresh ffmpeg process.
    """
    if end - start > max_s:
        mid = (start + end) / 2
        start, end = mid - max_s / 2, mid + max_s / 2
    start = max(start, 0.0)

    if waveform is not None:
        import base64

        seg = np.asarray(waveform[int(start * SR):int(end * SR)], dtype=np.float32)
        if len(seg) == 0:
            raise AutoVidError(f"section audio decoded to zero samples: {path}")
        if sample_rate != SR:
            n_out = max(1, int(round(len(seg) * sample_rate / SR)))
            seg = np.interp(
                np.linspace(0.0, len(seg) - 1.0, n_out), np.arange(len(seg), dtype=np.float64), seg
            )
        pcm = np.clip(seg, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        return base64.b64encode(pcm.tobytes()).decode("ascii")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        cmd = [
            ffmpeg_path(), "-nostdin", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(path),
            "-ac", "1", "-ar", str(sample_rate),
            "-f", "wav", str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise AutoVidError(f"section audio decode failed: {proc.stderr.strip()[-300:]}")
        import base64

        import soundfile as sf

        data, sr = sf.read(wav_path, dtype="int16")
        if len(data) == 0:
            raise AutoVidError(f"section audio decoded to zero samples: {path}")
        return base64.b64encode(np.asarray(data, dtype=np.int16).tobytes()).decode("ascii")
    finally:
        wav_path.unlink(missing_ok=True)


def analyze_music(path: str | Path) -> MusicAnalysis:
    """Full measurable analysis of a music/BGM file."""
    try:
        import librosa
    except ImportError as e:
        raise AutoVidError(
            f"music analysis needs librosa ({e}); install with: pip install librosa soundfile"
        ) from e

    path = Path(path)
    if not path.exists():
        raise AutoVidError(f"music file missing: {path}")
    y, sr = _decode_mono(path)
    if len(y) < sr:  # <1s of audio
        raise AutoVidError(f"music file too short: {path}")

    duration_s = round(len(y) / sr, 3)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr).round(3).tolist()
    tempo_bpm = round(float(np.atleast_1d(tempo)[0]), 2)

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset, sr=sr)

    # downbeat approximation: 4/4 bar phase with the strongest onsets
    downbeats: list[float] = []
    if len(beats) >= 8:
        beat_strengths = []
        for b in beats:
            idx = int(np.searchsorted(onset_times, b))
            beat_strengths.append(float(onset[max(0, min(idx, len(onset) - 1))]))
        best_phase = max(
            range(4),
            key=lambda phase: sum(
                beat_strengths[i] for i in range(phase, len(beats), 4)
            ),
        )
        downbeats = [round(beats[i], 3) for i in range(best_phase, len(beats), 4)]

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=512)
    raw_sections = _energy_sections(rms_times, rms)

    sections: list[MusicSection] = []
    for i, (start, end, mean) in enumerate(raw_sections):
        if i == 0:
            trend = 0
        else:
            prev = raw_sections[i - 1][2]
            trend = 1 if mean > prev * 1.15 else -1 if mean < prev * 0.85 else 0
        sections.append(MusicSection(start=start, end=end, energy_mean=round(mean, 5),
                                     energy_trend=trend))

    import librosa as _lib

    return MusicAnalysis(
        path=str(path),
        duration_s=duration_s,
        tempo_bpm=tempo_bpm,
        beats=beats,
        downbeats=downbeats,
        sections=sections,
        analyzer=f"librosa {_lib.__version__}",
        waveform=y,
    )
