"""Speech-to-text stage: asset audio → timestamped utterances in ``transcripts``.

Two backends, same output:

- ``whispercpp``: whisper.cpp with the Vulkan backend, runs on AMD (and Intel/NVIDIA)
  GPUs. Needs the Windows build + a ggml model under ``whispercpp_dir`` (see README).
- ``faster-whisper``: ctranslate2, CPU or NVIDIA CUDA only.

``auto`` picks whisper.cpp when its executable is present. Both drop silence
(VAD) and Whisper's usual junk, which otherwise shows up as hallucinated text.
d1-omni cannot do this: it only classifies audio into fixed labels
(see ``decide_music_mood``).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from . import db
from .config import Settings
from .errors import AutoVidError
from .ffmpeg_io.bins import ffmpeg_path
from .ingest import get_asset

logger = logging.getLogger(__name__)

Utterance = tuple[float, float, str]  # start_s, end_s, text


def _cli(settings: Settings) -> Path:
    return settings.whispercpp_dir / "bin" / "whisper-cli.exe"


def backend(settings: Settings) -> str:
    if settings.whisper_backend != "auto":
        return settings.whisper_backend
    return "whispercpp" if _cli(settings).exists() else "faster-whisper"


def load_model(settings: Settings):
    """The reusable model object; whisper.cpp is a subprocess per file, so it has none."""
    if backend(settings) == "whispercpp":
        return None
    from faster_whisper import WhisperModel  # heavy import, only when transcribing

    return WhisperModel(settings.whisper_model, device=settings.whisper_device,
                        compute_type=settings.whisper_compute_type)


def _confident(seg) -> bool:
    """Drop Whisper's usual junk: low-confidence guesses, "tik-tik-tik" loops, non-speech."""
    return seg.avg_logprob > -1.0 and seg.compression_ratio < 2.4 and seg.no_speech_prob < 0.6


def _run_faster_whisper(model, source: Path) -> tuple[list[Utterance], str]:
    # condition_on_previous_text=False stops one bad line from poisoning the next ones.
    segments, info = model.transcribe(str(source), vad_filter=True, condition_on_previous_text=False)
    return [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip() and _confident(s)], \
        info.language


def _run_whispercpp(settings: Settings, source: Path) -> tuple[list[Utterance], str]:
    root = settings.whispercpp_dir
    model = root / "models" / f"ggml-{settings.whisper_model}.bin"
    vad = root / "models" / "ggml-silero-v5.1.2.bin"
    for p in (_cli(settings), model, vad):
        if not p.exists():
            raise AutoVidError(f"whisper.cpp backend missing {p}")
    with tempfile.TemporaryDirectory() as tmp:
        wav, out = Path(tmp) / "audio.wav", Path(tmp) / "out"
        subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-i", str(source), "-vn",
                        "-ac", "1", "-ar", "16000", str(wav)], check=True, timeout=1800)
        # -mc 0 = no cross-segment context (same reason as condition_on_previous_text=False).
        # -vsd/-vmsd/-ml/-sow keep segments sentence-sized: the default VAD merges speech across
        # pauses into 10 s+ spans, which would make the script's cuts land on dead air.
        proc = subprocess.run(
            [str(_cli(settings)), "-m", str(model), "-f", str(wav), "-l", "auto", "-mc", "0",
             "--vad", "-vm", str(vad), "-vsd", "300", "-vmsd", "6", "-ml", "70", "-sow",
             "-oj", "-of", str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        if proc.returncode != 0:
            raise AutoVidError(f"whisper-cli failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
        if "ggml_vulkan: Found" not in proc.stderr:  # seen in the wild: "GPU acceleration on" but CPU
            logger.warning("whisper.cpp did not report a Vulkan device — this ran on the CPU")
        data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    rows = []
    for t in data["transcription"]:
        text = t["text"].strip()
        if not text:
            continue
        start, end = t["offsets"]["from"] / 1000, t["offsets"]["to"] / 1000
        # shortcut: whisper.cpp stretches a segment's end over trailing silence; cap it by a slow
        # speaking rate (9 chars/s). Upgrade: word-level timestamps (needs -dtw alignment heads).
        rows.append((start, min(end, start + len(text) / 9 + 0.5), text))
    return rows, data.get("result", {}).get("language")


def transcribe_asset(conn: sqlite3.Connection, settings: Settings, asset_id: str, *, model=None) -> int:
    """Transcribe one asset (replacing any earlier transcript); returns the utterance count."""
    asset = get_asset(conn, asset_id)
    if asset is None:
        raise AutoVidError(f"Unknown asset {asset_id}")
    source = Path(asset["path"])
    if not source.exists():
        raise AutoVidError(f"Source file missing: {source}")

    db.job_start(conn, asset_id, "transcribe")
    try:
        if backend(settings) == "whispercpp":
            utterances, language = _run_whispercpp(settings, source)
        else:
            utterances, language = _run_faster_whisper(model or load_model(settings), source)
        conn.execute("DELETE FROM transcripts WHERE asset_id = ?", (asset_id,))
        conn.executemany(
            "INSERT INTO transcripts (asset_id, start_s, end_s, text, language) VALUES (?,?,?,?,?)",
            [(asset_id, s, e, t, language) for s, e, t in utterances],
        )
        conn.commit()
        db.job_finish(conn, asset_id, "transcribe")
    except Exception as e:  # noqa: BLE001
        db.job_finish(conn, asset_id, "transcribe", error=str(e))
        raise
    return len(utterances)
