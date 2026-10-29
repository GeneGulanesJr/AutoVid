"""Transcribe stage tests: whisper.cpp JSON parsing + whisper-cli flag-set fallback.

The whisper.cpp binary itself is never run; the subprocess boundary is faked.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autovid import transcribe
from autovid.config import Settings
from autovid.errors import AutoVidError
from autovid.transcribe import _utterances_from_json

VULKAN_STDERR = "ggml_vulkan: Found 1 Vulkan devices: AMD Radeon\n"


# --- helpers: whisper.cpp JSON shapes ------------------------------------------


def _token(text: str, t_dtw: float | None = None) -> dict:
    tok = {"text": text}
    if t_dtw is not None:
        tok["t_dtw"] = t_dtw
    return tok


def _segment(text: str, frm: int, to: int, tokens: list[dict] | None = None) -> dict:
    seg = {"text": text, "offsets": {"from": frm, "to": to}}
    if tokens is not None:
        seg["tokens"] = tokens
    return seg


# --- pure JSON parsing ----------------------------------------------------------


def test_word_timestamps_trim_trailing_silence() -> None:
    # VAD-stretched segment: speech runs 1.2-3.4 s but offsets.to stretches to 9 s.
    data = {
        "result": {"language": "en"},
        "transcription": [_segment(
            " Hello world today", 1000, 9000,
            [_token("[_BEG_]"), _token("<|1.00|>"),  # special tokens: no timing
             _token(" Hello", 1200), _token(" world", 2200), _token(" today", 3400),
             _token("[_TT_4200_]")]),  # trailing special token
        ],
    }
    rows, language = _utterances_from_json(data)
    assert language == "en"
    start, end, text = rows[0]
    assert text == "Hello world today"
    assert start == 1.2  # first word's t_dtw, not the segment start
    # last word (3.4 s) + median inter-word gap (1.1 s), clamped well under the 9 s segment end
    assert end == 4.5
    assert end - 3.4 <= 1.2  # issue #10: no spoken clip ~1 s+ longer than its speech


def test_single_word_uses_fixed_tail() -> None:
    data = {"transcription": [_segment(
        " Yes", 2000, 8000, [_token(" Yes", 2100), _token("[_TT_2600_]")])]}
    start, end, text = _utterances_from_json(data)[0][0]
    # <3 words → 0.4 s fixed tail instead of the median; not the stretched 8 s segment end
    assert (start, end, text) == (2.1, 2.5, "Yes")


def test_end_never_before_start_plus_floor() -> None:
    # last word starts at the very end of the segment: clamping to offsets.to must not
    # push the end below start + 0.2 s
    data = {"transcription": [_segment(
        " Hi", 4000, 5000, [_token(" Hi", 4990)])]}
    start, end, _ = _utterances_from_json(data)[0][0]
    assert start == 4.99
    assert end >= start + 0.2


def test_no_tokens_uses_segment_offsets() -> None:
    data = {"result": {"language": "de"},
            "transcription": [_segment(" Hallo Welt", 500, 4000)]}
    rows, language = _utterances_from_json(data)
    assert rows == [(0.5, 4.0, "Hallo Welt")]  # verbatim offsets, no rate cap
    assert language == "de"


def test_tokens_without_dtw_fall_back_to_offsets() -> None:
    # plain -oj output: t_dtw missing or negative
    data = {"transcription": [_segment(
        " Hi", 300, 2500, [_token(" Hi", -1), _token("[_TT_900_]")])]}
    assert _utterances_from_json(data)[0] == [(0.3, 2.5, "Hi")]


def test_empty_text_segments_are_skipped_and_language_optional() -> None:
    data = {"transcription": [_segment("  ", 0, 500), _segment(" Go", 500, 900)]}
    rows, language = _utterances_from_json(data)
    assert rows == [(0.5, 0.9, "Go")]
    assert language is None


def test_import_keeps_faster_whisper_lazy() -> None:
    code = ("import sys; import autovid.transcribe; "
            "sys.exit(1 if 'faster_whisper' in sys.modules else 0)")
    root = Path(__file__).resolve().parents[1]
    assert subprocess.run([sys.executable, "-c", code], cwd=root,
                          capture_output=True).returncode == 0


# --- whisper-cli invocation and flag-set fallback --------------------------------


def _whispercpp_settings(tmp_path: Path) -> Settings:
    root = tmp_path / "wpp"
    (root / "bin").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "bin" / "whisper-cli.exe").touch()
    (root / "models" / "ggml-large-v3.bin").touch()
    (root / "models" / "ggml-silero-v5.1.2.bin").touch()
    return Settings(data_dir=tmp_path / "catalog", _env_file=None, vlm_api_key="",
                    whispercpp_dir=root)


def _fake_cli(monkeypatch: pytest.MonkeyPatch, payload: dict, behavior):
    """Fake subprocess.run inside autovid.transcribe: the ffmpeg call succeeds, whisper-cli
    calls get their rc from behavior(args) and write payload to <of>.json on success.
    Returns the recorded whisper-cli argv lists."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        args = [str(a) for a in args]
        if "ffmpeg" in Path(args[0]).name:
            return subprocess.CompletedProcess(args, 0)
        calls.append(args)
        rc = behavior(args)
        if rc == 0:
            out = Path(args[args.index("-of") + 1]).with_suffix(".json")
            out.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args, rc, stderr=VULKAN_STDERR)

    monkeypatch.setattr(transcribe, "subprocess", SimpleNamespace(run=fake_run))
    return calls


def test_whispercpp_argv_dtw_and_shaping_flags(tmp_path: Path, monkeypatch) -> None:
    payload = {"result": {"language": "en"}, "transcription": [_segment(
        " Hello", 1000, 9000, [_token(" Hello", 1200), _token("[_TT_2400_]")])]}
    calls = _fake_cli(monkeypatch, payload, lambda args: 0)
    settings = _whispercpp_settings(tmp_path)
    source = tmp_path / "clip.mp4"
    source.touch()

    rows, language = transcribe._run_whispercpp(settings, source)

    assert language == "en"
    assert rows == [(1.2, 1.6, "Hello")]  # first→last word + 0.4 s tail
    assert len(calls) == 1
    argv = calls[0]
    assert argv[-2:] == ["-dtw", "large-v3"]  # first attempt ends with the -dtw flag set
    assert "-ojf" in argv
    assert argv[argv.index("-mc") + 1] == "0"
    for flag in ("--vad", "-vm", "-vsd", "-vmsd", "-ml", "-sow", "-of"):
        assert flag in argv


def test_whispercpp_falls_back_without_dtw(tmp_path: Path, monkeypatch, caplog) -> None:
    # distilled/turbo models have no DTW alignment heads → -dtw run fails, -ojf retry works
    payload = {"result": {"language": "en"}, "transcription": [_segment(" Hi", 0, 900)]}
    calls = _fake_cli(monkeypatch, payload, lambda args: 1 if "-dtw" in args else 0)
    settings = _whispercpp_settings(tmp_path)
    source = tmp_path / "clip.mp4"
    source.touch()

    with caplog.at_level(logging.WARNING, logger="autovid.transcribe"):
        rows, _ = transcribe._run_whispercpp(settings, source)

    assert rows == [(0.0, 0.9, "Hi")]  # no word timing → segment offsets
    assert len(calls) == 2
    assert "-dtw" in calls[0]
    assert "-dtw" not in calls[1] and "-ojf" in calls[1]
    assert any("-dtw" in r.getMessage() for r in caplog.records)  # the warning


def test_whispercpp_falls_back_to_plain_oj_then_raises(tmp_path: Path, monkeypatch) -> None:
    payload = {"result": {"language": "en"}, "transcription": [_segment(" Hi", 0, 900)]}
    calls = _fake_cli(monkeypatch, payload, lambda args: 1)
    settings = _whispercpp_settings(tmp_path)
    source = tmp_path / "clip.mp4"
    source.touch()

    with pytest.raises(AutoVidError, match="whisper-cli failed \\(rc=1\\)"):
        transcribe._run_whispercpp(settings, source)

    assert len(calls) == 3
    assert calls[2][-1] == "-oj" and "-ojf" not in calls[2]  # original invocation as last resort
