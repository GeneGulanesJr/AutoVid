"""setup-whisper tests: pinned-manifest download verification, zip extraction, Vulkan self-test.

No network and no real whisper.cpp: httpx.MockTransport serves tiny fake payloads (the
same transport-injection pattern as test_vlm.py) and the subprocess boundary is faked,
so the download → verify → extract → self-test pipeline is exercised end to end offline.
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from autovid import whisper_setup
from autovid.config import Settings
from autovid.errors import AutoVidError
from autovid.whisper_setup import Artifact

VULKAN_STDERR = "ggml_vulkan: Found 1 Vulkan devices: AMD Radeon RX 9070 XT\n"
CPU_FALLBACK_STDERR = ("whisper_init_with_params_no_state: loading model\n"
                       "ggml_vulkan: Failed to create Vulkan instance\n"
                       "system_info: WHISPER : CPU\n")

MODEL_BYTES = b"fake ggml model"
VAD_BYTES = b"fake silero vad"
ZIP_MEMBERS = {"whisper-cli.exe": b"fake cli", "ggml.dll": b"fake ggml",
               "whisper.dll": b"fake whisper"}


# --- helpers ---------------------------------------------------------------------


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in ZIP_MEMBERS.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _artifact(name: str, url_name: str, pinned: bytes, rel_dest: str) -> Artifact:
    return Artifact(name=name, url=f"https://fake.test/{url_name}",
                    sha256=hashlib.sha256(pinned).hexdigest(), size=len(pinned),
                    rel_dest=Path(rel_dest))


def _install_fake_manifest(monkeypatch: pytest.MonkeyPatch, *, zip_served: bytes | None = None,
                           zip_pinned: bytes | None = None) -> None:
    """Swap the pinned manifest for tiny fakes; pins normally match the payloads the
    mock transport serves, but a test can pin different bytes to force a mismatch."""
    served = _zip_bytes() if zip_served is None else zip_served
    pinned_zip = served if zip_pinned is None else zip_pinned

    def manifest(model_name: str) -> tuple[Artifact, Artifact, Artifact]:
        return (
            _artifact("fake vulkan zip", "whisper.zip", pinned_zip, "downloads/whisper.zip"),
            _artifact("fake silero vad", "ggml-silero-v5.1.2.bin", VAD_BYTES,
                      "models/ggml-silero-v5.1.2.bin"),
            _artifact("fake ggml model", f"ggml-{model_name}.bin", MODEL_BYTES,
                      f"models/ggml-{model_name}.bin"),
        )

    monkeypatch.setattr(whisper_setup, "_manifest", manifest)


def _transport(payloads: dict[str, bytes], hits: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        for name, data in payloads.items():
            if request.url.path.endswith(name):
                return httpx.Response(200, content=data)
        return httpx.Response(404, text=f"no fake payload for {request.url}")

    return httpx.MockTransport(handler)


def _fake_subprocess(monkeypatch: pytest.MonkeyPatch, *, cli_rc: int = 0,
                     cli_stderr: str = VULKAN_STDERR) -> list[list[str]]:
    """Fake subprocess.run: ffmpeg succeeds, whisper-cli returns cli_rc/cli_stderr.
    Returns every argv seen (ffmpeg conversions included)."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        args = [str(a) for a in args]
        calls.append(args)
        if "ffmpeg" in Path(args[0]).name:
            return subprocess.CompletedProcess(args, 0)
        return subprocess.CompletedProcess(args, cli_rc, stderr=cli_stderr)

    monkeypatch.setattr(whisper_setup, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(whisper_setup, "ffmpeg_path", lambda: "ffmpeg")
    return calls


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "catalog", _env_file=None, vlm_api_key="",
                    whispercpp_dir=tmp_path / "wpp")


def _happy_transport(hits: list[str]) -> httpx.MockTransport:
    return _transport({"whisper.zip": _zip_bytes(), "ggml-large-v3.bin": MODEL_BYTES,
                       "ggml-silero-v5.1.2.bin": VAD_BYTES}, hits)


# --- the pinned manifest -----------------------------------------------------------


def test_manifest_pins_documented_hashes() -> None:
    zip_art, vad_art, model = whisper_setup._manifest("large-v3")
    assert zip_art.sha256 == "e0d20a0f92e31b98adc0faf71172efc810b701e6391a9d858ca045bff26f77cd"
    assert zip_art.size == 21_203_995
    assert zip_art.url.endswith(
        "lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-windows-vulkan-x64.zip")
    assert model.sha256 == "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
    assert model.size == 3_095_033_483
    assert model.url == "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
    assert vad_art.sha256 == "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf"
    assert vad_art.size == 885_098
    assert vad_art.url == "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"


def test_manifest_leaves_other_models_unpinned() -> None:
    # non-default models download from the same repo layout but only get a size>0 check
    _, _, turbo = whisper_setup._manifest("large-v3-turbo")
    assert turbo.url == "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
    assert turbo.sha256 == "" and turbo.size == 0


# --- download → verify → extract → self-test ---------------------------------------


def test_setup_downloads_extracts_and_self_tests(tmp_path: Path, monkeypatch) -> None:
    _install_fake_manifest(monkeypatch)
    hits: list[str] = []
    transport = _happy_transport(hits)
    calls = _fake_subprocess(monkeypatch)

    summary = whisper_setup.setup_whisper(_settings(tmp_path), transport=transport)

    root = tmp_path / "wpp"
    # extraction layout transcribe.py expects: bin/whisper-cli.exe + DLLs flat in bin/
    assert (root / "bin" / "whisper-cli.exe").read_bytes() == b"fake cli"
    assert (root / "bin" / "ggml.dll").read_bytes() == b"fake ggml"
    assert (root / "downloads" / "whisper.zip").exists()
    assert (root / "models" / "ggml-large-v3.bin").read_bytes() == MODEL_BYTES
    assert (root / "models" / "ggml-silero-v5.1.2.bin").read_bytes() == VAD_BYTES
    assert not list(root.rglob("*.part"))  # nothing left over after the atomic renames
    assert len([h for h in hits]) == 3
    cli_calls = [c for c in calls if Path(c[0]).name == "whisper-cli.exe"]
    assert len(cli_calls) == 1  # the self-test ran, exactly once
    assert cli_calls[0][0].endswith(str(root / "bin" / "whisper-cli.exe"))
    assert cli_calls[0][cli_calls[0].index("-m") + 1].endswith("ggml-large-v3.bin")
    assert "Vulkan" in summary and "RX 9070 XT" in summary  # device line made it into the summary


def test_setup_is_idempotent_when_hashes_match(tmp_path: Path, monkeypatch) -> None:
    _install_fake_manifest(monkeypatch)
    hits: list[str] = []
    transport = _happy_transport(hits)
    calls = _fake_subprocess(monkeypatch)
    settings = _settings(tmp_path)
    whisper_setup.setup_whisper(settings, transport=transport)
    assert len(hits) == 3

    whisper_setup.setup_whisper(settings, transport=transport)

    assert len(hits) == 3  # nothing re-downloaded: every file still matches its pin
    cli_runs = [c for c in calls if Path(c[0]).name == "whisper-cli.exe"]
    assert len(cli_runs) == 2  # but the self-test still re-runs — it is the GPU check


def test_corrupted_existing_file_is_re_downloaded(tmp_path: Path, monkeypatch) -> None:
    _install_fake_manifest(monkeypatch)
    drifted = tmp_path / "wpp" / "models" / "ggml-large-v3.bin"
    drifted.parent.mkdir(parents=True)
    drifted.write_bytes(b"bit rot")
    hits: list[str] = []
    _fake_subprocess(monkeypatch)

    whisper_setup.setup_whisper(_settings(tmp_path), transport=_happy_transport(hits))

    assert drifted.read_bytes() == MODEL_BYTES  # pin mismatch → re-download replaced it
    assert len(hits) == 3


def test_corrupt_zip_raises_with_both_hashes_and_deletes_part(tmp_path: Path, monkeypatch) -> None:
    good = _zip_bytes()
    bad = b"corrupted" + good[9:]  # same length, different bytes
    _install_fake_manifest(monkeypatch, zip_served=bad, zip_pinned=good)
    transport = _transport({"whisper.zip": bad, "ggml-large-v3.bin": MODEL_BYTES,
                            "ggml-silero-v5.1.2.bin": VAD_BYTES}, [])
    _fake_subprocess(monkeypatch)

    with pytest.raises(AutoVidError) as excinfo:
        whisper_setup.setup_whisper(_settings(tmp_path), transport=transport)

    msg = str(excinfo.value)
    assert "fake vulkan zip" in msg  # the artifact is named
    assert hashlib.sha256(good).hexdigest() in msg  # pinned hash
    assert hashlib.sha256(bad).hexdigest() in msg  # actual hash
    root = tmp_path / "wpp"
    assert not list(root.rglob("*.part"))  # partial download deleted
    assert not (root / "bin").exists()  # nothing extracted from an unverified zip


def test_short_download_raises_with_both_hashes_and_deletes_part(tmp_path: Path,
                                                                 monkeypatch) -> None:
    _install_fake_manifest(monkeypatch)  # pins match the FULL payloads
    truncated = MODEL_BYTES[:5]
    transport = _transport({"whisper.zip": _zip_bytes(), "ggml-large-v3.bin": truncated,
                            "ggml-silero-v5.1.2.bin": VAD_BYTES}, [])
    _fake_subprocess(monkeypatch)

    with pytest.raises(AutoVidError) as excinfo:
        whisper_setup.setup_whisper(_settings(tmp_path), transport=transport)

    msg = str(excinfo.value)
    assert "fake ggml model" in msg
    assert hashlib.sha256(MODEL_BYTES).hexdigest() in msg  # pinned hash
    assert hashlib.sha256(truncated).hexdigest() in msg  # actual hash
    assert str(len(MODEL_BYTES)) in msg  # and the size shortfall
    assert not list((tmp_path / "wpp").rglob("*.part"))
    assert not (tmp_path / "wpp" / "bin").exists()


# --- the self-test must fail loudly without a Vulkan device -------------------------


def test_self_test_fails_loudly_when_vulkan_line_missing(tmp_path: Path, monkeypatch) -> None:
    _install_fake_manifest(monkeypatch)
    transport = _happy_transport([])
    calls = _fake_subprocess(monkeypatch, cli_stderr=CPU_FALLBACK_STDERR)

    with pytest.raises(AutoVidError) as excinfo:
        whisper_setup.setup_whisper(_settings(tmp_path), transport=transport)

    msg = str(excinfo.value)
    assert "Vulkan" in msg and "CPU" in msg
    assert "driver" in msg  # tells the user what to do
    # everything downloaded and extracted fine — it is the GPU check that failed
    assert (tmp_path / "wpp" / "bin" / "whisper-cli.exe").exists()
    assert len([c for c in calls if Path(c[0]).name == "whisper-cli.exe"]) == 1


def test_self_test_fails_loudly_on_nonzero_rc(tmp_path: Path, monkeypatch) -> None:
    _install_fake_manifest(monkeypatch)
    transport = _happy_transport([])
    _fake_subprocess(monkeypatch, cli_rc=1, cli_stderr="whisper: failed to load the model")

    with pytest.raises(AutoVidError, match=r"self-test failed \(rc=1\)"):
        whisper_setup.setup_whisper(_settings(tmp_path), transport=transport)
