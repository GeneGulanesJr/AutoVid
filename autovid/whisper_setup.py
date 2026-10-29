"""One-command setup for the whisper.cpp Vulkan speech-to-text backend (``setup-whisper``).

Upstream whisper.cpp ships Windows binaries for CPU and CUDA only; AMD GPUs need the
Vulkan build from lemonade-sdk/whisper.cpp-rocm. This module downloads that build, the
ggml Whisper model and the silero VAD model into ``settings.whispercpp_dir`` (default
``~/.whisper-vulkan``, the layout ``transcribe.py`` expects), verifying pinned sha256
hashes and byte sizes along the way:

- bytes stream to ``<dest>.part`` and are hashed while streaming (a 3 GB model never
  sits in memory); only a verified file is atomically renamed into place, so a
  corrupted/truncated download never lands where ``transcribe`` would use it;
- a re-run skips any file whose hash already matches its pin (idempotent);
- it finishes with a one-second self-test that FAILS LOUDLY unless whisper.cpp's log
  shows ``ggml_vulkan: Found ... device`` — a silent CPU fallback (seen on RX 9070 XT)
  must not pass as a working GPU install.

Binary + model hashes are pinned in the manifest below (``ZIP_ARTIFACT``,
``VAD_ARTIFACT`` and :func:`_manifest`).
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Settings
from .errors import AutoVidError
from .ffmpeg_io.bins import ffmpeg_path

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024  # 1 MiB stream chunks — hash while streaming, never buffer the file
_PROGRESS_STEP_BYTES = 100 * 1024 * 1024  # log a heartbeat every ~100 MB so 3 GB doesn't look hung


@dataclass(frozen=True)
class Artifact:
    """One pinned download: source URL, expected sha256/size, destination relative
    to ``settings.whispercpp_dir``. ``sha256 == ""`` means "not pinned" (verify
    non-empty only) — used for models other than the pinned default."""

    name: str
    url: str
    sha256: str
    size: int
    rel_dest: Path


_ZIP_NAME = "whisper-v1.8.4-windows-vulkan-x64.zip"
_GITHUB_RELEASE = "https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4"
_HF_WHISPER = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
_HF_VAD = "https://huggingface.co/ggml-org/whisper-vad/resolve/main"

# The zip layout is FLAT (whisper-cli.exe + ggml/*.dll at its root); it is kept under
# downloads/ so re-runs skip the download, and extracted into bin/ where transcribe.py
# looks for <whispercpp_dir>/bin/whisper-cli.exe.
ZIP_ARTIFACT = Artifact(
    name="whisper.cpp v1.8.4 Windows Vulkan build (lemonade-sdk/whisper.cpp-rocm, MIT)",
    url=f"{_GITHUB_RELEASE}/{_ZIP_NAME}",
    sha256="e0d20a0f92e31b98adc0faf71172efc810b701e6391a9d858ca045bff26f77cd",
    size=21_203_995,
    rel_dest=Path("downloads") / _ZIP_NAME,
)
VAD_ARTIFACT = Artifact(
    name="silero VAD model (ggml-silero-v5.1.2.bin)",
    url=f"{_HF_VAD}/ggml-silero-v5.1.2.bin",
    sha256="29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    size=885_098,
    rel_dest=Path("models") / "ggml-silero-v5.1.2.bin",
)


def _manifest(model_name: str) -> tuple[Artifact, Artifact, Artifact]:
    """The download manifest, smallest first so a dead network fails fast.

    Only ``large-v3`` (the shipped default ``settings.whisper_model``) has a pinned
    hash. Any other model is fetched as ``ggml-<name>.bin`` from the same
    ggerganov/whisper.cpp repo layout and verified for size > 0 ONLY — pin its
    sha256/size here before relying on it for unattended use.
    """
    if model_name == "large-v3":
        model = Artifact(
            name="Whisper model (ggml-large-v3.bin)",
            url=f"{_HF_WHISPER}/ggml-large-v3.bin",
            sha256="64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
            size=3_095_033_483,
            rel_dest=Path("models") / "ggml-large-v3.bin",
        )
    else:
        model = Artifact(
            name=f"Whisper model (ggml-{model_name}.bin, hash not pinned)",
            url=f"{_HF_WHISPER}/ggml-{model_name}.bin",
            sha256="",
            size=0,
            rel_dest=Path("models") / f"ggml-{model_name}.bin",
        )
    return ZIP_ARTIFACT, VAD_ARTIFACT, model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(dest: Path, art: Artifact, transport: httpx.BaseTransport | None) -> None:
    """Stream the artifact to ``<dest>.part``, hashing while streaming.

    On any sha256/size mismatch the .part is deleted and :class:`AutoVidError`
    names the artifact plus both the pinned and the actual hash. ``transport`` is
    injectable for tests (httpx.MockTransport), mirroring ``vlm.py``.
    """
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    got = 0
    try:
        # HF "resolve/main" and GitHub release URLs both redirect to CDNs; long read
        # timeout because a 3 GB transfer stalls between chunks on a slow link.
        with httpx.Client(timeout=httpx.Timeout(60.0, read=600.0), follow_redirects=True,
                          transport=transport) as client:
            with client.stream("GET", art.url) as resp:
                resp.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in resp.iter_bytes(_CHUNK):
                        fh.write(chunk)
                        digest.update(chunk)
                        got += len(chunk)
                        if got // _PROGRESS_STEP_BYTES > (got - len(chunk)) // _PROGRESS_STEP_BYTES:
                            logger.info("%s: %.0f MB downloaded", art.name, got / 1e6)
    except (httpx.HTTPError, OSError) as e:
        part.unlink(missing_ok=True)
        raise AutoVidError(f"{art.name}: download from {art.url} failed: {e}") from e

    problems: list[str] = []
    if art.sha256 and digest.hexdigest() != art.sha256:
        problems.append(f"sha256 {digest.hexdigest()} does not match the pinned {art.sha256}")
    if art.size and got != art.size:
        problems.append(f"got {got} bytes, expected {art.size}")
    if not art.size and got == 0:
        problems.append("got 0 bytes")
    if problems:
        part.unlink(missing_ok=True)
        raise AutoVidError(
            f"{art.name}: download from {art.url} failed verification "
            f"({'; '.join(problems)}); the partial file was deleted")
    os.replace(part, dest)  # atomic within the same directory — dest never holds partial bytes


def _ensure_artifact(root: Path, art: Artifact, transport: httpx.BaseTransport | None) -> bool:
    """Download unless the destination already matches its pin. True when downloaded."""
    dest = root / art.rel_dest
    if dest.exists():
        if (art.sha256 and _sha256(dest) == art.sha256) or (not art.sha256 and dest.stat().st_size > 0):
            return False
        logger.warning("%s exists but does not match its pin — re-downloading", art.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download(dest, art, transport)
    return True


def _extract_zip(zip_path: Path, bin_dir: Path) -> None:
    """Extract the flat zip (whisper-cli.exe + DLLs at its root) into ``bin/``.

    Safe to extractall despite zip-slip concerns: the zip's sha256 was verified
    against the pin before we got here, so its member paths are known.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(bin_dir)


def _self_test(root: Path, model_name: str) -> str:
    """Transcribe one second of 440 Hz sine and require a Vulkan device in the log.

    whisper.cpp prints ``ggml_vulkan: Found ...`` while loading the model, so this
    pays the one-time 3 GB model load — once, here, instead of silently on a user's
    first real transcription. Returns the matching log line (device description).
    """
    cli = root / "bin" / "whisper-cli.exe"
    model = root / "models" / f"ggml-{model_name}.bin"
    for p in (cli, model):
        if not p.exists():
            raise AutoVidError(f"whisper.cpp setup incomplete: {p} is missing")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "sine.wav"
        subprocess.run([ffmpeg_path(), "-y", "-v", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=1", "-ac", "1", "-ar", "16000",
                        str(wav)], check=True, capture_output=True, timeout=120)
        # Generous timeout: the 3 GB model load from a cold disk dominates, the
        # one-second transcription itself is trivial.
        proc = subprocess.run([str(cli), "-m", str(model), "-f", str(wav)],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=1800)
    if proc.returncode != 0:
        raise AutoVidError(f"whisper-cli self-test failed (rc={proc.returncode}): "
                           f"{proc.stderr.strip()[-400:]}")
    vulkan_line = next((ln.strip() for ln in proc.stderr.splitlines()
                        if "ggml_vulkan: Found" in ln), None)
    if vulkan_line is None:
        # Silent CPU fallback: rc is 0 and text comes out, but every transcription
        # would crawl. transcribe.py only *warns* about this; setup must fail.
        raise AutoVidError(
            "whisper.cpp self-test did not report a Vulkan device "
            "('ggml_vulkan: Found ...' missing from its log) — it silently used the CPU. "
            "Update your GPU driver (Vulkan runtime), and if that does not help re-run "
            f"`autovid setup-whisper`; the build in {root / 'bin'} may lack Vulkan support.")
    return vulkan_line


def setup_whisper(settings: Settings, *, model: str | None = None,
                  transport: httpx.BaseTransport | None = None) -> str:
    """Set up the GPU transcription backend from nothing, then prove it works.

    Downloads + verifies the whisper.cpp Vulkan build, the silero VAD model and the
    ggml model (``model`` overrides ``settings.whisper_model``) into
    ``settings.whispercpp_dir``, extracts the build into ``bin/``, runs the Vulkan
    self-test and returns a short summary. Already-pinned files are not re-downloaded.
    """
    root = settings.whispercpp_dir
    model_name = model or settings.whisper_model
    zip_art, vad_art, model_art = _manifest(model_name)

    fetched: list[str] = []
    for art in (zip_art, vad_art, model_art):
        if _ensure_artifact(root, art, transport):
            fetched.append(art.name)
    bin_dir = root / "bin"
    if not (bin_dir / "whisper-cli.exe").exists():
        _extract_zip(root / zip_art.rel_dest, bin_dir)

    device = _self_test(root, model_name)
    summary = (f"whisper.cpp Vulkan backend ready under {root} "
               f"({len(fetched)} file(s) downloaded); self-test: {device}")
    logger.info(summary)
    return summary
