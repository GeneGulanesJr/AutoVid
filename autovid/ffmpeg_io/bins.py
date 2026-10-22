"""Locate the ffmpeg/ffprobe binaries.

Resolution order: explicit ``AUTOVID_FFMPEG_DIR`` → ``PATH`` → the vendored
``.tools/ffmpeg*/bin`` static build. Resolved once per process.
"""
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from ..errors import AutoVidError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=None)
def _bin(name: str, explicit_dir: str | None) -> str:
    candidates: list[Path] = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    which = shutil.which(name)
    if which:
        return which
    vendored = sorted(REPO_ROOT.glob(".tools/ffmpeg*/bin"))
    candidates.extend(vendored)
    for directory in candidates:
        exe = directory / f"{name}.exe"
        if exe.exists():  # windows
            return str(exe)
        posix = directory / name
        if posix.exists():
            return str(posix)
    raise AutoVidError(
        f"{name} not found. Set AUTOVID_FFMPEG_DIR to a folder containing "
        f"{name}.exe, install ffmpeg on PATH, or keep the vendored "
        f".tools/ffmpeg* build in the repo."
    )


def ffmpeg_path() -> str:
    return _bin("ffmpeg", os.environ.get("AUTOVID_FFMPEG_DIR"))


def ffprobe_path() -> str:
    return _bin("ffprobe", os.environ.get("AUTOVID_FFMPEG_DIR"))
