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
    # Order must match the docstring: explicit dir wins over whatever happens
    # to be on PATH, vendored build is the last resort.
    if explicit_dir:
        found = _exe_in(Path(explicit_dir), name)
        if found:
            return found
    which = shutil.which(name)
    if which:
        return which
    for vendored in sorted(REPO_ROOT.glob(".tools/ffmpeg*/bin")):
        found = _exe_in(vendored, name)
        if found:
            return found
    raise AutoVidError(
        f"{name} not found. Set AUTOVID_FFMPEG_DIR to a folder containing "
        f"{name}.exe, install ffmpeg on PATH, or keep the vendored "
        f".tools/ffmpeg* build in the repo."
    )


def _exe_in(directory: Path, name: str) -> str | None:
    exe = directory / f"{name}.exe"
    if exe.exists():  # windows
        return str(exe)
    posix = directory / name
    if posix.exists():
        return str(posix)
    return None


def ffmpeg_path() -> str:
    return _bin("ffmpeg", os.environ.get("AUTOVID_FFMPEG_DIR"))


def ffprobe_path() -> str:
    return _bin("ffprobe", os.environ.get("AUTOVID_FFMPEG_DIR"))
