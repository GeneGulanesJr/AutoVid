"""Render internals: per-EDL cross-process lock."""
from __future__ import annotations

from pathlib import Path

import pytest

from autovid.errors import AutoVidError
from autovid.render import _render_lock


def test_render_lock_excludes_second_holder(tmp_path: Path) -> None:
    lock = tmp_path / "renders" / "edl1" / ".render.lock"
    with _render_lock(lock):
        with pytest.raises(AutoVidError, match="already running"):
            with _render_lock(lock):
                pass  # pragma: no cover — the inner acquire must fail


def test_render_lock_releases_on_exit_and_crash(tmp_path: Path) -> None:
    lock = tmp_path / "renders" / "edl2" / ".render.lock"

    with _render_lock(lock):
        pass
    with _render_lock(lock):  # releasable after a clean exit
        pass

    # releasable after an exception mid-render (the finally must unlock)
    with pytest.raises(RuntimeError):
        with _render_lock(lock):
            raise RuntimeError("killed render")
    with _render_lock(lock):
        pass
