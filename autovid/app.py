"""Desktop shell: the app in a native window (pywebview), browser fallback.

``launch()`` starts the FastAPI server on a localhost port in a background
thread, waits for it to answer, then either opens a native window (pywebview
installed) or the default browser. Either way the UI is the same static page
talking to the same JSON API.
"""
from __future__ import annotations

import logging
import threading
import time
import webbrowser
from pathlib import Path

from .config import Settings

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8766


def launch(settings: Settings | None = None, *, port: int = DEFAULT_PORT,
           window: bool = True) -> None:
    """Start the app. With ``window=True`` and pywebview installed, opens a
    native window; otherwise opens the default browser tab."""
    import threading

    import uvicorn

    from .web import create_app

    app = create_app(settings)
    server = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port,
                                   log_level="warning"),
        daemon=True,
    )
    server.start()

    url = f"http://127.0.0.1:{port}/"
    _wait_ready(url)
    logger.info("AutoVid app ready at %s", url)

    if window:
        try:
            import webview  # pywebview

            webview.create_window("AutoVid", url, width=1280, height=820)
            webview.start()  # blocks until the window closes
            return
        except ImportError:
            logger.info("pywebview not installed — opening the browser instead "
                        "(pip install pywebview for the desktop window)")
        except Exception as e:  # noqa: BLE001 — never lose the UI to a GUI error
            logger.exception("native window failed (%s) — falling back to the browser", e)
    webbrowser.open(url)
    print(f"AutoVid is running at {url} — close this window to stop.")
    while server.is_alive():
        time.sleep(1)


def _wait_ready(url: str, timeout_s: float = 20.0) -> None:
    import httpx

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except Exception:  # noqa: BLE001 — server not up yet
            time.sleep(0.25)
    logger.warning("server did not answer within %.0fs", timeout_s)


def write_launcher_script(data_dir: Path) -> Path:
    """A double-clickable Windows launcher for non-technical use."""
    script = data_dir / "AutoVid.bat"
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"
    script.write_text(
        "@echo off\r\n"
        f"start \"AutoVid\" \"{venv_python}\" -m autovid.app\r\n",
        encoding="utf-8",
    )
    return script
