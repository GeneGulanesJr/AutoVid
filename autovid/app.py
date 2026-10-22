"""Desktop shell: the app in a native window (pywebview), browser fallback.

``launch()`` starts the FastAPI server on a localhost port in a background
thread, waits for it to answer, then either opens a native window (pywebview
installed) or the default browser. Either way the UI is the same static page
talking to the same JSON API.
"""
from __future__ import annotations

import atexit
import logging
import subprocess
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

    _start_decisionmcp(settings)
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


def _start_decisionmcp(settings: Settings) -> subprocess.Popen | None:
    """Spawn the sibling ../DecisionMCP server if the decision layer is local and down.

    shortcut: Windows venv layout and a fixed sibling folder, make the path
    configurable if DecisionMCP lives elsewhere.
    """
    import httpx
    from urllib.parse import urlparse

    url = urlparse(settings.decisionmcp_url)
    if url.hostname not in ("127.0.0.1", "localhost") or settings.decision_mode == "heuristic":
        return None
    health = f"{url.scheme}://{url.netloc}/health"
    try:
        if httpx.get(health, timeout=1.0).status_code == 200:
            return None
    except Exception:  # noqa: BLE001 — not running, try to start it
        pass
    root = Path(__file__).resolve().parent.parent.parent / "DecisionMCP"
    exe = root / ".venv" / "Scripts" / "decisionmcp.exe"
    if not exe.exists():
        logger.info("DecisionMCP not found at %s — heuristic fallbacks stay active", exe)
        return None
    proc = subprocess.Popen([str(exe)], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    atexit.register(proc.terminate)  # we started it, we stop it
    _wait_ready(health, timeout_s=90.0)  # model load takes a few seconds
    return proc


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
    # pythonw: no console window behind the app.
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "pythonw.exe"
    script.write_text(
        "@echo off\r\n"
        f"start \"\" \"{venv_python}\" -m autovid.app_main\r\n",
        encoding="utf-8",
    )
    return script
