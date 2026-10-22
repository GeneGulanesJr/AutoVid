"""Desktop entry: native window (pywebview) with a browser fallback."""
from __future__ import annotations

import argparse

from .app import launch
from .config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autovid-app", description="Run the AutoVid desktop app.")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--browser", action="store_true",
                        help="open in the default browser instead of a native window")
    args = parser.parse_args()
    launch(get_settings(), port=args.port, window=not args.browser)


if __name__ == "__main__":
    main()
