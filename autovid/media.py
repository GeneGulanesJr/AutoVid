"""Frame payload helpers: local files → base64 JPEG for model calls."""
from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image

from .errors import AutoVidError

DEFAULT_MAX_WIDTH = 768
DEFAULT_QUALITY = 80


def b64_jpeg(
    path: str | Path, *, max_width: int = DEFAULT_MAX_WIDTH, quality: int = DEFAULT_QUALITY
) -> str:
    """Read an image file and return it as a base64-encoded JPEG string.

    Frames are downscaled to ``max_width`` (never upscaled) before encoding —
    decision and VLM calls do not need full-resolution pixels, and payload size
    dominates cloud call cost.

    Results are cached per (path, mtime): the same immutable frame is encoded
    repeatedly across phases (continuity pairs, boundary probes, scene
    evidence) and the encode is deterministic. Bounded — payloads are large.
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError as e:
        raise AutoVidError(f"cannot encode frame {path}: {e}") from e
    return _b64_jpeg_cached(str(p), st.st_mtime_ns, max_width, quality)


@lru_cache(maxsize=128)
def _b64_jpeg_cached(path: str, mtime_ns: int, max_width: int, quality: int) -> str:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, max(1, round(img.height * ratio))))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
    except Exception as e:  # noqa: BLE001
        raise AutoVidError(f"cannot encode frame {path}: {e}") from e
    return base64.b64encode(buf.getvalue()).decode("ascii")


def data_uri(b64: str) -> str:
    return f"data:image/jpeg;base64,{b64}"
