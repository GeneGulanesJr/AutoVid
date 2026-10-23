"""Generate a heterogeneous synthetic bench fixture + ground truth.

Four clips with different resolutions, frame rates, scene counts and audio,
built from ffmpeg lavfi sources so the true cut timestamps are exact. The
media is NOT committed — regenerate it with:

    python tools/bench_fixture.py bench/fixture

Writes ``<folder>/truth.json`` mapping clip filename → true boundary seconds
(the input for `autovid bench` / tools/run_bench.py).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autovid.ffmpeg_io.bins import ffmpeg_path  # noqa: E402

# (name, width, height, fps, audio, [(lavfi source, seconds), …]) —
# deliberately heterogeneous: resolution, frame rate, scene count, audio.
CLIPS = [
    ("bench_a", 1280, 720, 30, False,
     [("smptebars", 8), ("testsrc2", 8), ("color=c=black", 8)]),
    ("bench_b", 1920, 1080, 24, True,
     [("gradients", 12), ("testsrc2", 12), ("color=c=red", 12), ("smptebars", 12)]),
    ("bench_c", 480, 270, 15, False,
     [("testsrc2", 10), ("smptebars", 10)]),
    ("bench_d", 640, 480, 25, False,
     [("gradients", 30)]),
]


def make_clip(path: Path, w: int, h: int, fps: int, audio: bool,
              scenes: list[tuple[str, int]]) -> None:
    cmd = [ffmpeg_path(), "-nostdin", "-y", "-v", "error"]
    for src, secs in scenes:
        # lavfi wants "name=opt=…:opt=…" — a source that already carries an
        # option ("color=c=black") continues with ':' instead of '='.
        sep = "=" if "=" not in src else ":"
        cmd += ["-f", "lavfi", "-i", f"{src}{sep}duration={secs}:size={w}x{h}:rate={fps}"]
    total = sum(s for _, s in scenes)
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={total}"]
    n = len(scenes)
    cmd += [
        "-filter_complex", f"{''.join(f'[{i}:v]' for i in range(n))}concat=n={n}:v=1[v]",
        "-map", "[v]",
    ]
    if audio:
        cmd += ["-map", f"{n}:a"]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "bench" / "fixture"
    folder.mkdir(parents=True, exist_ok=True)
    truth: dict[str, list[float]] = {}
    for name, w, h, fps, audio, scenes in CLIPS:
        path = folder / f"{name}.mp4"
        make_clip(path, w, h, fps, audio, scenes)
        cuts = [round(sum(s for _, s in scenes[:i]), 3) for i in range(1, len(scenes))]
        truth[path.name] = cuts
        print(f"{path.name}: {w}x{h}@{fps} audio={audio} cuts={cuts}")
    (folder / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"truth.json → {folder / 'truth.json'}")


if __name__ == "__main__":
    main()
