"""Shared fixtures: tiny real videos generated with the vendored ffmpeg."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autovid.config import Settings
from autovid.ffmpeg_io.bins import ffmpeg_path


def make_video(path: Path, *, duration: int = 4, width: int = 160, height: int = 120,
               fps: int = 12, audio: bool = False, hue_speed: float = 0.2) -> Path:
    """A short synthetic clip: animated color bars so scene/motion metrics vary."""
    src = f"testsrc2=duration={duration}:size={width}x{height}:rate={fps}"
    cmd = [ffmpeg_path(), "-y", "-v", "error", "-f", "lavfi", "-i", src]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


def make_multiscene_video(path: Path, *, scene_s: float = 3.0, scenes: int = 3,
                          width: int = 160, height: int = 120, fps: int = 12) -> Path:
    """Concatenated distinct sources → hard cuts at scene_s, 2*scene_s, …"""
    spec = f"duration={{d}}:size={width}x{height}:rate={fps}"
    sources = [
        "smptebars=" + spec,
        "testsrc2=" + spec,
        "color=red:" + spec,  # color takes the name positionally
        "gradients=" + spec,
    ]
    cmd = [ffmpeg_path(), "-y", "-v", "error"]
    for i in range(scenes):
        cmd += ["-f", "lavfi", "-i",
                sources[i % len(sources)].format(d=scene_s)]
    inputs = [f"[{i}:v]" for i in range(scenes)]
    cmd += ["-filter_complex", f"{''.join(inputs)}concat=n={scenes}:v=1[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Hermetic: never read the developer's real .env (a live key here would
    # turn offline tests into real API calls). Tests that need a key set one
    # explicitly via model_copy.
    return Settings(data_dir=tmp_path / "catalog", _env_file=None, vlm_api_key="")


@pytest.fixture
def video_folder(tmp_path: Path) -> Path:
    """Three clips + a duplicate + a non-video file."""
    folder = tmp_path / "footage"
    folder.mkdir()
    make_video(folder / "clip_a.mp4", duration=4, hue_speed=0.4)
    make_video(folder / "clip_b.mp4", duration=6, audio=True)
    sub = folder / "sub"
    sub.mkdir()
    make_video(sub / "clip_c.mov", duration=3)
    (folder / "notes.txt").write_text("not a video")
    return folder


def seed_asset(conn, asset_id: str = "a_test", *, duration: float = 30.0,
               width: int = 160, height: int = 120, fps: float = 12.0,
               path: str | None = None) -> str:
    """A catalog asset row without running the pipeline (FK target for tests)."""
    conn.execute(
        """INSERT INTO assets (id, sha256, path, original_path, original_name,
                               status, duration_s, width, height, fps)
           VALUES (?, ?, ?, ?, ?, 'organized', ?, ?, ?, ?)
           ON CONFLICT(id) DO NOTHING""",
        (asset_id, f"hash-{asset_id}", path or f"{asset_id}.mp4",
         path or f"{asset_id}.mp4", f"{asset_id}.mp4", duration, width, height, fps),
    )
    conn.commit()
    return asset_id


def seed_segment(conn, asset_id: str, segment_id: int, *, start: float = 0.0,
                 end: float = 10.0, description: str = "a test scene",
                 tags: list[str] | None = None, highlight: float = 0.8,
                 confidence: float = 0.9) -> int:
    """An accepted segment — storyboard-eligible by the same rule as the
    pipeline (verdict accept, review not pending)."""
    conn.execute(
        """INSERT INTO segments (id, asset_id, start_s, end_s, description, tags_json,
                                 highlight_score, decision_confidence, verdict, review_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'accept', 'auto')
           ON CONFLICT(id) DO NOTHING""",
        (segment_id, asset_id, start, end, description,
         json.dumps(tags or ["test"]), highlight, confidence),
    )
    conn.commit()
    return segment_id
