"""Footage kinds from metadata and transcripts."""
from __future__ import annotations

import json

from autovid import db
from autovid.kinds import classify


def _asset(conn, n, *, at, dur, fps=23.98, size=(3840, 2880), audio=1):
    tags = {"format": {"tags": {"creation_time": f"2026-10-06T{at}.000000Z"}}}
    conn.execute(
        "INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s, fps, width, height,"
        " has_audio, ffprobe_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (f"a{n}", f"s{n}", "p", "p", f"c{n}.mp4", dur, fps, size[0], size[1], audio, json.dumps(tags)))


def test_classify_flags_timelapse_other_camera_no_audio_and_speech(tmp_path):
    c = db.connect(tmp_path / "k.db")
    _asset(c, 0, at="02:11:49", dur=90)                            # ordinary, then the next starts 2 min later
    _asset(c, 1, at="02:14:14", dur=60)
    _asset(c, 2, at="02:26:38", dur=60)                            # 8 min wait after it: ratio 8, same format
    _asset(c, 3, at="02:35:00", dur=78, fps=29.97, size=(3840, 2160))  # other format, next clip 20 min later
    _asset(c, 4, at="02:55:00", dur=2, audio=0)
    c.execute("INSERT INTO transcripts (asset_id, start_s, end_s, text) VALUES ('a0', 0, 40, 'chika')")
    kinds = classify(c)
    assert kinds["a0"] == {"speech"}                               # 40 s of 90 s spoken
    assert kinds["a2"] == set()                                    # a long pause is not a timelapse
    assert kinds["a3"] == {"other_camera", "timelapse"}
    assert kinds["a4"] == {"no_audio"}                             # last clip: no gap to measure


def test_extreme_gap_alone_marks_a_timelapse_and_missing_metadata_is_harmless(tmp_path):
    c = db.connect(tmp_path / "k.db")
    _asset(c, 0, at="01:00:00", dur=30)
    _asset(c, 1, at="02:00:00", dur=30)                            # gap 3600 s / 30 s = 120x
    c.execute("INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s, ffprobe_json)"
              " VALUES ('x', 'sx', 'p', 'p', 'x.mp4', 10, 'not json')")
    kinds = classify(c)
    assert "timelapse" in kinds["a0"] and kinds["x"] <= {"no_audio", "other_camera"}
