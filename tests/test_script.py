"""Transcribe + script stages with a fake whisper model and a fake LLM."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autovid import db
from autovid.config import Settings
from autovid.errors import AutoVidError
from autovid.script import write_script
from autovid.transcribe import load_model, transcribe_asset


def _seg(start, end, text, *, logprob=-0.3, comp=1.2, no_speech=0.1):
    return SimpleNamespace(start=start, end=end, text=text, avg_logprob=logprob,
                           compression_ratio=comp, no_speech_prob=no_speech)


class FakeWhisper:
    def __init__(self, segments):
        self.segments = segments

    def transcribe(self, path, **kwargs):
        assert kwargs["vad_filter"] is True
        return iter(self.segments), SimpleNamespace(language="tl")


class FakeLLM:
    configured = True

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def ask_json(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    for n, name in enumerate(("DJI_0001.MP4", "DJI_0002.MP4")):
        f = tmp_path / name
        f.write_bytes(b"x")
        c.execute("INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s,"
                  " has_audio) VALUES (?,?,?,?,?,?,1)",
                  (f"a_{n}", f"sha{n}", str(f), str(f), name, 30.0))
    c.commit()
    return c


def _line(conn, asset, start, end, text):
    return conn.execute("INSERT INTO transcripts (asset_id, start_s, end_s, text) VALUES (?,?,?,?)",
                        (asset, start, end, text)).lastrowid


def _window(conn, asset, start, end, desc, foods, score=0.5):
    return conn.execute(
        "INSERT INTO analyses (asset_id, window_start, window_end, provider, model, response_json)"
        " VALUES (?,?,?,?,?,?)",
        (asset, start, end, "claude", "m",
         json.dumps({"description": desc, "foods": foods, "highlight_score": score, "uncertainty": 0.0}))).lastrowid


def test_transcribe_stores_confident_speech_only(conn):
    model = FakeWhisper([
        _seg(0, 2, "  Good morning, ma'am. "),
        _seg(2, 4, "tik-tik-tik-tik", comp=3.1),          # repetition loop
        _seg(4, 6, "Thank you for watching", no_speech=0.9),  # hallucinated on silence
        _seg(6, 8, "mumble", logprob=-1.8),                # low confidence
        _seg(8, 9, "   "),                                 # empty
    ])
    s = Settings(whisper_backend="faster-whisper")
    assert transcribe_asset(conn, s, "a_0", model=model) == 1
    row = conn.execute("SELECT text, language FROM transcripts").fetchone()
    assert (row["text"], row["language"]) == ("Good morning, ma'am.", "tl")
    # re-running replaces rather than duplicating
    transcribe_asset(conn, s, "a_0", model=model)
    assert conn.execute("SELECT count(*) FROM transcripts").fetchone()[0] == 1


def test_whispercpp_backend_parses_json_and_warns_on_cpu_fallback(conn, tmp_path, monkeypatch, caplog):
    import subprocess

    root = tmp_path / "wcpp"
    (root / "bin").mkdir(parents=True)
    (root / "models").mkdir()
    for p in (root / "bin" / "whisper-cli.exe", root / "models" / "ggml-large-v3.bin",
              root / "models" / "ggml-silero-v5.1.2.bin"):
        p.write_bytes(b"x")
    s = Settings(whispercpp_dir=root)  # auto → whispercpp because the exe exists
    assert load_model(s) is None
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "-of" in cmd:  # the whisper-cli call (ffmpeg has no -of)
            out = cmd[cmd.index("-of") + 1]
            Path(out + ".json").write_text(json.dumps({
                "result": {"language": "tl"},
                "transcription": [
                    # word timing says the speech ends at 0.8s; the segment stretches to 9.5s
                    {"offsets": {"from": 500, "to": 9500}, "text": " Tara na!",
                     "tokens": [{"text": "[_BEG_]", "t_dtw": -1}, {"text": " Tara", "t_dtw": 500},
                                {"text": " na!", "t_dtw": 800}, {"text": "[_END_]", "t_dtw": -1}]},
                    {"offsets": {"from": 3000, "to": 3500}, "text": "  "}]}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")  # no "ggml_vulkan: Found" line

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level("WARNING"):
        assert transcribe_asset(conn, s, "a_0") == 1
    assert "ran on the CPU" in caplog.text
    row = conn.execute("SELECT start_s, end_s, text, language FROM transcripts").fetchone()
    assert (row["start_s"], row["text"], row["language"]) == (0.5, "Tara na!", "tl")
    assert row["end_s"] == pytest.approx(1.2)  # first→last word + last-word tail, not the 9.5s stretch
    whisper_call = calls[1]
    assert "--vad" in whisper_call and whisper_call[whisper_call.index("-l") + 1] == "auto"
    assert "-ojf" in whisper_call and whisper_call[whisper_call.index("-dtw") + 1] == "large-v3"


def test_script_orders_lines_and_shots_and_merges_adjacent_speech(conn):
    a = _line(conn, "a_0", 1.0, 3.0, "Tara na!")
    b = _line(conn, "a_0", 3.5, 6.0, "Dito tayo kakain.")   # 0.5s gap → merges with a
    c = _line(conn, "a_1", 10.0, 12.0, "Ang sarap!")
    food = _window(conn, "a_1", 14.0, 22.0, "A plate of pancakes", ["pancakes"])
    reply = {"title": "Pancake day", "beats": [
        {"label": "intro", "order": [f"L{a}", f"L{b}"]},
        {"label": "payoff", "order": [f"S{food}", f"L{c}", "L9999", f"L{a}", "junk"]}]}
    llm = FakeLLM(reply, reply)   # 8s of speech vs 4s of food → one food-focus re-ask, same answer kept
    board, title = write_script(conn, llm, objective="pancakes")
    assert title == "Pancake day"
    assert [(i.asset_id, i.source_start, i.source_end) for i in board.items] == [
        ("a_0", 0.8, 6.3), ("a_1", 16.0, 20.0), ("a_1", 9.8, 12.3)]  # shot = middle 4s of the window
    assert board.items[0].note == "intro: Tara na! Dito tayo kakain."
    assert board.items[1].note == "payoff: [food] pancakes"
    assert all(i.audio == "clip" for i in board.items)
    assert board.target_duration_s == round(board.duration_s, 1)      # no target, records the result
    # items point at whole-asset script-source segments, one per asset, reused on re-run
    seg_ids = {i.clip_id for i in board.items}
    write_script(conn, FakeLLM({"title": "", "beats": [{"label": "x", "order": [f"L{a}"]}]}),
                 objective="pancakes")
    assert {r["id"] for r in conn.execute("SELECT id FROM segments")} == seg_ids


def test_every_food_moment_gets_a_shot_even_when_the_model_skips_it(conn):
    a = _line(conn, "a_0", 1.0, 3.0, "Tara na!")
    z = _line(conn, "a_1", 28.0, 29.0, "Sarap!")
    chicken = _window(conn, "a_0", 5.0, 13.0, "fried chicken on a plate", ["fried chicken"])
    waffle = _window(conn, "a_1", 10.0, 18.0, "a waffle with butter", ["waffle"])
    _window(conn, "a_1", 18.0, 26.0, "A menu board with pasta", ["pasta"])  # never a dish to show
    judge = {"dishes": [{"dish": "chicken", "best": chicken}, {"dish": "waffle", "best": waffle}]}
    llm = FakeLLM(judge, {"title": "t", "beats": [{"label": "all", "order": [f"L{a}", f"S{chicken}", f"L{z}"]}]})
    board, _ = write_script(conn, llm, objective="o")   # the model forgot the waffle
    notes = [i.note for i in board.items]
    assert not any("pasta" in n for n in notes)
    # inserted in capture order: after take 1's chicken, before the take-2 line at 28s
    assert [("[food] fried chicken" in n, "[food] waffle" in n, n.endswith("Sarap!")) for n in notes] == [
        (False, False, False), (True, False, False), (False, True, False), (False, False, True)]


def test_script_requires_transcripts(conn):
    with pytest.raises(AutoVidError, match="transcribe"):
        write_script(conn, FakeLLM(), objective="o")


def test_old_v1_catalog_upgrades_in_place(tmp_path):
    import sqlite3
    p = tmp_path / "old.db"
    raw = sqlite3.connect(p)
    raw.executescript(db._SCHEMA.split("-- Speech-to-text")[0])  # a v1 catalog: no transcripts
    raw.execute("PRAGMA user_version = 1")
    raw.commit()
    raw.close()
    assert db.connect(p).execute("SELECT count(*) FROM transcripts").fetchone()[0] == 0


def test_render_conforms_clips_from_different_cameras(tmp_path):
    """Mixed size / frame rate / no audio track used to leave a 12 s timestamp gap in the output."""
    import subprocess

    from autovid.edl import EDL, EDLItem
    from autovid.ffmpeg_io.bins import ffmpeg_path, ffprobe_path
    from autovid.ffmpeg_io.probe import probe
    from autovid.render import render_edl

    def make(name, size, rate, audio):
        p = tmp_path / name
        cmd = [ffmpeg_path(), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration=6"]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:a", "aac", "-shortest"]
        subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
        return p

    c = db.connect(tmp_path / "m.db")
    items = []
    # the two 160x120 clips are the majority, so the odd 120x160 one (first!) is conformed to them
    for n, (name, size, rate, audio) in enumerate([("b.mp4", "120x160", 30, False), ("a.mp4", "160x120", 24, True),
                                                   ("c.mp4", "160x120", 24, True)]):
        p = make(name, size, rate, audio)
        c.execute("INSERT INTO assets (id, sha256, path, original_path, original_name, duration_s)"
                  " VALUES (?,?,?,?,?,6)", (f"a{n}", f"s{n}", str(p), str(p), name))
        seg = c.execute("INSERT INTO segments (asset_id, start_s, end_s) VALUES (?,0,6)", (f"a{n}",)).lastrowid
        items.append(EDLItem(clip_id=seg, asset_id=f"a{n}", source_start=1.0, source_end=4.0, audio="clip"))
    board = EDL(name="mix", items=items)
    out = render_edl(c, Settings(data_dir=tmp_path / "data"), board)

    pts = sorted(float(x) for x in subprocess.run(
        [ffprobe_path(), "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.split() if x)
    assert max(b - a for a, b in zip(pts, pts[1:])) < 0.2          # no gaps
    info = probe(out)
    assert abs(info.duration_s - 9.0) < 0.3 and info.has_audio and (info.width, info.height) == (160, 120)


def test_food_shot_never_replays_footage_the_speech_already_used(conn):
    # speech at 1.4-3.8s of take 1; the food window is 0-8s of the same take
    a = _line(conn, "a_0", 1.6, 3.6, "Ang sarap!")      # padded to 1.4-3.9
    food = _window(conn, "a_0", 0.0, 8.0, "crispy fried chicken", ["fried chicken"])
    llm = FakeLLM({"title": "t", "beats": [{"label": "b", "order": [f"S{food}", f"L{a}"]}]})
    board, _ = write_script(conn, llm, objective="o")
    speech = next(i for i in board.items if "[food]" not in i.note)
    shot = next(i for i in board.items if "[food]" in i.note)
    assert shot.source_end <= speech.source_start or shot.source_start >= speech.source_end
    assert shot.source_end - shot.source_start >= 2.0                # the free 3.9-8.0s stretch, not dropped


def test_food_shot_is_dropped_when_speech_leaves_no_room(conn):
    a = _line(conn, "a_0", 0.5, 7.8, "Grabe ang sarap talaga nito")   # covers the whole 8s window
    food = _window(conn, "a_0", 0.0, 8.0, "pancakes", ["pancakes"])
    llm = FakeLLM({"title": "t", "beats": [{"label": "b", "order": [f"L{a}", f"S{food}"]}]})
    board, _ = write_script(conn, llm, objective="o")
    assert [("[food]" in i.note) for i in board.items] == [False]


def test_lookalike_shots_of_one_scene_are_dropped_and_not_re_added(conn):
    w1 = _window(conn, "a_1", 0.0, 8.0, "friends eating", ["fried rice"])
    w2 = _window(conn, "a_1", 8.0, 16.0, "friends eating", ["fried rice", "fried fish"])
    w3 = _window(conn, "a_1", 16.0, 24.0, "friends eating", ["noodle dish"])
    a = _line(conn, "a_0", 1.0, 3.0, "Kain na")

    def fp(path, ts):
        return 0 if ts < 16 else (1 << 64) - 1                       # w1, w2 identical; w3 differs in all 64 bits

    judge = {"dishes": [{"dish": "rice", "best": w1}, {"dish": "fish", "best": w2}, {"dish": "noodles", "best": w3}]}
    llm = FakeLLM(judge, {"title": "t", "beats": [{"label": "b", "order": [f"L{a}", f"S{w1}", f"S{w2}", f"S{w3}"]}]})
    board, _ = write_script(conn, llm, objective="o", fingerprint=fp)
    shots = [i for i in board.items if "[food]" in i.note]
    assert [i.note.split("[food] ")[1] for i in shots] == ["fried rice", "noodle dish"]  # w2 dropped, w3 kept
    assert not any("fried fish" in i.note for i in board.items)     # step 3 does not bring the look-alike back


def test_speech_heavy_story_is_reasked_once_with_a_budget_and_the_lighter_one_wins(conn):
    a = _line(conn, "a_0", 1.0, 9.0, "Mahabang kwento tungkol sa ordering")     # 8.5s with padding
    b = _line(conn, "a_1", 1.0, 3.0, "Ang sarap!")
    food = _window(conn, "a_1", 10.0, 18.0, "plate of pancakes", ["pancakes"])
    heavy = {"title": "heavy", "beats": [{"label": "x", "order": [f"L{a}", f"S{food}", f"L{b}"]}]}
    light = {"title": "light", "beats": [{"label": "x", "order": [f"S{food}", f"L{b}"]}]}
    llm = FakeLLM(heavy, light)
    board, title = write_script(conn, llm, objective="o")
    assert title == "light" and len(llm.prompts) == 2
    assert "at most 6s" in llm.prompts[1]                                          # 1.5 x the 4s of food
    assert [("[food]" in i.note) for i in board.items] == [True, False]


def test_speech_clips_of_one_take_never_replay_each_others_padding(conn):
    a = _line(conn, "a_0", 1.0, 4.0, "Una")
    b = _line(conn, "a_0", 4.6, 7.0, "Ikalawa")      # 0.6s gap: not merged, but the pads (0.3 + 0.2) touch at 4.3-4.4
    c = _line(conn, "a_0", 4.5, 6.0, "Overlapping")  # whisper segments can overlap
    llm = FakeLLM({"title": "t", "beats": [{"label": "x", "order": [f"L{a}", f"L{c}", f"L{b}"]}]})
    board, _ = write_script(conn, llm, objective="o")
    spans = sorted((i.source_start, i.source_end) for i in board.items)
    assert all(e1 <= s2 + 1e-9 for (_, e1), (s2, _) in zip(spans, spans[1:]))


def test_each_dish_is_shown_once_from_the_shot_the_judge_calls_best(conn):
    close = _window(conn, "a_0", 5.0, 13.0, "close-up of pancakes with syrup", ["pancakes"])
    wide = _window(conn, "a_0", 13.0, 21.0, "pancakes on a far table", ["pancakes"])   # same dish, worse shot
    waffle = _window(conn, "a_1", 10.0, 18.0, "a waffle with butter", ["waffle"])
    a = _line(conn, "a_0", 1.0, 3.0, "Tara na!")
    judge = {"dishes": [{"dish": "pancakes", "best": close}, {"dish": "waffle", "best": waffle}]}
    llm = FakeLLM(judge, {"title": "t", "beats": [{"label": "b", "order": [f"L{a}"]}]})
    board, _ = write_script(conn, llm, objective="o")
    judge_parts = llm.prompts[0]                                    # the judge saw every candidate (as content parts)
    assert isinstance(judge_parts, list) and sum(f"S{x} (" in p.get("text", "") for p in judge_parts for x in (close, wide, waffle)) == 3
    assert f"S{wide} " not in llm.prompts[1]                       # the story never sees the worse angle
    shots = [i.note for i in board.items if "[food]" in i.note]
    assert sorted(n.split("[food] ")[1] for n in shots) == ["pancakes", "waffle"]
    assert not any(i.source_start >= 13.0 and i.asset_id == "a_0" and "[food]" in i.note for i in board.items)


def test_failed_dish_judge_falls_back_to_every_food_shot(conn):
    from autovid.vlm import VLMUnavailable

    class Flaky(FakeLLM):
        def ask_json(self, system, user):
            if "judge" in system:
                raise VLMUnavailable("down")
            return super().ask_json(system, user)

    w1 = _window(conn, "a_0", 5.0, 13.0, "pancakes", ["pancakes"])
    w2 = _window(conn, "a_1", 10.0, 18.0, "waffle", ["waffle"])
    a = _line(conn, "a_0", 1.0, 3.0, "Tara na!")
    board, _ = write_script(conn, Flaky({"title": "t", "beats": [{"label": "b", "order": [f"L{a}"]}]}), objective="o")
    assert sum("[food]" in i.note for i in board.items) == 2


def test_food_shot_is_dropped_when_only_a_sliver_of_its_window_is_free(conn):
    a = _line(conn, "a_0", 0.0, 3.8, "Pancake dool")        # speech takes 0-4.1 and 6.0-8.0 of the 8s window
    b = _line(conn, "a_0", 5.9, 7.9, "Spicy chicken")
    food = _window(conn, "a_0", 0.0, 8.0, "a plate of pancakes", ["pancakes"])   # only ~1.9s free between
    llm = FakeLLM({"title": "t", "beats": [{"label": "b", "order": [f"L{a}", f"S{food}", f"L{b}"]}]})
    board, _ = write_script(conn, llm, objective="o")
    assert not any("[food]" in i.note for i in board.items)


def test_a_timelapse_contributes_at_most_its_best_shot_and_is_labelled_for_the_judge(conn, monkeypatch):
    import autovid.script as script_mod

    monkeypatch.setattr(script_mod, "classify", lambda c: {"a_0": set(), "a_1": {"timelapse"}})
    plain = _window(conn, "a_0", 5.0, 13.0, "close-up pancakes", ["pancakes"])
    t1 = _window(conn, "a_1", 0.0, 8.0, "table", ["fried rice"], score=0.4)
    t2 = _window(conn, "a_1", 8.0, 16.0, "table", ["noodles"], score=0.9)       # the timelapse's best
    t3 = _window(conn, "a_1", 16.0, 24.0, "table", ["fish"], score=0.5)
    a = _line(conn, "a_0", 1.0, 3.0, "Tara na!")
    judge = {"dishes": [{"dish": d, "best": i} for d, i in
                        (("pancakes", plain), ("rice", t1), ("noodles", t2), ("fish", t3))]}
    llm = FakeLLM(judge, {"title": "t", "beats": [{"label": "b", "order": [f"L{a}"]}]})
    board, _ = write_script(conn, llm, objective="o")
    assert any("TIMELAPSE" in p.get("text", "") for p in llm.prompts[0])        # the judge was told
    food = [i for i in board.items if "[food]" in i.note]
    assert sorted(i.note.split("[food] ")[1] for i in food) == ["noodles", "pancakes"]
