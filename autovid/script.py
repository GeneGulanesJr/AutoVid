"""Script stage: transcripts + scene descriptions → the *continuity* of a food vlog.

Output is an ordered EDL: the spoken lines and food shots, in story order. No
length target (the story decides) and no compositing (overlays, trims, music
belong to the editing stage). An LLM reads the noisy transcript and the scene
log in capture order, tells it as a small story that is mostly about the food,
and picks the lines and shots. Code then guarantees:

- each dish is shown once, from its single best shot (Haiku looks at the candidate
  frames and judges; no second angle or moment of the same dish), and
- nothing repeats: no stretch of footage plays twice (speech claims its time
  first, food shots take the free part of their window), and a shot whose frame
  is near-identical to an earlier shot of the same take is dropped (analysis
  windows of one continuous scene would otherwise yield a run of look-alikes).

shortcut: a food shot is a free 4 s slice of the analysis window that lists the
dish (the VLM says which dishes a window shows, not when); shorter analysis
windows (AUTOVID_COARSE_WINDOW_S=8) make this accurate enough.
Upgrade: per-frame dish timing.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from . import edl as edl_mod
from .edl import EDL, EDLItem
from .errors import AutoVidError
from .kinds import classify
from .media import b64_jpeg, data_uri
from .vlm import VLMClient, VLMUnavailable

logger = logging.getLogger(__name__)

MERGE_GAP_S = 1.5        # consecutive picks closer than this become one continuous clip
PAD_BEFORE_S = 0.2       # breathing room so words aren't clipped
PAD_AFTER_S = 0.3
SHOT_S = 4.0             # length of a food shot
MIN_SHOT_S = 1.5         # a free slice shorter than this is not worth a shot
MAX_CANDIDATES = 30       # food windows shown to the dish judge (best-quality first)
SPEECH_PER_FOOD = 1.5     # food focus: spoken seconds allowed per food second before one re-ask
SAME_SCENE_BITS = 18     # frame fingerprints (64 bits) closer than this = the same look
                         # (measured: one continuous scene 4-17, different scenes 22+)

_SYSTEM = """You are the story editor of a FOOD vlog. You get speech-to-text of raw footage \
(often Taglish; some lines are mis-heard or gibberish) plus scene descriptions, in capture \
order. Decide the continuity: which spoken lines and which food shots tell the story, and in \
what order. Someone else edits the pictures later; you only order the material.

The food is the star:
- Show EVERY food shot S you are given: each is already the single best shot of one dish (the \
editor adds any you skip), so there is nothing to choose between; only decide where each goes.
- Put most of the runtime on the food: arrival, close-ups, tasting and reactions. Spoken lines \
only where they carry the story (the plan, ordering, tasting reactions, the verdict): about a \
third of the runtime at most, never more than two or three spoken lines in a row without a food \
shot, and skip lines that just chat or repeat each other.
- Group the shots naturally (the dish ordered, then the dish arriving; or the table spread, then \
each dish), and follow a shot with the line that reacts to it when there is one.
- Never repeat yourself: each dish appears once.

Story shape: open on a hook (the most striking or funniest moment, ideally the best-looking \
dish), then brief setup (where we are, what we want), the outing and its small stakes, the \
payoff (the food and the reactions), and a closing line that resolves the story rather than \
repeating the intro. Keep chronological order except for the hook. No fixed length: include \
everything the story needs and nothing it does not (skip gibberish, filler, repeated lines, \
waiting chatter; never pick a menu board).

Spoken lines are marked L<id>; food shots (scenes showing a dish) are marked S<id>. Pick by id \
only; never invent ids. In a beat, list lines and shots in the order they should play.

Reply with JSON only: {"title": str, "beats": [{"label": str, "order": ["L12", "S40", ...]}, ...]}"""


class Beat(BaseModel):
    label: str
    order: list[str] = Field(default_factory=list)  # "L<id>" spoken line | "S<id>" food shot


class Script(BaseModel):
    title: str = ""
    beats: list[Beat]


Fingerprint = Callable[[str, float], "int | None"]


def _fingerprint(path: str, ts: float) -> int | None:
    """64-bit difference hash of the frame at ``ts`` (None when it cannot be read)."""
    from PIL import Image

    from .ffmpeg_io.frames import frame_at

    try:
        with tempfile.TemporaryDirectory() as tmp:
            img = Image.open(frame_at(path, ts, Path(tmp) / "f.jpg", width=96)).convert("L").resize((9, 8))
            px = list(img.getdata())
    except Exception as e:  # noqa: BLE001 — unreadable frame: treat as "not a repeat"
        logger.debug("no fingerprint for %s@%.1f: %s", path, ts, e)
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col + 1] > px[row * 9 + col])
    return bits


def _fmt_clock(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def _source_prompt(conn: sqlite3.Connection, keep: set[int] | None = None) -> tuple[str, dict[int, dict], dict[int, dict]]:
    """Footage log in capture order, transcript rows by id, food shots by id.

    ``keep``: only these food windows become S shots (the dish judge's picks); others stay plain scenes.
    """
    lines: dict[int, dict] = {}
    shots: dict[int, dict] = {}
    out: list[str] = []
    kinds = classify(conn)
    assets = conn.execute("SELECT id, original_name, path, duration_s FROM assets "
                          "ORDER BY original_name").fetchall()
    for n, a in enumerate(assets, 1):
        rows = conn.execute("SELECT * FROM transcripts WHERE asset_id = ? ORDER BY start_s",
                            (a["id"],)).fetchall()
        windows = conn.execute(
            "SELECT id, window_start, window_end, response_json FROM analyses "
            "WHERE asset_id = ? AND provider != 'none' ORDER BY window_start", (a["id"],)).fetchall()
        if not rows and not windows:
            continue
        out.append(f"\n## Take {n} ({a['original_name']}, {a['duration_s']:.0f}s)")
        events = []  # (time, text) so scenes, shots and speech interleave in time order
        for w in windows:
            j = json.loads(w["response_json"])
            desc = (j.get("description") or "")[:140]
            foods = [f.strip() for f in j.get("foods") or [] if f.strip()]
            if foods and "menu" not in desc.lower() and (keep is None or w["id"] in keep):
                quality = float(j.get("highlight_score") or 0) * (1 - float(j.get("uncertainty") or 0))
                shots[w["id"]] = {"id": w["id"], "asset_id": a["id"], "name": a["original_name"],
                                  "path": a["path"], "start": w["window_start"],
                                  "end": min(w["window_end"], a["duration_s"]), "foods": foods,
                                  "quality": quality, "timelapse": "timelapse" in kinds.get(a["id"], set())}
                events.append((w["window_start"], f"S{w['id']} @{_fmt_clock(w['window_start'])} "
                                                  f"[food: {', '.join(foods)}] {desc}"))
            else:
                events.append((w["window_start"], f"[scene @{_fmt_clock(w['window_start'])}] {desc}"))
        for r in rows:
            lines[r["id"]] = {**dict(r), "name": a["original_name"]}
            events.append((r["start_s"], f"L{r['id']} @{_fmt_clock(r['start_s'])}: {r['text']}"))
        out += [text for _, text in sorted(events, key=lambda e: e[0])]
    return "\n".join(out), lines, shots


_JUDGE_SYSTEM = """You judge candidate food shots for a food vlog. Each candidate is one image \
with an id (S<n>) and the dishes an earlier pass listed (noisy: the same plate can be named \
differently in each). Group the candidates by the DISTINCT DISH they actually show: different \
angles or moments of the same dish are one dish; a wide table scene with many plates is ONE \
"table spread" candidate, not one per plate. For each dish pick the single best shot: the dish is \
clearly visible and the subject, sharp, well lit, appetizing, not hidden by hands or people. A \
candidate marked TIMELAPSE is a time-compressed view of the whole meal: prefer another shot of a \
dish when there is one.

Reply with JSON only: {"dishes": [{"dish": str, "best": <S number>}, ...]}"""


def _pick_best_shots(vlm: VLMClient, shots: dict[int, dict]) -> set[int]:
    """Ids of the one best shot per distinct dish, judged by looking at each candidate's frame."""
    if len(shots) <= 1:
        return set(shots)
    from .ffmpeg_io.frames import frame_at

    candidates = sorted(shots.values(), key=lambda s: -s["quality"])[:MAX_CANDIDATES]
    parts: list[dict] = [{"type": "text", "text": "Candidates:"}]
    with tempfile.TemporaryDirectory() as tmp:
        for s in candidates:
            parts.append({"type": "text", "text": f"S{s['id']} (listed: {', '.join(s['foods'])}; "
                                                  f"{s['name']} @{_fmt_clock(s['start'])}"
                                                  f"{'; TIMELAPSE' if s['timelapse'] else ''})"})
            try:
                frame = frame_at(s["path"], (s["start"] + s["end"]) / 2, Path(tmp) / f"{s['id']}.jpg", width=512)
                parts.append({"type": "image_url", "image_url": {"url": data_uri(b64_jpeg(frame, max_width=512))}})
            except Exception as e:  # noqa: BLE001 — judge from the listing alone
                logger.warning("no frame for shot S%d: %s", s["id"], e)
    try:
        best = {int(d["best"]) for d in vlm.ask_json(_JUDGE_SYSTEM, parts).get("dishes", [])} & set(shots)
    except (VLMUnavailable, KeyError, TypeError, ValueError) as e:
        logger.warning("dish judge failed (%s) — keeping every food shot", e)
        best = set()
    best = best or set(shots)
    # a timelapse is one long, near-identical view of the table: at most its single best shot
    for asset_id in {shots[i]["asset_id"] for i in best if shots[i]["timelapse"]}:
        mine = sorted((i for i in best if shots[i]["asset_id"] == asset_id), key=lambda i: -shots[i]["quality"])
        best -= set(mine[1:])
    return best


def _ask(vlm: VLMClient, footage: str) -> Script:
    try:
        return Script.model_validate(vlm.ask_json(_SYSTEM, footage))
    except ValidationError as e:
        raise VLMUnavailable(f"script reply had the wrong shape: {e}") from e


def _picked(script: Script, lines: dict[int, dict], shots: dict[int, dict]) -> list[tuple[str, str, dict]]:
    """(beat label, 'line'|'shot', row|shot) in script order, unknown ids and repeats dropped."""
    seen: set[str] = set()
    picked: list[tuple[str, str, dict]] = []
    for beat in script.beats:
        for ref in beat.order:
            ref = ref.strip().upper()
            if ref in seen or len(ref) < 2 or not ref[1:].isdigit():
                continue
            kind, pool = ("line", lines) if ref[0] == "L" else ("shot", shots) if ref[0] == "S" else (None, {})
            if kind and int(ref[1:]) in pool:
                seen.add(ref)
                picked.append((beat.label, kind, pool[int(ref[1:])]))
    return picked


def _free_slice(window: tuple[float, float], blocked: list[tuple[float, float]]) -> tuple[float, float] | None:
    """The best SHOT_S-or-shorter slice of ``window`` that touches no ``blocked`` range, or None.

    Best = the longest free stretch (ties: closest to the window centre), centred in that stretch.
    """
    ws, we = window
    free, cursor = [], ws
    for bs, be in sorted(blocked):
        if be <= cursor or bs >= we:
            continue
        if bs > cursor:
            free.append((cursor, bs))
        cursor = max(cursor, be)
    if cursor < we:
        free.append((cursor, we))
    centre = (ws + we) / 2
    # a sliver squeezed between speech clips is not a shot: the dish is already on screen while they talk
    need = max(MIN_SHOT_S, min(SHOT_S, we - ws) / 2)
    free = [f for f in free if f[1] - f[0] >= need]
    if not free:
        return None
    s, e = max(free, key=lambda f: (min(f[1] - f[0], SHOT_S), -abs((f[0] + f[1]) / 2 - centre)))
    length = min(e - s, SHOT_S)
    start = min(max(centre - length / 2, s), e - length)
    return round(start, 3), round(start + length, 3)


def _cover_segment(conn: sqlite3.Connection, asset: sqlite3.Row) -> int:
    """A whole-asset segment the EDL can reference (render cuts sub-spans from the source).

    verdict 'script_source' keeps it out of the classic storyboard's candidate pool.
    """
    row = conn.execute("SELECT id FROM segments WHERE asset_id = ? AND verdict = 'script_source'",
                       (asset["id"],)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO segments (asset_id, start_s, end_s, description, verdict, review_status) "
        "VALUES (?, 0, ?, 'full footage (script source)', 'script_source', 'auto')",
        (asset["id"], asset["duration_s"]),
    )
    conn.commit()
    return int(cur.lastrowid)


def _build(
    conn: sqlite3.Connection, script: Script, lines: dict[int, dict], shots: dict[int, dict],
    assets: dict[str, sqlite3.Row], fingerprint: Fingerprint,
) -> tuple[list[EDLItem], float, float]:
    """Turn the LLM's story into EDL items; returns (items, speech seconds, food seconds)."""
    picked = _picked(script, lines, shots)
    if not picked:
        raise AutoVidError("the model picked no usable lines")

    # 1. Speech first: merge adjacent picks of one take into continuous clips, trimmed so no two
    #    spoken clips of a take replay the same moment (their padding can overlap).
    slots: list[dict] = []   # {"label", "kind": "line"|"shot", "key": (name, t), "item" | "shot"}
    prev: tuple[str, float] | None = None
    for label, kind, r in picked:
        if kind == "shot":
            slots.append({"label": label, "kind": "shot", "key": (r["name"], r["start"]), "shot": r})
            prev = None
            continue
        asset = assets[r["asset_id"]]
        if slots and slots[-1]["kind"] == "line" and prev and prev[0] == r["asset_id"] \
                and 0 <= r["start_s"] - prev[1] <= MERGE_GAP_S:
            item = slots[-1]["item"]
            item.source_end = round(min(asset["duration_s"], r["end_s"] + PAD_AFTER_S), 3)
            item.note += f" {r['text']}"
        else:
            start = max(0.0, r["start_s"] - PAD_BEFORE_S)
            end = min(asset["duration_s"], r["end_s"] + PAD_AFTER_S)
            for o in (s["item"] for s in slots if s["kind"] == "line" and s["item"].asset_id == r["asset_id"]):
                if start < o.source_end and end > o.source_start:       # overlaps an earlier clip
                    if start >= o.source_start:
                        start = o.source_end
                    else:
                        end = o.source_start
            if end - start >= 0.3:
                slots.append({"label": label, "kind": "line", "key": (r["name"], r["start_s"]), "item": EDLItem(
                    clip_id=_cover_segment(conn, asset), asset_id=r["asset_id"],
                    source_start=round(start, 3), source_end=round(end, 3),
                    note=f"{label}: {r['text']}", audio="clip")})
        prev = (r["asset_id"], r["end_s"])

    # 2. Food shots take what speech left free; look-alikes of an earlier shot are dropped.
    blocked: dict[str, list[tuple[float, float]]] = {}
    for s in slots:
        if s["kind"] == "line":
            blocked.setdefault(s["item"].asset_id, []).append((s["item"].source_start, s["item"].source_end))
    prints: dict[str, list[int]] = {}             # asset → fingerprints of kept shots

    def place(shot: dict) -> tuple[float, float] | None:
        """Fit ``shot`` (reserving its range) unless it overlaps used footage or repeats a kept look."""
        span = _free_slice((shot["start"], shot["end"]), blocked.get(shot["asset_id"], []))
        if span is None:
            return None
        fp = fingerprint(shot["path"], (span[0] + span[1]) / 2)
        if fp is not None and any(bin(fp ^ k).count("1") <= SAME_SCENE_BITS for k in prints.get(shot["asset_id"], [])):
            logger.info("dropped a look-alike shot (%s @%.0fs)", shot["name"], shot["start"])
            return None
        if fp is not None:
            prints.setdefault(shot["asset_id"], []).append(fp)
        blocked.setdefault(shot["asset_id"], []).append(span)
        return span

    kept: list[dict] = []
    for s in slots:
        if s["kind"] == "line":
            kept.append(s)
        elif (span := place(s["shot"])) is not None:
            s["span"] = span
            kept.append(s)
    slots = kept

    # 3. Every food moment the story skipped is added where it falls in capture order (best frames
    #    first, so a static scene keeps its best shot and the rest drop out as look-alikes).
    for shot in sorted(shots.values(), key=lambda x: -x["quality"]):
        if any(s["kind"] == "shot" and s["shot"] is shot for s in slots):
            continue
        if (span := place(shot)) is None:
            continue
        at = max((i + 1 for i, s in enumerate(slots) if s["key"] <= (shot["name"], shot["start"])),
                 default=min(1, len(slots)))
        slots.insert(at, {"label": slots[at - 1]["label"] if at else "Food", "kind": "shot",
                          "key": (shot["name"], shot["start"]), "shot": shot, "span": span})
        logger.info("added a food moment the story left out (%s @%.0fs)", shot["name"], shot["start"])

    items: list[EDLItem] = []
    for s in slots:
        if s["kind"] == "line":
            items.append(s["item"])
        else:
            r = s["shot"]
            items.append(EDLItem(clip_id=_cover_segment(conn, assets[r["asset_id"]]), asset_id=r["asset_id"],
                                 source_start=s["span"][0], source_end=s["span"][1],
                                 note=f"{s['label']}: [food] {', '.join(r['foods'])}", audio="clip"))
    food_s = sum(i.source_end - i.source_start for i in items if "[food]" in i.note)
    return items, sum(i.source_end - i.source_start for i in items) - food_s, food_s


def write_script(
    conn: sqlite3.Connection, vlm: VLMClient, *, objective: str, name: str = "script-cut",
    fingerprint: Fingerprint = _fingerprint,
) -> tuple[EDL, str]:
    """Ask the LLM for the story, turn it into an EDL; returns (EDL, title)."""
    footage, lines, shots = _source_prompt(conn)
    if not lines:
        raise AutoVidError("no transcripts in the catalog — run `transcribe` first")
    best = _pick_best_shots(vlm, shots)
    if best != set(shots):
        footage, lines, shots = _source_prompt(conn, keep=best)
    assets = {a["id"]: a for a in conn.execute("SELECT id, original_name, duration_s FROM assets")}
    prompt = f"Objective: {objective}\n{footage}"

    script = _ask(vlm, prompt)
    items, speech_s, food_s = _build(conn, script, lines, shots, assets, fingerprint)
    if food_s and speech_s > SPEECH_PER_FOOD * food_s:
        # Food focus is the point and the LLM drifts towards talk, so give it a measured budget once.
        budget = SPEECH_PER_FOOD * food_s
        logger.info("%.0fs of speech vs %.0fs of food — asking for a food-focused cut", speech_s, food_s)
        retry = _ask(vlm, f"{prompt}\n\nYour story has {speech_s:.0f}s of spoken lines but only {food_s:.0f}s "
                          f"of food. Make it food-focused: the spoken lines must total at most {budget:.0f}s. "
                          f"Keep the best lines (plan, ordering, reactions, verdict), cut the chatter, and "
                          f"keep every food shot.")
        items2, speech2, food2 = _build(conn, retry, lines, shots, assets, fingerprint)
        if speech2 / max(food2, 1) < speech_s / food_s:      # only accept an improvement
            script, items = retry, items2

    board = EDL(name=name, objective=objective, style="script", items=items)
    board.target_duration_s = round(board.duration_s, 1)  # no target: record what the story came to
    edl_mod.save(conn, board)
    return board, script.title
