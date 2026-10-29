"""Script stage: transcripts + scene descriptions → the *continuity* of a food vlog.

Output is an ordered EDL: the spoken lines and food shots, in story order. No
length target (the story decides) and no compositing (overlays, trims, music
belong to the editing stage). An LLM reads the noisy transcript and the scene
log in capture order, tells it as a small story that is mostly about the food,
and picks the lines and shots. A first call plans the story (premise, hook, threads that
need both halves, ending); the cut is checked against that plan and re-asked once if it
opens or ends wrongly or tells a thread by halves. Code then guarantees:

- each dish is shown once, from its single best shot (Haiku looks at the candidate
  frames and judges; no second angle or moment of the same dish), and
- nothing repeats: no stretch of footage plays twice (speech claims its time
  first, food shots take the free part of their window), and a shot whose frame
  is near-identical to an earlier shot of the same take is dropped (analysis
  windows of one continuous scene would otherwise yield a run of look-alikes), and
- a timelapse asset gives at most one food shot; with ``timelapse_montage`` it
  gives one continuous montage of up to half its own length instead (centred on
  its best window), falling back to the plain best shot when speech leaves no room.

Per-frame dish timing (``dish_moments``): the window analysis only lists which
dishes an 8 s window shows, not when, so after the dish judge picks the shots,
the 2 s grid frames of every kept food window are labelled by the VLM (one image
per call: which dishes are visible, which ONE dish is the subject, framing
close-up/wide), and each shot is cut around the first frame where its dish is
the subject — falling back to a frame that merely shows the dish (close-up
preferred) and to the plain window-centre slice when nothing matches. Cost:
about 240 single-image VLM calls per 8 minutes of food footage (~4 per 8 s
window — labelling stops once a frame makes the dish the subject).
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
FRAME_GRID_S = 2.0       # dish_moments: label one frame per grid step inside a food window
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
repeating the intro. Keep chronological order except for the hook. Within one take, list that \
take's lines and shots in the order they were captured (their @m:ss times ascending), unless it \
is the hook. No fixed length: include \
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


_SPINE_SYSTEM = """You plan the story of a vlog before it is cut. You get speech-to-text of raw \
footage (often Taglish; some lines are mis-heard) plus scene descriptions, in capture order. \
Spoken lines are marked L<id>, food shots S<id>. Do not cut anything yet; write the plan:

- premise: one sentence: what this video is about and why it is worth watching.
- acts: the outing from start to finish as 3 to 6 stages in order (where we are going, getting \
there, arriving, ordering, the food, the verdict...). refs = the 1 to 3 best ids that show that \
stage; only stages the footage really has.
- hook: up to 3 ids of the most striking moments to open on.
- threads: at most 4, the ones a viewer who was not there would be lost without: a remark and \
the reason for it, a plan and what came of it, a joke and its punchline, a dish and the story \
behind it. refs = only the few ids that are indispensable to it (usually 2 to 4), setup and \
payoff together; leave out chatter, repeats and garbled lines. Only threads whose pieces really \
are in the log.
- ending: up to 3 ids that can close the video as a payoff (the tasting, a reaction, the \
verdict, a line that resolves the story), never a line that only sets something up.

Pick by id only; never invent ids. Reply with JSON only: \
{"premise": str, "acts": [{"name": str, "refs": ["L3"]}], "hook": ["S4"], "threads": [{"name": str, "refs": ["L12", "S40"]}], "ending": ["L90"]}"""


class Thread(BaseModel):          # also used for an act: a named stage of the outing
    name: str = ""
    refs: list[str] = Field(default_factory=list)


class Spine(BaseModel):
    """The story plan the cut is checked against (all-empty = no rules, e.g. the plan call failed)."""
    premise: str = ""
    acts: list[Thread] = Field(default_factory=list)
    hook: list[str] = Field(default_factory=list)
    threads: list[Thread] = Field(default_factory=list)
    ending: list[str] = Field(default_factory=list)


Fingerprint = Callable[[str, float], "int | None"]


def _fingerprint(path: str, ts: float) -> int | None:
    """64-bit difference hash of the frame at ``ts`` (None when it cannot be read)."""
    from PIL import Image

    from .ffmpeg_io.frames import frame_at

    try:
        with tempfile.TemporaryDirectory() as tmp:
            _, frame = frame_at(path, ts, Path(tmp) / "f.jpg", width=96)
            img = Image.open(frame).convert("L").resize((9, 8))
            px = img.tobytes()          # raw 8-bit grayscale pixels, row-major (getdata is deprecated)
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
                frame = frame_at(s["path"], (s["start"] + s["end"]) / 2, Path(tmp) / f"{s['id']}.jpg", width=512)[1]
                parts.append({"type": "image_url", "image_url": {"url": data_uri(b64_jpeg(frame, max_width=512))}})
            except Exception as e:  # noqa: BLE001 — judge from the listing alone
                logger.warning("no frame for shot S%d: %s", s["id"], e)
    try:
        # the model answers "S64" as often as 64
        best = {int(str(d["best"]).lstrip("Ss")) for d in vlm.ask_json(_JUDGE_SYSTEM, parts).get("dishes", [])} & set(shots)
    except (VLMUnavailable, KeyError, TypeError, ValueError) as e:
        logger.warning("dish judge failed (%s) — keeping every food shot", e)
        best = set()
    best = best or set(shots)
    # a timelapse is one long, near-identical view of the table: at most its single best shot
    for asset_id in {shots[i]["asset_id"] for i in best if shots[i]["timelapse"]}:
        mine = sorted((i for i in best if shots[i]["asset_id"] == asset_id), key=lambda i: -shots[i]["quality"])
        best -= set(mine[1:])
    return best


_FRAME_SYSTEM = """You label one frame of a food-vlog shot. Say which dishes are visible in the \
image, which ONE dish is the subject of the frame (the one it is mostly about: clearly visible, \
in focus, framed — a wide table scene where every plate is equally small has no subject), and \
the framing (close-up = the dish fills the frame, wide = the dish sits in a bigger scene). The \
dish names an earlier pass listed for this shot are noisy: use your own words for what you see.

Reply with JSON only: {"dishes": [str], "subject": str | null, "framing": "close-up" | "wide"}"""


def _dish_moments(vlm: VLMClient, shots: dict[int, dict]) -> dict[int, float]:
    """Per-frame dish timing: {shot id: the moment its dish is the subject}.

    The window analysis only says which dishes a window shows, not when, so the 2 s grid
    frames of each window are labelled one image per ``ask_json`` call. A shot's moment is
    the first frame whose subject is its dish (labelling that window stops there); when no
    frame makes it the subject, the first frame that at least shows the dish — a close-up
    preferred — is the moment; when nothing matches, the shot id is omitted and the caller
    keeps the window-centre slice. Frames that cannot be extracted or labelled are skipped;
    once the VLM is unavailable, the remaining windows are not asked either.
    """
    if not shots:
        return {}
    from .ffmpeg_io.frames import frame_at

    def names(dish: str, text: str) -> bool:
        d, t = dish.strip().lower(), (text or "").strip().lower()
        return bool(d) and bool(t) and (d in t or t in d)   # noisy names: "pancakes" in "pancakes with syrup"

    moments: dict[int, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for sid, s in sorted(shots.items()):
            ws, we = s["start"], s["end"]
            grid, i = [], 0                                  # the 2 s grid, never at the exact window edge
            while ws + FRAME_GRID_S * i <= we:
                t = round(min(max(ws + FRAME_GRID_S * i, ws + 0.5), we - 0.5), 3)
                if not grid or t > grid[-1]:
                    grid.append(t)
                i += 1
            dish = s["foods"][0]
            fallback: tuple[float, bool] | None = None       # (ts, the dish fills the frame)
            try:
                for t in grid:
                    try:
                        frame = frame_at(s["path"], t, Path(tmp) / f"{sid}_{t:g}.jpg", width=512)[1]
                        reply = vlm.ask_json(_FRAME_SYSTEM, [
                            {"type": "text",
                             "text": f"Frame of shot S{sid} ({s['name']} @{_fmt_clock(t)}), "
                                     f"listed dish: {dish}."},
                            {"type": "image_url",
                             "image_url": {"url": data_uri(b64_jpeg(frame, max_width=512))}},
                        ])
                        subject = str(reply.get("subject") or "")
                        visible = any(names(dish, str(d)) for d in reply.get("dishes") or [])
                        close = "close" in str(reply.get("framing") or "").lower()
                    except VLMUnavailable:
                        raise
                    except Exception as e:  # noqa: BLE001 — no frame or a malformed reply: not the moment
                        logger.warning("no label for shot S%d @%.1fs: %s", sid, t, e)
                        continue
                    if names(dish, subject):
                        moments[sid] = t
                        break                                # the dish is the subject: the rest of the window cannot beat this
                    if visible and (fallback is None or (not fallback[1] and close)):
                        fallback = (t, close)
            except VLMUnavailable as e:
                logger.warning("frame labelling failed (%s) — keeping window-centre slices", e)
                break                                        # the service is down: stop asking about the rest
            if fallback is not None and sid not in moments:
                moments[sid] = fallback[0]
    return moments


def _ask(vlm: VLMClient, footage: str) -> Script:
    try:
        return Script.model_validate(vlm.ask_json(_SYSTEM, footage))
    except ValidationError as e:
        raise VLMUnavailable(f"script reply had the wrong shape: {e}") from e


def _reask(vlm: VLMClient, prompt: str) -> Script | None:
    """A second opinion on a cut already in hand: a malformed reply keeps the first cut."""
    try:
        return _ask(vlm, prompt)
    except VLMUnavailable as e:
        logger.warning("re-ask failed (%s) — keeping the first cut", e)
        return None


def _plan(vlm: VLMClient, prompt: str) -> Spine:
    """The story plan; a failed or malformed plan means no story rules, never a failed run."""
    try:
        return Spine.model_validate(vlm.ask_json(_SPINE_SYSTEM, prompt))
    except (VLMUnavailable, ValidationError) as e:
        logger.warning("story plan failed (%s) — cutting without story rules", e)
        return Spine()


def _plan_text(spine: Spine) -> str:
    """The plan as extra prompt lines for the story call."""
    if not (spine.acts or spine.hook or spine.threads or spine.ending):
        return ""
    out = ["\n\nStory plan (follow it):", f"- premise: {spine.premise}"]
    if spine.acts:
        out.append("- tell the whole outing in this order, with at least one of each stage's ids "
                   "(the food focus never cuts a stage): "
                   + "; ".join(f"{a.name} ({', '.join(a.refs)})" for a in spine.acts))
    if spine.hook:
        out.append(f"- open on one of: {', '.join(spine.hook)}")
    if spine.ending:
        out.append(f"- end on one of: {', '.join(spine.ending)}")
    out += [f"- thread \"{t.name}\" is all or nothing, keep every one of {', '.join(t.refs)} or none"
            for t in spine.threads if len(t.refs) > 1]
    return "\n".join(out)


def _story_problems(spine: Spine, script: Script, lines: dict[int, dict], shots: dict[int, dict]) -> list[str]:
    """Where the cut breaks the plan: wrong opening or ending, a thread told by halves."""
    def valid(refs: list[str]) -> list[str]:
        refs = [r.strip().upper() for r in refs]
        return [r for r in refs if len(r) > 1 and r[1:].isdigit()
                and int(r[1:]) in (lines if r[0] == "L" else shots if r[0] == "S" else {})]

    picks = [("L" if kind == "line" else "S") + str(r["id"]) for _, kind, r in _picked(script, lines, shots)]
    if not picks:
        return []
    out = []
    if (hook := valid(spine.hook)) and picks[0] not in hook:
        out.append(f"open on one of {', '.join(hook)}, not {picks[0]}")
    if (end := valid(spine.ending)) and picks[-1] not in end:
        out.append(f"end on one of {', '.join(end)}, not {picks[-1]}")
    have = set(picks) | {f"S{i}" for i in shots}      # a skipped food shot is added back later anyway
    for a in spine.acts:
        refs = valid(a.refs)
        if refs and not set(refs) & have:
            out.append(f"stage \"{a.name}\" is missing: add one of {', '.join(refs)}")
    for t in spine.threads:
        refs = valid(t.refs)
        missing = [r for r in refs if r not in have]
        if missing and len(missing) < len(refs):
            out.append(f"thread \"{t.name}\" is told by halves: add {', '.join(missing)}, or remove the rest of it")
    return out


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


def _free_slice(window: tuple[float, float], blocked: list[tuple[float, float]],
                cap: float = SHOT_S, centre: float | None = None) -> tuple[float, float] | None:
    """The best ``cap``-or-shorter slice of ``window`` that touches no ``blocked`` range, or None.

    Best = the longest free stretch (ties: closest to ``centre`` — the window centre, or a
    per-frame dish moment), centred in that stretch.
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
    c = (ws + we) / 2 if centre is None else min(max(centre, ws), we)
    # a sliver squeezed between speech clips is not a shot: the dish is already on screen while they talk
    need = max(MIN_SHOT_S, min(cap, we - ws) / 2)
    free = [f for f in free if f[1] - f[0] >= need]
    if not free:
        return None
    s, e = max(free, key=lambda f: (min(f[1] - f[0], cap), -abs((f[0] + f[1]) / 2 - c)))
    length = min(e - s, cap)
    start = min(max(c - length / 2, s), e - length)
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
    assets: dict[str, sqlite3.Row], fingerprint: Fingerprint, *, timelapse_montage: bool = False,
    moments: dict[int, float] | None = None,
) -> tuple[list[EDLItem], float, float]:
    """Turn the LLM's story into EDL items; returns (items, speech seconds, food seconds).

    ``moments``: {shot id: ts} from per-frame dish timing — a shot with a moment is placed
    centred on it instead of on its window centre (same free/replay/look-alike guards).
    """
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
        span = _free_slice((shot["start"], shot["end"]), blocked.get(shot["asset_id"], []),
                           cap=shot.get("cap", SHOT_S),
                           centre=None if moments is None else moments.get(shot["id"]))
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

    # Timelapse option: one continuous montage of up to half the asset, centred on the asset's
    # best food window, instead of the single best shot. Placed before the plain shots so it
    # claims its span first; if speech leaves less than half the span free (or the asset is too
    # short for a real montage), the plain best shot below plays instead.
    montages: list[dict] = []
    if timelapse_montage:
        best_tl: dict[str, dict] = {}
        for shot in shots.values():
            if shot["timelapse"]:
                shot["skip"] = False                       # rebuilds must not inherit an earlier verdict
                if shot["asset_id"] not in best_tl or shot["quality"] > best_tl[shot["asset_id"]]["quality"]:
                    best_tl[shot["asset_id"]] = shot
        for asset_id, b in best_tl.items():
            dur = float(assets[asset_id]["duration_s"] or 0.0)
            centre = (b["start"] + b["end"]) / 2
            length = min(dur / 2, 2 * min(centre, dur - centre))
            if length < SHOT_S:
                continue
            montage = {**b, "start": max(0.0, centre - length / 2), "end": min(dur, centre + length / 2),
                       "cap": length}
            if (span := place(montage)) is not None:
                b["skip"] = True                           # the montage replaces the single best shot
                montages.append({"key": (b["name"], b["start"]), "shot": montage, "span": span})

    kept: list[dict] = []
    for s in slots:
        if s["kind"] == "line":
            kept.append(s)
        elif not s["shot"].get("skip") and (span := place(s["shot"])) is not None:
            s["span"] = span
            kept.append(s)
    slots = kept
    for m in montages:                                     # capture order, like the shots added below
        at = max((i + 1 for i, s in enumerate(slots) if s["key"] <= m["key"]),
                 default=min(1, len(slots)))
        slots.insert(at, {"label": slots[at - 1]["label"] if at else "Food", "kind": "shot",
                          "key": m["key"], "shot": m["shot"], "span": m["span"]})

    # 3. Every food moment the story skipped is added where it falls in capture order (best frames
    #    first, so a static scene keeps its best shot and the rest drop out as look-alikes).
    for shot in sorted(shots.values(), key=lambda x: -x["quality"]):
        if shot.get("skip"):
            continue
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
    fingerprint: Fingerprint = _fingerprint, timelapse_montage: bool = False,
    dish_moments: bool = False,
) -> tuple[EDL, str]:
    """Ask the LLM for the story, turn it into an EDL; returns (EDL, title).

    ``timelapse_montage``: a timelapse asset contributes one continuous span of up to half its
    own length (centred on its best food window) instead of the single best shot.

    ``dish_moments``: after the judge's picks, label the 2 s grid frames of every kept food
    window with the VLM (about 240 single-image calls per 8 minutes of food footage) and cut
    each shot around the frame where its dish is the subject; shots with no matching frame
    keep the window-centre slice. Off by default.
    """
    footage, lines, shots = _source_prompt(conn)
    if not lines:
        raise AutoVidError("no transcripts in the catalog — run `transcribe` first")
    best = _pick_best_shots(vlm, shots)
    if best != set(shots):
        footage, lines, shots = _source_prompt(conn, keep=best)
    moments = _dish_moments(vlm, shots) if dish_moments and shots else {}
    assets = {a["id"]: a for a in conn.execute("SELECT id, original_name, duration_s FROM assets")}
    prompt = f"Objective: {objective}\n{footage}"
    spine = _plan(vlm, prompt)
    prompt += _plan_text(spine)

    script = _ask(vlm, prompt)
    items, speech_s, food_s = _build(conn, script, lines, shots, assets, fingerprint,
                                     timelapse_montage=timelapse_montage, moments=moments)
    if food_s and speech_s > SPEECH_PER_FOOD * food_s:
        # Food focus is the point and the LLM drifts towards talk, so give it a measured budget once.
        budget = SPEECH_PER_FOOD * food_s
        logger.info("%.0fs of speech vs %.0fs of food — asking for a food-focused cut", speech_s, food_s)
        retry = _ask(vlm, f"{prompt}\n\nYour story has {speech_s:.0f}s of spoken lines but only {food_s:.0f}s "
                          f"of food. Make it food-focused: the spoken lines must total at most {budget:.0f}s. "
                          f"Keep the best lines (plan, ordering, reactions, verdict), cut the chatter, and "
                          f"keep every food shot.")
        items2, speech2, food2 = _build(conn, retry, lines, shots, assets, fingerprint,
                                        timelapse_montage=timelapse_montage, moments=moments)
        if speech2 / max(food2, 1) < speech_s / food_s:      # only accept an improvement
            script, items = retry, items2

    board = EDL(name=name, objective=objective, style="script", items=items)
    board.target_duration_s = round(board.duration_s, 1)  # no target: record what the story came to
    edl_mod.save(conn, board)
    return board, script.title
