# AutoVid

AI-powered automatic video editor: point it at a folder of raw footage, and it
ingests, understands, organizes and rough-cuts the material with minimal
manual intervention.

Pipeline (each stage is resumable and logged; originals are never modified):

1. **ingest** — discover videos, dedupe by content hash, store ffprobe
   metadata, generate contact-sheet thumbnails.
2. **sample** — ffmpeg scene-change detection + fixed-interval grid frames
   with motion metrics.
3. **analyze** — per-window decisions from the local decision layer
   (DecisionMCP / d1-omni) drive adaptive frame extraction; a cloud VLM
   describes each window into a structured temporal map.
4. **cut** — D1-verified scene boundaries, candidate segments accepted by the
   decision layer, clips extracted with full source provenance.
5. **organize** — tag segments, flag quality, suggest recoverable filenames.
6. **storyboard** — objective/duration/style → script + storyboard (LLM agent
   with a deterministic planner fallback) → compiled, editable EDL.
7. **review** — minimal web UI for the catalog and the EDL.

## The app

Non-technical entry point — one command, one window:

```bash
.venv/Scripts/pip install -e ".[app]"   # pywebview (native window shell)
.venv/Scripts/autovid app               # opens the AutoVid window
```

The window has every step: **Setup** (first-run checks for ffmpeg /
DecisionMCP / VLM key, and a settings page that writes your `.env` — paste
your API key there, values are masked after saving), **Pipeline** (pick a
footage folder, run sample → analyze → cut → organize with live progress),
**Catalog** (searchable segments), **Review** (approve/reject flagged
segments), **Storyboard** (plan Script · Storyboard · EDL: generate, scene
cards with trim/lock/swap/reorder, script editor, version history, approve,
compile to an EDL), **Music** (pick a song → beat-aligned cut with the music
as the audio track), **EDL** (edit any plan, validate it, re-render).
Without pywebview the
app falls back to your browser automatically; `autovid app --browser` forces
that. `autovid app --make-launcher` writes a double-clickable `AutoVid.bat`
to your Desktop (`%USERPROFILE%\Desktop`, assuming the repo's `.venv`).

## Quick start (CLI)

Everything the app does is also scriptable:

```bash
py -3.10 -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
copy .env.example .env        # set AUTOVID_VLM_API_KEY (cloud VLM)

.venv/Scripts/autovid ingest C:/path/to/footage
.venv/Scripts/autovid sample --all
.venv/Scripts/autovid analyze --all
.venv/Scripts/autovid cut --all
.venv/Scripts/autovid status
```

Then turn the catalog into cuts:

```bash
.venv/Scripts/autovid agent "60s hike recap" --duration 45    # script + storyboard (agent, planner fallback) → EDL
.venv/Scripts/autovid storyboard "60s hike recap" --duration 45   # EDL straight from accepted segments
.venv/Scripts/autovid boards                                  # list saved storyboards
.venv/Scripts/autovid render <edl_id>                            # rough-cut preview MP4
.venv/Scripts/autovid musicvideo C:/path/to/song.mp3              # beat-aligned cut to music
.venv/Scripts/autovid search "campfire"                          # text search over segments
.venv/Scripts/autovid approve 42                                 # human review verdicts
.venv/Scripts/autovid bench truth.json                           # baseline vs decision-guided
```

ffmpeg/ffprobe are resolved from `AUTOVID_FFMPEG_DIR`, then `PATH`, then the
vendored static build in `.tools/ffmpeg*/bin`.

The decision layer is [DecisionMCP](../DecisionMCP) (`DECISIONMCP_URL`,
default `http://127.0.0.1:8765/mcp`): d1-omni answers bounded questions
(sampling policy, boundary verification, clip acceptance) in one forward pass
per call, while the cloud VLM does the expensive descriptive work. With
`AUTOVID_DECISION_MODE=auto` the pipeline keeps running on documented
heuristics when the server is down; every decision is logged to the
`decisions` table either way, so decision-guided processing stays measurable
against the fixed-interval baseline.

## Storyboards: Script · Storyboard · EDL

`autovid agent` (or the **Storyboard** tab) plans an edit as three logically
separate artifacts:

- **Script** — narration, dialogue and on-screen text per story beat.
- **Storyboard** — ordered scenes referencing catalog segments, with
  alternative shots, open issues and confidence.
- **EDL** — compiled from the storyboard: exact ranges and timeline
  positions.

The agent is a text LLM (`AUTOVID_AGENT_MODEL`, defaulting to the VLM
endpoint) that sees only a digest of the segment catalog — it cannot invent
footage. A draft referencing unknown segments or beats gets exactly one
repair round, then the deterministic planner takes over (the storyboard
records its origin either way). Every save is an append-only version and
`restore` brings an old version back as a new one; editing the script marks
the scenes illustrating a changed beat *stale*; scenes can be **locked** so
a full regeneration keeps them at their positions. **Compile** turns a
storyboard into an editable EDL — re-compiles update the same EDL id, so
render history and the staged-clip cache survive — and rendering refuses an
EDL with validation errors (unknown segments, bad ranges, timeline overlaps,
mixed resolutions; `✓ Validate` in the EDL tab lists them item by item).

## Web API

`autovid serve` (the same FastAPI app the app window runs; the window also
auto-starts DecisionMCP) exposes the UI and a JSON API on
`127.0.0.1:8766` — localhost only, no auth (local tooling, like
DecisionMCP's server). Catalog reads and edits are plain REST:
`/api/assets`, `/api/segments` (filters `asset_id`, `q`, `status`, `limit`),
`/api/edls`, `/api/settings`, `/api/setup`. EDL writes are full-document
replaces: `PUT /api/edls/{id}` and `POST /api/edls` require the complete
`items` list, and `GET /api/edls/{id}/validate` returns item-level issues
(errors block rendering). Storyboards live at `/api/storyboards`:
`POST /api/storyboards/generate` (job; planner fallback when no VLM key),
`GET/PUT /api/storyboards/{id}` (PUT is a full-document replace, validated
against the catalog), `PUT /api/storyboards/{id}/script` (marks dependent
scenes stale), `POST …/regenerate` (job; `{"scene_id": …}` for a single
scene — locked scenes survive a full regeneration), `POST …/compile`,
`POST …/approve`, `POST …/restore`.

Anything that takes minutes — pipeline stages, EDL renders, music planning —
runs as a background **job**: the POST returns a job view immediately; poll
`GET /api/jobs/{id}` until `state` is `done` (payload in `result`) or
`error` (message in `error`). One job runs at a time (the pipeline is
sequential per catalog), a POST that arrives while another job is running is
rejected with `409` — jobs are not queued — and the last 50 jobs are kept.
`POST /api/edls/{id}/render` and `POST /api/music/plan` are job endpoints —
they used to block and return the finished payload, so external callers must
poll now. Job results carry local file paths; the server also serves renders
at `GET /renders/{edl_id}/{name}` and segment clips at
`GET /clips/{segment_id}`.

## Performance notes

The pipeline is ffmpeg-bound, and the fast paths are built around that:

- One decode serves both contact-sheet grids and scene-change detection
  (split filtergraph); dense sampling windows seek the input instead of
  decoding from 0.
- Renders concat with stream copy; a clip re-encodes only when it carries a
  fade (music renders also re-encode the audio track). Re-renders
  reuse staged per-item clips in `catalog/renders/<edl_id>/` (named by
  segment + trim provenance), so a one-item edit re-extracts that item, not
  the whole timeline — those `item_*.mp4` files are the cache, not garbage;
  deleting a render dir drops it.
- SQLite runs WAL + `synchronous=NORMAL` with per-thread cached connections
  in the web app; analysis rows are upserted, so re-analyzing a window is
  idempotent.
- The decision layer keeps one persistent MCP session and the VLM one pooled
  HTTP client per pipeline run, reused across every call in that run.
- CLI commands import numpy/PIL/librosa only when the invoked command needs
  them.

Catalogs upgrade themselves: the SQLite schema is versioned
(`PRAGMA user_version`, currently v4) and an older catalog is migrated in
place on first connect — duplicate analysis rows collapse to the newest per
window, and pre-v4 catalogs gain the storyboard tables plus
`edls.storyboard_id`.

Run tests: `.venv/Scripts/python -m pytest` (real ffmpeg on synthetic clips,
mocked model calls — no network, no API key needed).

## Speech-to-text (transcribe + script stages)

`autovid transcribe --all` fills the `transcripts` table; `autovid script "<objective>" --duration 60`
then has the LLM pick the spoken lines that tell the story (speech-only EDL, render as usual).

Backends (`AUTOVID_WHISPER_BACKEND`, default `auto`):

- **whispercpp** (AMD/Intel/NVIDIA GPU via Vulkan). Put a Windows Vulkan build of whisper.cpp at
  `~/.whisper-vulkan/bin/whisper-cli.exe` (e.g. `whisper-*-windows-vulkan-x64.zip` from
  lemonade-sdk/whisper.cpp-rocm releases), plus `models/ggml-large-v3.bin`
  (huggingface.co/ggerganov/whisper.cpp) and `models/ggml-silero-v5.1.2.bin`
  (huggingface.co/ggml-org/whisper-vad). Override the folder with `AUTOVID_WHISPERCPP_DIR`.
  A warning is logged if whisper.cpp ran on the CPU instead of the GPU.
- **faster-whisper** (CPU, or NVIDIA CUDA). Used when the whisper.cpp executable is absent.

## One command: footage folder → story

```
autovid run "C:\path\to\footage" --objective "a food vlog: a day at the mall, then breakfast"
```

Runs ingest → sample → transcribe → analyze, then (with `--objective`) writes the story, lints it and renders a
preview. The catalog lives in `catalog/<folder name>` (override with `--data-dir`); re-running only does what is
missing, so you can add footage and run it again (`--force` redoes transcribe and analyze). It first makes one real
call to your VLM, so a bad key, base URL, model or parameter stops the run immediately with the server's own reason
(the Settings page has the same **Test connection** button).

- `autovid check [edl]` lints an EDL: footage that plays twice, food shots that look the same, tiny clips, food
  share of the runtime, a timelapse used too much, mixed formats. Exits non-zero on a defect. `script` and `run`
  print it automatically.
- `autovid analyze --force` re-analyzes assets already analyzed (replacing, never duplicating, their rows).
- Clips up to 10 minutes are analyzed in 8 s windows (`AUTOVID_FINE_WINDOW_S`), so a dish or reaction is placed to
  seconds, not a minute; longer footage keeps the 60 s windows.
- Each clip is tagged from its metadata and transcript: `speech` (skips the visual highlight gate in `cut`),
  `timelapse` (at most one food shot), `other_camera` and `no_audio` (the renderer conforms them so clips from
  different cameras join cleanly).
- The desktop app has the same stages (Transcribe, Story) and settings.
