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
6. **storyboard** — objective/duration/style → editable EDL + rendered
   rough-cut preview.
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
segments), **Music** (pick a song → beat-aligned cut with the music as the
audio track), **EDL** (edit any plan and re-render). Without pywebview the
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
.venv/Scripts/autovid storyboard "60s hike recap" --duration 45   # EDL from accepted segments
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

## Web API

`autovid serve` (the same FastAPI app the app window runs; the window also
auto-starts DecisionMCP) exposes the UI and a JSON API on
`127.0.0.1:8766` — localhost only, no auth (local tooling, like
DecisionMCP's server). Catalog reads and edits are plain REST:
`/api/assets`, `/api/segments` (filters `asset_id`, `q`, `status`, `limit`),
`/api/edls`, `/api/settings`, `/api/setup`. EDL writes are full-document
replaces: `PUT /api/edls/{id}` and `POST /api/edls` require the complete
`items` list.

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
(`PRAGMA user_version`, currently v3) and an older catalog is migrated in
place on first connect — duplicate analysis rows collapse to the newest per
window.

Run tests: `.venv/Scripts/python -m pytest` (real ffmpeg on synthetic clips,
mocked model calls — no network, no API key needed).
