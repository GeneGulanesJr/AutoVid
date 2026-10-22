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
5. **storyboard** — objective/duration/style → editable EDL + rendered
   rough-cut preview.
6. **review** — minimal web UI for the catalog and the EDL.

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
audio track), **EDL** (edit any plan and re-render). Without pywebview, use
`autovid app --browser`.

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

Run tests: `.venv/Scripts/python -m pytest` (real ffmpeg on synthetic clips,
mocked model calls — no network, no API key needed).
