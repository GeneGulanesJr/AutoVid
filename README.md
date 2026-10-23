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
