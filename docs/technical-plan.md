# AutoVid — Technical Implementation Plan

**AI-Powered Video Storyboard & Automatic Editing System · v1.0 · 2026-10-23**

This is the authoritative design document for AutoVid. The repository is not
greenfield — all nine roadmap phases below are implemented in `autovid/` and
validated (132 tests, live end-to-end runs; the Storyboard Agent, three-
artifact model, storyboard UI and EDL validator landed in the same PR as
this document). The plan marks every element as:

- **Established** — implemented + validated by tests and live E2E runs.
- **Benchmark required** — built, but its contribution is unproven at scale.

---

## 1. Executive Summary & Recommended Architecture

**Recommendation:** a **local-first, two-process architecture** — a
pipeline/application process (AutoVid) and a separate bounded
**decision-layer** process (DecisionMCP) — with three strictly separated
model roles and an immutable, stage-ledgered pipeline over a single SQLite
catalog.

| Element | Decision | Status |
|---|---|---|
| Pipeline host | Python 3.10 package `autovid`; FastAPI + pywebview desktop shell (`autovid app`), CLI parity (`autovid <stage>`) | Established |
| Decision layer | Separate repo **DecisionMCP**, streamable-HTTP MCP at `http://127.0.0.1:8765/mcp`; 21 tools, `d1-omni-tf` engine (LiquidAI/d1-omni-600M, transformers, CUDA/CPU) | Established |
| Primary vision model | OpenAI-compatible cloud VLM (`AUTOVID_VLM_*`, default Z.ai glm-4.6v); any local VLM server drop-in via the same interface | Established / local VLM to benchmark |
| Music analysis | librosa (beats, tempo, downbeat phase, RMS energy, energy sections) — deliberately **not** a model | Established |
| Catalog | Single SQLite DB (`catalog/`), `PRAGMA user_version` schema (v4) with in-place auto-migration, WAL + `synchronous=NORMAL` | Established |
| Media tools | ffmpeg/ffprobe 9.0.2, resolution chain `AUTOVID_FFMPEG_DIR` → `PATH` → vendored `.tools/ffmpeg*/bin` (winget is blocked by policy on this machine) | Established |
| Storyboard planning | Generative **Storyboard Agent** over a catalog digest (one repair round, then deterministic planner fallback); the deterministic planner also runs standalone | Established |
| UI | Setup / Pipeline / Catalog / Review / **Storyboard** / Music / EDL pages; scene cards, script editor, version history, approve→compile | Established |

**Why this shape:** the expensive understanding (cloud VLM) and the cheap
gating (local 600M model, one forward pass, no generated tokens) have
opposite cost/latency profiles and opposite failure modes; separating them
behind an MCP tool boundary lets either side fail without taking the
pipeline down (`AUTOVID_DECISION_MODE=auto` falls back to documented
heuristics; cold start with no VLM key still completes with pending
placeholder segments that a human approves). The single SQLite catalog keeps
the system one-folder portable and diffable; the app is localhost-only by
design.

**Key measured results so far** (Established): boundary-detection F1 = 1.00
against known cuts in both baseline and decision-guided modes; d1 correctly
accepted real cuts and flagged dead time; live VLM scoring honestly rated
synthetic patterns 0.1–0.2 (below the 0.4 highlight threshold); music E2E on
a 120 BPM click track produced a 5-cut plan over exactly 30 s rendered to
verified h264+aac. **Key open number** (Benchmark required): on 9 CPU-only
windows, decision-guided analysis cost 436 s vs 168 s baseline — d1 on CPU
is a correctness buy, not a speed buy, and its accuracy advantage on *real
ambiguous* footage is exactly what the bench harness must quantify before
its role expands.

---

## 2. Component Responsibilities & End-to-End Data Flow

### 2.1 Components

**AutoVid (this repo)**

| Module | Responsibility |
|---|---|
| `ingest.py` | Folder scan for supported containers; sha256 content-hash dedupe (flags same-content copies rather than deleting); ffprobe metadata; contact-sheet thumbnails; stable asset IDs; originals never touched |
| `sampling.py` / `ffmpeg_io/frames.py` | One shared decode per asset producing both scene-change detection (`scdet` scores) and fixed-interval grid frames + motion metrics; dense windows use input seeking with pts rebasing |
| `analyze.py` | 5-phase hierarchical segmentation (below); drives dense sampling from decision-layer output; writes per-window VLM analyses into the temporal map |
| `vlm.py` | OpenAI-compatible client; pooled HTTP client per run; structured-JSON window descriptions; per-window token accounting |
| `decisions.py` | MCP client for DecisionMCP; persistent session on a private background loop; decision modes `auto` / `d1` / `heuristic`; every decision logged to the `decisions` table with engine provenance |
| `refine.py` | Boundary verification/shift at suspected cuts (reuses nearest extracted frame within ±0.3 s before spawning ffmpeg) |
| `cut` stage | Segment acceptance → clip extraction with full source provenance |
| `organize.py` | Tags, quality flags, suggested filenames, recoverable rename journal |
| `planner.py` / `storyboard.py` / `edl.py` | Catalog → EDL: duration fitting (target ±1.2× tolerance), style pacing, music-mode section planning, alternative candidates; EDL v2 document model |
| `music.py` | librosa beat/downbeat/energy/sections; middle-window section audio for d1 mood |
| `render.py` | EDL validation → staged per-item clips (`*.part.mp4` + atomic rename, keyed by segment+trim provenance, reused across renders) → stream-copy concat, fade-only re-encode → music mux (`apad` + `-shortest`, 2 s afade out) |
| `web.py` / `app.py` | FastAPI on `127.0.0.1:8766`; JobRunner for long operations (single-flight, `409` if busy, last 50 kept); pywebview shell with browser fallback |
| `bench.py` | Ground-truth benchmark harness: baseline vs decision-guided, boundary P/R/F1 with tolerance, per-window token/cost logging |

**DecisionMCP (sibling repo)** — hosts d1-omni-600M behind bounded,
application-executable tools:

- `decision_video_region` → `sufficient | densify | inspect_more | flag_review`
- `decision_video_boundary` → `boundary | shift | not_boundary` (+ position hint)
- `decision_video_segment` → `accept | needs_evidence | reject | flag_review`
- `decision_video_continuity` → `same_scene | different_scene | needs_verification`
- `decision_music_mood` → `calm | neutral | energetic | driving` + energy + builds
- `decision_music_sequence` → `approve | replace_clip | slow_pace | fast_pace`

The application, never the model, executes the action. `predict /
predict_custom` carries base64 JPEG/audio; `require_vision()` gates tools
that need media. Integration tests opt in via `DECISIONMCP_INTEGRATION=1`;
default suites are mocked and hermetic.

### 2.2 Hierarchical temporal scene segmentation (the analysis core)

1. **Coarse windows** — pure time-based split, `AUTOVID_COARSE_WINDOW_S`
   (default 60 s). scdet is a hint, never the window definition.
2. **Multi-frame sampling** — grid frames per window (not a single frame),
   motion metrics attached.
3. **Window description** — cloud VLM writes a structured description per
   window (scene/activity, subjects, environment, shot type, highlights +
   confidence, tags, quality flags).
4. **Adjacent-window continuity** — `decision_video_continuity` compares
   neighbor pairs (text + optional end/start frames).
5. **Merge** — same-scene chains merge into logical scenes with real spans
   (`source='vlm-merge'`).
6. **Verification** — pairs with a scdet cut within 1.5 s get
   `decision_video_boundary` verification; confirmed cuts refine the shared
   edge; uncertain windows get densified via `decision_video_region`.
7. **Catalog** — scenes above `AUTOVID_HIGHLIGHT_THRESHOLD` (default 0.4)
   become segments; organize aggregates member windows per span.

**Scene-change vs. editing-opportunity distinction** is enforced
structurally: a camera-angle change without a continuity verdict stays
inside one logical scene; similar-looking footage with different activity
tags splits. The t=5 live incident — d1 refuted a real cut (conf 0.53) and
merged two look-alike colorful patterns — is the canonical example of why
boundary decisions stay **verifiable and benchmarked**, not trusted blindly.

### 2.3 End-to-end flow

```
footage folder
  → ingest: assets(+thumbs, hashes, metadata)          [catalog]
  → sample: windows + grid frames + scdet cuts          [catalog]
  → analyze: d1 gating ⇄ dense frames; VLM temporal map [catalog]
  → refine/merge: logical scenes (5-phase)              [catalog]
  → cut: accepted segments + extracted clips            [catalog + clips/]
  → organize: tags, names, review status                [catalog]
  → storyboard/planner: Script·Storyboard·EDL (§5)      [catalog]
  → render: staged clips → concat → (+music mux)        [renders/]
  → review UI / approval gate                           [HTTP 8766]
```

Every stage is ledgered and resumable: re-running a stage is idempotent
(analyses upsert per window), killed renders cannot poison later ones
(`*.part.mp4` + probe-failure re-extract), and incremental reanalysis
touches only changed assets — re-ingesting an unchanged folder
short-circuits on hash, and re-renders re-extract only the edited item.

---

## 3. Model Selection & Evaluation Strategy

### 3.1 Role table

| Role | Model | Runs | Contract | Status |
|---|---|---|---|---|
| Description/understanding | OpenAI-compatible VLM (default glm-4.6v) | Cloud | Structured JSON per window; image grids; `AUTOVID_VLM_MAX_TOKENS` default 4096 | Established |
| Bounded decisions | d1-omni-600M (`d1-omni-tf`) | Local (DecisionMCP) | Fixed action enums; one forward pass; no generated tokens | Established; accuracy-at-scale **benchmark required** |
| Storyboard Agent | Text-only generative LLM (same OpenAI-compatible config) | Cloud or local | Catalog digest in → Storyboard/Script JSON out; validator-gated (§5) | Established |
| Deterministic planner | `planner.py` rules | Local, no model | Always available; offline fallback; benchmark baseline | Established |
| Music timing | librosa | Local | Beats/downbeats/energy/sections as numbers, not model output | Established |
| Music mood / sequence review | d1-omni (audio + text) | Local | 16k mono PCM16, 30 s cap; mood labels; section approve/replace/slow/fast | Established |

### 3.2 Operational constraints discovered (must stay in the design)

- **Reasoning VLMs need token headroom.** glm-4.6v spent 561 of 600
  max_tokens thinking and truncated its JSON (`finish_reason: length`). The
  configurable max-token budget (default 4096) is a hard requirement for any
  GLM-4.6V-class model.
- **d1 media answers are uncalibrated by design** — the checkpoint skips
  temperature calibration for image/audio requests, so probabilities
  compress toward the middle. Consumers must *threshold and order*, never
  expect confident 0.9s from vision calls; tests assert direction, not exact
  labels.
- **d1 CPU latency:** preload ~24 s, text decision ~3.3 s, 1-frame vision
  ~13.6 s (Intel UHD, no NVIDIA). Accepted on this box because the same
  server/code moves to faster hardware unchanged; the accuracy-per-dollar
  verdict belongs to the bench harness.
- **Latency asymmetry:** ~30 s per VLM call dominates analysis wall time;
  the pooled-client, shared-decode, and input-seeking optimizations exist
  because ffmpeg decode + VLM round-trips — not Python arithmetic — are the
  bottleneck (2026-10-23 review conclusion).

### 3.3 Evaluation strategy (Benchmark required — ongoing)

`autovid bench truth.json` is the gate for every model-role claim:

- **Boundary quality:** precision/recall/F1 vs human ground truth with time
  tolerance. Established on synthetic/test cuts (F1 = 1.00); must be
  re-measured on real heterogeneous footage.
- **Guided vs baseline cost:** wall time, VLM tokens per window, d1 call
  count, per mode (`heuristic` = the conventional baseline of
  fixed-interval sampling + scdet; `d1` = adaptive). d1's role expands only
  where it beats baseline F1 by more than its cost penalty.
- **Highlight scoring honesty:** synthetic/dead content must stay under the
  0.4 threshold (verified: patterns 0.1–0.2, solid red 0.10).
- **Storyboard Agent eval (new, pending):** catalog-reference validity (0
  hallucinated asset IDs/timestamps — machine-checkable), duration
  adherence, coverage of highlights, and human edit distance (how much
  users change before approving).

---

## 4. Data Models & Structured Output Schemas

### 4.1 Catalog (SQLite, schema v4, versioned via `user_version`, in-place auto-migration)

| Table | Holds |
|---|---|
| `assets` | Stable ID, original path/filename, sha256, ffprobe metadata (codec/resolution/fps/duration/audio), thumbnail refs, dedupe status, processing state; indexed on `path` |
| `analyses` (windows) | Per-window VLM description JSON, frames refs, motion metrics, token usage; **upsert-per-window** (idempotent re-analysis); uniqueness enforced by migration-aware SQL |
| `scenes` | Logical scenes from phase-4 merge: real source spans, member windows, `source` provenance |
| `segments` | Accepted cut candidates: asset FK, in/out source timestamps, clip path, description, tags, score, confidence, review status (`pending/approved/rejected/flagged`) |
| `decisions` | Every d1/heuristic decision: tool, action, confidence, `details.model` engine provenance, `asset_id` (NULL for music-plan decisions), capped 4 KB `payload_json` |
| `edls` + items | EDL v2 documents (§4.3) |
| `jobs` | JobRunner state, results, pruning at 50 |
| music analyses | Per-track beats, tempo, downbeat phase, energy envelope, waveform, sections |
| rename journal | Recoverable original↔suggested filename mapping |

**Database recommendation:** stay on SQLite for the initial application —
single-file, WAL-safe, schema-versioned with a migration test suite that
builds a *pre-upgrade-shaped* DB and exercises new SQL on it (the v2→v3
migration bug — inline `UNIQUE` a no-op on existing tables — is the lesson
that produced this rule). Next indexing steps when catalogs grow: **FTS5**
virtual table over segment descriptions/tags for the `q` search (replacing
LIKE), and an optional embedding column with `sqlite-vec` for semantic
search — both local-first, no server. Postgres becomes justified only with
multi-user concurrent writes, which this app explicitly does not have.

### 4.2 Structured model outputs (JSON schemas, versioned)

- **WindowAnalysis** (VLM → catalog): `description`, `subjects[]`,
  `environment`, `shot_type`, `activities[]`,
  `highlights[] {ts, reason, score}`, `tags[]`, `quality_flags[]`,
  `confidence`.
- **DecisionResponse** (DecisionMCP → app): `tool`, `action` (enum per
  §2.1), `confidence`, `position_hint?`, `details.model`. The app maps
  action → operation; unknown actions fail closed to `flag_review`.
- **MusicAnalysis**: `tempo`, `beats[]`, `downbeats[]` (4/4 bar-phase
  approximation), `energy_envelope[]`, `sections[] {start, end, role}` where
  role ∈ intro/build/climax/outro by position and peak energy; plus d1
  `mood` enrichment.

### 4.3 The three artifacts (Script / Storyboard / EDL)

The spec's separation into three artifacts is the main item that was built
specifically for this document — implemented in `autovid/storyboard_doc.py`
alongside the agent (same PR). Model:

1. **Script** — narrative text: narration lines, dialogue, on-screen text,
   per story beat. No timeline data.
2. **Storyboard** — ordered scenes: `scene_id`, `narrative_role`, `title`,
   `description`, `purpose`, `selected_segment_refs[]`, `est_duration`,
   `music/pacing intent`, `transition intent`, `alternates[]` (other
   catalog segments), `open_issues[]`, `confidence`. References segments,
   not raw time.
3. **EDL (v2, established format)** — exact ranges: items carry
   `segment_id`, `source in/out`, `timeline_start/end`, `transition`
   (fade params), `beat_ref`, `speed`, `audio mode`; document carries
   `music_path`, `mode`, `duration_s` (prefers `timeline_end`).

**Propagation rule (implemented):** edits flow downward on request, never
silently — editing the Script marks dependent Storyboard scenes `stale` for
review; regenerating a Storyboard scene proposes a new version for approval.
The EDL is the single source of truth for rendering (compile refuses
out-of-asset ranges); the storyboard is the source of truth for intent. Both
are persisted; re-renders never require re-analysis.

---

## 5. Storyboard Generation & Editing Workflow

### 5.1 Established today (deterministic planner)

- Objective/duration/style in → score-ranked clip selection over accepted
  segments: no clip reuse until the pool is exhausted, tag-overlap dedupe,
  `duration ≥ slot` requirement, longest-clip-trim as last resort (never
  speed-changed footage).
- Music modes: `beat` (cuts snapped to beats), `cinematic`
  (structure-paced), `hybrid` (default: beat-snapped only above median
  energy). Section roles map to style-specific beats-per-cut
  (`STYLE_PACING`: intro 8–16, build 3–8, climax 2–4, outro 4–12), then d1
  mood enrichment adjusts (calm → ×2 and prefer longest; energetic/driving
  → ÷2, min 2).
- d1 reviews each section draft (`decision_music_sequence`); the plan is a
  JobRunner job; the EDL is fully editable and re-renderable, with duration
  tolerance ±1.2× verified in E2E.

### 5.2 The generative Storyboard Agent — implemented

- **Input:** a *catalog digest* (segments with tags/descriptions/scores/
  thumbnail refs, music sections, user objective/duration/style) — never
  raw video re-analysis.
- **Output:** the three artifacts of §4.3 as structured JSON.
- **Hallucination gate (hard requirement):** a validator rejects any
  storyboard referencing an asset/segment ID or timestamp the catalog
  doesn't support, returning the invalid refs to the agent for one bounded
  repair round, then failing to the deterministic planner. The agent may
  *request* missing shots ("no campfire-closeup in catalog") into a
  missing-footage list, never invent them.
- **Adaptivity:** no fixed narrative template — the agent receives music
  section roles and the catalog's actual structure and must fit the
  objective (montage / travel / vlog / tutorial / highlights / social); the
  deterministic planner covers the no-LLM and fallback paths.
- **Regeneration scopes:** single scene, section, or whole storyboard —
  each preserving user-locked items; version history per storyboard for
  recovery (§6).

---

## 6. Suggested Application Interface

**Established:** `autovid app` (pywebview shell; browser fallback;
`--make-launcher`) with pages **Setup** (first-run checks for
ffmpeg/DecisionMCP/VLM key; settings write `.env`, masked), **Pipeline**
(folder pick; sample→analyze→cut→organize with live progress), **Catalog**
(searchable segments; `/api/segments?asset_id&q&status&limit`), **Review**
(approve/reject flagged segments — also `autovid approve N`), **Music**
(track pick → beat-aligned plan/render), **EDL** (full-document edit via
`PUT /api/edls/{id}`, re-render). Long operations are jobs: POST returns a
job view, poll `GET /api/jobs/{id}`; renders stream at
`GET /renders/{edl_id}/{name}`, clips at `GET /clips/{segment_id}`.

**Implemented — storyboard-first board** (spec §6 → concrete controls):
thumbnail card grid per storyboard scene; clip preview per scene; reorder
(up/down); "swap with alternative" from the scene's `alternates[]`; trim
in/out per shot; inline narration/on-screen-text editing in the script
panel; objective/duration/style generation; scene-scoped or whole-board
regeneration with locked scenes preserved; missing-footage and open-issues
panel; **approve** toggle; restore from append-only version history.

---

## 7. Development Phases — Status, Acceptance Criteria, Risks

| # | Phase | Status | Acceptance criteria |
|---|---|---|---|
| 1 | Ingestion, metadata, thumbnails, catalog | **Done** (validated) | Discovery over folder; hashes dedupe; ffprobe rows; thumbs; originals untouched; resumable. ✅ E2E verified |
| 2 | Hierarchical scene segmentation + vision analysis | **Done** (validated) | 5-phase pipeline produces logical scenes with real spans; VLM maps per window; cold start degrades to pending placeholders. ✅ |
| 3 | D1 decision-layer evaluation | **Harness done; scale benchmark pending** | Boundary F1(guided) ≥ F1(baseline) on real footage AND guided cost within agreed budget; else d1 stays opt-in per decision type |
| 4 | Searchable scene catalog/library | **Done**; FTS5/embeddings optional next | Text search; status filters; API limits. Embeddings step: semantic search beats LIKE on a 20-query relevance set |
| 5 | Storyboard Agent + Script/Storyboard/EDL | **Done** (this PR) | 0 hallucinated refs on validation suite (validator-enforced); artifacts propagate per §4.3 rule; planner fallback path green |
| 6 | Storyboard review/editing UI | **Done** (this PR) | Storyboard tab with cards, reorder, swap, trim, lock, regenerate, approve, version restore; user edits survive scoped regeneration |
| 7 | Timeline compilation & EDL validation | **Done** (this PR) | Validator rejects invalid ranges, unknown segments, overlaps, mixed resolutions before render; warnings surface per item in UI/API; gaps + provenance drift warn |
| 8 | Music-driven pacing, stitching, rendering | **Done** (validated) | Beat-aligned plan on click track (✅ 30.08 s verified); staged-clip reuse; `*.part.mp4` atomicity; job-based render |
| 9 | Refinement, performance, quality evaluation | **First pass done** (PRs #3/#4/#5); ongoing | Bench suite green at 10× current footage volume; deferred perf items revisited only with new evidence |

**Dependencies:** 2←1, 3←2, 4←2, 5←4, 6←5, 7←5, 8←7, 9←all. With 5–7 done,
the remaining critical path is the scale benchmark (phase 3+9).

---

## 8. Major Risks, Trade-offs & Mitigations

| Risk | Mitigation | Status |
|---|---|---|
| d1 on CPU is slower than baseline (436 s vs 168 s / 9 windows) and made a genuine boundary mistake live (t=5 refutation, uncalibrated 0.53) | Decision modes let users run `heuristic`; every decision logged for measurement; role expansion gated on §3.3 benchmarks; hardware move is config-only | Managed |
| Reasoning VLM truncates JSON / costs | Token headroom default 4096 (`AUTOVID_VLM_MAX_TOKENS`); pooled client + backoff; per-window token accounting; key cleared from `.env` after live tests, tests hermetic (`_env_file=None`) | Established |
| ffmpeg 9 / Windows landmines (dead `fps` filter; `metadata=print` prefix pairing; `nullsink` for empty branches; input `-ss` pts rebase; output-extension muxer selection; UTF-16 BOMs; `mcp>=1.9,<2` pin; module-level Pydantic models) | Recipes centralized in `ffmpeg_io/frames.py` / `scenes.py`; gotcha list maintained; reuse over re-derivation | Established |
| Schema migration breaks upgraded catalogs (the v3 upsert crash) | Versioned schema + migration tests that build pre-upgrade-shaped DBs — now a standing rule for every schema change | Established rule |
| Storyboard Agent hallucinates footage | Catalog-reference validator + bounded repair + deterministic fallback (§5.2) | Established (validator-enforced) |
| Render cache poisoning / concurrency | Atomic `*.part.mp4` writes with probe-failure re-extract; per-thread SQLite connections; single-flight job runner (`409`), no queue | Established |
| Catalog scale (LIKE search, `/api/segments` default limit 1000) | FTS5 next; limit is explicit and documented; deferred items re-ranked only with evidence | Known |
| Staged-clip reuse vs source replaced in place with same length | Accepted deferred risk; document behavior; content-hash check if it bites | Known |
| Local inference vs cloud | Cloud VLM default (quality), local everything else (cost/privacy); any OpenAI-compatible local VLM server is drop-in — benchmark Qwen2.5-VL-class before switching | Standing |

---

## 9. Prioritized MVP Scope

**MVP-0 — achieved (this is the current system):** folder →
ingest/sample/analyze/cut/organize with d1-gated hierarchical segmentation;
searchable catalog; deterministic storyboard → editable EDL → rendered rough
cut; music beat-aligned mode; review UI with human approval; offline
fallbacks throughout; benchmark harness.

**MVP-1 — the next development slice, in priority order:**

1. ~~**Artifact separation + Storyboard Agent** (phase 5)~~ — **done in this
   PR**: three-artifact model, catalog validator, scoped regeneration,
   deterministic fallback.
2. ~~**Storyboard board UI** (phase 6)~~ — **done in this PR**: cards,
   reorder, swap-alternative, trim, lock, approve gate, version history.
3. ~~**EDL validator hardening** (phase 7)~~ — **done in this PR**:
   item-level error/warning reporting, render gate.
4. **Scale benchmark round** (phases 3+9): bench on real heterogeneous
   footage; decide d1's per-decision-type defaults from measured
   F1-vs-cost. *Harness validated 2026-10-23 on a synthetic heterogeneous
   fixture (docs/bench-synthetic-first-pass.md): boundary F1 = 1.00 for both
   modes, heuristic adaptivity cuts would-be VLM frames 61→46, d1 on CPU
   re-measured at ~13.8 s/window with no gain on unambiguous cuts. The
   real-footage round with human ground truth remains the gate.*
5. **Nice-to-have, explicitly unpromised:** audio transcription (local
   Whisper) as a segment metadata field; FTS5 search; embedding search;
   drag-and-drop reordering; music-summary input to the agent.

The system remains intentionally modular at each step: the Storyboard Agent
slots behind the existing planner interface, the UI upgrade rides the
existing FastAPI surface, and nothing above requires re-analysis of
already-cataloged footage.
