# The render-verify loop

`autovid check` lints the **plan**; `autovid verify` inspects the **render**.
Nothing in the pipeline watches the finished video — verify is the step that
does, and it is built so an agent (or a reviewer who won't scrub a 3-minute
MP4) can judge the result from PNGs and JSON alone.

## The loop

Run it after every render — and after every storyboard edit + re-render:

1. `autovid check <edl>` — the plan is linted first; a broken plan wastes the
   render. Non-zero exit stops here.
2. Render (`autovid render <edl>`, `run --objective …`, `musicvideo`, or the
   web job endpoint).
3. `autovid verify <edl>` — exit 0 = no defect found. Any `fail` stops the
   loop: fix the plan or the pipeline, re-render, start over.
4. Read the artifacts in `catalog/renders/<edl_id>/verify/`:
   - `sheet_item_NN.png` — one item's frames in a grid: shot order, whether a
     transition reads, caption legibility, is the food actually centered.
   - `sheet_overview.png` — the whole cut at a glance: rhythm, dead air,
     accidental black frames.
   - `waveform.png` — the audio with the cut marks on it: do the cuts land
     where the music moves?
5. Only then claim the render is good. A suspect frame deserves a full-size
   look — extract it and judge detail on the zoomed frame, never on the
   sheet's thumbnails:
   `ffmpeg -ss <t> -i <render.mp4> -frames:v 1 frame.png`

`verify --json` prints the machine-readable report on **stdout** (progress
goes to stderr): expected vs actual duration, per-item output windows, sheet
paths, audio numbers (LUFS / true peak / silences) and the findings list.
The exit code reflects only `fail`-level findings.

## What the findings mean

| level | meaning | examples |
|---|---|---|
| `fail` | the render diverges from the EDL — do not ship | render missing, no video stream, wrong duration |
| `warn` | worth a look; often legitimate | no visible cut (adjacent shots of one take), mid-shot pop, frozen stretch, clipping, mostly-silent item, `speed`/`timeline placement` the renderer ignores |
| `info` / `ok` | facts | stream properties, confirmed boundary, quiet stretch, no audio track |

Tolerances live as named constants at the top of `autovid/verify.py`
(`CUT_THRESHOLD`, `BOUNDARY_TOL_S`, `FREEZE_N`/`FREEZE_D`, LUFS band, …) —
tune there, not in call sites.

## Known limits

- The renderer stitches items back-to-back at 1.0×; `timeline_start`/`speed`
  on the EDL are plan-side only. Verify reports the divergence instead of
  pretending the file honors them.
- "No visible cut" is expected when consecutive items are adjacent stretches
  of one continuous take — the boundary warning is a reminder, not a defect.
- Freezes below `FREEZE_D` seconds and genuinely static b-roll surface as
  warnings; read the item's sheet before judging.
- Per-item loudness runs one small ffmpeg decode per item (`--no-audio` to
  skip when you only want structure and sheets).
