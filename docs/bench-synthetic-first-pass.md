# Benchmark first pass — synthetic heterogeneous fixture (2026-10-23)

This is the **harness-validation round** for the plan's scale-benchmark gate
(technical plan §3.3, §9 item 4). It proves the benchmark pipeline end-to-end
on heterogeneous content and re-measures d1's cost profile; it does **not**
settle d1's accuracy question — that needs real, ambiguous footage.

Reproduce with:

```
python tools/bench_fixture.py bench/fixture   # 4 clips + truth.json (media gitignored)
python tools/run_bench.py bench/fixture       # heuristic + d1 rounds → bench/<mode>/
```

## Fixture

| clip | size | fps | audio | scenes | true cuts (s) |
|---|---|---|---|---|---|
| bench_a | 1280×720 | 30 | no | 3 | 8, 16 |
| bench_b | 1920×1080 | 24 | yes | 4 | 12, 24, 36 |
| bench_c | 480×270 | 15 | no | 2 | 10 |
| bench_d | 640×480 | 25 | no | 1 | — |

122 s total, 6 ground-truth cuts, coarse windows at 10 s (13 windows).
Boundary tolerance ±0.5 s.

## Results

**Heuristic decision layer** (`bench/synthetic-heuristic/report.md`):

| metric | baseline | guided (heuristic) |
|---|---|---|
| boundary P/R/F1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| VLM frames (next round) | 61 | **46** |
| dense frames added | 0 | 10 |
| wall time (s) | 3.7 | 6.5 |

**Live d1-omni decision layer** (`bench/synthetic-d1/report.md`, DecisionMCP
server up, 21 tools):

| metric | baseline | guided (d1) |
|---|---|---|
| boundary P/R/F1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| wall time (s) | 6.2 | **179.5** |

## Findings

1. **The harness works end-to-end** on mixed resolutions (480p–1080p), frame
   rates (15–30), audio/no-audio mixes, and 1–4-scene clips. Truth
   resolution, catalog fork, both modes, and P/R/F1 all behave.
2. **scdet alone is perfect on hard synthetic cuts** (F1 = 1.00, zero FP/FN
   across 6 cuts) — consistent with the plan's existing synthetic result.
   Synthetic content is the easy case by construction.
3. **Heuristic adaptivity pays for itself**: the guided round densified 10
   frames and would send ~25 % fewer VLM frames next analysis (61 → 46) at
   equal boundary accuracy, for ~2.8 s of extra local wall time.
4. **d1 on CPU re-measured at ~13.8 s/window** (179.5 s vs 6.2 s for the same
   13 windows) with **zero accuracy gain on unambiguous content** — matching
   the plan's earlier 436 s vs 168 s observation. d1's value can only show up
   where scdet is wrong or silent (gradual transitions, ambiguous boundaries),
   which synthetic hard cuts never exercise.

## Limitations / what the real round must add

- The VLM is deliberately unconfigured here (hermetic, no API spend): windows
  are flagged unanalyzed, so **segment acceptance is not measured** — only
  boundary quality and would-be VLM frame counts.
- All cuts are hard lavfi-source switches. The real round needs footage with
  gradual transitions, similar-looking scenes, and handheld noise — the cases
  where the plan expects d1 to beat baseline — plus human ground truth.
- Decision cost here includes no VLM latency; with a live VLM the baseline
  round's ~30 s/window call dominates and the densify-driven frame reduction
  (finding 3) becomes the cost lever.
