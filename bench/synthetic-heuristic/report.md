# AutoVid benchmark: synthetic-heuristic

Decision mode: `heuristic` · tolerance ±0.5s · assets: a_339f246e4864a670, a_53509738aea427d9, a_6e7d1fee38a055d5, a_2b66d8c047868f99

| metric | baseline (fixed + scdet) | guided (decision layer) |
| --- | --- | --- |
| VLM frames sent | 61 | 46 |
| VLM calls | 13 | 13 |
| windows | 13 | 13 |
| dense frames added | 0 | 10 |
| flagged windows | 13 | 13 |
| boundaries kept | 6 | 6 |
| boundary P/R/F1 | P=1.00 R=1.00 F1=1.00 (tp=6 fp=0 fn=0) | P=1.00 R=1.00 F1=1.00 (tp=6 fp=0 fn=0) |
| accepted segments | 0 | 0 |
| wall time (s) | 3.73 | 6.51 |

Lower VLM frames at equal-or-better F1 means the decision layer is paying for itself; every guided decision is logged in the decisions table.
