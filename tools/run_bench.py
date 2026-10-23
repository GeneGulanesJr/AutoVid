"""Run the boundary benchmark on the synthetic fixture (first pass).

Creates a throwaway catalog, ingests + samples tools/bench_fixture.py output,
then runs bench.run_benchmark twice against the same fixture:

- ``synthetic-heuristic`` — baseline vs the heuristic decision layer (no model).
- ``synthetic-d1``        — baseline vs the live d1-omni decision layer
  (requires the DecisionMCP server; skipped when it is down).

The VLM is deliberately unconfigured: windows go unanalyzed (flagged), which
keeps the run hermetic and free, while the boundary P/R/F1 metric — scdet +
decision layer — stays fully meaningful. Reports are copied into
``bench/<name>/`` in the repo.

    python tools/bench_fixture.py bench/fixture
    python tools/run_bench.py bench/fixture
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autovid import db  # noqa: E402
from autovid.bench import run_benchmark  # noqa: E402
from autovid.config import Settings  # noqa: E402
from autovid.decisions import DecisionClient  # noqa: E402
from autovid.ingest import ingest_folder  # noqa: E402
from autovid.sampling import sample_asset  # noqa: E402
from autovid.vlm import VLMClient  # noqa: E402


def main() -> None:
    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "bench" / "fixture"
    truth = json.loads((fixture / "truth.json").read_text(encoding="utf-8"))

    data_dir = Path(tempfile.mkdtemp(prefix="autovid-bench-cat-"))
    base = Settings(data_dir=data_dir, _env_file=None, vlm_api_key="",
                    decision_mode="heuristic")
    conn = db.connect(base.db_path)
    report = ingest_folder(conn, base, fixture, recursive=False)
    print(f"ingested {len(report.added)} assets, duplicates {len(report.duplicates)}")
    for row in conn.execute("SELECT id FROM assets").fetchall():
        sample_asset(conn, base, row["id"])

    d1_up = False
    try:
        import httpx

        d1_up = httpx.get("http://127.0.0.1:8765/health", timeout=2.0).status_code == 200
    except Exception:  # noqa: BLE001
        pass
    modes = ["heuristic"] + (["d1"] if d1_up else [])
    print(f"decision modes to run: {modes} (d1 server {'up' if d1_up else 'down'})")

    for mode in modes:
        s = base.model_copy(update={"coarse_window_s": 10.0, "decision_mode": mode})
        decisions = DecisionClient(s, conn)
        started = time.perf_counter()
        rep = run_benchmark(conn, s, vlm=VLMClient(s), decisions=decisions,
                            ground_truth=truth, name=f"synthetic-{mode}")
        decisions.close()
        print(f"[{mode}] done in {time.perf_counter() - started:.0f}s — "
              f"baseline F1={rep['baseline']['boundaries']['prf']['f1'] if rep['baseline']['boundaries']['prf'] else 'n/a'} "
              f"guided F1={rep['guided']['boundaries']['prf']['f1'] if rep['guided']['boundaries']['prf'] else 'n/a'}")
        out = REPO / "bench" / f"synthetic-{mode}"
        out.mkdir(parents=True, exist_ok=True)
        for f in ("report.json", "report.md"):
            shutil.copy2(data_dir / "bench" / f"synthetic-{mode}" / f, out / f)
        print(f"report → {out}")

    conn.close()
    shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
