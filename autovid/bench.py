"""Benchmark: conventional baseline vs decision-guided analysis.

Re-answers the plan's core question on the current catalog: does the D1
decision layer improve boundary accuracy and analysis efficiency compared
with fixed-interval sampling + raw scene-change detection?

Both modes run against a throwaway fork of the catalog (DB + frames copied to
a temp dir), so the live catalog is never touched:

- **baseline** — no adaptive sampling (every region "sufficient"), boundaries
  kept exactly where scdet put them, segments accepted as-is.
- **guided** — the real decision layer (heuristics or d1 per DECISION_MODE)
  drives densification, boundary verification/shift and segment acceptance.

Metrics per mode: VLM frames sent, windows, dense frames added, flagged
windows, boundary precision/recall/F1 against a hand-labeled ground truth,
accepted segments, wall time. Report lands in ``catalog/bench/<name>/``.
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import db
from .analyze import analyze_asset
from .config import Settings
from .decisions import (
    BoundaryVerdict,
    DecisionClient,
    RegionAction,
    RegionStats,
    SegmentVerdict,
)
from .refine import cut_asset
from .vlm import VLMClient

logger = logging.getLogger(__name__)

BOUNDARY_TOLERANCE_S = 0.5


# ---------------------------------------------------------------------------
# Instrumented wrappers (count what the VLM saw; freeze decisions for baseline)
# ---------------------------------------------------------------------------


class RecordingVLM:
    """Proxy around any VLMClient that counts frames per call."""

    def __init__(self, inner: VLMClient) -> None:
        self._inner = inner
        self.calls = 0
        self.frames = 0
        self.unavailable = 0

    @property
    def configured(self) -> bool:
        return self._inner.configured

    @property
    def usage(self) -> list[dict[str, int]]:
        return self._inner.usage

    def describe_window(self, **kwargs) -> object:
        frames = kwargs.get("frames") or []
        self.calls += 1
        self.frames += len(frames)
        try:
            return self._inner.describe_window(**kwargs)
        except Exception:  # noqa: BLE001 — count outages, keep benchmarking
            self.unavailable += 1
            raise


class NullDecisions(DecisionClient):
    """The conventional pipeline: no adaptivity at all."""

    def __init__(self) -> None:
        super().__init__(Settings(decision_mode="heuristic"), None)

    def decide_region(self, stats: RegionStats) -> RegionAction:
        return RegionAction("sufficient", None, 0.5, 0, 0.0, "baseline", 0.0)

    def decide_boundary(self, asset_id, context, before, after) -> BoundaryVerdict:
        return BoundaryVerdict("boundary", "at", 1.0, 0.0, "baseline", 0.0)

    def decide_segment(self, asset_id, summary, frames, duration_s=None,
                       min_duration_s=0.5) -> SegmentVerdict:
        return SegmentVerdict("accept", 0.5, 0.5, 1, 0.0, "baseline", 0.0)


# ---------------------------------------------------------------------------
# Catalog fork (isolated sandbox per benchmark run)
# ---------------------------------------------------------------------------


def _fork_catalog(settings: Settings) -> tuple[Settings, Path]:
    fork_dir = Path(tempfile.mkdtemp(prefix="autovid-bench-"))
    fork_db = fork_dir / "autovid.db"
    # Backup API, not file copy: captures WAL-buffered writes from live connections.
    src = sqlite3.connect(str(settings.db_path))
    dst = sqlite3.connect(str(fork_db))
    src.backup(dst)
    dst.close()
    src.close()
    if settings.assets_dir.exists():
        shutil.copytree(settings.assets_dir, fork_dir / "assets")
    return settings.model_copy(update={"data_dir": fork_dir}), fork_dir


def _reset_derived(conn: sqlite3.Connection, fork: Settings) -> None:
    """Back to the 'ingested' state so a mode re-samples from scratch."""
    conn.executescript(
        """
        DELETE FROM analyses;
        DELETE FROM segments;
        DELETE FROM decisions;
        DELETE FROM scenes;
        DELETE FROM frames WHERE kind != 'thumb';
        DELETE FROM jobs;
        UPDATE assets SET status = 'ingested';
        """
    )
    conn.commit()
    if fork.assets_dir.exists():
        shutil.rmtree(fork.assets_dir)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def boundary_prf(pred: list[float], truth: list[float], tol: float = BOUNDARY_TOLERANCE_S
                 ) -> dict[str, float | int]:
    """Greedy one-to-one matching within ``tol`` seconds."""
    remaining = list(truth)
    tp = 0
    for p in sorted(pred):
        match = min((abs(p - t), t) for t in remaining) if remaining else None
        if match and match[0] <= tol:
            tp += 1
            remaining.remove(match[1])
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


@dataclass
class ModeResult:
    mode: str
    windows: int = 0
    vlm_frames: int = 0
    vlm_calls: int = 0
    vlm_outages: int = 0
    dense_frames: int = 0
    flagged: int = 0
    accepted_segments: int = 0
    boundaries: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


def _run_mode(
    mode: str,
    fork: Settings,
    asset_ids: list[str],
    vlm: VLMClient,
    decisions: DecisionClient | None,
) -> ModeResult:
    from .sampling import sample_asset  # local: re-sampling resets detector output

    conn = db.connect(fork.db_path)
    recording = RecordingVLM(vlm)
    client = NullDecisions() if decisions is None else decisions
    result = ModeResult(mode=mode)
    started = time.perf_counter()
    boundaries: list[float] = []
    accepted = 0

    for asset_id in asset_ids:
        _reset_derived(conn, fork)
        sample_asset(conn, fork, asset_id)  # fresh scdet boundaries + grid frames
        outcome = analyze_asset(conn, fork, asset_id, vlm=recording, decisions=client)
        result.windows += outcome.windows
        result.dense_frames += outcome.dense_frames
        result.flagged += outcome.flagged
        if decisions is not None:
            cut_asset(conn, fork, asset_id, decisions=client)
        else:  # baseline: raw scdet boundaries, all candidates accepted
            for row in conn.execute(
                "SELECT window_start, window_end, response_json FROM analyses "
                "WHERE asset_id = ?",
                (asset_id,),
            ).fetchall():
                data = json.loads(row["response_json"])
                if data.get("error"):
                    continue
                conn.execute(
                    """INSERT INTO segments (asset_id, start_s, end_s, description,
                                             highlight_score, verdict, decision_confidence,
                                             review_status)
                       VALUES (?, ?, ?, '', ?, 'accept', 0, 'auto')""",
                    (asset_id, row["window_start"], row["window_end"],
                     float(data.get("highlight_score") or 0.0)),
                )
            conn.commit()

        boundaries.extend(
            round(r["start_s"], 3)
            for r in conn.execute(
                "SELECT start_s FROM scenes WHERE source LIKE 'scdet%'"
            ).fetchall()
        )
        accepted += conn.execute(
            "SELECT COUNT(*) AS n FROM segments WHERE verdict = 'accept'"
        ).fetchone()["n"]

    result.vlm_calls = recording.calls
    result.vlm_frames = recording.frames
    result.vlm_outages = recording.unavailable
    result.elapsed_s = round(time.perf_counter() - started, 2)
    result.boundaries = {"count": len(boundaries), "timestamps": sorted(boundaries)}
    result.accepted_segments = accepted
    conn.close()
    return result


def run_benchmark(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    vlm: VLMClient,
    decisions: DecisionClient | None = None,
    ground_truth: dict[str, list[float]],
    name: str = "bench",
) -> dict:
    """Run both modes on a forked catalog; write report to ``catalog/bench/<name>``.

    ``ground_truth`` maps asset id (or original filename) → true boundary
    timestamps in seconds. Filenames are resolved against the catalog.
    """
    ids = [
        r["id"] for r in conn.execute(
            "SELECT id FROM assets WHERE status IN ('sampled','analyzed','cut','organized') "
            "ORDER BY created_at"
        ).fetchall()
    ]
    if not ids:
        raise ValueError("nothing to benchmark: ingest and sample an asset first")

    by_name: dict[str, str] = {}
    for r in conn.execute(
        "SELECT id, original_name, dup_paths_json FROM assets"
    ).fetchall():
        by_name[r["original_name"]] = r["id"]
        try:
            for dup in json.loads(r["dup_paths_json"] or "[]"):
                by_name[Path(dup).name] = r["id"]
        except ValueError:
            pass
    truth: dict[str, list[float]] = {}
    for key, stamps in ground_truth.items():
        asset_id = by_name.get(key, key)
        truth[asset_id] = [float(t) for t in stamps]

    fork, fork_dir = _fork_catalog(settings)
    try:
        baseline = _run_mode("baseline", fork, ids, vlm, decisions=None)
        guided = _run_mode("guided", fork, ids, vlm, decisions=decisions)
    finally:
        shutil.rmtree(fork_dir, ignore_errors=True)

    for mode in (baseline, guided):
        truth_all = [t for aid in ids for t in truth.get(aid, [])]
        pred = mode.boundaries.get("timestamps", [])
        mode.boundaries["prf"] = boundary_prf(pred, truth_all) if truth_all else None

    report = {
        "name": name,
        "assets": ids,
        "boundary_tolerance_s": BOUNDARY_TOLERANCE_S,
        "decision_mode": settings.decision_mode,
        "baseline": baseline.__dict__ | {"boundaries": baseline.boundaries},
        "guided": guided.__dict__ | {"boundaries": guided.boundaries},
    }

    out_dir = settings.data_dir / "bench" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict) -> str:
    b, g = report["baseline"], report["guided"]

    def prf(mode: dict) -> str:
        prf_ = mode["boundaries"].get("prf")
        if not prf_:
            return "n/a (no ground truth)"
        return (f"P={prf_['precision']:.2f} R={prf_['recall']:.2f} "
                f"F1={prf_['f1']:.2f} (tp={prf_['tp']} fp={prf_['fp']} fn={prf_['fn']})")

    lines = [
        f"# AutoVid benchmark: {report['name']}",
        "",
        f"Decision mode: `{report['decision_mode']}` · tolerance "
        f"±{report['boundary_tolerance_s']}s · assets: {', '.join(report['assets'])}",
        "",
        "| metric | baseline (fixed + scdet) | guided (decision layer) |",
        "| --- | --- | --- |",
        f"| VLM frames sent | {b['vlm_frames']} | {g['vlm_frames']} |",
        f"| VLM calls | {b['vlm_calls']} | {g['vlm_calls']} |",
        f"| windows | {b['windows']} | {g['windows']} |",
        f"| dense frames added | {b['dense_frames']} | {g['dense_frames']} |",
        f"| flagged windows | {b['flagged']} | {g['flagged']} |",
        f"| boundaries kept | {b['boundaries']['count']} | {g['boundaries']['count']} |",
        f"| boundary P/R/F1 | {prf(b)} | {prf(g)} |",
        f"| accepted segments | {b['accepted_segments']} | {g['accepted_segments']} |",
        f"| wall time (s) | {b['elapsed_s']} | {g['elapsed_s']} |",
        "",
        "Lower VLM frames at equal-or-better F1 means the decision layer is paying "
        "for itself; every guided decision is logged in the decisions table.",
    ]
    return "\n".join(lines) + "\n"
