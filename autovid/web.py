"""Minimal web review UI: catalog, review queue, EDL editor, render.

Binds to localhost only; no auth (like DecisionMCP's server — local tooling).
The UI is a single static page + JSON API; EDL edits are full-list replaces
(reorder/remove/trim happen client-side, then save), and human review
decisions land straight in the segments table.

The app surface also covers: settings (edits ``.env`` in place — the source
of truth pydantic-settings reads; secrets never round-trip), first-run setup
checks, and a background job runner so multi-minute pipeline stages report
live progress to the browser instead of blocking a request.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db
from .analyze import analyze_asset
from .config import Settings, get_settings
from .decisions import DecisionClient
from .edl import EDL, EDLItem, load as load_edl, save as save_edl
from .errors import AutoVidError
from .render import render_edl
from .vlm import VLMClient

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
REVIEW_STATUSES = ("auto", "approved", "rejected", "pending")


class EDLUpdate(BaseModel):
    """Body of the EDL edit endpoint (module level: FastAPI must resolve the hint)."""

    name: str | None = None
    objective: str | None = None
    style: str | None = None
    items: list[EDLItem]

class SettingsUpdate(BaseModel):
    values: dict[str, str]


class IngestParams(BaseModel):
    folder: str
    recursive: bool = True


class StageParams(BaseModel):
    asset_id: str | None = None


class MusicPlanParams(BaseModel):
    music_path: str
    objective: str = ""
    mode: str = "hybrid"
    style: str = "cinematic"
    name: str = "music-cut"
    render: bool = True


class Job:
    """One background pipeline run, polled by the browser."""

    def __init__(self, stage: str, params: dict) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.stage = stage
        self.params = params
        self.state = "running"  # running | done | error
        self.done = 0
        self.total = 0
        self.current = ""
        self.log: list[str] = []
        self.error: str | None = None
        self.started = time.time()

    def view(self) -> dict:
        return {
            "id": self.id, "stage": self.stage, "state": self.state,
            "done": self.done, "total": self.total, "current": self.current,
            "log": self.log[-8:], "error": self.error,
        }


class JobRunner:
    """In-process background jobs. One job at a time — the pipeline is
    inherently sequential per catalog, and stages write the same tables."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, stage: str, params: dict, target) -> Job:
        job = Job(stage, params)
        with self._lock:
            running = [j for j in self._jobs.values() if j.state == "running"]
            if running:
                raise HTTPException(409, f"job {running[0].id} ({running[0].stage}) "
                                         "is still running — wait for it first")
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._wrap, args=(job, target), daemon=True)
        thread.start()
        return job

    def _wrap(self, job: Job, target) -> None:
        try:
            target(job)
            job.state = "done"
        except Exception as e:  # noqa: BLE001 — surfaced to the browser
            logger.exception("job %s failed", job.id)
            job.state = "error"
            job.error = str(e)

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return job

    def list(self) -> list[dict]:
        return [j.view() for j in sorted(self._jobs.values(),
                                         key=lambda j: -j.started)]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="AutoVid review", docs_url=None, redoc_url=None)
    holder = {"settings": settings}  # refreshed after settings saves
    runner = JobRunner()

    def current_settings() -> Settings:
        return holder["settings"]

    def conn():
        return db.connect(holder["settings"].db_path)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # -- catalog -------------------------------------------------------------

    @app.get("/api/assets")
    def assets() -> list[dict]:
        c = conn()
        rows = c.execute(
            """SELECT a.id, a.original_name, a.status, a.duration_s, a.has_audio,
                      a.poster_path, a.dup_paths_json,
                      (SELECT COUNT(*) FROM segments g WHERE g.asset_id = a.id) AS segments,
                      (SELECT COUNT(*) FROM decisions d WHERE d.asset_id = a.id) AS decisions
               FROM assets a ORDER BY a.created_at"""
        ).fetchall()
        return [
            {
                "id": r["id"], "name": r["original_name"], "status": r["status"],
                "duration_s": r["duration_s"], "has_audio": bool(r["has_audio"]),
                "poster": r["poster_path"], "segments": r["segments"],
                "decisions": r["decisions"],
                "duplicates": json.loads(r["dup_paths_json"] or "[]"),
            }
            for r in rows
        ]

    @app.get("/api/segments")
    def segments(asset_id: str | None = None, q: str | None = None) -> list[dict]:
        c = conn()
        sql = (
            "SELECT s.*, a.original_name FROM segments s JOIN assets a ON a.id = s.asset_id "
            "WHERE 1=1"
        )
        args: list = []
        if asset_id:
            sql += " AND s.asset_id = ?"
            args.append(asset_id)
        if q:
            pattern = f"%{q.lower()}%"
            sql += " AND (lower(s.description) LIKE ? OR lower(s.tags_json) LIKE ?)"
            args += [pattern, pattern]
        sql += " ORDER BY s.highlight_score DESC"
        return [dict(r) for r in c.execute(sql, args).fetchall()]

    @app.post("/api/segments/{segment_id}/review")
    def review(segment_id: int, status: str) -> dict:
        if status not in REVIEW_STATUSES:
            raise HTTPException(400, f"status must be one of {REVIEW_STATUSES}")
        c = conn()
        cur = c.execute(
            "UPDATE segments SET review_status = ? WHERE id = ?", (status, segment_id)
        )
        c.commit()
        if cur.rowcount == 0:
            raise HTTPException(404, f"unknown segment {segment_id}")
        return {"segment_id": segment_id, "review_status": status}

    @app.get("/clips/{segment_id}")
    def clip(segment_id: int) -> FileResponse:
        c = conn()
        row = c.execute(
            "SELECT clip_path FROM segments WHERE id = ?", (segment_id,)
        ).fetchone()
        if row is None or not row["clip_path"]:
            raise HTTPException(404, f"segment {segment_id} has no clip")
        path = Path(row["clip_path"])
        if not path.exists():
            raise HTTPException(404, f"clip file missing: {path.name}")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    # -- EDL -----------------------------------------------------------------

    @app.get("/api/edls")
    def edls() -> list[dict]:
        c = conn()
        return [dict(r) for r in c.execute(
            "SELECT id, name, objective, style, target_duration_s, updated_at FROM edls "
            "ORDER BY updated_at DESC"
        ).fetchall()]

    @app.get("/api/edls/{edl_id}")
    def get_edl(edl_id: str) -> dict:
        try:
            return load_edl(conn(), edl_id).model_dump()
        except AutoVidError as e:
            raise HTTPException(404, str(e))

    @app.put("/api/edls/{edl_id}")
    def put_edl(edl_id: str, update: EDLUpdate) -> dict:
        c = conn()
        try:
            board = load_edl(c, edl_id)
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        if update.name is not None:
            board.name = update.name
        if update.objective is not None:
            board.objective = update.objective
        if update.style is not None:
            board.style = update.style
        board.items = update.items
        save_edl(c, board)
        return board.model_dump()

    @app.post("/api/edls")
    def make_edl(board: EDL) -> dict:
        save_edl(conn(), board)
        return board.model_dump()

    @app.post("/api/edls/{edl_id}/render")
    def render(edl_id: str) -> dict:
        c = conn()
        try:
            board = load_edl(c, edl_id)
            out = render_edl(c, current_settings(), board)
        except AutoVidError as e:
            raise HTTPException(409, str(e))
        return {"edl_id": edl_id, "path": str(out), "duration_s": board.duration_s}

    @app.get("/renders/{edl_id}/{name}")
    def rendered(edl_id: str, name: str) -> FileResponse:
        data_dir = current_settings().data_dir
        path = (data_dir / "renders" / edl_id / name).resolve()
        if not path.is_relative_to((data_dir / "renders").resolve()) or not path.exists():
            raise HTTPException(404, "render not found")
        return FileResponse(path, media_type="video/mp4", filename=name)

    # -- settings (edits .env in place) --------------------------------------

    @app.get("/api/settings")
    def get_api_settings() -> dict:
        from .settings_store import current_settings_view

        return current_settings_view(current_settings())

    @app.put("/api/settings")
    def put_api_settings(update: SettingsUpdate) -> dict:
        from .config import get_settings
        from .settings_store import EDITABLE_KEYS, upsert_env_values

        unknown = set(update.values) - EDITABLE_KEYS
        if unknown:
            raise HTTPException(400, f"not editable: {sorted(unknown)}")
        upsert_env_values(update.values)
        get_settings.cache_clear()  # next read re-parses .env
        holder["settings"] = get_settings()
        return {"saved": sorted(update.values), "restart_free": True}

    # -- first-run setup checks -----------------------------------------------

    @app.get("/api/setup")
    def setup_checks() -> dict:
        s = current_settings()
        import httpx

        from .ffmpeg_io.bins import ffmpeg_path

        checks: list[dict] = []
        try:
            ffmpeg_path()
            checks.append({"name": "ffmpeg", "ok": True, "detail": "found"})
        except Exception as e:  # noqa: BLE001
            checks.append({"name": "ffmpeg", "ok": False, "detail": str(e)[:160]})

        base = s.decisionmcp_url.rsplit("/mcp", 1)[0]
        try:
            r = httpx.get(f"{base}/health", timeout=2.0)
            body = r.json()
            checks.append({"name": "DecisionMCP", "ok": r.status_code == 200,
                           "detail": f"{body.get('tools', '?')} tools"})
        except Exception as e:  # noqa: BLE001
            checks.append({"name": "DecisionMCP", "ok": False,
                           "detail": f"unreachable ({s.decisionmcp_url}) — heuristic "
                                     "fallbacks stay active"})

        checks.append({"name": "VLM key", "ok": bool(s.vlm_api_key),
                       "detail": "configured" if s.vlm_api_key else
                                 "not set — windows will need manual review"})

        checks.append({"name": "Catalog", "ok": s.db_path.exists(),
                       "detail": str(s.db_path)})
        return {"checks": checks,
                "ok": all(c["ok"] for c in checks if c["name"] != "DecisionMCP")}

    # -- background pipeline jobs ----------------------------------------------

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return runner.list()

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict:
        return runner.get(job_id).view()

    @app.post("/api/pipeline/ingest")
    def start_ingest(params: IngestParams) -> dict:
        folder = Path(params.folder)
        if not folder.is_dir():
            raise HTTPException(400, f"not a folder: {params.folder}")

        def target(job: Job) -> None:
            s = current_settings()
            c = db.connect(s.db_path)
            from .ingest import ingest_folder

            report = ingest_folder(
                c, s, folder, recursive=params.recursive,
                progress=lambda p, i, n: (
                    setattr(job, "done", i), setattr(job, "total", n),
                    setattr(job, "current", Path(p).name)),
            )
            job.log.append(f"added {len(report.added)}, duplicates {len(report.duplicates)}, "
                           f"unchanged {len(report.skipped)}, failed {len(report.failed)}")

        return runner.start("ingest", params.model_dump(), target).view()

    def _stage_target(stage: str, asset_id: str | None):
        def target(job: Job) -> None:
            s = current_settings()
            c = db.connect(s.db_path)
            vlm = VLMClient(s)
            decisions = DecisionClient(s, c)
            from .refine import cut_asset
            from .sampling import pending_assets, sample_asset

            status_for = {"sample": "ingested", "analyze": "sampled",
                          "cut": "analyzed", "organize": "cut"}
            targets = [asset_id] if asset_id else [
                r["id"] for r in pending_assets(c, status_for[stage])]
            job.total = len(targets)
            for aid in targets:
                job.current = aid
                if stage == "sample":
                    sampling.sample_asset(c, s, aid)
                elif stage == "analyze":
                    analyze_asset(c, s, aid, vlm=vlm, decisions=decisions)
                elif stage == "cut":
                    cut_asset(c, s, aid, decisions=decisions)
                else:
                    from .organize import organize_asset

                    organize_asset(c, s, aid)
                job.done += 1
                job.log.append(f"{aid}: done")

        return target

    @app.post("/api/pipeline/{stage}")
    def start_stage(stage: str, params: StageParams) -> dict:
        if stage not in ("sample", "analyze", "cut", "organize"):
            raise HTTPException(400, "stage must be sample | analyze | cut | organize")
        if stage != "sample" and params.asset_id is None:
            count = conn().execute(
                "SELECT COUNT(*) AS n FROM assets WHERE status = ?",
                ({"sample": "ingested", "analyze": "sampled",
                  "cut": "analyzed", "organize": "cut"}[stage],),
            ).fetchone()["n"]
            if count == 0:
                raise HTTPException(409, f"no assets ready for {stage}")
        return runner.start(stage, params.model_dump(),
                            _stage_target(stage, params.asset_id)).view()

    # -- music ------------------------------------------------------------------

    @app.post("/api/music/plan")
    def music_plan(params: MusicPlanParams) -> dict:
        s = current_settings()
        c = conn()
        music_file = Path(params.music_path)
        if not music_file.exists():
            raise HTTPException(400, f"music file not found: {params.music_path}")
        from .music import analyze_music
        from .planner import plan_music_video

        try:
            analysis = analyze_music(music_file)
            decisions = DecisionClient(s, c)
            board = plan_music_video(
                c, s, analysis, objective=params.objective, mode=params.mode,
                style=params.style, name=params.name, decisions=decisions,
            )
        except (AutoVidError, ValueError) as e:
            raise HTTPException(409, str(e))
        out = {"edl_id": board.id, "items": len(board.items),
               "duration_s": board.duration_s, "tempo_bpm": analysis.tempo_bpm}
        if params.render:
            try:
                out["render_path"] = str(render_edl(c, s, board))
            except AutoVidError as e:
                out["render_error"] = str(e)
        return out

    return app


def serve(settings: Settings | None = None, host: str = "127.0.0.1", port: int = 8766) -> None:
    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
