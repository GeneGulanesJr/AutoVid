"""Minimal web review UI: catalog, review queue, EDL editor, render.

Binds to localhost only; no auth (like DecisionMCP's server — local tooling).
A middleware rejects any request whose ``Host`` header is not localhost, so a
browser tab cannot reach the API via DNS rebinding. The UI is a single static
page + JSON API; EDL edits are full-list replaces (reorder/remove/trim happen
client-side, then save), and human review decisions land straight in the
segments table.

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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import agent as agent_mod
from . import db
from . import storyboard_doc as doc_mod
from .analyze import analyze_asset
from .config import Settings, get_settings
from .decisions import DecisionClient
from .edl import EDL, EDLItem, load as load_edl, save as save_edl, validate_edl
from .errors import AutoVidError
from .render import render_edl
from .storyboard_doc import ScriptBeat, ScriptDocument, StoryboardScene
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


class ModelsRequest(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""   # used by /api/vlm/test only


class ScriptParams(BaseModel):
    objective: str
    name: str = "script-cut"
    render: bool = False


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


class StoryboardGenerateParams(BaseModel):
    objective: str
    target_duration_s: float = 30.0
    style: str = "cinematic"
    name: str = "storyboard"


class StoryboardUpdate(BaseModel):
    """Full-document replace of the current storyboard's scenes."""

    scenes: list[StoryboardScene]
    notes: str | None = None
    missing_footage: list[str] | None = None


class ScriptUpdate(BaseModel):
    title: str | None = None
    synopsis: str | None = None
    beats: list[ScriptBeat]


class RegenerateParams(BaseModel):
    scene_id: str | None = None  # None = regenerate the whole board


class RestoreParams(BaseModel):
    version: int


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
        self.result: dict = {}  # endpoint-specific payload, set when done
        self.started = time.time()

    def view(self) -> dict:
        return {
            "id": self.id, "stage": self.stage, "state": self.state,
            "done": self.done, "total": self.total, "current": self.current,
            "log": self.log[-8:], "error": self.error, "result": self.result,
        }


class JobRunner:
    """In-process background jobs. One job at a time — the pipeline is
    inherently sequential per catalog, and stages write the same tables."""

    MAX_JOBS = 50  # long-lived desktop process: don't grow the history forever

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
            if len(self._jobs) > self.MAX_JOBS:
                by_age = sorted(self._jobs.values(), key=lambda j: j.started)
                for old in by_age[: len(self._jobs) - self.MAX_JOBS]:
                    if old.state != "running":
                        self._jobs.pop(old.id, None)
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
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: -j.started)
        return [j.view() for j in jobs]


# The API is unauthenticated; anything answering a non-loopback Host header is
# reachable by DNS rebinding from a random web page in the user's browser.
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="AutoVid review", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def localhost_only(request: Request, call_next):
        header = request.headers.get("host") or ""
        if header.startswith("["):  # bracketed IPv6, e.g. [::1]:8766
            host = header[1:header.index("]") if "]" in header else len(header)]
        elif header.count(":") == 1:
            host = header.rsplit(":", 1)[0]
        else:
            host = header
        if host.lower() not in _ALLOWED_HOSTS:
            return JSONResponse({"detail": "localhost only"}, status_code=403)
        return await call_next(request)

    holder = {"settings": settings}  # refreshed after settings saves
    runner = JobRunner()

    def current_settings() -> Settings:
        return holder["settings"]

    conn_cache = threading.local()

    def conn():
        """One cached connection per (thread, database path).

        Per-thread, not global: sqlite3 connections must not interleave
        statements from concurrent request threads — a shared connection
        would merge separate requests' writes into one implicit transaction.
        Cached because connect+pragmas+init_db per request was pure churn;
        bounded by the threadpool size. Job threads open (and close) their
        own connections.
        """
        s = holder["settings"]
        key = str(Path(s.db_path).resolve())
        per_path = getattr(conn_cache, "per_path", None)
        if per_path is None:
            per_path = conn_cache.per_path = {}
        c = per_path.get(key)
        if c is None:
            c = db.connect(s.db_path)
            per_path[key] = c
        return c

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
    def segments(
        asset_id: str | None = None, q: str | None = None,
        status: str | None = None, limit: int = 1000,
    ) -> list[dict]:
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
        if status:
            sql += " AND s.review_status = ?"
            args.append(status)
        sql += " ORDER BY s.highlight_score DESC LIMIT ?"
        args.append(max(int(limit), 1))
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
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        validation = validate_edl(c, board)
        if not validation.ok:
            raise HTTPException(400, "; ".join(e.message for e in validation.errors))

        # Renders take minutes (dozens of ffmpeg spawns) — run as a job so the
        # request returns at once and the browser polls progress. Also keeps
        # the one-job-at-a-time guard: a render must not interleave with a
        # pipeline stage writing the same tables.
        def target(job: Job) -> None:
            s = current_settings()
            c = db.connect(s.db_path)
            try:
                board = load_edl(c, edl_id)
                job.current = board.name
                out = render_edl(c, s, board)
                job.result = {"edl_id": edl_id, "path": str(out),
                              "duration_s": board.duration_s}
                job.log.append(f"rendered {out.name}")
            finally:
                c.close()

        return runner.start("render", {"edl_id": edl_id}, target).view()

    @app.get("/renders/{edl_id}/{name}")
    def rendered(edl_id: str, name: str) -> FileResponse:
        data_dir = current_settings().data_dir
        path = (data_dir / "renders" / edl_id / name).resolve()
        if not path.is_relative_to((data_dir / "renders").resolve()) or not path.exists():
            raise HTTPException(404, "render not found")
        return FileResponse(path, media_type="video/mp4", filename=name)

    @app.get("/api/edls/{edl_id}/validate")
    def edl_validation(edl_id: str) -> dict:
        try:
            board = load_edl(conn(), edl_id)
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        return validate_edl(conn(), board).model_dump()

    # -- storyboards: Script · Storyboard · EDL ------------------------------

    @app.get("/api/storyboards")
    def storyboards_list() -> list[dict]:
        return [dict(r) for r in doc_mod.list_boards(conn())]

    @app.post("/api/storyboards/generate")
    def storyboards_generate(params: StoryboardGenerateParams) -> dict:
        # The agent is one or more LLM round trips (~30s each) — a job, like
        # render/music. Falls back to the deterministic planner (recorded in
        # the result origin) when no key is set or the draft fails the
        # catalog gate twice.
        def target(job: Job) -> None:
            s = current_settings()
            c = db.connect(s.db_path)
            try:
                job.current = "planning storyboard"
                board_id, origin = agent_mod.generate_with_fallback(
                    c, s, objective=params.objective,
                    target_duration_s=params.target_duration_s,
                    style=params.style, name=params.name,
                )
                row = doc_mod.get_board(c, board_id)
                document, version = doc_mod.load_document(c, board_id)
                job.result = {
                    "storyboard_id": board_id, "origin": origin, "version": version,
                    "name": row["name"], "scenes": len(document.scenes),
                    "duration_s": round(document.duration_s, 2),
                    "missing_footage": document.missing_footage,
                }
                job.log.append(
                    f"storyboard {board_id} via {origin}: "
                    f"{len(document.scenes)} scenes, {document.duration_s:.1f}s")
            finally:
                c.close()

        return runner.start("storyboard-generate", params.model_dump(), target).view()

    @app.get("/api/storyboards/{storyboard_id}")
    def storyboards_get(storyboard_id: str) -> dict:
        c = conn()
        try:
            row = doc_mod.get_board(c, storyboard_id)
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        document, version = doc_mod.load_document(c, storyboard_id)
        script = doc_mod.load_script(c, storyboard_id)
        return {
            **dict(row), "current_version": version,
            "document": document.model_dump(),
            "script": script.model_dump() if script else None,
            "versions": [dict(v) for v in doc_mod.list_versions(c, storyboard_id)],
        }

    @app.put("/api/storyboards/{storyboard_id}")
    def storyboards_put(storyboard_id: str, update: StoryboardUpdate) -> dict:
        c = conn()
        try:
            doc, _ = doc_mod.load_document(c, storyboard_id)
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        document = doc.model_copy(update={
            "scenes": update.scenes,
            "notes": update.notes if update.notes is not None else doc.notes,
            "missing_footage": (update.missing_footage if update.missing_footage is not None
                                else doc.missing_footage),
        })
        script = doc_mod.load_script(c, storyboard_id) \
            or ScriptDocument(storyboard_id=storyboard_id)
        problems = agent_mod.validate_document(c, document, script)
        if problems:
            raise HTTPException(400, "; ".join(problems))
        version = doc_mod.save_document(c, storyboard_id, document, origin="user_edit")
        return {"storyboard_id": storyboard_id, "version": version}

    @app.put("/api/storyboards/{storyboard_id}/script")
    def script_put(storyboard_id: str, update: ScriptUpdate) -> dict:
        c = conn()
        try:
            doc_mod.get_board(c, storyboard_id)
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        script = doc_mod.load_script(c, storyboard_id) \
            or ScriptDocument(storyboard_id=storyboard_id)
        if update.title is not None:
            script.title = update.title
        if update.synopsis is not None:
            script.synopsis = update.synopsis
        script.beats = update.beats
        doc_mod.save_script(c, script)  # marks dependent scenes stale + bumps version
        document, version = doc_mod.load_document(c, storyboard_id)
        return {
            "storyboard_id": storyboard_id, "version": version,
            "stale_scenes": [s.id for s in document.scenes if s.stale],
        }

    @app.post("/api/storyboards/{storyboard_id}/regenerate")
    def storyboards_regenerate(storyboard_id: str, params: RegenerateParams) -> dict:
        try:
            doc_mod.get_board(conn(), storyboard_id)
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        if not current_settings().vlm_api_key:
            raise HTTPException(
                409, "no VLM key configured — regeneration needs the agent; "
                     "edit scenes manually or re-run generate (planner fallback)")

        def target(job: Job) -> None:
            s = current_settings()
            c = db.connect(s.db_path)
            sb_agent = agent_mod.StoryboardAgent(s)
            try:
                job.current = "regenerating " + (params.scene_id or "whole storyboard")
                version = sb_agent.regenerate(c, storyboard_id, scene_id=params.scene_id)
                job.result = {"storyboard_id": storyboard_id, "version": version}
                job.log.append(f"regenerated {params.scene_id or 'all'} → v{version}")
            finally:
                sb_agent.client.close()
                c.close()

        return runner.start(
            "storyboard-regenerate",
            {"storyboard_id": storyboard_id, **params.model_dump()}, target,
        ).view()

    @app.post("/api/storyboards/{storyboard_id}/compile")
    def storyboards_compile(storyboard_id: str) -> dict:
        c = conn()
        try:
            doc_mod.get_board(c, storyboard_id)
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        try:
            board = doc_mod.compile_to_edl(c, storyboard_id)
        except AutoVidError as e:
            raise HTTPException(400, str(e))
        validation = validate_edl(c, board)
        return {
            "edl_id": board.id, "duration_s": board.duration_s,
            "items": len(board.items),
            "warnings": [w.model_dump() for w in validation.warnings],
        }

    @app.post("/api/storyboards/{storyboard_id}/approve")
    def storyboards_approve(storyboard_id: str, approve: bool = True) -> dict:
        try:
            doc_mod.set_status(conn(), storyboard_id,
                               "approved" if approve else "draft")
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        return {"storyboard_id": storyboard_id,
                "status": "approved" if approve else "draft"}

    @app.post("/api/storyboards/{storyboard_id}/restore")
    def storyboards_restore(storyboard_id: str, params: RestoreParams) -> dict:
        c = conn()
        try:
            document, restored_from = doc_mod.load_document(c, storyboard_id, params.version)
        except AutoVidError as e:
            raise HTTPException(404, str(e))
        version = doc_mod.save_document(
            c, storyboard_id, document, origin="restore",
            note=f"restored from v{restored_from}")
        return {"storyboard_id": storyboard_id, "version": version,
                "restored_from": restored_from}

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

    @app.post("/api/models")
    def list_vlm_models(req: ModelsRequest) -> dict:
        from .vlm import VLMUnavailable, list_models

        s = current_settings()
        # Form values win over saved ones so the list works before saving.
        try:
            return {"models": list_models(req.base_url or s.vlm_base_url,
                                          req.api_key or s.vlm_api_key)}
        except VLMUnavailable as e:
            raise HTTPException(502, str(e)) from e

    @app.post("/api/vlm/test")
    def test_vlm(req: ModelsRequest) -> dict:
        """One real call with the form's (or saved) key / base URL / model: shows the server's own reason."""
        from .vlm import VLMUnavailable

        s = current_settings().model_copy(update={k: v for k, v in (
            ("vlm_base_url", req.base_url), ("vlm_api_key", req.api_key), ("vlm_model", req.model)) if v})
        try:
            VLMClient(s).ping()
        except VLMUnavailable as e:
            raise HTTPException(502, str(e)) from e
        return {"ok": True, "model": s.vlm_model}

    @app.post("/api/script")
    def write_story(params: ScriptParams) -> dict:
        """Order the spoken lines and food shots into a story (needs transcripts) and lint the result."""
        from dataclasses import asdict

        from .check import check_edl
        from .script import write_script

        s = current_settings()
        c = conn()
        try:
            board, title = write_script(c, VLMClient(s), objective=params.objective, name=params.name,
                                        timelapse_montage=s.timelapse_montage, dish_moments=s.dish_moments)
        except (AutoVidError, ValueError) as e:
            raise HTTPException(409, str(e)) from e
        out = {"edl_id": board.id, "title": title, "items": len(board.items), "duration_s": board.duration_s,
               "findings": [asdict(f) for f in check_edl(c, board)]}
        if params.render:
            try:
                out["render_path"] = str(render_edl(c, s, board))
            except AutoVidError as e:
                out["render_error"] = str(e)
        return out

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
            try:
                from .ingest import ingest_folder

                report = ingest_folder(
                    c, s, folder, recursive=params.recursive,
                    progress=lambda p, i, n: (
                        setattr(job, "done", i), setattr(job, "total", n),
                        setattr(job, "current", Path(p).name)),
                )
                job.log.append(f"added {len(report.added)}, duplicates {len(report.duplicates)}, "
                               f"unchanged {len(report.skipped)}, failed {len(report.failed)}")
            finally:
                c.close()

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
            whisper = None
            if stage == "transcribe":  # not status-driven: every asset with audio not yet transcribed
                from .transcribe import load_model, transcribe_asset

                targets = [asset_id] if asset_id else [
                    r["id"] for r in c.execute("SELECT id FROM assets WHERE has_audio = 1 ORDER BY original_name")
                    if not db.job_done(c, r["id"], "transcribe")]
                whisper = load_model(s)
            else:
                targets = [asset_id] if asset_id else [
                    r["id"] for r in pending_assets(c, status_for[stage])]
            job.total = len(targets)
            try:
                if stage == "analyze" and vlm.configured:
                    vlm.ping()  # a bad key / model / parameter stops the job here, with the server's reason
                for aid in targets:
                    job.current = aid
                    if stage == "sample":
                        sample_asset(c, s, aid)
                    elif stage == "analyze":
                        analyze_asset(c, s, aid, vlm=vlm, decisions=decisions)
                    elif stage == "cut":
                        cut_asset(c, s, aid, decisions=decisions)
                    elif stage == "transcribe":
                        job.log.append(f"{aid}: {transcribe_asset(c, s, aid, model=whisper)} utterances")
                    else:
                        from .organize import organize_asset

                        organize_asset(c, s, aid)
                    job.done += 1
                    job.log.append(f"{aid}: done")
            finally:
                vlm.close()
                decisions.close()
                c.close()

        return target

    @app.post("/api/pipeline/{stage}")
    def start_stage(stage: str, params: StageParams) -> dict:
        if stage not in ("sample", "transcribe", "analyze", "cut", "organize"):
            raise HTTPException(400, "stage must be sample | transcribe | analyze | cut | organize")
        if stage not in ("sample", "transcribe") and params.asset_id is None:
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
        music_file = Path(params.music_path)
        if not music_file.exists():
            raise HTTPException(400, f"music file not found: {params.music_path}")

        # Librosa analysis + per-section mood calls + render take minutes —
        # job, not request (same reasoning as the render endpoint).
        def target(job: Job) -> None:
            s = current_settings()
            c = db.connect(s.db_path)
            from .music import analyze_music
            from .planner import plan_music_video

            try:
                job.current = "analyzing music"
                analysis = analyze_music(music_file)
                decisions = DecisionClient(s, c)
                try:
                    board = plan_music_video(
                        c, s, analysis, objective=params.objective, mode=params.mode,
                        style=params.style, name=params.name, decisions=decisions,
                    )
                finally:
                    decisions.close()
                out = {"edl_id": board.id, "items": len(board.items),
                       "duration_s": board.duration_s, "tempo_bpm": analysis.tempo_bpm}
                job.log.append(f"planned {len(board.items)} cuts @ "
                               f"{analysis.tempo_bpm:.0f} BPM")
                if params.render:
                    job.current = "rendering"
                    try:
                        out["render_path"] = str(render_edl(c, s, board))
                    except AutoVidError as e:
                        out["render_error"] = str(e)
                job.result = out
            finally:
                c.close()

        return runner.start("music-plan", params.model_dump(), target).view()

    return app


def serve(settings: Settings | None = None, host: str = "127.0.0.1", port: int = 8766) -> None:
    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
