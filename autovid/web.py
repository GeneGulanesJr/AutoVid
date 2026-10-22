"""Minimal web review UI: catalog, review queue, EDL editor, render.

Binds to localhost only; no auth (like DecisionMCP's server — local tooling).
The UI is a single static page + JSON API; EDL edits are full-list replaces
(reorder/remove/trim happen client-side, then save), and human review
decisions land straight in the segments table.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db
from .config import Settings, get_settings
from .edl import EDL, EDLItem, load as load_edl, save as save_edl
from .errors import AutoVidError
from .render import render_edl

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
REVIEW_STATUSES = ("auto", "approved", "rejected", "pending")


class EDLUpdate(BaseModel):
    """Body of the EDL edit endpoint (module level: FastAPI must resolve the hint)."""

    name: str | None = None
    objective: str | None = None
    style: str | None = None
    items: list[EDLItem]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="AutoVid review", docs_url=None, redoc_url=None)

    def conn():
        return db.connect(settings.db_path)

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
            out = render_edl(c, settings, board)
        except AutoVidError as e:
            raise HTTPException(409, str(e))
        return {"edl_id": edl_id, "path": str(out), "duration_s": board.duration_s}

    @app.get("/renders/{edl_id}/{name}")
    def rendered(edl_id: str, name: str) -> FileResponse:
        path = (settings.data_dir / "renders" / edl_id / name).resolve()
        if not path.is_relative_to((settings.data_dir / "renders").resolve()) or not path.exists():
            raise HTTPException(404, "render not found")
        return FileResponse(path, media_type="video/mp4", filename=name)

    return app


def serve(settings: Settings | None = None, host: str = "127.0.0.1", port: int = 8766) -> None:
    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
