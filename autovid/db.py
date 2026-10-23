"""SQLite catalog: assets, frames, scenes, analyses, segments, decisions.

stdlib sqlite3 (WAL) in the style of DecisionMCP's usage store — no ORM.
Schema version tracked via ``PRAGMA user_version``; every derived artifact
references its source asset so all timestamps stay traceable to originals.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2  # 2: transcripts (every statement is IF NOT EXISTS, so v1 catalogs upgrade in place)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id            TEXT PRIMARY KEY,          -- 'a_' + 16 hex of sha256 (stable)
    sha256        TEXT NOT NULL UNIQUE,
    path          TEXT NOT NULL,             -- current absolute path
    original_path TEXT NOT NULL,
    original_name TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'ingested',
        -- ingested → sampled → analyzed → refined → extracted → organized | error
    duration_s    REAL NOT NULL DEFAULT 0,
    width         INTEGER,
    height        INTEGER,
    fps           REAL NOT NULL DEFAULT 0,
    has_audio     INTEGER NOT NULL DEFAULT 0,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    poster_path   TEXT,
    dup_paths_json TEXT NOT NULL DEFAULT '[]',  -- other paths with identical content
    ffprobe_json  TEXT NOT NULL DEFAULT '{}',
    error         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS frames (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id     TEXT NOT NULL REFERENCES assets(id),
    ts           REAL NOT NULL,              -- seconds on the source timeline
    kind         TEXT NOT NULL,              -- thumb | grid | dense
    path         TEXT NOT NULL,
    scene_score  REAL,                       -- scene-change likelihood at ts
    motion_score REAL,
    UNIQUE (asset_id, kind, ts)
);
CREATE INDEX IF NOT EXISTS idx_frames_asset ON frames(asset_id, kind, ts);

CREATE TABLE IF NOT EXISTS scenes (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(id),
    start_s  REAL NOT NULL,                  -- boundary instant (scdet) or span start (vlm)
    end_s    REAL,                           -- span end when known (vlm/manual)
    score    REAL,                           -- detection confidence
    source   TEXT NOT NULL DEFAULT 'scdet'   -- scdet | d1 | vlm | manual
);
CREATE INDEX IF NOT EXISTS idx_scenes_asset ON scenes(asset_id, start_s);

CREATE TABLE IF NOT EXISTS analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id          TEXT NOT NULL REFERENCES assets(id),
    window_start      REAL NOT NULL,
    window_end        REAL NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    response_json     TEXT NOT NULL DEFAULT '{}',
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    confidence        REAL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_analyses_asset ON analyses(asset_id, window_start);

CREATE TABLE IF NOT EXISTS segments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id            TEXT NOT NULL REFERENCES assets(id),
    start_s             REAL NOT NULL,
    end_s               REAL NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    tags_json           TEXT NOT NULL DEFAULT '[]',
    highlight_score     REAL,
    quality_json        TEXT NOT NULL DEFAULT '{}',
    verdict             TEXT,               -- accept | needs_evidence | reject | flag_review
    decision_confidence REAL,
    review_status       TEXT NOT NULL DEFAULT 'auto',  -- auto | approved | rejected | pending
    clip_path           TEXT,
    suggested_name      TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_segments_asset ON segments(asset_id, start_s);

-- Every decision-layer call, for measurement/review (mirrors usage.db ideas).
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id      TEXT REFERENCES assets(id),
    tool          TEXT NOT NULL,             -- decision_video_region | ...
    action        TEXT,
    payload_json  TEXT NOT NULL DEFAULT '{}',
    confidence    REAL,
    engine_version TEXT,
    elapsed_ms    REAL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_decisions_asset ON decisions(asset_id, tool);

-- Recoverable filesystem changes (renames) — old↔new mapping, undo support.
CREATE TABLE IF NOT EXISTS renames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT NOT NULL REFERENCES assets(id),
    old_path    TEXT NOT NULL,
    new_path    TEXT NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
    reverted_at TEXT
);

CREATE TABLE IF NOT EXISTS edls (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    objective         TEXT NOT NULL DEFAULT '',
    style             TEXT NOT NULL DEFAULT '',
    target_duration_s REAL NOT NULL DEFAULT 0,
    items_json        TEXT NOT NULL DEFAULT '[]',
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-asset stage ledger; resumability = skip (asset, stage) rows marked done.
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT NOT NULL REFERENCES assets(id),
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',  -- running | done | error
    detail      TEXT,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_asset_stage ON jobs(asset_id, stage);

-- Speech-to-text utterances (faster-whisper), source-timeline seconds.
CREATE TABLE IF NOT EXISTS transcripts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(id),
    start_s  REAL NOT NULL,
    end_s    REAL NOT NULL,
    text     TEXT NOT NULL,
    language TEXT
);
CREATE INDEX IF NOT EXISTS idx_transcripts_asset ON transcripts(asset_id, start_s);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the catalog DB with sane pragmas."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return
    conn.executescript(_SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


# ---------------------------------------------------------------------------
# Job ledger — resumability: a (asset, stage) pair marked done is never redone.
# ---------------------------------------------------------------------------


def job_done(conn: sqlite3.Connection, asset_id: str, stage: str) -> bool:
    row = conn.execute(
        "SELECT status FROM jobs WHERE asset_id = ? AND stage = ?", (asset_id, stage)
    ).fetchone()
    return row is not None and row["status"] == "done"


def job_start(conn: sqlite3.Connection, asset_id: str, stage: str) -> None:
    """Reset any previous attempt and mark the stage running (idempotent)."""
    conn.execute(
        """INSERT INTO jobs (asset_id, stage, status)
           VALUES (?, ?, 'running')
           ON CONFLICT(asset_id, stage) DO UPDATE SET
               status = 'running', detail = NULL,
               started_at = datetime('now'), finished_at = NULL""",
        (asset_id, stage),
    )
    conn.commit()


def job_finish(
    conn: sqlite3.Connection, asset_id: str, stage: str, *, error: str | None = None
) -> None:
    status = "error" if error else "done"
    conn.execute(
        """UPDATE jobs SET status = ?, detail = ?, finished_at = datetime('now')
           WHERE asset_id = ? AND stage = ?""",
        (status, error, asset_id, stage),
    )
    conn.commit()
