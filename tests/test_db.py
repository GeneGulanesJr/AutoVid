"""Schema migration tests: pre-v3 catalogs must end up usable by the new code.

Regression guard for the review finding that the analyses window-uniqueness
lived only in CREATE TABLE IF NOT EXISTS — a no-op on existing databases,
which made analyze_asset's upsert crash on every upgraded catalog.
"""
from __future__ import annotations

import sqlite3

import pytest

from autovid import db

_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
    original_path TEXT NOT NULL, original_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ingested', duration_s REAL NOT NULL DEFAULT 0,
    width INTEGER, height INTEGER, fps REAL NOT NULL DEFAULT 0,
    has_audio INTEGER NOT NULL DEFAULT 0, size_bytes INTEGER NOT NULL DEFAULT 0,
    poster_path TEXT, dup_paths_json TEXT NOT NULL DEFAULT '[]',
    ffprobe_json TEXT NOT NULL DEFAULT '{}', error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(id),
    window_start REAL NOT NULL, window_end REAL NOT NULL,
    provider TEXT NOT NULL, model TEXT NOT NULL,
    response_json TEXT NOT NULL DEFAULT '{}',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_analyses_asset ON analyses(asset_id, window_start);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(id), ts REAL NOT NULL, kind TEXT NOT NULL,
    path TEXT NOT NULL, scene_score REAL, motion_score REAL,
    UNIQUE (asset_id, kind, ts)
);
CREATE INDEX IF NOT EXISTS idx_frames_asset ON frames(asset_id, kind, ts);
"""


def _make_v1_catalog() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_V1_SCHEMA)
    conn.execute(
        "INSERT INTO assets (id, sha256, path, original_path, original_name) "
        "VALUES ('a_old', 'beef', 'x.mp4', 'x.mp4', 'x.mp4')"
    )
    # duplicate windows from a re-run on the old insert-only code
    conn.executemany(
        "INSERT INTO analyses (asset_id, window_start, window_end, provider, model) "
        "VALUES ('a_old', 0.0, 60.0, ?, 'm')",
        [("old-run",), ("newest-run",)],
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    return conn


def test_migration_dedupes_and_enables_upsert() -> None:
    conn = _make_v1_catalog()
    db.init_db(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    # newest row per window survives the dedupe
    rows = conn.execute(
        "SELECT provider FROM analyses WHERE asset_id = 'a_old'"
    ).fetchall()
    assert [r["provider"] for r in rows] == ["newest-run"]

    # the analyze_asset upsert must work on the MIGRATED database — this is
    # exactly what broke when uniqueness lived only in CREATE TABLE IF NOT EXISTS
    conn.execute(
        """INSERT INTO analyses (asset_id, window_start, window_end, provider, model,
                                 response_json, prompt_tokens, completion_tokens, confidence)
           VALUES ('a_old', 0.0, 60.0, 'rerun', 'm', '{}', 1, 2, 0.9)
           ON CONFLICT(asset_id, window_start) DO UPDATE SET
               provider = excluded.provider, prompt_tokens = excluded.prompt_tokens"""
    )
    conn.commit()
    row = conn.execute(
        "SELECT provider, prompt_tokens FROM analyses WHERE asset_id = 'a_old'"
    ).fetchone()
    assert row["provider"] == "rerun" and row["prompt_tokens"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM analyses").fetchone()["n"] == 1


def test_fresh_database_gets_the_same_indexes() -> None:
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    assert "idx_analyses_asset_window" in names
    assert "idx_frames_asset" not in names  # duplicate index stays gone
    assert "idx_assets_path" in names


# --- v3 → v4: storyboards/scripts/versions + edls.storyboard_id -------------

_V3_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
    original_path TEXT NOT NULL, original_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ingested', duration_s REAL NOT NULL DEFAULT 0,
    width INTEGER, height INTEGER, fps REAL NOT NULL DEFAULT 0,
    has_audio INTEGER NOT NULL DEFAULT 0, size_bytes INTEGER NOT NULL DEFAULT 0,
    poster_path TEXT, dup_paths_json TEXT NOT NULL DEFAULT '[]',
    ffprobe_json TEXT NOT NULL DEFAULT '{}', error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(id),
    window_start REAL NOT NULL, window_end REAL NOT NULL,
    provider TEXT NOT NULL, model TEXT NOT NULL,
    response_json TEXT NOT NULL DEFAULT '{}',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_analyses_asset_window ON analyses(asset_id, window_start);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(id), ts REAL NOT NULL, kind TEXT NOT NULL,
    path TEXT NOT NULL, scene_score REAL, motion_score REAL,
    UNIQUE (asset_id, kind, ts)
);
CREATE TABLE IF NOT EXISTS edls (
    id TEXT PRIMARY KEY, name TEXT NOT NULL,
    objective TEXT NOT NULL DEFAULT '', style TEXT NOT NULL DEFAULT '',
    target_duration_s REAL NOT NULL DEFAULT 0, items_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _make_v3_catalog() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_V3_SCHEMA)
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    return conn


def test_v3_catalog_migrates_to_v4() -> None:
    conn = _make_v3_catalog()
    db.init_db(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"storyboards", "storyboard_versions", "scripts"} <= tables

    # the ALTER landed on the pre-existing edls table
    edl_cols = {r[1] for r in conn.execute("PRAGMA table_info(edls)")}
    assert "storyboard_id" in edl_cols

    # the new storyboard-version SQL must work on the MIGRATED database —
    # the standing rule from the v3 upsert crash
    conn.execute(
        "INSERT INTO assets (id, sha256, path, original_path, original_name) "
        "VALUES ('a_old', 'beef', 'x.mp4', 'x.mp4', 'x.mp4')")
    conn.execute(
        "INSERT INTO storyboards (id, name) VALUES ('sb1', 'test')")
    conn.execute(
        """INSERT INTO storyboard_versions (storyboard_id, version, origin, document_json)
           VALUES ('sb1', 1, 'planner', '{}')""")
    conn.execute(
        """INSERT INTO edls (id, name, items_json, storyboard_id)
           VALUES ('e1', 'n', '{}', 'sb1')""")
    conn.commit()
    assert conn.execute(
        "SELECT storyboard_id FROM edls WHERE id = 'e1'").fetchone()["storyboard_id"] == "sb1"


def test_fresh_database_has_v4_tables_and_columns() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(edls)")}
    assert "storyboard_id" in cols
    # one script per storyboard
    conn.execute("INSERT INTO storyboards (id, name) VALUES ('sb1', 'n')")
    conn.execute(
        "INSERT INTO scripts (id, storyboard_id, beats_json) VALUES ('sc1', 'sb1', '[]')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO scripts (id, storyboard_id, beats_json) VALUES ('sc2', 'sb1', '[]')")
