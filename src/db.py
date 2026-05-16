"""
src/db.py
SQLite helpers for the xAI hallucination detection project.

Schema
------
    runs          — one row per experimental run
    narratives    — one row per generated narrative
    evaluations   — one row per evaluated narrative (1-to-1 with narratives)

Public API
----------
    init_db(db_path)                     -> None
    insert_run(conn, run_id, ...)        -> None
    insert_narrative(conn, ...)          -> None
    insert_evaluation(conn, ...)         -> None
    get_narratives_for_run(conn, run_id) -> List[dict]
    get_evaluations_for_run(conn, run_id)-> List[dict]
    get_run(conn, run_id)                -> dict | None
    open_connection(db_path)             -> sqlite3.Connection
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    run_name    TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

_CREATE_NARRATIVES = """
CREATE TABLE IF NOT EXISTS narratives (
    narrative_id    TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    dataset         TEXT NOT NULL,
    instance_id     INTEGER NOT NULL,
    model_id        TEXT NOT NULL,
    prompt_strategy TEXT NOT NULL,
    narrative_text  TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""

_CREATE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS evaluations (
    eval_id              TEXT PRIMARY KEY,
    narrative_id         TEXT NOT NULL REFERENCES narratives(narrative_id),
    sign_inversion       INTEGER NOT NULL DEFAULT 0,
    rank_swap            INTEGER NOT NULL DEFAULT 0,
    feature_fabrication  INTEGER NOT NULL DEFAULT 0,
    magnitude_distortion INTEGER NOT NULL DEFAULT 0,
    omission             INTEGER NOT NULL DEFAULT 0,
    any_hallucination    INTEGER NOT NULL DEFAULT 0,
    notes                TEXT,
    evaluated_at         TEXT NOT NULL
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_narratives_run ON narratives(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_narratives_model ON narratives(model_id);",
    "CREATE INDEX IF NOT EXISTS idx_narratives_dataset ON narratives(dataset);",
    "CREATE INDEX IF NOT EXISTS idx_evals_narrative ON evaluations(narrative_id);",
]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str | Path) -> None:
    """
    Create the database file and apply the schema (idempotent).
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(_CREATE_RUNS)
        conn.execute(_CREATE_NARRATIVES)
        conn.execute(_CREATE_EVALUATIONS)
        for idx_sql in _INDEXES:
            conn.execute(idx_sql)
        conn.commit()


def open_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a SQLite connection with sensible defaults.
    Caller is responsible for closing.
    Use as a context manager or call .close() explicitly.
    """
    path = Path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def db_connection(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that opens, yields, and closes a connection."""
    conn = open_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def insert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_name: str,
    config_json: str | dict,
    created_at: str,
) -> None:
    """Insert a row into the runs table."""
    if isinstance(config_json, dict):
        config_json = json.dumps(config_json)
    conn.execute(
        "INSERT INTO runs (run_id, run_name, config_json, created_at) VALUES (?, ?, ?, ?)",
        (run_id, run_name, config_json, created_at),
    )
    conn.commit()


def insert_narrative(
    conn: sqlite3.Connection,
    *,
    narrative_id: str,
    run_id: str,
    dataset: str,
    instance_id: int,
    model_id: str,
    prompt_strategy: str,
    narrative_text: str,
    created_at: str,
) -> None:
    """Insert a row into the narratives table."""
    conn.execute(
        """
        INSERT INTO narratives
            (narrative_id, run_id, dataset, instance_id, model_id,
             prompt_strategy, narrative_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            narrative_id, run_id, dataset, instance_id, model_id,
            prompt_strategy, narrative_text, created_at,
        ),
    )
    conn.commit()


def insert_evaluation(
    conn: sqlite3.Connection,
    *,
    eval_id: str,
    narrative_id: str,
    sign_inversion: bool = False,
    rank_swap: bool = False,
    feature_fabrication: bool = False,
    magnitude_distortion: bool = False,
    omission: bool = False,
    notes: Optional[str] = None,
    evaluated_at: str,
) -> None:
    """Insert a row into the evaluations table."""
    any_hallucination = any(
        [sign_inversion, rank_swap, feature_fabrication, magnitude_distortion, omission]
    )
    conn.execute(
        """
        INSERT INTO evaluations
            (eval_id, narrative_id, sign_inversion, rank_swap,
             feature_fabrication, magnitude_distortion, omission,
             any_hallucination, notes, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            eval_id, narrative_id,
            int(sign_inversion), int(rank_swap), int(feature_fabrication),
            int(magnitude_distortion), int(omission), int(any_hallucination),
            notes, evaluated_at,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    """Return a single run row as a dict, or None if not found."""
    row = conn.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return dict(row) if row else None


def get_narratives_for_run(conn: sqlite3.Connection, run_id: str) -> List[dict]:
    """Return all narrative rows for a given run_id."""
    rows = conn.execute(
        "SELECT * FROM narratives WHERE run_id = ? ORDER BY dataset, instance_id, model_id, prompt_strategy",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_evaluations_for_run(conn: sqlite3.Connection, run_id: str) -> List[dict]:
    """
    Return all evaluation rows joined with their narrative for a given run_id.
    """
    rows = conn.execute(
        """
        SELECT e.*, n.dataset, n.instance_id, n.model_id, n.prompt_strategy
        FROM evaluations e
        JOIN narratives n ON e.narrative_id = n.narrative_id
        WHERE n.run_id = ?
        ORDER BY n.dataset, n.instance_id, n.model_id, n.prompt_strategy
        """,
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_narrative(conn: sqlite3.Connection, narrative_id: str) -> Optional[dict]:
    """Return a single narrative row as a dict, or None if not found."""
    row = conn.execute(
        "SELECT * FROM narratives WHERE narrative_id = ?", (narrative_id,)
    ).fetchone()
    return dict(row) if row else None


def list_runs(conn: sqlite3.Connection) -> List[dict]:
    """Return all run rows ordered by creation time."""
    rows = conn.execute(
        "SELECT run_id, run_name, created_at FROM runs ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def narrative_exists(
    conn: sqlite3.Connection,
    run_id: str,
    dataset: str,
    instance_id: int,
    model_id: str,
) -> bool:
    """
    Return True if a narrative already exists for this combination.

    Now that there is a single prompt strategy (Martens-style narrative),
    the resume key is just (run_id, dataset, instance_id, model_id) — the
    prompt_strategy column is left at its placeholder value.

    Used by the generator to skip work that was already completed in a
    previous (possibly interrupted) run, enabling crash-safe resume.
    """
    row = conn.execute(
        """
        SELECT 1 FROM narratives
        WHERE run_id = ? AND dataset = ? AND instance_id = ?
          AND model_id = ?
        LIMIT 1
        """,
        (run_id, dataset, instance_id, model_id),
    ).fetchone()
    return row is not None
