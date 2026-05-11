"""SQLite-backed face embeddings store.

Schema is intentionally simple: one row per training photo, embedding stored
as binary blob (numpy float32[128]). All embeddings for a subject are loaded
into memory at process start for fast matching.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np


class FaceRow(NamedTuple):
    id: int
    subject: str
    photo_filename: str
    embedding: np.ndarray
    created_at: datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    photo_filename TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_faces_subject ON faces(subject);
"""


class FaceDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def add(self, subject: str, photo_filename: str, embedding: np.ndarray) -> int:
        blob = embedding.astype(np.float32).tobytes()
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO faces (subject, photo_filename, embedding, created_at) "
                "VALUES (?, ?, ?, ?)",
                (subject, photo_filename, blob, now),
            )
            return cur.lastrowid or 0

    def all(self) -> list[FaceRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, subject, photo_filename, embedding, created_at FROM faces"
            ).fetchall()
        return [
            FaceRow(
                id=r["id"],
                subject=r["subject"],
                photo_filename=r["photo_filename"],
                embedding=np.frombuffer(r["embedding"], dtype=np.float32),
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def list_subjects(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject, COUNT(*) AS n FROM faces GROUP BY subject"
            ).fetchall()
        return {r["subject"]: r["n"] for r in rows}

    def delete_subject(self, subject: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM faces WHERE subject = ?", (subject,))
            return cur.rowcount
