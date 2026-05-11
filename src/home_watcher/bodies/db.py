"""SQLite stores for known and unknown person body embeddings (Re-ID)."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np


class UnknownBodyRow(NamedTuple):
    id: int
    detected_at: datetime
    camera: str
    crop_filename: str
    snapshot_filename: str
    bbox_top: int
    bbox_right: int
    bbox_bottom: int
    bbox_left: int
    width_px: int
    height_px: int
    embedding: np.ndarray
    labeled: int


SCHEMA = """
CREATE TABLE IF NOT EXISTS unknown_bodies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    camera TEXT NOT NULL,
    crop_filename TEXT NOT NULL,
    snapshot_filename TEXT NOT NULL,
    bbox_top INTEGER NOT NULL,
    bbox_right INTEGER NOT NULL,
    bbox_bottom INTEGER NOT NULL,
    bbox_left INTEGER NOT NULL,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    labeled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bodies_labeled ON unknown_bodies(labeled);
CREATE INDEX IF NOT EXISTS idx_bodies_detected_at ON unknown_bodies(detected_at);

CREATE TABLE IF NOT EXISTS known_bodies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    crop_filename TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_known_bodies_subject ON known_bodies(subject);
"""


class BodyDB:
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

    # --- unknown bodies ---

    def add_unknown(
        self,
        *,
        camera: str,
        crop_filename: str,
        snapshot_filename: str,
        bbox: tuple[int, int, int, int],
        width_px: int,
        height_px: int,
        embedding: np.ndarray,
    ) -> int:
        top, right, bottom, left = bbox
        blob = embedding.astype(np.float32).tobytes()
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO unknown_bodies
                (detected_at, camera, crop_filename, snapshot_filename,
                 bbox_top, bbox_right, bbox_bottom, bbox_left,
                 width_px, height_px, embedding, labeled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (now, camera, crop_filename, snapshot_filename,
                 top, right, bottom, left,
                 width_px, height_px, blob),
            )
            return cur.lastrowid or 0

    def list_unlabeled(self, limit: int = 100) -> list[UnknownBodyRow]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM unknown_bodies WHERE labeled = 0
                ORDER BY detected_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_named(r) for r in rows]

    def get_unknown(self, body_id: int) -> UnknownBodyRow | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM unknown_bodies WHERE id = ?", (body_id,)
            ).fetchone()
        return self._row_to_named(row) if row else None

    def mark_labeled(self, body_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE unknown_bodies SET labeled = 1 WHERE id = ?", (body_id,)
            )

    def mark_discarded(self, body_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE unknown_bodies SET labeled = 2 WHERE id = ?", (body_id,)
            )

    # --- known bodies ---

    def add_known(self, subject: str, crop_filename: str, embedding: np.ndarray) -> int:
        blob = embedding.astype(np.float32).tobytes()
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO known_bodies (subject, crop_filename, embedding, created_at)
                VALUES (?, ?, ?, ?)""",
                (subject, crop_filename, blob, now),
            )
            return cur.lastrowid or 0

    def list_known_subjects(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject, COUNT(*) AS n FROM known_bodies GROUP BY subject"
            ).fetchall()
        return {r["subject"]: r["n"] for r in rows}

    def all_known_by_subject(self) -> dict[str, list[np.ndarray]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject, embedding FROM known_bodies"
            ).fetchall()
        out: dict[str, list[np.ndarray]] = {}
        for r in rows:
            out.setdefault(r["subject"], []).append(
                np.frombuffer(r["embedding"], dtype=np.float32)
            )
        return out

    def delete_known_subject(self, subject: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM known_bodies WHERE subject = ?", (subject,))
            return cur.rowcount

    @staticmethod
    def _row_to_named(row: sqlite3.Row) -> UnknownBodyRow:
        return UnknownBodyRow(
            id=row["id"],
            detected_at=datetime.fromisoformat(row["detected_at"]),
            camera=row["camera"],
            crop_filename=row["crop_filename"],
            snapshot_filename=row["snapshot_filename"],
            bbox_top=row["bbox_top"],
            bbox_right=row["bbox_right"],
            bbox_bottom=row["bbox_bottom"],
            bbox_left=row["bbox_left"],
            width_px=row["width_px"],
            height_px=row["height_px"],
            embedding=np.frombuffer(row["embedding"], dtype=np.float32),
            labeled=row["labeled"],
        )
