"""SQLite store for detected-but-unlabeled faces.

When the recognizer sees a face it can't match, we save the embedding +
the face crop + the full snapshot to disk, plus a row in this table.
The admin UI lets the user browse these and assign a subject name.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np


class UnknownFaceRow(NamedTuple):
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
    embedding: np.ndarray
    labeled: int


SCHEMA = """
CREATE TABLE IF NOT EXISTS unknown_faces (
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
    embedding BLOB NOT NULL,
    labeled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_unknown_labeled ON unknown_faces(labeled);
CREATE INDEX IF NOT EXISTS idx_unknown_detected_at ON unknown_faces(detected_at);
"""


class UnknownFaceDB:
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

    def add(
        self,
        *,
        camera: str,
        crop_filename: str,
        snapshot_filename: str,
        bbox: tuple[int, int, int, int],
        width_px: int,
        embedding: np.ndarray,
    ) -> int:
        top, right, bottom, left = bbox
        blob = embedding.astype(np.float32).tobytes()
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO unknown_faces
                (detected_at, camera, crop_filename, snapshot_filename,
                 bbox_top, bbox_right, bbox_bottom, bbox_left, width_px,
                 embedding, labeled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (now, camera, crop_filename, snapshot_filename,
                 top, right, bottom, left, width_px, blob),
            )
            return cur.lastrowid or 0

    def list_unlabeled(self, limit: int = 100) -> list[UnknownFaceRow]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM unknown_faces WHERE labeled = 0
                ORDER BY detected_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_named(r) for r in rows]

    def get(self, face_id: int) -> UnknownFaceRow | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM unknown_faces WHERE id = ?", (face_id,)
            ).fetchone()
        return self._row_to_named(row) if row else None

    def mark_labeled(self, face_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE unknown_faces SET labeled = 1 WHERE id = ?", (face_id,)
            )

    def mark_discarded(self, face_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE unknown_faces SET labeled = 2 WHERE id = ?", (face_id,)
            )

    def prune_older_than(self, cutoff: datetime) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM unknown_faces WHERE detected_at < ?",
                (cutoff.isoformat(),),
            )
            return cur.rowcount

    @staticmethod
    def _row_to_named(row: sqlite3.Row) -> UnknownFaceRow:
        return UnknownFaceRow(
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
            embedding=np.frombuffer(row["embedding"], dtype=np.float32),
            labeled=row["labeled"],
        )
