"""SQLite stores for known and unknown (unlabeled) pet detections."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np


class UnknownPetRow(NamedTuple):
    id: int
    detected_at: datetime
    camera: str
    species: str
    confidence: float
    crop_filename: str
    snapshot_filename: str
    bbox_top: int
    bbox_right: int
    bbox_bottom: int
    bbox_left: int
    width_px: int
    height_px: int
    labeled: int


SCHEMA = """
CREATE TABLE IF NOT EXISTS unknown_pets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    camera TEXT NOT NULL,
    species TEXT NOT NULL,
    confidence REAL NOT NULL,
    crop_filename TEXT NOT NULL,
    snapshot_filename TEXT NOT NULL,
    bbox_top INTEGER NOT NULL,
    bbox_right INTEGER NOT NULL,
    bbox_bottom INTEGER NOT NULL,
    bbox_left INTEGER NOT NULL,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    labeled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pets_labeled ON unknown_pets(labeled);
CREATE INDEX IF NOT EXISTS idx_pets_detected_at ON unknown_pets(detected_at);

CREATE TABLE IF NOT EXISTS known_pets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    species TEXT NOT NULL,
    crop_filename TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_known_pets_subject ON known_pets(subject);
"""


class PetDB:
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

    # --- unknown pets ---

    def add_unknown(
        self,
        *,
        camera: str,
        species: str,
        confidence: float,
        crop_filename: str,
        snapshot_filename: str,
        bbox: tuple[int, int, int, int],
        width_px: int,
        height_px: int,
    ) -> int:
        top, right, bottom, left = bbox
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO unknown_pets
                (detected_at, camera, species, confidence,
                 crop_filename, snapshot_filename,
                 bbox_top, bbox_right, bbox_bottom, bbox_left,
                 width_px, height_px, labeled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (now, camera, species, confidence,
                 crop_filename, snapshot_filename,
                 top, right, bottom, left,
                 width_px, height_px),
            )
            return cur.lastrowid or 0

    def list_unlabeled(self, limit: int = 100) -> list[UnknownPetRow]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM unknown_pets WHERE labeled = 0
                ORDER BY detected_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_named(r) for r in rows]

    def get_unknown(self, pet_id: int) -> UnknownPetRow | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM unknown_pets WHERE id = ?", (pet_id,)
            ).fetchone()
        return self._row_to_named(row) if row else None

    def mark_labeled(self, pet_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE unknown_pets SET labeled = 1 WHERE id = ?", (pet_id,)
            )

    def mark_discarded(self, pet_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE unknown_pets SET labeled = 2 WHERE id = ?", (pet_id,)
            )

    # --- known pets ---

    def add_known(
        self,
        subject: str,
        species: str,
        crop_filename: str,
        embedding: np.ndarray | None = None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        emb_bytes = embedding.tobytes() if embedding is not None else None
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO known_pets (subject, species, crop_filename, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (subject, species, crop_filename, emb_bytes, now),
            )
            return cur.lastrowid or 0

    def list_known_subjects(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject, COUNT(*) AS n FROM known_pets GROUP BY subject"
            ).fetchall()
        return {r["subject"]: r["n"] for r in rows}

    def all_known_by_subject(self) -> dict[str, list[np.ndarray]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject, embedding FROM known_pets WHERE embedding IS NOT NULL"
            ).fetchall()
        result: dict[str, list[np.ndarray]] = {}
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
            result.setdefault(r["subject"], []).append(emb)
        return result

    def species_by_subject(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT subject, species FROM known_pets"
            ).fetchall()
        return {r["subject"]: r["species"] for r in rows}

    def delete_known_subject(self, subject: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM known_pets WHERE subject = ?", (subject,))
            return cur.rowcount

    @staticmethod
    def _row_to_named(row: sqlite3.Row) -> UnknownPetRow:
        return UnknownPetRow(
            id=row["id"],
            detected_at=datetime.fromisoformat(row["detected_at"]),
            camera=row["camera"],
            species=row["species"],
            confidence=row["confidence"],
            crop_filename=row["crop_filename"],
            snapshot_filename=row["snapshot_filename"],
            bbox_top=row["bbox_top"],
            bbox_right=row["bbox_right"],
            bbox_bottom=row["bbox_bottom"],
            bbox_left=row["bbox_left"],
            width_px=row["width_px"],
            height_px=row["height_px"],
            labeled=row["labeled"],
        )
