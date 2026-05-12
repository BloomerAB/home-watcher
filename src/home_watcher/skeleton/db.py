"""SQLite storage for skeleton biometric profiles."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass
class SkeletonRow:
    id: int
    camera: str
    detected_at: str
    profile_vector: bytes
    subject: str | None
    status: str
    shoulder_ratio: float
    torso_ratio: float
    leg_ratio: float
    arm_ratio: float
    height_px: float

    def feature_array(self) -> np.ndarray:
        return np.frombuffer(self.profile_vector, dtype=np.float32).copy()


class SkeletonDB:
    def __init__(self, db_path: str | object) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS skeleton_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                profile_vector BLOB NOT NULL,
                subject TEXT,
                status TEXT NOT NULL DEFAULT 'unknown',
                shoulder_ratio REAL NOT NULL DEFAULT 0.0,
                torso_ratio REAL NOT NULL DEFAULT 0.0,
                leg_ratio REAL NOT NULL DEFAULT 0.0,
                arm_ratio REAL NOT NULL DEFAULT 0.0,
                height_px REAL NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_skel_status
                ON skeleton_profiles(status);
        """)
        self._conn.commit()

    def add(
        self,
        camera: str,
        profile_vector: np.ndarray,
        shoulder_ratio: float,
        torso_ratio: float,
        leg_ratio: float,
        arm_ratio: float,
        height_px: float,
        subject: str | None = None,
    ) -> int:
        status = "known" if subject else "unknown"
        cur = self._conn.execute(
            """INSERT INTO skeleton_profiles
               (camera, detected_at, profile_vector, subject, status,
                shoulder_ratio, torso_ratio, leg_ratio, arm_ratio, height_px)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                camera,
                datetime.now(timezone.utc).isoformat(),
                profile_vector.tobytes(),
                subject,
                status,
                shoulder_ratio,
                torso_ratio,
                leg_ratio,
                arm_ratio,
                height_px,
            ),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def list_unknown(self, limit: int = 50) -> list[SkeletonRow]:
        rows = self._conn.execute(
            """SELECT * FROM skeleton_profiles WHERE status = 'unknown'
               ORDER BY detected_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [SkeletonRow(**dict(r)) for r in rows]

    def get(self, profile_id: int) -> SkeletonRow | None:
        row = self._conn.execute(
            "SELECT * FROM skeleton_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return SkeletonRow(**dict(row)) if row else None

    def label(self, profile_id: int, subject: str) -> None:
        self._conn.execute(
            "UPDATE skeleton_profiles SET subject = ?, status = 'known' WHERE id = ?",
            (subject, profile_id),
        )
        self._conn.commit()

    def discard(self, profile_id: int) -> None:
        self._conn.execute(
            "UPDATE skeleton_profiles SET status = 'discarded' WHERE id = ?",
            (profile_id,),
        )
        self._conn.commit()

    def known_by_subject(self) -> dict[str, list[np.ndarray]]:
        rows = self._conn.execute(
            "SELECT * FROM skeleton_profiles WHERE status = 'known' AND subject IS NOT NULL"
        ).fetchall()
        result: dict[str, list[np.ndarray]] = {}
        for row in rows:
            r = SkeletonRow(**dict(row))
            result.setdefault(r.subject or "", []).append(r.feature_array())
        return result
