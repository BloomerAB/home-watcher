"""SQLite storage for trajectories — known patterns and unknown for labeling."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .tracker import Position, Trajectory


@dataclass
class TrajectoryRow:
    id: int
    camera: str
    detected_at: str
    positions_json: str
    feature_vector: bytes
    subject: str | None
    status: str
    direction_angle: float | None
    speed: float
    entry_zone: str
    exit_zone: str

    def to_trajectory(self) -> Trajectory:
        raw = json.loads(self.positions_json)
        return Trajectory(
            camera=self.camera,
            positions=[Position(x=p["x"], y=p["y"], t=p["t"]) for p in raw],
        )

    def feature_array(self) -> np.ndarray:
        return np.frombuffer(self.feature_vector, dtype=np.float32).copy()


class TrajectoryDB:
    def __init__(self, db_path: str | object) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                positions_json TEXT NOT NULL,
                feature_vector BLOB NOT NULL,
                subject TEXT,
                status TEXT NOT NULL DEFAULT 'unknown',
                direction_angle REAL,
                speed REAL NOT NULL DEFAULT 0.0,
                entry_zone TEXT NOT NULL DEFAULT '0,0',
                exit_zone TEXT NOT NULL DEFAULT '0,0'
            );
            CREATE INDEX IF NOT EXISTS idx_traj_camera_subject
                ON trajectories(camera, subject);
            CREATE INDEX IF NOT EXISTS idx_traj_status
                ON trajectories(status);
        """)
        self._conn.commit()

    def add(self, trajectory: Trajectory, subject: str | None = None) -> int:
        positions_json = json.dumps([
            {"x": p.x, "y": p.y, "t": p.t} for p in trajectory.positions
        ])
        feature = trajectory.to_feature_vector().tobytes()
        status = "known" if subject else "unknown"
        angle = trajectory.direction_angle
        entry = f"{trajectory.entry_zone[0]},{trajectory.entry_zone[1]}"
        exit_ = f"{trajectory.exit_zone[0]},{trajectory.exit_zone[1]}"

        cur = self._conn.execute(
            """INSERT INTO trajectories
               (camera, detected_at, positions_json, feature_vector, subject,
                status, direction_angle, speed, entry_zone, exit_zone)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trajectory.camera,
                datetime.now(timezone.utc).isoformat(),
                positions_json,
                feature,
                subject,
                status,
                angle,
                trajectory.speed_px_per_sec,
                entry,
                exit_,
            ),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def list_unknown(self, limit: int = 50) -> list[TrajectoryRow]:
        rows = self._conn.execute(
            """SELECT * FROM trajectories WHERE status = 'unknown'
               ORDER BY detected_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [TrajectoryRow(**dict(r)) for r in rows]

    def get(self, trajectory_id: int) -> TrajectoryRow | None:
        row = self._conn.execute(
            "SELECT * FROM trajectories WHERE id = ?", (trajectory_id,)
        ).fetchone()
        return TrajectoryRow(**dict(row)) if row else None

    def label(self, trajectory_id: int, subject: str) -> None:
        self._conn.execute(
            "UPDATE trajectories SET subject = ?, status = 'known' WHERE id = ?",
            (subject, trajectory_id),
        )
        self._conn.commit()

    def discard(self, trajectory_id: int) -> None:
        self._conn.execute(
            "UPDATE trajectories SET status = 'discarded' WHERE id = ?",
            (trajectory_id,),
        )
        self._conn.commit()

    def known_by_camera(self) -> dict[str, dict[str, list[np.ndarray]]]:
        """Return {camera: {subject: [feature_vectors]}}."""
        rows = self._conn.execute(
            "SELECT * FROM trajectories WHERE status = 'known' AND subject IS NOT NULL"
        ).fetchall()
        result: dict[str, dict[str, list[np.ndarray]]] = {}
        for row in rows:
            r = TrajectoryRow(**dict(row))
            cam_dict = result.setdefault(r.camera, {})
            cam_dict.setdefault(r.subject or "", []).append(r.feature_array())
        return result

    def prune_older_than(self, days: int = 30) -> int:
        cutoff = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """DELETE FROM trajectories
               WHERE status IN ('unknown', 'discarded')
               AND detected_at < datetime(?, '-' || ? || ' days')""",
            (cutoff, days),
        )
        self._conn.commit()
        return cur.rowcount
