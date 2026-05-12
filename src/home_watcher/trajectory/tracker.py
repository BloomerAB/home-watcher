"""Trajectory tracking from burst snapshots.

Takes multiple snapshots during a motion event and tracks person
positions across frames to build movement trajectories. Family members
follow predictable paths; strangers don't.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    t: float

    def distance_to(self, other: Position) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Trajectory:
    camera: str
    positions: list[Position]
    frame_width: int = 1920
    frame_height: int = 1080

    @property
    def direction_angle(self) -> float | None:
        """Movement direction in degrees (0=right, 90=down, 180=left, 270=up).
        None if no movement detected."""
        if len(self.positions) < 2:
            return None
        start = self.positions[0]
        end = self.positions[-1]
        dx = end.x - start.x
        dy = end.y - start.y
        if abs(dx) < 5 and abs(dy) < 5:
            return None
        return math.degrees(math.atan2(dy, dx)) % 360

    @property
    def speed_px_per_sec(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        start = self.positions[0]
        end = self.positions[-1]
        dt = end.t - start.t
        if dt <= 0:
            return 0.0
        return start.distance_to(end) / dt

    @property
    def entry_zone(self) -> tuple[int, int]:
        return self._to_zone(self.positions[0])

    @property
    def exit_zone(self) -> tuple[int, int]:
        return self._to_zone(self.positions[-1])

    @property
    def zone_sequence(self) -> list[tuple[int, int]]:
        """Sequence of zones visited, with consecutive duplicates removed."""
        zones = [self._to_zone(p) for p in self.positions]
        deduped: list[tuple[int, int]] = []
        for z in zones:
            if not deduped or deduped[-1] != z:
                deduped.append(z)
        return deduped

    @property
    def total_displacement(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        return self.positions[0].distance_to(self.positions[-1])

    @property
    def is_stationary(self) -> bool:
        return self.total_displacement < 20.0

    def _to_zone(self, pos: Position) -> tuple[int, int]:
        """Map position to a 3x3 grid zone (row, col)."""
        col = min(int(pos.x / (self.frame_width / 3)), 2)
        row = min(int(pos.y / (self.frame_height / 3)), 2)
        return (row, col)

    def to_feature_vector(self) -> np.ndarray:
        """Compact representation for similarity matching."""
        angle = self.direction_angle
        return np.array([
            self.positions[0].x / self.frame_width,
            self.positions[0].y / self.frame_height,
            self.positions[-1].x / self.frame_width if len(self.positions) > 1 else 0.0,
            self.positions[-1].y / self.frame_height if len(self.positions) > 1 else 0.0,
            math.cos(math.radians(angle)) if angle is not None else 0.0,
            math.sin(math.radians(angle)) if angle is not None else 0.0,
            min(self.speed_px_per_sec / 500.0, 1.0),
            float(self.entry_zone[0]) / 2.0,
            float(self.entry_zone[1]) / 2.0,
            float(self.exit_zone[0]) / 2.0,
            float(self.exit_zone[1]) / 2.0,
        ], dtype=np.float32)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Extract center point from YOLO bbox (top, right, bottom, left)."""
    top, right, bottom, left = bbox
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def associate_detections(
    prev_positions: list[tuple[float, float]],
    curr_bboxes: list[tuple[tuple[int, int, int, int], float]],
) -> list[int | None]:
    """Match current detections to previous positions by nearest neighbor.

    Returns a list parallel to curr_bboxes where each element is the index
    into prev_positions that this detection corresponds to, or None if new.
    """
    if not prev_positions or not curr_bboxes:
        return [None] * len(curr_bboxes)

    curr_centers = [bbox_center(bbox) for bbox, _ in curr_bboxes]
    used_prev: set[int] = set()
    assignments: list[int | None] = [None] * len(curr_bboxes)

    pairs: list[tuple[float, int, int]] = []
    for ci, cc in enumerate(curr_centers):
        for pi, pc in enumerate(prev_positions):
            dist = math.hypot(cc[0] - pc[0], cc[1] - pc[1])
            pairs.append((dist, ci, pi))
    pairs.sort()

    used_curr: set[int] = set()
    for dist, ci, pi in pairs:
        if ci in used_curr or pi in used_prev:
            continue
        if dist > 400:
            continue
        assignments[ci] = pi
        used_curr.add(ci)
        used_prev.add(pi)

    return assignments


@dataclass
class BurstTracker:
    """Tracks persons across burst snapshots to build trajectories."""
    camera: str
    frame_width: int = 1920
    frame_height: int = 1080
    _tracks: dict[int, list[Position]] = field(default_factory=dict)
    _last_positions: list[tuple[float, float]] = field(default_factory=list)
    _next_track_id: int = 0

    def add_frame(
        self,
        bboxes: list[tuple[tuple[int, int, int, int], float]],
        timestamp: float,
    ) -> None:
        """Add a frame's person detections to the tracker."""
        if not bboxes:
            return

        assignments = associate_detections(self._last_positions, bboxes)

        new_positions: list[tuple[float, float]] = []
        for i, (bbox, _conf) in enumerate(bboxes):
            cx, cy = bbox_center(bbox)
            pos = Position(x=cx, y=cy, t=timestamp)

            prev_idx = assignments[i]
            if prev_idx is not None:
                track_ids = [
                    tid for tid, positions in self._tracks.items()
                    if len(positions) > 0
                    and abs(positions[-1].x - self._last_positions[prev_idx][0]) < 1
                    and abs(positions[-1].y - self._last_positions[prev_idx][1]) < 1
                ]
                if track_ids:
                    self._tracks[track_ids[0]].append(pos)
                else:
                    self._tracks[self._next_track_id] = [pos]
                    self._next_track_id += 1
            else:
                self._tracks[self._next_track_id] = [pos]
                self._next_track_id += 1

            new_positions.append((cx, cy))

        self._last_positions = new_positions

    def get_trajectories(self) -> list[Trajectory]:
        """Return all tracked trajectories with ≥2 positions."""
        return [
            Trajectory(
                camera=self.camera,
                positions=positions,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )
            for positions in self._tracks.values()
            if len(positions) >= 2
        ]
