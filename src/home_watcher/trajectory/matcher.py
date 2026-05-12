"""Match observed trajectories against known movement patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracker import Trajectory


@dataclass
class TrajectoryMatch:
    matched_subject: str | None
    similarity: float
    matched_camera: str | None = None

    @property
    def is_known(self) -> bool:
        return self.matched_subject is not None


class TrajectoryMatcher:
    def __init__(self, similarity_threshold: float = 0.80) -> None:
        self.threshold = similarity_threshold
        self._known: dict[str, dict[str, list[np.ndarray]]] = {}

    def reload(self, known: dict[str, dict[str, list[np.ndarray]]]) -> None:
        self._known = known

    def match(self, trajectory: Trajectory) -> TrajectoryMatch:
        """Match a trajectory against known patterns for its camera."""
        camera_patterns = self._known.get(trajectory.camera)
        if not camera_patterns:
            return TrajectoryMatch(matched_subject=None, similarity=0.0)

        query = trajectory.to_feature_vector()
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return TrajectoryMatch(matched_subject=None, similarity=0.0)
        query_normalized = query / query_norm

        best_subject: str | None = None
        best_sim = -1.0

        for subject, vectors in camera_patterns.items():
            for known_vec in vectors:
                known_norm = np.linalg.norm(known_vec)
                if known_norm == 0:
                    continue
                sim = float(np.dot(query_normalized, known_vec / known_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_subject = subject

        if best_sim >= self.threshold:
            return TrajectoryMatch(
                matched_subject=best_subject,
                similarity=best_sim,
                matched_camera=trajectory.camera,
            )
        return TrajectoryMatch(
            matched_subject=None,
            similarity=best_sim,
            matched_camera=trajectory.camera,
        )
