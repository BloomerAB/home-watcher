"""Match skeleton profiles against known biometric profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SkeletonMatch:
    matched_subject: str | None
    similarity: float

    @property
    def is_known(self) -> bool:
        return self.matched_subject is not None


class SkeletonMatcher:
    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.threshold = similarity_threshold
        self._known: dict[str, list[np.ndarray]] = {}

    def reload(self, known: dict[str, list[np.ndarray]]) -> None:
        self._known = known

    def match(self, profile_vector: np.ndarray) -> SkeletonMatch:
        if not self._known:
            return SkeletonMatch(matched_subject=None, similarity=0.0)

        query_norm = float(np.linalg.norm(profile_vector))
        if query_norm == 0:
            return SkeletonMatch(matched_subject=None, similarity=0.0)
        query = profile_vector / query_norm

        best_subject: str | None = None
        best_sim = -1.0

        for subject, vectors in self._known.items():
            for known_vec in vectors:
                known_norm = float(np.linalg.norm(known_vec))
                if known_norm == 0:
                    continue
                sim = float(np.dot(query, known_vec / known_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_subject = subject

        if best_sim >= self.threshold:
            return SkeletonMatch(matched_subject=best_subject, similarity=best_sim)
        return SkeletonMatch(matched_subject=None, similarity=best_sim)
