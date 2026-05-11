"""Face detection + recognition using dlib (via face_recognition library).

Embeddings for known subjects are loaded into memory at startup. Each
incoming snapshot triggers face_locations + face_encodings, then we compute
distances against the in-memory cache.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .db import FaceDB

if TYPE_CHECKING:
    pass


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]
    """(top, right, bottom, left) in pixels."""
    width_px: int
    matched_subject: str | None
    distance: float
    """Cosine distance to closest known embedding. Lower = better match."""

    @property
    def is_known(self) -> bool:
        return self.matched_subject is not None


class FaceRecognizer:
    def __init__(self, db: FaceDB, tolerance: float, min_face_width_px: int) -> None:
        self.db = db
        self.tolerance = tolerance
        self.min_face_width_px = min_face_width_px
        self._embeddings: dict[str, list[np.ndarray]] = {}
        self.reload()

    def reload(self) -> None:
        embeddings: dict[str, list[np.ndarray]] = {}
        for row in self.db.all():
            embeddings.setdefault(row.subject, []).append(row.embedding)
        self._embeddings = embeddings

    def known_subjects(self) -> list[str]:
        return list(self._embeddings.keys())

    def recognize(self, image_bytes: bytes) -> list[DetectedFace]:
        import face_recognition

        image = self._load_image(image_bytes)
        locations = face_recognition.face_locations(image, model="hog")
        if not locations:
            return []
        encodings = face_recognition.face_encodings(image, known_face_locations=locations)

        results: list[DetectedFace] = []
        for bbox, encoding in zip(locations, encodings, strict=True):
            top, right, bottom, left = bbox
            width = right - left
            matched, distance = self._match(encoding)
            results.append(
                DetectedFace(
                    bbox=bbox,
                    width_px=width,
                    matched_subject=matched,
                    distance=distance,
                )
            )
        return results

    def encode_for_training(self, image_bytes: bytes) -> np.ndarray | None:
        """Returns the encoding of the LARGEST face in the image, or None."""
        import face_recognition

        image = self._load_image(image_bytes)
        locations = face_recognition.face_locations(image, model="hog")
        if not locations:
            return None
        largest = max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        encodings = face_recognition.face_encodings(image, known_face_locations=[largest])
        if not encodings:
            return None
        return encodings[0]

    @staticmethod
    def _load_image(image_bytes: bytes) -> np.ndarray:
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(img)

    def _match(self, encoding: np.ndarray) -> tuple[str | None, float]:
        import face_recognition

        best_subject: str | None = None
        best_distance = float("inf")
        for subject, known_encodings in self._embeddings.items():
            distances = face_recognition.face_distance(known_encodings, encoding)
            min_dist = float(distances.min())
            if min_dist < best_distance:
                best_distance = min_dist
                best_subject = subject
        if best_distance > self.tolerance:
            return None, best_distance
        return best_subject, best_distance


def save_training_photo(data_dir: Path, subject: str, filename: str, image_bytes: bytes) -> Path:
    subject_dir = data_dir / "photos" / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    dest = subject_dir / filename
    dest.write_bytes(image_bytes)
    return dest
