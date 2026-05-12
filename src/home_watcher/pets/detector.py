"""Pet detection via YOLOv8 (ultralytics).

Runs YOLO on snapshots to find animals. We use the nano model (~6MB on disk,
~100ms per inference on CPU) which is plenty for homelab use.

YOLOv8 returns COCO classes; we filter to animals: cat, dog, bird, horse,
sheep, cow. The model is loaded lazily at first inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

PET_CLASSES = {
    "cat", "dog", "bird", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

PERSON_CLASS = "person"


@dataclass
class DetectedPet:
    species: str
    confidence: float
    bbox: tuple[int, int, int, int]
    """(top, right, bottom, left) in pixels."""
    width_px: int
    height_px: int

    @property
    def area_px(self) -> int:
        return self.width_px * self.height_px


class PetDetector:
    def __init__(self, model_name: str = "yolov8n.pt", min_confidence: float = 0.50) -> None:
        self.model_name = model_name
        self.min_confidence = min_confidence
        self._model = None
        self._class_names: dict[int, str] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)
        self._class_names = dict(self._model.names) if hasattr(self._model, "names") else {}

    def _run(self, image_bytes: bytes) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Single-pass YOLO inference. Returns (class, conf, (y1,x2,y2,x1)) tuples
        for all detections above min_confidence regardless of class."""
        self._ensure_loaded()
        assert self._model is not None

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        results = self._model(arr, verbose=False)
        if not results or results[0].boxes is None:
            return []

        out: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if conf < self.min_confidence:
                continue
            name = self._class_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            out.append((name, conf, (y1, x2, y2, x1)))
        return out

    def detect(self, image_bytes: bytes) -> list[DetectedPet]:
        """Detect animals only."""
        return [
            DetectedPet(
                species=name,
                confidence=conf,
                bbox=bbox,
                width_px=bbox[1] - bbox[3],
                height_px=bbox[2] - bbox[0],
            )
            for name, conf, bbox in self._run(image_bytes)
            if name in PET_CLASSES
        ]

    def detect_vehicles(self, image_bytes: bytes) -> list[DetectedPet]:
        """Detect vehicles (car, truck, bus, motorcycle)."""
        return [
            DetectedPet(
                species=name,
                confidence=conf,
                bbox=bbox,
                width_px=bbox[1] - bbox[3],
                height_px=bbox[2] - bbox[0],
            )
            for name, conf, bbox in self._run(image_bytes)
            if name in VEHICLE_CLASSES
        ]

    def detect_persons(self, image_bytes: bytes) -> int:
        """Returns count of persons detected above min_confidence."""
        return sum(1 for name, _, _ in self._run(image_bytes) if name == PERSON_CLASS)

    def detect_person_bboxes(
        self, image_bytes: bytes
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Returns list of (bbox, confidence) for each person detected.

        bbox is (top, right, bottom, left) in pixels. Same convention as DetectedPet.
        """
        return [
            (bbox, conf)
            for name, conf, bbox in self._run(image_bytes)
            if name == PERSON_CLASS
        ]

    def detect_all(self, image_bytes: bytes) -> dict[str, int]:
        """Returns map of {class_name: count} for all relevant classes."""
        counts: dict[str, int] = {}
        for name, _, _ in self._run(image_bytes):
            if name in PET_CLASSES or name == PERSON_CLASS or name in ("car", "truck", "bus"):
                counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def crop(image_bytes: bytes, bbox: tuple[int, int, int, int], pad: int = 20) -> bytes:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        top, right, bottom, left = bbox
        w, h = img.size
        crop = img.crop((
            max(0, left - pad),
            max(0, top - pad),
            min(w, right + pad),
            min(h, bottom + pad),
        ))
        out = BytesIO()
        crop.save(out, format="JPEG", quality=85)
        return out.getvalue()
