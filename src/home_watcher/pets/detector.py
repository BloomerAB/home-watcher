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
    def __init__(self, model_name: str = "yolov8n.pt", min_confidence: float = 0.35) -> None:
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

    def detect(self, image_bytes: bytes) -> list[DetectedPet]:
        self._ensure_loaded()
        assert self._model is not None  # for type checkers

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)

        results = self._model(arr, verbose=False)
        if not results:
            return []
        result = results[0]
        if result.boxes is None:
            return []

        out: list[DetectedPet] = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            name = self._class_names.get(cls_id, str(cls_id))
            if name not in PET_CLASSES:
                continue
            if conf < self.min_confidence:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            out.append(
                DetectedPet(
                    species=name,
                    confidence=conf,
                    bbox=(y1, x2, y2, x1),
                    width_px=x2 - x1,
                    height_px=y2 - y1,
                )
            )
        return out

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
