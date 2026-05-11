"""Person Re-Identification via OSNet (torchreid).

OSNet (Omni-Scale Network) is trained on Market-1501 / DukeMTMC to identify
the SAME person across different camera views. It produces a 512-d embedding
per person crop that captures clothing, build, posture — characteristics that
hold across face-angle/distance changes where face_recognition fails.

We use osnet_x0_25 (smallest variant, ~5MB) — sufficient for homelab single-
family use case, runs at ~50ms on CPU per crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass
class BodyMatch:
    matched_subject: str | None
    similarity: float
    """Cosine similarity to closest known embedding (-1..1, higher=more similar)."""

    @property
    def is_known(self) -> bool:
        return self.matched_subject is not None


class BodyReID:
    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.threshold = similarity_threshold
        self._model = None
        self._embeddings: dict[str, list[np.ndarray]] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchreid

        model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1000,
            pretrained=True,
        )
        model.eval()
        self._model = model
        self._torch = torch

    def reload(self, embeddings_by_subject: dict[str, list[np.ndarray]]) -> None:
        """Replace in-memory known-bodies cache."""
        self._embeddings = embeddings_by_subject

    def known_subjects(self) -> list[str]:
        return list(self._embeddings.keys())

    def embed(self, crop_bytes: bytes) -> np.ndarray:
        """Compute 512-d embedding for a person crop JPEG."""
        self._ensure_loaded()
        assert self._model is not None
        assert self._torch is not None

        img = Image.open(BytesIO(crop_bytes)).convert("RGB")
        # OSNet expects 256x128 (H x W) — resize, normalize ImageNet mean/std
        img = img.resize((128, 256))
        arr = np.array(img).astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        # HWC -> CHW, add batch dim
        tensor = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with self._torch.no_grad():
            features = self._model(tensor)
        emb = features[0].numpy()
        # L2-normalize for cosine similarity
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def match(self, embedding: np.ndarray) -> BodyMatch:
        if not self._embeddings:
            return BodyMatch(matched_subject=None, similarity=0.0)

        best_subject: str | None = None
        best_sim = -1.0
        for subject, known_embs in self._embeddings.items():
            for known in known_embs:
                sim = float(np.dot(embedding, known))
                if sim > best_sim:
                    best_sim = sim
                    best_subject = subject

        if best_sim >= self.threshold:
            return BodyMatch(matched_subject=best_subject, similarity=best_sim)
        return BodyMatch(matched_subject=None, similarity=best_sim)
