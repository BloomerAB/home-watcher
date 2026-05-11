"""Person Re-Identification baseline using torchvision's ResNet18.

Why ResNet18 baseline (not OSNet):
  - torchreid on PyPI is stuck at 0.2.5 (2019), incompatible with modern torch
  - Installing from git source adds significant build complexity
  - ResNet18 is in torchvision (already a transitive dep via ultralytics)
  - ResNet18 pretrained on ImageNet gives ~70-75% accuracy for person Re-ID
  - Good enough for homelab family identification, ships fast

If accuracy proves insufficient over time, we can swap to OSNet via:
  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
or hand-port the OSNet model definition into this repo.

Pipeline per snapshot:
  1. YOLO gives person bbox → crop region (already done in main.py)
  2. Resize crop to 224x224, ImageNet-normalize
  3. ResNet18 feature extractor (last fc layer removed) → 512-d embedding
  4. L2-normalize → cosine similarity vs known embeddings
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
    def __init__(self, similarity_threshold: float = 0.85) -> None:
        # 0.85 is conservative for ResNet18-baseline. Will tune from real data.
        self.threshold = similarity_threshold
        self._model = None
        self._torch = None
        self._embeddings: dict[str, list[np.ndarray]] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        import torch.nn as nn
        import torchvision.models as tv_models

        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        # Strip the final classification layer — use as feature extractor.
        # ResNet18.fc maps 512 -> 1000; we want the 512-d features instead.
        model.fc = nn.Identity()
        model.eval()
        self._model = model
        self._torch = torch

    def reload(self, embeddings_by_subject: dict[str, list[np.ndarray]]) -> None:
        """Replace in-memory known-bodies cache."""
        self._embeddings = embeddings_by_subject

    def known_subjects(self) -> list[str]:
        return list(self._embeddings.keys())

    def embed(self, crop_bytes: bytes) -> np.ndarray:
        """Compute 512-d L2-normalized embedding for a person crop JPEG."""
        self._ensure_loaded()
        assert self._model is not None
        assert self._torch is not None

        img = Image.open(BytesIO(crop_bytes)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        # ImageNet normalization (what ResNet18 was trained on)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        # HWC -> CHW, add batch dim
        tensor = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with self._torch.no_grad():
            features = self._model(tensor)
        emb = features[0].numpy().astype(np.float32)
        # L2-normalize for cosine similarity
        norm = float(np.linalg.norm(emb))
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
