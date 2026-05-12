"""Pet Re-Identification using ResNet18 embeddings.

Same approach as body ReID — ResNet18 feature extractor gives 512-d
embedding per animal crop. Cosine similarity for matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass
class PetMatch:
    matched_subject: str | None
    species: str | None
    similarity: float

    @property
    def is_known(self) -> bool:
        return self.matched_subject is not None


class PetReID:
    def __init__(self, similarity_threshold: float = 0.60) -> None:
        self.threshold = similarity_threshold
        self._model = None
        self._torch = None
        self._embeddings: dict[str, list[np.ndarray]] = {}
        self._species: dict[str, str] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        import torch.nn as nn
        import torchvision.models as tv_models

        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model.eval()
        self._model = model
        self._torch = torch

    def reload(
        self,
        embeddings_by_subject: dict[str, list[np.ndarray]],
        species_by_subject: dict[str, str] | None = None,
    ) -> None:
        self._embeddings = embeddings_by_subject
        self._species = species_by_subject or {}

    def embed(self, crop_bytes: bytes) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        assert self._torch is not None

        img = Image.open(BytesIO(crop_bytes)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with self._torch.no_grad():
            features = self._model(tensor)
        emb = features[0].numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb

    def match(self, embedding: np.ndarray) -> PetMatch:
        if not self._embeddings:
            return PetMatch(matched_subject=None, species=None, similarity=0.0)

        best_subject: str | None = None
        best_sim = -1.0
        for subject, known_embs in self._embeddings.items():
            for known in known_embs:
                sim = float(np.dot(embedding, known))
                if sim > best_sim:
                    best_sim = sim
                    best_subject = subject

        if best_sim >= self.threshold and best_subject:
            return PetMatch(
                matched_subject=best_subject,
                species=self._species.get(best_subject),
                similarity=best_sim,
            )
        return PetMatch(matched_subject=None, species=None, similarity=best_sim)
