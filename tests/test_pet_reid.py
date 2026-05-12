"""Pet ReID + PetDB embedding tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from home_watcher.pets.db import PetDB
from home_watcher.pets.reid import PetMatch, PetReID


@pytest.fixture
def db(tmp_path: Path) -> PetDB:
    return PetDB(tmp_path / "pets.db")


def _fake_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestPetReID:
    def test_match_empty_returns_unknown(self) -> None:
        reid = PetReID()
        result = reid.match(_fake_embedding(0))
        assert not result.is_known
        assert result.similarity == 0.0

    def test_match_above_threshold(self) -> None:
        reid = PetReID(similarity_threshold=0.5)
        emb = _fake_embedding(42)
        noisy = emb + np.random.default_rng(1).standard_normal(512).astype(np.float32) * 0.02
        noisy = noisy / np.linalg.norm(noisy)
        reid.reload({"Bella": [emb]}, {"Bella": "dog"})
        result = reid.match(noisy)
        assert result.is_known
        assert result.matched_subject == "Bella"
        assert result.species == "dog"
        assert result.similarity > 0.5

    def test_match_below_threshold(self) -> None:
        reid = PetReID(similarity_threshold=0.99)
        reid.reload({"Bella": [_fake_embedding(1)]})
        result = reid.match(_fake_embedding(999))
        assert not result.is_known

    def test_best_match_among_multiple(self) -> None:
        reid = PetReID(similarity_threshold=0.3)
        target = _fake_embedding(10)
        close = target + np.random.default_rng(2).standard_normal(512).astype(np.float32) * 0.05
        close = close / np.linalg.norm(close)
        reid.reload({
            "Bella": [_fake_embedding(100)],
            "Whiskers": [target],
        })
        result = reid.match(close)
        assert result.matched_subject == "Whiskers"


class TestPetDBEmbeddings:
    def test_add_known_with_embedding(self, db: PetDB) -> None:
        emb = _fake_embedding(5)
        db.add_known("Bella", "dog", "bella.jpg", emb)
        known = db.all_known_by_subject()
        assert "Bella" in known
        assert len(known["Bella"]) == 1
        np.testing.assert_allclose(known["Bella"][0], emb, atol=1e-6)

    def test_add_known_without_embedding_excluded(self, db: PetDB) -> None:
        db.add_known("Ghost", "cat", "ghost.jpg")
        known = db.all_known_by_subject()
        assert "Ghost" not in known

    def test_species_by_subject(self, db: PetDB) -> None:
        db.add_known("Bella", "dog", "b.jpg", _fake_embedding(1))
        db.add_known("Whiskers", "cat", "w.jpg", _fake_embedding(2))
        species = db.species_by_subject()
        assert species == {"Bella": "dog", "Whiskers": "cat"}
