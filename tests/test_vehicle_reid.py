"""Vehicle ReID + VehicleDB tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from home_watcher.vehicles.db import VehicleDB
from home_watcher.vehicles.reid import VehicleMatch, VehicleReID


@pytest.fixture
def db(tmp_path: Path) -> VehicleDB:
    return VehicleDB(tmp_path / "vehicles.db")


def _fake_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestVehicleReID:
    def test_match_empty_returns_unknown(self) -> None:
        reid = VehicleReID()
        result = reid.match(_fake_embedding(0))
        assert not result.is_known
        assert result.similarity == 0.0

    def test_match_above_threshold(self) -> None:
        reid = VehicleReID(similarity_threshold=0.5)
        emb = _fake_embedding(42)
        noisy = emb + np.random.default_rng(1).standard_normal(512).astype(np.float32) * 0.02
        noisy = noisy / np.linalg.norm(noisy)
        reid.reload({"Malins bil": [emb]})
        result = reid.match(noisy)
        assert result.is_known
        assert result.matched_subject == "Malins bil"
        assert result.similarity > 0.5

    def test_match_below_threshold(self) -> None:
        reid = VehicleReID(similarity_threshold=0.99)
        reid.reload({"Malins bil": [_fake_embedding(1)]})
        result = reid.match(_fake_embedding(999))
        assert not result.is_known

    def test_best_match_among_multiple(self) -> None:
        reid = VehicleReID(similarity_threshold=0.3)
        target = _fake_embedding(10)
        close = target + np.random.default_rng(2).standard_normal(512).astype(np.float32) * 0.05
        close = close / np.linalg.norm(close)
        reid.reload({
            "Malins bil": [_fake_embedding(100)],
            "Maddes bil": [target],
        })
        result = reid.match(close)
        assert result.matched_subject == "Maddes bil"


class TestVehicleDB:
    def test_add_unknown_and_list(self, db: VehicleDB) -> None:
        vid = db.add_unknown(
            camera="Entrance",
            vehicle_class="car",
            confidence=0.85,
            crop_filename="crop.jpg",
            snapshot_filename="snap.jpg",
            bbox=(100, 300, 250, 50),
            width_px=250,
            height_px=150,
        )
        assert vid > 0
        unlabeled = db.list_unlabeled()
        assert len(unlabeled) == 1
        assert unlabeled[0].vehicle_class == "car"

    def test_mark_labeled(self, db: VehicleDB) -> None:
        vid = db.add_unknown(
            camera="Entrance",
            vehicle_class="truck",
            confidence=0.90,
            crop_filename="c.jpg",
            snapshot_filename="s.jpg",
            bbox=(10, 30, 25, 5),
            width_px=25,
            height_px=15,
        )
        db.mark_labeled(vid)
        assert len(db.list_unlabeled()) == 0

    def test_mark_discarded(self, db: VehicleDB) -> None:
        vid = db.add_unknown(
            camera="Entrance",
            vehicle_class="bus",
            confidence=0.70,
            crop_filename="c.jpg",
            snapshot_filename="s.jpg",
            bbox=(10, 30, 25, 5),
            width_px=25,
            height_px=15,
        )
        db.mark_discarded(vid)
        assert len(db.list_unlabeled()) == 0

    def test_add_known_with_embedding(self, db: VehicleDB) -> None:
        emb = _fake_embedding(5)
        db.add_known("Malins bil", "car", "crop.jpg", emb)
        known = db.all_known_by_subject()
        assert "Malins bil" in known
        assert len(known["Malins bil"]) == 1
        np.testing.assert_allclose(known["Malins bil"][0], emb, atol=1e-6)

    def test_add_known_without_embedding_excluded(self, db: VehicleDB) -> None:
        db.add_known("Ghost car", "car", "ghost.jpg")
        known = db.all_known_by_subject()
        assert "Ghost car" not in known

    def test_list_known_subjects(self, db: VehicleDB) -> None:
        db.add_known("Malins bil", "car", "a.jpg", _fake_embedding(1))
        db.add_known("Malins bil", "car", "b.jpg", _fake_embedding(2))
        db.add_known("Maddes bil", "car", "c.jpg", _fake_embedding(3))
        subjects = db.list_known_subjects()
        assert subjects == {"Malins bil": 2, "Maddes bil": 1}

    def test_delete_known_subject(self, db: VehicleDB) -> None:
        db.add_known("Malins bil", "car", "a.jpg", _fake_embedding(1))
        deleted = db.delete_known_subject("Malins bil")
        assert deleted == 1
        assert db.list_known_subjects() == {}
