"""PetDB tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from home_watcher.pets.db import PetDB


@pytest.fixture
def db(tmp_path: Path) -> PetDB:
    return PetDB(tmp_path / "pets.db")


def test_add_unknown_and_retrieve(db: PetDB) -> None:
    pid = db.add_unknown(
        camera="Ocean",
        species="cat",
        confidence=0.82,
        crop_filename="c.jpg",
        snapshot_filename="s.jpg",
        bbox=(50, 250, 200, 100),
        width_px=150,
        height_px=150,
    )
    assert pid > 0
    row = db.get_unknown(pid)
    assert row is not None
    assert row.camera == "Ocean"
    assert row.species == "cat"
    assert row.confidence == pytest.approx(0.82)
    assert row.width_px == 150
    assert row.labeled == 0


def test_list_unlabeled_orders_by_newest(db: PetDB) -> None:
    for species in ("dog", "cat", "bird"):
        db.add_unknown(
            camera="x", species=species, confidence=0.5,
            crop_filename=f"{species}.jpg", snapshot_filename="s.jpg",
            bbox=(0, 100, 100, 0), width_px=100, height_px=100,
        )
    rows = db.list_unlabeled()
    assert [r.species for r in rows] == ["bird", "cat", "dog"]


def test_label_unknown_pet_removes_from_unlabeled(db: PetDB) -> None:
    pid = db.add_unknown(
        camera="x", species="dog", confidence=0.8,
        crop_filename="c.jpg", snapshot_filename="s.jpg",
        bbox=(0, 100, 100, 0), width_px=100, height_px=100,
    )
    db.mark_labeled(pid)
    assert db.list_unlabeled() == []


def test_known_pets_round_trip(db: PetDB) -> None:
    db.add_known("Bella", "dog", "bella1.jpg")
    db.add_known("Bella", "dog", "bella2.jpg")
    db.add_known("Whiskers", "cat", "w.jpg")
    assert db.list_known_subjects() == {"Bella": 2, "Whiskers": 1}
    n = db.delete_known_subject("Bella")
    assert n == 2
    assert db.list_known_subjects() == {"Whiskers": 1}


def test_discarded_pet_not_in_unlabeled(db: PetDB) -> None:
    pid = db.add_unknown(
        camera="x", species="dog", confidence=0.7,
        crop_filename="c.jpg", snapshot_filename="s.jpg",
        bbox=(0, 100, 100, 0), width_px=100, height_px=100,
    )
    db.mark_discarded(pid)
    assert db.list_unlabeled() == []
    row = db.get_unknown(pid)
    assert row is not None and row.labeled == 2
