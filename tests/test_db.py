"""FaceDB tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from home_watcher.faces.db import FaceDB


@pytest.fixture
def db(tmp_path: Path) -> FaceDB:
    return FaceDB(tmp_path / "faces.db")


def test_add_and_retrieve(db: FaceDB) -> None:
    embedding = np.random.rand(128).astype(np.float32)
    row_id = db.add("Malin", "photo1.jpg", embedding)
    assert row_id > 0

    rows = db.all()
    assert len(rows) == 1
    assert rows[0].subject == "Malin"
    assert rows[0].photo_filename == "photo1.jpg"
    np.testing.assert_array_almost_equal(rows[0].embedding, embedding)


def test_list_subjects_counts(db: FaceDB) -> None:
    for i in range(3):
        db.add("Malin", f"m{i}.jpg", np.random.rand(128).astype(np.float32))
    for i in range(2):
        db.add("Anna", f"a{i}.jpg", np.random.rand(128).astype(np.float32))

    subjects = db.list_subjects()
    assert subjects == {"Malin": 3, "Anna": 2}


def test_delete_subject_removes_all_rows(db: FaceDB) -> None:
    for i in range(3):
        db.add("Malin", f"m{i}.jpg", np.random.rand(128).astype(np.float32))
    db.add("Anna", "a.jpg", np.random.rand(128).astype(np.float32))

    deleted = db.delete_subject("Malin")
    assert deleted == 3
    assert db.list_subjects() == {"Anna": 1}


def test_embedding_dtype_preserved(db: FaceDB) -> None:
    embedding = np.linspace(0, 1, 128, dtype=np.float32)
    db.add("Test", "p.jpg", embedding)
    rows = db.all()
    assert rows[0].embedding.dtype == np.float32
