"""UnknownFaceDB tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from home_watcher.faces.unknown_db import UnknownFaceDB


@pytest.fixture
def db(tmp_path: Path) -> UnknownFaceDB:
    return UnknownFaceDB(tmp_path / "unknown.db")


def test_add_and_retrieve(db: UnknownFaceDB) -> None:
    emb = np.random.rand(128).astype(np.float32)
    fid = db.add(
        camera="Entrance",
        crop_filename="c.jpg",
        snapshot_filename="s.jpg",
        bbox=(10, 200, 100, 100),
        width_px=100,
        embedding=emb,
    )
    assert fid > 0
    row = db.get(fid)
    assert row is not None
    assert row.camera == "Entrance"
    assert row.crop_filename == "c.jpg"
    assert row.width_px == 100
    np.testing.assert_array_almost_equal(row.embedding, emb)
    assert row.labeled == 0


def test_list_unlabeled_orders_by_newest(db: UnknownFaceDB) -> None:
    for cam in ("c1", "c2", "c3"):
        db.add(
            camera=cam,
            crop_filename=f"{cam}.jpg",
            snapshot_filename=f"{cam}-s.jpg",
            bbox=(0, 100, 100, 0),
            width_px=100,
            embedding=np.random.rand(128).astype(np.float32),
        )
    rows = db.list_unlabeled()
    assert [r.camera for r in rows] == ["c3", "c2", "c1"]


def test_mark_labeled_removes_from_unlabeled_list(db: UnknownFaceDB) -> None:
    fid = db.add(
        camera="x",
        crop_filename="c.jpg",
        snapshot_filename="s.jpg",
        bbox=(0, 100, 100, 0),
        width_px=100,
        embedding=np.random.rand(128).astype(np.float32),
    )
    db.mark_labeled(fid)
    assert db.list_unlabeled() == []
    row = db.get(fid)
    assert row is not None and row.labeled == 1


def test_mark_discarded(db: UnknownFaceDB) -> None:
    fid = db.add(
        camera="x",
        crop_filename="c.jpg",
        snapshot_filename="s.jpg",
        bbox=(0, 100, 100, 0),
        width_px=100,
        embedding=np.random.rand(128).astype(np.float32),
    )
    db.mark_discarded(fid)
    assert db.list_unlabeled() == []
    row = db.get(fid)
    assert row is not None and row.labeled == 2


def test_prune_older_than(db: UnknownFaceDB) -> None:
    db.add(
        camera="x",
        crop_filename="c.jpg",
        snapshot_filename="s.jpg",
        bbox=(0, 100, 100, 0),
        width_px=100,
        embedding=np.random.rand(128).astype(np.float32),
    )
    n = db.prune_older_than(datetime.now(UTC) + timedelta(days=1))
    assert n == 1
    assert db.list_unlabeled() == []
