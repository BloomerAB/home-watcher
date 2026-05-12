"""Trajectory tracking and matching tests."""

from __future__ import annotations

import numpy as np

from home_watcher.trajectory.tracker import (
    BurstTracker,
    Position,
    Trajectory,
    associate_detections,
    bbox_center,
)
from home_watcher.trajectory.matcher import TrajectoryMatcher


def test_bbox_center() -> None:
    assert bbox_center((100, 200, 300, 0)) == (100.0, 200.0)


def test_trajectory_direction_right() -> None:
    t = Trajectory(
        camera="test",
        positions=[Position(0, 500, 0), Position(500, 500, 2)],
    )
    assert t.direction_angle is not None
    assert abs(t.direction_angle - 0.0) < 5


def test_trajectory_direction_down() -> None:
    t = Trajectory(
        camera="test",
        positions=[Position(500, 0, 0), Position(500, 500, 2)],
    )
    assert t.direction_angle is not None
    assert abs(t.direction_angle - 90.0) < 5


def test_trajectory_speed() -> None:
    t = Trajectory(
        camera="test",
        positions=[Position(0, 0, 0), Position(300, 0, 3)],
    )
    assert abs(t.speed_px_per_sec - 100.0) < 1


def test_trajectory_stationary() -> None:
    t = Trajectory(
        camera="test",
        positions=[Position(500, 500, 0), Position(505, 500, 2)],
    )
    assert t.is_stationary


def test_trajectory_zone_mapping() -> None:
    t = Trajectory(
        camera="test",
        positions=[Position(100, 100, 0)],
        frame_width=1920,
        frame_height=1080,
    )
    assert t.entry_zone == (0, 0)

    t2 = Trajectory(
        camera="test",
        positions=[Position(1800, 900, 0)],
        frame_width=1920,
        frame_height=1080,
    )
    assert t2.entry_zone == (2, 2)


def test_trajectory_zone_sequence() -> None:
    t = Trajectory(
        camera="test",
        positions=[
            Position(100, 100, 0),
            Position(700, 100, 1),
            Position(1300, 100, 2),
        ],
        frame_width=1920,
        frame_height=1080,
    )
    seq = t.zone_sequence
    assert len(seq) == 3
    assert seq[0] == (0, 0)
    assert seq[-1] == (0, 2)


def test_feature_vector_shape() -> None:
    t = Trajectory(
        camera="test",
        positions=[Position(100, 200, 0), Position(800, 200, 2)],
    )
    vec = t.to_feature_vector()
    assert vec.shape == (11,)
    assert vec.dtype == np.float32


def test_associate_detections_nearest() -> None:
    prev = [(100.0, 100.0), (500.0, 500.0)]
    curr = [
        ((480, 520, 520, 480), 0.9),
        ((80, 120, 120, 80), 0.9),
    ]
    assignments = associate_detections(prev, curr)
    assert assignments[0] == 1
    assert assignments[1] == 0


def test_associate_detections_new_person() -> None:
    prev = [(100.0, 100.0)]
    curr = [
        ((80, 120, 120, 80), 0.9),
        ((900, 920, 920, 900), 0.9),
    ]
    assignments = associate_detections(prev, curr)
    assert assignments[0] == 0
    assert assignments[1] is None


def test_burst_tracker_builds_trajectories() -> None:
    tracker = BurstTracker(camera="Entrance")
    tracker.add_frame([((100, 200, 200, 100), 0.9)], 0.0)
    tracker.add_frame([((100, 250, 200, 150), 0.9)], 2.0)
    tracker.add_frame([((100, 300, 200, 200), 0.9)], 4.0)

    trajectories = tracker.get_trajectories()
    assert len(trajectories) == 1
    assert len(trajectories[0].positions) == 3
    assert trajectories[0].positions[0].x == 150.0
    assert trajectories[0].positions[-1].x == 250.0


def test_matcher_no_known_patterns() -> None:
    matcher = TrajectoryMatcher()
    t = Trajectory(
        camera="test",
        positions=[Position(100, 200, 0), Position(800, 200, 2)],
    )
    result = matcher.match(t)
    assert not result.is_known
    assert result.similarity == 0.0


def test_matcher_with_known_pattern() -> None:
    t1 = Trajectory(
        camera="Entrance",
        positions=[Position(100, 200, 0), Position(800, 200, 2)],
    )
    vec = t1.to_feature_vector()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    matcher = TrajectoryMatcher(similarity_threshold=0.8)
    matcher.reload({"Entrance": {"Malin": [vec]}})

    result = matcher.match(t1)
    assert result.is_known
    assert result.matched_subject == "Malin"
    assert result.similarity > 0.99


def test_matcher_different_pattern_no_match() -> None:
    t_known = Trajectory(
        camera="Entrance",
        positions=[Position(100, 200, 0), Position(800, 200, 2)],
    )
    vec = t_known.to_feature_vector()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    matcher = TrajectoryMatcher(similarity_threshold=0.8)
    matcher.reload({"Entrance": {"Malin": [vec]}})

    t_new = Trajectory(
        camera="Entrance",
        positions=[Position(1800, 900, 0), Position(100, 100, 2)],
    )
    result = matcher.match(t_new)
    assert not result.is_known
