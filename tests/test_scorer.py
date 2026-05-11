"""Decision logic tests — verifies multi-signal scoring rules."""

from __future__ import annotations

from datetime import datetime

import pytest

from home_watcher.config import CameraConfig
from home_watcher.decision.scorer import Decision, ScoringContext, decide
from home_watcher.faces.recognizer import DetectedFace


def _face(known: bool = True, width: int = 100, subject: str | None = "Malin") -> DetectedFace:
    return DetectedFace(
        bbox=(0, width, 100, 0),
        width_px=width,
        matched_subject=subject if known else None,
        distance=0.3 if known else 0.8,
    )


def _ctx(
    *,
    camera_name: str = "Entrance",
    smart_detect_types: list[str] | None = None,
    faces: list[DetectedFace] | None = None,
    now: datetime | None = None,
    family_at_home: bool = True,
    camera_cfg: CameraConfig | None = None,
) -> ScoringContext:
    return ScoringContext(
        camera_id="cam1",
        camera_name=camera_name,
        smart_detect_types=smart_detect_types or ["person"],
        faces=faces or [],
        now=now or datetime(2026, 5, 11, 14, 0),  # midday, default
        family_at_home=family_at_home,
        camera_cfg=camera_cfg or CameraConfig(),
    )


def test_vehicle_on_entrance_always_alerts() -> None:
    cfg = CameraConfig(always_alert_objects=["vehicle"])
    result = decide(
        _ctx(smart_detect_types=["vehicle"], camera_cfg=cfg),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.ALERT
    assert result.score == 1.0
    assert "always_alert: vehicle" in result.reasons[0]


def test_vehicle_on_other_cameras_ignored() -> None:
    result = decide(
        _ctx(smart_detect_types=["vehicle"], camera_cfg=CameraConfig()),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.IGNORE


def test_animal_notifies() -> None:
    result = decide(
        _ctx(smart_detect_types=["animal"]),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.NOTIFY_ANIMAL


def test_known_family_silent() -> None:
    result = decide(
        _ctx(smart_detect_types=["person"], faces=[_face(known=True)]),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.KNOWN_FAMILY
    assert "Malin" in result.matched_subjects


def test_unknown_face_at_daytime_with_family_home() -> None:
    """Unknown face = 0.7, family home = 0, midday = 0 → above 0.6 threshold."""
    result = decide(
        _ctx(faces=[_face(known=False)], family_at_home=True),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.ALERT
    assert result.score >= 0.7


def test_no_face_visible_with_family_home_daytime_low_camera_weight() -> None:
    """No face = 0.3, family home = 0, midday = 0, default camera = 0 → below 0.6."""
    result = decide(
        _ctx(faces=[], family_at_home=True),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.SILENT


def test_no_face_at_night_no_family_home() -> None:
    """No face = 0.3, no family = 0.3, night = 0.4 → 1.0 = alert."""
    result = decide(
        _ctx(
            faces=[],
            now=datetime(2026, 5, 11, 3, 0),
            family_at_home=False,
        ),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.ALERT
    assert result.score >= 0.9


def test_small_face_under_threshold_treated_as_no_face() -> None:
    """Face width 40px (< 60px threshold) means we can't trust the match."""
    small_known = _face(known=True, width=40)
    result = decide(
        _ctx(faces=[small_known], family_at_home=True),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    # Not KNOWN_FAMILY because face is too small to trust
    assert result.decision != Decision.KNOWN_FAMILY


def test_camera_alert_weight_increases_score() -> None:
    high_weight = CameraConfig(alert_weight=0.5)
    result = decide(
        _ctx(faces=[], family_at_home=True, camera_cfg=high_weight),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    # No-face 0.3 + weight 0.5 = 0.8 → alert
    assert result.decision == Decision.ALERT


def test_mixed_faces_known_and_unknown_alerts() -> None:
    """If any usable face is unknown, alert."""
    result = decide(
        _ctx(faces=[_face(known=True), _face(known=False)]),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.ALERT


def test_no_relevant_object_ignored() -> None:
    result = decide(
        _ctx(smart_detect_types=["package"]),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.IGNORE


@pytest.mark.parametrize("hour", [22, 23, 0, 3, 5])
def test_night_hours_add_score(hour: int) -> None:
    result = decide(
        _ctx(faces=[], family_at_home=True, now=datetime(2026, 5, 11, hour, 0)),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    # No face 0.3 + night 0.4 = 0.7
    assert result.score >= 0.7
