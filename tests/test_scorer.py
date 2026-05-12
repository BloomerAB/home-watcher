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
    body_person_count: int = 0,
    body_matches: list[str] | None = None,
    family_members_home: list[str] | None = None,
) -> ScoringContext:
    members = family_members_home or (["Malin"] if family_at_home else [])
    return ScoringContext(
        camera_id="cam1",
        camera_name=camera_name,
        smart_detect_types=smart_detect_types or ["person"],
        faces=faces or [],
        now=now or datetime(2026, 5, 11, 14, 0),
        family_at_home=family_at_home,
        camera_cfg=camera_cfg or CameraConfig(),
        body_person_count=body_person_count,
        body_matches=body_matches or [],
        family_members_home=members,
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


def test_no_face_visible_with_family_home_known_via_presence() -> None:
    """No face but family phone on WiFi + 1 person → presence-count match."""
    result = decide(
        _ctx(faces=[], family_at_home=True),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.KNOWN_FAMILY


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


def test_small_face_under_threshold_still_known_via_presence() -> None:
    """Face too small to trust, but presence-count still matches → known family."""
    small_known = _face(known=True, width=40)
    result = decide(
        _ctx(faces=[small_known], family_at_home=True),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.KNOWN_FAMILY


def test_small_face_no_family_home_not_known() -> None:
    """Face too small and no phones home → not KNOWN_FAMILY."""
    small_known = _face(known=True, width=40)
    result = decide(
        _ctx(faces=[small_known], family_at_home=False, family_members_home=[]),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision != Decision.KNOWN_FAMILY


def test_camera_alert_weight_increases_score() -> None:
    """Camera weight matters when no family phones are home."""
    high_weight = CameraConfig(alert_weight=0.5)
    result = decide(
        _ctx(faces=[], family_at_home=False, family_members_home=[], camera_cfg=high_weight),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
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
        _ctx(
            faces=[],
            now=datetime(2026, 5, 11, hour, 0),
            family_at_home=False,
            family_members_home=[],
        ),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.score >= 0.7


def test_presence_count_1_person_2_phones_known_family() -> None:
    """1 person detected, 2 family phones home → known family."""
    result = decide(
        _ctx(
            body_person_count=1,
            family_members_home=["Malin", "Madde"],
        ),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.KNOWN_FAMILY
    assert "presence-count" in result.reasons[0]


def test_presence_count_equal_persons_and_phones() -> None:
    """3 persons, 3 phones → all accounted for."""
    result = decide(
        _ctx(
            body_person_count=3,
            family_members_home=["Malin", "Madde", "Loe"],
        ),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.KNOWN_FAMILY


def test_presence_count_more_persons_than_phones_alerts() -> None:
    """4 persons but only 2 phones → 2 unknown people."""
    result = decide(
        _ctx(
            body_person_count=4,
            family_members_home=["Malin", "Madde"],
        ),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.ALERT
    assert any("beyond" in r for r in result.reasons)


def test_presence_count_no_phones_home_alerts() -> None:
    """Person detected, no family phones → alert."""
    result = decide(
        _ctx(
            body_person_count=1,
            family_at_home=False,
            family_members_home=[],
        ),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision == Decision.ALERT


def test_presence_count_blocked_by_unknown_face() -> None:
    """Even if phones match count, unknown face overrides."""
    result = decide(
        _ctx(
            body_person_count=1,
            faces=[_face(known=False)],
            family_members_home=["Malin"],
        ),
        alert_threshold=0.6,
        min_face_width_px=60,
    )
    assert result.decision != Decision.KNOWN_FAMILY


def test_known_vehicle_on_entrance_silent() -> None:
    """Known family car on always-alert camera → KNOWN_FAMILY, not ALERT."""
    cfg = CameraConfig(always_alert_objects=["vehicle"])
    ctx = _ctx(smart_detect_types=["vehicle"], camera_cfg=cfg)
    ctx.vehicle_matches = ["Malins bil"]
    ctx.vehicle_count = 1
    result = decide(ctx, alert_threshold=0.6, min_face_width_px=60)
    assert result.decision == Decision.KNOWN_FAMILY
    assert "Malins bil" in result.matched_subjects


def test_unknown_vehicle_on_entrance_alerts() -> None:
    """Unknown vehicle on always-alert camera → ALERT."""
    cfg = CameraConfig(always_alert_objects=["vehicle"])
    ctx = _ctx(smart_detect_types=["vehicle"], camera_cfg=cfg)
    ctx.vehicle_matches = []
    ctx.vehicle_count = 1
    result = decide(ctx, alert_threshold=0.6, min_face_width_px=60)
    assert result.decision == Decision.ALERT


def test_mixed_vehicles_on_entrance_alerts() -> None:
    """One known + one unknown vehicle on always-alert camera → ALERT."""
    cfg = CameraConfig(always_alert_objects=["vehicle"])
    ctx = _ctx(smart_detect_types=["vehicle"], camera_cfg=cfg)
    ctx.vehicle_matches = ["Malins bil"]
    ctx.vehicle_count = 2
    result = decide(ctx, alert_threshold=0.6, min_face_width_px=60)
    assert result.decision == Decision.ALERT


def test_known_vehicle_on_other_camera_known_family() -> None:
    """Known vehicle on non-driveway camera → KNOWN_FAMILY."""
    ctx = _ctx(smart_detect_types=["vehicle"], camera_cfg=CameraConfig())
    ctx.vehicle_matches = ["Malins bil"]
    ctx.vehicle_count = 1
    result = decide(ctx, alert_threshold=0.6, min_face_width_px=60)
    assert result.decision == Decision.KNOWN_FAMILY
