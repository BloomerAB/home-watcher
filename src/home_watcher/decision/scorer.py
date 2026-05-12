"""Multi-signal decision logic.

Combines face-recognition results with contextual signals (time of day,
family presence, per-camera weights, object-type rules) into a single
alert/silent decision.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import CameraConfig
from ..faces.recognizer import DetectedFace


class Decision(str, Enum):
    ALERT = "alert"
    SILENT = "silent"
    KNOWN_FAMILY = "known_family"
    NOTIFY_ANIMAL = "notify_animal"
    IGNORE = "ignore"


@dataclass
class ScoringContext:
    camera_id: str
    camera_name: str
    smart_detect_types: list[str]
    faces: list[DetectedFace]
    now: datetime
    family_at_home: bool
    camera_cfg: CameraConfig
    body_matches: list[str] = field(default_factory=list)
    body_person_count: int = 0
    family_members_home: list[str] = field(default_factory=list)
    trajectory_matches: list[str] = field(default_factory=list)
    skeleton_matches: list[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    decision: Decision
    score: float
    reasons: list[str] = field(default_factory=list)
    matched_subjects: list[str] = field(default_factory=list)


NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}


def decide(ctx: ScoringContext, *, alert_threshold: float, min_face_width_px: int) -> DecisionResult:
    reasons: list[str] = []

    # 1. Per-camera always-alert rules (short-circuit)
    for obj in ctx.smart_detect_types:
        if obj in ctx.camera_cfg.always_alert_objects:
            return DecisionResult(
                decision=Decision.ALERT,
                score=1.0,
                reasons=[f"always_alert: {obj} on {ctx.camera_name}"],
            )

    # 2. No person detected — handle animal / vehicle / nothing
    if "person" not in ctx.smart_detect_types:
        if "animal" in ctx.smart_detect_types:
            return DecisionResult(
                decision=Decision.NOTIFY_ANIMAL,
                score=0.5,
                reasons=["animal detected, no person"],
            )
        if "vehicle" in ctx.smart_detect_types:
            return DecisionResult(
                decision=Decision.IGNORE,
                score=0.0,
                reasons=["vehicle on non-driveway camera = noise"],
            )
        return DecisionResult(decision=Decision.IGNORE, score=0.0, reasons=["no relevant object"])

    # 3. Person detected — score from signals
    score = 0.0

    # 3a. Face + Body Re-ID — known family check
    usable_faces = [f for f in ctx.faces if f.width_px >= min_face_width_px]
    all_faces_known = bool(usable_faces) and all(f.is_known for f in usable_faces)
    any_unknown_face = bool(usable_faces) and any(not f.is_known for f in usable_faces)
    all_bodies_matched = (
        ctx.body_person_count > 0 and len(ctx.body_matches) == ctx.body_person_count
    )

    trajectory_matched = bool(ctx.trajectory_matches)
    skeleton_matched = bool(ctx.skeleton_matches)

    # If any strong signal identifies all persons as family → KNOWN_FAMILY.
    recognized_by = (
        (all_faces_known and not any_unknown_face)
        or (all_bodies_matched and not any_unknown_face)
        or (trajectory_matched and not any_unknown_face)
        or (skeleton_matched and not any_unknown_face)
    )
    if recognized_by:
        subjects: list[str] = []
        if usable_faces:
            subjects.extend(f.matched_subject for f in usable_faces if f.matched_subject)
        subjects.extend(ctx.body_matches)
        subjects.extend(ctx.trajectory_matches)
        subjects.extend(ctx.skeleton_matches)
        unique_subjects = list(dict.fromkeys(subjects))
        if all_faces_known:
            signal = "face"
        elif skeleton_matched:
            signal = "skeleton"
        elif trajectory_matched:
            signal = "trajectory"
        else:
            signal = "body"
        return DecisionResult(
            decision=Decision.KNOWN_FAMILY,
            score=0.0,
            reasons=[f"recognized ({signal}): {', '.join(unique_subjects) or 'family'}"],
            matched_subjects=unique_subjects,
        )

    # 3b. Presence-count matching: if the number of detected persons matches
    # or is fewer than family phones on WiFi, it's very likely all family.
    n_family_home = len(ctx.family_members_home)
    n_persons = max(ctx.body_person_count, 1)

    if n_family_home > 0 and n_persons <= n_family_home and not any_unknown_face:
        return DecisionResult(
            decision=Decision.KNOWN_FAMILY,
            score=0.0,
            reasons=[
                f"presence-count: {n_persons} person(s) detected, "
                f"{n_family_home} family phone(s) home "
                f"({', '.join(ctx.family_members_home)})",
            ],
            matched_subjects=ctx.family_members_home,
        )

    # 3c. More persons than family phones → some are unknown
    if n_family_home > 0 and n_persons > n_family_home:
        extra = n_persons - n_family_home
        score += 0.6
        reasons.append(
            f"{extra} unknown person(s) beyond {n_family_home} family phone(s) home"
        )
    elif any_unknown_face:
        score += 0.7
        reasons.append(f"unknown face detected (width≥{min_face_width_px}px)")
    elif not ctx.family_at_home:
        score += 0.7
        reasons.append("person detected, no family phones on WiFi")
    elif ctx.body_person_count > 0 and not ctx.body_matches:
        score += 0.3
        reasons.append("person detected, family home but no body match")
    else:
        score += 0.3
        reasons.append("person present but unclear identification")

    # 3d. Time of day
    if ctx.now.hour in NIGHT_HOURS:
        score += 0.4
        reasons.append(f"night hour ({ctx.now.hour:02d}:00)")

    # 3e. No family at home — strongest contextual signal
    if not ctx.family_at_home:
        score += 0.3
        reasons.append("no family at home")

    # 3f. Camera weight
    if ctx.camera_cfg.alert_weight > 0:
        score += ctx.camera_cfg.alert_weight
        reasons.append(f"camera alert_weight=+{ctx.camera_cfg.alert_weight}")

    matched = [f.matched_subject for f in ctx.faces if f.matched_subject]
    if score >= alert_threshold:
        return DecisionResult(
            decision=Decision.ALERT, score=score, reasons=reasons, matched_subjects=matched
        )
    return DecisionResult(
        decision=Decision.SILENT, score=score, reasons=reasons, matched_subjects=matched
    )
