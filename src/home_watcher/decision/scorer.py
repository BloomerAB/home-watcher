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

    # 3a. Face recognition
    usable_faces = [f for f in ctx.faces if f.width_px >= min_face_width_px]
    if usable_faces and all(f.is_known for f in usable_faces):
        return DecisionResult(
            decision=Decision.KNOWN_FAMILY,
            score=0.0,
            reasons=[f"all faces known: {', '.join(f.matched_subject or '?' for f in usable_faces)}"],
            matched_subjects=[f.matched_subject for f in usable_faces if f.matched_subject],
        )

    if usable_faces and any(not f.is_known for f in usable_faces):
        score += 0.7
        reasons.append(f"unknown face detected (width≥{min_face_width_px}px)")
    elif not ctx.faces:
        score += 0.3
        reasons.append("no face detected — person present but face hidden/too far")
    else:
        score += 0.3
        reasons.append(f"only small faces (<{min_face_width_px}px)")

    # 3b. Time of day
    if ctx.now.hour in NIGHT_HOURS:
        score += 0.4
        reasons.append(f"night hour ({ctx.now.hour:02d}:00)")

    # 3c. Family presence
    if not ctx.family_at_home:
        score += 0.3
        reasons.append("no family at home")

    # 3d. Camera weight
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
