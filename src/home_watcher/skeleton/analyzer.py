"""Skeleton analysis using YOLOv8-pose for biometric proportions and gait.

COCO 17 keypoints:
  0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
  5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
  9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
  13:left_knee 14:right_knee 15:left_ankle 16:right_ankle

Biometric proportions (stable across clothing/time):
  - shoulder_width / height
  - torso_length / height  (mid-shoulder to mid-hip)
  - leg_length / height    (mid-hip to mid-ankle)
  - arm_length / height    (shoulder to wrist)

Gait features (from burst frames):
  - stride_length (ankle displacement between frames)
  - arm_swing (wrist displacement between frames)
  - cadence (steps per second estimated from knee oscillation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

MIN_KEYPOINT_CONF = 0.3


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


@dataclass(frozen=True)
class BodyProportions:
    """Biometric ratios normalized to total height. Stable across clothing."""
    shoulder_ratio: float
    torso_ratio: float
    leg_ratio: float
    arm_ratio: float
    height_px: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.shoulder_ratio,
            self.torso_ratio,
            self.leg_ratio,
            self.arm_ratio,
        ], dtype=np.float32)


@dataclass(frozen=True)
class GaitFeatures:
    """Per-frame gait measurements from skeleton keypoints across burst frames."""
    stride_length: float
    arm_swing: float
    knee_angle_change: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.stride_length,
            self.arm_swing,
            self.knee_angle_change,
        ], dtype=np.float32)


@dataclass
class SkeletonProfile:
    """Combined biometric profile: proportions + gait signature."""
    proportions: BodyProportions | None
    gait: list[GaitFeatures]
    keypoints_sequence: list[np.ndarray]

    def to_vector(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.proportions:
            parts.append(self.proportions.to_vector())
        if self.gait:
            avg_gait = np.mean([g.to_vector() for g in self.gait], axis=0)
            parts.append(avg_gait)
        if not parts:
            return np.zeros(7, dtype=np.float32)
        vec = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class SkeletonAnalyzer:
    def __init__(self) -> None:
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO
        self._model = YOLO("yolov8n-pose.pt")

    def detect_keypoints(self, image_bytes: bytes) -> list[np.ndarray]:
        """Detect person skeletons. Returns list of (17, 3) arrays [x, y, conf]."""
        self._ensure_loaded()
        assert self._model is not None
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        results = self._model(np.array(img), verbose=False)
        persons: list[np.ndarray] = []
        for r in results:
            if r.keypoints is None:
                continue
            for kps in r.keypoints.data:
                arr = kps.cpu().numpy()
                if arr.shape == (17, 3):
                    persons.append(arr)
        return persons

    def extract_proportions(self, keypoints: np.ndarray) -> BodyProportions | None:
        """Extract biometric proportions from a single skeleton."""
        if keypoints.shape != (17, 3):
            return None

        def kp(idx: int) -> np.ndarray | None:
            if keypoints[idx, 2] < MIN_KEYPOINT_CONF:
                return None
            return keypoints[idx, :2]

        l_shoulder = kp(KP_L_SHOULDER)
        r_shoulder = kp(KP_R_SHOULDER)
        l_hip = kp(KP_L_HIP)
        r_hip = kp(KP_R_HIP)
        l_ankle = kp(KP_L_ANKLE)
        r_ankle = kp(KP_R_ANKLE)
        l_wrist = kp(KP_L_WRIST)
        r_wrist = kp(KP_R_WRIST)
        nose = kp(KP_NOSE)

        if any(p is None for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
            return None
        assert l_shoulder is not None and r_shoulder is not None
        assert l_hip is not None and r_hip is not None

        mid_shoulder = _midpoint(l_shoulder, r_shoulder)
        mid_hip = _midpoint(l_hip, r_hip)

        top = nose if nose is not None else mid_shoulder
        ankles = [a for a in [l_ankle, r_ankle] if a is not None]
        if not ankles:
            return None
        mid_ankle = np.mean(ankles, axis=0)

        height = _dist(top, mid_ankle)
        if height < 50:
            return None

        shoulder_width = _dist(l_shoulder, r_shoulder)
        torso_length = _dist(mid_shoulder, mid_hip)

        leg_length = _dist(mid_hip, mid_ankle)

        wrists = [w for w in [l_wrist, r_wrist] if w is not None]
        shoulders = [l_shoulder, r_shoulder]
        if wrists and shoulders:
            arm_lengths = []
            if l_wrist is not None:
                arm_lengths.append(_dist(l_shoulder, l_wrist))
            if r_wrist is not None:
                arm_lengths.append(_dist(r_shoulder, r_wrist))
            arm_length = float(np.mean(arm_lengths))
        else:
            arm_length = 0.0

        return BodyProportions(
            shoulder_ratio=shoulder_width / height,
            torso_ratio=torso_length / height,
            leg_ratio=leg_length / height,
            arm_ratio=arm_length / height,
            height_px=height,
        )

    def extract_gait(
        self, kp_sequence: list[np.ndarray],
    ) -> list[GaitFeatures]:
        """Extract gait features from a sequence of skeletons (burst frames)."""
        if len(kp_sequence) < 2:
            return []

        features: list[GaitFeatures] = []
        for i in range(1, len(kp_sequence)):
            prev, curr = kp_sequence[i - 1], kp_sequence[i]

            stride = self._ankle_displacement(prev, curr)
            arm_swing = self._wrist_displacement(prev, curr)
            knee_change = self._knee_angle_change(prev, curr)

            features.append(GaitFeatures(
                stride_length=stride,
                arm_swing=arm_swing,
                knee_angle_change=knee_change,
            ))
        return features

    def build_profile(
        self, snapshots: list[bytes],
    ) -> SkeletonProfile | None:
        """Build a complete skeleton profile from burst snapshots."""
        all_keypoints: list[np.ndarray] = []
        for snap in snapshots:
            persons = self.detect_keypoints(snap)
            if persons:
                all_keypoints.append(persons[0])

        if not all_keypoints:
            return None

        proportions = self.extract_proportions(all_keypoints[0])
        gait = self.extract_gait(all_keypoints)

        return SkeletonProfile(
            proportions=proportions,
            gait=gait,
            keypoints_sequence=all_keypoints,
        )

    def _ankle_displacement(self, prev: np.ndarray, curr: np.ndarray) -> float:
        displacements: list[float] = []
        for idx in [KP_L_ANKLE, KP_R_ANKLE]:
            if prev[idx, 2] >= MIN_KEYPOINT_CONF and curr[idx, 2] >= MIN_KEYPOINT_CONF:
                displacements.append(_dist(prev[idx, :2], curr[idx, :2]))
        return float(np.mean(displacements)) if displacements else 0.0

    def _wrist_displacement(self, prev: np.ndarray, curr: np.ndarray) -> float:
        displacements: list[float] = []
        for idx in [KP_L_WRIST, KP_R_WRIST]:
            if prev[idx, 2] >= MIN_KEYPOINT_CONF and curr[idx, 2] >= MIN_KEYPOINT_CONF:
                displacements.append(_dist(prev[idx, :2], curr[idx, :2]))
        return float(np.mean(displacements)) if displacements else 0.0

    def _knee_angle_change(self, prev: np.ndarray, curr: np.ndarray) -> float:
        angles: list[float] = []
        for hip_idx, knee_idx, ankle_idx in [
            (KP_L_HIP, KP_L_KNEE, KP_L_ANKLE),
            (KP_R_HIP, KP_R_KNEE, KP_R_ANKLE),
        ]:
            if all(prev[i, 2] >= MIN_KEYPOINT_CONF and curr[i, 2] >= MIN_KEYPOINT_CONF
                   for i in [hip_idx, knee_idx, ankle_idx]):
                prev_angle = self._joint_angle(prev[hip_idx, :2], prev[knee_idx, :2], prev[ankle_idx, :2])
                curr_angle = self._joint_angle(curr[hip_idx, :2], curr[knee_idx, :2], curr[ankle_idx, :2])
                angles.append(abs(curr_angle - prev_angle))
        return float(np.mean(angles)) if angles else 0.0

    @staticmethod
    def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at joint b formed by segments a-b and b-c, in degrees."""
        ba = a - b
        bc = c - b
        cos_angle = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8))
        return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
