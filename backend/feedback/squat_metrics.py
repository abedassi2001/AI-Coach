"""Low-level squat biomechanical measurements from pose and features."""

from __future__ import annotations

from backend.pose.keypoint_schema import PoseFrame


def heel_lift_detected(pose: PoseFrame, threshold: float = 0.02) -> tuple[bool, float]:
    """
    Heel higher than ankle in image (smaller y) suggests lift.

    Unreliable from pure side view — check heel_detection_confidence.
    """
    lifts: list[float] = []
    for side in ("left", "right"):
        ankle = pose.landmarks.get(f"{side}_ankle")
        heel = pose.landmarks.get(f"{side}_heel")
        if ankle and heel and ankle.visibility > 0.5 and heel.visibility > 0.5:
            lifts.append(ankle.y - heel.y)
    if not lifts:
        return False, 0.0
    max_lift = max(lifts)
    return max_lift > threshold, max_lift


def knee_valgus_score(pose: PoseFrame, torso_length: float) -> float:
    """
    Rough valgus proxy: knee x deviates toward body midline vs hip-ankle line.

    Works best with front-angled camera; side view is weak.
    """
    if torso_length <= 0:
        return 0.0
    scores: list[float] = []
    for side in ("left", "right"):
        hip = pose.landmarks.get(f"{side}_hip")
        knee = pose.landmarks.get(f"{side}_knee")
        ankle = pose.landmarks.get(f"{side}_ankle")
        if not all(p and p.visibility > 0.5 for p in (hip, knee, ankle)):
            continue
        mid_x = (hip.x + ankle.x) / 2.0
        deviation = abs(knee.x - mid_x) / torso_length
        scores.append(deviation)
    return max(scores) if scores else 0.0
