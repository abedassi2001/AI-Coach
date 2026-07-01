"""Continuous 0–100 squat dimension scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.features.feature_pipeline import FrameFeatures
from backend.features.rep_segmentation import Repetition
from backend.feedback.squat_metrics import heel_lift_detected, knee_valgus_score
from backend.feedback.scoring import linear_score, weighted_average
from backend.pose.keypoint_schema import PoseFrame


@dataclass
class DimensionScores:
    depth_score: float
    knee_tracking_score: float
    torso_control_score: float
    symmetry_score: float
    stability_score: float
    heel_control_score: float
    overall_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "depth_score": self.depth_score,
            "knee_tracking_score": self.knee_tracking_score,
            "torso_control_score": self.torso_control_score,
            "symmetry_score": self.symmetry_score,
            "stability_score": self.stability_score,
            "heel_control_score": self.heel_control_score,
            "overall_score": self.overall_score,
        }


@dataclass
class RepConfidence:
    pose_confidence: float
    camera_angle_confidence: str
    heel_detection_confidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose_confidence": round(self.pose_confidence, 3),
            "camera_angle_confidence": self.camera_angle_confidence,
            "heel_detection_confidence": self.heel_detection_confidence,
        }


def _dim_cfg(scoring_cfg: dict[str, Any], name: str) -> dict[str, float]:
    return scoring_cfg.get("dimensions", {}).get(name, {})


def score_depth(bottom_knee_angle: float, cfg: dict[str, Any]) -> float:
    """Smaller knee angle at bottom = deeper squat = higher score."""
    d = _dim_cfg(cfg, "depth")
    return round(
        linear_score(
            bottom_knee_angle,
            ideal_value=float(d.get("ideal", 80)),
            warning_value=float(d.get("warning", 95)),
            fail_value=float(d.get("fail", 115)),
            higher_is_worse=True,
        ),
        1,
    )


def score_torso_control(torso_lean: float, cfg: dict[str, Any]) -> float:
    """Smaller forward lean at bottom = higher score."""
    if torso_lean != torso_lean:
        return 50.0
    d = _dim_cfg(cfg, "torso_control")
    return round(
        linear_score(
            torso_lean,
            ideal_value=float(d.get("ideal", 28)),
            warning_value=float(d.get("warning", 42)),
            fail_value=float(d.get("fail", 55)),
            higher_is_worse=True,
        ),
        1,
    )


def score_symmetry(knee_asymmetry: float, cfg: dict[str, Any]) -> float:
    """Smaller left/right knee asymmetry = higher score."""
    if knee_asymmetry != knee_asymmetry:
        return 50.0
    d = _dim_cfg(cfg, "symmetry")
    return round(
        linear_score(
            knee_asymmetry,
            ideal_value=float(d.get("ideal", 4)),
            warning_value=float(d.get("warning", 12)),
            fail_value=float(d.get("fail", 20)),
            higher_is_worse=True,
        ),
        1,
    )


def score_stability(knee_angle_std: float, cfg: dict[str, Any]) -> float:
    """Lower knee-angle variability over the rep = higher stability score."""
    if knee_angle_std != knee_angle_std:
        return 50.0
    d = _dim_cfg(cfg, "stability")
    return round(
        linear_score(
            knee_angle_std,
            ideal_value=float(d.get("ideal", 6)),
            warning_value=float(d.get("warning", 16)),
            fail_value=float(d.get("fail", 26)),
            higher_is_worse=True,
        ),
        1,
    )


def score_knee_tracking(valgus: float, cfg: dict[str, Any]) -> float:
    """Lower valgus proxy = better knee tracking."""
    if valgus != valgus:
        return 50.0
    d = _dim_cfg(cfg, "knee_tracking")
    return round(
        linear_score(
            valgus,
            ideal_value=float(d.get("ideal", 0.06)),
            warning_value=float(d.get("warning", 0.13)),
            fail_value=float(d.get("fail", 0.22)),
            higher_is_worse=True,
        ),
        1,
    )


def score_heel_control(
    lift_signal: float | None,
    heels_visible: bool,
    cfg: dict[str, Any],
) -> float:
    """Lower heel lift signal = heels stay down. Neutral if heels not visible."""
    d = _dim_cfg(cfg, "heel_control")
    if not heels_visible or lift_signal is None or lift_signal != lift_signal:
        return float(d.get("missing_score", 75))
    return round(
        linear_score(
            lift_signal,
            ideal_value=float(d.get("ideal", 0.005)),
            warning_value=float(d.get("warning", 0.02)),
            fail_value=float(d.get("fail", 0.045)),
            higher_is_worse=True,
        ),
        1,
    )


def _pose_confidence(bottom_pose: PoseFrame | None) -> float:
    if bottom_pose is None:
        return 0.0
    names = (
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_shoulder", "right_shoulder",
    )
    vis = [bottom_pose.landmarks[n].visibility for n in names if n in bottom_pose.landmarks]
    if not vis:
        return 0.0
    return sum(vis) / len(vis)


def _heel_visibility(bottom_pose: PoseFrame | None) -> tuple[bool, str]:
    if bottom_pose is None:
        return False, "low"
    heels = [
        bottom_pose.landmarks[n].visibility
        for n in ("left_heel", "right_heel")
        if n in bottom_pose.landmarks
    ]
    if not heels:
        return False, "low"
    avg = sum(heels) / len(heels)
    if avg >= 0.6:
        return True, "high"
    if avg >= 0.35:
        return True, "medium"
    return False, "low"


def _camera_angle_confidence(
    bottom: FrameFeatures | None,
    bottom_pose: PoseFrame | None,
) -> str:
    """Side view is best for depth/lean; front view is better for valgus."""
    if bottom is None:
        return "low"
    frontality = bottom.derived.get("camera_frontality", float("nan"))
    if frontality == frontality:
        if frontality < 0.85:
            return "high"  # side-dominant
        if frontality < 1.15:
            return "medium"
        return "medium"  # front-dominant — valgus ok, depth less reliable
    if bottom_pose is not None:
        lm = bottom_pose.landmarks
        if "left_hip" in lm and "right_hip" in lm:
            hip_span = abs(lm["left_hip"].x - lm["right_hip"].x)
            if hip_span < 0.08:
                return "high"
            if hip_span < 0.14:
                return "medium"
    return "low"


def compute_rep_scores(
    rep: Repetition,
    rep_frames: list[FrameFeatures],
    bottom: FrameFeatures | None,
    bottom_pose: PoseFrame | None,
    scoring_cfg: dict[str, Any],
    *,
    knee_angle_std: float | None = None,
    heel_lift_signal: float | None = None,
    valgus: float | None = None,
) -> tuple[DimensionScores, RepConfidence, dict[str, float]]:
    """Compute all dimension scores and raw metrics for one rep."""
    metrics: dict[str, float] = {"bottom_knee_angle": rep.bottom_knee_angle}

    torso_lean = float("nan")
    asymmetry = float("nan")
    if bottom:
        torso_lean = bottom.angles.get("torso_lean", float("nan"))
        asymmetry = bottom.derived.get("knee_asymmetry_deg", float("nan"))
        metrics["bottom_torso_lean"] = torso_lean
        metrics["bottom_knee_asymmetry"] = asymmetry

    if knee_angle_std is not None and knee_angle_std == knee_angle_std:
        metrics["knee_angle_std"] = knee_angle_std
    if heel_lift_signal is not None and heel_lift_signal == heel_lift_signal:
        metrics["heel_lift_signal"] = heel_lift_signal
    if valgus is not None and valgus == valgus:
        metrics["knee_valgus_score"] = valgus

    heels_visible, heel_conf = _heel_visibility(bottom_pose)

    scores_map = {
        "depth_score": score_depth(rep.bottom_knee_angle, scoring_cfg),
        "torso_control_score": score_torso_control(torso_lean, scoring_cfg),
        "symmetry_score": score_symmetry(asymmetry, scoring_cfg),
        "stability_score": score_stability(
            knee_angle_std if knee_angle_std is not None else float("nan"),
            scoring_cfg,
        ),
        "knee_tracking_score": score_knee_tracking(
            valgus if valgus is not None else float("nan"),
            scoring_cfg,
        ),
        "heel_control_score": score_heel_control(heel_lift_signal, heels_visible, scoring_cfg),
    }

    weights = scoring_cfg.get("weights", {})
    overall = weighted_average(scores_map, weights)

    confidence = RepConfidence(
        pose_confidence=_pose_confidence(bottom_pose),
        camera_angle_confidence=_camera_angle_confidence(bottom, bottom_pose),
        heel_detection_confidence=heel_conf,
    )

    dims = DimensionScores(
        depth_score=scores_map["depth_score"],
        knee_tracking_score=scores_map["knee_tracking_score"],
        torso_control_score=scores_map["torso_control_score"],
        symmetry_score=scores_map["symmetry_score"],
        stability_score=scores_map["stability_score"],
        heel_control_score=scores_map["heel_control_score"],
        overall_score=overall,
    )
    return dims, confidence, metrics


def measure_heel_and_valgus(
    bottom_pose: PoseFrame | None,
    bottom: FrameFeatures | None,
) -> tuple[float | None, float | None, bool]:
    """Return (heel_lift_signal, valgus, heels_visible)."""
    if bottom_pose is None or bottom is None:
        return None, None, False

    heels_visible, _ = _heel_visibility(bottom_pose)
    _, lift_val = heel_lift_detected(bottom_pose)
    torso = bottom.torso_length if bottom.torso_length == bottom.torso_length else 0.1
    valgus = knee_valgus_score(bottom_pose, torso)
    return lift_val, valgus, heels_visible
