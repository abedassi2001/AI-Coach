"""Rule-based squat form analysis from features and rep segments."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.features.feature_pipeline import FrameFeatures
from src.features.rep_segmentation import (
    Repetition,
    load_frame_features_from_csv,
    load_frame_features_from_json,
)
from src.feedback.templates import format_mistake
from src.pose.keypoint_schema import Keypoint, PoseFrame
from src.utils.config import load_config, resolve_path


@dataclass
class MistakeFinding:
    mistake_id: str
    severity: str
    message: str
    value: float
    threshold: float
    frame_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepFormAnalysis:
    rep_id: int
    form_score: float
    quality: str
    mistakes: list[MistakeFinding] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rep_id": self.rep_id,
            "form_score": self.form_score,
            "quality": self.quality,
            "mistakes": [m.to_dict() for m in self.mistakes],
            "metrics": self.metrics,
        }


@dataclass
class FormAnalysisResult:
    source_id: str
    exercise: str
    rep_analyses: list[RepFormAnalysis]
    overall_score: float
    overall_quality: str
    output_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "exercise": self.exercise,
            "overall_score": self.overall_score,
            "overall_quality": self.overall_quality,
            "rep_count": len(self.rep_analyses),
            "repetitions": [r.to_dict() for r in self.rep_analyses],
        }

    def save_json(self, path: Path | None = None) -> Path:
        out = path or (self.output_dir / "form_analysis.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out


def load_repetitions(reps_path: str | Path) -> tuple[str, str, list[Repetition]]:
    with Path(reps_path).open(encoding="utf-8") as f:
        data = json.load(f)
    reps = []
    for r in data["repetitions"]:
        reps.append(
            Repetition(
                rep_id=r["rep_id"],
                start_frame=r["start_frame"],
                end_frame=r["end_frame"],
                bottom_frame=r["bottom_frame"],
                start_time_sec=r["start_time_sec"],
                end_time_sec=r["end_time_sec"],
                duration_sec=r["duration_sec"],
                bottom_knee_angle=r["bottom_knee_angle"],
            )
        )
    return data["source_id"], data["exercise"], reps


def load_pose_frames(keypoints_path: str | Path) -> dict[int, PoseFrame]:
    with Path(keypoints_path).open(encoding="utf-8") as f:
        data = json.load(f)
    by_index: dict[int, PoseFrame] = {}
    for fr in data["frames"]:
        landmarks = {n: Keypoint(**v) for n, v in fr["landmarks"].items()}
        by_index[fr["frame_index"]] = PoseFrame(
            frame_index=fr["frame_index"],
            timestamp_sec=fr["timestamp_sec"],
            landmarks=landmarks,
            width=fr.get("width", 0),
            height=fr.get("height", 0),
        )
    return by_index


def _frames_for_rep(
    all_frames: list[FrameFeatures], rep: Repetition
) -> list[FrameFeatures]:
    return [f for f in all_frames if rep.start_frame <= f.frame_index <= rep.end_frame]


def _frame_by_index(
    all_frames: list[FrameFeatures], frame_index: int
) -> FrameFeatures | None:
    for f in all_frames:
        if f.frame_index == frame_index:
            return f
    return None


def _heel_lift_detected(pose: PoseFrame, threshold: float = 0.02) -> tuple[bool, float]:
    """Heel higher than ankle in image (smaller y) suggests lift."""
    lifts: list[float] = []
    for side in ("left", "right"):
        ankle = pose.landmarks.get(f"{side}_ankle")
        heel = pose.landmarks.get(f"{side}_heel")
        if ankle and heel and ankle.visibility > 0.5 and heel.visibility > 0.5:
            # y grows downward; heel above ankle => ankle.y - heel.y > 0
            lifts.append(ankle.y - heel.y)
    if not lifts:
        return False, 0.0
    max_lift = max(lifts)
    return max_lift > threshold, max_lift


def _knee_valgus_score(pose: PoseFrame, torso_length: float) -> float:
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


class SquatFormAnalyzer:
    """Apply configurable rules to each segmented repetition."""

    def __init__(self, rules: dict[str, Any] | None = None) -> None:
        cfg = load_config()
        self.rules = rules or cfg.get("form_rules", {})

    def analyze_rep(
        self,
        rep: Repetition,
        rep_frames: list[FrameFeatures],
        bottom_pose: PoseFrame | None = None,
    ) -> RepFormAnalysis:
        mistakes: list[MistakeFinding] = []
        bottom = _frame_by_index(rep_frames, rep.bottom_frame) or (
            rep_frames[0] if rep_frames else None
        )

        metrics: dict[str, float] = {
            "bottom_knee_angle": rep.bottom_knee_angle,
        }

        if bottom:
            metrics["bottom_torso_lean"] = bottom.angles.get("torso_lean", float("nan"))
            metrics["bottom_knee_asymmetry"] = bottom.derived.get(
                "knee_asymmetry_deg", float("nan")
            )

        # --- insufficient depth ---
        max_depth_angle = self.rules.get("min_depth_knee_angle", 90)
        if rep.bottom_knee_angle > max_depth_angle:
            mistakes.append(
                MistakeFinding(
                    mistake_id="insufficient_depth",
                    severity="high",
                    message=format_mistake(
                        "insufficient_depth", rep.bottom_knee_angle, max_depth_angle
                    ),
                    value=rep.bottom_knee_angle,
                    threshold=max_depth_angle,
                    frame_index=rep.bottom_frame,
                )
            )

        # --- forward lean at bottom ---
        if bottom:
            lean = bottom.angles.get("torso_lean", float("nan"))
            max_lean = self.rules.get("max_torso_lean_deg", 45)
            if lean == lean and lean > max_lean:
                mistakes.append(
                    MistakeFinding(
                        mistake_id="excessive_forward_lean",
                        severity="medium",
                        message=format_mistake("excessive_forward_lean", lean, max_lean),
                        value=lean,
                        threshold=max_lean,
                        frame_index=rep.bottom_frame,
                    )
                )

        # --- asymmetry at bottom ---
        if bottom:
            asym = bottom.derived.get("knee_asymmetry_deg", float("nan"))
            asym_threshold = self.rules.get("max_knee_asymmetry_deg", 15)
            if asym == asym and asym > asym_threshold:
                mistakes.append(
                    MistakeFinding(
                        mistake_id="asymmetry",
                        severity="medium",
                        message=format_mistake("asymmetry", asym, asym_threshold),
                        value=asym,
                        threshold=asym_threshold,
                        frame_index=rep.bottom_frame,
                    )
                )

        # --- unstable path (knee angle std dev over rep) ---
        knee_series = [
            f.derived.get("knee_angle_min", float("nan"))
            for f in rep_frames
            if f.derived.get("knee_angle_min", float("nan")) == f.derived.get("knee_angle_min", float("nan"))
        ]
        if len(knee_series) >= 3:
            instability = statistics.pstdev(knee_series)
            metrics["knee_angle_std"] = instability
            unstable_threshold = self.rules.get("max_knee_angle_std", 25)
            if instability > unstable_threshold:
                mistakes.append(
                    MistakeFinding(
                        mistake_id="unstable_path",
                        severity="low",
                        message=format_mistake("unstable_path", instability, unstable_threshold),
                        value=instability,
                        threshold=unstable_threshold,
                    )
                )

        # --- keypoint-based checks at bottom ---
        if bottom_pose and bottom:
            torso = bottom.torso_length if bottom.torso_length == bottom.torso_length else 0.1

            lifted, lift_val = _heel_lift_detected(bottom_pose)
            metrics["heel_lift_signal"] = lift_val
            if lifted:
                mistakes.append(
                    MistakeFinding(
                        mistake_id="heel_lift",
                        severity="medium",
                        message=format_mistake("heel_lift", lift_val, 0.02),
                        value=lift_val,
                        threshold=0.02,
                        frame_index=rep.bottom_frame,
                    )
                )

            valgus = _knee_valgus_score(bottom_pose, torso)
            metrics["knee_valgus_score"] = valgus
            valgus_threshold = self.rules.get("knee_valgus_threshold", 0.15)
            if valgus > valgus_threshold:
                mistakes.append(
                    MistakeFinding(
                        mistake_id="knee_valgus",
                        severity="medium",
                        message=format_mistake("knee_valgus", valgus, valgus_threshold),
                        value=valgus,
                        threshold=valgus_threshold,
                        frame_index=rep.bottom_frame,
                    )
                )

        score = self._score_from_mistakes(mistakes)
        quality = "good" if score >= 80 and not mistakes else "needs_work"

        return RepFormAnalysis(
            rep_id=rep.rep_id,
            form_score=score,
            quality=quality,
            mistakes=mistakes,
            metrics=metrics,
        )

    @staticmethod
    def _score_from_mistakes(mistakes: list[MistakeFinding]) -> float:
        penalties = {"high": 25, "medium": 15, "low": 8}
        score = 100.0
        for m in mistakes:
            score -= penalties.get(m.severity, 10)
        return max(0.0, round(score, 1))

    def analyze(
        self,
        features_path: str | Path,
        reps_path: str | Path,
        keypoints_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> FormAnalysisResult:
        fpath = Path(features_path)
        if fpath.suffix.lower() == ".csv":
            source_id, exercise, frames = load_frame_features_from_csv(fpath)
        else:
            source_id, exercise, frames = load_frame_features_from_json(fpath)

        _, _, repetitions = load_repetitions(reps_path)
        pose_by_frame = load_pose_frames(keypoints_path) if keypoints_path else {}

        rep_analyses: list[RepFormAnalysis] = []
        for rep in repetitions:
            rep_frames = _frames_for_rep(frames, rep)
            bottom_pose = pose_by_frame.get(rep.bottom_frame)
            rep_analyses.append(self.analyze_rep(rep, rep_frames, bottom_pose))

        overall = (
            round(sum(r.form_score for r in rep_analyses) / len(rep_analyses), 1)
            if rep_analyses
            else 0.0
        )
        overall_quality = "good" if overall >= 80 else "needs_work"

        out_dir = (
            Path(output_dir)
            if output_dir
            else resolve_path("data/processed/analysis") / source_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        result = FormAnalysisResult(
            source_id=source_id,
            exercise=exercise,
            rep_analyses=rep_analyses,
            overall_score=overall,
            overall_quality=overall_quality,
            output_dir=out_dir,
        )
        result.save_json()
        return result
