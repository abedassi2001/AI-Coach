"""Rule-based squat form analysis from features and rep segments."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backend.features.feature_pipeline import FrameFeatures
from backend.features.rep_segmentation import (
    Repetition,
    load_frame_features_from_csv,
    load_frame_features_from_json,
)
from backend.feedback.rep_coaching import (
    build_video_summary,
    flags_from_scores,
    generate_rep_coaching,
    mistakes_from_flags,
)
from backend.feedback.scoring import quality_label
from backend.feedback.squat_dimensions import (
    compute_rep_scores,
    measure_heel_and_valgus,
)
from backend.feedback.squat_metrics import heel_lift_detected, knee_valgus_score
from backend.feedback.templates import format_mistake
from backend.pose.keypoint_schema import Keypoint, PoseFrame
from backend.utils.config import get_project_root, load_config, resolve_path

ANALYZER_VERSION = "0.2.0-continuous-scoring"


@dataclass
class MistakeFinding:
    mistake_id: str
    severity: str
    message: str
    value: float
    threshold: float
    frame_index: int | None = None
    flag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepFormAnalysis:
    rep_id: int
    form_score: float  # backward compat alias for overall_score
    quality: str
    scores: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    coaching: dict[str, Any] = field(default_factory=dict)
    mistakes: list[MistakeFinding] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        return self.scores.get("overall_score", self.form_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rep_id": self.rep_id,
            "overall_score": self.overall_score,
            "form_score": self.form_score,  # deprecated alias
            "quality": self.quality,
            "scores": self.scores,
            "confidence": self.confidence,
            "flags": self.flags,
            "feedback": self.feedback,
            "coaching": self.coaching,
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
    analyzer_version: str = ANALYZER_VERSION
    video_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        reps = [r.to_dict() for r in self.rep_analyses]
        return {
            "analyzer_version": self.analyzer_version,
            "source_id": self.source_id,
            "exercise": self.exercise,
            "overall_score": self.overall_score,
            "overall_quality": self.overall_quality,
            "rep_count": len(self.rep_analyses),
            "video_summary": self.video_summary,
            "repetitions": reps,
            "reps": reps,  # alias for GPT / newer consumers
        }

    def save_json(self, path: Path | None = None) -> Path:
        out = path or (self.output_dir / "form_analysis.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return out


def load_scoring_config(exercise: str = "squat") -> dict[str, Any]:
    """Load continuous scoring thresholds and weights for an exercise."""
    path = get_project_root() / "configs/form_scoring" / f"{exercise}.yaml"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault("analyzer_version", ANALYZER_VERSION)
        return cfg
    return {"analyzer_version": ANALYZER_VERSION, "weights": {}, "dimensions": {}}


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


class SquatFormAnalyzer:
    """
    Apply configurable biomechanical rules to each segmented repetition.

    Produces continuous 0–100 scores per form dimension plus deterministic
    coaching feedback. The optional ML classifier (form_classifier.joblib) is
  separate and experimental — this analyzer is the primary scoring engine.
    """

    def __init__(
        self,
        rules: dict[str, Any] | None = None,
        scoring_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = load_config()
        self.rules = rules or cfg.get("form_rules", {})
        self.scoring_config = scoring_config or load_scoring_config("squat")

    def analyze_rep(
        self,
        rep: Repetition,
        rep_frames: list[FrameFeatures],
        bottom_pose: PoseFrame | None = None,
    ) -> RepFormAnalysis:
        bottom = _frame_by_index(rep_frames, rep.bottom_frame) or (
            rep_frames[0] if rep_frames else None
        )

        knee_series = [
            f.derived.get("knee_angle_min", float("nan"))
            for f in rep_frames
            if f.derived.get("knee_angle_min", float("nan"))
            == f.derived.get("knee_angle_min", float("nan"))
        ]
        knee_std = statistics.pstdev(knee_series) if len(knee_series) >= 3 else float("nan")

        heel_signal: float | None = None
        valgus: float | None = None
        if bottom_pose and bottom:
            heel_signal, valgus, _ = measure_heel_and_valgus(bottom_pose, bottom)
        elif bottom_pose is None and bottom is None:
            heel_signal, valgus = None, None

        dims, confidence, metrics = compute_rep_scores(
            rep,
            rep_frames,
            bottom,
            bottom_pose,
            self.scoring_config,
            knee_angle_std=knee_std,
            heel_lift_signal=heel_signal,
            valgus=valgus,
        )

        scores = dims.as_dict()
        flag_thresholds = self.scoring_config.get("flag_thresholds", {})
        flags = flags_from_scores(scores, flag_thresholds)

        legacy_map = self.scoring_config.get("legacy_mistake_map", {})
        mistake_dicts = mistakes_from_flags(
            flags,
            scores,
            metrics,
            legacy_map,
            frame_index=rep.bottom_frame,
        )
        mistakes = [MistakeFinding(**m) for m in mistake_dicts]

        coaching = generate_rep_coaching(scores, flags, confidence.as_dict())

        bands = self.scoring_config.get("quality_bands", {})
        quality = quality_label(
            dims.overall_score,
            good_min=float(bands.get("good_min", 80)),
            acceptable_min=float(bands.get("acceptable_min", 70)),
        )

        return RepFormAnalysis(
            rep_id=rep.rep_id,
            form_score=dims.overall_score,
            quality=quality,
            scores=scores,
            confidence=confidence.as_dict(),
            flags=flags,
            feedback=coaching["feedback"],
            coaching=coaching,
            mistakes=mistakes,
            metrics=metrics,
        )

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
            round(sum(r.overall_score for r in rep_analyses) / len(rep_analyses), 1)
            if rep_analyses
            else 0.0
        )
        bands = self.scoring_config.get("quality_bands", {})
        overall_quality = quality_label(
            overall,
            good_min=float(bands.get("good_min", 80)),
            acceptable_min=float(bands.get("acceptable_min", 70)),
        )

        analyzer_version = str(
            self.scoring_config.get("analyzer_version", ANALYZER_VERSION)
        )
        rep_dicts = [r.to_dict() for r in rep_analyses]
        video_summary = build_video_summary(rep_dicts, analyzer_version)

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
            analyzer_version=analyzer_version,
            video_summary=video_summary,
        )
        result.save_json()
        return result


# Re-export for backward compatibility with tests importing private helpers
_heel_lift_detected = heel_lift_detected
_knee_valgus_score = knee_valgus_score
