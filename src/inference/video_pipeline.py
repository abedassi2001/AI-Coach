"""End-to-end video analysis pipeline for the demo app and API."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.features.feature_pipeline import FeatureExtractionPipeline
from src.features.rep_segmentation import RepSegmentationPipeline
from src.feedback.coaching_pipeline import generate_coaching
from src.feedback.form_analyzer import SquatFormAnalyzer
from src.pose.pose_pipeline import PoseExtractionPipeline
from src.utils.config import get_project_root, resolve_path
from src.visualization.evaluation_overlay import (
    build_evaluation_report,
    load_evaluation_artifacts,
    write_evaluation_video,
    write_knee_angle_chart,
)

ProgressCallback = Callable[[str, float], None]


def _notify(cb: ProgressCallback | None, message: str, progress: float) -> None:
    if cb is not None:
        cb(message, progress)


def sanitize_source_id(name: str) -> str:
    """Turn an upload filename into a safe source_id."""
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9_-]+", "_", stem)
    stem = stem.strip("_")
    return stem or "upload"


@dataclass
class PipelineResult:
    source_id: str
    exercise: str
    video_path: Path
    evaluation_video: Path | None = None
    chart_path: Path | None = None
    coaching_text_path: Path | None = None
    report: dict[str, Any] = field(default_factory=dict)
    coaching_provider: str | None = None


def default_model_path() -> Path:
    return get_project_root() / "models/checkpoints/baseline/form_classifier.joblib"


def list_demo_videos() -> list[dict[str, str]]:
    """Return sample videos already in data/raw/videos."""
    root = get_project_root()
    videos_dir = root / "data/raw/videos"
    if not videos_dir.exists():
        return []
    items: list[dict[str, str]] = []
    for path in sorted(videos_dir.glob("*")):
        if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            items.append({"source_id": path.stem, "path": str(path.resolve())})
    return items


def stage_uploaded_video(upload_path: Path, source_id: str | None = None) -> tuple[Path, str]:
    """Copy an uploaded file into data/raw/videos and return (path, source_id)."""
    sid = source_id or sanitize_source_id(upload_path.name)
    dest_dir = resolve_path("data/raw/videos")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sid}{upload_path.suffix.lower() or '.mp4'}"
    shutil.copy2(upload_path, dest)
    return dest, sid


def run_full_pipeline(
    video_path: Path,
    *,
    exercise: str = "squat",
    source_id: str | None = None,
    model_path: Path | None = None,
    coaching_provider: str = "template",
    generate_evaluation_video: bool = True,
    generate_coaching_report: bool = True,
    on_progress: ProgressCallback | None = None,
) -> PipelineResult:
    """
    Run pose → features → reps → rules → ML overlay → optional coaching.

    Progress callback receives (step_message, fraction_0_to_1).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    sid = source_id or video_path.stem
    root = get_project_root()
    model = model_path if model_path is not None else default_model_path()
    if not model.exists():
        model = None

    _notify(on_progress, "Extracting pose keypoints…", 0.05)
    pose_pipeline = PoseExtractionPipeline()
    pose_pipeline.process_video(video_path, exercise=exercise)

    keypoints = root / "data/processed/pose" / sid / "keypoints.json"
    _notify(on_progress, "Computing joint angles and features…", 0.30)
    feat_result = FeatureExtractionPipeline().extract_from_keypoints_file(keypoints)

    _notify(on_progress, "Segmenting repetitions…", 0.45)
    RepSegmentationPipeline().segment_from_features_file(feat_result.csv_path)

    reps_path = root / "data/processed/reps" / sid / "reps.json"
    _notify(on_progress, "Analyzing form with rules…", 0.55)
    SquatFormAnalyzer().analyze(
        features_path=feat_result.csv_path,
        reps_path=reps_path,
        keypoints_path=keypoints,
    )

    _notify(on_progress, "Loading predictions and building report…", 0.70)
    artifacts = load_evaluation_artifacts(
        sid,
        video_path=video_path,
        model_path=model,
        root=root,
    )
    report = build_evaluation_report(artifacts)

    eval_dir = root / "data/processed/evaluation" / sid
    eval_dir.mkdir(parents=True, exist_ok=True)

    evaluation_video: Path | None = None
    chart_path: Path | None = None
    if generate_evaluation_video:
        _notify(on_progress, "Rendering annotated evaluation video…", 0.80)
        evaluation_video = eval_dir / f"{sid}_evaluation.mp4"
        write_evaluation_video(artifacts, evaluation_video)
        chart_path = write_knee_angle_chart(artifacts, eval_dir / "knee_angle_chart.png")

    coaching_text_path: Path | None = None
    provider_used: str | None = None
    if generate_coaching_report:
        _notify(on_progress, "Generating coaching feedback…", 0.92)
        coaching = generate_coaching(
            sid,
            provider=coaching_provider,
            model_path=model,
            allow_fallback=coaching_provider in ("auto", "ollama", "openai"),
        )
        coaching_text_path = coaching.text_path
        provider_used = coaching.provider

    _notify(on_progress, "Done", 1.0)

    return PipelineResult(
        source_id=sid,
        exercise=exercise,
        video_path=video_path.resolve(),
        evaluation_video=evaluation_video.resolve() if evaluation_video else None,
        chart_path=chart_path.resolve() if chart_path else None,
        coaching_text_path=coaching_text_path.resolve() if coaching_text_path else None,
        report=report,
        coaching_provider=provider_used,
    )


def load_existing_result(source_id: str) -> PipelineResult | None:
    """Load outputs if this video was already processed."""
    root = get_project_root()
    keypoints = root / "data/processed/pose" / source_id / "keypoints.json"
    reps = root / "data/processed/reps" / source_id / "reps.json"
    if not keypoints.exists() or not reps.exists():
        return None

    video = root / "data/raw/videos" / f"{source_id}.mp4"
    if not video.exists():
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            candidate = root / "data/raw/videos" / f"{source_id}{ext}"
            if candidate.exists():
                video = candidate
                break

    model = default_model_path()
    model_path = model if model.exists() else None
    artifacts = load_evaluation_artifacts(
        source_id,
        video_path=video if video.exists() else None,
        model_path=model_path,
        root=root,
    )
    report = build_evaluation_report(artifacts)

    eval_dir = root / "data/processed/evaluation" / source_id
    eval_video = eval_dir / f"{source_id}_evaluation.mp4"
    chart = eval_dir / "knee_angle_chart.png"
    coaching_txt = root / "data/processed/coaching" / source_id / "coaching_report.txt"

    return PipelineResult(
        source_id=source_id,
        exercise=artifacts.exercise,
        video_path=artifacts.video_path,
        evaluation_video=eval_video if eval_video.exists() else None,
        chart_path=chart if chart.exists() else None,
        coaching_text_path=coaching_txt if coaching_txt.exists() else None,
        report=report,
    )
