"""Overlay skeleton, rep phases, rule analysis, and ML predictions on video."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.feedback.form_analyzer import load_pose_frames, load_repetitions
from src.inference.rep_classifier import RepQualityPredictor
from src.pose.keypoint_schema import Keypoint, PoseFrame
from src.visualization.skeleton import draw_skeleton_on_frame

# BGR
_COLOR_GOOD = (0, 200, 80)
_COLOR_BAD = (0, 80, 255)
_COLOR_NEUTRAL = (220, 220, 220)
_COLOR_PANEL = (30, 30, 30)
_COLOR_PHASE = {
    "standing": (180, 180, 180),
    "descending": (0, 180, 255),
    "bottom": (0, 140, 255),
    "ascending": (255, 180, 0),
    "finished": (160, 255, 160),
}


@dataclass
class RepContext:
    rep_id: int
    start_frame: int
    end_frame: int
    bottom_frame: int
    bottom_knee_angle: float
    phases: list[dict[str, Any]] = field(default_factory=list)
    form_score: float | None = None
    rule_quality: str | None = None
    mistakes: list[dict[str, Any]] = field(default_factory=list)
    prediction: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationArtifacts:
    source_id: str
    exercise: str
    video_path: Path
    keypoints_path: Path
    reps_path: Path
    features_path: Path
    analysis_path: Path | None
    reps: list[RepContext]
    pose_by_frame: dict[int, PoseFrame]
    knee_angles: dict[int, float]
    frame_count: int
    fps: float


def _quality_color(label: str | None) -> tuple[int, int, int]:
    if not label:
        return _COLOR_NEUTRAL
    low = label.lower()
    if low in ("good", "finished"):
        return _COLOR_GOOD
    if low in ("bad", "poor"):
        return _COLOR_BAD
    return (0, 200, 255)  # needs_work / unknown


def load_evaluation_artifacts(
    source_id: str,
    video_path: Path | None = None,
    model_path: Path | None = None,
    root: Path | None = None,
) -> EvaluationArtifacts:
    """Load processed pipeline outputs and optional model predictions."""
    from src.utils.config import get_project_root

    root = root or get_project_root()
    pose_dir = root / "data/processed/pose" / source_id
    feat_dir = root / "data/processed/features" / source_id
    rep_dir = root / "data/processed/reps" / source_id
    analysis_dir = root / "data/processed/analysis" / source_id

    keypoints_path = pose_dir / "keypoints.json"
    features_path = feat_dir / "features.csv"
    reps_path = rep_dir / "reps.json"
    analysis_path = analysis_dir / "form_analysis.json" if analysis_dir.exists() else None

    if video_path is None:
        video_path = root / "data/raw/videos" / f"{source_id}.mp4"
        if not video_path.exists():
            for ext in (".mp4", ".mov", ".avi", ".mkv"):
                candidate = root / "data/raw/videos" / f"{source_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break

    if not keypoints_path.exists():
        raise FileNotFoundError(f"Missing keypoints: {keypoints_path}")
    if not reps_path.exists():
        raise FileNotFoundError(f"Missing reps: {reps_path}")

    with reps_path.open(encoding="utf-8") as f:
        reps_data = json.load(f)

    _, exercise, _ = load_repetitions(reps_path)
    pose_by_frame = load_pose_frames(keypoints_path)

    analysis_by_rep: dict[int, dict[str, Any]] = {}
    if analysis_path and analysis_path.exists():
        with analysis_path.open(encoding="utf-8") as f:
            analysis = json.load(f)
        for rep_a in analysis.get("repetitions", []):
            analysis_by_rep[int(rep_a["rep_id"])] = rep_a

    predictions_by_rep: dict[int, dict[str, Any]] = {}
    if model_path and Path(model_path).exists():
        predictor = RepQualityPredictor.load(str(model_path))
        for pred in predictor.predict_source(source_id):
            predictions_by_rep[int(pred["rep_id"])] = pred

    reps: list[RepContext] = []
    for r in reps_data.get("repetitions", []):
        rid = int(r["rep_id"])
        rep_a = analysis_by_rep.get(rid, {})
        pred = predictions_by_rep.get(rid, {})
        reps.append(
            RepContext(
                rep_id=rid,
                start_frame=int(r["start_frame"]),
                end_frame=int(r["end_frame"]),
                bottom_frame=int(r["bottom_frame"]),
                bottom_knee_angle=float(r.get("bottom_knee_angle", 0)),
                phases=r.get("phases", []),
                form_score=rep_a.get("form_score"),
                rule_quality=rep_a.get("quality"),
                mistakes=rep_a.get("mistakes", []),
                prediction=pred.get("prediction"),
                confidence=pred.get("confidence"),
                probabilities=pred.get("probabilities", {}),
            )
        )

    knee_angles: dict[int, float] = {}
    if features_path.exists():
        import csv

        with features_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["frame_index"])
                if "knee_angle_min" in row:
                    knee_angles[idx] = float(row["knee_angle_min"])

    fps = 30.0
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            cap.release()

    frame_count = int(reps_data.get("frame_count", len(pose_by_frame)))

    return EvaluationArtifacts(
        source_id=source_id,
        exercise=exercise,
        video_path=video_path,
        keypoints_path=keypoints_path,
        reps_path=reps_path,
        features_path=features_path,
        analysis_path=analysis_path,
        reps=reps,
        pose_by_frame=pose_by_frame,
        knee_angles=knee_angles,
        frame_count=frame_count,
        fps=fps,
    )


def rep_at_frame(artifacts: EvaluationArtifacts, frame_idx: int) -> RepContext | None:
    candidates = [
        rep for rep in artifacts.reps if rep.start_frame <= frame_idx <= rep.end_frame
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Overlapping segments: assign frame to rep whose bottom is nearest (deepest point).
    return min(candidates, key=lambda r: abs(r.bottom_frame - frame_idx))


def phase_at_frame(rep: RepContext, frame_idx: int) -> str | None:
    for seg in rep.phases:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return seg["phase"]
    return None


def _draw_panel(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int] | None]],
    origin: tuple[int, int] = (10, 10),
    width: int = 420,
) -> np.ndarray:
    out = frame.copy()
    x, y = origin
    line_h = 26
    panel_h = line_h * len(lines) + 16
    cv2.rectangle(out, (x, y), (x + width, y + panel_h), _COLOR_PANEL, -1)
    cv2.rectangle(out, (x, y), (x + width, y + panel_h), (80, 80, 80), 1)
    ty = y + 22
    for text, color in lines:
        cv2.putText(
            out,
            text,
            (x + 10, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color or _COLOR_NEUTRAL,
            1,
            cv2.LINE_AA,
        )
        ty += line_h
    return out


def draw_evaluation_overlay(
    frame: np.ndarray,
    frame_idx: int,
    artifacts: EvaluationArtifacts,
) -> np.ndarray:
    """Draw skeleton + HUD for one frame."""
    pose = artifacts.pose_by_frame.get(frame_idx)
    out = draw_skeleton_on_frame(frame, pose)

    rep = rep_at_frame(artifacts, frame_idx)
    lines: list[tuple[str, tuple[int, int, int] | None]] = [
        (f"{artifacts.exercise} | {artifacts.source_id}", _COLOR_NEUTRAL),
        (f"Frame {frame_idx} / {max(artifacts.frame_count - 1, 0)}", _COLOR_NEUTRAL),
    ]

    if rep is None:
        lines.append(("Between reps", _COLOR_NEUTRAL))
    else:
        phase = phase_at_frame(rep, frame_idx) or "?"
        lines.append((f"Rep {rep.rep_id} | phase: {phase}", _COLOR_PHASE.get(phase, _COLOR_NEUTRAL)))
        if rep.form_score is not None:
            lines.append(
                (f"Rules: {rep.rule_quality} (score {rep.form_score:.0f})", _quality_color(rep.rule_quality)),
            )
        if rep.prediction:
            conf = f"{rep.confidence:.0%}" if rep.confidence is not None else "?"
            lines.append(
                (f"Model: {rep.prediction} ({conf})", _quality_color(rep.prediction)),
            )
        knee = artifacts.knee_angles.get(frame_idx)
        if knee is not None:
            lines.append((f"Knee angle (min): {knee:.0f} deg", _COLOR_NEUTRAL))

        if frame_idx == rep.bottom_frame:
            lines.append((f"BOTTOM @ {rep.bottom_knee_angle:.0f} deg", (0, 220, 255)))

        active_mistakes = [
            m for m in rep.mistakes if m.get("frame_index") == frame_idx or frame_idx == rep.bottom_frame
        ]
        for m in active_mistakes[:2]:
            msg = m.get("mistake_id", "issue").replace("_", " ")
            lines.append((f"! {msg}", _COLOR_BAD))

    out = _draw_panel(out, lines)

    # Timeline bar at bottom
    h, w = out.shape[:2]
    bar_h = 18
    y0 = h - bar_h - 8
    cv2.rectangle(out, (8, y0), (w - 8, y0 + bar_h), (40, 40, 40), -1)
    total = max(artifacts.frame_count - 1, 1)

    for rep in artifacts.reps:
        x1 = int(8 + (rep.start_frame / total) * (w - 16))
        x2 = int(8 + (rep.end_frame / total) * (w - 16))
        color = _quality_color(rep.prediction or rep.rule_quality)
        cv2.rectangle(out, (x1, y0), (x2, y0 + bar_h), color, -1)
        bx = int(8 + (rep.bottom_frame / total) * (w - 16))
        cv2.line(out, (bx, y0 - 2), (bx, y0 + bar_h + 2), (255, 255, 255), 2)

    cx = int(8 + (frame_idx / total) * (w - 16))
    cv2.line(out, (cx, y0 - 4), (cx, y0 + bar_h + 4), (255, 255, 255), 2)

    return out


def write_evaluation_video(
    artifacts: EvaluationArtifacts,
    output_path: Path,
    max_frames: int | None = None,
) -> Path:
    """Render annotated MP4 from source video + pipeline artifacts."""
    if not artifacts.video_path.exists():
        raise FileNotFoundError(f"Video not found: {artifacts.video_path}")

    cap = cv2.VideoCapture(str(artifacts.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {artifacts.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or artifacts.fps or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break
        annotated = draw_evaluation_overlay(frame, idx, artifacts)
        writer.write(annotated)
        idx += 1

    cap.release()
    writer.release()
    return output_path


def write_rep_snapshots(
    artifacts: EvaluationArtifacts,
    output_dir: Path,
) -> list[Path]:
    """Save one annotated image per rep at the bottom (deepest) frame."""
    if not artifacts.video_path.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(artifacts.video_path))
    saved: list[Path] = []

    for rep in artifacts.reps:
        target = rep.bottom_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            continue
        annotated = draw_evaluation_overlay(frame, target, artifacts)
        out_path = output_dir / f"rep_{rep.rep_id:02d}_bottom_frame_{target:04d}.jpg"
        cv2.imwrite(str(out_path), annotated)
        saved.append(out_path)

    cap.release()
    return saved


def write_knee_angle_chart(artifacts: EvaluationArtifacts, output_path: Path) -> Path | None:
    """Plot knee angle over time with rep boundaries."""
    if not artifacts.knee_angles:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    frames = sorted(artifacts.knee_angles.keys())
    values = [artifacts.knee_angles[f] for f in frames]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, values, color="#2ecc71", linewidth=2, label="knee_angle_min")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Knee angle (deg)")
    ax.set_title(f"{artifacts.source_id} — knee angle & rep segments")
    ax.grid(True, alpha=0.3)

    for rep in artifacts.reps:
        ax.axvspan(rep.start_frame, rep.end_frame, alpha=0.15, color="steelblue")
        ax.axvline(rep.bottom_frame, color="orange", linestyle="--", alpha=0.8)
        label = rep.prediction or rep.rule_quality or f"rep{rep.rep_id}"
        ax.text(
            rep.bottom_frame,
            min(values) - 5,
            f"R{rep.rep_id}:{label}",
            ha="center",
            fontsize=8,
        )

    ax.legend(loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def build_evaluation_report(artifacts: EvaluationArtifacts) -> dict[str, Any]:
    """JSON-serializable summary for inspection."""
    return {
        "source_id": artifacts.source_id,
        "exercise": artifacts.exercise,
        "video": str(artifacts.video_path),
        "frame_count": artifacts.frame_count,
        "rep_count": len(artifacts.reps),
        "repetitions": [
            {
                "rep_id": r.rep_id,
                "frames": f"{r.start_frame}-{r.end_frame}",
                "bottom_frame": r.bottom_frame,
                "bottom_knee_angle": r.bottom_knee_angle,
                "rule_quality": r.rule_quality,
                "form_score": r.form_score,
                "model_prediction": r.prediction,
                "model_confidence": r.confidence,
                "mistakes": [m.get("mistake_id") for m in r.mistakes],
            }
            for r in artifacts.reps
        ],
    }
