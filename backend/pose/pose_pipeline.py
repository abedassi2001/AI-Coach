"""Orchestrate pose extraction from videos or pre-extracted frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from backend.data.video_loader import VideoLoader
from backend.pose.base import PoseEstimator
from backend.pose.factory import create_pose_estimator
from backend.pose.keypoint_schema import PoseFrame, PoseSequence
from backend.utils.config import load_config, resolve_path


@dataclass
class PoseExtractionResult:
    """Artifacts from a pose extraction run."""

    sequence: PoseSequence
    output_dir: Path
    saved_paths: dict[str, Path]
    overlay_video_path: Path | None = None


class PoseExtractionPipeline:
    """
    Exercise-agnostic pose extraction.

    The `exercise` field tags output for downstream analyzers (squat rules,
    deadlift rules, etc.) but does not change how landmarks are detected.
    """

    def __init__(
        self,
        estimator: PoseEstimator | None = None,
        backend: str | None = None,
        pose_config: dict[str, Any] | None = None,
        video_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = load_config()
        self.pose_config = pose_config or cfg.get("pose", {})
        self.video_config = video_config or cfg.get("video", {})

        if estimator is not None:
            self.estimator = estimator
        else:
            backend_name = backend or self.pose_config.get("backend", "mediapipe")
            self.estimator = create_pose_estimator(
                backend=backend_name,
                min_detection_confidence=self.pose_config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=self.pose_config.get("min_tracking_confidence", 0.5),
            )

    def close(self) -> None:
        self.estimator.close()

    def __enter__(self) -> PoseExtractionPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def process_video(
        self,
        video_path: str | Path,
        exercise: str = "squat",
        output_dir: str | Path | None = None,
        output_format: str | None = None,
        write_overlay_video: bool = False,
        max_frames: int | None = None,
    ) -> PoseExtractionResult:
        """Extract pose directly from a video file."""
        video_path = Path(video_path)
        source_id = video_path.stem
        out_dir = self._resolve_output_dir(output_dir, source_id)
        fmt = output_format or self.pose_config.get("output_format", "json")

        frames_data: list[tuple[Any, int, float]] = []
        with VideoLoader(video_path) as loader:
            meta = loader.get_metadata()
            cap = loader.open()
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = idx / meta.fps if meta.fps > 0 else 0.0
                frames_data.append((frame, idx, timestamp))
                idx += 1
                if max_frames is not None and idx >= max_frames:
                    break

        sequence = self._estimate_sequence(
            frames_data=frames_data,
            source_id=source_id,
            exercise=exercise,
            extra_metadata={
                "source_video": str(video_path.resolve()),
                "source_fps": meta.fps,
                "source_frame_count": meta.frame_count,
            },
        )

        saved = sequence.save(out_dir, output_format=fmt)
        overlay_path = None
        if write_overlay_video:
            overlay_path = self._write_overlay_video(
                frames_data, sequence, out_dir / f"{source_id}_skeleton.mp4", meta.fps
            )

        return PoseExtractionResult(
            sequence=sequence,
            output_dir=out_dir,
            saved_paths=saved,
            overlay_video_path=overlay_path,
        )

    def process_frames_dir(
        self,
        frames_dir: str | Path,
        exercise: str = "squat",
        manifest_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        output_format: str | None = None,
        write_overlay_video: bool = False,
        max_frames: int | None = None,
    ) -> PoseExtractionResult:
        """Extract pose from a directory of images (e.g. Phase 2 output)."""
        frames_dir = Path(frames_dir)
        source_id = frames_dir.name
        out_dir = self._resolve_output_dir(output_dir, source_id)
        fmt = output_format or self.pose_config.get("output_format", "json")

        manifest = self._load_manifest(manifest_path, frames_dir)
        image_paths = sorted(frames_dir.glob("frame_*.jpg"))
        if not image_paths:
            image_paths = sorted(frames_dir.glob("frame_*.png"))
        if max_frames is not None:
            image_paths = image_paths[:max_frames]

        frames_data: list[tuple[Any, int, float]] = []
        for i, path in enumerate(image_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if manifest and i < len(manifest.get("frames", [])):
                entry = manifest["frames"][i]
                frame_index = int(entry.get("source_index", entry.get("output_index", i)))
                timestamp = float(entry.get("timestamp_sec", 0.0))
            else:
                frame_index = i
                timestamp = 0.0
            frames_data.append((frame, frame_index, timestamp))

        sequence = self._estimate_sequence(
            frames_data=frames_data,
            source_id=source_id,
            exercise=exercise,
            extra_metadata={"frames_dir": str(frames_dir.resolve())},
        )

        saved = sequence.save(out_dir, output_format=fmt)
        overlay_path = None
        if write_overlay_video and frames_data:
            fps = float(manifest.get("effective_fps", 30)) if manifest else 30.0
            overlay_path = self._write_overlay_video(
                frames_data, sequence, out_dir / f"{source_id}_skeleton.mp4", fps
            )

        return PoseExtractionResult(
            sequence=sequence,
            output_dir=out_dir,
            saved_paths=saved,
            overlay_video_path=overlay_path,
        )

    def _estimate_sequence(
        self,
        frames_data: list[tuple[Any, int, float]],
        source_id: str,
        exercise: str,
        extra_metadata: dict[str, Any],
    ) -> PoseSequence:
        sequence = PoseSequence(
            source_id=source_id,
            exercise=exercise,
            backend=self.estimator.backend_name,
            landmark_names=self.estimator.landmark_names,
            metadata=extra_metadata,
        )
        for frame, frame_index, timestamp in frames_data:
            pose_frame = self.estimator.estimate(frame, frame_index, timestamp)
            if pose_frame is not None:
                sequence.frames.append(pose_frame)
        sequence.metadata["detected_frames"] = len(sequence.frames)
        sequence.metadata["total_frames"] = len(frames_data)
        return sequence

    def _resolve_output_dir(self, output_dir: str | Path | None, source_id: str) -> Path:
        if output_dir is not None:
            out = Path(output_dir)
        else:
            base = resolve_path("data/processed/pose")
            out = base / source_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _load_manifest(manifest_path: str | Path | None, frames_dir: Path) -> dict | None:
        path = Path(manifest_path) if manifest_path else frames_dir / "manifest.json"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _write_overlay_video(
        self,
        frames_data: list[tuple[Any, int, float]],
        sequence: PoseSequence,
        output_path: Path,
        fps: float,
    ) -> Path | None:
        from backend.visualization.skeleton import draw_skeleton_on_frame

        if not frames_data:
            return None

        pose_by_index = {f.frame_index: f for f in sequence.frames}
        height, width = frames_data[0][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1.0), (width, height))

        for frame, frame_index, _ in frames_data:
            pose = pose_by_index.get(frame_index)
            annotated = draw_skeleton_on_frame(frame, pose) if pose else frame
            writer.write(annotated)

        writer.release()
        return output_path
