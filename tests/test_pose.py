"""Integration tests for MediaPipe pose estimation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_create_mediapipe_estimator():
    pytest.importorskip("mediapipe")
    from backend.pose.factory import create_pose_estimator

    estimator = create_pose_estimator("mediapipe")
    assert estimator.backend_name == "mediapipe"
    assert len(estimator.landmark_names) == 33
    estimator.close()


def test_create_unknown_backend_raises():
    from backend.pose.factory import create_pose_estimator

    with pytest.raises(ValueError, match="Unknown pose backend"):
        create_pose_estimator("invalid")


def test_mediapipe_estimator_on_blank_frame():
    pytest.importorskip("mediapipe")
    np = pytest.importorskip("numpy")
    from backend.pose.factory import create_pose_estimator

    estimator = create_pose_estimator("mediapipe", min_detection_confidence=0.1)
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame, frame_index=0, timestamp_sec=0.0)
        assert result is None
    finally:
        estimator.close()


def test_pose_pipeline_process_frames_dir(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from backend.pose.keypoint_schema import Keypoint, PoseFrame
    from backend.pose.pose_pipeline import PoseExtractionPipeline

    frames_dir = tmp_path / "squat_clip"
    frames_dir.mkdir()
    for i in range(3):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

    mock_frame = PoseFrame(
        frame_index=0,
        timestamp_sec=0.0,
        landmarks={"left_hip": Keypoint(0.5, 0.5)},
        width=320,
        height=240,
    )
    mock_estimator = MagicMock()
    mock_estimator.backend_name = "mediapipe"
    mock_estimator.landmark_names = ["left_hip"]
    mock_estimator.estimate.return_value = mock_frame

    pipeline = PoseExtractionPipeline(estimator=mock_estimator)
    result = pipeline.process_frames_dir(
        frames_dir=frames_dir,
        exercise="squat",
        output_dir=tmp_path / "pose_out",
        output_format="json",
    )
    pipeline.close()

    assert len(result.sequence.frames) == 3
    assert result.sequence.exercise == "squat"
    assert (tmp_path / "pose_out" / "keypoints.json").exists()
