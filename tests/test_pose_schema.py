"""Tests for pose schema and exercise config (no OpenCV/MediaPipe)."""

from __future__ import annotations

import json
from pathlib import Path

from src.pose.keypoint_schema import Keypoint, PoseFrame, PoseSequence
from src.utils.exercise_config import list_exercises, load_exercise_config


def test_list_exercises_includes_squat():
    exercises = list_exercises()
    assert "squat" in exercises


def test_load_squat_exercise_config():
    cfg = load_exercise_config("squat")
    assert cfg["id"] == "squat"
    assert "landmark_groups" in cfg
    assert "left_knee" in cfg["landmark_groups"]["lower_body_left"]


def test_keypoint_roundtrip():
    kp = Keypoint(x=0.5, y=0.3, z=-0.1, visibility=0.9)
    d = kp.to_dict()
    assert d["x"] == 0.5
    assert d["visibility"] == 0.9


def test_pose_sequence_save_json_csv(tmp_path: Path):
    frame = PoseFrame(
        frame_index=0,
        timestamp_sec=0.0,
        landmarks={"left_knee": Keypoint(0.4, 0.6, 0.0, 1.0)},
        width=640,
        height=480,
    )
    seq = PoseSequence(
        source_id="test_vid",
        exercise="squat",
        backend="mediapipe",
        landmark_names=["left_knee"],
        frames=[frame],
    )
    json_path = seq.save_json(tmp_path / "keypoints.json")
    csv_path = seq.save_csv(tmp_path / "keypoints.csv")

    assert json_path.exists()
    assert csv_path.exists()
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["exercise"] == "squat"
    assert data["frame_count"] == 1
