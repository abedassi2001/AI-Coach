"""Tests for joint angles and feature extraction (Phase 4)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from backend.features.angles import angle_at_point_degrees, compute_configured_angles, compute_derived_features
from backend.features.feature_pipeline import FeatureExtractionPipeline, load_pose_sequence, smooth_series
from backend.features.normalization import compute_body_scale, euclidean_distance, midpoint
from backend.pose.keypoint_schema import Keypoint, PoseFrame, PoseSequence


def _kp(x: float, y: float) -> Keypoint:
    return Keypoint(x=x, y=y, visibility=1.0)


def test_angle_right_angle():
    # L-shape: angle at origin between (1,0) and (0,1) vectors from (0,0)
    a = _kp(1, 0)
    b = _kp(0, 0)
    c = _kp(0, 1)
    assert angle_at_point_degrees(a, b, c) == pytest.approx(90.0, abs=0.01)


def test_angle_straight_line():
    a = _kp(0, 0)
    b = _kp(1, 0)
    c = _kp(2, 0)
    assert angle_at_point_degrees(a, b, c) == pytest.approx(180.0, abs=0.01)


def test_midpoint():
    m = midpoint(_kp(0, 0), _kp(2, 4))
    assert m.x == pytest.approx(1.0)
    assert m.y == pytest.approx(2.0)


def test_knee_artifact_clamped_to_hip_proxy():
    derived = compute_derived_features(
        {"left_knee": 6.0, "right_knee": 8.0, "left_hip": 95.0, "right_hip": 92.0}
    )
    assert derived["knee_from_hip_proxy"] == 1.0
    assert derived["squat_depth_angle"] == pytest.approx(92.0)


def test_torso_length():
    frame = PoseFrame(
        frame_index=0,
        timestamp_sec=0.0,
        landmarks={
            "left_shoulder": _kp(0.4, 0.3),
            "right_shoulder": _kp(0.6, 0.3),
            "left_hip": _kp(0.4, 0.6),
            "right_hip": _kp(0.6, 0.6),
        },
    )
    scale = compute_body_scale(frame)
    assert scale is not None
    assert scale.torso_length == pytest.approx(0.3, abs=0.01)


def test_compute_configured_angles_squat_frame():
    frame = PoseFrame(
        frame_index=0,
        timestamp_sec=0.0,
        landmarks={
            "left_shoulder": _kp(0.4, 0.2),
            "right_shoulder": _kp(0.6, 0.2),
            "left_hip": _kp(0.4, 0.5),
            "right_hip": _kp(0.6, 0.5),
            "left_knee": _kp(0.4, 0.7),
            "right_knee": _kp(0.6, 0.7),
            "left_ankle": _kp(0.4, 0.9),
            "right_ankle": _kp(0.6, 0.9),
        },
    )
    angle_defs = [
        {"name": "left_knee", "points": ["left_hip", "left_knee", "left_ankle"]},
        {"name": "torso_lean", "points": ["mid_hip", "mid_shoulder", "vertical_up"]},
    ]
    angles = compute_configured_angles(frame, angle_defs)
    assert "left_knee" in angles
    assert angles["left_knee"] == pytest.approx(180.0, abs=1.0)  # straight leg
    assert "torso_lean" in angles


def test_derived_knee_features():
    derived = compute_derived_features({"left_knee": 100.0, "right_knee": 110.0})
    assert derived["knee_angle_avg"] == pytest.approx(105.0)
    assert derived["knee_angle_min"] == pytest.approx(100.0)
    assert derived["knee_asymmetry_deg"] == pytest.approx(10.0)


def test_smooth_series():
    assert smooth_series([1.0, 5.0, 3.0], window=3) == pytest.approx([3.0, 3.0, 4.0])


def test_feature_pipeline_on_sample_keypoints():
    keypoints = Path("data/processed/pose/sample_squat/keypoints.json")
    if not keypoints.exists():
        pytest.skip("sample keypoints not present — run extract_pose first")

    pipeline = FeatureExtractionPipeline(smoothing_window=1)
    result = pipeline.extract_from_keypoints_file(keypoints)

    assert len(result.frames) > 0
    assert result.csv_path.exists()
    assert result.json_path.exists()
    assert "angle_left_knee" in result.frames[0].to_flat_dict(result.source_id, result.exercise)
    assert result.frames[0].derived.get("knee_angle_min", 0) > 0
