"""Tests for rule-based form analysis with continuous scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.features.feature_pipeline import FrameFeatures
from src.features.rep_segmentation import Repetition
from src.feedback.form_analyzer import SquatFormAnalyzer, load_scoring_config


def _bottom_frame(knee_min: float, lean: float, asym: float) -> FrameFeatures:
    return FrameFeatures(
        frame_index=10,
        timestamp_sec=1.0,
        angles={"torso_lean": lean},
        derived={"knee_angle_min": knee_min, "knee_asymmetry_deg": asym},
        torso_length=0.15,
    )


def test_depth_rule_flags_shallow_squat():
    analyzer = SquatFormAnalyzer()
    rep = Repetition(
        rep_id=1,
        start_frame=0,
        end_frame=20,
        bottom_frame=10,
        start_time_sec=0.0,
        end_time_sec=2.0,
        duration_sec=2.0,
        bottom_knee_angle=110.0,
    )
    frames = [
        FrameFeatures(0, 0.0, {}, {"knee_angle_min": 160.0}),
        _bottom_frame(110.0, 20.0, 5.0),
        FrameFeatures(20, 2.0, {}, {"knee_angle_min": 150.0}),
    ]
    result = analyzer.analyze_rep(rep, frames)
    ids = [m.mistake_id for m in result.mistakes]
    assert "insufficient_depth" in ids
    assert result.scores["depth_score"] < 70
    assert "shallow_depth" in result.flags


def test_good_depth_no_depth_flag():
    analyzer = SquatFormAnalyzer()
    rep = Repetition(
        rep_id=1,
        start_frame=0,
        end_frame=20,
        bottom_frame=10,
        start_time_sec=0.0,
        end_time_sec=2.0,
        duration_sec=2.0,
        bottom_knee_angle=80.0,
    )
    frames = [_bottom_frame(80.0, 20.0, 5.0)]
    result = analyzer.analyze_rep(rep, frames)
    assert "insufficient_depth" not in [m.mistake_id for m in result.mistakes]
    assert result.scores["depth_score"] >= 70


def test_forward_lean_rule():
    analyzer = SquatFormAnalyzer()
    rep = Repetition(
        rep_id=1, start_frame=0, end_frame=10, bottom_frame=5,
        start_time_sec=0.0, end_time_sec=1.0, duration_sec=1.0, bottom_knee_angle=85.0,
    )
    frames = [_bottom_frame(85.0, lean=50.0, asym=5.0)]
    result = analyzer.analyze_rep(rep, frames)
    assert "excessive_forward_lean" in [m.mistake_id for m in result.mistakes]
    assert result.scores["torso_control_score"] < 70


def test_rep_json_schema_fields():
    analyzer = SquatFormAnalyzer()
    rep = Repetition(
        rep_id=1, start_frame=0, end_frame=10, bottom_frame=5,
        start_time_sec=0.0, end_time_sec=1.0, duration_sec=1.0, bottom_knee_angle=85.0,
    )
    frames = [_bottom_frame(85.0, lean=25.0, asym=5.0)]
    result = analyzer.analyze_rep(rep, frames)
    data = result.to_dict()
    assert "scores" in data
    assert "confidence" in data
    assert "flags" in data
    assert "feedback" in data
    assert "coaching" in data
    assert set(data["scores"].keys()) >= {
        "depth_score",
        "knee_tracking_score",
        "torso_control_score",
        "symmetry_score",
        "stability_score",
        "heel_control_score",
        "overall_score",
    }


def test_scoring_config_loads():
    cfg = load_scoring_config("squat")
    assert "weights" in cfg
    assert cfg["weights"]["depth_score"] == 0.25


def test_analyze_sample_pipeline():
    features = Path("data/processed/features/sample_squat/features.csv")
    reps = Path("data/processed/reps/sample_squat/reps.json")
    if not features.exists() or not reps.exists():
        pytest.skip("run feature + rep pipeline first")

    analyzer = SquatFormAnalyzer()
    result = analyzer.analyze(features, reps)
    assert result.output_dir.joinpath("form_analysis.json").exists()
    assert len(result.rep_analyses) >= 1
    assert result.video_summary["num_reps"] == len(result.rep_analyses)
    assert result.analyzer_version.startswith("0.2")
