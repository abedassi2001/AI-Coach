"""Tests for continuous scoring utilities."""

from __future__ import annotations

import pytest

from backend.feedback.rep_coaching import (
    build_video_summary,
    flags_from_scores,
    generate_rep_coaching,
)
from backend.feedback.scoring import (
    clamp,
    linear_score,
    quality_label,
    score_angle_range,
    weighted_average,
)
from backend.feedback.squat_dimensions import score_depth, score_torso_control


def test_clamp():
    assert clamp(150) == 100.0
    assert clamp(-5) == 0.0
    assert clamp(50) == 50.0


def test_linear_score_ideal():
    assert linear_score(80, ideal_value=90, warning_value=100, fail_value=120) == 100.0


def test_linear_score_degrades_smoothly():
    mid = linear_score(100, ideal_value=90, warning_value=110, fail_value=130)
    bad = linear_score(125, ideal_value=90, warning_value=110, fail_value=130)
    assert 70 <= mid <= 100
    assert bad < mid
    assert bad >= 0


def test_linear_score_nan_returns_neutral():
    assert linear_score(float("nan"), 0, 1, 2) == 50.0


def test_score_angle_range_inside_band():
    assert score_angle_range(85, ideal_min=80, ideal_max=95) == 100.0


def test_weighted_average_renormalizes():
    scores = {"a": 100.0, "b": 50.0}
    weights = {"a": 0.5, "b": 0.5}
    assert weighted_average(scores, weights) == 75.0


def test_quality_label_bands():
    assert quality_label(85) == "good"
    assert quality_label(75) == "acceptable"
    assert quality_label(55) == "needs_work"


def test_depth_score_shallow_is_low():
    cfg = {"dimensions": {"depth": {"ideal": 80, "warning": 95, "fail": 115}}}
    deep = score_depth(75, cfg)
    shallow = score_depth(110, cfg)
    assert deep > shallow
    assert shallow < 70


def test_torso_control_high_lean_is_low():
    cfg = {"dimensions": {"torso_control": {"ideal": 28, "warning": 42, "fail": 55}}}
    good = score_torso_control(25, cfg)
    bad = score_torso_control(50, cfg)
    assert good > bad


def test_flags_from_scores():
    thresholds = {
        "depth_score": [["shallow_depth", 70], ["severe_shallow_depth", 40]],
    }
    flags = flags_from_scores({"depth_score": 55}, thresholds)
    assert "shallow_depth" in flags
    assert "severe_shallow_depth" not in flags

    severe = flags_from_scores({"depth_score": 30}, thresholds)
    assert "severe_shallow_depth" in severe


def test_generate_rep_coaching_has_required_fields():
    scores = {
        "depth_score": 55,
        "knee_tracking_score": 88,
        "torso_control_score": 72,
        "symmetry_score": 80,
        "stability_score": 78,
        "heel_control_score": 75,
        "overall_score": 72,
    }
    out = generate_rep_coaching(scores, ["shallow_depth"])
    assert "overall_summary" in out
    assert out["top_issues"]
    assert out["correction_cue"]
    assert out["feedback"]


def test_build_video_summary_schema():
    reps = [
        {
            "overall_score": 70,
            "scores": {
                "depth_score": 60,
                "knee_tracking_score": 90,
                "torso_control_score": 75,
                "symmetry_score": 80,
                "stability_score": 78,
                "heel_control_score": 72,
            },
            "flags": ["shallow_depth"],
        },
        {
            "overall_score": 80,
            "scores": {
                "depth_score": 85,
                "knee_tracking_score": 88,
                "torso_control_score": 77,
                "symmetry_score": 82,
                "stability_score": 79,
                "heel_control_score": 74,
            },
            "flags": [],
        },
    ]
    summary = build_video_summary(reps, "0.2.0-test")
    assert summary["num_reps"] == 2
    assert summary["average_overall_score"] == 75.0
    assert summary["worst_dimension"] == "depth_score"
    assert summary["best_dimension"] == "knee_tracking_score"
    assert "shallow_depth" in summary["main_issues"]
