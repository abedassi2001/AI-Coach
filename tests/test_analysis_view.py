"""Tests for frontend analysis view models."""

from __future__ import annotations

from app.analysis_view import build_full_view, build_summary
from app.issue_copy import issue_copy_for, performance_label


def test_performance_labels():
    assert performance_label(95) == "Excellent"
    assert performance_label(80) == "Good"
    assert performance_label(65) == "Needs Work"
    assert performance_label(50) == "Poor Form"
    assert performance_label(30) == "High Risk"


def test_issue_copy_human_readable():
    copy = issue_copy_for("shallow_depth")
    assert "depth" in copy["title"].lower()
    assert copy["cue"]


def test_build_summary_from_minimal_json():
    analysis = {
        "source_id": "test",
        "overall_score": 76,
        "video_summary": {
            "num_reps": 3,
            "average_scores": {
                "depth_score": 55,
                "knee_tracking_score": 88,
                "torso_control_score": 72,
                "symmetry_score": 80,
                "stability_score": 78,
                "heel_control_score": 75,
            },
            "main_issues": ["shallow_depth"],
            "worst_dimension": "depth_score",
            "best_dimension": "knee_tracking_score",
        },
        "repetitions": [],
    }
    s = build_summary(analysis)
    assert s.overall_score == 76
    assert s.performance_label == "Good"
    assert s.main_issue_title
    assert s.quick_fix
    assert s.positive_title


def test_build_summary_fallback_empty():
    s = build_summary(None)
    assert s.overall_score == 0
    assert "incomplete" in s.main_issue_title.lower()


def test_build_full_view_computes_best_worst():
    analysis = {
        "source_id": "x",
        "overall_score": 70,
        "video_summary": {"average_scores": {"depth_score": 50, "knee_tracking_score": 90}},
        "repetitions": [{"rep_id": 1, "overall_score": 70, "scores": {"depth_score": 50}}],
    }
    v = build_full_view(analysis)
    assert v.worst_dimension == "depth_score"
    assert v.best_dimension == "knee_tracking_score"
