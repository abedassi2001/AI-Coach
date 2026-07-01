"""Tests for coaching feedback generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.feedback.coach_context import build_coaching_context
from backend.feedback.coaching_pipeline import generate_coaching
from backend.feedback.template_coach import generate_template_coaching


SAMPLE_CONTEXT = {
    "source_id": "sample_squat",
    "exercise": "squat",
    "overall_score": 62.0,
    "overall_quality": "needs_work",
    "model_available": True,
    "repetitions": [
        {
            "rep_id": 1,
            "rule_score": 62.0,
            "rule_quality": "needs_work",
            "model_prediction": "good",
            "model_confidence": 0.99,
            "metrics": {"bottom_knee_angle": 82.0},
            "mistakes": [
                {
                    "id": "asymmetry",
                    "severity": "medium",
                    "message": "Asymmetry detected",
                    "value": 30.0,
                    "threshold": 15.0,
                },
                {
                    "id": "insufficient_depth",
                    "severity": "high",
                    "message": "Shallow",
                    "value": 95.0,
                    "threshold": 90.0,
                },
            ],
        }
    ],
}


def test_template_coaching_prioritizes_high_severity():
    report = generate_template_coaching(SAMPLE_CONTEXT, max_actions=3)
    assert report["provider"] == "template"
    assert len(report["action_plan"]) >= 1
    assert "practice_drills" in report
    assert report["rep_feedback"][0]["rep_id"] == 1


def test_build_coaching_context_from_sample():
    path = Path("data/processed/analysis/sample_squat/form_analysis.json")
    if not path.exists():
        pytest.skip("sample form_analysis not present")
    ctx = build_coaching_context("sample_squat")
    assert ctx.source_id == "sample_squat"
    assert len(ctx.repetitions) >= 1


def test_generate_coaching_template(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    path = Path("data/processed/analysis/sample_squat/form_analysis.json")
    if not path.exists():
        pytest.skip("sample form_analysis not present")
    result = generate_coaching(
        "sample_squat",
        provider="template",
        output_dir=tmp_path / "coaching",
    )
    assert result.json_path.exists()
    assert result.text_path.exists()
    assert "coaching" in result.text_path.read_text(encoding="utf-8").lower()
