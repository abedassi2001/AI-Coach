"""Build structured coaching context from analysis + model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import get_project_root


@dataclass
class CoachingContext:
    """Structured facts passed to the coach (rules + ML — not raw video)."""

    source_id: str
    exercise: str
    overall_score: float
    overall_quality: str
    repetitions: list[dict[str, Any]] = field(default_factory=list)
    model_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "exercise": self.exercise,
            "overall_score": self.overall_score,
            "overall_quality": self.overall_quality,
            "model_available": self.model_available,
            "repetitions": self.repetitions,
        }


def load_form_analysis(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_model_predictions(source_id: str, model_path: Path | None = None) -> list[dict[str, Any]]:
    root = get_project_root()
    ckpt = model_path or root / "models/checkpoints/baseline/form_classifier.joblib"
    from src.inference.model_loader import try_load_predictor

    predictor = try_load_predictor(ckpt)
    if predictor is None:
        return []
    return predictor.predict_source(source_id)


def build_coaching_context(
    source_id: str,
    analysis_path: Path | None = None,
    model_path: Path | None = None,
) -> CoachingContext:
    root = get_project_root()
    analysis_path = analysis_path or root / "data/processed/analysis" / source_id / "form_analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Form analysis not found: {analysis_path}. Run: python scripts/analyze_form.py {source_id}"
        )

    analysis = load_form_analysis(analysis_path)
    predictions = load_model_predictions(source_id, model_path)
    pred_by_rep = {int(p["rep_id"]): p for p in predictions}

    reps_out: list[dict[str, Any]] = []
    for rep in analysis.get("repetitions", []):
        rid = int(rep["rep_id"])
        pred = pred_by_rep.get(rid, {})
        mistakes = [
            {
                "id": m.get("mistake_id"),
                "severity": m.get("severity"),
                "message": m.get("message"),
                "value": m.get("value"),
                "threshold": m.get("threshold"),
            }
            for m in rep.get("mistakes", [])
        ]
        reps_out.append(
            {
                "rep_id": rid,
                "rule_score": rep.get("overall_score", rep.get("form_score")),
                "rule_quality": rep.get("quality"),
                "scores": rep.get("scores", {}),
                "flags": rep.get("flags", []),
                "feedback": rep.get("feedback", []),
                "confidence": rep.get("confidence", {}),
                "coaching": rep.get("coaching", {}),
                "model_prediction": pred.get("prediction"),
                "model_confidence": pred.get("confidence"),
                "metrics": rep.get("metrics", {}),
                "mistakes": mistakes,
            }
        )

    return CoachingContext(
        source_id=analysis.get("source_id", source_id),
        exercise=analysis.get("exercise", "squat"),
        overall_score=float(analysis.get("overall_score", 0)),
        overall_quality=str(analysis.get("overall_quality", "unknown")),
        repetitions=reps_out,
        model_available=bool(predictions),
    )