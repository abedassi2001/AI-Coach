"""Inference wrapper for rep-level form classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.ml.baseline_classifier import BaselineFormClassifier
from backend.training.rep_dataset import build_rep_rows_for_source


class RepQualityPredictor:
    """Load trained classifier and predict on processed recordings."""

    def __init__(self, model: BaselineFormClassifier) -> None:
        self.model = model

    @classmethod
    def load(cls, path: str) -> RepQualityPredictor:
        return cls(BaselineFormClassifier.load(path))

    def predict_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        preds = self.model.predict(rows)
        probas = self.model.predict_proba(rows)
        classes = self.model.classes_
        results = []
        for i, row in enumerate(rows):
            proba = probas[i]
            best_idx = int(proba.argmax())
            results.append(
                {
                    "source_id": row.get("source_id"),
                    "exercise": row.get("exercise"),
                    "rep_id": row.get("rep_id"),
                    "prediction": preds[i],
                    "confidence": float(proba[best_idx]),
                    "probabilities": {classes[j]: float(proba[j]) for j in range(len(classes))},
                }
            )
        return results

    def predict_source(self, source_id: str) -> list[dict[str, Any]]:
        rows = build_rep_rows_for_source(source_id)
        return self.predict_rows(rows)
