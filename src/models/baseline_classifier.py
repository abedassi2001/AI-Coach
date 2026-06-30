"""Scikit-learn baseline form classifier — scalable across exercises."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.protocol import FormClassifier


class BaselineFormClassifier(FormClassifier):
    """
    Gradient boosting / random forest on rep-level features.

    - `exercise` is one-hot encoded → add deadlift/bench without retraining architecture
    - Numeric rep aggregates are standardized
    - Unknown exercises at inference get empty one-hot branch (handle_unknown='ignore')
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        random_state: int = 42,
        feature_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.feature_columns = feature_columns or []
        self.categorical_columns = categorical_columns or ["exercise"]
        self.pipeline: Pipeline | None = None
        self.classes_: list[str] = []

    @property
    def name(self) -> str:
        return f"baseline_{self.model_type}"

    def _build_estimator(self) -> Any:
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=self.random_state,
            )
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=1000, class_weight="balanced")
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            random_state=self.random_state,
        )

    def _build_pipeline(self) -> Pipeline:
        numeric = self.feature_columns
        categorical = self.categorical_columns

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("clf", self._build_estimator()),
            ]
        )

    def _to_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, list):
            return pd.DataFrame(X)
        raise TypeError("X must be DataFrame or list of dicts")

    def _prepare_matrix(self, X: Any) -> pd.DataFrame:
        df = self._to_dataframe(X)
        cols = self.feature_columns + self.categorical_columns
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        out = df[cols].copy()
        for col in self.feature_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def fit(self, X: Any, y: Any, groups: Any | None = None) -> None:
        df = self._to_dataframe(X)
        if not self.feature_columns:
            exclude = {
                "source_id", "rep_id", "label", "weak_label", "synthetic",
                "synthetic_mistake", "label_source",
                *self.categorical_columns,
            }
            self.feature_columns = [
                c
                for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
        self.pipeline = self._build_pipeline()
        matrix = self._prepare_matrix(df)
        self.pipeline.fit(matrix, y)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)

    def predict(self, X: Any) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted")
        return self.pipeline.predict(self._prepare_matrix(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted")
        return self.pipeline.predict_proba(self._prepare_matrix(X))

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "random_state": self.random_state,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "classes": self.classes_,
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, out)

    @classmethod
    def load(cls, path: str) -> BaselineFormClassifier:
        payload = joblib.load(path)
        obj = cls(
            model_type=payload["model_type"],
            random_state=payload.get("random_state", 42),
            feature_columns=payload.get("feature_columns", []),
            categorical_columns=payload.get("categorical_columns", ["exercise"]),
        )
        obj.pipeline = payload["pipeline"]
        obj.classes_ = payload.get("classes", [])
        return obj

    def metadata_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "classes": self.classes_,
            "n_features": len(self.feature_columns),
        }

    def save_metadata(self, path: str) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.metadata_json(), f, indent=2)
