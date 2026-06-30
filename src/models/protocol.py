"""Abstract interface for exercise form classifiers (baseline + deep models)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FormClassifier(ABC):
    """Predict rep-level form quality. Implementations: sklearn baseline, LSTM, etc."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier stored in checkpoints."""

    @abstractmethod
    def fit(self, X: Any, y: Any, groups: Any | None = None) -> None:
        """Train on feature matrix X, labels y, optional groups for split-by-video."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict class labels."""

    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities."""

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> FormClassifier:
        ...
