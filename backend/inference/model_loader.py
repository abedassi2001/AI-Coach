"""Safe loading of the optional ML classifier checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.inference.rep_classifier import RepQualityPredictor

logger = logging.getLogger(__name__)


def try_load_predictor(model_path: str | Path | None) -> RepQualityPredictor | None:
    """
    Load RepQualityPredictor if the checkpoint exists and is compatible.

    Returns None instead of raising when the file is missing or sklearn versions
    mismatch (common after upgrading scikit-learn). Rule-based scoring still works.
    """
    path = Path(model_path) if model_path else None
    if path is None or not path.exists():
        return None

    try:
        from backend.inference.rep_classifier import RepQualityPredictor

        return RepQualityPredictor.load(str(path))
    except Exception as exc:
        logger.warning("Could not load ML model at %s: %s", path, exc)
        return None


def model_load_hint(model_path: Path) -> str:
    return (
        f"ML model unavailable ({model_path.name}). "
        "Rule-based scores still work. Retrain with: python scripts/train_classifier.py --demo"
    )
