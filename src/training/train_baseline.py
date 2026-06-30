"""Train exercise-scalable baseline form classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.models.baseline_classifier import BaselineFormClassifier
from src.training.metrics import compute_metrics, plot_confusion_matrix, save_evaluation_report
from src.training.rep_dataset import RepDatasetBuilder
from src.utils.config import get_project_root


@dataclass
class TrainingResult:
    model_path: Path
    report_path: Path
    metrics: dict[str, Any]
    n_train: int
    n_val: int
    feature_count: int


def load_training_config() -> dict[str, Any]:
    path = get_project_root() / "configs/training/baseline.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _group_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by source_id so reps from the same video stay in one fold."""
    if "source_id" not in df.columns or df["source_id"].nunique() < 2:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"] if df["label"].nunique() > 1 else None,
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["source_id"].values
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx], df.iloc[val_idx]


def train_baseline_classifier(
    config: dict[str, Any] | None = None,
    source_ids: list[str] | None = None,
    demo_augment: bool = False,
) -> TrainingResult:
    cfg = config or load_training_config()
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})

    min_samples = int(train_cfg.get("min_samples", 8))
    test_size = float(train_cfg.get("test_size", 0.2))
    random_state = int(model_cfg.get("random_state", 42))

    builder = RepDatasetBuilder()
    df = builder.labeled_only(builder.build(source_ids=source_ids))

    if demo_augment and len(df) < min_samples and len(df) > 0:
        from src.training.synthetic import augment_labeled_reps

        df = augment_labeled_reps(df, target_size=max(min_samples, 20))

    if len(df) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} labeled reps to train. Found {len(df)}. "
            "Add rows to data/raw/labels/rep_labels.csv, run scripts/bootstrap_labels.py, "
            "or use --demo for synthetic augmentation (development only)."
        )

    if df["label"].nunique() < 2:
        raise ValueError(
            "Need at least 2 classes (good/bad) in labels. "
            f"Found: {df['label'].unique().tolist()}"
        )

    train_df, val_df = _group_split(df, test_size=test_size, random_state=random_state)

    model = BaselineFormClassifier(
        model_type=model_cfg.get("type", "gradient_boosting"),
        random_state=random_state,
    )
    model.fit(train_df, train_df["label"].values, groups=train_df.get("source_id"))

    y_pred = model.predict(val_df)
    metrics = compute_metrics(
        val_df["label"].values,
        y_pred,
        labels=sorted(df["label"].unique().tolist()),
    )
    metrics["n_train"] = len(train_df)
    metrics["n_val"] = len(val_df)
    metrics["exercises"] = sorted(df["exercise"].unique().tolist())
    metrics["model"] = model.metadata_json()

    root = get_project_root()
    ckpt_dir = Path(paths_cfg.get("checkpoint_dir", "models/checkpoints/baseline"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    reports_dir = Path(paths_cfg.get("reports_dir", ckpt_dir / "reports"))
    if not reports_dir.is_absolute():
        reports_dir = root / reports_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "form_classifier.joblib"
    model.save(str(model_path))
    model.save_metadata(str(ckpt_dir / "model_metadata.json"))

    report_path = reports_dir / "evaluation.json"
    save_evaluation_report(metrics, report_path)

    cm_path = reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["labels"],
        cm_path,
        title="Baseline Form Classifier",
    )

    return TrainingResult(
        model_path=model_path,
        report_path=report_path,
        metrics=metrics,
        n_train=len(train_df),
        n_val=len(val_df),
        feature_count=len(model.feature_columns),
    )
