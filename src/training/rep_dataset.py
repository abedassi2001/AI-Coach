"""Build labeled rep datasets from processed pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.features.rep_features import extract_rep_features
from src.features.rep_segmentation import (
    Repetition,
    load_frame_features_from_csv,
)
from src.feedback.form_analyzer import load_repetitions
from src.feedback.form_analyzer import SquatFormAnalyzer
from src.utils.config import get_project_root, load_config, resolve_path


def load_label_table(path: str | Path) -> pd.DataFrame:
    """Load rep_labels.csv: source_id, exercise, rep_id, label."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["source_id", "exercise", "rep_id", "label"])
    return pd.read_csv(p)


def _frames_for_rep(frames, rep: Repetition):
    return [f for f in frames if rep.start_frame <= f.frame_index <= rep.end_frame]


def build_rep_rows_for_source(
    source_id: str,
    features_dir: Path | None = None,
    reps_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract rep-level feature rows for one recording."""
    root = get_project_root()
    feat_path = features_dir or root / "data/processed/features" / source_id / "features.csv"
    reps_path = reps_dir or root / "data/processed/reps" / source_id / "reps.json"

    if not feat_path.exists() or not reps_path.exists():
        return []

    _, exercise, frames = load_frame_features_from_csv(feat_path)
    _, ex2, repetitions = load_repetitions(reps_path)
    exercise = exercise or ex2

    rows: list[dict[str, Any]] = []
    for rep in repetitions:
        rep_frames = _frames_for_rep(frames, rep)
        row = extract_rep_features(rep, rep_frames, exercise)
        row["source_id"] = source_id
        rows.append(row)
    return rows


def discover_processed_sources() -> list[str]:
    root = resolve_path("data/processed/features")
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "features.csv").exists())


def merge_labels(
    rows: list[dict[str, Any]],
    labels: pd.DataFrame,
) -> list[dict[str, Any]]:
    if labels.empty:
        return rows
    label_map = {
        (str(r.source_id), str(r.exercise), int(r.rep_id)): str(r.label)
        for r in labels.itertuples(index=False)
    }
    for row in rows:
        key = (str(row["source_id"]), str(row["exercise"]), int(row["rep_id"]))
        if key in label_map:
            row["label"] = label_map[key]
    return rows


def bootstrap_weak_labels(
    rows: list[dict[str, Any]],
    positive_class: str = "good",
    negative_class: str = "bad",
    score_threshold: float = 80.0,
) -> list[dict[str, Any]]:
    """
    Assign labels from rule-based form analyzer when human labels are missing.

    Weak supervision — useful to bootstrap; replace with human labels for production.
    """
    root = get_project_root()
    analyzer = SquatFormAnalyzer()
    for row in rows:
        if row.get("label"):
            continue
        sid = row["source_id"]
        features = root / "data/processed/features" / sid / "features.csv"
        reps = root / "data/processed/reps" / sid / "reps.json"
        keypoints = root / "data/processed/pose" / sid / "keypoints.json"
        if not features.exists() or not reps.exists():
            continue
        try:
            result = analyzer.analyze(
                features,
                reps,
                keypoints if keypoints.exists() else None,
            )
            for rep_a in result.rep_analyses:
                if int(rep_a.rep_id) == int(row["rep_id"]):
                    row["label"] = (
                        positive_class
                        if rep_a.form_score >= score_threshold
                        else negative_class
                    )
                    row["weak_label"] = True
                    break
        except Exception:
            continue
    return rows


class RepDatasetBuilder:
    """Assemble training DataFrame from processed artifacts + labels."""

    def __init__(self, labels_path: str | Path | None = None) -> None:
        cfg = load_config()
        train_cfg_path = get_project_root() / "configs/training/baseline.yaml"
        train_cfg: dict = {}
        if train_cfg_path.exists():
            import yaml

            with train_cfg_path.open(encoding="utf-8") as f:
                train_cfg = yaml.safe_load(f) or {}

        labels_cfg = train_cfg.get("labels", {})
        self.labels_path = Path(
            labels_path or labels_cfg.get("path", "data/raw/labels/rep_labels.csv")
        )
        if not self.labels_path.is_absolute():
            self.labels_path = get_project_root() / self.labels_path
        self.positive_class = labels_cfg.get("positive_class", "good")
        self.negative_class = labels_cfg.get("negative_class", "bad")
        self.use_weak_labels = train_cfg.get("training", {}).get("use_weak_labels", True)

    def build(self, source_ids: list[str] | None = None) -> pd.DataFrame:
        ids = source_ids or discover_processed_sources()
        labels = load_label_table(self.labels_path)
        all_rows: list[dict[str, Any]] = []
        for sid in ids:
            all_rows.extend(build_rep_rows_for_source(sid))

        all_rows = merge_labels(all_rows, labels)
        if self.use_weak_labels:
            all_rows = bootstrap_weak_labels(
                all_rows,
                positive_class=self.positive_class,
                negative_class=self.negative_class,
            )

        df = pd.DataFrame(all_rows)
        if "label" not in df.columns:
            df["label"] = None
        return df

    def labeled_only(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["label"].notna() & (df["label"].astype(str).str.len() > 0)].copy()
