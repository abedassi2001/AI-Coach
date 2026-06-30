#!/usr/bin/env python3
"""
Demo: show model predictions on real reps vs synthetic mistake variants.

Useful to verify the classifier learned from human labels, not rule-engine noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.inference.rep_classifier import RepQualityPredictor
from src.training.rep_dataset import build_rep_rows_for_source, merge_labels, load_label_table
from src.training.synthetic import MISTAKE_PERTURBATIONS, generate_contrastive_bad_reps


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo model on real + synthetic mistake reps.")
    parser.add_argument("source_id", nargs="?", default="sample_squat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/checkpoints/baseline/form_classifier.joblib"),
    )
    args = parser.parse_args()

    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    if not model_path.exists():
        print(f"Train first: python scripts/train_classifier.py --demo", file=sys.stderr)
        return 1

    predictor = RepQualityPredictor.load(str(model_path))
    rows = build_rep_rows_for_source(args.source_id)
    labels = load_label_table(PROJECT_ROOT / "data/raw/labels/rep_labels.csv")
    rows = merge_labels(rows, labels)

    print(f"\n=== Real reps: {args.source_id} ===\n")
    print(f"{'Rep':>4}  {'Human':>6}  {'Model':>6}  {'Conf':>6}  {'Bottom knee':>12}")
    print("-" * 48)
    for r in predictor.predict_rows(rows):
        human = next((x.get("label") for x in rows if int(x["rep_id"]) == int(r["rep_id"])), "?")
        bottom = next((x.get("rep_bottom_knee_angle") for x in rows if int(x["rep_id"]) == int(r["rep_id"])), 0)
        print(
            f"{r['rep_id']:>4}  {str(human):>6}  {r['prediction']:>6}  "
            f"{r['confidence']:>5.0%}  {float(bottom):>10.1f}°"
        )

    good_rows = [r for r in rows if str(r.get("label", "")).lower() == "good"]
    if not good_rows:
        good_rows = rows[:1]

    bad_variants = generate_contrastive_bad_reps(good_rows, variants_per_rep=len(MISTAKE_PERTURBATIONS))
    print(f"\n=== Synthetic mistake variants (should predict BAD) ===\n")
    print(f"{'Mistake':<18}  {'Model':>6}  {'Conf':>6}  {'P(good)':>8}  {'P(bad)':>8}")
    print("-" * 56)
    preds = predictor.predict_rows(bad_variants)
    for row, pred in zip(bad_variants, preds):
        mistake = row.get("synthetic_mistake", "?")
        pg = pred["probabilities"].get("good", 0)
        pb = pred["probabilities"].get("bad", 0)
        mark = "OK" if pred["prediction"] == "bad" else "MISS"
        print(
            f"{mistake:<18}  {pred['prediction']:>6}  {pred['confidence']:>5.0%}  "
            f"{pg:>7.0%}  {pb:>7.0%}  [{mark}]"
        )

    caught = sum(1 for p in preds if p["prediction"] == "bad")
    print(f"\nCaught {caught}/{len(preds)} synthetic mistakes as bad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
