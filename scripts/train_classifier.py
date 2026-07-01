#!/usr/bin/env python3
"""CLI: train scalable baseline form classifier (Phase 7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.train_baseline import train_baseline_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train exercise-aware baseline classifier on rep-level features."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Augment with synthetic reps if too few labels (development only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = train_baseline_classifier(demo_augment=args.demo)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Model saved:  {result.model_path.resolve()}")
    print(f"Report:       {result.report_path.resolve()}")
    print(f"Train / val:  {result.n_train} / {result.n_val} reps")
    print(f"Features:     {result.feature_count} numeric + exercise (one-hot)")
    print(f"Accuracy:     {result.metrics['accuracy']:.3f}")
    print(f"F1:           {result.metrics['f1']:.3f}")
    print(f"Exercises:    {result.metrics.get('exercises', [])}")
    print("\nConfusion matrix:")
    for row in result.metrics["confusion_matrix"]:
        print(" ", row)
    if args.demo:
        print("\nNote: --demo used synthetic augmentation. Replace with real labels for production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
