#!/usr/bin/env python3
"""CLI: predict rep form quality with trained baseline model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.rep_classifier import RepQualityPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict form quality for one recording.")
    parser.add_argument("source_id", help="e.g. sample_squat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/checkpoints/baseline/form_classifier.joblib"),
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        print("Train first: python scripts/train_classifier.py --demo", file=sys.stderr)
        return 1

    predictor = RepQualityPredictor.load(str(args.model))
    results = predictor.predict_source(args.source_id)

    print(f"Source: {args.source_id}")
    for r in results:
        print(
            f"  Rep {r['rep_id']}: {r['prediction']} "
            f"(confidence {r['confidence']:.2f}) exercise={r['exercise']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
