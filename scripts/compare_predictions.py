#!/usr/bin/env python3
"""Compare model predictions across multiple processed recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.rep_classifier import RepQualityPredictor
from src.training.rep_dataset import discover_processed_sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare predictions on all processed videos.")
    parser.add_argument(
        "source_ids",
        nargs="*",
        help="Source ids (default: all processed)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/checkpoints/baseline/form_classifier.joblib"),
    )
    args = parser.parse_args()

    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    if not model_path.exists():
        print("Train first: python scripts/train_classifier.py --demo", file=sys.stderr)
        return 1

    ids = args.source_ids or discover_processed_sources()
    predictor = RepQualityPredictor.load(str(model_path))

    print(f"\n{'Source':<22} {'Rep':>4} {'Model':>6} {'Conf':>6} {'Bottom°':>8}  Evaluation video")
    print("-" * 95)
    for sid in ids:
        preds = predictor.predict_source(sid)
        eval_video = PROJECT_ROOT / "data/processed/evaluation" / sid / f"{sid}_evaluation.mp4"
        for p in preds:
            from src.training.rep_dataset import build_rep_rows_for_source

            rows = build_rep_rows_for_source(sid)
            bottom = next(
                (r.get("rep_bottom_knee_angle") for r in rows if int(r["rep_id"]) == int(p["rep_id"])),
                0,
            )
            vid = str(eval_video.resolve()) if eval_video.exists() else "(run visualize_evaluation.py)"
            print(
                f"{sid:<22} {p['rep_id']:>4} {p['prediction']:>6} {p['confidence']:>5.0%} "
                f"{float(bottom):>8.1f}  {vid}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
