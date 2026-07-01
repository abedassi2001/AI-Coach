#!/usr/bin/env python3
"""CLI: rule-based squat form analysis (Phase 6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feedback.form_analyzer import SquatFormAnalyzer
from src.utils.config import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze squat form using rule-based checks.")
    parser.add_argument("source_id", type=str, help="Recording id (e.g. sample_squat)")
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Path to features.csv (default: data/processed/features/<id>/features.csv)",
    )
    parser.add_argument(
        "--reps",
        type=Path,
        default=None,
        help="Path to reps.json (default: data/processed/reps/<id>/reps.json)",
    )
    parser.add_argument(
        "--keypoints",
        type=Path,
        default=None,
        help="Path to keypoints.json for heel/valgus checks",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sid = args.source_id
    features = args.features or resolve_path(f"data/processed/features/{sid}/features.csv")
    reps = args.reps or resolve_path(f"data/processed/reps/{sid}/reps.json")
    keypoints = args.keypoints or resolve_path(f"data/processed/pose/{sid}/keypoints.json")

    if not features.exists():
        print(f"Error: features not found: {features}", file=sys.stderr)
        return 1
    if not reps.exists():
        print(f"Error: reps not found: {reps}", file=sys.stderr)
        return 1

    kp = keypoints if keypoints.exists() else None
    analyzer = SquatFormAnalyzer()
    result = analyzer.analyze(
        features_path=features,
        reps_path=reps,
        keypoints_path=kp,
        output_dir=args.output_dir,
    )

    print(f"Source:         {result.source_id}")
    print(f"Exercise:       {result.exercise}")
    print(f"Overall score:  {result.overall_score}/100 ({result.overall_quality})")
    print(f"Output:         {(result.output_dir / 'form_analysis.json').resolve()}")

    for rep in result.rep_analyses:
        print(f"\n--- Rep {rep.rep_id} | overall {rep.overall_score} | {rep.quality} ---")
        if rep.scores:
            print(
                "  Scores: "
                + ", ".join(f"{k}={v}" for k, v in rep.scores.items() if k != "overall_score")
            )
        if rep.feedback:
            print(f"  Summary: {rep.feedback[0]}")
        if not rep.mistakes:
            print("  No mistake flags.")
        for m in rep.mistakes:
            print(f"  [{m.severity}] {m.mistake_id}: {m.message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
