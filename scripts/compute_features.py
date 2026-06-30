#!/usr/bin/env python3
"""CLI: compute joint angles and movement features from pose keypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_pipeline import FeatureExtractionPipeline
from src.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute joint angles and features from keypoints.json (Phase 4)."
    )
    parser.add_argument(
        "keypoints",
        type=Path,
        nargs="?",
        default=None,
        help="Path to keypoints.json (default: data/processed/pose/<id>/keypoints.json)",
    )
    parser.add_argument("-e", "--exercise", type=str, default=None)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument(
        "--smooth",
        type=int,
        default=None,
        help="Moving-average window for angles (default from config)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    feat_cfg = cfg.get("features", {})

    if args.keypoints is None:
        print("Error: provide keypoints.json path", file=sys.stderr)
        print(
            "Example: python scripts/compute_features.py "
            "data/processed/pose/sample_squat/keypoints.json",
            file=sys.stderr,
        )
        return 1

    if not args.keypoints.exists():
        print(f"Error: file not found: {args.keypoints}", file=sys.stderr)
        return 1

    pipeline = FeatureExtractionPipeline(
        smoothing_window=args.smooth if args.smooth is not None else feat_cfg.get("smoothing_window", 1),
    )
    result = pipeline.extract_from_keypoints_file(
        keypoints_path=args.keypoints,
        output_dir=args.output_dir,
        exercise_id=args.exercise,
    )

    print(f"Source:   {result.source_id}")
    print(f"Exercise: {result.exercise}")
    print(f"Frames:   {len(result.frames)}")
    print(f"CSV:      {result.csv_path.resolve()}")
    print(f"JSON:     {result.json_path.resolve()}")

    if result.frames:
        sample = result.frames[0]
        print("\nSample frame 0 angles (degrees):")
        for name, val in sample.angles.items():
            print(f"  {name}: {val:.1f}")
        if sample.derived:
            print("Derived:")
            for name, val in sample.derived.items():
                print(f"  {name}: {val:.1f}")

        deepest = min(result.frames, key=lambda f: f.derived.get("knee_angle_min", 999))
        print(f"\nDeepest squat frame: {deepest.frame_index} @ {deepest.timestamp_sec:.2f}s")
        print(f"  knee_angle_min: {deepest.derived.get('knee_angle_min', float('nan')):.1f}°")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
