#!/usr/bin/env python3
"""CLI: segment squat repetitions from features.csv or features.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.rep_segmentation import RepSegmentationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment repetitions from per-frame features (Phase 5)."
    )
    parser.add_argument(
        "features",
        type=Path,
        help="Path to features.csv or features.json",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument(
        "--signal",
        type=str,
        default="knee_angle_min",
        help="Feature column used to find squat bottoms (default: knee_angle_min)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum rep duration in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.features.exists():
        print(f"Error: not found: {args.features}", file=sys.stderr)
        return 1

    pipeline = RepSegmentationPipeline(
        signal_column=args.signal,
        min_rep_duration_sec=args.min_duration,
    )
    result = pipeline.segment_from_features_file(
        args.features,
        output_dir=args.output_dir,
    )

    print(f"Source:   {result.source_id}")
    print(f"Exercise: {result.exercise}")
    print(f"Signal:   {result.signal_column}")
    print(f"Frames:   {result.frame_count}")
    print(f"Reps:     {len(result.repetitions)}")
    print(f"Output:   {(result.output_dir / 'reps.json').resolve()}")

    for rep in result.repetitions:
        print(
            f"\n  Rep {rep.rep_id}: frames {rep.start_frame}–{rep.end_frame} "
            f"({rep.start_time_sec:.2f}s–{rep.end_time_sec:.2f}s)"
        )
        print(f"    bottom @ frame {rep.bottom_frame}, knee_angle={rep.bottom_knee_angle}°")
        for phase in rep.phases:
            if phase.start_frame == phase.end_frame and phase.phase in ("standing", "finished"):
                print(f"    {phase.phase}: frame {phase.start_frame}")
            elif phase.start_frame != phase.end_frame:
                print(
                    f"    {phase.phase}: frames {phase.start_frame}–{phase.end_frame}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
