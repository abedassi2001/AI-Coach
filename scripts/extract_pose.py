#!/usr/bin/env python3
"""CLI: extract body pose keypoints from a video or frame directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.pose_pipeline import PoseExtractionPipeline
from src.utils.config import load_config, resolve_path
from src.utils.exercise_config import list_exercises, load_exercise_config


def parse_args() -> argparse.Namespace:
    exercises = list_exercises()
    parser = argparse.ArgumentParser(
        description="Extract exercise-agnostic pose keypoints (MediaPipe)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Input video path")
    source.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory of extracted frames (from extract_frames.py)",
    )
    parser.add_argument(
        "-e",
        "--exercise",
        type=str,
        default=None,
        choices=exercises if exercises else None,
        help=f"Exercise tag for output metadata (default from config). Options: {', '.join(exercises)}",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--backend", type=str, default=None, help="Pose backend (mediapipe)")
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["json", "csv", "both"],
        help="Keypoint output format",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Write skeleton overlay video",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    exercise = args.exercise or cfg.get("project", {}).get("exercise", "squat")

    try:
        ex_cfg = load_exercise_config(exercise)
        print(f"Exercise: {ex_cfg.get('display_name', exercise)} ({exercise})")
    except FileNotFoundError as e:
        print(f"Warning: {e}", file=sys.stderr)

    pipeline = PoseExtractionPipeline(backend=args.backend)
    try:
        if args.video is not None:
            if not args.video.exists():
                print(f"Error: video not found: {args.video}", file=sys.stderr)
                return 1
            result = pipeline.process_video(
                video_path=args.video,
                exercise=exercise,
                output_dir=args.output_dir,
                output_format=args.format,
                write_overlay_video=args.overlay,
                max_frames=args.max_frames,
            )
        else:
            if not args.frames_dir or not args.frames_dir.exists():
                print(f"Error: frames directory not found: {args.frames_dir}", file=sys.stderr)
                return 1
            result = pipeline.process_frames_dir(
                frames_dir=args.frames_dir,
                exercise=exercise,
                output_dir=args.output_dir,
                output_format=args.format,
                write_overlay_video=args.overlay,
                max_frames=args.max_frames,
            )
    finally:
        pipeline.close()

    seq = result.sequence
    print(f"Source:    {seq.source_id}")
    print(f"Backend:   {seq.backend}")
    print(f"Detected:  {len(seq.frames)} / {seq.metadata.get('total_frames', '?')} frames")
    print(f"Landmarks: {len(seq.landmark_names)} per frame")
    for fmt, path in result.saved_paths.items():
        print(f"Saved {fmt}: {path.resolve()}")
    if result.overlay_video_path:
        print(f"Overlay:   {result.overlay_video_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
