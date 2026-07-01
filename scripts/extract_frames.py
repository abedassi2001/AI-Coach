#!/usr/bin/env python3
"""CLI: extract frames from an exercise video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python scripts/extract_frames.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.frame_extractor import FrameExtractor
from backend.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract resized frames from a gym exercise video."
    )
    parser.add_argument("video", type=Path, help="Path to input video (.mp4, .avi, ...)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for extracted frames (default: data/interim/<video_stem>)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="YAML config path (default: configs/default.yaml)",
    )
    parser.add_argument("--target-fps", type=float, default=None, help="Override target FPS")
    parser.add_argument("--max-width", type=int, default=None, help="Override max frame width")
    parser.add_argument("--max-height", type=int, default=None, help="Override max frame height")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N extracted frames (useful for quick previews)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Analyze only; do not write frame images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    video_cfg = cfg.get("video", {})

    target_fps = args.target_fps if args.target_fps is not None else video_cfg.get("target_fps", 30)
    max_width = args.max_width if args.max_width is not None else video_cfg.get("max_width", 1280)
    max_height = args.max_height if args.max_height is not None else video_cfg.get("max_height", 720)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = resolve_path("data/interim") / args.video.stem

    extractor = FrameExtractor(
        target_fps=target_fps,
        max_width=max_width,
        max_height=max_height,
    )
    result = extractor.extract(
        video_path=args.video,
        output_dir=output_dir,
        save_frames=not args.no_save,
        save_manifest=not args.no_save,
        max_frames=args.max_frames,
    )

    meta = result.metadata
    print(f"Video:     {meta.path}")
    print(f"Source:    {meta.width}x{meta.height} @ {meta.fps:.2f} fps, {meta.frame_count} frames")
    print(f"Duration:  {meta.duration_sec:.2f}s")
    print(f"Extracted: {len(result.frames)} frames (stride={result.sample_stride}, ~{result.effective_fps:.2f} fps)")
    if not args.no_save:
        print(f"Output:    {output_dir.resolve()}")
        print(f"Manifest:  {(output_dir / 'manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
