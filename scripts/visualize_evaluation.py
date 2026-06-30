#!/usr/bin/env python3
"""
Visual evaluation: annotated video + per-rep snapshots + knee-angle chart.

Runs the full pipeline on a recording (optional --refresh), loads rule analysis
and ML predictions, and writes reviewable artifacts under data/processed/evaluation/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_project_root
from src.visualization.evaluation_overlay import (
    build_evaluation_report,
    load_evaluation_artifacts,
    write_evaluation_video,
    write_knee_angle_chart,
    write_rep_snapshots,
)


def _run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def refresh_pipeline(source_id: str, video: Path, train_demo: bool, bootstrap_labels: bool) -> None:
    """Re-run pose → features → reps → rules → labels → train."""
    py = sys.executable
    _run(
        [
            py,
            "scripts/extract_pose.py",
            "--video",
            str(video),
            "--exercise",
            "squat",
        ]
    )
    keypoints = PROJECT_ROOT / "data/processed/pose" / source_id / "keypoints.json"
    _run([py, "scripts/compute_features.py", str(keypoints)])
    features = PROJECT_ROOT / "data/processed/features" / source_id / "features.csv"
    _run([py, "scripts/segment_reps.py", str(features)])
    _run([py, "scripts/analyze_form.py", source_id])
    if bootstrap_labels:
        _run([py, "scripts/bootstrap_labels.py"])
    model = PROJECT_ROOT / "models/checkpoints/baseline/form_classifier.joblib"
    if train_demo or not model.exists():
        _run([py, "scripts/train_classifier.py", "--demo"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual evaluation video and snapshots for one recording."
    )
    parser.add_argument("source_id", help="Recording id, e.g. sample_squat")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Override video path (default: data/raw/videos/<source_id>.mp4)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/checkpoints/baseline/form_classifier.joblib"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/processed/evaluation/<source_id>)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run full pipeline on the video before visualizing",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="With --refresh, skip training if model already exists",
    )
    parser.add_argument(
        "--bootstrap-labels",
        action="store_true",
        help="With --refresh, update rep_labels.csv from rules (skipped by default with --no-train)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit output video length (debug)",
    )
    parser.add_argument(
        "--coaching",
        action="store_true",
        help="Also generate AI coaching report (OpenAI or template fallback)",
    )
    parser.add_argument(
        "--coaching-provider",
        choices=["auto", "template", "ollama", "openai"],
        default="template",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    source_id = args.source_id

    video = args.video
    if video is None:
        video = root / "data/raw/videos" / f"{source_id}.mp4"
    if not video.is_absolute():
        video = root / video

    if args.refresh:
        if not video.exists():
            print(f"Error: video not found for refresh: {video}", file=sys.stderr)
            return 1
        refresh_pipeline(
            source_id,
            video,
            train_demo=not args.no_train,
            bootstrap_labels=args.bootstrap_labels,
        )

    model_path = args.model if args.model.is_absolute() else root / args.model
    if not model_path.exists():
        print(f"Warning: model not found at {model_path} — overlay will omit ML predictions.")
        print("Train with: python scripts/train_classifier.py --demo")
        model_path = None

    try:
        artifacts = load_evaluation_artifacts(
            source_id,
            video_path=video,
            model_path=model_path,
            root=root,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run with --refresh or process the video first.", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (root / "data/processed/evaluation" / source_id)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    video_out = out_dir / f"{source_id}_evaluation.mp4"
    write_evaluation_video(artifacts, video_out, max_frames=args.max_frames)

    snapshots = write_rep_snapshots(artifacts, out_dir / "snapshots")
    chart_path = write_knee_angle_chart(artifacts, out_dir / "knee_angle_chart.png")

    report = build_evaluation_report(artifacts)
    report["outputs"] = {
        "video": str(video_out.resolve()),
        "snapshots": [str(p.resolve()) for p in snapshots],
        "chart": str(chart_path.resolve()) if chart_path else None,
    }
    report_path = out_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Visual evaluation complete ===")
    print(f"Annotated video:  {video_out.resolve()}")
    print(f"Report:           {report_path.resolve()}")
    if chart_path:
        print(f"Knee angle chart: {chart_path.resolve()}")
    print(f"Rep snapshots:    {len(snapshots)} images in {(out_dir / 'snapshots').resolve()}")
    print("\nPer-rep summary:")
    for r in report["repetitions"]:
        print(
            f"  Rep {r['rep_id']}: frames {r['frames']}, bottom @ {r['bottom_frame']} "
            f"({r['bottom_knee_angle']:.0f}°) | rules={r['rule_quality']} "
            f"| model={r.get('model_prediction', 'n/a')}"
        )
        if r["mistakes"]:
            print(f"    mistakes: {', '.join(r['mistakes'])}")

    if args.coaching:
        from src.feedback.coaching_pipeline import generate_coaching

        coaching = generate_coaching(
            source_id,
            provider=args.coaching_provider,
            model_path=model_path,
            allow_fallback=args.coaching_provider in ("auto", "ollama", "openai"),
        )
        print("\n=== Coaching report ===")
        print(f"Provider: {coaching.provider}")
        print(f"Text:     {coaching.text_path.resolve()}")
        print()
        print(coaching.text_path.read_text(encoding="utf-8"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
