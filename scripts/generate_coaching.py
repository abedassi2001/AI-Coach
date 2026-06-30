#!/usr/bin/env python3
"""Generate personalized squat coaching from analysis + ML (OpenAI or templates)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feedback.coaching_pipeline import generate_coaching
from src.feedback.form_analyzer import SquatFormAnalyzer
from src.utils.config import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate actionable coaching feedback (OpenAI or template fallback)."
    )
    parser.add_argument("source_id", help="Recording id, e.g. sample_squat")
    parser.add_argument(
        "--provider",
        choices=["auto", "template", "ollama", "openai"],
        default="template",
        help="template=free built-in (default), ollama=free local AI, openai=paid API",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/checkpoints/baseline/form_classifier.joblib"),
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument(
        "--refresh-analysis",
        action="store_true",
        help="Re-run rule-based analyze_form if missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sid = args.source_id
    analysis = resolve_path(f"data/processed/analysis/{sid}/form_analysis.json")

    if args.refresh_analysis or not analysis.exists():
        features = resolve_path(f"data/processed/features/{sid}/features.csv")
        reps = resolve_path(f"data/processed/reps/{sid}/reps.json")
        keypoints = resolve_path(f"data/processed/pose/{sid}/keypoints.json")
        if not features.exists() or not reps.exists():
            print("Run pose → features → reps pipeline first.", file=sys.stderr)
            return 1
        kp = keypoints if keypoints.exists() else None
        SquatFormAnalyzer().analyze(features, reps, kp)

    allow_fallback = args.provider in ("auto", "ollama", "openai")

    try:
        result = generate_coaching(
            sid,
            provider=args.provider if args.provider != "auto" else "auto",
            model_path=args.model,
            output_dir=args.output_dir,
            allow_fallback=allow_fallback,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Provider:  {result.provider}")
    print(f"JSON:      {result.json_path.resolve()}")
    print(f"Text:      {result.text_path.resolve()}")
    print()
    print(result.text_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
