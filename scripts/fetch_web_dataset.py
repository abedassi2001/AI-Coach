#!/usr/bin/env python3
"""
Download curated squat videos, run the pipeline, and build rep_labels.csv.

Uses configs/datasets/web_squats.yaml for URLs and human quality labels.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.rep_dataset import discover_processed_sources


def _run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def download_video(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  skip download (exists): {dest.name}")
        return
    import urllib.request

    print(f"  downloading {url} -> {dest.name}")
    urllib.request.urlretrieve(url, dest)


def process_source(source_id: str, video_path: Path, py: str) -> None:
    keypoints = PROJECT_ROOT / "data/processed/pose" / source_id / "keypoints.json"
    if not keypoints.exists():
        _run([py, "scripts/extract_pose.py", "--video", str(video_path), "--exercise", "squat"])
    features = PROJECT_ROOT / "data/processed/features" / source_id / "features.csv"
    if not features.exists():
        _run([py, "scripts/compute_features.py", str(keypoints)])
    reps = PROJECT_ROOT / "data/processed/reps" / source_id / "reps.json"
    if not reps.exists():
        _run([py, "scripts/segment_reps.py", str(features), "--signal", "squat_depth_angle"])
    analysis = PROJECT_ROOT / "data/processed/analysis" / source_id / "form_analysis.json"
    if not analysis.exists():
        _run([py, "scripts/analyze_form.py", source_id])


def build_labels(manifest: list[dict], holdout_ids: set[str]) -> pd.DataFrame:
    import json

    rows: list[dict] = []
    for entry in manifest:
        sid = entry["source_id"]
        if entry.get("holdout"):
            continue
        quality = entry["quality"]
        label = "good" if quality == "good" else "bad"
        reps_path = PROJECT_ROOT / "data/processed/reps" / sid / "reps.json"
        if not reps_path.exists():
            print(f"  warning: no reps for {sid}, skipping labels")
            continue
        with reps_path.open(encoding="utf-8") as f:
            data = json.load(f)
        for rep in data.get("repetitions", []):
            rows.append(
                {
                    "source_id": sid,
                    "exercise": data.get("exercise", "squat"),
                    "rep_id": int(rep["rep_id"]),
                    "label": label,
                    "label_source": "human",
                    "notes": entry.get("notes", ""),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch web squat dataset and build labels.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/web_squats.yaml"),
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=Path("data/raw/labels/rep_labels.csv"),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Render evaluation video per source")
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    videos = cfg.get("videos", [])
    py = sys.executable
    video_dir = PROJECT_ROOT / "data/raw/videos"

    for entry in videos:
        sid = entry["source_id"]
        dest = video_dir / f"{sid}.mp4"
        if not args.skip_download:
            url = entry.get("url")
            if url:
                download_video(url, dest)
            elif not dest.exists():
                alt = video_dir / f"{sid.replace('sample_', '')}.mp4"
                if sid == "sample_squat" and (video_dir / "sample_squat.mp4").exists():
                    pass
                else:
                    print(f"  warning: missing video for {sid}")
                    continue
        if not args.skip_process and dest.exists():
            print(f"\n=== Processing {sid} ({entry.get('quality')}) ===")
            process_source(sid, dest, py)

    holdout_ids = {e["source_id"] for e in videos if e.get("holdout")}
    labels_df = build_labels(videos, holdout_ids)
    if labels_df.empty:
        print("No labels built.", file=sys.stderr)
        return 1

    out = args.labels_out if args.labels_out.is_absolute() else PROJECT_ROOT / args.labels_out
    out.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(out, index=False)

    print(f"\n=== Labels ({len(labels_df)} reps) ===")
    print(labels_df.groupby(["label", "source_id"]).size().to_string())
    print(f"\nWrote {out.resolve()}")
    print(f"Processed sources: {discover_processed_sources()}")
    print(f"Holdout (not in labels): {sorted(holdout_ids)}")

    if args.visualize:
        for entry in videos:
            sid = entry["source_id"]
            if not (PROJECT_ROOT / "data/processed/reps" / sid / "reps.json").exists():
                continue
            _run([py, "scripts/visualize_evaluation.py", sid, "--no-train"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
