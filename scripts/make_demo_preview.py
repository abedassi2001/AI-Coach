#!/usr/bin/env python3
"""Generate preview images and summary from a pipeline test run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from backend.pose.keypoint_schema import PoseSequence
from backend.visualization.skeleton import draw_skeleton_on_frame


def load_sequence(json_path: Path) -> PoseSequence:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    from backend.pose.keypoint_schema import Keypoint, PoseFrame

    frames = []
    for fr in data["frames"]:
        landmarks = {
            name: Keypoint(**vals) for name, vals in fr["landmarks"].items()
        }
        frames.append(
            PoseFrame(
                frame_index=fr["frame_index"],
                timestamp_sec=fr["timestamp_sec"],
                landmarks=landmarks,
                width=fr.get("width", 0),
                height=fr.get("height", 0),
            )
        )
    return PoseSequence(
        source_id=data["source_id"],
        exercise=data["exercise"],
        backend=data["backend"],
        landmark_names=data["landmark_names"],
        frames=frames,
        metadata=data.get("metadata", {}),
    )


def main() -> int:
    video = PROJECT_ROOT / "data" / "raw" / "videos" / "sample_squat.mp4"
    keypoints = PROJECT_ROOT / "data" / "processed" / "pose" / "sample_squat" / "keypoints.json"
    out_dir = PROJECT_ROOT / "data" / "interim" / "demo_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video.exists() or not keypoints.exists():
        print("Run extract_pose first.", file=sys.stderr)
        return 1

    seq = load_sequence(keypoints)
    pose_by_idx = {f.frame_index: f for f in seq.frames}

    cap = cv2.VideoCapture(str(video))
    picks = [0, 8, 16, 24]
    saved = []
    idx = 0
    while cap.isOpened() and len(saved) < len(picks):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in picks:
            pose = pose_by_idx.get(idx)
            preview = draw_skeleton_on_frame(frame, pose)
            out_path = out_dir / f"preview_frame_{idx:04d}.jpg"
            cv2.imwrite(str(out_path), preview)
            saved.append((idx, out_path, pose))
        idx += 1
        if idx > max(picks):
            break
    cap.release()

    summary = {
        "video": str(video),
        "exercise": seq.exercise,
        "detected_frames": len(seq.frames),
        "total_frames_processed": seq.metadata.get("total_frames"),
        "preview_images": [str(p) for _, p, _ in saved],
        "sample_landmarks_frame_0": (
            {k: v.to_dict() for k, v in saved[0][2].landmarks.items()}
            if saved and saved[0][2]
            else None
        ),
    }
    summary_path = out_dir / "demo_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
