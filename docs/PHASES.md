# Development phases — commit checklist

Each phase ends with: **files changed**, **what was implemented**, **how to test**, **commit message**.

---

## Phase 1 — Repository setup ✅

**Files changed:** Full tree under `src/`, `configs/`, `data/`, `docs/`, `tests/`, root configs.

**Implemented:** Project skeleton, README, requirements, Docker scaffold, default YAML config, dataset docs, minimal smoke test.

**How to test:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest tests/test_package.py -v
```

**Commit message:**
```
chore: initialize AI Gym Form Coach project structure

Set up src package layout, configs, docs, Docker scaffold,
requirements, and gitignore for squat-form MVP pipeline.
```

---

## Phase 2 — Video processing ✅

**Files changed:** `src/utils/config.py`, `src/data/video_loader.py`, `src/data/frame_extractor.py`, `scripts/extract_frames.py`, `tests/conftest.py`, `tests/test_video.py`

**Implemented:** YAML config loader, OpenCV video metadata, FPS-aware frame sampling, aspect-ratio resize, frame export + JSON manifest, CLI script.

**How to test:**
```bash
pytest tests/test_video.py -v

# With your own squat video:
python scripts/extract_frames.py path/to/squat.mp4
python scripts/extract_frames.py path/to/squat.mp4 --max-frames 10
```

**Commit message:**
```
feat(data): add video loading and frame extraction pipeline
```

---

## Phase 3 — Pose extraction ✅

**Files changed:** `src/pose/*`, `src/visualization/skeleton.py`, `src/utils/exercise_config.py`, `configs/exercises/`, `scripts/extract_pose.py`, `tests/test_pose*.py`, `docs/EXERCISES.md`

**Implemented:** Exercise-agnostic pose schema, MediaPipe backend with factory pattern, pose pipeline (video or frames), JSON/CSV export, skeleton overlay, per-exercise YAML configs.

**How to test:**
```bash
pytest tests/test_pose_schema.py tests/test_pose.py -v

python scripts/extract_pose.py --video path/to/squat.mp4 --exercise squat --overlay
python scripts/extract_pose.py --frames-dir data/interim/<name> --exercise squat
```

**Commit message:**
```
feat(pose): add scalable MediaPipe pose extraction pipeline

Exercise-agnostic keypoint schema, backend factory, per-exercise configs,
skeleton overlay, and CLI for video or pre-extracted frames.
```

---

## Phase 4 — Feature engineering (next)

**Planned files:** `src/features/angles.py`, `src/features/normalization.py`, `scripts/compute_features.py`, `tests/test_features.py`

**Commit message (suggested):**
```
feat(features): compute joint angles and normalized movement features
```
