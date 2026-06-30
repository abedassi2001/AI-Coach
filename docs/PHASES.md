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

## Phase 4 — Feature engineering ✅

**Files changed:** `src/features/angles.py`, `src/features/normalization.py`, `src/features/feature_pipeline.py`, `scripts/compute_features.py`, `tests/test_features.py`

**Implemented:** Joint angle math, body scale (torso length), exercise-config-driven angles, derived features (knee avg/min/asymmetry), optional smoothing, CSV/JSON export.

**How to test:**
```bash
pytest tests/test_features.py -v
python scripts/compute_features.py data/processed/pose/sample_squat/keypoints.json
```

**Commit message:**
```
feat(features): compute joint angles and normalized movement features
```

---

## Phase 5 — Rep segmentation ✅

**Files changed:** `src/features/rep_segmentation.py`, `scripts/segment_reps.py`, `tests/test_reps.py`

**Implemented:** Valley detection on knee angle signal, rep boundaries, phase labels (standing/descending/bottom/ascending/finished), reps.json export.

**How to test:**
```bash
pytest tests/test_reps.py -v
python scripts/segment_reps.py data/processed/features/sample_squat/features.csv
```

**Commit message:**
```
feat(features): segment squat repetitions from knee angle time series
```

---

## Phase 6 — Rule-based form analyzer (next)

**Planned files:** `src/feedback/rules_engine.py` or `src/features/form_analyzer.py`, `scripts/analyze_form.py`, `tests/test_form_rules.py`

**Commit message (suggested):**
```
feat(feedback): add rule-based squat form mistake detection
```
