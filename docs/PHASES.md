# Development phases — commit checklist

Each phase ends with: **files changed**, **what was implemented**, **how to test**, **commit message**.

---

## Phase 1 — Repository setup ✅

**Files changed:** Full tree under `backend/`, `configs/`, `data/`, `docs/`, `tests/`, root configs.

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

**Files changed:** `backend/utils/config.py`, `backend/data/video_loader.py`, `backend/data/frame_extractor.py`, `scripts/extract_frames.py`, `tests/conftest.py`, `tests/test_video.py`

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

**Files changed:** `backend/pose/*`, `backend/visualization/skeleton.py`, `backend/utils/exercise_config.py`, `configs/exercises/`, `scripts/extract_pose.py`, `tests/test_pose*.py`, `docs/EXERCISES.md`

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

**Files changed:** `backend/features/angles.py`, `backend/features/normalization.py`, `backend/features/feature_pipeline.py`, `scripts/compute_features.py`, `tests/test_features.py`

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

**Files changed:** `backend/features/rep_segmentation.py`, `scripts/segment_reps.py`, `tests/test_reps.py`

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

## Phase 6 — Rule-based form analyzer ✅

**Files changed:** `backend/feedback/form_analyzer.py`, `backend/feedback/templates.py`, `scripts/analyze_form.py`, `tests/test_form_rules.py`

**Implemented:** Depth, lean, asymmetry, instability, heel lift, valgus proxy rules; form score; templated feedback messages.

**How to test:**
```bash
pytest tests/test_form_rules.py -v
python scripts/analyze_form.py sample_squat
```

**Commit message:**
```
feat(feedback): add rule-based squat form analysis and coaching messages
```

---

## Phase 7 — ML baseline classifier ✅

**Files changed:** `backend/features/rep_features.py`, `backend/ml/*`, `backend/training/*`, `backend/inference/rep_classifier.py`, `configs/training/baseline.yaml`, `scripts/train_classifier.py`, `scripts/bootstrap_labels.py`, `scripts/predict_rep_quality.py`, `docs/MODELS.md`

**Implemented:** Exercise-scalable rep features, gradient boosting pipeline with one-hot exercise, group split by video, weak-label bootstrap, metrics + confusion matrix.

**How to test:**
```bash
pytest tests/test_baseline_model.py -v
python scripts/bootstrap_labels.py
python scripts/train_classifier.py --demo
python scripts/predict_rep_quality.py sample_squat
```

**Commit message:**
```
feat(models): add scalable sklearn baseline for multi-exercise rep form quality
```

---

## Phase 8 — Deep learning sequence model (next)

**Planned files:** `backend/ml/sequence_lstm.py`, `backend/training/train_sequence.py`, `scripts/train_sequence.py`

**Commit message (suggested):**
```
feat(models): add LSTM sequence classifier for rep-level form quality
```
