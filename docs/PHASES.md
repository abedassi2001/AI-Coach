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

## Phase 3 — Pose extraction (next)

**Planned files:** `src/pose/mediapipe_estimator.py`, `src/pose/keypoint_schema.py`, `scripts/extract_pose.py`, `tests/test_pose.py`

**Commit message (suggested):**
```
feat(pose): add MediaPipe keypoint extraction and skeleton overlay
```
