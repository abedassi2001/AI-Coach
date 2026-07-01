# Backend — CV & ML pipeline

Python package for video analysis: pose estimation, feature engineering, rep segmentation, rule-based scoring, optional ML classifier, and coaching.

## Layout

```
backend/
├── data/           # Video I/O, frame extraction
├── pose/           # MediaPipe pose estimation
├── features/       # Angles, normalization, rep segmentation
├── feedback/       # Form rules, continuous scoring, coaching
├── inference/      # End-to-end video pipeline
├── ml/             # Classifier architectures (sklearn baseline)
├── training/       # Dataset builders, training loops, metrics
├── visualization/  # Skeleton overlay, evaluation videos
└── utils/          # Config, paths, web video helpers
```

## Import convention

```python
from backend.inference.video_pipeline import run_full_pipeline
from backend.feedback.form_analyzer import SquatFormAnalyzer
from backend.ml.baseline_classifier import BaselineFormClassifier
```

## Entry points

CLI scripts live in `scripts/` at the repo root and call into this package.

## Configuration

YAML configs in `configs/` — project root is resolved via `backend.utils.config.get_project_root()`.

## Saved weights

Trained checkpoints are **not** in this folder. See `models/` at the repo root.
