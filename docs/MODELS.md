# ML model design — scalable across exercises

## Design principles

1. **Pose is exercise-agnostic** (Phase 3) — same MediaPipe landmarks for all movements.
2. **Frame features are config-driven** (Phase 4) — `configs/exercises/<id>.yaml` defines angles.
3. **Rep features auto-aggregate** (Phase 7) — min/max/mean/std/at_bottom for every numeric column.
4. **`exercise` is a categorical input** — one-hot encoded; add deadlift without new model code.
5. **Split by `source_id`** — reps from the same video never leak between train and validation.

## Model stack

| Stage | Model | Input | Output |
|-------|--------|-------|--------|
| Phase 7 | Gradient boosting (sklearn) | Rep feature vector + exercise | good / bad |
| Phase 8 | LSTM / Transformer (PyTorch) | Sequence `(T, F)` per rep | good / bad + mistakes |
| Phase 9 | OpenAI | Structured JSON | Coaching text |

Phase 7 and 8 share the same **rep feature schema** — Phase 8 adds temporal modeling.

## Adding a new exercise

1. Add `configs/exercises/deadlift.yaml` with angle definitions.
2. Run pose → features → reps pipeline with `--exercise deadlift`.
3. Label reps in `data/raw/labels/rep_labels.csv`.
4. Retrain — `exercise` column lets one model learn all movements.

## Labels

**Human labels (best):** `data/raw/labels/rep_labels.csv`

```csv
source_id,exercise,rep_id,label
my_video,deadlift,1,good
```

**Weak labels (bootstrap):** `python scripts/bootstrap_labels.py` from Phase 6 rules.

**Synthetic (dev only):** `python scripts/train_classifier.py --demo`

## Training

```bash
python scripts/bootstrap_labels.py
python scripts/train_classifier.py --demo   # development
python scripts/train_classifier.py          # production (8+ real labels)
python scripts/predict_rep_quality.py sample_squat
```

Checkpoints: `models/checkpoints/baseline/form_classifier.joblib`
