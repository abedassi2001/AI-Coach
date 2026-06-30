# Architecture

## Pipeline overview

```mermaid
flowchart LR
    A[Video upload] --> B[Frame extraction]
    B --> C[Pose estimation]
    C --> D[Normalize keypoints]
    D --> E[Joint angles & features]
    E --> F[Rep segmentation]
    F --> G[Phase classification]
    G --> H{Analyzer}
    H --> I[Rule-based]
    H --> J[ML classifier]
    H --> K[Sequence model]
    I --> L[Feedback generator]
    J --> L
    K --> L
    L --> M[User report + visuals]
```

## Module responsibilities

### `src/data`
Load videos with OpenCV, extract frames at target FPS, resize, and persist frames or metadata. Handles train/val splits and label files.

### `src/pose`
Wrap MediaPipe Pose (or YOLO Pose). Output per-frame keypoints (x, y, visibility) as JSON/CSV.

### `src/features`
Compute geometry: hip, knee, ankle, torso angles; shoulder–hip alignment; velocity and smoothness. Normalize by torso length for camera invariance.

### `src/models`
- Baseline: sklearn classifiers on aggregated rep features
- Deep: LSTM / Temporal CNN / Transformer on `(batch, seq_len, n_features)` tensors

### `src/training`
Training loops, loss functions, optimizers, early stopping, checkpointing, evaluation metrics.

### `src/inference`
Orchestrates the full pipeline on a single video path and returns structured results.

### `src/feedback`
Maps detected mistakes to templated coaching messages. LLM integration stays optional and separate.

### `src/visualization`
Skeleton overlays, angle plots, rep markers on timeline.

### `src/utils`
Config parsing (`configs/default.yaml`), logging, angle math, file I/O helpers.

## Tensor shapes (Phase 8 preview)

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input sequence | `(B, T, F)` | Batch, time steps (frames per rep), feature dim |
| LSTM hidden | `(B, H)` | Pooled representation |
| Output logits | `(B, C)` | C = mistake classes or quality score |

## Separation of concerns

- **Detection logic** (rules + models) lives in `src/features`, `src/models`, `src/inference`
- **Natural language** lives in `src/feedback` — swappable templates or LLM without changing core models
