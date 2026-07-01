# Models — saved artifacts

Gitignored runtime directory for trained weights and pose assets.

```
models/
├── checkpoints/          # sklearn / PyTorch checkpoints
│   └── baseline/
│       └── form_classifier.joblib
├── exports/              # ONNX / TorchScript (future)
└── pose/                 # MediaPipe .task files
    └── pose_landmarker_lite.task
```

## Train or refresh the baseline

```bash
python scripts/train_classifier.py --demo
python scripts/ensure_model.py    # auto-retrain if sklearn version mismatch
```

## Code vs artifacts

| Path | Purpose |
|------|---------|
| `backend/ml/` | Python classifier **code** |
| `models/checkpoints/` | Serialized **weights** on disk |

Configs reference artifact paths in `configs/default.yaml` and `configs/training/baseline.yaml`.
