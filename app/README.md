# Iron Form Coach — Web UI

## Stack

**Streamlit** (Python) — not React/TypeScript. The UX patterns (modal, states, component layers) mirror a modern SPA architecture in Python modules.

## Run

```bash
pip install -r requirements.txt
python scripts/ensure_model.py   # retrains ML checkpoint if sklearn version mismatch
python scripts/run_app.py
# → http://localhost:8501
```

Use the project **virtual environment** (`.venv`) so Python, sklearn, and the saved model versions match.

If you see `_RemainderColsList` or pickle errors, run `python scripts/ensure_model.py` to retrain the optional ML model. **Rule-based scoring works without it.**

## User flow

```
Idle (upload area + explanation)
  → User uploads squat video
  → "Analyze my squat"
  → Loading: "Analyzing your squat form…"
  → Summary modal (auto-opens)
       • Overall score / performance label
       • Main issue + quick fix
       • Positive point
       • [View full analysis] [Analyze another video] [Watch replay]
  → Full analysis section
       • Coach narrative
       • Dimension score cards (Depth, Knees, Torso, …)
       • Rep-by-rep breakdown
       • Annotated video replay
```

## Architecture

| Module | Role |
|--------|------|
| `streamlit_app.py` | Main page, tabs, analysis state machine |
| `analysis_view.py` | Build summary/full views from `form_analysis.json` |
| `issue_copy.py` | Beginner-friendly issue & dimension copy |
| `ui_states.py` | Empty, loading, error, score ring/badge |
| `ui_score.py` | Dimension cards, rep cards, issue explanations |
| `ui_panels.py` | Summary modal + full analysis panel |
| `components.py` | Video dialog, load analysis JSON |
| `gym_theme.py` | Dark gym CSS theme |

## Data source

Reads `data/processed/analysis/<source_id>/form_analysis.json` — continuous 0–100 scores per dimension. Missing fields use safe fallbacks (computed best/worst dimension, generic copy).

## Scoring UI

| Score | Label |
|-------|--------|
| 90–100 | Excellent |
| 75–89 | Good |
| 60–74 | Needs Work |
| 40–59 | Poor Form |
| 0–39 | High Risk |

## Screenshots

_Add screenshots here after running the app:_
- Upload idle state
- Summary modal
- Full analysis panel

## Future improvements

- GPT-generated personalized coaching in modal
- Timeline video annotations (click rep → jump to frame)
- Rep-to-rep comparison chart
- Mobile recording flow (camera upload)
- User history dashboard
- Migrate to React/Next.js for production if needed
