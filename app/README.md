# Legacy `app/` path

The Streamlit UI was moved to **`frontend/`** as part of the backend / frontend / models layout.

| Old | New |
|-----|-----|
| `app/streamlit_app.py` | `frontend/streamlit_app.py` |
| `from app.*` | `from frontend.*` |
| `src/` | `backend/` |

`app/streamlit_app.py` remains as a **compatibility shim** so older run commands still work.

**Run the demo:**

```bash
python scripts/run_app.py
```
