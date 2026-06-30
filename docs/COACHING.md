# Coaching feedback (Phase 9) — all free options

## Free options (no paid API key)

### 1. Built-in coach (default) — **recommended**

100% free, works offline, no signup:

```bash
python scripts/generate_coaching.py sample_squat
# same as --provider template
```

Uses your rule + ML analysis to produce priorities, per-rep cues, and practice drills.

### 2. Ollama (optional) — free local AI

If you want more natural language without paying:

1. Install [Ollama](https://ollama.com) (free)
2. In a terminal:
   ```bash
   ollama pull llama3.2
   ```
3. Run coaching:
   ```bash
   python scripts/generate_coaching.py sample_squat --provider ollama
   ```

If Ollama is not running, it falls back to the built-in coach automatically.

### 3. Auto

```bash
python scripts/generate_coaching.py sample_squat --provider auto
```

Tries Ollama if running, otherwise built-in coach.

## Paid option (optional)

OpenAI only if you already have a key:

```bash
set OPENAI_API_KEY=sk-...
python scripts/generate_coaching.py sample_squat --provider openai
```

## How it works

```
form_analysis.json + ML predictions → coach → coaching_report.txt
```

The AI never sees your video — only structured angles, mistakes, and scores.

## Output

`data/processed/coaching/<video_id>/coaching_report.txt`

## Config

`configs/feedback/openai.yaml` — default provider is `template` (free).
