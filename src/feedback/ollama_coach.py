"""Free local LLM coaching via Ollama (https://ollama.com)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

SYSTEM_PROMPT = """You are a friendly gym coach. You receive JSON squat analysis from a computer vision app.
Give practical, encouraging advice. Do not diagnose injuries. Base advice ONLY on the JSON.
Return valid JSON only:
{
  "overall_summary": "2-4 sentences",
  "action_plan": ["fix 1", "fix 2", ... up to 5],
  "practice_drills": ["optional drill 1", ...],
  "rep_feedback": [
    {"rep_id": 1, "summary": "one sentence", "focus_areas": ["mistake_id"], "cues": ["cue 1"]}
  ]
}
"""


def ollama_is_running(base_url: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def generate_ollama_coaching(
    context_dict: dict[str, Any],
    *,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.4,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Call a free local Ollama model — no API key, runs on your PC."""
    if not ollama_is_running(base_url):
        raise RuntimeError(
            "Ollama is not running. Install from https://ollama.com then run:\n"
            "  ollama pull llama3.2\n"
            "  ollama serve"
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Coaching JSON for this squat session:\n"
                + json.dumps(context_dict, indent=2),
            },
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }

    url = f"{base_url.rstrip('/')}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    raw = body.get("message", {}).get("content", "{}")
    parsed = json.loads(raw)
    parsed["provider"] = "ollama"
    parsed["model"] = model
    parsed["source_id"] = context_dict.get("source_id")
    parsed["exercise"] = context_dict.get("exercise")
    return parsed
