"""OpenAI-powered coaching from structured form analysis."""

from __future__ import annotations

import json
import os
from typing import Any

SYSTEM_PROMPT = """You are an encouraging strength-and-conditioning coach for recreational lifters.

You receive structured JSON from a computer-vision squat analyzer (pose landmarks, joint angles, rule flags, and an optional ML quality label). You do NOT see the video.

Rules:
- Base advice ONLY on the JSON facts provided. Do not invent injuries or diagnoses.
- Prioritize fixes by mistake severity (high > medium > low).
- Give concrete, actionable cues a beginner can try on the next set.
- Keep tone supportive, not harsh.
- Mention 1–2 things that are already working when overall quality is not terrible.
- Output valid JSON matching the schema exactly — no markdown fences.

Schema:
{
  "overall_summary": "2-4 sentences",
  "action_plan": ["top priority fix 1", "fix 2", ... up to 5 strings],
  "rep_feedback": [
    {
      "rep_id": 1,
      "summary": "one sentence for this rep",
      "focus_areas": ["mistake_id", ...],
      "cues": ["specific cue 1", "cue 2"]
    }
  ]
}
"""


def _get_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def generate_openai_coaching(
    context_dict: dict[str, Any],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    max_tokens: int = 1200,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Call OpenAI Chat Completions with structured analysis context."""
    key = api_key or _get_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export it or use --provider template for offline coaching."
        )

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai: pip install openai") from e

    client = OpenAI(api_key=key)
    user_content = (
        "Analyze this squat session and return coaching JSON.\n\n"
        f"{json.dumps(context_dict, indent=2)}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    parsed["provider"] = "openai"
    parsed["model"] = model
    parsed["source_id"] = context_dict.get("source_id")
    parsed["exercise"] = context_dict.get("exercise")
    return parsed
