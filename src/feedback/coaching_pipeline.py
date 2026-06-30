"""Orchestrate coaching report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.feedback.coach_context import build_coaching_context
from src.feedback.ollama_coach import generate_ollama_coaching, ollama_is_running
from src.feedback.openai_coach import generate_openai_coaching
from src.feedback.template_coach import generate_template_coaching
from src.utils.config import get_project_root


@dataclass
class CoachingResult:
    source_id: str
    exercise: str
    provider: str
    report: dict[str, Any]
    output_dir: Path
    json_path: Path
    text_path: Path


def load_feedback_config() -> dict[str, Any]:
    path = get_project_root() / "configs/feedback/openai.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _format_text_report(report: dict[str, Any], disclaimer: str) -> str:
    lines = [
        f"=== {report.get('exercise', 'exercise').title()} coaching: {report.get('source_id')} ===",
        f"(Provider: {report.get('provider', 'template')} — free, no paid API required)",
        "",
        report.get("overall_summary", ""),
        "",
        "TOP PRIORITIES",
        "--------------",
    ]
    for i, item in enumerate(report.get("action_plan", []), 1):
        lines.append(f"{i}. {item}")

    drills = report.get("practice_drills", [])
    if drills:
        lines.extend(["", "PRACTICE DRILLS", "---------------"])
        for i, d in enumerate(drills, 1):
            lines.append(f"{i}. {d}")

    lines.extend(["", "PER-REP NOTES", "-------------"])
    for rep in report.get("rep_feedback", []):
        lines.append(f"\nRep {rep.get('rep_id')}")
        lines.append(f"  {rep.get('summary', '')}")
        for cue in rep.get("cues", []):
            lines.append(f"  - {cue}")

    lines.extend(["", "---", disclaimer.strip()])
    return "\n".join(lines)


def _template_fallback(ctx_dict: dict, max_actions: int, reason: str) -> dict[str, Any]:
    report = generate_template_coaching(ctx_dict, max_actions=max_actions)
    report["fallback_reason"] = reason
    return report


def generate_coaching(
    source_id: str,
    provider: str | None = None,
    analysis_path: Path | None = None,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    allow_fallback: bool = True,
) -> CoachingResult:
    cfg = load_feedback_config()
    coach_cfg = cfg.get("coaching", {})
    openai_cfg = cfg.get("openai", {})
    ollama_cfg = cfg.get("ollama", {})
    chosen = provider or cfg.get("provider", "template")

    context = build_coaching_context(source_id, analysis_path=analysis_path, model_path=model_path)
    ctx_dict = context.to_dict()
    max_actions = int(coach_cfg.get("max_action_items", 5))
    disclaimer = str(cfg.get("disclaimer", ""))

    if chosen == "auto":
        base = str(ollama_cfg.get("base_url", "http://localhost:11434"))
        if ollama_is_running(base):
            chosen = "ollama"
        else:
            chosen = "template"

    if chosen == "template":
        report = generate_template_coaching(ctx_dict, max_actions=max_actions)
    elif chosen == "ollama":
        try:
            report = generate_ollama_coaching(
                ctx_dict,
                model=str(ollama_cfg.get("model", "llama3.2")),
                base_url=str(ollama_cfg.get("base_url", "http://localhost:11434")),
                temperature=float(ollama_cfg.get("temperature", 0.4)),
                timeout_sec=int(ollama_cfg.get("timeout_sec", 120)),
            )
        except (RuntimeError, json.JSONDecodeError, KeyError) as exc:
            if not allow_fallback:
                raise
            report = _template_fallback(ctx_dict, max_actions, str(exc))
    elif chosen == "openai":
        try:
            report = generate_openai_coaching(
                ctx_dict,
                model=str(openai_cfg.get("model", "gpt-4o-mini")),
                temperature=float(openai_cfg.get("temperature", 0.4)),
                max_tokens=int(openai_cfg.get("max_tokens", 1200)),
            )
        except RuntimeError as exc:
            if not allow_fallback:
                raise
            report = _template_fallback(ctx_dict, max_actions, str(exc))
    else:
        raise ValueError(f"Unknown provider: {chosen}. Use template, ollama, openai, or auto.")

    report["disclaimer"] = disclaimer
    report["input_context"] = ctx_dict

    root = get_project_root()
    out_root = output_dir or Path(cfg.get("paths", {}).get("output_dir", "data/processed/coaching"))
    if not out_root.is_absolute():
        out_root = root / out_root
    out_dir = out_root / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "coaching_report.json"
    text_path = out_dir / "coaching_report.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    text_path.write_text(_format_text_report(report, disclaimer), encoding="utf-8")

    return CoachingResult(
        source_id=source_id,
        exercise=report.get("exercise", context.exercise),
        provider=str(report.get("provider", chosen)),
        report=report,
        output_dir=out_dir,
        json_path=json_path,
        text_path=text_path,
    )
