"""Run the question set against a model and score the agent's behaviour.

Scoring is rule based, so results are repeatable and cheap. It checks what the agent did (which tools,
which sources) as well as what it said. A right answer with no evidence trail fails.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .agent import Agent, Result
from .guardrails import NO_DATA


def score(case: dict, result: Result) -> dict:
    answer = result.answer.lower()
    tools = {call["name"] for call in result.tool_calls}
    checks = {}
    if "tools_any" in case:
        checks["tools"] = bool(tools & set(case["tools_any"]))
    if "cite_any" in case:
        checks["cite_any"] = any(c.startswith(p) for c in result.citations for p in case["cite_any"])
    if "cite_all" in case:
        checks["cite_all"] = all(any(c.startswith(p) for c in result.citations) for p in case["cite_all"])
    if "contains_all" in case:
        checks["contains_all"] = all(s.lower() in answer for s in case["contains_all"])
    if "contains_any" in case:
        checks["contains_any"] = any(s.lower() in answer for s in case["contains_any"])
    if "excludes" in case:
        checks["excludes"] = not any(s.lower() in answer for s in case["excludes"])
    if case.get("no_data"):
        checks["no_data"] = NO_DATA.lower() in answer and not result.citations
    checks["grounded"] = not result.problems
    valid = [c for c in result.citations if c in result.retrieved]
    return {
        "id": case["id"],
        "passed": all(checks.values()),
        "checks": checks,
        "citation_precision": round(len(valid) / len(result.citations), 3) if result.citations else None,
        "steps": result.steps,
        "tool_calls": len(result.tool_calls),
        "seconds": result.seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "flags": result.flags,
        "run_id": result.run_id,
    }


def summarise(model: str, scores: list[dict]) -> dict:
    precision = [s["citation_precision"] for s in scores if s["citation_precision"] is not None]
    checks = [ok for s in scores for ok in s["checks"].values()]
    return {
        "model": model,
        "questions": len(scores),
        "passed": sum(s["passed"] for s in scores),
        "check_pass_rate": round(sum(checks) / len(checks), 3),
        "citation_precision": round(sum(precision) / len(precision), 3) if precision else None,
        "mean_seconds": round(sum(s["seconds"] for s in scores) / len(scores), 1),
        "mean_tool_calls": round(sum(s["tool_calls"] for s in scores) / len(scores), 1),
        "total_tokens": sum(s["input_tokens"] + s["output_tokens"] for s in scores),
    }


def run_eval(
    agent: Agent, questions: Path, out_dir: Path | None = None, only: list[str] | None = None, echo=print
) -> dict:
    cases = yaml.safe_load(questions.read_text())
    if only:
        cases = [c for c in cases if c["id"] in only]
    scores, answers = [], []
    for case in cases:
        result = agent.run(case["question"])
        scored = score(case, result)
        scores.append(scored)
        answers.append(asdict(result))
        failed = [name for name, ok in scored["checks"].items() if not ok]
        echo(
            f"{'PASS' if scored['passed'] else 'FAIL'}  {case['id']:<22} {scored['seconds']:>6.1f}s"
            + (f"  failed: {', '.join(failed)}" if failed else "")
        )
    report = {"summary": summarise(agent.model.name, scores), "scores": scores, "answers": answers}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        name = agent.model.name.replace(":", "-").replace("/", "-")
        (out_dir / f"{stamp}-{name}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def comparison_table(reports: list[dict]) -> str:
    rows = [
        "| Model | Passed | Checks passed | Citation precision | Mean time | Mean tool calls | Tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        s = r["summary"]
        rows.append(
            f"| {s['model']} | {s['passed']}/{s['questions']} | {s['check_pass_rate']:.0%} | "
            f"{s['citation_precision'] if s['citation_precision'] is not None else 'n/a'} | "
            f"{s['mean_seconds']}s | {s['mean_tool_calls']} | {s['total_tokens']:,} |"
        )
    return "\n".join(rows)
