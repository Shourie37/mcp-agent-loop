"""Evaluation harness: bounded tasks, executable acceptance checks, and config comparison."""

from __future__ import annotations

import dataclasses
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import Agent, AgentConfig, RunResult, must_contain, non_empty
from .client import InProcessClient
from .models import Model, make_model
from .server import ToolServer
from .tools import build_tools
from .trace import Tracer


@dataclass(frozen=True)
class EvalTask:
    id: str
    goal: str
    plan: list[dict[str, Any]]
    checks: list[dict[str, Any]]
    fail_first_calls: int = 0
    must_mention: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalTask:
        return cls(id=data["id"], goal=data["goal"], plan=data.get("plan", []), checks=data["checks"],
                   fail_first_calls=data.get("fail_first_calls", 0),
                   must_mention=tuple(data.get("must_mention", ())))


def load_tasks(path: Path) -> list[EvalTask]:
    return [EvalTask.from_dict(t) for t in json.loads(path.read_text(encoding="utf-8"))["tasks"]]


def load_configs(path: Path) -> dict[str, dict[str, Any]]:
    return dict(json.loads(path.read_text(encoding="utf-8"))["configs"])


def run_check(check: dict[str, Any], result: RunResult) -> str | None:
    """Return None if the check passes, else a short failure message."""
    kind = check["type"]
    if kind == "status":
        ok = result.status == check["equals"]
        return None if ok else f"status {result.status!r} != {check['equals']!r}"
    if kind == "answer_contains":
        ok = check["value"].lower() in result.answer.lower()
        return None if ok else f"answer lacks {check['value']!r}"
    if kind == "answer_regex":
        return None if re.search(check["pattern"], result.answer) else f"answer does not match /{check['pattern']}/"
    if kind == "tool_called":
        return None if check["name"] in result.tool_calls else f"tool {check['name']!r} was not called"
    if kind == "max_steps":
        return None if result.steps <= check["value"] else f"used {result.steps} steps > {check['value']}"
    if kind == "min_tool_errors":
        ok = result.tool_errors >= check["value"]
        return None if ok else f"tool_errors {result.tool_errors} < {check['value']}"
    raise ValueError(f"unknown check type: {kind}")


@dataclass
class TaskOutcome:
    task_id: str
    passed: bool
    status: str
    steps: int
    latency: float
    tool_errors: int
    retries: int
    failures: list[str] = field(default_factory=list)


@dataclass
class Report:
    config: str
    outcomes: list[TaskOutcome]

    @property
    def success_rate(self) -> float:
        return sum(o.passed for o in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    def summary(self) -> dict[str, Any]:
        lat = [o.latency for o in self.outcomes] or [0.0]
        return {
            "config": self.config,
            "tasks": len(self.outcomes),
            "success_rate": round(self.success_rate, 3),
            "mean_steps": round(statistics.fmean(o.steps for o in self.outcomes), 2) if self.outcomes else 0,
            "p50_latency_s": round(statistics.median(lat), 4),
            "max_latency_s": round(max(lat), 4),
            "tool_errors": sum(o.tool_errors for o in self.outcomes),
            "retries": sum(o.retries for o in self.outcomes),
        }


def run_task(task: EvalTask, config: AgentConfig, root: Path, model: Model | None = None,
             trace_dir: Path | None = None) -> TaskOutcome:
    server = ToolServer(build_tools(root), fail_first_calls=task.fail_first_calls)
    client = InProcessClient(server)
    check = must_contain(*task.must_mention) if task.must_mention else non_empty
    cfg = dataclasses.replace(config, completion_check=check)
    tracer = Tracer(trace_dir / f"{task.id}.jsonl" if trace_dir else None, run_id=task.id)
    try:
        agent = Agent(model or make_model("scripted", task.plan), client, cfg, tracer, sleep=lambda _s: None)
        result = agent.run(task.goal)
    finally:
        tracer.close()
        client.close()
    failures = [msg for c in task.checks if (msg := run_check(c, result))]
    return TaskOutcome(task.id, not failures, result.status, result.steps, result.elapsed,
                       result.tool_errors, result.retries, failures)


def run_eval(tasks: list[EvalTask], name: str, config: AgentConfig, root: Path, model_kind: str = "scripted",
             trace_dir: Path | None = None) -> Report:
    outcomes = []
    for task in tasks:
        model = None if model_kind == "scripted" else make_model(model_kind)
        outcomes.append(run_task(task, config, root, model, trace_dir / name if trace_dir else None))
    return Report(name, outcomes)


def format_report(report: Report) -> str:
    lines = [f"== {report.config} =="]
    for o in report.outcomes:
        mark = "PASS" if o.passed else "FAIL"
        detail = f"  ({'; '.join(o.failures)})" if o.failures else ""
        lines.append(f"  {mark}  {o.task_id:<22} status={o.status:<12} steps={o.steps} "
                     f"errors={o.tool_errors} retries={o.retries}{detail}")
    return "\n".join(lines)


def format_comparison(a: Report, b: Report) -> str:
    sa, sb = a.summary(), b.summary()
    keys = ["success_rate", "mean_steps", "p50_latency_s", "max_latency_s", "tool_errors", "retries"]
    lines = [f"{'metric':<16}{a.config:>12}{b.config:>12}{'delta':>10}"]
    for k in keys:
        delta = sb[k] - sa[k]
        lines.append(f"{k:<16}{sa[k]:>12}{sb[k]:>12}{delta:>+10.3g}")
    flips = [f"{x.task_id}: {'FAIL->PASS' if y.passed else 'PASS->FAIL'}"
             for x, y in zip(a.outcomes, b.outcomes) if x.passed != y.passed]
    if flips:
        lines.append("changed tasks: " + ", ".join(flips))
    return "\n".join(lines)
