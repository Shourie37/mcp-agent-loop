"""Command line: serve | demo | eval | workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agent import Agent, AgentConfig, RunResult
from .client import StdioClient
from .evals import format_comparison, format_report, load_configs, load_tasks, run_eval
from .graph import END, Graph, State, auto_approve, console_approver, human_review
from .models import Model, ModelError, make_model
from .server import ToolServer, serve_stdio
from .tools import build_tools
from .trace import Tracer

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT / "examples" / "workspace"

DEMO_GOAL = "Report project health: the version, open TODOs, and whether the sample tests pass."
DEMO_PLAN: list[dict[str, Any]] = [
    {"thought": "Orient: see what is in the workspace.", "tool": "list_dir", "args": {"path": "."}},
    {"thought": "Read the version file.", "tool": "read_file", "args": {"path": "VERSION.txt"}},
    {"thought": "Try a shell grep for TODOs.", "tool": "run_shell", "args": {"cmd": "grep -r TODO ."}},
    {"thought": "Shell is not allowed; use the search tool.", "tool": "search_text", "args": {"query": "TODO"}},
    {"thought": "Run the sample test suite.", "tool": "run_tests", "args": {"path": "sample_repo"}},
    {"thought": "Enough evidence; summarize.",
     "finish": "version: {obs[1]}\nopen TODOs:\n{obs[3]}\ntests:\n{obs[4]}"},
]


def _model(kind: str) -> Model:
    return make_model(kind, DEMO_PLAN)


def _format_event(e: dict[str, Any]) -> str:
    kind, step = e["event"], e["step"]
    if kind == "run_start":
        return f"[start] model={e['model']} tools={','.join(e['tools'])}"
    if kind == "decision":
        action = f"call {e['tool']}({json.dumps(e['arguments'])})" if e["tool"] else "finish"
        return f"[{step}] think: {e['thought'] or '-'}  ->  {action}"
    if kind == "retry":
        return f"[{step}]   retry #{e['attempt'] + 1} in {e['delay']:.2f}s ({e['error']})"
    if kind == "permission_denied":
        return f"[{step}]   DENIED: {e['tool']} is not in the allowlist"
    if kind == "tool_result":
        text = (e.get("preview") or e.get("error") or "").replace("\n", " | ")
        return f"[{step}]   {'error' if e['is_error'] else 'ok'}: {text[:90]}"
    if kind == "completion_rejected":
        return f"[{step}]   finish rejected: {e['reason']}"
    if kind == "run_end":
        return (f"[end] status={e['status']} steps={step} tokens~{e['tokens']} "
                f"tool_errors={e['tool_errors']} retries={e['retries']} elapsed={e['elapsed']}s")
    return ""


def run_demo(model_kind: str, root: Path, trace_path: Path, fail_first_calls: int) -> RunResult:
    trace_path.unlink(missing_ok=True)
    tracer = Tracer(trace_path, run_id="demo")
    client = StdioClient(root, fail_first_calls=fail_first_calls)
    config = AgentConfig(max_steps=8, max_seconds=60, allowed_tools=frozenset(
        {"read_file", "list_dir", "search_text", "run_tests"}))
    try:
        result = Agent(_model(model_kind), client, config, tracer).run(DEMO_GOAL)
    finally:
        client.close()
        tracer.close()
    for line in filter(None, map(_format_event, tracer.events)):
        print(line)
    print("\nAnswer:\n" + result.answer)
    print(f"\nTrace: {trace_path}")
    return result


def build_workflow(model_kind: str, root: Path, auto: bool) -> Graph:
    def collect(state: State) -> State:
        state["attempts"] = state.get("attempts", 0) + 1
        client = StdioClient(root)
        try:
            res = Agent(_model(model_kind), client, AgentConfig(max_steps=8)).run(DEMO_GOAL)
        finally:
            client.close()
        return {"status": res.status, "summary": res.answer}

    def after_collect(state: State) -> str:
        if state["status"] == "completed":
            return "review"
        return "collect" if state["attempts"] < 2 else "escalate"

    graph = Graph()
    graph.add_node("collect", collect, retries=1)
    graph.add_node("review", human_review(auto_approve if auto else console_approver))
    graph.add_node("publish", lambda s: {"published": True})
    graph.add_node("escalate", lambda s: {"published": False})
    graph.add_branch("collect", after_collect)
    graph.add_branch("review", lambda s: "publish" if s["approved"] else "escalate")
    graph.add_edge("publish", END).add_edge("escalate", END)
    return graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcp_agent_loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the MCP tool server on stdio")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--fail-first-calls", type=int, default=0)

    p = sub.add_parser("demo", help="run one agent task end to end over stdio")
    p.add_argument("--model", choices=["scripted", "claude"], default="scripted")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--trace", type=Path, default=PROJECT / "traces" / "demo.jsonl")
    p.add_argument("--fail-first-calls", type=int, default=1, help="inject transient failures")

    p = sub.add_parser("eval", help="run the eval set and compare two configs")
    p.add_argument("--tasks", type=Path, default=PROJECT / "evals" / "tasks.json")
    p.add_argument("--configs", type=Path, default=PROJECT / "evals" / "configs.json")
    p.add_argument("--a", default="baseline")
    p.add_argument("--b", default="hardened")
    p.add_argument("--model", choices=["scripted", "claude"], default="scripted")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--trace-dir", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None, help="write both summaries as JSON")

    p = sub.add_parser("workflow", help="run the review workflow graph")
    p.add_argument("--model", choices=["scripted", "claude"], default="scripted")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--auto-approve", action="store_true")

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "serve":
        serve_stdio(ToolServer(build_tools(args.root), args.fail_first_calls), sys.stdin, sys.stdout)
        return 0
    if args.cmd == "demo":
        return 0 if run_demo(args.model, args.root, args.trace, args.fail_first_calls).ok else 1
    if args.cmd == "eval":
        tasks, configs = load_tasks(args.tasks), load_configs(args.configs)
        reports = [run_eval(tasks, name, AgentConfig.from_dict(configs[name]), args.root, args.model, args.trace_dir)
                   for name in (args.a, args.b)]
        for r in reports:
            print(format_report(r) + "\n")
        print(format_comparison(*reports))
        if args.json:
            args.json.write_text(json.dumps([r.summary() for r in reports], indent=2) + "\n")
        return 0
    result = build_workflow(args.model, args.root, args.auto_approve).run("collect")
    print(f"path: {' -> '.join(result.path)}  status={result.status}")
    print(f"published={result.state.get('published')} approved={result.state.get('approved')}")
    return 0 if result.state.get("published") else 1
