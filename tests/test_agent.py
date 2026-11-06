from pathlib import Path
from typing import Any

from mcp_agent_loop.agent import Agent, AgentConfig, must_contain
from mcp_agent_loop.models import Decision, History, ModelError, ScriptedModel
from mcp_agent_loop.trace import Tracer, read_trace

from conftest import make_client

READ = {"tool": "read_file", "args": {"path": "VERSION.txt"}}


class FakeClock:
    def __init__(self, tick: float) -> None:
        self.now, self.tick = 0.0, tick

    def __call__(self) -> float:
        self.now += self.tick
        return self.now


def run(workspace: Path, plan: list[dict[str, Any]], fail_first_calls: int = 0, **cfg: Any) -> tuple:
    client = make_client(workspace, fail_first_calls)
    sleeps: list[float] = []
    tracer = Tracer()
    agent = Agent(ScriptedModel(plan), client, AgentConfig(**cfg), tracer, sleep=sleeps.append)
    return agent.run("goal"), tracer, sleeps, client


def test_completes_with_answer(workspace: Path) -> None:
    result, *_ = run(workspace, [READ, {"finish": "v{last}"}])
    assert result.ok and result.answer == "v1.4.2" and result.steps == 2


def test_stops_at_max_steps(workspace: Path) -> None:
    result, tracer, *_ = run(workspace, [READ] * 10, max_steps=3)
    assert result.status == "max_steps" and result.steps == 3
    assert len(result.tool_calls) == 3


def test_stops_on_time_budget(workspace: Path) -> None:
    client = make_client(workspace)
    agent = Agent(ScriptedModel([READ] * 10), client, AgentConfig(max_seconds=3.5), clock=FakeClock(1.0))
    result = agent.run("goal")
    assert result.status == "time_budget" and result.steps < 10


def test_stops_on_token_budget(workspace: Path) -> None:
    result, *_ = run(workspace, [READ] * 10, max_tokens=12)
    assert result.status == "token_budget" and 0 < result.steps < 10


def test_retries_transient_errors_with_exponential_backoff(workspace: Path) -> None:
    result, tracer, sleeps, client = run(workspace, [READ, {"finish": "{last}"}], fail_first_calls=2,
                                         max_retries=3, backoff_base=0.1)
    assert result.ok and result.answer == "1.4.2"
    assert sleeps == [0.1, 0.2] and result.retries == 2
    assert client.server.call_count == 3


def test_gives_up_after_max_retries(workspace: Path) -> None:
    result, _, sleeps, _ = run(workspace, [READ, {"finish": "{last}"}], fail_first_calls=5, max_retries=1)
    assert sleeps == [0.05] and result.tool_errors == 1
    assert result.answer.startswith("Tool call failed")


def test_tool_error_result_is_not_retried(workspace: Path) -> None:
    plan = [{"tool": "read_file", "args": {"path": "missing.txt"}}, {"finish": "{last}"}]
    result, _, sleeps, client = run(workspace, plan)
    assert sleeps == [] and result.tool_errors == 1 and client.server.call_count == 1


def test_permission_denied_never_reaches_server(workspace: Path) -> None:
    plan = [{"tool": "run_tests", "args": {}}, {"finish": "{last}"}]
    result, tracer, _, client = run(workspace, plan, allowed_tools=frozenset({"read_file"}))
    assert client.server.call_count == 0
    assert result.permission_denials == 1 and "Permission denied" in result.answer
    assert [e["event"] for e in tracer.events].count("permission_denied") == 1


def test_allowlist_hides_tools_from_model(workspace: Path) -> None:
    _, tracer, *_ = run(workspace, [{"finish": "x"}], allowed_tools=frozenset({"read_file"}))
    assert tracer.events[0]["tools"] == ["read_file"]


def test_rejected_completion_feeds_back_and_continues(workspace: Path) -> None:
    plan = [{"finish": "no idea"}, READ, {"finish": "version {last}"}]
    result, tracer, *_ = run(workspace, plan, completion_check=must_contain("1.4.2"))
    assert result.ok and result.steps == 3
    assert any(e["event"] == "completion_rejected" for e in tracer.events)


def test_model_error_stops_run(workspace: Path) -> None:
    class Broken:
        name = "broken"

        def decide(self, goal: str, tools: list[dict[str, Any]], history: History) -> Decision:
            raise ModelError("boom")

    result = Agent(Broken(), make_client(workspace)).run("goal")
    assert result.status == "model_error" and result.steps == 1


def test_jsonl_trace_records_every_step(workspace: Path, tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    tracer = Tracer(path, run_id="r1")
    agent = Agent(ScriptedModel([READ, {"finish": "{last}"}]), make_client(workspace, 1), AgentConfig(),
                  tracer, sleep=lambda _s: None)
    agent.run("goal")
    tracer.close()
    events = read_trace(path)
    assert [e["event"] for e in events] == [
        "run_start", "decision", "tool_call", "retry", "tool_call", "tool_result", "decision", "run_end"]
    assert all(e["run_id"] == "r1" and "ts" in e for e in events)
    assert events[-1]["status"] == "completed" and events[-1]["retries"] == 1
