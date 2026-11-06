from pathlib import Path

import pytest

from mcp_agent_loop.agent import AgentConfig, RunResult
from mcp_agent_loop.evals import (EvalTask, format_comparison, load_configs, load_tasks, run_check, run_eval,
                                  run_task)

from conftest import PROJECT

TASKS = PROJECT / "evals" / "tasks.json"
CONFIGS = PROJECT / "evals" / "configs.json"


def result(**kw: object) -> RunResult:
    base = dict(status="completed", answer="Version 1.4.2", steps=2, tokens=10, elapsed=0.1,
                tool_calls=["read_file"], tool_errors=0)
    return RunResult(**{**base, **kw})  # type: ignore[arg-type]


@pytest.mark.parametrize("check, passes", [
    ({"type": "status", "equals": "completed"}, True),
    ({"type": "status", "equals": "max_steps"}, False),
    ({"type": "answer_contains", "value": "version 1.4"}, True),
    ({"type": "answer_regex", "pattern": r"\d+\.\d+\.\d+"}, True),
    ({"type": "tool_called", "name": "run_tests"}, False),
    ({"type": "max_steps", "value": 1}, False),
    ({"type": "min_tool_errors", "value": 1}, False),
])
def test_run_check(check: dict, passes: bool) -> None:
    assert (run_check(check, result()) is None) is passes


def test_unknown_check_type_is_an_error() -> None:
    with pytest.raises(ValueError):
        run_check({"type": "vibes"}, result())


def test_task_set_is_bounded() -> None:
    tasks = load_tasks(TASKS)
    assert 5 <= len(tasks) <= 10
    assert len({t.id for t in tasks}) == len(tasks)
    assert all(t.checks for t in tasks)


def test_compare_configs(workspace: Path) -> None:
    tasks, configs = load_tasks(TASKS), load_configs(CONFIGS)
    base = run_eval(tasks, "baseline", AgentConfig.from_dict(configs["baseline"]), workspace)
    hard = run_eval(tasks, "hardened", AgentConfig.from_dict(configs["hardened"]), workspace)
    assert hard.success_rate == 1.0
    assert base.success_rate == pytest.approx(5 / 8)
    failed = {o.task_id for o in base.outcomes if not o.passed}
    assert failed == {"run-tests", "flaky-read", "multi-file-summary"}
    table = format_comparison(base, hard)
    assert "success_rate" in table and "flaky-read: FAIL->PASS" in table
    summary = hard.summary()
    assert summary["tasks"] == 8 and summary["retries"] == 1


def test_failed_task_reports_reasons(workspace: Path) -> None:
    task = EvalTask("t", "g", [{"finish": "nothing"}], [{"type": "answer_contains", "value": "1.4.2"}])
    outcome = run_task(task, AgentConfig(), workspace)
    assert not outcome.passed and outcome.failures == ["answer lacks '1.4.2'"]
