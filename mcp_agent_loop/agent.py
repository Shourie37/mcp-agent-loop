"""The agent loop: plan -> call tool -> observe -> decide, under explicit limits."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .client import ToolClient, TransportError
from .models import Decision, History, Model, ModelError, Observation, ToolCall, estimate_tokens
from .trace import Tracer

# A completion check returns None when the answer is acceptable, else the reason it is not.
CompletionCheck = Callable[[str], "str | None"]


def non_empty(answer: str) -> str | None:
    return None if answer.strip() else "answer is empty"


def must_contain(*needles: str) -> CompletionCheck:
    def check(answer: str) -> str | None:
        missing = [n for n in needles if n.lower() not in answer.lower()]
        return f"answer must mention: {', '.join(missing)}" if missing else None
    return check


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    max_seconds: float = 60.0
    max_tokens: int = 50_000
    max_retries: int = 2
    backoff_base: float = 0.05
    allowed_tools: frozenset[str] | None = None  # None allows every tool the server lists.
    completion_check: CompletionCheck = non_empty

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        data = dict(data)
        if data.get("allowed_tools") is not None:
            data["allowed_tools"] = frozenset(data["allowed_tools"])
        return cls(**data)


@dataclass
class RunResult:
    status: str  # completed | max_steps | time_budget | token_budget | model_error
    answer: str
    steps: int
    tokens: int
    elapsed: float
    tool_calls: list[str] = field(default_factory=list)
    tool_errors: int = 0
    retries: int = 0
    permission_denials: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class Agent:
    def __init__(
        self,
        model: Model,
        client: ToolClient,
        config: AgentConfig | None = None,
        tracer: Tracer | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.client = client
        self.config = config or AgentConfig()
        self.tracer = tracer or Tracer()
        self._sleep = sleep
        self._clock = clock

    def run(self, goal: str) -> RunResult:
        cfg, trace = self.config, self.tracer
        start = self._clock()
        tools = [t for t in self.client.list_tools() if self._allowed(t["name"])]
        history: History = []
        result = RunResult(status="max_steps", answer="", steps=0, tokens=0, elapsed=0.0)
        trace.emit("run_start", 0, goal=goal, model=self.model.name, tools=[t["name"] for t in tools],
                   config={"max_steps": cfg.max_steps, "max_seconds": cfg.max_seconds,
                           "max_tokens": cfg.max_tokens, "max_retries": cfg.max_retries})

        for step in range(1, cfg.max_steps + 1):
            stop = self._budget_exceeded(start, result.tokens)
            if stop:
                result.status = stop
                break
            result.steps = step
            try:
                decision = self.model.decide(goal, tools, history)
            except ModelError as exc:
                trace.emit("model_error", step, error=str(exc))
                result.status = "model_error"
                break
            result.tokens += decision.tokens
            trace.emit("decision", step, thought=decision.thought, tokens=decision.tokens,
                       tool=decision.tool_call.name if decision.tool_call else None,
                       arguments=decision.tool_call.arguments if decision.tool_call else None,
                       answer=decision.answer)

            if decision.tool_call is None:
                answer = decision.answer or ""
                reason = cfg.completion_check(answer)
                if reason is None:
                    result.status, result.answer = "completed", answer
                    break
                trace.emit("completion_rejected", step, reason=reason)
                history.append((decision, Observation(None, f"Not done yet: {reason}", True)))
                continue

            obs = self._execute(step, decision.tool_call, result)
            result.tokens += estimate_tokens(obs.text)
            history.append((decision, obs))

        result.elapsed = round(self._clock() - start, 4)
        trace.emit("run_end", result.steps, status=result.status, answer=result.answer, tokens=result.tokens,
                   elapsed=result.elapsed, tool_errors=result.tool_errors, retries=result.retries)
        return result

    def _allowed(self, name: str) -> bool:
        return self.config.allowed_tools is None or name in self.config.allowed_tools

    def _budget_exceeded(self, start: float, tokens: int) -> str | None:
        if self._clock() - start >= self.config.max_seconds:
            return "time_budget"
        if tokens >= self.config.max_tokens:
            return "token_budget"
        return None

    def _execute(self, step: int, call: ToolCall, result: RunResult) -> Observation:
        trace = self.tracer
        result.tool_calls.append(call.name)
        # Enforce at call time too: a model can name a tool it was never shown.
        if not self._allowed(call.name):
            result.permission_denials += 1
            result.tool_errors += 1
            trace.emit("permission_denied", step, tool=call.name)
            return Observation(call, f"Permission denied: '{call.name}' is not in the allowlist", True)

        attempt = 0
        while True:
            trace.emit("tool_call", step, tool=call.name, arguments=call.arguments, attempt=attempt)
            t0 = self._clock()
            try:
                res = self.client.call_tool(call.name, call.arguments)
            except TransportError as exc:
                if exc.retryable and attempt < self.config.max_retries:
                    delay = self.config.backoff_base * (2 ** attempt)
                    trace.emit("retry", step, tool=call.name, attempt=attempt, delay=delay, error=str(exc))
                    result.retries += 1
                    attempt += 1
                    self._sleep(delay)
                    continue
                result.tool_errors += 1
                trace.emit("tool_result", step, tool=call.name, is_error=True, error=str(exc),
                           latency=round(self._clock() - t0, 4))
                return Observation(call, f"Tool call failed: {exc}", True)
            if res.is_error:
                result.tool_errors += 1
            trace.emit("tool_result", step, tool=call.name, is_error=res.is_error,
                       preview=res.text[:200], latency=round(self._clock() - t0, 4))
            return Observation(call, res.text, res.is_error)


__all__ = ["Agent", "AgentConfig", "Decision", "RunResult", "must_contain", "non_empty"]
