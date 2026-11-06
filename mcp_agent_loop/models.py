"""Model interface plus two adapters: a scripted offline model and an optional Claude model."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


class ModelError(Exception):
    """The model could not produce a decision. The run stops."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True)
class Decision:
    """One model turn: either a tool call or a final answer."""

    thought: str = ""
    tool_call: ToolCall | None = None
    answer: str | None = None
    tokens: int = 0


@dataclass(frozen=True)
class Observation:
    """What the loop feeds back. `call` is None for loop feedback such as a rejected finish."""

    call: ToolCall | None
    text: str
    is_error: bool


History = list[tuple[Decision, Observation]]


class Model(Protocol):
    name: str

    def decide(self, goal: str, tools: list[dict[str, Any]], history: History) -> Decision: ...


def estimate_tokens(text: str) -> int:
    """Rough count (4 chars per token). Good enough for an offline budget."""
    return max(1, len(text) // 4)


@dataclass
class ScriptedModel:
    """Replay a fixed plan. Step i of the plan answers turn i, so the run is deterministic.

    A plan step is {"tool": name, "args": {...}} or {"finish": template}. The template can use
    {last} (last observation text) and {obs[i]} (observation i).
    """

    plan: list[dict[str, Any]]
    name: str = "scripted"
    _calls: int = field(default=0, init=False)

    def decide(self, goal: str, tools: list[dict[str, Any]], history: History) -> Decision:
        index = len(history)
        if index >= len(self.plan):
            return Decision(thought="plan exhausted", answer="", tokens=1)
        step = self.plan[index]
        thought = step.get("thought", "")
        if "finish" in step:
            texts = [obs.text.strip() for _, obs in history]
            answer = str(step["finish"]).format_map({"last": texts[-1] if texts else "", "obs": texts})
            return Decision(thought=thought, answer=answer, tokens=estimate_tokens(thought + answer))
        self._calls += 1
        call = ToolCall(step["tool"], dict(step.get("args", {})), f"call_{self._calls}")
        return Decision(thought=thought, tool_call=call, tokens=estimate_tokens(thought + str(call.arguments)))


SYSTEM_PROMPT = (
    "You are a careful engineering agent working in a sandboxed repository. "
    "Use the tools to gather facts. Call one tool at a time. "
    "When you have enough evidence, reply with the final answer as plain text and call no tool."
)


class ClaudeModel:
    """Claude adapter. Keeps an append-only message list so prior assistant turns go back unchanged."""

    def __init__(self, model_id: str | None = None, client: Any = None, max_tokens: int = 16_000) -> None:
        self.name = model_id or os.environ.get("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL)
        self.max_tokens = max_tokens
        if client is None:
            try:
                import anthropic  # lazy: the offline path must not need the package
            except ImportError as exc:
                raise ModelError("install the 'anthropic' package to use the Claude model") from exc
            client = anthropic.Anthropic()
        self._client = client
        self._messages: list[dict[str, Any]] = []

    def decide(self, goal: str, tools: list[dict[str, Any]], history: History) -> Decision:
        if not history:
            self._messages = [{"role": "user", "content": goal}]
        else:
            self._messages.append({"role": "user", "content": _feedback(history[-1][1])})
        try:
            resp = self._client.beta.messages.create(
                model=self.name,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[_to_claude_tool(t) for t in tools],
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=self._messages,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK error as a model error
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise ModelError("model refused the request")
        self._messages.append({"role": "assistant", "content": resp.content})
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        uses = [b for b in resp.content if b.type == "tool_use"]
        if uses:
            call = ToolCall(uses[0].name, dict(uses[0].input), uses[0].id)
            return Decision(thought=text, tool_call=call, tokens=tokens)
        if resp.stop_reason == "max_tokens":
            raise ModelError("response hit max_tokens before an answer")
        return Decision(thought="", answer=text, tokens=tokens)


def _to_claude_tool(spec: dict[str, Any]) -> dict[str, Any]:
    return {"name": spec["name"], "description": spec.get("description", ""), "input_schema": spec["inputSchema"]}


def _feedback(obs: Observation) -> Any:
    if obs.call is None:
        return obs.text
    return [{"type": "tool_result", "tool_use_id": obs.call.call_id, "content": obs.text, "is_error": obs.is_error}]


def make_model(kind: str, plan: list[dict[str, Any]] | None = None) -> Model:
    """Build a model by name. 'claude' needs ANTHROPIC_API_KEY."""
    if kind == "scripted":
        return ScriptedModel(plan or [])
    if kind == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelError("set ANTHROPIC_API_KEY to use the Claude model")
        return ClaudeModel()
    raise ModelError(f"unknown model kind: {kind}")
