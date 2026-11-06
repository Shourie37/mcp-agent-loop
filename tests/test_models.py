from types import SimpleNamespace
from typing import Any

import pytest

from mcp_agent_loop.models import (ClaudeModel, ModelError, Observation, ScriptedModel, ToolCall, make_model)

TOOLS = [{"name": "read_file", "description": "d", "inputSchema": {"type": "object", "properties": {}}}]


def test_scripted_model_is_deterministic() -> None:
    plan = [{"tool": "read_file", "args": {"path": "a"}}, {"finish": "got {last} / {obs[0]}"}]
    model = ScriptedModel(plan)
    first = model.decide("g", TOOLS, [])
    assert first.tool_call and first.tool_call.name == "read_file"
    done = model.decide("g", TOOLS, [(first, Observation(first.tool_call, " x \n", False))])
    assert done.answer == "got x / x"


def test_make_model_claude_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ModelError, match="ANTHROPIC_API_KEY"):
        make_model("claude")


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses, self.calls = responses, []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def response(stop: str, *blocks: Any) -> Any:
    return SimpleNamespace(stop_reason=stop, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_claude_adapter_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    use = SimpleNamespace(type="tool_use", id="tu_1", name="read_file", input={"path": "VERSION.txt"})
    msgs = FakeMessages([
        response("tool_use", SimpleNamespace(type="text", text="Reading."), use),
        response("end_turn", SimpleNamespace(type="text", text="Version 1.4.2")),
    ])
    model = ClaudeModel(client=SimpleNamespace(beta=SimpleNamespace(messages=msgs)))
    assert model.name == "claude-sonnet-5-5"

    d1 = model.decide("goal", TOOLS, [])
    assert d1.tool_call == ToolCall("read_file", {"path": "VERSION.txt"}, "tu_1") and d1.tokens == 15
    assert msgs.calls[0]["tools"][0]["input_schema"] == TOOLS[0]["inputSchema"]

    d2 = model.decide("goal", TOOLS, [(d1, Observation(d1.tool_call, "1.4.2", False))])
    assert d2.answer == "Version 1.4.2"
    sent = msgs.calls[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "1.4.2",
                                     "is_error": False}


def test_claude_refusal_is_model_error() -> None:
    msgs = FakeMessages([response("refusal")])
    model = ClaudeModel(client=SimpleNamespace(beta=SimpleNamespace(messages=msgs)))
    with pytest.raises(ModelError, match="refused"):
        model.decide("goal", TOOLS, [])
