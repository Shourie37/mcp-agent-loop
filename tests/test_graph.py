import pytest

from mcp_agent_loop.graph import END, Graph, auto_approve, human_review


def test_branching_follows_state() -> None:
    g = Graph()
    g.add_node("start", lambda s: {"n": s["n"] * 2})
    g.add_node("big", lambda s: {"label": "big"})
    g.add_node("small", lambda s: {"label": "small"})
    g.add_branch("start", lambda s: "big" if s["n"] > 5 else "small")
    assert g.run("start", {"n": 4}).state["label"] == "big"
    result = g.run("start", {"n": 1})
    assert result.state["label"] == "small" and result.path == ["start", "small"] and result.status == "done"


def test_node_retry_then_success() -> None:
    calls = {"n": 0}

    def flaky(state: dict) -> dict:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("try again")
        return {"ok": True}

    g = Graph().add_node("flaky", flaky, retries=2)
    result = g.run("flaky")
    assert result.status == "done" and result.state["ok"] and len(result.state["_errors"]) == 2


def test_node_fails_after_retries() -> None:
    g = Graph().add_node("bad", lambda s: 1 / 0, retries=1)
    result = g.run("bad")
    assert result.status == "failed" and "ZeroDivisionError" in (result.error or "")


@pytest.mark.parametrize("approve, expected", [(True, "publish"), (False, "escalate")])
def test_human_review_routes_on_decision(approve: bool, expected: str) -> None:
    g = Graph()
    g.add_node("review", human_review(auto_approve if approve else (lambda s: False)))
    g.add_node("publish", lambda s: {"out": "publish"})
    g.add_node("escalate", lambda s: {"out": "escalate"})
    g.add_branch("review", lambda s: "publish" if s["approved"] else "escalate")
    g.add_edge("publish", END).add_edge("escalate", END)
    assert g.run("review").state["out"] == expected


def test_cycle_is_bounded() -> None:
    g = Graph().add_node("loop", lambda s: {"i": s.get("i", 0) + 1})
    g.add_edge("loop", "loop")
    result = g.run("loop", max_transitions=5)
    assert result.status == "max_transitions" and result.state["i"] == 5


def test_rejects_duplicate_routes_and_unknown_nodes() -> None:
    g = Graph().add_node("a", lambda s: None)
    g.add_edge("a", END)
    with pytest.raises(ValueError):
        g.add_branch("a", lambda s: END)
    with pytest.raises(ValueError):
        g.add_edge("missing", END)
    assert Graph().run("missing").status == "failed"
