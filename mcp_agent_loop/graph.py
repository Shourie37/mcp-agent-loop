"""A small workflow-graph engine: nodes, edges, branches, shared state, retries, human review."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

END = "__end__"

State = dict[str, Any]
NodeFn = Callable[[State], "State | None"]
Router = Callable[[State], str]
Approver = Callable[[State], bool]


@dataclass
class GraphResult:
    status: str  # done | failed | max_transitions
    state: State
    path: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class _Node:
    fn: NodeFn
    retries: int


class Graph:
    def __init__(self) -> None:
        self._nodes: dict[str, _Node] = {}
        self._edges: dict[str, str] = {}
        self._branches: dict[str, Router] = {}

    def add_node(self, name: str, fn: NodeFn, retries: int = 0) -> Graph:
        if name in self._nodes or name == END:
            raise ValueError(f"duplicate or reserved node name: {name}")
        self._nodes[name] = _Node(fn, retries)
        return self

    def add_edge(self, src: str, dst: str) -> Graph:
        self._check_source(src)
        self._edges[src] = dst
        return self

    def add_branch(self, src: str, router: Router) -> Graph:
        """Route to the node name the router returns, based on state."""
        self._check_source(src)
        self._branches[src] = router
        return self

    def _check_source(self, src: str) -> None:
        if src not in self._nodes:
            raise ValueError(f"unknown node: {src}")
        if src in self._edges or src in self._branches:
            raise ValueError(f"node already has an outgoing route: {src}")

    def run(self, start: str, state: State | None = None, max_transitions: int = 50) -> GraphResult:
        state = dict(state or {})
        path: list[str] = []
        current = start
        for _ in range(max_transitions):
            if current == END:
                return GraphResult("done", state, path)
            node = self._nodes.get(current)
            if node is None:
                return GraphResult("failed", state, path, error=f"unknown node: {current}")
            path.append(current)
            error = self._run_node(current, node, state)
            if error:
                return GraphResult("failed", state, path, error=error)
            current = self._next(current, state)
        return GraphResult("max_transitions", state, path)

    def _run_node(self, name: str, node: _Node, state: State) -> str | None:
        last = ""
        for attempt in range(node.retries + 1):
            try:
                update = node.fn(state)
            except Exception as exc:  # noqa: BLE001 - a node failure is data, not a crash
                last = f"{name}: {type(exc).__name__}: {exc}"
                state.setdefault("_errors", []).append({"node": name, "attempt": attempt, "error": str(exc)})
                continue
            if update:
                state.update(update)
            return None
        return last

    def _next(self, name: str, state: State) -> str:
        if name in self._branches:
            return self._branches[name](state)
        return self._edges.get(name, END)


def auto_approve(state: State) -> bool:
    return True


def console_approver(state: State) -> bool:
    reply = input(f"Approve result? {state.get('summary', '')!r} [y/N] ")
    return reply.strip().lower() in {"y", "yes"}


def human_review(approver: Approver, key: str = "approved") -> NodeFn:
    """Build a node that asks a reviewer and stores the decision in state[key]."""
    def node(state: State) -> State:
        return {key: bool(approver(state))}
    return node
