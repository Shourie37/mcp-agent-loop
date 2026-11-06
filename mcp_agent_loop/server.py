"""MCP-shaped tool server: initialize, tools/list, tools/call over JSON-RPC 2.0."""

from __future__ import annotations

from typing import Any, TextIO

from . import __version__, protocol
from .protocol import RpcError
from .tools import Tool, ToolError

PROTOCOL_VERSION = "2025-06-18"

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list,
}


def validate_arguments(schema: dict[str, Any], args: Any) -> None:
    """Check required keys, unknown keys, and top-level types. Raise INVALID_PARAMS."""
    if not isinstance(args, dict):
        raise RpcError(protocol.INVALID_PARAMS, "arguments must be an object")
    props: dict[str, Any] = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise RpcError(protocol.INVALID_PARAMS, f"missing required argument(s): {', '.join(missing)}")
    if schema.get("additionalProperties") is False:
        extra = sorted(set(args) - set(props))
        if extra:
            raise RpcError(protocol.INVALID_PARAMS, f"unknown argument(s): {', '.join(extra)}")
    for key, value in args.items():
        expected = _JSON_TYPES.get(props.get(key, {}).get("type", ""))
        if expected and not isinstance(value, expected):
            raise RpcError(protocol.INVALID_PARAMS, f"argument '{key}' must be {props[key]['type']}")


class ToolServer:
    """Stateful request handler. Transport-agnostic: takes and returns dicts or lines."""

    def __init__(self, tools: dict[str, Tool], fail_first_calls: int = 0) -> None:
        self.tools = tools
        self.initialized = False
        # Fault injection for tests and evals: the first N tools/call requests fail transiently.
        self._failures_left = fail_first_calls
        self.call_count = 0

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one parsed message. Return None for notifications."""
        msg_id = msg.get("id")
        is_notification = "id" not in msg
        try:
            result = self._dispatch(msg["method"], msg.get("params") or {})
        except RpcError as err:
            return None if is_notification else protocol.failure(msg_id, err)
        except Exception as exc:  # noqa: BLE001 - never crash the transport loop
            internal = RpcError(protocol.INTERNAL_ERROR, f"internal error: {type(exc).__name__}")
            return None if is_notification else protocol.failure(msg_id, internal)
        return None if is_notification else protocol.success(msg_id, result)

    def handle_line(self, line: str) -> str | None:
        try:
            msg = protocol.parse(line)
        except RpcError as err:
            return protocol.encode(protocol.failure(None, err))
        if "method" not in msg:
            return None  # A response sent to the server: nothing to do.
        reply = self.handle(msg)
        return protocol.encode(reply) if reply is not None else None

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            self.initialized = True
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mcp-agent-loop", "version": __version__},
            }
        if method == "notifications/initialized" or method == "ping":
            return {}
        if not self.initialized:
            raise RpcError(protocol.INVALID_REQUEST, "server not initialized")
        if method == "tools/list":
            return {"tools": [t.spec() for t in self.tools.values()]}
        if method == "tools/call":
            return self._call(params)
        raise RpcError(protocol.METHOD_NOT_FOUND, f"method not found: {method}")

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None:
            raise RpcError(protocol.INVALID_PARAMS, f"unknown tool: {name}")
        args = params.get("arguments", {})
        validate_arguments(tool.input_schema, args)
        self.call_count += 1
        if self._failures_left > 0:
            self._failures_left -= 1
            raise RpcError(protocol.TRANSIENT_ERROR, "transient failure (injected)")
        try:
            text, is_error = tool.handler(args), False
        except ToolError as exc:
            text, is_error = str(exc), True
        return {"content": [{"type": "text", "text": text}], "isError": is_error}


def serve_stdio(server: ToolServer, stdin: TextIO, stdout: TextIO) -> None:
    """Read requests line by line until EOF and write replies."""
    for line in stdin:
        if not line.strip():
            continue
        reply = server.handle_line(line)
        if reply is not None:
            stdout.write(reply)
            stdout.flush()
