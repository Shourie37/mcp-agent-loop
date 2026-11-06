"""JSON-RPC 2.0 framing for the MCP stdio transport (one JSON object per line)."""

from __future__ import annotations

import json
from typing import Any

JSONRPC = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# Server-defined code: the call did not run and is safe to retry.
TRANSIENT_ERROR = -32001

RETRYABLE_CODES = frozenset({TRANSIENT_ERROR})


class RpcError(Exception):
    """A JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def request(msg_id: int | str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": JSONRPC, "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": JSONRPC, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def success(msg_id: int | str | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC, "id": msg_id, "result": result}


def failure(msg_id: int | str | None, err: RpcError) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC, "id": msg_id, "error": err.to_dict()}


def encode(msg: dict[str, Any]) -> str:
    """Serialize one message as a single newline-terminated line."""
    return json.dumps(msg, separators=(",", ":")) + "\n"


def parse(line: str) -> dict[str, Any]:
    """Parse and validate one incoming line. Raise RpcError on bad input."""
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RpcError(PARSE_ERROR, f"Parse error: {exc.msg}") from exc
    if not isinstance(msg, dict) or msg.get("jsonrpc") != JSONRPC:
        raise RpcError(INVALID_REQUEST, "Invalid request: expected a JSON-RPC 2.0 object")
    is_response = "result" in msg or "error" in msg
    if not is_response and not isinstance(msg.get("method"), str):
        raise RpcError(INVALID_REQUEST, "Invalid request: 'method' must be a string")
    if "params" in msg and not isinstance(msg["params"], dict):
        raise RpcError(INVALID_PARAMS, "Invalid params: 'params' must be an object")
    return msg
