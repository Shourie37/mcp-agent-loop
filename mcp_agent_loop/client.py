"""Tool clients: in-process (fast, for tests and evals) and stdio subprocess (real transport)."""

from __future__ import annotations

import itertools
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import protocol
from .server import PROTOCOL_VERSION, ToolServer


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool


class TransportError(Exception):
    """The call did not produce a tool result."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolClient(Protocol):
    def list_tools(self) -> list[dict[str, Any]]: ...
    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...
    def close(self) -> None: ...


class _RpcClient:
    """Shared request/response logic. Subclasses supply one line round trip."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)

    def _roundtrip(self, line: str) -> str:
        raise NotImplementedError

    def _notify(self, line: str) -> None:
        raise NotImplementedError

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        msg_id = next(self._ids)
        reply = protocol.parse(self._roundtrip(protocol.encode(protocol.request(msg_id, method, params))))
        if reply.get("id") != msg_id:
            raise TransportError(f"response id mismatch: {reply.get('id')} != {msg_id}", retryable=False)
        if "error" in reply:
            err = reply["error"]
            retryable = err.get("code") in protocol.RETRYABLE_CODES
            raise TransportError(f"rpc error {err.get('code')}: {err.get('message')}", retryable=retryable)
        return reply["result"]

    def initialize(self) -> dict[str, Any]:
        info = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mcp-agent-loop-client", "version": "0"},
        })
        self._notify(protocol.encode(protocol.notification("notifications/initialized")))
        return dict(info)

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._request("tools/list")["tools"])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        return ToolResult(text=text, is_error=bool(result.get("isError")))


class InProcessClient(_RpcClient):
    """Run the server in the same process but still pass every message through JSON lines."""

    def __init__(self, server: ToolServer) -> None:
        super().__init__()
        self.server = server
        self.initialize()

    def _roundtrip(self, line: str) -> str:
        reply = self.server.handle_line(line)
        if reply is None:
            raise TransportError("no reply from server", retryable=False)
        return reply

    def _notify(self, line: str) -> None:
        self.server.handle_line(line)

    def close(self) -> None:
        pass


class StdioClient(_RpcClient):
    """Spawn `python -m mcp_agent_loop serve` and talk to it over stdin/stdout."""

    def __init__(self, root: Path, fail_first_calls: int = 0, timeout: float = 30.0) -> None:
        super().__init__()
        self.timeout = timeout
        cmd = [sys.executable, "-m", "mcp_agent_loop", "serve", "--root", str(root),
               "--fail-first-calls", str(fail_first_calls)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._lines: queue.Queue[str] = queue.Queue()
        # A reader thread lets us apply a timeout to each response.
        threading.Thread(target=self._pump, daemon=True).start()
        self.initialize()

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.put(line)
        self._lines.put("")

    def _write(self, line: str) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise TransportError("server process exited", retryable=False) from exc

    def _roundtrip(self, line: str) -> str:
        self._write(line)
        try:
            reply = self._lines.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TransportError(f"no response within {self.timeout:g}s", retryable=True) from exc
        if not reply:
            raise TransportError("server closed the stream", retryable=False)
        return reply

    def _notify(self, line: str) -> None:
        self._write(line)

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


__all__ = ["InProcessClient", "StdioClient", "ToolClient", "ToolResult", "TransportError"]
