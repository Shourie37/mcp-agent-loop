import json
from pathlib import Path

import pytest

from mcp_agent_loop import protocol
from mcp_agent_loop.client import StdioClient, TransportError
from mcp_agent_loop.server import PROTOCOL_VERSION, ToolServer, validate_arguments
from mcp_agent_loop.protocol import RpcError
from mcp_agent_loop.tools import build_tools


def call(server: ToolServer, method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    reply = server.handle_line(protocol.encode(protocol.request(msg_id, method, params)))
    assert reply is not None
    return json.loads(reply)


def init(server: ToolServer) -> dict:
    return call(server, "initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}})


def test_initialize_returns_capabilities(server: ToolServer) -> None:
    reply = init(server)
    assert reply["jsonrpc"] == "2.0" and reply["id"] == 1
    assert reply["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert "tools" in reply["result"]["capabilities"]


def test_tools_require_initialize(server: ToolServer) -> None:
    reply = call(server, "tools/list")
    assert reply["error"]["code"] == protocol.INVALID_REQUEST


def test_tools_list_has_schemas(server: ToolServer) -> None:
    init(server)
    tools = call(server, "tools/list", msg_id=2)["result"]["tools"]
    assert {t["name"] for t in tools} == {"read_file", "list_dir", "search_text", "run_tests"}
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


def test_tools_call_returns_text_content(server: ToolServer) -> None:
    init(server)
    result = call(server, "tools/call", {"name": "read_file", "arguments": {"path": "VERSION.txt"}})["result"]
    assert result["isError"] is False
    assert result["content"] == [{"type": "text", "text": "1.4.2\n"}]


def test_tool_failure_is_a_result_not_a_protocol_error(server: ToolServer) -> None:
    init(server)
    result = call(server, "tools/call", {"name": "read_file", "arguments": {"path": "nope.txt"}})["result"]
    assert result["isError"] is True and "not a file" in result["content"][0]["text"]


@pytest.mark.parametrize("params, fragment", [
    ({"name": "no_such_tool", "arguments": {}}, "unknown tool"),
    ({"name": "read_file", "arguments": {}}, "missing required"),
    ({"name": "read_file", "arguments": {"path": 3}}, "must be string"),
    ({"name": "read_file", "arguments": {"path": "a", "mode": "w"}}, "unknown argument"),
])
def test_invalid_params(server: ToolServer, params: dict, fragment: str) -> None:
    init(server)
    err = call(server, "tools/call", params)["error"]
    assert err["code"] == protocol.INVALID_PARAMS and fragment in err["message"]


def test_unknown_method(server: ToolServer) -> None:
    init(server)
    assert call(server, "resources/list")["error"]["code"] == protocol.METHOD_NOT_FOUND


def test_parse_error_and_invalid_request(server: ToolServer) -> None:
    assert json.loads(server.handle_line("{not json"))["error"]["code"] == protocol.PARSE_ERROR
    bad = json.loads(server.handle_line('{"jsonrpc":"1.0","id":1,"method":"ping"}'))
    assert bad["error"]["code"] == protocol.INVALID_REQUEST and bad["id"] is None


def test_notification_gets_no_reply(server: ToolServer) -> None:
    assert server.handle_line(protocol.encode(protocol.notification("notifications/initialized"))) is None


def test_injected_failure_is_retryable_code(workspace: Path) -> None:
    server = ToolServer(build_tools(workspace), fail_first_calls=1)
    init(server)
    err = call(server, "tools/call", {"name": "list_dir", "arguments": {}})["error"]
    assert err["code"] in protocol.RETRYABLE_CODES


def test_validate_arguments_rejects_non_object() -> None:
    with pytest.raises(RpcError):
        validate_arguments({"type": "object"}, ["x"])


def test_stdio_subprocess_end_to_end(workspace: Path) -> None:
    client = StdioClient(workspace, fail_first_calls=1, timeout=10)
    try:
        assert len(client.list_tools()) == 4
        with pytest.raises(TransportError) as exc:
            client.call_tool("read_file", {"path": "VERSION.txt"})
        assert exc.value.retryable
        assert client.call_tool("read_file", {"path": "VERSION.txt"}).text.strip() == "1.4.2"
    finally:
        client.close()
