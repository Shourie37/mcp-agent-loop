from pathlib import Path

import pytest

from mcp_agent_loop.client import InProcessClient
from mcp_agent_loop.server import ToolServer
from mcp_agent_loop.tools import build_tools

PROJECT = Path(__file__).resolve().parent.parent
WORKSPACE = PROJECT / "examples" / "workspace"


@pytest.fixture
def workspace() -> Path:
    return WORKSPACE


@pytest.fixture
def server(workspace: Path) -> ToolServer:
    return ToolServer(build_tools(workspace))


def make_client(workspace: Path, fail_first_calls: int = 0) -> InProcessClient:
    return InProcessClient(ToolServer(build_tools(workspace), fail_first_calls=fail_first_calls))
