from pathlib import Path

import pytest

from mcp_agent_loop.tools import Sandbox, ToolError, build_tools


def test_read_file(workspace: Path) -> None:
    assert build_tools(workspace)["read_file"].handler({"path": "VERSION.txt"}).strip() == "1.4.2"


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "notes/../../x"])
def test_sandbox_rejects_escape(workspace: Path, path: str) -> None:
    with pytest.raises(ToolError, match="escapes the sandbox"):
        Sandbox(workspace).resolve(path)


def test_sandbox_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("secret")
    (root / "link.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(ToolError, match="escapes the sandbox"):
        build_tools(root)["read_file"].handler({"path": "link.txt"})


def test_list_dir_marks_directories(workspace: Path) -> None:
    listing = build_tools(workspace)["list_dir"].handler({}).splitlines()
    assert "notes/" in listing and "VERSION.txt" in listing


def test_search_text_reports_file_and_line(workspace: Path) -> None:
    out = build_tools(workspace)["search_text"].handler({"query": "todo", "path": "sample_repo"})
    assert out.startswith("sample_repo/calc.py:")


def test_search_text_no_matches(workspace: Path) -> None:
    assert build_tools(workspace)["search_text"].handler({"query": "zzz-not-there"}) == "no matches"


def test_run_tests_passes_on_sample_repo(workspace: Path) -> None:
    out = build_tools(workspace)["run_tests"].handler({"path": "sample_repo"})
    assert out.startswith("exit_code=0") and "Ran 3 tests" in out


def test_run_tests_times_out(tmp_path: Path) -> None:
    (tmp_path / "test_slow.py").write_text(
        "import time, unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        time.sleep(5)\n")
    with pytest.raises(ToolError, match="timed out"):
        build_tools(tmp_path, test_timeout=0.5)["run_tests"].handler({"path": "."})
