"""Demo tools. Every path is resolved inside one sandbox root."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_READ_BYTES = 64_000
MAX_SEARCH_HITS = 50


class ToolError(Exception):
    """The tool ran and failed. The agent sees the message as an observation."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def spec(self) -> dict[str, Any]:
        """Return the MCP tools/list entry."""
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


class Sandbox:
    """Resolve relative paths and reject any that leave the root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, rel: str) -> Path:
        # resolve() follows symlinks, so a link that points outside is also rejected.
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root):
            raise ToolError(f"path '{rel}' escapes the sandbox")
        return target

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


def _schema(props: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def build_tools(root: Path, test_timeout: float = 20.0) -> dict[str, Tool]:
    """Build the tool registry for one sandbox root."""
    box = Sandbox(root)

    def read_file(args: dict[str, Any]) -> str:
        path = box.resolve(args["path"])
        if not path.is_file():
            raise ToolError(f"not a file: {args['path']}")
        data = path.read_bytes()[:MAX_READ_BYTES]
        return data.decode("utf-8", errors="replace")

    def list_dir(args: dict[str, Any]) -> str:
        path = box.resolve(args.get("path", "."))
        if not path.is_dir():
            raise ToolError(f"not a directory: {args.get('path', '.')}")
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir() if not p.name.startswith("."))
        return "\n".join(names)

    def search_text(args: dict[str, Any]) -> str:
        needle = args["query"].lower()
        start = box.resolve(args.get("path", "."))
        hits: list[str] = []
        for file in sorted(p for p in start.rglob("*") if p.is_file()):
            if any(part.startswith(".") or part == "__pycache__" for part in file.relative_to(box.root).parts):
                continue
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for num, line in enumerate(lines, 1):
                if needle in line.lower():
                    hits.append(f"{box.rel(file)}:{num}: {line.strip()}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "no matches"

    def run_tests(args: dict[str, Any]) -> str:
        path = box.resolve(args.get("path", "."))
        if not path.is_dir():
            raise ToolError(f"not a directory: {args.get('path', '.')}")
        cmd = [sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", ".", "-t", "."]
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(cmd, cwd=path, env=env, capture_output=True, text=True, timeout=test_timeout)
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"tests timed out after {test_timeout:g}s") from exc
        output = (proc.stdout + proc.stderr).strip()
        return f"exit_code={proc.returncode}\n{output[-4000:]}"

    tools = [
        Tool("read_file", "Read a UTF-8 text file inside the sandbox.",
             _schema({"path": {"type": "string"}}, ["path"]), read_file),
        Tool("list_dir", "List entries of a directory inside the sandbox.",
             _schema({"path": {"type": "string"}}, []), list_dir),
        Tool("search_text", "Case-insensitive substring search over sandbox files.",
             _schema({"query": {"type": "string"}, "path": {"type": "string"}}, ["query"]), search_text),
        Tool("run_tests", "Run the unittest suite in a sandbox directory, with a timeout.",
             _schema({"path": {"type": "string"}}, []), run_tests),
    ]
    return {t.name: t for t in tools}
