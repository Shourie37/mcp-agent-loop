"""JSONL execution trace: one event per line, flushed as it happens."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


class Tracer:
    """Record events in memory and, if a path is given, append them to a JSONL file."""

    def __init__(self, path: Path | None = None, run_id: str = "run") -> None:
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self._fh: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")
        self.path = path

    def emit(self, event: str, step: int, **fields: Any) -> dict[str, Any]:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "step": step, "event": event, **fields}
        self.events.append(record)
        if self._fh is not None:
            # Flush per event so a crash still leaves a usable trace.
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
