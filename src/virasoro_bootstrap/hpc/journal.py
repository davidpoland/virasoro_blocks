"""Atomic per-task progress journal.

A task's durable mid-flight state (completed probe verdicts, the last usable
SDPB checkpoint, requeue counters) lives in one JSON file next to its outputs.
Writes are atomic (tmp + rename), so a kill at any moment leaves the previous
consistent state on disk; a resumed worker reads it and skips finished work.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def progress_counter(document, key: str) -> int:
    """Read a nonnegative JSON counter; malformed fields carry no usable count."""
    value = document.get(key) if isinstance(document, dict) else None
    return value if type(value) is int and value >= 0 else 0


def progress_mapping(document, key: str) -> dict:
    """Copy saved work entries; malformed fields or entries must be recomputed."""
    value = document.get(key) if isinstance(document, dict) else None
    if not isinstance(value, dict):
        return {}
    return {name: entry for name, entry in value.items()
            if isinstance(name, str) and isinstance(entry, dict)}


class Journal:
    """A dict persisted to *path* with atomic whole-file writes."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data: dict = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                self.data = loaded if isinstance(loaded, dict) else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A corrupt journal (e.g. torn write on an exotic filesystem)
                # only costs re-doing work; never block the task on it.
                self.data = {}

    def update(self, **updates) -> None:
        candidate = {**self.data, **updates}
        encoded = json.dumps(candidate, indent=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(encoded)
        tmp.replace(self.path)
        self.data = candidate
