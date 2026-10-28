"""Validate user configuration and authoritative run metadata at their boundaries."""
from __future__ import annotations

import argparse
from functools import wraps
import json
import os
from pathlib import Path
import sys


class HpcInputError(ValueError):
    """An invalid configuration or saved input, safe to explain at the CLI."""


def cli_input_errors(prog: str):
    """Give expected input errors a failing CLI status; keep internal errors visible."""
    def decorate(main):
        @wraps(main)
        def wrapped(*args, **kwargs):
            try:
                return main(*args, **kwargs)
            except HpcInputError as exc:
                print(f"{prog}: error: {exc}", file=sys.stderr)
                raise SystemExit(2) from None
        return wrapped
    return decorate


def input_integer(name, value, *, minimum: int) -> int:
    """Use the task integer contract for configuration and manifest counts."""
    from .tasks import _task_integer

    try:
        return _task_integer(name, value, minimum=minimum)
    except ValueError as exc:
        raise HpcInputError(str(exc)) from exc


def _integer_argument(value: str, minimum: int) -> int:
    try:
        integer = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}") from exc
    if integer < minimum:
        raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {value!r}")
    return integer


def nonnegative_integer(value: str) -> int:
    return _integer_argument(value, 0)


def positive_integer(value: str) -> int:
    return _integer_argument(value, 1)


def path_argument(value: str) -> Path:
    if not value.strip() or "\0" in value:
        raise argparse.ArgumentTypeError("must be a nonempty path without NUL")
    return Path(value)


def resolve_local_runs_root(value=None) -> Path:
    """Expand local path controls consistently before any output is created."""
    from ..paths import runs_root

    try:
        raw = os.fspath(runs_root() if value is None else value)
        if not isinstance(raw, str) or not raw.strip() or "\0" in raw:
            raise HpcInputError("runs_root must be a nonempty path string without NUL")
        expanded = os.path.expandvars(raw)
        if not expanded.strip():
            raise HpcInputError("runs_root expands to an empty path")
        return Path(expanded).expanduser()
    except (TypeError, RuntimeError) as exc:
        raise HpcInputError(f"cannot resolve runs_root {value!r}: {exc}") from exc


def read_job_spec(path: Path, *, allow_missing: bool = False,
                  require_output: bool = True) -> dict | None:
    """Read a spec without creating outputs; inspection may use the default path."""
    try:
        spec = json.loads(path.read_text())
    except FileNotFoundError as exc:
        if allow_missing:
            return None
        raise HpcInputError(f"no spec at {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HpcInputError(f"cannot read job spec {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise HpcInputError(f"{path}: job spec must be a JSON object")
    task = spec.get("task")
    if not isinstance(task, str) or not task.strip():
        raise HpcInputError(f"{path}: job spec task must be a nonempty string")
    if not isinstance(spec.get("params", {}), dict):
        raise HpcInputError(f"{path}: job spec params must be a JSON object")
    if require_output or "output_dir" in spec:
        output = spec.get("output_dir")
        if not isinstance(output, str) or not output.strip() or "\0" in output:
            raise HpcInputError(f"{path}: job spec output_dir must be a nonempty path string")
    return spec


def load_run_manifest(run_dir: Path) -> tuple[dict, int]:
    """Read the manifest and require a positive, integral task count."""
    path = run_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HpcInputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise HpcInputError(f"{path} must contain a JSON object")
    if "n_tasks" not in manifest:
        raise HpcInputError(f"{path} is missing n_tasks")
    try:
        n_tasks = input_integer("n_tasks", manifest["n_tasks"], minimum=1)
    except HpcInputError as exc:
        raise HpcInputError(f"{path}: {exc}") from exc
    return manifest, n_tasks
