"""Worker-process counts that are safe in the caller's script.

Under the ``spawn`` / ``forkserver`` start methods (the default on macOS and Windows, and on Linux from Python
3.14) every worker process re-runs the main script's top level before it can take work.  A script without an
``if __name__ == "__main__":`` guard then rebuilds the whole problem in each worker, and the nested pool start
kills it: ``BrokenProcessPool`` after the full cost of the build.  The library's parallel paths (``num_threads``,
default up to 4) ask :func:`usable_workers` first and run serially, with a one-time warning, where workers cannot
start.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import multiprocessing
import os
import sys
import tokenize
import warnings

__all__ = ["usable_workers", "validate_worker_count"]

_WARNED = False


def _main_guard_test(test) -> bool:
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return any(_main_guard_test(value) for value in test.values)
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)):
        return False
    left, right = test.left, test.comparators[0]
    return any(isinstance(name, ast.Name) and name.id == "__name__"
               and isinstance(label, ast.Constant) and label.value == "__main__"
               for name, label in ((left, right), (right, left)))


@lru_cache(maxsize=16)
def _script_has_main_guard(path, version) -> bool:
    # The stat signature invalidates a changed script without reparsing a large
    # CLI module for every table. Respect Python's source encoding and BOM too.
    with tokenize.open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        # Delayed guards inside definitions can be unrelated to module startup.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If) and _main_guard_test(node.test):
            return True
        pending.extend(ast.iter_child_nodes(node))
    return False


def _main_script_unguarded() -> bool:
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if not path:
        return False            # interactive session / notebook: workers do not re-run anything
    try:
        path = os.path.abspath(path)
        info = os.stat(path)
        version = (info.st_mtime_ns, info.st_ctime_ns, info.st_size, info.st_ino)
        return not _script_has_main_guard(path, version)
    except (OSError, SyntaxError, UnicodeError):
        # A stdin script has __file__ == '<stdin>'; spawn still tries to reload
        # that nonexistent path. Missing/unreadable saved scripts fail likewise.
        return True


def validate_worker_count(requested) -> int:
    """Normalize a positive integral worker count without truncating inputs."""
    from .requests import _integer_field, RequestValidationError

    requested = _integer_field("num_threads", requested)
    if requested < 1:
        raise RequestValidationError("num_threads must be >= 1")
    return requested


def usable_workers(requested) -> int:
    """``requested``, or 1 when workers cannot safely reload the main script."""
    global _WARNED

    from .requests import _integer_field

    # Certification callers document nonpositive counts as serial, clamping
    # this result to one. Interpolation validates its positive count separately.
    requested = _integer_field("num_threads", requested)
    if requested <= 1:
        return requested
    process = multiprocessing.current_process()
    if process.daemon or getattr(process, "_inheriting", False):
        # Daemon pool workers cannot create children under any start method;
        # a worker still bootstrapping cannot safely nest a pool either.
        return 1
    method = multiprocessing.get_start_method(allow_none=True) or multiprocessing.get_context().get_start_method()
    if method == "fork" or not _main_script_unguarded():
        return requested
    if not _WARNED:
        _WARNED = True
        warnings.warn(
            f"running serially instead of with {requested} worker processes: under the {method!r} start method each "
            "worker must reload the main script, which is unavailable or has no `if __name__ == \"__main__\":` guard. "
            "Run a saved script with its work under that guard to use the workers "
            "(or pass num_threads=1 to silence this).",
            RuntimeWarning, stacklevel=3,
        )
    return 1
