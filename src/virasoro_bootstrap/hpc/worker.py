"""Job-spec worker: run one task inside one allocation (or local process).

    python -m virasoro_bootstrap.hpc.worker artifacts/runs/<id>/specs/0003.json

Contract:

- **Idempotent**: if the task's ``result.json`` exists and matches both the
  supplied parameters and the defaults that fill the rest, exit 0 immediately.
- **Resumable**: the task journals probe-level progress (``progress.json``,
  stamped with the spec hash); a rerun after any kill continues mid-chain, and
  a journal left by a different (since edited) spec is set aside, not replayed.
- **Interruption-aware**: SIGUSR1/SIGTERM (sent by SLURM via
  ``--signal=B:USR1@600`` before the walltime, or on preemption) set a stop
  flag the task polls between solver calls. The task raises
  :class:`~.tasks.TaskInterrupted`; the worker records ``interrupted`` status,
  requeues its own SLURM (array) task, and exits 3. Nothing about this path is
  an error: interruption is the normal case.
- **Exclusive**: a stable ``.worker.lock`` in the output directory serializes
  overlapping submissions, including scratch aliases. A busy attempt leaves
  the active worker's metadata untouched, attempts to requeue and exits 3.

Exit codes: 0 done/skipped, 1 failed, 2 bad spec or no task lock, 3 interrupted/busy (requeue attempted).

The sbatch template ``exec``s this module so the batch-step signal lands on
the Python process itself, not a wrapping shell.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import errno
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from .journal import Journal, progress_counter
from .inputs import read_job_spec
from .provenance import execution_matches, execution_settings, settings_digest
from .tasks import TaskContext, TaskInterrupted, get_task, result_error

RESULT_NAME = "result.json"
STATUS_NAME = "status.json"
JOURNAL_NAME = "progress.json"
_LOCK_NAME = ".worker.lock"


class _TaskLockError(OSError):
    """The task's output lock could not be acquired."""


class _TaskBusy(_TaskLockError):
    """Another worker or maintenance process owns the task output."""


@contextmanager
def _task_output_lock(output_dir):
    """Lock a stable inode through execution, publication and housekeeping.

    Atomic journal/status replacements cannot serve as lock anchors. Keep one
    empty file per task, opened for writing so exclusive flock also works on
    NFS. Scratch and durable aliases reach the same file.
    """
    import fcntl

    path = Path(output_dir) / _LOCK_NAME
    descriptor = None
    acquired = False
    try:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise _TaskBusy(f"task output is already in use: {output_dir}") from exc
        except OSError as exc:
            unsupported = exc.errno in {errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS}
            raise _TaskLockError(
                f"cannot lock {path} ({errno.errorcode.get(exc.errno, exc.errno)}: {exc.strerror})"
                + ("; this filesystem does not provide file locks (e.g. NFS without lockd)" if unsupported else "")
            ) from exc
        yield
    finally:
        if descriptor is not None:
            try:
                if acquired:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def spec_hash(task_name: str, params: dict) -> str:
    """Content hash identifying "the same work" for idempotence checks."""
    canonical = json.dumps({"task": task_name, "params": params}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def result_doc_failure(doc: dict) -> str | None:
    """The failure a ``result.json`` document records, or None if it is an answer.

    A task that catches its own failure returns an error payload (see
    :func:`~.tasks.result_error`); the worker marks that document ``failed``.
    Older documents without the mark are judged by their payload.
    """
    failed = doc.get("failed")
    if failed:
        return str(failed)
    return result_error(doc.get("result"))


# Tasks whose answer is an OPE coefficient. The spin-0 scalar_G convention sets its scale, so a result of
# another convention is neither reused nor collected with new ones.
SCALE_DEPENDENT_TASKS = frozenset({
    "isolated_scalar_four_point_ope_bounds", "isolated_scalar_four_point_fixed_ope", "scalar_ope_max_scan",
})


def result_doc_convention(doc: dict) -> str:
    """The spin-0 scalar_G convention a ``result.json`` document was computed in."""
    from virasoro_blocks._conventions import SCALAR_G_SPIN0_DOUBLE, SCALAR_G_SPIN0_KEY

    return doc.get(SCALAR_G_SPIN0_KEY, SCALAR_G_SPIN0_DOUBLE)


def result_doc_convention_matches(doc: dict) -> bool:
    """False for a scale-dependent result computed in the other spin-0 scalar_G convention than this process's."""
    from virasoro_blocks._conventions import scalar_G_spin0_convention

    return doc.get("task") not in SCALE_DEPENDENT_TASKS or result_doc_convention(doc) == scalar_G_spin0_convention()


def result_doc_is_final(doc: dict | None, digest: str, *, execution: dict | None = None) -> bool:
    """True when *doc* is a successful result of the spec hashing to *digest*.

    The one rule the worker (skip on rerun) and ``vb-submit --retry`` (what
    is pending) share: a stale hash, a recorded failure, or a scale-dependent
    result of the other spin-0 scalar_G convention is not final. Reuse also
    requires the same task defaults (``execution``); historical collection and
    cleanup omit that check because an earlier result remains a valid record.
    """
    return (
        isinstance(doc, dict)
        and doc.get("spec_hash") == digest
        and execution_matches(doc.get("execution_settings"), execution,
                              resolved_settings=(doc.get("result") or {}).get("resolved_settings")
                              if isinstance(doc.get("result"), dict) else None)
        and result_doc_failure(doc) is None
        and result_doc_convention_matches(doc)
    )


def _task_output_dir(run_dir: Path, index: int, output_dir=None) -> Path:
    """Resolve a saved output path just as the worker does for a spec."""
    if output_dir is None:
        return run_dir / "tasks" / f"{index:04d}"
    if not isinstance(output_dir, str) or not output_dir.strip() or '\0' in output_dir:
        raise ValueError("job spec output_dir must be a nonempty path string")
    path = Path(output_dir)
    return path if path.is_absolute() else run_dir / path


def _task_result_path(run_dir: Path, index: int, output_dir=None) -> Path:
    """Use the durable result when a scratch task's result has been purged."""
    task_dir = _task_output_dir(run_dir, index, output_dir)
    result_path = task_dir / RESULT_NAME
    if task_dir.is_symlink() and not result_path.exists():
        return run_dir / "results_mirror" / f"{index:04d}.json"
    return result_path


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _attempt_number(document):
    return progress_counter(document, 'attempt')


def _cleanup_finished_inputs(output_dir):
    from .storage import cleanup_task_solver_inputs, keep_solver_inputs

    try:
        if keep_solver_inputs(os.environ.get("VB_KEEP_SOLVER_INPUTS", "")):
            return
        stats = cleanup_task_solver_inputs(output_dir)
        if stats["removed_files"]:
            print(f"removed {stats['removed_files']} generated SDP input files; "
                  "retained exact certification samples")
    except Exception as exc:
        # Housekeeping must not turn a completed physics result into a retry.
        print(f"warning: solver-input cleanup failed: {exc}", file=sys.stderr)


#: errors a shared cluster filesystem can return for a moment (quota or space briefly exhausted by other jobs)
_TRANSIENT_WRITE_ERRNOS = frozenset({errno.EDQUOT, errno.ENOSPC, errno.EIO})
_WRITE_RETRY_DELAYS = (5, 15, 30, 60, 120, 240, 300, 300)  # seconds: about 18 minutes in all


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``path`` atomically; ride out a transient quota/space error rather than lose a finished task."""
    text = json.dumps(payload, indent=2)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    for delay in (*_WRITE_RETRY_DELAYS, None):
        try:
            tmp.write_text(text)
            tmp.replace(path)
            return
        except OSError as exc:
            if delay is None or exc.errno not in _TRANSIENT_WRITE_ERRNOS:
                raise
            print(f"warning: writing {path.name} failed ({exc.strerror}); retrying in {delay} s", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            time.sleep(delay)


def _slurm_task_id() -> str | None:
    """The id `scontrol requeue` wants: array element if in an array, else job."""
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"{array_job}_{array_task}"
    return os.environ.get("SLURM_JOB_ID")


def _requeue_self() -> bool:
    """Requeue this SLURM (array) task; False when off-SLURM or on failure."""
    task_id = _slurm_task_id()
    if task_id is None:
        return False
    try:
        subprocess.run(
            ["scontrol", "requeue", task_id], check=True, capture_output=True, text=True
        )
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warning: scontrol requeue {task_id} failed: {exc}", file=sys.stderr)
        return False


def load_spec(spec_path: Path) -> tuple[str, dict, Path]:
    """Parse a job spec, returning (task_name, params, output_dir).

    A relative ``output_dir`` resolves against the *run* directory (the spec
    file's parent's parent: specs live in ``runs/<id>/specs/``), so a whole
    run directory can be moved or rsynced without breaking its specs.
    """
    spec = read_job_spec(spec_path)
    task_name = spec["task"]
    params = spec.get("params", {})
    out = Path(spec["output_dir"])
    if not out.is_absolute():
        out = spec_path.resolve().parent.parent / out

    # Bulky per-solve intermediates (sdp dirs, blocks, sdpb_out) dwarf the
    # results: measured at Lambda=19, one task is ~670 MB of which result.json
    # is 100 KB. Those belong on scratch, not on the smaller project volume.
    # Redirect the whole task dir to $VB_SCRATCH_ROOT and leave a symlink
    # behind, so every existing `output_dir / key / ...` path -- certification,
    # checkpoint reuse, collection -- keeps resolving unchanged.
    scratch = os.environ.get("VB_SCRATCH_ROOT")
    if scratch:
        run_id = spec_path.resolve().parent.parent.name
        target = (Path(os.path.expandvars(scratch)).expanduser() / run_id / spec_path.stem).resolve()
        if out.is_symlink():
            # Scratch can purge the target while leaving our durable link.
            # Restore only the target this mapping owns, including when its
            # empty directory was already recreated. Unrelated dangling
            # output links must still be diagnosed as bad specs.
            if out.resolve() == target and not (out / RESULT_NAME).exists():
                target.mkdir(parents=True, exist_ok=True)
                try:
                    with _task_output_lock(out):
                        mirror = spec_path.resolve().parent.parent / "results_mirror" / spec_path.name
                        try:
                            saved = json.loads(mirror.read_text())
                        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
                            saved = None
                        if not (out / RESULT_NAME).exists() and result_doc_is_final(
                                saved, spec_hash(task_name, params),
                                execution=execution_settings(task_name, params)):
                            _atomic_write_json(out / RESULT_NAME, saved)
                except _TaskBusy:
                    # Recovery must not publish an old mirror over a live task.
                    pass
            return task_name, params, out
        target.mkdir(parents=True, exist_ok=True)
        if out.is_dir():
            # A real directory from an earlier un-redirected attempt: keep it
            # rather than silently stranding partial work.
            return task_name, params, out
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            # Another worker may have just installed this same mapping.
            if not out.is_symlink() or out.resolve() != target:
                raise
    return task_name, params, out


def _open_journal(path: Path, digest: str, execution: dict | None = None) -> Journal:
    """The task's progress journal, reset if it belongs to a different spec.

    The journal holds probe verdicts and checkpoints for one spec. A journal
    with another spec's hash is moved to ``progress.<old-hash>.json`` and a
    fresh one is started. A journal recorded with other task defaults
    (profile or SDPB flags) is archived the same way. A
    journal of another spin-0 scalar_G convention is moved to
    ``progress.scalar_G_spin0-<convention>.json`` the same way, so a resume
    never mixes the two normalizations.  A journal that cannot be set aside
    is refused rather than overwritten.
    """
    journal = Journal(path)
    stored = journal.data.get("spec_hash")
    if stored is not None and stored != digest:
        attempt = _attempt_number(journal.data)
        stale = path.with_name(f"{path.stem}.{stored}{path.suffix}")
        try:
            path.replace(stale)
        except OSError as exc:
            raise RuntimeError(f"{path}: cannot set aside progress for a different spec: {exc}") from exc
        print(f"{path}: journal is for spec {stored}, not {digest}; starting afresh")
        journal = Journal(path)
        # Starts reset for the new spec; the ordinal still orders attempts
        # across edits, including a later return to the original inputs.
        journal.data = {'attempt': attempt} if attempt else {}
    # The saved verdicts, OPE values, and checkpoints were computed in one spin-0 scalar_G normalization; resuming
    # them in another would mix two scales in one result (only the worker's own counters carry no scale).
    from virasoro_blocks._conventions import SCALAR_G_SPIN0_DOUBLE, SCALAR_G_SPIN0_KEY, scalar_G_spin0_convention

    active = scalar_G_spin0_convention()
    if execution is not None:
        execution_hash = settings_digest(execution)
        saved_work = set(journal.data) - {"spec_hash", "starts", "attempt", SCALAR_G_SPIN0_KEY}
        if saved_work and not execution_matches(journal.data.get("execution_settings"), execution,
                                               resolved_settings=journal.data.get("resolved_settings")):
            attempt = _attempt_number(journal.data)
            previous = journal.data.get("execution_hash", "unrecorded")
            stale = path.with_name(f"{path.stem}.execution-{previous}-{attempt}{path.suffix}")
            try:
                path.replace(stale)
            except OSError as exc:
                raise RuntimeError(f"{path}: cannot set aside progress with different task defaults: {exc}") from exc
            print(f"{path}: task defaults differ; saved {stale.name}, starting afresh")
            journal = Journal(path)
            journal.data = {'attempt': attempt} if attempt else {}
    saved_work = set(journal.data) - {"spec_hash", "starts", "attempt"}
    recorded = journal.data.get(SCALAR_G_SPIN0_KEY, SCALAR_G_SPIN0_DOUBLE if saved_work else None)
    if recorded is not None and recorded != active:
        attempt = _attempt_number(journal.data)
        stale = path.with_name(f"{path.stem}.{SCALAR_G_SPIN0_KEY}-{recorded}{path.suffix}")
        try:
            path.replace(stale)
        except OSError as exc:
            raise RuntimeError(f"{path}: cannot set aside a journal of scalar_G_spin0 = {recorded!r} "
                               f"before starting afresh: {exc}") from exc
        print(f"{path}: journal was written with scalar_G_spin0 = {recorded!r}, but this worker uses {active!r}; "
              "not resuming its saved work, starting afresh")
        journal = Journal(path)
        journal.data = {'attempt': attempt} if attempt else {}
    if journal.data.get("spec_hash") != digest or journal.data.get(SCALAR_G_SPIN0_KEY) != active:
        journal.update(spec_hash=digest, **{SCALAR_G_SPIN0_KEY: active})
    if execution is not None and journal.data.get("execution_settings") != execution:
        journal.update(execution_hash=execution_hash, execution_settings=execution)
    return journal


def run_spec(spec_path: Path | str) -> int:
    spec_path = Path(spec_path)
    try:
        task_name, params, output_dir = load_spec(spec_path)
        task_fn = get_task(task_name)
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError) as exc:
        print(f"bad spec {spec_path}: {exc}", file=sys.stderr)
        return 2

    try:
        with _task_output_lock(output_dir):
            return _run_spec_locked(spec_path, task_name, params, output_dir, task_fn)
    except _TaskBusy as exc:
        requeued = _requeue_self()
        print(f"{spec_path.name}: {exc}; requeued={requeued}")
        return 3
    except _TaskLockError as exc:
        # A completed run can be copied onto a read-only filesystem. Pure
        # result reuse needs no lock; repairs and cleanup still require one.
        try:
            saved = json.loads((output_dir / RESULT_NAME).read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            saved = None
        if result_doc_is_final(saved, spec_hash(task_name, params),
                               execution=execution_settings(task_name, params)):
            print(f"{spec_path.name}: result present, skipping without housekeeping ({exc})")
            return 0
        print(f"{spec_path.name}: cannot run without the task lock: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{spec_path.name}: cannot initialize or finalize task: {exc}", file=sys.stderr)
        return 1


def _run_spec_locked(spec_path, task_name, params, output_dir, task_fn):
    digest = spec_hash(task_name, params)
    execution = execution_settings(task_name, params)
    execution_hash = settings_digest(execution)
    result_path = output_dir / RESULT_NAME
    status_path = output_dir / STATUS_NAME

    existing = None
    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            existing = None
        if result_doc_is_final(existing, digest, execution=execution):
            _repair_final_status(status_path, spec_path, existing, task_name=task_name)
            _mirror_result(spec_path, output_dir, result_path)
            _cleanup_finished_inputs(output_dir)
            print(f"{spec_path.name}: result present, skipping")
            return 0

    journal = _open_journal(output_dir / JOURNAL_NAME, digest, execution)
    starts = progress_counter(journal.data, "starts") + 1
    try:
        prior_status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        prior_status = None
    attempt = 1 + max(_attempt_number(document)
                      for document in (journal.data, existing, prior_status))
    journal.update(starts=starts, attempt=attempt)

    stop = threading.Event()

    def on_signal(signum, _frame):
        print(f"received {signal.Signals(signum).name}; stopping at next safe point")
        stop.set()

    old_handlers = {
        sig: signal.signal(sig, on_signal) for sig in (signal.SIGUSR1, signal.SIGTERM)
    }

    started = _now()
    base_status = {
        "task": task_name,
        "spec": spec_path.name,
        "spec_hash": digest,
        "execution_hash": execution_hash,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_task": _slurm_task_id(),
        "starts": starts,
        "attempt": attempt,
        "started_at": started,
    }
    try:
        _atomic_write_json(status_path, {**base_status, "state": "running"})
        ctx = TaskContext(
            params=params, output_dir=output_dir, journal=journal, should_stop=stop.is_set,
            execution=execution
        )
        result = task_fn(ctx)
        failure = result_error(result)
        from virasoro_blocks._conventions import SCALAR_G_SPIN0_KEY, scalar_G_spin0_convention

        doc = {"spec_hash": digest, "task": task_name, "params": params,
               "execution_hash": execution_hash, "execution_settings": execution,
               SCALAR_G_SPIN0_KEY: scalar_G_spin0_convention(), "starts": starts, "attempt": attempt,
               "started_at": started, "completed_at": _now(), "result": result}
        if failure is not None:
            # Keep the payload (its logs are the diagnosis) but mark it failed,
            # so neither worker reuse nor submission retry takes it as done.
            doc["failed"] = failure
        # Encoding and publication can fail too. Record those failures just
        # like task exceptions, rather than leaving a dead worker "running".
        _atomic_write_json(result_path, doc)
    except TaskInterrupted as exc:
        requeued = _requeue_self()
        _atomic_write_json(
            status_path,
            {**base_status, "state": "interrupted", "detail": str(exc),
             "requeued": requeued, "ended_at": _now()},
        )
        print(f"{spec_path.name}: interrupted ({exc}); requeued={requeued}")
        return 3
    except Exception as exc:
        _atomic_write_json(
            status_path,
            {**base_status, "state": "failed",
             "detail": f"{exc.__class__.__name__}: {exc}",
             "traceback": traceback.format_exc(), "ended_at": _now()},
        )
        print(f"{spec_path.name}: FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    if failure is not None:
        _atomic_write_json(
            status_path,
            {**base_status, "state": "failed", "detail": failure, "ended_at": _now()},
        )
        _mirror_result(spec_path, output_dir, result_path)
        print(f"{spec_path.name}: FAILED: {failure}", file=sys.stderr)
        return 1
    _atomic_write_json(status_path, {**base_status, "state": "done", "ended_at": _now()})
    _mirror_result(spec_path, output_dir, result_path)
    _cleanup_finished_inputs(output_dir)
    print(f"{spec_path.name}: done")
    return 0


def _repair_final_status(status_path: Path, spec_path: Path, result: dict, *, task_name: str) -> None:
    """Finish metadata publication after a kill between result and status writes."""
    try:
        try:
            status = json.loads(status_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            status = {}
        if not isinstance(status, dict) or status.get("spec_hash") != result["spec_hash"]:
            status = {}
        if status.get("state") == "done":
            return
        for name in ("detail", "traceback", "requeued"):
            status.pop(name, None)
        status.update(task=task_name, spec=spec_path.name,
                      spec_hash=result["spec_hash"], state="done",
                      ended_at=result.get("completed_at") or _now())
        if result.get("started_at") is not None:
            status["started_at"] = result["started_at"]
        for name in ('starts', 'attempt'):
            if type(result.get(name)) is int and result[name] >= 0:
                status[name] = result[name]
        _atomic_write_json(status_path, status)
    except OSError as exc:
        print(f"{spec_path.name}: final status repair failed: {exc}", file=sys.stderr)


def _mirror_result(spec_path: Path, output_dir: Path, result_path: Path) -> None:
    """Copy the (small) result back to the run dir when the task dir is on scratch.

    With VB_SCRATCH_ROOT set, ``output_dir`` is a symlink onto scratch -- which
    is typically purged on a fixed schedule. The intermediates are
    disposable but the result is not, so keep a durable copy beside the specs.
    """
    if not output_dir.is_symlink():
        return
    temporary = None
    try:
        mirror = spec_path.resolve().parent.parent / "results_mirror"
        mirror.mkdir(parents=True, exist_ok=True)
        destination = mirror / spec_path.name
        temporary = destination.with_suffix(destination.suffix + f".tmp{os.getpid()}")
        shutil.copy2(result_path, temporary)
        temporary.replace(destination)
    except OSError as exc:  # never fail a completed task over the mirror
        print(f"{spec_path.name}: result mirror failed: {exc}", file=sys.stderr)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m virasoro_bootstrap.hpc.worker", description=__doc__
    )
    parser.add_argument("spec", type=Path, help="job-spec JSON file")
    args = parser.parse_args(argv)
    return run_spec(args.spec)


if __name__ == "__main__":
    sys.exit(main())
