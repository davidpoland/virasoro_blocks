"""Worker/submit/collect layer: specs, idempotence, journal resume, sbatch render.

Uses throwaway registered tasks -- no blocks are built and no SDPB runs, so
these are fast pure-plumbing tests.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from virasoro_bootstrap.hpc import collect, submit, worker
from virasoro_bootstrap.hpc.journal import Journal
from virasoro_bootstrap.hpc.tasks import TASKS, TaskContext, TaskInterrupted, task

pytestmark = pytest.mark.usefixtures("_clean_fake_tasks")

CALLS: list[str] = []


@pytest.fixture
def _clean_fake_tasks():
    """Register the fake tasks for the duration of each test."""
    added = []

    def add(name, fn):
        TASKS[name] = fn
        added.append(name)

    add("fake_echo", _fake_echo)
    add("fake_probe_chain", _fake_probe_chain)
    add("fake_interrupt_once", _fake_interrupt_once)
    CALLS.clear()
    yield
    for name in added:
        TASKS.pop(name, None)


def _fake_echo(ctx: TaskContext) -> dict:
    CALLS.append(f"echo:{ctx.params.get('c')}")
    return {"c": ctx.params.get("c"), "value": 42}


def _fake_probe_chain(ctx: TaskContext) -> dict:
    """Three 'probes', each journaled; counts real solves in CALLS."""
    probes = dict(ctx.journal.data.get("probes", {}))
    for key in ("p1", "p2", "p3"):
        if key in probes:
            continue
        CALLS.append(f"solve:{key}")
        probes[key] = {"allowed": True}
        ctx.journal.update(probes=probes)
    return {"n_probes": len(probes)}


def _fake_interrupt_once(ctx: TaskContext) -> dict:
    """Interrupt on the first run (after journaling one probe); finish on the second."""
    probes = dict(ctx.journal.data.get("probes", {}))
    if "p1" not in probes:
        probes["p1"] = {"allowed": True}
        ctx.journal.update(probes=probes)
        raise TaskInterrupted("simulated pre-wall stop")
    return {"resumed": True}


def _write_spec(run_dir: Path, index: int, task_name: str, params: dict) -> Path:
    (run_dir / "specs").mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "specs" / f"{index:04d}.json"
    spec_path.write_text(json.dumps(
        {"task": task_name, "params": params, "output_dir": f"tasks/{index:04d}"}
    ))
    return spec_path


# --- worker ------------------------------------------------------------------

def test_worker_runs_and_writes_result(tmp_path):
    spec = _write_spec(tmp_path, 0, "fake_echo", {"c": "8"})
    assert worker.run_spec(spec) == 0
    doc = json.loads((tmp_path / "tasks" / "0000" / "result.json").read_text())
    assert doc["result"] == {"c": "8", "value": 42}
    assert doc["spec_hash"] == worker.spec_hash("fake_echo", {"c": "8"})
    status = json.loads((tmp_path / "tasks" / "0000" / "status.json").read_text())
    assert status["state"] == "done"


def _register_feasibility_defaults(monkeypatch, *names):
    from virasoro_bootstrap.hpc import tasks
    for name in names:
        monkeypatch.setitem(tasks.TASK_SDPB_DEFAULTS, name, 'DEFAULT_FEASIBILITY_SDPB_OPTIONS')


def _change_defaults(monkeypatch, change):
    from virasoro_bootstrap import defaults
    from virasoro_bootstrap.hpc import tasks

    if change in ('profile', 'other_profile', 'comment'):
        config = defaults.load_default_params()
        if change == 'comment':
            config['lambda19']['_comment'] = 'Edited explanation.'
        else:
            config['lambda19' if change == 'profile' else 'lambda7']['q_order'] += 2
        monkeypatch.setattr(defaults, 'load_default_params', lambda: config)
    else:
        name = 'DEFAULT_FEASIBILITY_SDPB_OPTIONS' if change == 'solver' else 'DEFAULT_OPE_MAX_SDPB_OPTIONS'
        monkeypatch.setattr(tasks, name, {**getattr(tasks, name), '--precision': 1024})


@pytest.mark.parametrize('change', ['profile', 'solver'])
def test_changed_task_defaults_recompute_saved_work(tmp_path, monkeypatch, change):
    _register_feasibility_defaults(monkeypatch, 'fake_probe_chain')
    params = {'lam': 19}
    spec = _write_spec(tmp_path, 0, 'fake_probe_chain', params)
    assert worker.run_spec(spec) == 0
    out = tmp_path / 'tasks/0000'
    saved = json.loads((out / 'result.json').read_text())
    _change_defaults(monkeypatch, change)
    # Historical results stay readable, but cannot satisfy a new execution.
    assert worker.result_doc_is_final(saved, worker.spec_hash('fake_probe_chain', params))
    assert submit._pending_indices(tmp_path, 1) == [0]
    assert worker.run_spec(spec) == 0
    assert CALLS == ['solve:p1', 'solve:p2', 'solve:p3'] * 2
    assert list(out.glob('progress.execution-*.json'))
    assert submit._pending_indices(tmp_path, 1) == []
    assert worker.run_spec(spec) == 0
    assert len(CALLS) == 6


@pytest.mark.parametrize('change', ['other_profile', 'other_solver', 'comment', 'legacy'])
def test_unrelated_changes_keep_saved_work(tmp_path, monkeypatch, change):
    _register_feasibility_defaults(monkeypatch, 'fake_probe_chain')
    spec = _write_spec(tmp_path, 0, 'fake_probe_chain', {'lam': 19})
    assert worker.run_spec(spec) == 0
    out = tmp_path / 'tasks/0000'
    if change == 'legacy':
        # Work saved before task defaults were recorded is adopted.
        for name in ('progress.json', 'result.json'):
            doc = json.loads((out / name).read_text())
            doc.pop('execution_hash')
            doc.pop('execution_settings')
            (out / name).write_text(json.dumps(doc))
    else:
        _change_defaults(monkeypatch, change)
    assert submit._pending_indices(tmp_path, 1) == []
    assert worker.run_spec(spec) == 0
    assert CALLS == ['solve:p1', 'solve:p2', 'solve:p3']


def test_changed_defaults_do_not_resume_a_partial_chain(tmp_path, monkeypatch):
    _register_feasibility_defaults(monkeypatch, 'fake_interrupt_once')
    spec = _write_spec(tmp_path, 0, 'fake_interrupt_once', {})
    monkeypatch.setattr(worker, '_requeue_self', lambda: False)
    assert worker.run_spec(spec) == 3
    _change_defaults(monkeypatch, 'solver')
    # A fresh first probe interrupts again, rather than reusing the old one.
    assert worker.run_spec(spec) == 3
    assert worker.run_spec(spec) == 0


def test_task_defaults_name_the_flags_each_task_uses():
    from virasoro_bootstrap.hpc import tasks
    real = set(tasks.TASKS) - {'fake_echo', 'fake_probe_chain', 'fake_interrupt_once'}
    assert set(tasks.TASK_SDPB_DEFAULTS) == real - {'moment_recertify'}
    for name, constant in tasks.TASK_SDPB_DEFAULTS.items():
        assert isinstance(getattr(tasks, constant), dict), name


def test_failed_progress_archive_preserves_previous_work(tmp_path, monkeypatch):
    spec = _write_spec(tmp_path, 0, 'fake_probe_chain', {'lam': 19})
    assert worker.run_spec(spec) == 0
    progress = tmp_path / 'tasks/0000/progress.json'
    previous = progress.read_bytes()
    _change_defaults(monkeypatch, 'profile')
    replace = Path.replace
    def refuse_archive(path, target):
        if path == progress:
            raise PermissionError('archive is read-only')
        return replace(path, target)
    monkeypatch.setattr(Path, 'replace', refuse_archive)
    assert worker.run_spec(spec) == 1
    assert progress.read_bytes() == previous
    assert len(CALLS) == 3


def test_worker_is_idempotent(tmp_path, monkeypatch):
    spec = _write_spec(tmp_path, 0, "fake_echo", {"c": "8"})
    assert worker.run_spec(spec) == 0
    assert worker.run_spec(spec) == 0
    assert CALLS == ["echo:8"]  # second run skipped, no re-execution
    original = os.open
    def read_only_lock(path, *args, **kwargs):
        if Path(path).name == '.worker.lock':
            raise PermissionError('read-only completed output')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, 'open', read_only_lock)
    assert worker.run_spec(spec) == 0
    assert CALLS == ['echo:8']


def test_unsupported_lock_names_the_lock_file_and_errno(tmp_path, monkeypatch, capsys):
    import errno
    import fcntl

    spec = _write_spec(tmp_path, 0, "fake_echo", {"c": "8"})
    def no_lockd(*args):
        raise OSError(errno.ENOLCK, "No locks available")
    monkeypatch.setattr(fcntl, "flock", no_lockd)
    assert worker.run_spec(spec) == 2
    err = capsys.readouterr().err
    assert ".worker.lock" in err and "ENOLCK" in err and "bad spec" not in err
    assert CALLS == []


def test_busy_worker_preserves_live_state_and_scratch_mirror_until_lock_release(tmp_path, monkeypatch):
    import fcntl

    requeues = []
    cleanups = []
    monkeypatch.setattr(worker, '_requeue_self', lambda: requeues.append(True) or False)
    monkeypatch.setattr(worker, '_cleanup_finished_inputs', lambda out: cleanups.append(out))
    for scratch in (False, True):
        for cached in (False, True):
            CALLS.clear()
            run = tmp_path / f'run-{scratch}-{cached}'
            if scratch:
                monkeypatch.setenv('VB_SCRATCH_ROOT', str(tmp_path / 'scratch'))
            else:
                monkeypatch.delenv('VB_SCRATCH_ROOT', raising=False)
            spec = _write_spec(run, 0, 'fake_echo', {'c': '8'})
            _, _, out = worker.load_spec(spec)
            out.mkdir(parents=True, exist_ok=True)
            digest = worker.spec_hash('fake_echo', {'c': '8'})
            saved = {'spec_hash': digest, 'result': {'c': '8', 'value': 42},
                     'execution_settings': worker.execution_settings('fake_echo', {'c': '8'})}
            (out / 'status.json').write_text(json.dumps({'state': 'running', 'spec_hash': digest}))
            (out / 'progress.json').write_text(json.dumps({'starts': 4, 'spec_hash': digest}))
            if cached:
                (out / 'result.json').write_text(json.dumps(saved))
            if scratch:
                mirror = run / 'results_mirror/0000.json'
                mirror.parent.mkdir()
                mirror.write_text(json.dumps(saved))
            paths = [out / name for name in ('status.json', 'progress.json', 'result.json')]
            before = {path: path.read_bytes() if path.exists() else None for path in paths}
            cleanup_count, requeue_count = len(cleanups), len(requeues)
            with (out / '.worker.lock').open('a+b') as anchor:
                fcntl.flock(anchor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                assert worker.run_spec(spec) == 3
                assert not CALLS and len(cleanups) == cleanup_count
                assert len(requeues) == requeue_count + 1
                assert before == {path: path.read_bytes() if path.exists() else None for path in paths}
            assert worker.run_spec(spec) == 0
            assert CALLS == (['echo:8'] if not scratch and not cached else [])
            assert json.loads((out / 'status.json').read_text())['state'] == 'done'


def test_completed_worker_repairs_status_and_scratch_mirror_without_rerunning(tmp_path, monkeypatch):
    monkeypatch.setenv("VB_SCRATCH_ROOT", str(tmp_path / "scratch"))
    spec = _write_spec(tmp_path / "run", 0, "fake_echo", {"c": "8"})
    assert worker.run_spec(spec) == 0
    output = spec.parent.parent / "tasks/0000"
    result = (output / "result.json").read_bytes()
    mirror = spec.parent.parent / "results_mirror/0000.json"
    status_path = output / "status.json"
    original = json.loads(status_path.read_text())
    # A kill can occur after publishing result.json, or a temporary mirror
    # failure can leave the completed result on scratch only.
    for status in ({**original, "state": "running"},
                   {**original, "state": "failed", "detail": "old error", "traceback": "old trace"},
                   {"state": "running", "spec_hash": "old"}, [], None):
        mirror.unlink()
        if status is None:
            status_path.unlink()
        else:
            status_path.write_text(json.dumps(status))
        assert worker.run_spec(spec) == 0
        repaired = json.loads(status_path.read_text())
        assert repaired["state"] == "done" and repaired["spec_hash"] == original["spec_hash"]
        assert repaired["task"] == "fake_echo"
        assert "detail" not in repaired and "traceback" not in repaired
        assert mirror.read_bytes() == result == (output / "result.json").read_bytes()
    assert CALLS == ["echo:8"]


def test_worker_recreates_only_its_owned_purged_scratch_target(tmp_path, monkeypatch):
    import shutil

    scratch = tmp_path / "scratch"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VB_SCRATCH_ROOT", "scratch")
    spec = _write_spec(tmp_path / "run", 0, "fake_echo", {})
    _, _, output = worker.load_spec(spec)
    assert output.is_dir() and output.resolve() == scratch / "run/0000"
    target = output.resolve()
    original_link = output.readlink()
    shutil.rmtree(target)
    assert output.is_symlink() and not output.exists()
    _, _, restored = worker.load_spec(spec)
    assert restored == output and restored.is_dir() and restored.readlink() == original_link

    # An unrelated user-created broken link is not adopted as scratch.
    output.unlink()
    unrelated = tmp_path / "missing_user_directory"
    output.symlink_to(unrelated, target_is_directory=True)
    assert worker.run_spec(spec) == 2
    assert output.is_symlink() and not unrelated.exists()

    # The same mapping installed by a racing worker is already valid.
    raced = _write_spec(tmp_path / 'run', 1, 'fake_echo', {})
    symlink = Path.symlink_to
    def installed_by_other_worker(path, target, **kwargs):
        symlink(path, target, **kwargs)
        raise FileExistsError('mapping just installed by another worker')
    monkeypatch.setattr(Path, 'symlink_to', installed_by_other_worker)
    assert worker.run_spec(raced) == 0
    assert (raced.parent.parent / 'tasks/0001').resolve() == scratch / 'run/0001'


def test_failed_result_mirror_publication_preserves_previous_copy(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setenv("VB_SCRATCH_ROOT", str(tmp_path / "scratch"))
    spec = _write_spec(tmp_path / "run", 0, "fake_echo", {})
    assert worker.run_spec(spec) == 0
    output = spec.parent.parent / "tasks/0000"
    mirror = spec.parent.parent / "results_mirror/0000.json"
    saved = mirror.read_bytes()
    (output / "result.json").write_text('{"new":"complete result"}')

    def fail_copy(source, destination):
        Path(destination).write_text('{"partial":')
        raise OSError("interrupted mirror copy")

    monkeypatch.setattr(shutil, "copy2", fail_copy)
    worker._mirror_result(spec, output, output / "result.json")
    assert mirror.read_bytes() == saved
    assert list(mirror.parent.iterdir()) == [mirror]


def test_purged_scratch_uses_current_mirrored_results_and_retries_stale_or_failed_ones(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setenv("VB_SCRATCH_ROOT", str(tmp_path / "scratch"))
    run = tmp_path / "run"
    spec = _write_spec(run, 0, "fake_echo", {"c": "8"})
    (run / "manifest.json").write_text(json.dumps({"run_id": "run", "n_tasks": 1}))
    assert worker.run_spec(spec) == 0
    output = run / "tasks/0000"
    original = (output / "result.json").read_bytes()
    for recreate_empty in (False, True):
        target = output.resolve()
        shutil.rmtree(target)
        if recreate_empty:
            target.mkdir()
        assert collect.load_run(run)[1][0].state == "done"
        assert submit._pending_indices(run, 1) == []
        assert worker.run_spec(spec) == 0
        assert (output / "result.json").read_bytes() == original
        assert json.loads((output / "status.json").read_text())["state"] == "done"
        assert CALLS == ["echo:8"]

    target = output.resolve()
    shutil.rmtree(target)
    target.mkdir()
    _write_spec(run, 0, "fake_echo", {"c": "9"})
    assert collect.load_run(run)[1][0].state == "stale"
    assert submit._pending_indices(run, 1) == [0]
    assert worker.run_spec(spec) == 0
    assert CALLS == ["echo:8", "echo:9"]

    mirror = run / "results_mirror/0000.json"
    saved = json.loads(mirror.read_text())
    for payload in (json.dumps({**saved, "result": {"reason": "solve_error"}}), '{"partial":'):
        shutil.rmtree(output.resolve())
        mirror.write_text(payload)
        assert collect.load_run(run)[1][0].state in {"failed", "pending"}
        assert submit._pending_indices(run, 1) == [0]
        assert worker.run_spec(spec) == 0
    assert CALLS == ["echo:8", "echo:9", "echo:9", "echo:9"]


def test_custom_output_paths_with_shared_names_remain_independent_on_scratch(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setenv("VB_SCRATCH_ROOT", str(tmp_path / "scratch"))
    run = tmp_path / "run"
    outputs = []
    for index, charge in enumerate(("8", "9")):
        spec = _write_spec(run, index, "fake_echo", {"c": charge})
        document = json.loads(spec.read_text())
        output = run / f"tasks/group{index}/shared_name"
        document["output_dir"] = (str(output) if index else "tasks/group0/shared_name")
        spec.write_text(json.dumps(document))
        assert worker.run_spec(spec) == 0
        outputs.append(output)
    (run / "manifest.json").write_text(json.dumps({"run_id": "run", "n_tasks": 2}))
    assert outputs[0].resolve() != outputs[1].resolve()
    for output, charge in zip(outputs, ("8", "9")):
        assert json.loads((output / "result.json").read_text())["result"]["c"] == charge
    assert [view.result["c"] for view in collect.load_run(run)[1]] == ["8", "9"]
    assert submit._pending_indices(run, 2) == []

    for output in outputs:
        shutil.rmtree(output.resolve())
    assert [view.result["c"] for view in collect.load_run(run)[1]] == ["8", "9"]
    assert submit._pending_indices(run, 2) == []
    for index in range(2):
        assert worker.run_spec(run / "specs" / f"{index:04d}.json") == 0
    assert CALLS == ["echo:8", "echo:9"]


@pytest.mark.parametrize("payload", [b"null", b"\xff"])
def test_corrupt_result_remains_retryable(tmp_path, payload):
    spec = _write_spec(tmp_path, 0, "fake_echo", {"c": "8"})
    task_dir = tmp_path / "tasks" / "0000"
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_bytes(payload)
    assert submit._pending_indices(tmp_path, 1) == [0]
    assert worker.run_spec(spec) == 0
    assert CALLS == ["echo:8"]


@pytest.mark.parametrize("result", [None, "done"])
def test_invalid_task_result_is_failed_and_retryable(tmp_path, monkeypatch, result):
    spec = _write_spec(tmp_path, 0, "fake_echo", {})
    monkeypatch.setitem(TASKS, "fake_echo", lambda ctx: result)
    assert worker.run_spec(spec) == 1
    assert submit._pending_indices(tmp_path, 1) == [0]
    doc = json.loads((tmp_path / "tasks" / "0000" / "result.json").read_text())
    assert "malformed" in doc["failed"]
    (tmp_path / 'manifest.json').write_text(json.dumps({'n_tasks': 1}))
    original = (tmp_path / 'tasks/0000/result.json').read_bytes()
    view = collect.load_run(tmp_path)[1][0]
    assert view.state == 'failed' and isinstance(view.result, dict)
    assert view.result['reason'] == 'invalid_result' and view.result['raw_result'] == result
    assert collect.collect_main([str(tmp_path)]) == 0
    assert json.loads((tmp_path / 'results.json').read_text()) == [view.result]
    assert (tmp_path / 'tasks/0000/result.json').read_bytes() == original


def test_matching_spec_hash_without_result_is_not_complete():
    assert not worker.result_doc_is_final({"spec_hash": "abc"}, "abc")


def test_worker_reruns_when_params_change(tmp_path):
    spec = _write_spec(tmp_path, 0, "fake_echo", {"c": "8"})
    assert worker.run_spec(spec) == 0
    _write_spec(tmp_path, 0, "fake_echo", {"c": "9"})
    assert worker.run_spec(spec) == 0
    assert CALLS == ["echo:8", "echo:9"]


def test_worker_journal_resume_skips_done_probes(tmp_path):
    spec = _write_spec(tmp_path, 0, "fake_probe_chain", {})
    assert worker.run_spec(spec) == 0
    assert CALLS == ["solve:p1", "solve:p2", "solve:p3"]
    # Wipe the result but keep the journal: rerun must not re-solve anything.
    (tmp_path / "tasks" / "0000" / "result.json").unlink()
    CALLS.clear()
    assert worker.run_spec(spec) == 0
    assert CALLS == []


def test_worker_interrupted_then_resumes(tmp_path, monkeypatch):
    # Off-SLURM: requeue is a no-op returning False; exit code is still 3.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    spec = _write_spec(tmp_path, 0, "fake_interrupt_once", {})
    assert worker.run_spec(spec) == 3
    status = json.loads((tmp_path / "tasks" / "0000" / "status.json").read_text())
    assert status["state"] == "interrupted"
    assert status["requeued"] is False
    # Second run picks the journal up and completes.
    assert worker.run_spec(spec) == 0
    doc = json.loads((tmp_path / "tasks" / "0000" / "result.json").read_text())
    assert doc["result"] == {"resumed": True}


def test_worker_failure_writes_status(tmp_path, monkeypatch):
    import signal

    def boom(ctx):
        raise RuntimeError('boom')
    cyclic = {}
    cyclic['cycle'] = cyclic
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        spec = _write_spec(tmp_path, 0, "fake_boom", {})
        result_path = tmp_path / 'tasks/0000/result.json'
        result_path.parent.mkdir(parents=True)
        previous = json.dumps({'spec_hash': 'old-spec', 'result': {'previous': True}}).encode()
        result_path.write_bytes(previous)
        for fn, error in ((boom, 'boom'),
                          (lambda ctx: {'bad': object()}, 'not JSON serializable'),
                          (lambda ctx: cyclic, 'Circular reference')):
            TASKS['fake_boom'] = fn
            assert worker.run_spec(spec) == 1
            status = json.loads((tmp_path / "tasks" / "0000" / "status.json").read_text())
            assert status["state"] == "failed"
            assert error in status["detail"]
            assert 'traceback' in status
            assert result_path.read_bytes() == previous
            assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
        TASKS['fake_boom'] = lambda ctx: {'recovered': True}
        original = Path.replace
        def fail_publication(path, target):
            if Path(target) == result_path:
                raise OSError('result publication failed')
            return original(path, target)
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'replace', fail_publication)
            assert worker.run_spec(spec) == 1
        status = json.loads((result_path.parent / 'status.json').read_text())
        assert status['state'] == 'failed' and 'result publication failed' in status['detail']
        assert result_path.read_bytes() == previous
        assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
        assert worker.run_spec(spec) == 0  # failure released its output lock
    finally:
        TASKS.pop("fake_boom", None)


def test_worker_bad_spec(tmp_path):
    bad = tmp_path / "specs" / "0000.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps({"task": "no_such_task", "params": {}, "output_dir": "t"}))
    assert worker.run_spec(bad) == 2


# --- journal -----------------------------------------------------------------

def test_journal_atomic_and_corruption_tolerant(tmp_path):
    j = Journal(tmp_path / "progress.json")
    j.update(starts=1, probes={"a": 1})
    j2 = Journal(tmp_path / "progress.json")
    assert j2.data == {"starts": 1, "probes": {"a": 1}}
    (tmp_path / "progress.json").write_text("{corrupt")
    assert Journal(tmp_path / "progress.json").data == {}


# --- submit ------------------------------------------------------------------

def _write_sweep(tmp_path: Path, task_name="fake_echo") -> Path:
    sweep = tmp_path / "sweep.toml"
    sweep.write_text(
        'name = "unit"\n'
        f'task = "{task_name}"\n'
        "[resources]\ncpus_per_task = 4\ntime = \"01:00:00\"\n"
        "[params]\nlam = 7\n"
        "[sweep]\nc = [\"2\", \"4\", \"8\"]\n"
    )
    return sweep


def _write_cluster(tmp_path: Path, kind="slurm") -> Path:
    profile = tmp_path / "cluster.toml"
    profile.write_text(
        f'kind = "{kind}"\n'
        'partition = "day"\n'
        "max_array_throttle = 16\n"
        'modules = ["miniconda", "apptainer"]\n'
        'setup = ["source activate vb"]\n'
        "[env]\n"
        'VB_SDPB_MODE = "apptainer"\n'
        'VB_SDPB_IMAGE = "/images/sdpb.sif"\n'
    )
    return profile


def test_expand_sweep_cartesian():
    combos = submit.expand_sweep(
        {"params": {"lam": 7}, "sweep": {"c": ["2", "4"], "tol": ["0.1", "0.05"]}}
    )
    assert len(combos) == 4
    assert {"lam": 7, "c": "2", "tol": "0.1"} in combos
    assert all(p["lam"] == 7 for p in combos)


def test_expand_sweep_rows_crossed_with_grid():
    combos = submit.expand_sweep({
        "params": {"lam": 19},
        "sweep_rows": [
            {"q_order": 60, "npoles": 20},
            {"q_order": 80, "npoles": 60},
        ],
        "sweep": {"delta_ext": ["1.5", "2.0"]},
    })
    assert len(combos) == 4  # 2 rows x 2 grid points
    assert {"lam": 19, "q_order": 60, "npoles": 20, "delta_ext": "1.5"} in combos
    assert {"lam": 19, "q_order": 80, "npoles": 60, "delta_ext": "2.0"} in combos
    assert all(p["lam"] == 19 for p in combos)


def test_expand_sweep_rows_without_grid():
    combos = submit.expand_sweep({
        "params": {"lam": 19},
        "sweep_rows": [{"mpiprocs": 8}, {"mpiprocs": 16}, {"mpiprocs": 32}],
    })
    assert combos == [
        {"lam": 19, "mpiprocs": 8},
        {"lam": 19, "mpiprocs": 16},
        {"lam": 19, "mpiprocs": 32},
    ]


def test_expand_sweep_rows_absent_matches_old_cartesian_only_behavior():
    # No [[sweep_rows]] at all must behave exactly like before its addition.
    combos = submit.expand_sweep({"params": {"lam": 7}, "sweep": {"c": ["2", "4"]}})
    assert combos == [{"lam": 7, "c": "2"}, {"lam": 7, "c": "4"}]


def test_expand_sweep_rows_rejects_non_table_entries():
    from virasoro_bootstrap.hpc.params import validate_sweep

    for rows in (["not-a-table"], False, 0, '', None, {}):
        config = {'task': 'scalar_gap_bisection', 'params': {'c': '2'}, 'sweep_rows': rows}
        with pytest.raises(ValueError, match="sweep_rows"):
            submit.expand_sweep(config)
        assert any('sweep_rows' in error for error in validate_sweep(config))
    config = {'task': 'scalar_gap_bisection', 'params': {'c': '2'}, 'sweep_rows': []}
    assert submit.expand_sweep(config) == []
    assert any('no tasks' in error for error in validate_sweep(config))


def test_submit_dry_run_renders_everything(tmp_path, monkeypatch):
    run_dir = submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=tmp_path / "runs", run_id="testrun", dry_run=True,
    )
    specs = sorted((run_dir / "specs").glob("*.json"))
    assert [s.name for s in specs] == ["0000.json", "0001.json", "0002.json"]
    spec0 = json.loads(specs[0].read_text())
    assert spec0 == {"task": "fake_echo", "params": {"lam": 7, "c": "2"},
                     "output_dir": "tasks/0000"}
    script = (run_dir / "job.sbatch").read_text()
    assert "#SBATCH --array=0-2%16" in script
    assert "#SBATCH --partition=day" in script
    assert "#SBATCH --requeue" in script
    assert f"#SBATCH --signal=B:USR1@{submit.SIGNAL_LEAD_S}" in script
    assert "#SBATCH --open-mode=append" in script
    assert "module load miniconda" in script
    assert "source activate vb" in script
    assert 'export VB_SDPB_MODE="apptainer"' in script
    assert script.rstrip().endswith(
        'exec "python" -m virasoro_bootstrap.hpc.worker '
        '"specs/$(printf \'%04d\' "$SLURM_ARRAY_TASK_ID").json"'
    )
    monkeypatch.chdir(tmp_path)
    for name, directory in (('audit run', tmp_path / 'run with spaces'),
                            ('audit#run', tmp_path / 'run#tag'),
                            ('audit"run', tmp_path / 'run"tag'),
                            ("audit'run", tmp_path / "run'tag"),
                            ('audit\\run', tmp_path / 'run\\tag'),
                            ('audit%a%j', tmp_path / 'run%a%j'),
                            ('relative run', Path('relative%a\\tag'))):
        rendered = submit.render_sbatch(name=name, run_dir=directory, n_tasks=1,
                                        resources={}, cluster={})
        directives = [shlex.split(line.removeprefix('#SBATCH '), comments=True)
                      for line in rendered.splitlines() if line.startswith('#SBATCH ')]
        assert ['--job-name=' + name] in directives
        # Slurm expands patterns before opening stdout. Only the basename's
        # array placeholder belongs in that pattern; the work directory is literal.
        assert ['--chdir=' + str(directory.resolve())] in directives
        assert ['--output=logs/task-%a.out'] in directives
    for invalid in ('bad\nname', 'bad\rname', 'bad\0name'):
        with pytest.raises(ValueError, match='job name'):
            submit.render_sbatch(name=invalid, run_dir=tmp_path, n_tasks=1,
                                 resources={}, cluster={})


def test_toml_numbers_preserve_exact_values_in_saved_tasks(tmp_path, monkeypatch):
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.hpc.params import validate_sweep
    from virasoro_bootstrap.hpc.tasks import _task_integer

    monkeypatch.setattr(submit, 'python_preflight', lambda *args: [])
    monkeypatch.setattr(submit, 'sdpb_preflight', lambda *args: [])
    literals = ('0.7', '0.700_000_000_000_000_01', '1.00000000000000000001',
                '1.0e-500', '1e23', '7.0', '9_007_199_254_740_993.0', '1.0e5000')
    sweep = tmp_path / 'exact.toml'
    sweep.write_text('task = "scalar_gap_bisection"\n[resources]\ncpus_per_task = 8.0\n'
                     '[params]\nlam = 7.0\nspins = [0.0, 2.0]\n[sweep]\nc = ['
                     + ', '.join(literals) + ']\n')
    cluster = tmp_path / 'cluster.toml'
    cluster.write_text('max_array_throttle = 8.0\n[env]\nVERSION = 1.0\n')
    parsed = submit.load_toml(sweep)
    assert validate_sweep(parsed, expand=submit.expand_sweep) == []
    run = submit.create_run(sweep, cluster, runs_root=tmp_path / 'runs',
                            run_id='exact', dry_run=True)
    manifest = json.loads((run / 'manifest.json').read_text())
    for i, token in enumerate(literals):
        spec = json.loads((run / 'specs' / f'{i:04d}.json').read_text())
        expected = literal_number(token.replace('_', ''))
        assert literal_number(spec['params']['c']) == expected, token
        assert literal_number(manifest['sweep']['sweep']['c'][i]) == expected, token
        assert _task_integer('lam', spec['params']['lam']) == 7
    assert 'export VERSION="1.0"' in (run / 'job.sbatch').read_text()
    assert submit._array_throttle(manifest['cluster']) == 8
    submit.retry_run(run, dry_run=True)
    for token in ('nan', 'inf', '-inf'):
        sweep.write_text(f'task = "scalar_gap_bisection"\n[params]\nc = {token}\n')
        assert any('c =' in error for error in validate_sweep(submit.load_toml(sweep)))
    for token in ('0.99999999999999999999', '7.00000000000000000001'):
        sweep.write_text(f'task = "scalar_gap_bisection"\n[params]\nc = 2\nlam = {token}\n')
        assert any('lam' in error for error in validate_sweep(submit.load_toml(sweep)))


def test_submit_python_key_overrides_interpreter(tmp_path):
    profile = tmp_path / "cluster.toml"
    profile.write_text(
        'kind = "slurm"\npartition = "day"\n'
        'python = "$HOME/.conda/envs/vb/bin/python"\n'
    )
    run_dir = submit.create_run(
        _write_sweep(tmp_path), profile,
        runs_root=tmp_path / "runs", run_id="pyrun", dry_run=True,
    )
    script = (run_dir / "job.sbatch").read_text()
    assert 'exec "$HOME/.conda/envs/vb/bin/python" -m virasoro_bootstrap.hpc.worker' in script
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["n_tasks"] == 3
    assert manifest["slurm_job_id"] is None


def test_submit_local_kind_runs_tasks(tmp_path, monkeypatch):
    # Register both vars with monkeypatch so whatever _apply_local_env writes
    # is rolled back after the test (delenv alone wouldn't restore-on-teardown
    # for vars that were absent before the test).
    monkeypatch.setenv("VB_SDPB_MODE", "docker")
    monkeypatch.setenv("VB_SDPB_IMAGE", "sentinel")
    run_dir = submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path, kind="local"),
        runs_root=tmp_path / "runs", run_id="localrun",
    )
    assert CALLS == ["echo:2", "echo:4", "echo:8"]
    import os
    assert os.environ["VB_SDPB_MODE"] == "apptainer"  # profile env applied
    for i in range(3):
        assert (run_dir / "tasks" / f"{i:04d}" / "result.json").exists()


def test_retry_only_resubmits_pending(tmp_path, capsys, monkeypatch):
    run_dir = submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=tmp_path / "runs", run_id="retryrun", dry_run=True,
    )
    # Complete task 1 by hand; 0 and 2 stay pending.
    assert worker.run_spec(run_dir / "specs" / "0001.json") == 0
    submit.retry_run(run_dir, dry_run=True)
    out = capsys.readouterr().out
    assert "retrying 2 task(s): [0, 2]" in out
    assert "--array=0,2%16" in out
    result = run_dir / 'tasks/0001/result.json'
    original = Path.read_text
    def disappearing_result(path, *args, **kwargs):
        if path == result:
            path.unlink()
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_text', disappearing_result)
        assert submit._pending_indices(run_dir, 3) == [0, 1, 2]


def _make_bare_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "tasks" / "0000").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "x", "task": "t", "n_tasks": 1, "slurm_job_id": None}
    ))
    (run_dir / "specs" / "0000.json").write_text(json.dumps(
        {"task": "t", "params": {}, "output_dir": "tasks/0000"}
    ))
    return run_dir


def test_status_tracks_retries_and_edited_specs_without_hiding_new_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('VB_SCRATCH_ROOT', raising=False)
    monkeypatch.setattr(worker, '_requeue_self', lambda: False)
    monkeypatch.setattr(worker, '_now', lambda: '2026-01-01T00:00:00+00:00')
    run = tmp_path / 'run'
    spec = _write_spec(run, 0, 'fake_echo', {'c': '8'})
    (run / 'manifest.json').write_text(json.dumps({'run_id': 'run', 'task': 'fake_echo', 'n_tasks': 1}))
    payload = {'reason': 'solve_error:original', 'error': 'old failure'}
    monkeypatch.setitem(TASKS, 'fake_echo', lambda ctx: payload)
    assert worker.run_spec(spec) == 1
    result_path = run / 'tasks/0000/result.json'
    original = result_path.read_bytes()

    def unserializable(ctx):
        view = collect.load_run(run)[1][0]
        assert view.state == 'running' and view.result is None
        return {'bad': object()}
    monkeypatch.setitem(TASKS, 'fake_echo', unserializable)
    assert worker.run_spec(spec) == 1
    view = collect.load_run(run)[1][0]
    assert view.state == 'failed' and 'not JSON serializable' in view.detail
    assert view.result == payload and view.starts == 2
    assert result_path.read_bytes() == original
    assert collect.status_main([str(run)]) == 0
    assert 'not JSON serializable' in capsys.readouterr().out

    # A new failed payload can become visible before the preceding status is
    # replaced. Its newer ordinal prevents an old status from hiding it.
    future = {**json.loads(original), 'starts': 3, 'attempt': 3,
              'failed': 'solve_error:latest: new payload',
              'result': {'reason': 'solve_error:latest', 'error': 'new payload'}}
    result_path.write_text(json.dumps(future))
    view = collect.load_run(run)[1][0]
    assert 'new payload' in view.detail and view.starts == 3
    result_path.write_bytes(original)

    def interrupt(ctx):
        view = collect.load_run(run)[1][0]
        assert view.state == 'running' and view.result is None
        raise TaskInterrupted('next attempt interrupted')
    monkeypatch.setitem(TASKS, 'fake_echo', interrupt)
    assert worker.run_spec(spec) == 3
    view = collect.load_run(run)[1][0]
    assert view.state == 'interrupted' and view.result is None and view.starts == 3

    # A -> B -> A resets each spec's starts, but ordinals still identify the
    # active attempt even when the original A failure remains on disk.
    monkeypatch.setitem(TASKS, 'fake_echo', unserializable)
    for charge in ('9', '8'):
        if charge == '8':
            # Losing the journal still leaves the last status ordinal usable.
            (run / 'tasks/0000/progress.json').unlink()
        _write_spec(run, 0, 'fake_echo', {'c': charge})
        # B has no matching payload; revisiting A still exposes its saved
        # failure until the new A attempt actually starts.
        assert collect.load_run(run)[1][0].state == ('stale' if charge == '9' else 'failed')
        assert worker.run_spec(spec) == 1
        view = collect.load_run(run)[1][0]
        assert view.state == 'failed' and 'not JSON serializable' in view.detail
        assert view.starts == 1 and result_path.read_bytes() == original
        assert submit._pending_indices(run, 1) == [0]
    monkeypatch.setitem(TASKS, 'fake_echo', _fake_echo)
    assert worker.run_spec(spec) == 0
    view = collect.load_run(run)[1][0]
    assert view.state == 'done' and view.result['c'] == '8' and view.starts == 2

    _write_spec(run, 0, 'fake_echo', {'c': '9'})
    monkeypatch.setitem(TASKS, 'fake_echo', unserializable)
    assert worker.run_spec(spec) == 1
    view = collect.load_run(run)[1][0]
    assert view.state == 'failed' and 'not JSON serializable' in view.detail and view.result is None
    monkeypatch.setitem(TASKS, 'fake_echo', _fake_echo)
    assert worker.run_spec(spec) == 0
    doc = json.loads(result_path.read_text())
    assert doc['attempt'] == 8 and doc['starts'] == 2
    assert collect.load_run(run)[1][0].result['c'] == '9'


def test_load_run_retries_transient_done_without_result(tmp_path, monkeypatch):
    """Retry when an NFS client sees done status before the matching result.

    The worker writes result.json before status.json, but another client can
    observe the renames in the opposite order.
    """
    run_dir = _make_bare_run(tmp_path)
    digest = worker.spec_hash('t', {})
    (run_dir / "tasks" / "0000" / "status.json").write_text(json.dumps({
        "state": "done", 'spec_hash': digest, 'attempt': 2, 'starts': 2,
        'started_at': '2026-01-02T00:00:00+00:00'}))
    result_path = run_dir / "tasks" / "0000" / "result.json"
    real_read_json = collect._read_json
    calls = {"n": 0}

    older = None
    def flaky_read(path):
        if path == result_path:
            calls["n"] += 1
            if calls["n"] < 3:
                return older  # absent or an older failed result still visible
            return {'spec_hash': digest, 'attempt': 2, 'starts': 2,
                    "result": {"delta_gap": "1.0"}}
        return real_read_json(path)

    monkeypatch.setattr(collect, "_read_json", flaky_read)
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)  # don't slow the test
    for older in (None,
                  {'spec_hash': digest, 'attempt': 1, 'result': {'reason': 'solve_error'}},
                  {'spec_hash': digest, 'started_at': '2026-01-01T00:00:00+00:00',
                   'result': {'reason': 'solve_error'}}):
        calls['n'] = 0
        manifest, views = collect.load_run(run_dir)
        assert views[0].state == "done"
        assert views[0].result == {"delta_gap": "1.0"}
        assert calls["n"] == 3


def test_load_run_genuine_failure_still_reported(tmp_path, monkeypatch):
    """Report failure when done status has no readable result after the retries."""
    run_dir = _make_bare_run(tmp_path)
    (run_dir / "tasks" / "0000" / "status.json").write_text(json.dumps({"state": "done"}))
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)
    manifest, views = collect.load_run(run_dir)
    assert views[0].state == "failed"
    assert "no readable result.json" in views[0].detail
    task = run_dir / 'tasks/0000'
    old = worker.spec_hash('t', {'edited': True})
    (task / 'status.json').write_text(json.dumps({'state': 'running', 'spec_hash': old}))
    (task / 'progress.json').write_text(json.dumps({'starts': 12, 'spec_hash': old, 'probes': {'old': {}}}))
    view = collect.load_run(run_dir)[1][0]
    assert view.state == 'stale' and view.starts == 0
    assert 'probe(s)' not in view.detail


def test_malformed_status_fields_do_not_hide_valid_results_or_crash_inspection(tmp_path, capsys):
    run = _make_bare_run(tmp_path)
    task = run / 'tasks/0000'
    status = task / 'status.json'
    result = task / 'result.json'
    digest = worker.spec_hash('t', {})
    (task / 'progress.json').write_text(json.dumps({'probes': {'p': {}}}))
    for field, value in (('state', []), ('state', 'unknown'),
                         ('detail', None), ('detail', ['bad'])):
        status.write_text(json.dumps({'spec_hash': digest, 'state': 'running', field: value}))
        original = status.read_bytes()
        result.write_text(json.dumps({'spec_hash': digest, 'result': {'value': 7}}))
        assert collect.load_run(run)[1][0].state == 'done'
        result.unlink()
        view = collect.load_run(run)[1][0]
        assert view.state == ('failed' if field == 'state' else 'running')
        assert 'invalid status' in view.detail and '1 probe(s)' in view.detail
        assert collect.status_main([str(run)]) == 0
        assert 'invalid status' in capsys.readouterr().out
        assert status.read_bytes() == original


def test_saved_results_require_the_authoritative_job_spec(tmp_path, capsys):
    run = _make_bare_run(tmp_path)
    spec = run / 'specs/0000.json'
    saved_spec = spec.read_bytes()
    spec.unlink()
    # A manifest-only laptop mirror remains inspectable before outputs arrive.
    assert collect.load_run(run)[1][0].state == 'pending'
    task = run / 'tasks/0000'
    aggregate = run / 'results.json'
    aggregate.write_text('[{"saved": true}]')
    original_aggregate = aggregate.read_bytes()
    result = task / 'result.json'
    status = task / 'status.json'
    digest = worker.spec_hash('t', {})
    for document in ({'result': {'value': 7}},
                     {'task': 't', 'params': {}, 'spec_hash': digest, 'result': {'value': 7}}):
        result.write_text(json.dumps(document))
        original_result = result.read_bytes()
        for main in (collect.collect_main, collect.status_main):
            with pytest.raises(SystemExit) as exc:
                main([str(run)])
            assert exc.value.code == 2 and '0000.json' in capsys.readouterr().err
        assert aggregate.read_bytes() == original_aggregate
        assert result.read_bytes() == original_result
    spec.write_bytes(saved_spec)
    assert collect.load_run(run)[1][0].state == 'done'
    spec.unlink()
    result.unlink()
    status.write_text(json.dumps({'state': 'done', 'spec_hash': digest}))
    with pytest.raises(ValueError, match='0000.json'):
        collect.load_run(run)


# --- collect / status --------------------------------------------------------

def test_collect_and_status(tmp_path, capsys, monkeypatch):
    run_dir = submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=tmp_path / "runs", run_id="collectrun", dry_run=True,
    )
    worker.run_spec(run_dir / "specs" / "0000.json")
    worker.run_spec(run_dir / "specs" / "0002.json")

    output = (run_dir / 'results.json').resolve()
    original = Path.replace
    staged = []
    def overlapping_collect(path, target):
        if Path(target).resolve() == output:
            staged.append(path)
            if len(staged) == 1:
                assert collect.collect_main([str(run_dir)]) == 0
        return original(path, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', overlapping_collect)
        assert collect.collect_main([str(run_dir)]) == 0
    assert len(staged) == 2 and staged[0] != staged[1]
    assert all(not path.exists() for path in staged)
    aggregate = json.loads((run_dir / "results.json").read_text())
    assert aggregate[0] == {"c": "2", "value": 42}
    assert aggregate[1]["reason"] == "pending"
    assert aggregate[1]["c"] == "4"
    assert aggregate[2] == {"c": "8", "value": 42}

    capsys.readouterr()
    collect.status_main([str(run_dir)])
    out = capsys.readouterr().out
    assert "done=2" in out and "pending=1" in out
    # Done rows carry the result's scalar fields inline.
    assert "value=42" in out
    previous = output.read_bytes()
    assert worker.run_spec(run_dir / 'specs/0001.json') == 0
    def failed_publication(path, target):
        if Path(target).resolve() == output:
            raise OSError('collection publication failed')
        return original(path, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', failed_publication)
        with pytest.raises(OSError, match='collection publication failed'):
            collect.collect_main([str(run_dir)])
    assert output.read_bytes() == previous
    assert not list(output.parent.glob('.results.json-*'))
    assert collect.collect_main([str(run_dir)]) == 0
    assert all(row['value'] == 42 for row in json.loads(output.read_text()))


def _set_slurm_job_id(run_dir: Path, job_id: str) -> None:
    manifest_path = run_dir / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["slurm_job_id"] = job_id
    manifest_path.write_text(json.dumps(m))


def test_status_and_collect_accept_bare_slurm_job_id(tmp_path, capsys):
    roots = tmp_path / "runs"
    run_dir = submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=roots, run_id="jobidrun", dry_run=True,
    )
    _set_slurm_job_id(run_dir, "12345")
    worker.run_spec(run_dir / "specs" / "0000.json")
    (roots / 'notes.txt').write_text('ordinary files are not runs')
    unknown = roots / 'unknown-date'
    unknown.mkdir()
    (unknown / 'manifest.json').write_text(json.dumps({'slurm_job_id': '12345', 'created_at': {}}))

    capsys.readouterr()
    assert collect.status_main(["12345", "--runs-root", str(roots)]) == 0
    assert "done=1" in capsys.readouterr().out

    assert collect.collect_main(["12345", "--runs-root", str(roots)]) == 0
    assert json.loads((run_dir / "results.json").read_text())[0] == {"c": "2", "value": 42}


def test_status_accepts_array_task_suffixed_job_id(tmp_path, capsys):
    # squeue shows "<job>_<array-idx>" (e.g. from our own sbatch arrays);
    # that suffix must be stripped before matching the manifest's plain id.
    roots = tmp_path / "runs"
    run_dir = submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=roots, run_id="arrayjob", dry_run=True,
    )
    _set_slurm_job_id(run_dir, "12345")
    capsys.readouterr()
    assert collect.status_main(["12345_0", "--runs-root", str(roots)]) == 0
    assert "n_tasks=" in capsys.readouterr().out


def test_status_unknown_job_id_raises_clear_error(tmp_path):
    roots = tmp_path / "runs"
    submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=roots, run_id="x", dry_run=True,
    )
    with pytest.raises(SystemExit, match="no run under .* has SLURM job id 99999"):
        collect.status_main(["99999", "--runs-root", str(roots)])


def test_status_bare_job_id_uses_cluster_runs_root_without_explicit_flag(tmp_path, capsys):
    # No --runs-root at all: must fall back to --cluster's runs_root (the
    # whole point -- vb-status 12345678 on the cluster, relying on $VB_CLUSTER-
    # defaulted --cluster, with no other flags).
    roots = tmp_path / "actual_runs"
    run_dir = submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=roots, run_id="viacluster", dry_run=True,
    )
    _set_slurm_job_id(run_dir, "55555")
    cluster_path = tmp_path / "cluster.toml"
    cluster_path.write_text(f'kind = "slurm"\nruns_root = "{roots}"\n')

    capsys.readouterr()
    rc = collect.status_main(["55555", "--cluster", str(cluster_path)])
    assert rc == 0
    assert "n_tasks=" in capsys.readouterr().out


def test_literal_existing_path_still_works_unchanged(tmp_path, capsys):
    # A real run directory whose *name* happens to look like a job id (all
    # digits) must resolve as the literal path, not trigger a manifest scan.
    roots = tmp_path / "runs"
    run_dir = submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=roots, run_id="99999", dry_run=True,
    )
    _set_slurm_job_id(run_dir, "11111")  # deliberately different from the dir name
    capsys.readouterr()
    assert collect.status_main([str(run_dir)]) == 0
    assert "run 99999" in capsys.readouterr().out


def test_list_shows_runs_newest_first(tmp_path, capsys):
    roots = tmp_path / "runs"
    submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=roots, run_id="older", dry_run=True,
    )
    submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=roots, run_id="newer", dry_run=True,
    )
    # create_run's created_at has second resolution; force a deterministic
    # order rather than relying on two calls landing in different seconds.
    older_manifest = roots / "older" / "manifest.json"
    newer_manifest = roots / "newer" / "manifest.json"
    m = json.loads(older_manifest.read_text())
    m["created_at"] = "2020-01-01T00:00:00+00:00"
    older_manifest.write_text(json.dumps(m))
    m = json.loads(newer_manifest.read_text())
    m["created_at"] = "2025-01-01T00:00:00+00:00"
    newer_manifest.write_text(json.dumps(m))

    capsys.readouterr()  # discard the two create_run "run <id>: ..." prints
    rc = collect.list_main(["--runs-root", str(roots)])
    assert rc == 0
    out = capsys.readouterr().out
    # Match the actual list-entry format, not the bare word: on macOS the tmp
    # path itself contains "/folders/", which trivially contains "older" as a
    # substring and would false-match a naive out.index("older").
    assert out.index("\n  newer  task=") < out.index("\n  older  task=")
    assert "task=fake_echo" in out
    assert "n_tasks=3" in out
    # Damaged display metadata must not prevent inspecting the readable runs.
    m['created_at'] = {}
    newer_manifest.write_text(json.dumps(m))
    assert collect.list_main(['--runs-root', str(roots)]) == 0
    out = capsys.readouterr().out
    assert out.index('\n  older  task=') < out.index('\n  newer  task=')


def test_list_respects_limit(tmp_path, capsys):
    roots = tmp_path / "runs"
    for i in range(3):
        submit.create_run(
            _write_sweep(tmp_path), _write_cluster(tmp_path),
            runs_root=roots, run_id=f"run{i}", dry_run=True,
        )
    capsys.readouterr()
    rc = collect.list_main(["--runs-root", str(roots), "-n", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("task=fake_echo") == 2
    assert "1 more" in out
    assert collect.list_main(['--runs-root', str(roots), '-n', '0']) == 0
    out = capsys.readouterr().out
    assert 'task=fake_echo' not in out and '3 more' in out
    with pytest.raises(SystemExit) as exc:
        collect.list_main(['--runs-root', str(roots), '-n', '-1'])
    assert exc.value.code == 2
    assert '--limit' in capsys.readouterr().err
    # A runs root can also hold notes or unrelated files, not just run directories.
    (roots / 'notes.txt').write_text('keep this file')
    assert collect.list_main(['--runs-root', str(roots)]) == 0
    assert capsys.readouterr().out.count('task=fake_echo') == 3


def test_malformed_run_manifest_cannot_publish_results_or_retry(tmp_path, monkeypatch, capsys):
    from virasoro_bootstrap.hpc import recertify

    run = tmp_path / "run"
    run.mkdir()
    manifest = run / "manifest.json"
    output = run / "results.json"
    output.write_text('[{"saved": true}]')
    original = output.read_bytes()
    new_output = tmp_path / "new-output" / "results.json"
    monkeypatch.setattr(submit, '_pending_indices',
                        lambda *args: pytest.fail('invalid manifest reached retry dispatch'))
    documents = [[], {}, *({'n_tasks': count} for count in
                           (True, False, 1.5, -1, 0, '1_0', None))]
    for document in documents:
        manifest.write_text(json.dumps(document))
        with pytest.raises(ValueError, match='manifest.json|n_tasks'):
            collect.load_run(run)
        with pytest.raises(ValueError, match='manifest.json|n_tasks'):
            submit.retry_run(run)
        for selection in (['--task', '0'], ['--all-tasks']):
            with pytest.raises(SystemExit) as exc:
                recertify.main([str(run), *selection])
            assert exc.value.code == 2
        for destination in (output, new_output):
            with pytest.raises(SystemExit) as exc:
                collect.collect_main([str(run), '-o', str(destination)])
            assert exc.value.code == 2
            assert output.read_bytes() == original
            assert not new_output.parent.exists()
        assert 'manifest.json' in capsys.readouterr().err
    for count in (2, 2.0, ' +02 '):
        manifest.write_text(json.dumps({'n_tasks': count}))
        assert len(collect.load_run(run)[1]) == 2
    for field in ('run_id', 'task'):
        for invalid in (None, [], ''):
            document = {'n_tasks': 1, 'run_id': 'run', 'task': 'fake_echo'}
            document[field] = invalid
            manifest.write_text(json.dumps(document))
            with pytest.raises(SystemExit) as exc:
                collect.status_main([str(run)])
            assert exc.value.code == 2 and field in capsys.readouterr().err


def test_bad_progress_fields_do_not_block_worker_or_collection(tmp_path, monkeypatch):
    monkeypatch.delenv('VB_SCRATCH_ROOT', raising=False)
    for index, count in enumerate(('bad', None, True, 1.5, -1)):
        run = tmp_path / str(index)
        spec = _write_spec(run, 0, 'fake_echo', {})
        (run / 'manifest.json').write_text(json.dumps({'n_tasks': 1}))
        progress = run / 'tasks/0000/progress.json'
        progress.parent.mkdir(parents=True)
        progress.write_text(json.dumps({'starts': count, 'probes': None}))
        view = collect.load_run(run)[1][0]
        assert view.starts == 0 and 'probe(s)' not in view.detail
        assert worker.run_spec(spec) == 0
        assert collect.load_run(run)[1][0].starts == 1
        assert Journal(progress).data['starts'] == 1


def test_cli_configuration_errors_are_clear_and_internal_errors_propagate(tmp_path, monkeypatch, capsys):
    from virasoro_bootstrap.hpc import recertify

    monkeypatch.setattr(submit, 'python_preflight', lambda *args: [])
    sweep = tmp_path / 'sweep.toml'
    sweep.write_text('task = "scalar_gap_bisection"\n[params]\nc = "2"\n')
    profile = tmp_path / 'cluster.toml'
    root = tmp_path / 'runs'
    base = [str(sweep), '--cluster', str(profile), '--runs-root', str(root), '--dry-run']
    for text, field in (('kind = "loacl"', 'kind'),
                        ('max_array_throttle = 2.5', 'max_array_throttle'),
                        ('kind =\n', 'cluster.toml'),
                        ('recorded_at = 2026-10-06T10:00:00Z', 'cluster.toml'),
                        ('[env]\nVB_SDPB_MODE = "bogus"', 'bogus')):
        profile.write_text(text)
        with pytest.raises(SystemExit) as exc:
            submit.main(base)
        assert exc.value.code == 2
        error = capsys.readouterr().err
        assert error.startswith('vb-submit: error:') and field in error
        assert not root.exists()
    profile.write_text('kind = "local"')
    for extra, field in ((['--run-id', '../outside'], 'run_id'),
                         (['--cluster', str(tmp_path / 'missing.toml')], 'missing.toml')):
        with pytest.raises(SystemExit) as exc:
            submit.main([*base, *extra])
        assert exc.value.code == 2 and field in capsys.readouterr().err
        assert not root.exists()
    for root_value in ('[]', 'false', '["/tmp/wrong"]'):
        profile.write_text(f'runs_root = {root_value}\n')
        calls = ((collect.list_main, ['--cluster', str(profile)]),
                 (collect.status_main, ['absent', '--cluster', str(profile)]),
                 (collect.collect_main, ['absent', '--cluster', str(profile)]),
                 (recertify.main, ['absent', '--task', '0', '--cluster', str(profile)]))
        for main, argv in calls:
            for extra in ([], ['--runs-root', str(root)]):
                with pytest.raises(SystemExit) as exc:
                    main([*argv, *extra])
                assert exc.value.code == 2 and 'runs_root' in capsys.readouterr().err
    # pytest's pythonpath setting is not inherited by subprocesses. In an
    # environment with an older installed wheel, use this same package tree.
    package_root = str(Path(collect.__file__).resolve().parents[2])
    child_env = {**os.environ, 'PYTHONPATH': package_root}
    result = subprocess.run([sys.executable, '-m', 'virasoro_bootstrap.hpc.collect',
                             'list', '--cluster', str(profile)],
                            capture_output=True, text=True, timeout=10, env=child_env)
    assert result.returncode == 2 and 'runs_root' in result.stderr
    assert 'Traceback' not in result.stderr and not result.stdout

    def internal_error(*args, **kwargs):
        raise ValueError('unexpected implementation failure')
    monkeypatch.setattr(submit, 'create_run', internal_error)
    with pytest.raises(ValueError, match='unexpected implementation failure'):
        submit.main(base)


def test_list_empty_runs_root(tmp_path, capsys):
    rc = collect.list_main(["--runs-root", str(tmp_path / "nope")])
    assert rc == 0
    assert "no runs under" in capsys.readouterr().out


def test_list_expands_home_and_vars(monkeypatch, tmp_path, capsys):
    from virasoro_bootstrap.hpc import recertify
    from virasoro_bootstrap.hpc.inputs import resolve_local_runs_root

    # Profile runs_root values use ~ or $HOME;
    # vb-list must expand both, same as create_run already does.
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_local_runs_root("~/vb_runs") == tmp_path / "vb_runs"
    assert resolve_local_runs_root("$HOME/vb_runs") == tmp_path / "vb_runs"
    monkeypatch.setenv('VB_TEST_EMPTY_ROOT', '')
    sweep, profile = _write_sweep(tmp_path), _write_cluster(tmp_path)
    for raw_root in ('', '  ', '\0', '$VB_TEST_EMPTY_ROOT',
                     '~__vb_missing_home_user_20261006__/runs'):
        with pytest.raises(ValueError, match='runs_root'):
            resolve_local_runs_root(raw_root)
        for main, args in ((submit.main, [str(sweep), '--cluster', str(profile), '--dry-run']),
                           (collect.list_main, []), (collect.status_main, ['absent']),
                           (collect.collect_main, ['absent']),
                           (recertify.main, ['absent', '--task', '0'])):
            with pytest.raises(SystemExit) as exc:
                main([*args, '--runs-root', raw_root])
            assert exc.value.code == 2 and 'runs' in capsys.readouterr().err
    monkeypatch.chdir(tmp_path)
    for index, raw_root in enumerate(('~/vb_runs', '$HOME/vb_runs')):
        run_id = f'expanded_{index}'
        run = submit.create_run(_write_sweep(tmp_path), _write_cluster(tmp_path),
                                runs_root=Path(raw_root), run_id=run_id, dry_run=True)
        assert run == (tmp_path / 'vb_runs' / run_id).resolve()
        capsys.readouterr()
        assert collect.list_main(['--runs-root', raw_root]) == 0
        assert run_id in capsys.readouterr().out
        assert collect.status_main([run_id, '--runs-root', raw_root]) == 0
        assert f'run {run_id}' in capsys.readouterr().out
        assert collect.collect_main([run_id, '--runs-root', raw_root]) == 0
        assert len(json.loads((run / 'results.json').read_text())) == 3


def test_list_cluster_without_host_uses_profile_runs_root_no_ssh(tmp_path, capsys, monkeypatch):
    # Standing directly on the cluster (no --host, no SSH needed): --cluster
    # alone should source the profile's runs_root instead of a bare ./runs.
    real_runs = tmp_path / "actual_vb_runs"
    submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=real_runs, run_id="oncluster", dry_run=True,
    )
    cluster_path = tmp_path / "cluster.toml"
    cluster_path.write_text(f'kind = "slurm"\nruns_root = "{real_runs}"\n')

    monkeypatch.chdir(tmp_path)  # so a bare ./runs (empty/absent) would find nothing
    capsys.readouterr()
    rc = collect.list_main(["--cluster", str(cluster_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "oncluster" in out
    assert "task=fake_echo" in out


def test_vb_cluster_env_var_is_default_for_submit(tmp_path, monkeypatch):
    cluster_path = _write_cluster(tmp_path)
    monkeypatch.setenv("VB_CLUSTER", str(cluster_path))
    run_dir = submit.create_run(
        _write_sweep(tmp_path), cluster_path,  # create_run itself unaffected; test main()'s parsing
        runs_root=tmp_path / "runs1", run_id="viaenv", dry_run=True,
    )
    assert run_dir.exists()
    # The actual behavior under test: main() with no --cluster flag at all.
    rc = submit.main([
        str(_write_sweep(tmp_path)), "--runs-root", str(tmp_path / "runs2"),
        "--run-id", "viaenv2", "--dry-run",
    ])
    assert rc == 0
    assert (tmp_path / "runs2" / "viaenv2" / "job.sbatch").exists()


def test_vb_cluster_env_var_missing_still_requires_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("VB_CLUSTER", raising=False)
    with pytest.raises(SystemExit):
        submit.main([str(_write_sweep(tmp_path)), "--dry-run"])


def test_vb_cluster_env_var_is_default_for_bare_list(tmp_path, monkeypatch, capsys):
    # The behavior this actually has to prove: `vb-list` with NO arguments at
    # all (no --cluster, no --host, no --runs-root) still finds the right
    # runs_root, purely from $VB_CLUSTER -- exactly the "use the current
    # cluster by default" ask. --cluster isn't otherwise required for
    # status/collect/list in local mode, so testing those wouldn't prove
    # the env var default is doing anything.
    real_runs = tmp_path / "actual_vb_runs"
    cluster_path = tmp_path / "cluster.toml"
    cluster_path.write_text(f'kind = "slurm"\nruns_root = "{real_runs}"\n')
    submit.create_run(
        _write_sweep(tmp_path), cluster_path,
        runs_root=real_runs, run_id="envdefaulted", dry_run=True,
    )
    monkeypatch.setenv("VB_CLUSTER", str(cluster_path))
    monkeypatch.chdir(tmp_path)  # so a bare ./runs would find nothing

    capsys.readouterr()
    assert collect.list_main([]) == 0
    assert "envdefaulted" in capsys.readouterr().out


def test_list_explicit_runs_root_beats_cluster(tmp_path, capsys):
    real_runs = tmp_path / "real"
    other_runs = tmp_path / "other"
    submit.create_run(
        _write_sweep(tmp_path), _write_cluster(tmp_path),
        runs_root=real_runs, run_id="x", dry_run=True,
    )
    cluster_path = tmp_path / "cluster.toml"
    cluster_path.write_text(f'kind = "slurm"\nruns_root = "{other_runs}"\n')

    capsys.readouterr()
    rc = collect.list_main(["--cluster", str(cluster_path), "--runs-root", str(real_runs)])
    assert rc == 0
    assert "x" in capsys.readouterr().out


def test_run_dir_with_only_staged_inputs_is_not_a_previous_run(tmp_path):
    """--host stages the sweep and profile into <run-dir>/inputs first.

    The guard against clobbering a previous run must not fire on that: the
    directory exists, but nothing has been submitted into it yet.
    """
    sweep = tmp_path / "s.toml"
    sweep.write_text('name = "s"\ntask = "scalar_gap_bisection"\n[sweep]\nc = ["2"]\n')
    cluster = tmp_path / "c.toml"
    cluster.write_text('kind = "slurm"\n')
    staged = tmp_path / "runs" / "run1" / "inputs"
    staged.mkdir(parents=True)
    (staged / "s.toml").write_text("")

    run_dir = submit.create_run(
        sweep, cluster, runs_root=tmp_path / "runs", run_id="run1", dry_run=True
    )
    assert (run_dir / "specs" / "0000.json").is_file()
    assert (run_dir / "inputs" / "s.toml").is_file()


def test_run_dir_with_a_real_run_still_refuses(tmp_path):
    sweep = tmp_path / "s.toml"
    sweep.write_text('name = "s"\ntask = "scalar_gap_bisection"\n[sweep]\nc = ["2"]\n')
    cluster = tmp_path / "c.toml"
    cluster.write_text('kind = "slurm"\n')
    (tmp_path / "runs" / "run1" / "specs").mkdir(parents=True)

    with pytest.raises(SystemExit, match="already exists"):
        submit.create_run(
            sweep, cluster, runs_root=tmp_path / "runs", run_id="run1"
        )


def test_sdpb_preflight_flags_a_missing_apptainer_image(tmp_path):
    warnings = submit.sdpb_preflight({
        "kind": "slurm",
        "env": {"VB_SDPB_MODE": "apptainer",
                "VB_SDPB_IMAGE": str(tmp_path / "absent.sif")},
    })
    assert warnings and "does not exist here" in warnings[0]
    warnings = submit.sdpb_preflight({
        'env': {'VB_SDPB_MODE': 'apptainer',
                'VB_SDPB_IMAGE': '~__vb_missing_home_user_20261006__/sdpb.sif'},
    })
    assert warnings and 'does not exist here' in warnings[0]


def test_sdpb_preflight_accepts_an_image_that_exists(tmp_path, monkeypatch):
    sif = tmp_path / "sdpb.sif"
    sif.write_text("")
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('VB_TEST_IMAGE_DIR', str(tmp_path))
    for image in (str(sif), '~/sdpb.sif', '$VB_TEST_IMAGE_DIR/sdpb.sif'):
        assert submit.sdpb_preflight({
            "kind": "slurm",
            "env": {"VB_SDPB_MODE": "apptainer", "VB_SDPB_IMAGE": image},
        }) == []


def test_sdpb_preflight_flags_a_missing_native_toolchain(tmp_path):
    warnings = submit.sdpb_preflight({
        "kind": "slurm",
        "env": {"VB_SDPB_MODE": "native", "VB_SDPB_BIN_DIR": str(tmp_path)},
    })
    assert warnings and "sdpb" in warnings[0]


def test_sdpb_preflight_stays_quiet_when_modules_build_the_environment():
    """A cluster may provide the toolchain only on compute nodes."""
    assert submit.sdpb_preflight({
        "kind": "slurm",
        "modules": ["sdpb"],
        "env": {"VB_SDPB_MODE": "native", "VB_SDPB_BIN_DIR": "/not/here"},
    }) == []


def test_sdpb_preflight_does_not_leak_env(tmp_path):
    before = dict(os.environ)
    submit.sdpb_preflight({
        "env": {"VB_SDPB_MODE": "apptainer", "VB_SDPB_IMAGE": str(tmp_path / "x.sif")},
    })
    assert dict(os.environ) == before


def test_python_preflight_flags_an_interpreter_without_the_package(tmp_path, monkeypatch):
    """The default profile python is bare "python", often a bare system one.

    Without this the failure is `No module named virasoro_bootstrap` inside a
    SLURM log, after the task has been queued and scheduled.
    """
    fake = tmp_path / 'environment with spaces' / 'python'
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\nprintf 'profile interpreter reached\\n'\nexit 1\n")
    fake.chmod(0o755)
    warnings = submit.python_preflight({"python": str(fake)})
    assert warnings and "cannot import virasoro_bootstrap" in warnings[0]
    quoted = fake.with_name('py"thon')
    quoted.write_text(fake.read_text())
    quoted.chmod(0o755)
    monkeypatch.setenv('VB_PROFILE_TEST_INTERPRETER', str(fake))
    for interpreter in (str(fake), '~/' + os.path.relpath(os.path.realpath(fake), os.path.realpath(Path.home())),
                        '$VB_PROFILE_TEST_INTERPRETER', str(quoted)):
        script = tmp_path / 'job.sbatch'
        script.write_text(submit.render_sbatch(name='example', run_dir=tmp_path,
                          n_tasks=1, resources={}, cluster={'python': interpreter}))
        result = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                                env={**os.environ, 'SLURM_ARRAY_TASK_ID': '0'})
        assert result.returncode == 1 and result.stdout.strip() == 'profile interpreter reached', result.stderr
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$VB_PROFILE_TEST_DATA\"\n")
    monkeypatch.setenv('VB_PROFILE_TEST_PREFIX', 'prefix with spaces')
    for value in ('a "quoted" value\\line', '$VB_PROFILE_TEST_PREFIX/path', 'line1\nline2', 3, False):
        script.write_text(submit.render_sbatch(name='example', run_dir=tmp_path,
                          n_tasks=1, resources={}, cluster={
                              'python': str(fake), 'env': {'VB_PROFILE_TEST_DATA': value}}))
        result = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                                env={**os.environ, 'SLURM_ARRAY_TASK_ID': '0'})
        assert result.returncode == 0 and result.stdout == os.path.expandvars(str(value)) + '\n', result.stderr
    fake.write_text("#!/bin/sh\nprintf 'unexpected interpreter launch\\n'\n")
    missing = tmp_path / 'removed run'
    missing.mkdir()
    script.write_text(submit.render_sbatch(name='example', run_dir=missing,
                      n_tasks=1, resources={}, cluster={'python': str(fake)}))
    missing.rmdir()
    result = subprocess.run(['bash', str(script)], cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, 'SLURM_ARRAY_TASK_ID': '0'})
    assert result.returncode != 0 and result.stdout == '' and 'removed run' in result.stderr


def test_python_preflight_flags_a_missing_interpreter(monkeypatch):
    warnings = submit.python_preflight({"python": "/no/such/python"})
    assert warnings and "not found here" in warnings[0]
    def unresolved_home(path):
        raise RuntimeError('Could not determine home directory')
    monkeypatch.setattr(Path, 'expanduser', unresolved_home)
    warnings = submit.python_preflight({'python': '~other_user/vb/bin/python'})
    assert warnings and 'cannot be resolved here' in warnings[0]
    for invalid in (None, False, [], '', '   ', '\0'):
        with pytest.raises(ValueError, match='profile python'):
            submit.render_sbatch(name='example', run_dir=Path('/tmp/example'), n_tasks=1,
                                 resources={}, cluster={'python': invalid})


def test_python_preflight_accepts_this_interpreter():
    assert submit.python_preflight({"python": sys.executable}) == []


def test_python_preflight_stays_quiet_when_modules_build_the_environment():
    assert submit.python_preflight({
        "modules": ["miniconda"], "python": "/no/such/python",
    }) == []


@pytest.mark.parametrize('content', [b'[]', b'\xff'])
def test_journal_and_collector_ignore_malformed_record_shapes(tmp_path, content, monkeypatch):
    from virasoro_bootstrap.hpc.collect import _read_json

    path = tmp_path / 'progress.json'
    path.write_bytes(content)
    journal = Journal(path)
    assert journal.data == {}
    assert _read_json(path) is None
    journal.update(starts=1)
    assert Journal(path).data == {'starts': 1}
    original = Path.read_text
    def disappearing_record(source, *args, **kwargs):
        if source == path:
            source.unlink()
        return original(source, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', disappearing_record)
    assert _read_json(path) is None


def test_failed_journal_update_preserves_memory_and_disk(tmp_path):
    journal = Journal(tmp_path / 'progress.json')
    journal.update(starts=1)
    with pytest.raises(TypeError):
        journal.update(unserializable=object())
    assert journal.data == {'starts': 1}
    assert Journal(journal.path).data == journal.data
