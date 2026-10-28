"""Completed-task cleanup preserves exact certification and retry behavior."""
import json
import os
from pathlib import Path

import mpmath as mp
import pytest

pytest.importorskip("fcntl", reason="HPC cleanup uses POSIX file locks")

from virasoro_bootstrap.certification import _direct_sdp_block_detail
from virasoro_bootstrap.hpc import worker
from virasoro_bootstrap.hpc.storage import (
    SAMPLES_NAME, RECORD_NAME, cleanup_sdp_inputs, load_compacted_sdp_block,
    regenerate_sdp_inputs, record_sdp_source, recorded_sdp_source,
)
from virasoro_bootstrap.sdp_writer import write_sdp_directory


def document():
    return {"objective": ["0", "1", "0"], "normalization": ["1", "0", "0"],
            "PositiveMatrixWithPrefactorArray": [
                {"polynomials": [[[["1", ".7"], ["-2"], ["-3"]]]],
                 "samplePoints": ["0", "1"], "sampleScalings": ["1", ".5"]},
                {"polynomials": [[[["1/8"], ["-2"], ["-3"]]]],
                 "samplePoints": ["0"], "sampleScalings": ["1"]}]}


def prepare(path):
    doc = document()
    # SDP/PMP numbers are serialized decimals, rather than API rational labels.
    doc["PositiveMatrixWithPrefactorArray"][1]["polynomials"][0][0][0] = [".125"]
    return write_sdp_directory(doc, path, eval_dps=80)


def _require_writable_exclusive_locks(monkeypatch):
    """Emulate Linux NFS's documented writable-descriptor requirement."""
    import errno
    import fcntl
    import os

    original = fcntl.flock
    def nfs_lock(descriptor, operation):
        if (operation & fcntl.LOCK_EX
                and fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY):
            raise OSError(errno.EBADF, 'exclusive NFS lock requires a writable descriptor')
        return original(descriptor, operation)
    monkeypatch.setattr(fcntl, 'flock', nfs_lock)


def test_exact_samples_and_certification_survive_cleanup(tmp_path):
    sdp = prepare(tmp_path / "sdp")
    output = tmp_path / "sdpb_out"
    record_sdp_source(sdp, output, state="completed")
    original = {i: json.loads((sdp / f"block_data_{i}.json").read_text()) for i in range(2)}
    metadata = {name: (sdp / name).read_bytes() for name in ("normalization.json", "control.json", "objectives.json")}
    with mp.workdps(80):
        before = [_direct_sdp_block_detail([mp.mpf(".3"), mp.mpf(".7")], sdp, i) for i in range(2)]
        stats = cleanup_sdp_inputs(sdp)
        after = [_direct_sdp_block_detail([mp.mpf(".3"), mp.mpf(".7")], sdp, i) for i in range(2)]
    assert before == after
    assert stats["removed_files"] == 4
    for i in range(2):
        assert load_compacted_sdp_block(sdp, i) == {key: original[i][key] for key in ("c", "B")}
        assert not (sdp / f"block_data_{i}.json").exists()
        assert not (sdp / f"block_info_{i}.json").exists()
    assert metadata == {name: (sdp / name).read_bytes() for name in metadata}
    assert recorded_sdp_source(output).resolve() == sdp.resolve()
    assert cleanup_sdp_inputs(sdp)["removed_files"] == 0


def test_regeneration_keeps_the_original_nearly_tied_normalization_basis(tmp_path, monkeypatch):
    from virasoro_bootstrap import sdp_writer

    solve = tmp_path/'solve'
    doc = {'objective': ['0', '0'], 'normalization': ['1', '1.'+'0'*49+'1'],
           'PositiveMatrixWithPrefactorArray': [{
               'polynomials': [[[['3'], ['5']]]], 'samplePoints': ['0'], 'sampleScalings': ['1']}]}
    sdp = write_sdp_directory(doc, solve/'sdp', eval_dps=40)
    record_sdp_source(sdp, solve/'sdpb_out', state="completed")
    (solve/'pmp.json').write_text(json.dumps(doc))
    # A prior solve's log must not override the current retained basis.
    (solve/'pmp2sdp.log').write_text('sdp_writer (eval_dps=80, output_digits=80)')
    norm = json.loads((sdp/'normalization.json').read_text())
    assert norm['eliminated_index'] == 0 and norm['eval_dps'] == 40
    original = json.loads((sdp/'block_data_0.json').read_text())
    cleanup_sdp_inputs(sdp)
    from virasoro_bootstrap.hpc import storage

    with monkeypatch.context() as patch:
        patch.setattr(storage, 'load_compacted_sdp_block',
                      lambda *_: pytest.fail('invalid output precision reached sample reconstruction'))
        for invalid in (True,):
            with pytest.raises(ValueError, match='output_digits'):
                regenerate_sdp_inputs(solve, output_digits=invalid)
    captured = []
    writer = sdp_writer.write_sdp_directory

    def inspect_samples(*args, **kwargs):
        captured.append((kwargs['eval_dps'], kwargs['output_digits'], kwargs['sample_values_by_block']))
        return writer(*args, **kwargs)

    monkeypatch.setattr(sdp_writer, 'write_sdp_directory', inspect_samples)
    regenerate_sdp_inputs(solve)
    assert captured == [(40, 40, {0: [[mp.mpf(3), mp.mpf(5)]]})]
    assert json.loads((sdp/'block_data_0.json').read_text()) == original
    assert recorded_sdp_source(solve/'sdpb_out').resolve() == sdp.resolve()


@pytest.mark.parametrize('legacy_record', [False, True])
def test_interrupted_pruning_can_resume_without_losing_samples(tmp_path, monkeypatch, legacy_record):
    sdp = prepare(tmp_path / "sdp")
    original = Path.unlink
    def fail(path, *args, **kwargs):
        if path.name == "block_info_0.json":
            raise OSError("interrupted deletion")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        with pytest.raises(OSError, match="interrupted"):
            cleanup_sdp_inputs(sdp)
    if legacy_record:
        path = sdp / RECORD_NAME
        record = json.loads(path.read_text())
        for name in ('control.json', 'normalization.json'):
            record['source_sha256'].pop(name)
        path.write_text(json.dumps(record))
    assert load_compacted_sdp_block(sdp, 0) is not None
    assert load_compacted_sdp_block(sdp, 1) is not None
    assert cleanup_sdp_inputs(sdp)["removed_files"] == 2


def test_overlapping_cleanup_retains_inputs_until_lock_is_released(tmp_path, monkeypatch):
    import fcntl
    _require_writable_exclusive_locks(monkeypatch)
    sdp = prepare(tmp_path / "sdp")
    with (sdp / "control.json").open("r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        assert cleanup_sdp_inputs(sdp)["removed_files"] == 0
        assert (sdp / "block_data_0.json").is_file()
        assert not (sdp / RECORD_NAME).exists()
    assert cleanup_sdp_inputs(sdp)["removed_files"] == 4


@pytest.mark.parametrize("field,value", [("num_blocks", 3)])
def test_resume_refuses_incompatible_record_before_deletion(tmp_path, monkeypatch, field, value):
    sdp = prepare(tmp_path / "sdp")
    original = Path.unlink
    def stop(path, *args, **kwargs):
        if _is_original(path):
            raise OSError("interrupted deletion")
        return original(path, *args, **kwargs)
    def _is_original(path):
        return path.parent == sdp and path.name.startswith("block_")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", stop)
        with pytest.raises(OSError, match="interrupted deletion"):
            cleanup_sdp_inputs(sdp)
    record_path = sdp / RECORD_NAME
    record = json.loads(record_path.read_text())
    record[field] = value
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incompatible input-cleanup record"):
        cleanup_sdp_inputs(sdp)
    assert all((sdp / f"block_data_{i}.json").exists() for i in range(2))


def test_changed_regeneration_source_prevents_input_deletion(tmp_path, monkeypatch):
    import gzip
    sdp = prepare(tmp_path / "solve/sdp")
    pmp = sdp.parent / "pmp.json"
    pmp.write_text(json.dumps(document()))
    original = gzip.open
    def change_source(path, mode, *args, **kwargs):
        if mode == "wt":
            pmp.write_text('{"changed":true}')
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(gzip, "open", change_source)
    with pytest.raises(ValueError, match="PMP changed during cleanup"):
        cleanup_sdp_inputs(sdp)
    assert all((sdp / f"block_data_{i}.json").exists() for i in range(2))
    assert not (sdp / RECORD_NAME).exists()


def test_metadata_changed_after_record_publication_prevents_input_deletion(tmp_path, monkeypatch):
    sdp = prepare(tmp_path / 'sdp')
    originals = {p.name: p.read_bytes() for p in sdp.glob('block_*.json')}
    replace = Path.replace

    def change_metadata(path, target):
        result = replace(path, target)
        if Path(target).name == RECORD_NAME:
            normalization = sdp / 'normalization.json'
            data = json.loads(normalization.read_text())
            data['eval_dps'] = 40
            normalization.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(Path, 'replace', change_metadata)
    with pytest.raises(ValueError, match='metadata changed before removal'):
        cleanup_sdp_inputs(sdp)
    assert {p.name: p.read_bytes() for p in sdp.glob('block_*.json')} == originals


def test_changed_regeneration_source_prevents_resuming_pruning(tmp_path, monkeypatch):
    sdp = prepare(tmp_path / "solve/sdp")
    pmp = sdp.parent / "pmp.json"
    pmp.write_text(json.dumps(document()))
    original = Path.unlink
    def stop(path, *args, **kwargs):
        if path.name == "block_data_1.json":
            raise OSError("interrupted deletion")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", stop)
        with pytest.raises(OSError, match="interrupted"):
            cleanup_sdp_inputs(sdp)
    pmp.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="PMP changed since cleanup"):
        cleanup_sdp_inputs(sdp)
    assert (sdp / "block_data_1.json").exists()


@pytest.mark.parametrize('operation', ['cleanup', 'functional'])
def test_changed_normalization_basis_refuses_recovery_without_removing_inputs(tmp_path, monkeypatch, operation):
    from virasoro_bootstrap.certification import _reconstruct_full_y

    solve = tmp_path / 'solve'
    doc = document()
    doc['normalization'] = ['1', '1', '0']
    doc['PositiveMatrixWithPrefactorArray'][1]['polynomials'][0][0][0] = ['.125']
    sdp = write_sdp_directory(doc, solve / 'sdp', eval_dps=80)
    (solve / 'pmp.json').write_text(json.dumps(doc))
    (solve / 'pmp2sdp.log').write_text('sdp_writer (eval_dps=80, output_digits=80)')
    unlink = Path.unlink

    def interrupt(path, *args, **kwargs):
        if path.name == 'block_data_0.json':
            raise OSError('interrupted deletion')
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', interrupt)
        with pytest.raises(OSError, match='interrupted'):
            cleanup_sdp_inputs(sdp)
    # Exercise the cached metadata verification as well as a fresh lookup.
    assert load_compacted_sdp_block(sdp, 0) is not None
    path = sdp / 'normalization.json'
    normalization = json.loads(path.read_text())
    normalization['eliminated_index'] = 1
    path.write_text(json.dumps(normalization))
    before = {p.name: p.read_bytes() for p in sdp.iterdir()}
    actions = {'cleanup': lambda: cleanup_sdp_inputs(sdp),
               'samples': lambda: load_compacted_sdp_block(sdp, 0),
               'restore': lambda: regenerate_sdp_inputs(solve),
               'functional': lambda: _reconstruct_full_y([mp.mpf(0), mp.mpf(0)], solve / 'sdpb_out')}
    with pytest.raises(ValueError, match='metadata changed since cleanup'):
        actions[operation]()
    assert {p.name: p.read_bytes() for p in sdp.iterdir()} == before


def test_corrupt_bundle_cannot_be_used_for_certification(tmp_path):
    sdp = prepare(tmp_path / "sdp")
    cleanup_sdp_inputs(sdp)
    (sdp / SAMPLES_NAME).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _direct_sdp_block_detail([mp.mpf(0), mp.mpf(0)], sdp, 0)


def test_present_invalid_metadata_hash_is_not_treated_as_an_older_record(tmp_path):
    sdp = prepare(tmp_path / 'sdp')
    cleanup_sdp_inputs(sdp)
    path = sdp / RECORD_NAME
    record = json.loads(path.read_text())
    record['source_sha256']['normalization.json'] = None
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='metadata changed since cleanup'):
        load_compacted_sdp_block(sdp, 0)


def test_certification_handles_removal_between_lookup_and_open(tmp_path, monkeypatch):
    sdp = prepare(tmp_path / "sdp")
    data = (sdp / "block_data_0.json").read_bytes()
    expected = _direct_sdp_block_detail([mp.mpf(0), mp.mpf(0)], sdp, 0)
    cleanup_sdp_inputs(sdp)
    source = sdp / "block_data_0.json"
    source.write_bytes(data)
    original = Path.open
    def racing_open(path, *args, **kwargs):
        if path == source:
            path.unlink()
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", racing_open)
    assert _direct_sdp_block_detail([mp.mpf(0), mp.mpf(0)], sdp, 0) == expected


def test_fresh_sdp_supersedes_compacted_inputs(tmp_path):
    sdp = prepare(tmp_path / "sdp")
    cleanup_sdp_inputs(sdp)
    prepare(sdp)
    assert not (sdp / RECORD_NAME).exists()
    assert not (sdp / SAMPLES_NAME).exists()
    assert (sdp / "block_data_0.json").exists()
    assert cleanup_sdp_inputs(sdp)["removed_files"] == 4


@pytest.mark.parametrize("directory,explicit", [("sdp", False), ("custom_inputs", True)])
def test_inputs_regenerate_without_blocks_or_solver_and_keep_exact_samples(tmp_path, directory, explicit):
    solve = tmp_path / "solve"
    sdp = prepare(solve / directory)
    doc = document()
    doc["PositiveMatrixWithPrefactorArray"][1]["polynomials"][0][0][0] = [".125"]
    (solve / "pmp.json").write_text(json.dumps(doc))
    (solve / "pmp2sdp.log").write_text("sdp_writer (eval_dps=80, output_digits=80)")
    original = {i: json.loads((sdp / f"block_data_{i}.json").read_text()) for i in range(2)}
    cleanup_sdp_inputs(sdp)
    regenerate_sdp_inputs(sdp if explicit else solve)
    for i in range(2):
        regenerated = json.loads((sdp / f"block_data_{i}.json").read_text())
        assert regenerated == original[i]
    assert not (sdp / RECORD_NAME).exists()
    assert cleanup_sdp_inputs(sdp)["removed_files"] == 4
    # Direct modular solves and log-suppressed runs still retain the writer
    # precision in normalization.json. Recovery needs no conversion log.
    (solve / "pmp2sdp.log").unlink()
    regenerate_sdp_inputs(sdp if explicit else solve)
    assert {i: json.loads((sdp / f"block_data_{i}.json").read_text()) for i in range(2)} == original
    # Older normalization files keep the log fallback.
    path = sdp / "normalization.json"
    normalization = json.loads(path.read_text())
    normalization.pop("eval_dps")
    path.write_text(json.dumps(normalization))
    (solve / "pmp2sdp.log").write_text("sdp_writer (eval_dps=80, output_digits=80)")
    cleanup_sdp_inputs(sdp)
    regenerate_sdp_inputs(sdp if explicit else solve)
    assert {i: json.loads((sdp / f"block_data_{i}.json").read_text()) for i in range(2)} == original


def test_restoration_refuses_overlapping_cleanup_without_changing_saved_samples(tmp_path, monkeypatch):
    import fcntl
    sdp = prepare(tmp_path / "solve/sdp")
    cleanup_sdp_inputs(sdp)
    _require_writable_exclusive_locks(monkeypatch)
    saved = (sdp / SAMPLES_NAME).read_bytes()
    with (sdp / "control.json").open("r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="already in progress"):
            regenerate_sdp_inputs(sdp.parent)
    assert (sdp / SAMPLES_NAME).read_bytes() == saved
    assert not (sdp / "block_data_0.json").exists()


@pytest.mark.parametrize("directory", ["custom_inputs"])
def test_public_moment_certifier_preserves_verdict_and_details_after_cleanup(tmp_path, directory):
    from types import SimpleNamespace
    from virasoro_bootstrap.certification import certify_dual_from_sdpb_output
    sdp = prepare(tmp_path / "solve" / directory)
    output = sdp.parent / "sdpb_out"
    output.mkdir()
    (output / "y.txt").write_text("2 1\n0\n0\n")
    doc = document()
    doc["PositiveMatrixWithPrefactorArray"][1]["polynomials"][0][0][0] = [".125"]
    for block in doc["PositiveMatrixWithPrefactorArray"]:
        block["DampedRational"] = {"base": ".5", "constant": "1", "poles": []}
    problem = SimpleNamespace(
        spec=SimpleNamespace(dps=80, coordinate="zzb", spin_list=(0, 2)),
        F_pairs=(), spin_channels=(SimpleNamespace(ell=0), SimpleNamespace(ell=2)),
        pmp_build=SimpleNamespace(doc=doc), pmp_metadata={"moment": {"kind": "normalized"}})
    record_sdp_source(sdp, output, state="completed")
    before = certify_dual_from_sdpb_output(problem, output, warn=False)
    cleanup_sdp_inputs(sdp)
    assert before[0] is True
    assert certify_dual_from_sdpb_output(problem, output, warn=False) == before


@pytest.mark.parametrize("kind", ["symlink", "malformed"])
def test_unsafe_inputs_leave_all_other_originals_untouched(tmp_path, kind):
    sdp = prepare(tmp_path / "sdp")
    source = sdp / "block_data_0.json"
    if kind == "missing":
        source.unlink()
    elif kind == "symlink":
        saved = tmp_path / "external.json"
        source.rename(saved)
        source.symlink_to(saved)
    else:
        source.write_text('{"c":[],"B":[]}')
    other = (sdp / "block_data_1.json").read_bytes()
    with pytest.raises(ValueError):
        cleanup_sdp_inputs(sdp)
    assert (sdp / "block_data_1.json").read_bytes() == other
    assert not (sdp / RECORD_NAME).exists()


@pytest.mark.parametrize("outcome", ["done", "interrupted", "keep_whitespace", "invalid_flag"])
def test_worker_cleans_only_final_results(tmp_path, monkeypatch, outcome):
    from virasoro_bootstrap.hpc.tasks import TaskInterrupted
    spec = tmp_path / "run/specs/0000.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps({"task": "cleanup_test", "params": {}, "output_dir": "tasks/0000"}))
    task = tmp_path / "run/tasks/0000"
    def run(ctx):
        prepare(ctx.output_dir / "solve/sdp")
        doc = document()
        doc["PositiveMatrixWithPrefactorArray"][1]["polynomials"][0][0][0] = [".125"]
        (ctx.output_dir / "solve/pmp.json").write_text(json.dumps(doc))
        out = ctx.output_dir / "solve/sdpb_out"
        out.mkdir()
        (out / "out.txt").write_text("completed")
        if outcome == "interrupted":
            raise TaskInterrupted("stop")
        return {"reason": "solve_error"} if outcome == "failed" else {"answer": "ok"}
    monkeypatch.setattr(worker, "get_task", lambda _: run)
    monkeypatch.setattr(worker, "_requeue_self", lambda: False)
    monkeypatch.delenv("VB_SCRATCH_ROOT", raising=False)
    flags = {"keep": "1", "keep_whitespace": " TRUE ", "invalid_flag": "typo"}
    monkeypatch.setenv("VB_KEEP_SOLVER_INPUTS", flags.get(outcome, "0"))
    assert worker.run_spec(spec) == {"failed": 1, "interrupted": 3}.get(outcome, 0)
    assert (task / "solve/sdp/block_data_0.json").exists() == (outcome != "done")
    if outcome == "done":
        assert worker.run_spec(spec) == 0
        assert load_compacted_sdp_block(task / "solve/sdp", 0) is not None


def test_preparation_only_inputs_are_retained(tmp_path):
    from virasoro_bootstrap.hpc.storage import cleanup_task_solver_inputs
    sdp = prepare(tmp_path / "task/solve/sdp")
    assert cleanup_task_solver_inputs(tmp_path / "task")["removed_files"] == 0
    assert (sdp / "block_data_0.json").exists()


def test_maintenance_protects_failed_stale_symlinked_and_opted_in_runs(tmp_path, monkeypatch):
    import fcntl
    import numpy as np
    from virasoro_bootstrap.hpc import cleanup
    queries = []
    monkeypatch.setattr(cleanup, "_active_jobs", lambda: queries.append(None) or set())
    root = tmp_path / "runs"
    for kind in ("done", "failed", "stale", "symlink", "keep", "missing", "busy"):
        run = root / kind
        specs = run / "specs"
        specs.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({"cluster": {"env": {
            "VB_KEEP_SOLVER_INPUTS": "1" if kind == "keep" else "0"}}}))
        task = run / "tasks/0000"
        if kind == "symlink":
            task.parent.mkdir()
            external = tmp_path / "external"
            external.mkdir()
            task.symlink_to(external)
        sdp = prepare(task / "solve/sdp")
        doc = document()
        doc["PositiveMatrixWithPrefactorArray"][1]["polynomials"][0][0][0] = [".125"]
        (sdp.parent / "pmp.json").write_text(json.dumps(doc))
        out = sdp.parent / "sdpb_out"
        out.mkdir()
        (out / "out.txt").write_text("completed")
        spec = {"task": "example", "params": {}, "output_dir": "tasks/0000"}
        (specs / "0000.json").write_text(json.dumps(spec))
        (task / "status.json").write_text(json.dumps({"state": "failed" if kind == "failed" else "done"}))
        (task / "result.json").write_text(json.dumps({"spec_hash": "old" if kind == "stale" else
            worker.spec_hash("example", {}), "result": {"answer": "ok"}}))
        if kind == "missing":
            (task / "result.json").unlink()
    for bad in ("true", np.bool_(True)):
        with pytest.raises(ValueError, match="execute must be bool"):
            cleanup.cleanup_runs(root, execute=bad)
    assert not queries
    assert cleanup.cleanup_runs(root, execute=False)["removed_files"] == 0
    for kind in ("done", "failed", "stale", "symlink", "keep", "missing", "busy"):
        assert (root / kind / "tasks/0000/solve/sdp/block_data_0.json").exists()
    with (root / 'busy/tasks/0000/.worker.lock').open('a+b') as anchor:
        fcntl.flock(anchor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stats = cleanup.cleanup_runs(root, execute=True, workers=2)
    assert not stats["errors"]
    assert stats["removed_files"] == 4
    for kind in ("failed", "stale", "symlink", "keep", "missing", "busy"):
        assert (root / kind / "tasks/0000/solve/sdp/block_data_0.json").exists()


def test_maintenance_skips_queued_array_runs_with_one_squeue_call(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc import cleanup
    calls = []
    monkeypatch.setattr(cleanup.shutil, "which", lambda name: "/usr/bin/squeue")
    # array element 1001 of base 1000 (%A %F), and a plain job 2001
    monkeypatch.setattr(cleanup.subprocess, "check_output",
                        lambda cmd, text: calls.append(cmd) or "1001 1000\n2001 2001\n")
    root = tmp_path / "runs"
    for name, job in (("array", 1000), ("single", 2001), ("finished", 3000)):
        (root / name / "specs").mkdir(parents=True)
        (root / name / "manifest.json").write_text(json.dumps({"slurm_job_id": job}))
        (root / name / "specs/0000.json").write_text("[]")   # an error only when the run is inspected
    errors = cleanup.cleanup_runs(root)["errors"]
    assert [Path(error["spec"]).parts[-3] for error in errors] == ["finished"]
    assert len(calls) == 1


def test_stale_solver_outputs_do_not_make_preparation_inputs_disposable(tmp_path):
    from virasoro_bootstrap.hpc.storage import cleanup_task_solver_inputs
    solve = tmp_path / "task/solve"
    (solve / "sdpb_out").mkdir(parents=True)
    (solve / "sdpb_out/out.txt").write_text("earlier solve")
    sdp = prepare(solve / "sdp")
    (solve / "pmp.json").write_text("{}")
    # make the earlier solve strictly older: coarse filesystem clocks can give both files the same mtime
    newer = (sdp / "control.json").stat().st_mtime_ns
    os.utime(solve / "sdpb_out/out.txt", ns=(newer-10**9, newer-10**9))
    assert cleanup_task_solver_inputs(tmp_path / "task")["removed_files"] == 0
    assert (sdp / "block_data_0.json").exists()


@pytest.mark.parametrize("bad", [[], {"cluster": {"env": {"VB_KEEP_SOLVER_INPUTS": "typo"}}}])
def test_malformed_run_metadata_is_reported_without_stopping_other_runs(tmp_path, monkeypatch, bad):
    from virasoro_bootstrap.hpc import cleanup
    monkeypatch.setattr(cleanup, "_active_jobs", lambda: set())
    root = tmp_path / "runs"
    for name, manifest in (("bad", bad), ("good", {})):
        run = root / name
        (run / "specs").mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps(manifest))
        (run / "specs/0000.json").write_text(json.dumps({
            "task": "example", "params": {}, "output_dir": "tasks/0000"}))
        task = run / "tasks/0000"
        sdp = prepare(task / "solve/sdp")
        (sdp.parent / "pmp.json").write_text("{}")
        (sdp.parent / "sdpb_out").mkdir()
        (sdp.parent / "sdpb_out/out.txt").write_text("completed")
        (task / "status.json").write_text(json.dumps({"state": "done"}))
        (task / "result.json").write_text(json.dumps({
            "spec_hash": worker.spec_hash("example", {}), "result": {"answer": "ok"}}))
    stats = cleanup.cleanup_runs(root, execute=True)
    assert len(stats["errors"]) == 1
    assert stats["errors"][0]["run"] == str(root / "bad")
    assert stats["removed_files"] == 4
    assert (root / "bad/tasks/0000/solve/sdp/block_data_0.json").exists()
    assert not (root / "good/tasks/0000/solve/sdp/block_data_0.json").exists()
