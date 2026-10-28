"""Regressions for the cluster-workflow review (hpc layer, 2026-09-28).

Each test pins one reproduced defect: a hotstart checkpoint chained across a
change of SDP structure, an error payload recorded as done, a sweep submitted
with a misspelled key, a run id that vb-status could not find, an SDP directory
not called ``sdp``, an ``--flag=value`` SDPB option, and help text written for
developers. No SDPB runs: the solver is replaced where a solve is involved.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import pytest

from virasoro_bootstrap.hpc import collect, params as hpc_params, submit, worker
from virasoro_bootstrap.hpc.journal import Journal
from virasoro_bootstrap.hpc.tasks import LAMBDA_PRESETS, TASKS, TaskContext

REPO = Path(__file__).resolve().parents[2]


def _ctx(tmp_path: Path, params: dict) -> TaskContext:
    return TaskContext(params=params, output_dir=tmp_path, journal=Journal(tmp_path / "progress.json"))


# --- 1. match_gap hotstart: chain a checkpoint only onto the same SDP structure --------

@dataclass
class _FakeRun:
    sdpb_status: str | None
    sdpb_summary: str | None
    checkpoint_dir: Path | None
    sdpb_log_path: Path | None = None
    sdpb_out_dir: Path | None = None


class _FakeProblem:
    def __init__(self, n_blocks: int):
        block = {"polynomials": [[[[0] * 10] * 3]], "samplePoints": [0] * 10}
        self.pmp_build = type("B", (), {"doc": {
            "objective": [0, 0, 0],
            "PositiveMatrixWithPrefactorArray": [block] * n_blocks,
        }})()
        self.n_blocks = n_blocks


def _fake_solver(monkeypatch, *, bound="1.3", broken_seed=False, always=None):
    """Replace the build + SDPB run of the gap task.

    The block count drops by one above gap 1.0 (a tower slice crossing the gap,
    as under twist_gap="match_gap"). Like SDPB, the fake solver aborts when
    seeded from a checkpoint of a different structure; ``broken_seed`` makes
    every seeded solve fail, ``always`` makes every solve fail with that cause.
    """
    import virasoro_bootstrap.identical_scalar as mod

    calls = []

    def build(spec, prepared=None):
        problem = _FakeProblem(6 if spec.delta_gap < 1 else 5)
        problem.gap = spec.delta_gap
        return problem

    def run(problem, output_dir, *, initial_checkpoint_dir=None, **_kw):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        seeded_from = None
        if initial_checkpoint_dir is not None:
            seeded_from = int((Path(initial_checkpoint_dir) / "blocks").read_text())
        calls.append((output_dir.name, problem.n_blocks, seeded_from))
        if always is not None:
            return _FakeRun("solver_error", always, None)
        if seeded_from is not None and (broken_seed or seeded_from != problem.n_blocks):
            return _FakeRun("solver_error", "Assertion 'local_height == block.LocalHeight()' failed", None)
        ck = output_dir / "sdp.ck"
        ck.mkdir(exist_ok=True)
        (ck / "blocks").write_text(str(problem.n_blocks))
        gap = problem.gap
        status = "primal_feasible" if gap < mp.mpf(bound) else "dual_feasible"
        return _FakeRun(status, None, ck)

    monkeypatch.setattr(mod, "build_identical_scalar_gap_problem", build)
    monkeypatch.setattr(mod, "run_identical_scalar_gap_problem", run)
    return calls


_GAP_PARAMS = {"c": "2", "lam": 7, "spins": [0, 2], "gap_lo": "0.5", "gap_hi": "2.0",
               "tol": "0.2", "twist_gap": "match_gap", "large_spin_double_twists": 3}


def test_match_gap_hotstart_never_seeds_a_different_sdp_structure(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc.tasks import scalar_gap_bisection

    calls = _fake_solver(monkeypatch)
    result = scalar_gap_bisection(_ctx(tmp_path, dict(_GAP_PARAMS)))
    assert result.get("reason") is None, result
    assert mp.mpf("1.1") <= mp.mpf(result["delta_gap"]) <= mp.mpf("1.5")
    seeded = [(n, s) for _name, n, s in calls if s is not None]
    assert seeded, "hotstart should still chain between probes of the same structure"
    assert all(n == s for n, s in seeded)
    assert any("SDP structure differs" in line for line in result["logs"])


def test_resume_uses_only_explicit_boolean_probe_verdicts(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc.tasks import _label, scalar_gap_bisection

    calls = _fake_solver(monkeypatch, bound="1.3")
    low_key, high_key = _label(mp.mpf("0.5")), _label(mp.mpf("2"))
    for index, bad_entry in enumerate(({'allowed': 'false'}, {'status': 'dual_feasible'})):
        calls.clear()
        ctx = _ctx(tmp_path / str(index), dict(_GAP_PARAMS, max_hi="2", hotstart=False))
        ctx.journal.update(probes={low_key: {'allowed': True}, high_key: bad_entry})
        result = scalar_gap_bisection(ctx)
        assert result.get('reason') is None, result
        assert mp.mpf(result['delta_gap_lo']) <= mp.mpf('1.3') <= mp.mpf(result['delta_gap_hi'])
        names = [name for name, _, _ in calls]
        assert 'gap_' + low_key not in names, 'valid saved work must still be reused'
        assert 'gap_' + high_key in names, 'invalid verdicts must be solved again'
        assert ctx.journal.data['probes'][high_key]['allowed'] is False


def test_grid_resume_recomputes_invalid_verdicts_and_keeps_valid_sector(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from virasoro_bootstrap import identical_scalar, modular_crossing
    from virasoro_bootstrap.hpc.tasks import isolated_relevant_scalar_grid_point

    calls = []
    monkeypatch.setattr(modular_crossing, 'build_modular_gap_problem', lambda *args: object())
    def solve(*args, **kwargs):
        calls.append('modular')
        return SimpleNamespace(sdpb_status='dual_feasible', sdpb_log_path=tmp_path / 'sdpb.log')
    monkeypatch.setattr(modular_crossing, 'run_modular_gap_problem', solve)
    monkeypatch.setattr(identical_scalar, 'build_identical_scalar_gap_problem',
                        lambda *args: pytest.fail('valid four-point verdict was not reused'))
    for index, invalid in enumerate((None, {'allowed': 'false', 'status': 'dual_feasible'},
                                     {'allowed': True, 'status': 'dual_feasible'})):
        calls.clear()
        ctx = _ctx(tmp_path / str(index), dict(c='2', delta_phi='.7', lam=7))
        ctx.journal.update(results={'modular': invalid,
                                    'four_point': {'allowed': True, 'status': 'primal_feasible'}})
        result = isolated_relevant_scalar_grid_point(ctx)
        assert calls == ['modular']
        assert result['modular_allowed'] is False
        assert result['four_point_allowed'] is True
        assert result['relaxed_coupled_light_allowed'] is False


def test_solver_error_retry_drops_the_checkpoint(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc.tasks import scalar_gap_bisection

    calls = _fake_solver(monkeypatch, broken_seed=True)
    result = scalar_gap_bisection(_ctx(tmp_path, dict(_GAP_PARAMS)))
    assert result.get("reason") is None, result
    by_probe: dict[str, list] = {}
    for name, _n, seeded_from in calls:
        by_probe.setdefault(name, []).append(seeded_from)
    # every probe that failed on its seed was retried cold, once
    for attempts in by_probe.values():
        if attempts[0] is not None:
            assert attempts[1:] == [None]
    assert any("without the hotstart checkpoint" in line for line in result["logs"])


def test_solver_error_result_carries_the_cause_from_the_sdpb_log(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc.tasks import result_error, scalar_gap_bisection

    cause = "docker: Error response from daemon: pull access denied for nope/sdpb"
    _fake_solver(monkeypatch, always=cause)
    result = scalar_gap_bisection(_ctx(tmp_path, dict(_GAP_PARAMS)))
    assert result["reason"] == "solve_error"
    assert "pull access denied" in result["error"]
    assert "pull access denied" in result_error(result)
    assert any("pull access denied" in line for line in result["logs"])


def test_scalar_only_probe_reports_a_solver_error_as_a_failure(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc.tasks import result_error, scalar_only_gap_probe

    _fake_solver(monkeypatch, always="docker: Error response from daemon: pull access denied")
    result = scalar_only_gap_probe(_ctx(tmp_path, {"c": "2", "delta": "1.0", "lam": 7}))
    assert result["reason"] == "solve_error"
    assert "pull access denied" in result_error(result)


def test_sdp_structure_signature_tracks_the_block_count():
    from virasoro_bootstrap.hpc.tasks import sdp_structure_signature

    assert sdp_structure_signature(_FakeProblem(6)) == sdp_structure_signature(_FakeProblem(6))
    assert sdp_structure_signature(_FakeProblem(6)) != sdp_structure_signature(_FakeProblem(5))


# --- 2. an error payload is a failure, and --retry reruns it -----------------------------

@pytest.fixture
def _error_task():
    calls = []

    def fake(ctx):
        calls.append(1)
        return {"delta_gap": None, "reason": "solve_error", "error": "boom", "logs": ["x"]}

    TASKS["fake_solve_error"] = fake
    yield calls
    TASKS.pop("fake_solve_error", None)


def _one_task_run(tmp_path: Path, task_name: str) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "run", "task": task_name, "n_tasks": 1, "cluster": {"kind": "local"}}
    ))
    (run_dir / "specs" / "0000.json").write_text(json.dumps(
        {"task": task_name, "params": {"c": "2"}, "output_dir": "tasks/0000"}
    ))
    return run_dir


def test_error_payload_is_recorded_failed_and_retried(tmp_path, _error_task, capsys):
    run_dir = _one_task_run(tmp_path, "fake_solve_error")
    spec = run_dir / "specs" / "0000.json"
    assert worker.run_spec(spec) == 1
    task_dir = run_dir / "tasks" / "0000"
    status = json.loads((task_dir / "status.json").read_text())
    assert status["state"] == "failed" and "solve_error: boom" in status["detail"]
    doc = json.loads((task_dir / "result.json").read_text())
    assert doc["failed"] and doc["result"]["reason"] == "solve_error"

    # vb-status/vb-collect: failed, but the payload is still collected
    _manifest, views = collect.load_run(run_dir)
    assert views[0].state == "failed" and views[0].result["reason"] == "solve_error"
    # vb-submit --retry and the worker agree that it is not done
    assert submit._pending_indices(run_dir, 1) == [0]
    capsys.readouterr()
    submit.retry_run(run_dir)
    assert "nothing to retry" not in capsys.readouterr().out
    assert len(_error_task) == 2


def test_worker_rejects_malformed_specs_before_creating_task_outputs(tmp_path, monkeypatch, capsys):
    from virasoro_bootstrap.hpc import recertify

    run = tmp_path/'run'
    specs = run/'specs'
    specs.mkdir(parents=True)
    path = specs/'0000.json'
    scratch = tmp_path/'scratch'
    monkeypatch.setenv('VB_SCRATCH_ROOT', str(scratch))
    (run/'manifest.json').write_text(json.dumps({'run_id': 'run', 'task': 't', 'n_tasks': 1}))
    aggregate = run/'results.json'
    aggregate.write_text('[{"saved": true}]')
    saved_aggregate = aggregate.read_bytes()
    monkeypatch.setattr(submit, '_sbatch',
                        lambda *args: pytest.fail('invalid saved spec reached submission'))
    valid = {'task': 'scalar_gap_bisection', 'params': {}, 'output_dir': 'tasks/0000'}
    bad = [[], {**valid, 'task': ''}, {**valid, 'params': None}, {**valid, 'output_dir': '\0'}]
    for document in bad:
        path.write_text(json.dumps(document))
        assert worker.run_spec(path) == 2
        assert 'bad spec' in capsys.readouterr().err
        with pytest.raises(SystemExit) as exc:
            collect.collect_main([str(run)])
        assert exc.value.code == 2 and '0000.json' in capsys.readouterr().err
        assert aggregate.read_bytes() == saved_aggregate
        with pytest.raises(SystemExit, match='0000.json'):
            recertify.probe_dirs(run, 0)
        with pytest.raises(SystemExit) as exc:
            submit.main(['--retry', str(run)])
        assert exc.value.code == 2 and '0000.json' in capsys.readouterr().err
        assert not (run/'tasks').exists() and not scratch.exists()
    path.write_text(json.dumps(valid))
    assert collect.load_run(run)[1][0].state == 'pending'
    assert recertify.probe_dirs(run, 0) == []
    assert submit._pending_indices(run, 1) == [0]
    assert not (run/'tasks').exists() and not scratch.exists()
    monkeypatch.delenv('VB_SCRATCH_ROOT')
    occupied = tmp_path/'occupied'
    occupied.write_text('existing user file')
    path.write_text(json.dumps({**valid, 'output_dir': str(occupied)}))
    assert worker.run_spec(path) == 2
    assert 'bad spec' in capsys.readouterr().err
    assert occupied.read_text() == 'existing user file'


def test_legacy_error_result_without_the_mark_is_still_pending(tmp_path, _error_task):
    run_dir = _one_task_run(tmp_path, "fake_solve_error")
    task_dir = run_dir / "tasks" / "0000"
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text(json.dumps({
        "spec_hash": worker.spec_hash("fake_solve_error", {"c": "2"}),
        "result": {"delta_gap": None, "reason": "solve_error"},
    }))
    assert submit._pending_indices(run_dir, 1) == [0]
    assert collect.load_run(run_dir)[1][0].state == "failed"


def test_answers_with_a_reason_are_not_failures():
    from virasoro_bootstrap.hpc.tasks import result_error

    for reason in ("lo_excluded", "hi_allowed_at_cap", "lo_allowed", "hi_excluded_at_cap", "inconclusive"):
        assert result_error({"delta_gap": None, "reason": reason}) is None
    assert result_error({"reason": "prepare_error:ValueError"}) == "prepare_error:ValueError"


# --- 3. sweeps are validated at submit time ---------------------------------------------

def _errors(**config):
    config.setdefault("task", "scalar_gap_bisection")
    return hpc_params.validate_sweep(config, expand=submit.expand_sweep)


def test_a_valid_sweep_passes():
    assert _errors(params={"lam": 7, "gap_lo": "0.5", "tol": "0.05", "spins": [0, 2],
                           "twist_gap": "match_gap", "hotstart": False,
                           "sdpb_options": {"--precision": 512, "--findDualFeasible": True}},
                   sweep={"c": ["2", "8/7"]}, resources={"cpus_per_task": 4}) == []
    assert _errors(params={"c": "2", "twist_gap": "1/3"}) == []
    for lam in (7.0, ' +07 '):
        assert _errors(params={'c': '2', 'lam': lam, 'spins': [0, ' +02 ']}) == []
    # This task explicitly permits lower precision than the sphere backend.
    assert _errors(task='modular_scalar_gap_bisection', params={'c': '2', 'dps': 1}) == []
    assert _errors(task='isolated_relevant_scalar_grid_point', params={
        'c': '2', 'delta_phi': '.7', 'dps': 1},
        sweep_rows=[{'four_point_dps': 40}, {'four_point_dps': 80}]) == []
    assert _errors(task='moment_nu1_over_nu0', params={
        'c': '2', 'delta_ext': '.7', 'large_spin_npoles': None, 'moment_cert_q_order': None,
        'large_spin_twists': None, 'large_spin_low_twist_step': None,
        'initial_checkpoint_source_dir': None}) == []
    assert _errors(params={'c': '2', 'spins': [2, 0, 2]}) == []
    assert _errors(params={'c': '2', 'spins': [0, 1]},
                   sweep_rows=[{'spins': [0, 2]}]) == []
    for spins in (None, [0, 1, 2], [' +02 ', 0]):
        assert _errors(task='isolated_relevant_scalar_grid_point',
                       params={'c': '2', 'delta_phi': '.7', 'modular_spins': spins}) == []
    assert _errors(task='isolated_relevant_scalar_degeneracy_intervals', params={
        'c': '2', 'delta_phi': '.7', 'degeneracy_grid': [0, '1/3', '.7']}) == []
    assert _errors(task='scalar_ope_max_scan', params={
        'c': '2', 'delta_stars': ['.7', 0, '.7'], 'spin_star': ' +02 '}) == []
    for task in ('scalar_gap_bisection', 'modular_scalar_gap_bisection'):
        assert _errors(task=task, params={'c': '2', 'gap_lo': 0, 'gap_hi': '.1'}) == []
        assert _errors(task=task, params={'c': '2', 'gap_lo': 1,
                       'gap_hi': f'{10**200+1}/{10**200}'}) == []
        assert _errors(task=task, params={'c': '2', 'gap_lo': -1},
                       sweep_rows=[{'gap_lo': 0}]) == []
    assert _errors(task='modular_scalar_gap_bisection', params={'c': '12', 'gap_lo': 3}) == []


def test_unknown_task_is_rejected_with_a_suggestion():
    (message,) = _errors(task="gap_bisection", params={"c": "8"})
    assert "unknown task 'gap_bisection'" in message and "scalar_gap_bisection" in message


def test_misspelled_param_key_is_rejected_not_defaulted():
    errors = _errors(params={"lam": 7, "c": "8", "toll": "0.1", "spin_list": [0, 2]})
    assert any("'toll'" in e and "did you mean 'tol'" in e for e in errors)
    assert any("'spin_list'" in e for e in errors)


def test_values_the_task_cannot_convert_are_rejected():
    errors = _errors(params={"lam": 7, "c": "2", "gap_lo": float("nan"), "gap_hi": "abc",
                             "hotstart": "false", "spins": [0, "x"],
                             "twist_gap": float("inf"),
                             "sdpb_options": {"--precision": "abc"}})
    text = "\n".join(errors)
    for key in ("gap_lo", "gap_hi", "hotstart", "spins", "twist_gap", "--precision"):
        assert key in text, key
    for key, value in (('num_threads', '2_0'), ('q_order', 0), ('dps', 9)):
        errors = _errors(params={'c': '2', key: value})
        assert any(key in error for error in errors), (key, value, errors)
    # A denominator of zero must be reported alongside other bad values.
    errors = _errors(params={'c': '1/0', 'hotstart': 'false'})
    assert any('c =' in error for error in errors) and any('hotstart' in error for error in errors)
    assert any('dps' in error for error in _errors(task='isolated_relevant_scalar_grid_point',
        params={'c': '2', 'delta_phi': '.7', 'dps': 1}))
    assert any('dps' in error for error in _errors(task='isolated_relevant_scalar_grid_point',
        params={'c': '2', 'delta_phi': '.7', 'dps': 1}, sweep_rows=[{'four_point_dps': 40}, {}]))
    cases = (
        ('scalar_gap_bisection', {'c': '2'}, 'spins', ([0, 1],)),
        ('isolated_relevant_scalar_grid_point', {'c': '2', 'delta_phi': '.7'},
         'modular_spins', ([0, True],)),
        ('isolated_scalar_four_point_ope_bounds', {'c': '2', 'delta_ext': '.7'},
         'directions', (['lowre'],)),
        ('moment_nu1_over_nu0', {'c': '2', 'delta_ext': '.7'},
         'identity_modes', (['inculde'],)),
        ('scalar_ope_max_scan', {'c': '2'}, 'spin_star', (1,)),
        ('isolated_relevant_scalar_delta_phi_intervals', {'c': '2'},
         'delta_phi_grid', (['.8', '.4'], ['1/5', '.2'])),
        ('isolated_relevant_scalar_degeneracy_intervals', {'c': '2', 'delta_phi': '.7'},
         'degeneracy_grid', ([2, 1],)),
        ('scalar_ope_max_scan', {'c': '2'}, 'delta_stars', ([-1, 0],)),
    )
    for task, base, key, invalid in cases:
        for value in invalid:
            errors = _errors(task=task, params={**base, key: value})
            assert any(key in error for error in errors), (task, key, value, errors)
    for task, base, key in (
        ('scalar_gap_bisection', {'c': '2'}, 'tol'),
        ('modular_scalar_gap_bisection', {'c': '2'}, 'tol'),
        ('isolated_relevant_scalar_delta_phi_intervals',
         {'c': '2', 'delta_phi_grid': ['.5', '.6']}, 'delta_phi_tol'),
        ('isolated_relevant_scalar_degeneracy_intervals',
         {'c': '2', 'delta_phi': '.7', 'degeneracy_grid': [0, 1]}, 'degeneracy_tol'),
    ):
        assert any(key in error for error in _errors(task=task, params={**base, key: '-1/10'}))
        assert _errors(task=task, params={**base, key: '1e-500'}) == []
    for task in ('scalar_gap_bisection', 'modular_scalar_gap_bisection'):
        for bracket in ({'gap_lo': -1}, {'gap_lo': 2, 'gap_hi': 1}):
            errors = _errors(task=task, params={'c': '2', **bracket})
            assert any('gap bracket' in error for error in errors), (task, bracket, errors)
        assert any('gap bracket' in error for error in _errors(task=task, params={'c': '2'},
                   sweep={'gap_lo': ['.1', '3'], 'gap_hi': ['2']}))
    assert any('gap bracket' in error for error in _errors(task='modular_scalar_gap_bisection',
               params={'gap_lo': 3}, sweep={'c': ['12', '2']}))
    assert any('gap bracket' in error for error in _errors(task='modular_scalar_gap_bisection',
               params={'c': '-12'}))


def test_missing_required_param_and_unknown_lam():
    for lam in (5, 5.0, ' +5 '):
        errors = _errors(params={"lam": lam})
        assert any("needs 'c'" in e for e in errors)
        assert any("has no packaged profile" in e and "7, 11, 15, 19" in e for e in errors)


def test_empty_sweep_is_rejected():
    errors = _errors(params={"lam": 7}, sweep={"c": []})
    assert any("[sweep] c is an empty list" in e for e in errors)
    with pytest.raises(ValueError):
        submit.render_sbatch(name="x", run_dir=Path("/tmp/x"), n_tasks=0, resources={}, cluster={})


def test_dry_run_refuses_an_invalid_sweep_before_writing_anything(tmp_path):
    sweep = tmp_path / "typo.toml"
    sweep.write_text('name = "y"\ntask = "scalar_gap_bisection"\n[params]\nlam = 7\nc = "8"\ntoll = "0.1"\n')
    cluster = tmp_path / "c.toml"
    cluster.write_text(f'kind = "local"\nruns_root = "{tmp_path / "runs"}"\n')
    with pytest.raises(SystemExit, match="toll"):
        submit.main([str(sweep), "--cluster", str(cluster), "--dry-run"])
    assert not (tmp_path / "runs").exists()
    for key, literal in (('c', '"1/0"'), ('spins', '[0, 1]'), ('gap_lo', '3')):
        params = {'c': '"2"', 'lam': '7', key: literal}
        sweep.write_text('name = "y"\ntask = "scalar_gap_bisection"\n[params]\n' +
                         '\n'.join(f'{name} = {value}' for name, value in params.items()) + '\n')
        with pytest.raises(SystemExit, match=key):
            submit.main([str(sweep), '--cluster', str(cluster), '--dry-run'])
        assert not (tmp_path / 'runs').exists()


def test_submission_resource_counts_validate_before_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, 'python_preflight', lambda *args: [])
    monkeypatch.setattr(submit, 'sdpb_preflight', lambda *args: [])
    sweep = tmp_path / 'sweep.toml'
    cluster = tmp_path / 'cluster.toml'
    root = tmp_path / 'runs'
    cluster.write_text(f'kind = "local"\nruns_root = "{root}"\n')
    for stanza in ('resources = "large"', '[resources]\ncpus_per_task = 4.5',
                   '[resources]\ncpus_per_task = 0'):
        sweep.write_text('task = "scalar_gap_bisection"\n' + stanza + '\n[params]\nc = "2"\n')
        with pytest.raises(SystemExit, match='resources|cpus_per_task'):
            submit.create_run(sweep, cluster, dry_run=True)
        assert not root.exists()
    sweep.write_text('task = "scalar_gap_bisection"\n[params]\nc = "2"\n')
    for literal in ('true', '-1'):
        cluster.write_text(f'kind = "local"\nruns_root = "{root}"\nmax_array_throttle = {literal}\n')
        with pytest.raises(ValueError, match='max_array_throttle'):
            submit.create_run(sweep, cluster, dry_run=True)
        assert not root.exists()
        cluster.write_text(f'kind = "slurm"\nruns_root = "{root}"\nmax_array_throttle = {literal}\n')
        with monkeypatch.context() as transport:
            transport.setattr(submit, '_remote_expand',
                              lambda *args: pytest.fail('invalid profile reached remote transport'))
            with pytest.raises(ValueError, match='max_array_throttle'):
                submit.submit_remote(sweep, cluster, host='example', dry_run=True)
    cluster.write_text(f'kind = "local"\nruns_root = "{root}"\n')
    for index, literal in enumerate(('4', '" +04 "')):
        sweep.write_text('task = "scalar_gap_bisection"\n[resources]\ncpus_per_task = ' +
                         literal + '\n[params]\nc = "2"\n')
        run = submit.create_run(sweep, cluster, dry_run=True, run_id=f'valid_{index}')
        assert '#SBATCH --cpus-per-task=4\n' in (run / 'job.sbatch').read_text()
        assert json.loads((run / 'manifest.json').read_text())['n_tasks'] == 1
    for invalid in (1.5,):
        with pytest.raises(ValueError, match='n_tasks'):
            submit.render_sbatch(name='example', run_dir=root, n_tasks=invalid,
                                 resources={}, cluster={})
    for cap in (None, 0):
        script = submit.render_sbatch(name='example', run_dir=root, n_tasks=2,
                                     resources={}, cluster={'max_array_throttle': cap})
        assert '#SBATCH --array=0-1\n' in script
    for cap in (2, ' +02 '):
        script = submit.render_sbatch(name='example', run_dir=root, n_tasks=2,
                                     resources={}, cluster={'max_array_throttle': cap})
        assert '#SBATCH --array=0-1%2\n' in script
    retry = tmp_path / 'retry'
    retry.mkdir()
    manifest_path = retry / 'manifest.json'
    submissions = []
    monkeypatch.setattr(submit, '_pending_indices', lambda *args: [0, 1])
    monkeypatch.setattr(submit, '_sbatch', lambda args, path: submissions.append(args) or '123')
    for cap in (None, 0, 2):
        manifest_path.write_text(json.dumps({'n_tasks': 2, 'cluster': {'max_array_throttle': cap}}))
        submit.retry_run(retry)
        expected = '--array=0,1' + ('%2' if cap == 2 else '')
        assert submissions[-1] == [expected, str(retry / 'job.sbatch')]
    count = len(submissions)
    for cap in (True, -1):
        manifest_path.write_text(json.dumps({'n_tasks': 2, 'cluster': {'max_array_throttle': cap}}))
        with pytest.raises(ValueError, match='max_array_throttle'):
            submit.retry_run(retry)
        assert len(submissions) == count
    for count in (1.5, 0):
        manifest_path.write_text(json.dumps({'n_tasks': count}))
        with pytest.raises(ValueError, match='n_tasks'):
            submit.retry_run(retry)


def test_retry_skips_tasks_still_queued_or_running(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'manifest.json').write_text(json.dumps({'n_tasks': 3, 'cluster': {}, 'slurm_job_id': '500'}))
    submissions = []
    monkeypatch.setattr(submit, '_pending_indices', lambda *args: [0, 1, 2])
    monkeypatch.setattr(submit, '_sbatch', lambda args, path: submissions.append(args) or '600')
    monkeypatch.setattr(submit.shutil, 'which', lambda name: '/usr/bin/squeue')
    # squeue -r -o "%F %K": elements 0 and 2 of the original array, and another user's job
    monkeypatch.setattr(submit.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(
        args, 0, stdout='500 0\n500 2\n777 1\n'))
    submit.retry_run(run)
    assert submissions == [['--array=1', str(run / 'job.sbatch')]]
    assert 'still queued or running: [0, 2]' in capsys.readouterr().out
    assert json.loads((run / 'manifest.json').read_text())['slurm_job_ids'] == ['500', '600']


def test_cluster_profiles_refuse_bad_shapes_before_creating_or_staging_runs(tmp_path, monkeypatch):
    sdpb_preflight = submit.sdpb_preflight
    monkeypatch.setattr(submit, 'python_preflight', lambda *args: [])
    monkeypatch.setattr(submit, 'sdpb_preflight', lambda *args: [])
    monkeypatch.setattr(submit, '_remote_expand',
                        lambda *args: pytest.fail('invalid profile reached remote staging'))
    sweep = tmp_path / 'sweep.toml'
    sweep.write_text('task = "scalar_gap_bisection"\n[params]\nc = "2"\n')
    profile = tmp_path / 'cluster.toml'
    root = tmp_path / 'runs'
    cases = (
        ('kind = "loacl"', 'kind'), ('modules = "gcc"', 'modules'),
        ('setup = [true]', 'setup'), ('sbatch_extra = "--exclusive"', 'sbatch_extra'),
        ('[env]\n"bad-name" = "value"', 'env'), ('runs_root = ""', 'runs_root'),
        ('remote_repo = true', 'remote_repo'),
    )
    for stanza, field in cases:
        prefix = '' if field == 'runs_root' else f'runs_root = "{root}"\n'
        profile.write_text(prefix + stanza + '\n')
        with pytest.raises(ValueError, match=field):
            submit.create_run(sweep, profile, dry_run=True)
        assert not root.exists()
        with pytest.raises(ValueError, match=field):
            submit.submit_remote(sweep, profile, host='example', dry_run=True)
    monkeypatch.setattr(submit, 'sdpb_preflight', sdpb_preflight)
    profile.write_text(f'kind = "local"\nruns_root = "{root}"\n[env]\nVB_SDPB_MODE = "bogus"\n')
    with pytest.raises(ValueError, match='bogus'):
        submit.create_run(sweep, profile, dry_run=True)
    assert not root.exists()
    for stanza in ('recorded_at = 2026-10-06T10:00:00Z', '[metadata]\nrecorded_at = [2026-10-06]'):
        profile.write_text(f'kind = "slurm"\nruns_root = "{root}"\n{stanza}\n')
        with pytest.raises(ValueError, match='cluster.toml.*JSON'):
            submit.create_run(sweep, profile, dry_run=True)
        with pytest.raises(ValueError, match='cluster.toml.*JSON'):
            submit.submit_remote(sweep, profile, host='example', dry_run=True)
        assert not root.exists()
    profile.write_text('partition = "2026-10-06T10:00:00Z"\n')
    assert submit.load_toml(profile)['partition'] == '2026-10-06T10:00:00Z'
    sweep.write_text('task = "scalar_gap_bisection"\n[params]\nc = "2"\n'
                     '[params.sdpb_options]\nrecorded_at = 2026-10-06\n')
    with pytest.raises(ValueError, match='sweep.toml.*JSON'):
        submit.submit_remote(sweep, profile, host='example', dry_run=True)


def test_every_builtin_task_is_introspectable():
    from virasoro_bootstrap.hpc import tasks as tasks_mod

    builtin = [n for n, fn in TASKS.items() if fn.__module__ == tasks_mod.__name__]
    assert builtin
    for name in builtin:
        sig = hpc_params.task_signature(name)
        assert sig is not None, f"{name}: parameters could not be read off its implementation"
        assert "lam" in sig.params
        if "c" in sig.params:
            assert sig.params["c"].kind == "real", name
    gap = hpc_params.task_signature("scalar_gap_bisection").params
    assert gap["c"].required and gap["c"].kind == "real"
    assert gap["tol"].default == '"0.05"' and gap["spins"].kind == "list[int]"
    assert gap["hotstart"].kind == "bool" and gap["sdpb_options"].kind == "mapping"
    # keys read in a helper the task hands p to are found too
    moment = hpc_params.task_signature("moment_nu1_over_nu0").params
    assert {"large_spin_twists", "shared_sample_points", "delta_ext"} <= set(moment)


def test_cluster_docs_task_table_is_generated_from_the_tasks():
    text = (REPO / "docs" / "CLUSTER.md").read_text()
    match = re.search(
        r"<!-- BEGIN GENERATED TASK TABLE -->\n(.*)<!-- END GENERATED TASK TABLE -->", text, re.S
    )
    assert match, "docs/CLUSTER.md lost its generated task table"
    assert match.group(1) == hpc_params.markdown_table(), (
        "docs/CLUSTER.md task table is stale: regenerate with python -m virasoro_bootstrap.hpc.params"
    )
    assert ", ".join(str(k) for k in sorted(LAMBDA_PRESETS)) in text
    assert "vb-certify" in text


# --- 4. run ids resolve under runs_root ---------------------------------------------------

def test_status_and_collect_accept_a_run_id(tmp_path, capsys):
    roots = tmp_path / "runs"
    run_dir = _one_task_run(roots, "fake_whatever")
    target = roots / "20260928-120000-gap"
    run_dir.rename(target)
    assert collect.resolve_run_dir_arg("20260928-120000-gap", roots) == target
    capsys.readouterr()
    assert collect.status_main(["20260928-120000-gap", "--runs-root", str(roots)]) == 0
    assert "n_tasks=1" in capsys.readouterr().out
    assert collect.collect_main(["20260928-120000-gap", "--runs-root", str(roots)]) == 0
    assert (target / "results.json").exists()
    with pytest.raises(SystemExit, match="neither a run directory nor a run id"):
        collect.resolve_run_dir_arg("no-such-run", roots)


# --- 5. certification finds an SDP directory not named sdp -------------------------------

def test_certification_uses_the_runs_actual_sdp_dir(tmp_path):
    from virasoro_bootstrap.certification import _reconstruct_full_y, _sdp_dir_for_output

    out = tmp_path / "sdpb_out"
    out.mkdir()
    custom = tmp_path / "sdp_custom"
    custom.mkdir()
    (custom / "control.json").write_text("{}")
    (custom / "normalization.json").write_text(json.dumps({"normalization": ["1", "2", "0.5"]}))
    assert _sdp_dir_for_output(out) == custom
    # the eliminated component (largest |n|) is recovered from n . y = 1
    full = _reconstruct_full_y([mp.mpf(1), mp.mpf(2)], out)
    assert len(full) == 3 and mp.almosteq(full[0] * 1 + full[1] * 2 + full[2] * mp.mpf("0.5"), 1)

    # the SDPB log's record wins over a stray directory, and a plain sdp over a guess
    other = tmp_path / "sdp_other"
    other.mkdir()
    (other / "control.json").write_text("{}")
    (tmp_path / "sdpb.log").write_text('SDP directory   : "sdp_other"\n')
    assert _sdp_dir_for_output(out) == other
    (tmp_path / "sdpb.log").unlink()
    plain = tmp_path / "sdp"
    plain.mkdir()
    (plain / "control.json").write_text("{}")
    assert _sdp_dir_for_output(out) == plain

    # Distinct solves can share a parent, including concatenated SDPB logs.
    shared = tmp_path / "shared"
    shared.mkdir()
    logs = []
    for suffix, normalization in (("a", ["1", "0"]), ("b", ["0", "1"])):
        inputs = shared / f"sdp_{suffix}"
        inputs.mkdir()
        (inputs / "control.json").write_text("{}")
        (inputs / "normalization.json").write_text(json.dumps({"normalization": normalization}))
        (shared / f"out_{suffix}").mkdir()
        text = f'SDP directory   : "sdp_{suffix}"\nout directory   : "out_{suffix}"\n'
        log = shared / f"{suffix}.log"
        log.write_text(text)
        logs.append(log)
    second = shared / "out_b"
    assert _sdp_dir_for_output(second) == shared / "sdp_b"
    assert _reconstruct_full_y([mp.mpf(2)], second) == [2, 1]
    combined = "".join(log.read_text() for log in logs)
    for log in logs:
        log.unlink()
    (shared / "combined.log").write_text(combined)
    assert _sdp_dir_for_output(shared / "out_a") == shared / "sdp_a"
    assert _sdp_dir_for_output(second) == shared / "sdp_b"
    (shared / "conflicting.log").write_text(
        'SDP directory : "sdp_a"\nout directory : "out_b"\n')
    with pytest.raises(ValueError, match="ambiguous SDP directories"):
        _sdp_dir_for_output(second)
    (shared / "conflicting.log").unlink()
    # SDPB quotes and escapes path names in its log.
    quoted = shared / 'sdp "with spaces"'
    (shared / "sdp_b").rename(quoted)
    (shared / "combined.log").write_text(
        f'SDP directory : {json.dumps(quoted.name)}\nout directory : "out_b"\n')
    assert _sdp_dir_for_output(second) == quoted


def test_solver_source_record_binds_nested_outputs_and_the_launched_basis(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from virasoro_bootstrap import identical_scalar as mod
    from virasoro_bootstrap.certification import _sdp_dir_for_output
    from virasoro_bootstrap.hpc.runner import SdpbRunner
    from virasoro_bootstrap.hpc.storage import SDP_SOURCE_NAME

    doc = {"objective": ["0", "1"], "normalization": ["1", "0"],
           "PositiveMatrixWithPrefactorArray": [{
               "polynomials": [[[["1"], ["2"]]]],
               "samplePoints": ["0"], "sampleScalings": ["1"]}]}
    problem = SimpleNamespace(spec=SimpleNamespace(dps=40),
                              pmp_build=SimpleNamespace(doc=doc), sample_values_by_block=None)

    def write_problem(problem, output):
        output.mkdir(parents=True, exist_ok=True)
        (output / "pmp.json").write_text(json.dumps(problem.pmp_build.doc))
        return output / "manifest.json", output / "pmp.json"

    outcome = [0]
    launched = []

    def solve(command, log_path, *, check=True, cwd=None):
        assert log_path is None
        output = cwd / command[command.index("-o") + 1]
        source = cwd / command[command.index("-s") + 1]
        record = json.loads((output / SDP_SOURCE_NAME).read_text())
        assert record["state"] == "running"
        with pytest.raises(ValueError, match="no completed attempt"):
            _sdp_dir_for_output(output)
        launched.append(record["metadata_sha256"])
        if outcome[0] == "change":
            (source / "normalization.json").write_text('{"normalization":["0","1"]}')
            return 0
        if outcome[0] == "interrupt":
            raise RuntimeError("interrupted solver")
        return outcome[0]

    monkeypatch.setattr(mod, "write_identical_scalar_gap_problem", write_problem)
    monkeypatch.setattr(mod, "_run_command", solve)
    monkeypatch.setattr(mod.shutil, "which", lambda name: str(name))
    root = tmp_path / "run"

    def run():
        return mod.run_identical_scalar_gap_problem(
            problem, root, runner=SdpbRunner(mode="native", bin_dir="/unused"),
            sdp_dir_name="inputs/custom", sdpb_out_dir_name="outputs/custom",
            pmp2sdp_log_name=None, sdpb_log_name=None, eval_dps=40)

    result = run()
    out = result.sdpb_out_dir
    assert _sdp_dir_for_output(out).resolve() == result.sdp_dir.resolve()
    saved = json.loads((out / SDP_SOURCE_NAME).read_text())
    assert saved["state"] == "completed" and saved["metadata_sha256"] == launched[-1]
    previous_doc = {**doc, "objective": ["0", "2"]}
    (root / "pmp.json").write_text(json.dumps(previous_doc))
    previous_files = {path.relative_to(root): path.read_bytes()
                      for path in root.rglob("*") if path.is_file()}
    for invalid in ({"backend": "typo"}, {"sdpb_options": {"--precision": 0}}, {"eval_dps": True}):
        with pytest.raises(ValueError):
            mod.run_identical_scalar_gap_problem(
                problem, root, runner=SdpbRunner(mode="native", bin_dir="/unused"), **invalid)
        assert {path.relative_to(root): path.read_bytes()
                for path in root.rglob("*") if path.is_file()} == previous_files
        assert len(launched) == 1
    # The relative reference survives moving a whole run, without logs.
    moved = tmp_path / "moved"
    root.rename(moved)
    moved_out = moved / "outputs/custom"
    assert _sdp_dir_for_output(moved_out).resolve() == (moved / "inputs/custom").resolve()
    moved.rename(root)
    for value in (1, "change", "interrupt"):
        outcome[0] = value
        if value == "interrupt":
            with pytest.raises(RuntimeError, match="interrupted solver"):
                run()
        else:
            result = run()
            assert result.sdpb_status == ("solver_error" if value == 1 else None)
        saved = json.loads((out / SDP_SOURCE_NAME).read_text())
        assert saved["metadata_sha256"] == launched[-1]
        with pytest.raises(ValueError, match="no completed attempt|metadata changed since solver launch"):
            _sdp_dir_for_output(out)
    # A broken record link must not silently fall back to another SDP.
    (out / SDP_SOURCE_NAME).unlink()
    (out / SDP_SOURCE_NAME).symlink_to("missing.json")
    with pytest.raises(ValueError, match="symlinked SDP source record"):
        _sdp_dir_for_output(out)


# --- 6. --flag=value SDPB options ----------------------------------------------------------

def test_split_sdpb_options_accepts_equals_forms():
    from virasoro_bootstrap.identical_scalar import _split_sdpb_options

    bits, procs, flags = _split_sdpb_options(
        ["--precision=640", "--findDualFeasible", "--maxIterations=50", "--mpiProcs=2"], 300
    )
    assert (bits, procs) == (640, 2)
    assert flags == ["--findDualFeasible", "--maxIterations=50"]
    bits, procs, flags = _split_sdpb_options({"--precision=512": True, "--mpiProcs": 4}, 300)
    assert (bits, procs, flags) == (512, 4, [])
    assert _split_sdpb_options(["--precision", "704"], 300)[0] == 704
    with pytest.raises(ValueError, match="--precision needs an integer"):
        _split_sdpb_options(["--precision=high"], 300)
    for value in ("--precision=512", 0):
        with pytest.raises(ValueError, match="sdpb_options must be"):
            _split_sdpb_options(value, 300)


# --- 7. vb-certify reports where a gap-path channel fails -------------------------------

def test_vb_certify_names_the_gap_path_failure_location(capsys):
    from virasoro_bootstrap.hpc.recertify import _summarize

    details = [{"ell": 2, "passed": False, "min_value": -1.5e-3, "min_delta": 4.25,
                "n_negative": 3, "n_test_points": 100}]
    info = {"probe": "gap_1p0", "probe_value": mp.mpf(1)}
    assert _summarize(False, details, info) is False
    out = capsys.readouterr().out
    assert "None" not in out
    assert "Delta~4.25" in out and "-0.0015" in out


def test_vb_certify_incomplete_checks_fail_without_claiming_negativity(tmp_path, monkeypatch, capsys):
    from virasoro_bootstrap.hpc import recertify

    (tmp_path / "manifest.json").write_text('{"n_tasks": 1}')
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs/0000.json").write_text(json.dumps({
        "task": "scalar_gap_bisection", "params": {}, "output_dir": "task",
    }))
    (tmp_path / "task/gap_1p0/sdpb_out").mkdir(parents=True)
    info = {"probe": "gap_1p0", "probe_value": None}
    for details in ([{"passed": False, "reason": "direct_sdp_dimension_mismatch"}], [],
                    [{"passed": True, "exact_positivity_undecided": "no_full_y_vector"}]):
        # Even a contradictory success flag cannot turn unchecked channels
        # into a valid functional or a witnessed negative channel.
        monkeypatch.setattr(recertify, "certify_probe", lambda *args, **kwargs: (True, details, info))
        assert recertify.main([str(tmp_path), "--task", "0", "--all-probes"]) == 1
        output = capsys.readouterr().out
        assert "NOT PROVEN" in output and "NEGATIVE" not in output
    for detail in ({"passed": False, "reason": "non_finite_dual_vector"},
                   {"passed": False, "reason": "normalization_violated", "normalization_value": 0}):
        assert not recertify._summarize(False, [detail], info)
        output = capsys.readouterr().out
        assert "INVALID functional" in output and "NEGATIVE" not in output
    assert recertify._summarize(True, [{"passed": True}], info)
    assert "VALID functional" in capsys.readouterr().out


# --- 8. help texts are for users ------------------------------------------------------------

@pytest.mark.parametrize("main, prog", [
    (submit.main, "vb-submit"),
    (collect.status_main, "vb-status"),
    (collect.collect_main, "vb-collect"),
    (collect.list_main, "vb-list"),
    ("recertify", "vb-certify"),
])
def test_cli_help_is_user_facing(main, prog, capsys):
    if main == "recertify":
        from virasoro_bootstrap.hpc.recertify import main
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert text.startswith(f"usage: {prog}")
    assert "examples:" in text
    for forbidden in ("``", "PROBLEM_BUILDERS", ":func:"):
        assert forbidden not in text, forbidden
    assert len(text.splitlines()) < 60


def test_task_descriptions_cite_no_private_runs(capsys):
    assert submit.main(["--list-tasks"]) == 0
    text = capsys.readouterr().out
    assert "scalar_gap_bisection" in text and "toll" not in text
    for name in sorted(TASKS):
        submit.main(["--list-tasks", name])
    text += capsys.readouterr().out
    assert not re.search(r"\brun 20\d{6}-\d{6}\b", text)


def test_probe_labels_distinguish_close_values_and_fit_filename_limits():
    from virasoro_bootstrap.hpc.tasks import _label

    with mp.workdps(1000):
        first = mp.mpf(1) + mp.mpf("1e-900")
        second = mp.mpf(1) + mp.mpf("2e-900")
        labels = (_label(first), _label(second))
        assert labels[0] != labels[1]
        assert all(len("degeneracy_" + label) < 255 for label in labels)
        assert _label(mp.mpf(".5")) == "0p5"
    with mp.workdps(15):
        assert (_label(first), _label(second)) == labels


@pytest.mark.parametrize("output_path", ["tasks/0000", "absolute"])
def test_last_probe_selection_uses_manifest_value_without_ambient_rounding(tmp_path, monkeypatch, output_path):
    from virasoro_bootstrap.hpc import recertify

    (tmp_path / "manifest.json").write_text('{"n_tasks": 1}')
    (tmp_path / "specs").mkdir()
    task_dir = tmp_path / ("custom/shared" if output_path == "absolute" else output_path)
    (tmp_path / "specs/0000.json").write_text(json.dumps({
        "task": "scalar_gap_bisection", "params": {},
        "output_dir": str(task_dir) if output_path == "absolute" else output_path,
    }))
    for name, gap in [("gap_1p0~aaa", "1.0000000000000000002"),
                      ("gap_1p0~zzz", "1.0000000000000000001")]:
        probe = task_dir / name
        (probe / "sdpb_out").mkdir(parents=True)
        (probe / "sdpb_out/y.txt").write_text("0 1\n")
        (probe / "manifest.json").write_text(json.dumps({"spec": {"delta_gap": gap}}))
    unfinished = task_dir / "gap_9p0"   # launched, never finished: not the last finished probe
    (unfinished / "sdpb_out").mkdir(parents=True)
    (unfinished / "manifest.json").write_text(json.dumps({"spec": {"delta_gap": "9"}}))
    selected = []
    def certify(run_dir, task, probe, **kwargs):
        selected.append(probe)
        return True, [], {"probe": probe, "probe_value": None}
    monkeypatch.setattr(recertify, "certify_probe", certify)
    monkeypatch.setattr(recertify, "_summarize", lambda *args: True)
    with mp.workdps(15):
        assert recertify.main([str(tmp_path), "--task", "0", "--last-probe"]) == 0
    assert selected == ["gap_1p0~aaa"]


def test_vb_certify_skips_unfinished_probes_and_continues(tmp_path, capsys):
    from virasoro_bootstrap.hpc import recertify

    (tmp_path / "manifest.json").write_text('{"n_tasks": 1}')
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs/0000.json").write_text(json.dumps({
        "task": "scalar_gap_bisection", "params": {}, "output_dir": "task",
    }))
    running = tmp_path / "task/gap_1p0/sdpb_out"
    running.mkdir(parents=True)
    (running / "vb_sdp_source.json").write_text(json.dumps({
        "schema_version": 1, "state": "running", "sdp_directory": "../sdp", "metadata_sha256": {}}))
    (tmp_path / "task/gap_2p0/sdpb_out").mkdir(parents=True)   # no y.txt
    assert recertify.main([str(tmp_path), "--task", "0", "--all-probes"]) == 1
    out = capsys.readouterr().out
    assert "gap_1p0" in out and "SDPB attempt running" in out
    assert "gap_2p0" in out and "no y.txt" in out


@pytest.mark.parametrize("args", [["--task", "0"], ["--all-tasks"]])
def test_recertification_does_not_report_success_when_no_probes_were_checked(tmp_path, args):
    from virasoro_bootstrap.hpc.recertify import main

    (tmp_path / "manifest.json").write_text('{"n_tasks": 1}')
    (tmp_path / "specs").mkdir()
    assert main([str(tmp_path), *args]) != 0


def test_certification_task_selection_uses_manifest_and_normalizes_cluster_path(tmp_path, monkeypatch, capsys):
    from virasoro_bootstrap.hpc import recertify

    run = tmp_path / 'runs/run'
    specs = run / 'specs'
    specs.mkdir(parents=True)
    manifest = run / 'manifest.json'
    manifest.write_text(json.dumps({'n_tasks': 2}))
    spec = {'task': 'scalar_gap_bisection', 'params': {}}
    (specs / '0000.json').write_text(json.dumps(spec))
    (run / 'tasks/0000/gap_1p0/sdpb_out').mkdir(parents=True)
    checked = []
    def certify(run_dir, task, probe, **kwargs):
        checked.append(task)
        return True, [{'passed': True}], {'probe': probe, 'probe_value': None}
    monkeypatch.setattr(recertify, 'certify_probe', certify)
    assert recertify.main([str(run), '--all-tasks']) == 1
    assert checked == [0] and 'task 0001: SKIPPED' in capsys.readouterr().out
    (specs / '0001.json').write_text(json.dumps(spec))
    (run / 'tasks/0001/gap_1p0/sdpb_out').mkdir(parents=True)
    (specs / 'notes.json').write_text('{}')
    (specs / '0020.json').write_text(json.dumps(spec))
    checked.clear()
    assert recertify.main([str(run), '--all-tasks']) == 0
    assert checked == [0, 1]
    checked.clear()
    for index in ('2', '20'):
        with pytest.raises(SystemExit) as exc:
            recertify.main([str(run), '--task', index])
        assert exc.value.code == 2 and 'task index' in capsys.readouterr().err
    assert not checked
    cluster = tmp_path / 'cluster/profile.toml'
    cluster.parent.mkdir()
    cluster.write_text(f'kind = "local"\nruns_root = "{run.parent}"\n')
    monkeypatch.setenv('VB_RESEARCH_ROOT', str(tmp_path))
    assert recertify.main(['run', '--all-tasks', '--cluster', 'clusters/profile.toml']) == 0
    assert checked == [0, 1]


def test_inspection_and_certification_reject_bad_cli_controls_before_loading_runs(monkeypatch, capsys):
    from virasoro_bootstrap.hpc import recertify

    monkeypatch.setattr(recertify, 'resolve_run_dir_arg',
                        lambda *args: pytest.fail('invalid controls reached run loading'))
    monkeypatch.setattr(collect, '_local_runs_root',
                        lambda *args: pytest.fail('invalid controls reached profile loading'))
    for flags in ([], ['--task', '0', '--all-tasks'], ['--task', '0', '--extra-spins', '-1'],
                  ['--task', '0', '--probe', 'named', '--last-probe']):
        with pytest.raises(SystemExit) as exc:
            recertify.main(['absent', *flags])
        assert exc.value.code == 2 and 'error:' in capsys.readouterr().err
    for main, argv in ((collect.list_main, ['--limit', '-1']),
                       (collect.status_main, ['absent', '--requeue-warn', '-1'])):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2 and 'error:' in capsys.readouterr().err


def test_tiny_gap_bisection_keeps_distinct_probes_and_resumes_them(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc.tasks import scalar_gap_bisection

    with mp.workdps(80):
        unit = mp.mpf("1e-60")
        text = lambda x: mp.nstr(x, 85)
        bound = text(1 + 5 * unit)
        params = dict(_GAP_PARAMS, dps=80, gap_lo=text(1 + unit),
                      gap_hi=text(1 + 9 * unit), tol="1e-61")
    calls = _fake_solver(monkeypatch, bound=bound)
    result = scalar_gap_bisection(_ctx(tmp_path, params))
    assert result.get("reason") is None, result
    with mp.workdps(80):
        lo, hi = mp.mpf(result["delta_gap_lo"]), mp.mpf(result["delta_gap_hi"])
        assert lo < hi
        assert lo <= mp.mpf(bound) <= hi
        assert hi - lo <= mp.mpf(params["tol"])
    count = len(calls)
    assert count > 3
    assert len({name for name, _, _ in calls}) == count
    again = scalar_gap_bisection(_ctx(tmp_path, params))
    assert again.get("reason") is None, again
    assert len(calls) == count  # all completed probes came from the journal


def test_saved_result_number_survives_a_lower_ambient_precision():
    from fractions import Fraction
    from virasoro_bootstrap.hpc.tasks import _result_number

    with mp.workdps(100):
        value = mp.mpf(1) + mp.mpf("1e-90")
    with mp.workdps(15):
        encoded = _result_number(value)
        assert _result_number(Fraction(7, 10)) == "0.7"
        assert _result_number(Fraction(1, 3)) == "1/3"
    with mp.workdps(100):
        assert mp.mpf(encoded) == value


@pytest.mark.parametrize("tol,task_name,tol_key,extra", [
    ("0", "scalar_gap_bisection", "tol", {}),
    ("nan", "isolated_relevant_scalar_delta_phi_intervals", "delta_phi_tol", {"delta_phi_grid": [".1", ".2"]}),
    ("-0.1", "isolated_relevant_scalar_degeneracy_intervals", "degeneracy_tol",
     {"delta_phi": ".1", "degeneracy_grid": ["0", "1"]}),
])
def test_bisection_rejects_invalid_tolerance_before_build(tmp_path, monkeypatch, tol, task_name, tol_key, extra):
    from virasoro_bootstrap.hpc import tasks
    import virasoro_bootstrap.modular_crossing as modular
    import virasoro_bootstrap.identical_scalar as scalar

    def unexpected(*args, **kwargs):
        pytest.fail("invalid bisection inputs reached a block build")
    monkeypatch.setattr(modular, "prepare_modular_gap_build", unexpected)
    monkeypatch.setattr(scalar, "prepare_identical_scalar_gap_build", unexpected)
    with pytest.raises(ValueError, match="finite and positive"):
        getattr(tasks, task_name)(_ctx(tmp_path, dict(_GAP_PARAMS, **extra, **{tol_key: tol})))


def test_bisection_reports_insufficient_precision_instead_of_stalling(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc import tasks

    _fake_solver(monkeypatch)
    original_label = tasks._label
    evaluations = 0
    def guarded_label(value):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 200:
            raise RuntimeError("audit watchdog: bisection stalled")
        return original_label(value)
    monkeypatch.setattr(tasks, "_label", guarded_label)
    result = tasks.scalar_gap_bisection(_ctx(tmp_path, dict(_GAP_PARAMS, dps=15, tol="1e-100")))
    assert result["reason"] == "solve_error"
    assert "increase dps" in result["error"]
    assert evaluations < 200


@pytest.mark.parametrize("task_name,lo,hi", [
    ("scalar_gap_bisection", "2", "1"), ("modular_scalar_gap_bisection", "nan", "2"),
])
def test_gap_bisection_rejects_invalid_bracket_before_build(tmp_path, task_name, lo, hi):
    from virasoro_bootstrap.hpc import tasks

    with pytest.raises(ValueError, match="gap bracket must be finite"):
        getattr(tasks, task_name)(_ctx(tmp_path, dict(_GAP_PARAMS, gap_lo=lo, gap_hi=hi)))


@pytest.mark.parametrize("task_name,grid_key,value", [
    ("isolated_relevant_scalar_delta_phi_intervals", "delta_phi_grid", "nan"),
    ("isolated_relevant_scalar_degeneracy_intervals", "degeneracy_grid", "inf"),
])
def test_interval_bisection_rejects_nonfinite_grid(tmp_path, task_name, grid_key, value):
    from virasoro_bootstrap.hpc import tasks

    with pytest.raises(ValueError, match="entries must be finite"):
        getattr(tasks, task_name)(_ctx(tmp_path, dict(_GAP_PARAMS, delta_phi=".1", **{grid_key: [value]})))


@pytest.mark.parametrize("flag,value,form", [
    ("--precision", "0", "pair"), ("--mpiProcs", "", "equals"), ("--precision", "-1", "mapping"),
])
def test_solver_integer_options_reject_nonpositive_or_empty_values(flag, value, form):
    from virasoro_bootstrap.identical_scalar import _split_sdpb_options

    options = {"pair": [flag, value], "equals": [f"{flag}={value}"], "mapping": {flag: value}}[form]
    with pytest.raises(ValueError, match=flag):
        _split_sdpb_options(options, 300)


@pytest.mark.parametrize("flag", ["--mpiProcs"])
def test_solver_integer_options_reject_missing_values(flag):
    from virasoro_bootstrap.identical_scalar import _split_sdpb_options

    with pytest.raises(ValueError, match=flag):
        _split_sdpb_options([flag], 300)


@pytest.mark.parametrize("flag,setting", [("--precision", 0), ("--mpiProcs", "-2")])
def test_sweep_rejects_nonpositive_solver_counts(flag, setting):
    errors = hpc_params._check_sdpb_options({flag: setting})
    assert errors and flag in errors[0]


@pytest.mark.parametrize("flag,setting", [("--mpiProcs", False)])
def test_sweep_accepts_disabled_solver_options(flag, setting):
    from virasoro_bootstrap.identical_scalar import _split_sdpb_options

    assert hpc_params._check_sdpb_options({flag: setting}) == []
    assert _split_sdpb_options({flag: setting}, 300) == (300, None, [])


@pytest.mark.parametrize("flag", ["--precision="])
def test_sweep_rejects_empty_inline_solver_count(flag):
    assert hpc_params._check_sdpb_options({flag: 512})


@pytest.mark.parametrize("nested", [
    {"lower_status": "solve_error", "lower": None},
    {"lower": {"error": "missing saved manifest"}},
])
def test_nested_moment_errors_are_not_final_results(nested):
    from virasoro_bootstrap.hpc.tasks import result_error

    payload = {"c": "3", "include": nested}
    assert result_error(payload) is not None
    document = {"spec_hash": "digest", "result": payload}
    assert not worker.result_doc_is_final(document, "digest")


@pytest.mark.parametrize("status", ["optimal_uncertified"])
def test_completed_moment_verdicts_remain_final(status):
    from virasoro_bootstrap.hpc.tasks import result_error

    assert result_error({"include": {"lower_status": status, "lower": None}}) is None
    assert result_error({"include": {"lower": {"certified": False}}}) is None


def test_atomic_write_rides_out_a_transient_quota_error(tmp_path, monkeypatch):
    import errno

    calls = {"n": 0}
    real = Path.write_text

    def flaky(self, text, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    worker._atomic_write_json(tmp_path / "status.json", {"state": "done"})
    assert json.loads((tmp_path / "status.json").read_text()) == {"state": "done"} and calls["n"] == 3


def test_atomic_write_raises_other_errors_at_once(tmp_path, monkeypatch):
    import errno

    def denied(self, text, *args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(Path, "write_text", denied)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: pytest.fail("no retry for EACCES"))
    with pytest.raises(PermissionError):
        worker._atomic_write_json(tmp_path / "status.json", {})
