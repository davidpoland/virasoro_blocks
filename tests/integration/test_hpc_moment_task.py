"""moment_nu1_over_nu0 task: params/profile wiring, journal resume, error paths.

Every expensive call (block prepare, problem build, SDPB run, certification)
is monkeypatched to a cheap fake -- these tests check the task's own logic
(profile loading, overrides, per-solve journaling, interruption), not the
numerics, which the existing moment-bound tests already cover.
"""
from __future__ import annotations

from pathlib import Path

import mpmath as mp
import pytest

import virasoro_bootstrap.certification as certification_mod
import virasoro_bootstrap.identical_scalar as identical_scalar_mod
import virasoro_bootstrap.moments as moments_mod
from virasoro_bootstrap.hpc.journal import Journal
from virasoro_bootstrap.hpc.tasks import (
    TaskContext,
    TaskInterrupted,
    _load_default_params_profile,
    moment_nu1_over_nu0,
)


class _FakePrepared:
    pass


class _FakeProblem:
    def __init__(self, direction, identity_mode):
        self.direction = direction
        self.identity_mode = identity_mode


class _FakeRun:
    def __init__(self, *, status="optimal", sdpb_out_dir="fake_out", checkpoint_dir=None, log_path=None):
        self.sdpb_status = status
        self.sdpb_out_dir = sdpb_out_dir
        self.sdpb_log_path = log_path
        self.checkpoint_dir = checkpoint_dir


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Patch every expensive call with something that runs in microseconds.

    Returns a dict of call logs the tests can assert against, plus a
    `solve_status` override the test can mutate mid-run.
    """
    calls = {
        "prepare": 0, "build_lower": [], "build_upper": [], "run": [],
        "certify": 0, "certify_kwargs": [],
    }
    state = {"status": "optimal"}

    def fake_prepare(spec):
        calls["prepare"] += 1
        return _FakePrepared()

    def fake_build_lower(spec, numerator, denominator, *, identity_mode, prepared):
        calls["build_lower"].append(identity_mode)
        return _FakeProblem("lower", identity_mode)

    def fake_build_upper(spec, numerator, denominator, *, identity_mode, prepared):
        calls["build_upper"].append(identity_mode)
        return _FakeProblem("upper", identity_mode)

    def fake_run(problem, output_dir, **kwargs):
        calls["run"].append((problem.identity_mode, problem.direction))
        return _FakeRun(status=state["status"])

    def fake_read_sdpb_objective(out_dir):
        return {"dual": mp.mpf("1.5")}

    def fake_certify(problem, out_dir, **kwargs):
        calls["certify"] += 1
        calls["certify_kwargs"].append(kwargs)
        return True, [{"passed": True}]

    monkeypatch.setattr(identical_scalar_mod, "prepare_identical_scalar_gap_build", fake_prepare)
    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", fake_run)
    monkeypatch.setattr(identical_scalar_mod, "read_sdpb_objective", fake_read_sdpb_objective)
    monkeypatch.setattr(moments_mod, "build_identical_scalar_normalized_moment_lower_problem", fake_build_lower)
    monkeypatch.setattr(moments_mod, "build_identical_scalar_normalized_moment_upper_problem", fake_build_upper)
    monkeypatch.setattr(moments_mod, "normalized_moment_lower_bound_from_objective", lambda v: v)
    monkeypatch.setattr(moments_mod, "normalized_moment_upper_bound_from_objective", lambda v: v)
    monkeypatch.setattr(certification_mod, "certify_dual_from_sdpb_output", fake_certify)

    calls["state"] = state
    return calls


def _ctx(tmp_path: Path, params: dict, *, should_stop=lambda: False) -> TaskContext:
    return TaskContext(
        params=params, output_dir=tmp_path,
        journal=Journal(tmp_path / "progress.json"), should_stop=should_stop,
    )


def test_profile_loader_reads_real_config():
    profile = _load_default_params_profile(19)
    assert profile["q_order"] == 80
    assert profile["npoles"] == 60
    assert profile["spin_list"][-1] == 18


def test_moment_task_happy_path_all_four_solves(tmp_path, fake_pipeline):
    ctx = _ctx(tmp_path, {"c": "8/7", "delta_ext": "2.0", "lam": 19})
    result = moment_nu1_over_nu0(ctx)

    assert fake_pipeline["prepare"] == 1  # one prepare shared by all 4 solves
    assert sorted(fake_pipeline["run"]) == [
        ("exclude", "lower"), ("exclude", "upper"),
        ("include", "lower"), ("include", "upper"),
    ]
    assert result["q_order"] == 80  # from the real lambda19 profile
    assert result["npoles"] == 60
    for mode in ("include", "exclude"):
        assert result[mode]["lower"] == "1.5"
        assert result[mode]["upper"] == "1.5"
        assert result[mode]["lower_status"] == "optimal"
        assert isinstance(result[mode]["lower_solve_seconds"], float)


def test_moment_task_param_overrides_flow_into_spec(tmp_path, fake_pipeline, monkeypatch):
    captured_specs = []
    orig_build_lower = moments_mod.build_identical_scalar_normalized_moment_lower_problem

    def capturing_build_lower(spec, *a, **kw):
        captured_specs.append(spec)
        return orig_build_lower(spec, *a, **kw)

    monkeypatch.setattr(moments_mod, "build_identical_scalar_normalized_moment_lower_problem", capturing_build_lower)
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "q_order": 40, "npoles": 20,  # override below the profile's 80/60
        "directions": ["lower"], "identity_modes": ["include"],
    })
    result = moment_nu1_over_nu0(ctx)
    assert result["q_order"] == 40
    assert result["npoles"] == 20
    assert captured_specs[0].q_order == 40
    assert captured_specs[0].npoles == 20
    # Only the requested subset ran.
    assert fake_pipeline["run"] == [("include", "lower")]
    assert "upper" not in result["include"]


def test_moment_task_forwards_physical_certification_controls(tmp_path, fake_pipeline):
    ctx = _ctx(tmp_path, {
        "c": "1.00001", "delta_ext": "0.5", "lam": 11,
        "directions": ["upper"], "identity_modes": ["exclude"],
        "physical_moment_blocks": True,
        "force_exact_included": True,
        "interstitial_twists": 3,
        "cert_n_test_points": 240,
        "cert_delta_range": 40,
        "cert_log_near_points": 12,
    })
    moment_nu1_over_nu0(ctx)
    kwargs = fake_pipeline["certify_kwargs"][0]
    assert kwargs["physical_moment_blocks"] is True
    assert kwargs["force_exact_included"] is True
    assert kwargs["interstitial_twists"] == 3
    assert kwargs["n_test_points"] == 240
    assert kwargs["delta_range"] == 40.0
    assert kwargs["log_near_points"] == 12


def test_moment_task_mpiprocs_override_reaches_run(tmp_path, fake_pipeline, monkeypatch):
    captured_options = []

    def fake_run(problem, output_dir, *, sdpb_options, **kwargs):
        captured_options.append(sdpb_options)
        return _FakeRun()

    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", fake_run)
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "directions": ["lower"], "identity_modes": ["include"],
        "sdpb_options": {"--mpiProcs": 32},
    })
    moment_nu1_over_nu0(ctx)
    assert captured_options[0]["--mpiProcs"] == 32
    assert captured_options[0]["--precision"] == 1024  # profile default, not overridden


def test_moment_task_default_drops_no_checkpoint_flag_absent(tmp_path, fake_pipeline, monkeypatch):
    captured_options = []

    def fake_run(problem, output_dir, *, sdpb_options, **kwargs):
        captured_options.append(sdpb_options)
        return _FakeRun()

    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", fake_run)
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "directions": ["lower"], "identity_modes": ["include"],
    })
    moment_nu1_over_nu0(ctx)
    # Default: no checkpoint kept (most sweeps just want the verdict).
    assert captured_options[0]["--noFinalCheckpoint"] is True


def test_moment_task_keep_checkpoint_drops_no_final_checkpoint_flag(tmp_path, fake_pipeline, monkeypatch):
    captured_options = []

    def fake_run(problem, output_dir, *, sdpb_options, **kwargs):
        captured_options.append(sdpb_options)
        return _FakeRun()

    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", fake_run)
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "directions": ["lower"], "identity_modes": ["include"],
        "keep_checkpoint": True,
    })
    moment_nu1_over_nu0(ctx)
    assert "--noFinalCheckpoint" not in captured_options[0]


def test_moment_task_initial_checkpoint_source_dir_passed_through(tmp_path, fake_pipeline, monkeypatch):
    source = tmp_path / "prior_run"
    ck_dir = source / "include_lower" / "sdp.ck"
    ck_dir.mkdir(parents=True)

    captured = {}

    def fake_run(problem, output_dir, *, initial_checkpoint_dir, **kwargs):
        captured["initial_checkpoint_dir"] = initial_checkpoint_dir
        return _FakeRun()

    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", fake_run)
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "directions": ["lower"], "identity_modes": ["include"],
        "initial_checkpoint_source_dir": str(source),
    })
    moment_nu1_over_nu0(ctx)
    assert captured["initial_checkpoint_dir"] == ck_dir


def test_moment_task_initial_checkpoint_missing_raises_clearly(tmp_path, fake_pipeline):
    source = tmp_path / "prior_run"  # no sdp.ck under it at all
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "directions": ["lower"], "identity_modes": ["include"],
        "initial_checkpoint_source_dir": str(source),
    })
    with pytest.raises(FileNotFoundError, match="keep_checkpoint=true"):
        moment_nu1_over_nu0(ctx)


def test_moment_task_journal_resume_skips_completed_solves(tmp_path, fake_pipeline):
    params = {"c": "8/7", "delta_ext": "2.0", "lam": 19}
    ctx = _ctx(tmp_path, params)
    moment_nu1_over_nu0(ctx)
    assert len(fake_pipeline["run"]) == 4

    fake_pipeline["run"].clear()
    fake_pipeline["prepare"] = 0
    ctx2 = _ctx(tmp_path, params)  # fresh context, same output_dir -> same journal file
    result = moment_nu1_over_nu0(ctx2)
    assert fake_pipeline["run"] == []  # every solve was journaled; nothing re-run
    assert result["include"]["lower"] == "1.5"
    valid = dict(ctx2.journal.data['solved'])
    for invalid in (None, {}, {'status': 'optimal', 'value': 'nan'},
                    {**valid['include_lower'], 'value': 'nan'},
                    {**valid['include_lower'], 'value': True}):
        solved = {**valid, 'include_lower': invalid}
        ctx2.journal.update(solved=solved)
        fake_pipeline['run'].clear()
        retried = moment_nu1_over_nu0(_ctx(tmp_path, params))
        assert fake_pipeline['run'] == [('include', 'lower')]
        assert retried['include']['lower'] == '1.5'


@pytest.mark.parametrize('old_setting', [{'large_spin_lmin': 20}, {'q_order': 78}, None])
def test_moment_resume_rebuilds_all_bounds_after_default_changes(
    tmp_path, fake_pipeline, monkeypatch, old_setting,
):
    from dataclasses import replace
    import virasoro_bootstrap.hpc.tasks as tasks

    params = {'c': '8/7', 'delta_ext': '2.0', 'lam': 19, 'spins': list(range(0, 25, 2)),
              'identity_modes': ['include'], 'directions': ['lower', 'upper']}
    current_builder = tasks._moment_nu1_over_nu0_spec
    stopped = [False]
    built = []
    for direction in ('lower', 'upper'):
        name = f'build_identical_scalar_normalized_moment_{direction}_problem'
        original = getattr(moments_mod, name)
        def record(spec, *args, _original=original, _direction=direction, **kwargs):
            built.append((_direction, spec.large_spin_lmin, spec.q_order))
            stopped[0] = True
            return _original(spec, *args, **kwargs)
        monkeypatch.setattr(moments_mod, name, record)
    if old_setting is not None:
        monkeypatch.setattr(tasks, '_moment_nu1_over_nu0_spec',
                            lambda *a, **kw: replace(current_builder(*a, **kw), **old_setting))
    ctx = _ctx(tmp_path, params, should_stop=lambda: stopped[0])
    with pytest.raises(TaskInterrupted):
        moment_nu1_over_nu0(ctx)
    assert len(built) == 1
    if old_setting is None:
        # A pre-provenance journal cannot establish which defaults it used.
        ctx.journal.data.pop('resolved_settings')
        ctx.journal.update()
    monkeypatch.setattr(tasks, '_moment_nu1_over_nu0_spec', current_builder)
    result = moment_nu1_over_nu0(_ctx(tmp_path, params))
    assert built[1:] == [('lower', 26, 80), ('upper', 26, 80)]
    assert result['large_spin_lmin'] == 26
    assert mp.mpf(result['resolved_settings']['spec']['large_spin_lmin']) == 26
    assert list(tmp_path.glob('progress.settings-*.json'))
    # Once the settings agree, neither bound is solved again.
    moment_nu1_over_nu0(_ctx(tmp_path, params))
    assert len(built) == 3


def test_moment_task_interrupted_mid_task_then_resumes(tmp_path, fake_pipeline, monkeypatch):
    stop = {"flag": False}
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "identity_modes": ["include"], "directions": ["lower", "upper"],
    }, should_stop=lambda: stop["flag"])

    real_run = identical_scalar_mod.run_identical_scalar_gap_problem

    def run_then_stop(problem, *a, **kw):
        result = real_run(problem, *a, **kw)
        if problem.direction == "lower":
            stop["flag"] = True  # simulate the wall approaching after 1 solve
        return result

    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", run_then_stop)
    with pytest.raises(TaskInterrupted):
        moment_nu1_over_nu0(ctx)

    assert fake_pipeline["run"] == [("include", "lower")]

    # Resume: should_stop no longer set, only the remaining solve runs.
    stop["flag"] = False
    fake_pipeline["run"].clear()
    ctx2 = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "identity_modes": ["include"], "directions": ["lower", "upper"],
    }, should_stop=lambda: stop["flag"])
    result = moment_nu1_over_nu0(ctx2)
    assert fake_pipeline["run"] == [("include", "upper")]
    assert result["include"]["lower"] == "1.5"
    assert result["include"]["upper"] == "1.5"


def test_moment_task_uncertified_optimal_reports_no_bound(tmp_path, fake_pipeline, monkeypatch):
    monkeypatch.setattr(
        certification_mod, "certify_dual_from_sdpb_output",
        lambda *a, **kw: (False, [{"passed": False}]),
    )
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "identity_modes": ["include"], "directions": ["lower"],
    })
    result = moment_nu1_over_nu0(ctx)
    assert result["include"]["lower"] is None
    assert result["include"]["lower_status"] == "optimal_uncertified"


def test_moment_task_does_not_publish_finite_grid_as_bound(tmp_path, fake_pipeline, monkeypatch):
    monkeypatch.setattr(
        certification_mod, "_is_normalized_moment_problem", lambda p: True
    )
    monkeypatch.setattr(
        certification_mod, "certify_dual_from_sdpb_output",
        lambda *a, **kw: (
            True,
            [{
                "passed": True,
                "basis": "physical_numeric_blocks",
                "certificate_scope": "finite_true_block_grid",
            }],
        ),
    )
    ctx = _ctx(tmp_path, {
        "c": "1.00001", "delta_ext": "0.5", "lam": 11,
        "identity_modes": ["exclude"], "directions": ["upper"],
    })
    result = moment_nu1_over_nu0(ctx)
    assert result["exclude"]["upper"] is None
    assert result["exclude"]["upper_solved_value"] == "1.5"
    assert result["exclude"]["upper_status"] == "optimal_uncertified"
    assert result["exclude"]["upper_certificate_scope"] == [
        "finite_true_block_grid"
    ]


def test_moment_task_max_iterations_gets_labeled(tmp_path, fake_pipeline, monkeypatch):
    log_path = tmp_path / "sdpb.log"
    log_path.write_text("...\n-----maxIterations exceeded-----\n")

    def fake_run(problem, output_dir, **kwargs):
        return _FakeRun(status=None, log_path=log_path)

    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", fake_run)
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "identity_modes": ["include"], "directions": ["lower"],
    })
    result = moment_nu1_over_nu0(ctx)
    assert result["include"]["lower"] is None
    assert result["include"]["lower_status"] == "max_iterations"


def test_moment_task_max_runtime_interrupts_not_journals(tmp_path, fake_pipeline, monkeypatch):
    log_path = tmp_path / "sdpb.log"
    log_path.write_text("...\n-----maxRuntime exceeded-----\n")

    def fake_run(problem, output_dir, **kwargs):
        return _FakeRun(status=None, log_path=log_path)

    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", fake_run)
    ctx = _ctx(tmp_path, {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "identity_modes": ["include"], "directions": ["lower"],
    })
    with pytest.raises(TaskInterrupted, match="maxRuntime"):
        moment_nu1_over_nu0(ctx)
    # Not journaled as a (valueless) completed solve -- a resumed task must
    # retry this exact solve from scratch, not treat it as done.
    assert "solved" not in ctx.journal.data or "include_lower" not in ctx.journal.data.get("solved", {})


def test_moment_task_prepare_failure_reported(tmp_path, monkeypatch):
    def boom(spec):
        raise RuntimeError("block build exploded")

    monkeypatch.setattr(identical_scalar_mod, "prepare_identical_scalar_gap_build", boom)
    ctx = _ctx(tmp_path, {"c": "8/7", "delta_ext": "2.0", "lam": 19})
    result = moment_nu1_over_nu0(ctx)
    assert result["reason"] == "prepare_error:RuntimeError"


def test_scratch_root_redirects_the_task_dir_but_keeps_legacy_paths(tmp_path, monkeypatch):
    """Bulky solve intermediates belong on scratch, not the project volume.

    One Lambda=19 task writes ~670 MB of SDP intermediates against a 100 KB
    result, so a 654-task sweep is ~440 GB of disposable data. VB_SCRATCH_ROOT
    redirects the task dir and leaves a symlink, so every existing
    ``output_dir / key / ...`` reader keeps resolving unchanged.
    """
    import json

    from virasoro_bootstrap.hpc.worker import load_spec

    run = tmp_path / "run"
    (run / "specs").mkdir(parents=True)
    (run / "tasks").mkdir()
    scratch = tmp_path / "scratch"
    (run / "specs" / "0000.json").write_text(
        json.dumps({"task": "t", "params": {}, "output_dir": "tasks/0000"})
    )

    monkeypatch.setenv("VB_SCRATCH_ROOT", str(scratch))
    _name, _params, out = load_spec(run / "specs" / "0000.json")
    assert out.is_symlink()
    assert str(scratch) in str(out.resolve())

    # The whole point: paths built the old way must still work.
    (out / "include_upper" / "sdpb_out").mkdir(parents=True)
    assert (run / "tasks" / "0000" / "include_upper" / "sdpb_out").is_dir()


def test_no_scratch_root_leaves_the_task_dir_alone(tmp_path, monkeypatch):
    import json

    from virasoro_bootstrap.hpc.worker import load_spec

    run = tmp_path / "run"
    (run / "specs").mkdir(parents=True)
    (run / "specs" / "0000.json").write_text(
        json.dumps({"task": "t", "params": {}, "output_dir": "tasks/0000"})
    )
    monkeypatch.delenv("VB_SCRATCH_ROOT", raising=False)
    _name, _params, out = load_spec(run / "specs" / "0000.json")
    assert not out.is_symlink()


def test_moment_worker_retries_only_failed_solves(tmp_path, fake_pipeline, monkeypatch):
    import json
    from virasoro_bootstrap.hpc import worker, submit, collect

    run_dir = tmp_path / "run"
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "run", "task": "moment_nu1_over_nu0", "n_tasks": 1,
    }))
    spec_path = run_dir / "specs" / "0000.json"
    spec_path.write_text(json.dumps({
        "task": "moment_nu1_over_nu0", "params": {"c": "8/7", "delta_ext": "2.0", "lam": 19},
        "output_dir": "tasks/0000",
    }))
    original = identical_scalar_mod.run_identical_scalar_gap_problem
    failed_once = []
    def transient_failure(problem, *args, **kwargs):
        if not failed_once:
            failed_once.append((problem.identity_mode, problem.direction))
            raise RuntimeError("temporary solver failure")
        return original(problem, *args, **kwargs)
    monkeypatch.setattr(identical_scalar_mod, "run_identical_scalar_gap_problem", transient_failure)
    assert worker.run_spec(spec_path) == 1
    assert collect.load_run(run_dir)[1][0].state == "failed"
    assert submit._pending_indices(run_dir, 1) == [0]
    assert len(fake_pipeline["run"]) == 3
    fake_pipeline["run"].clear()
    assert worker.run_spec(spec_path) == 0
    assert fake_pipeline["run"] == failed_once
    assert submit._pending_indices(run_dir, 1) == []


@pytest.mark.parametrize("spot_check", [False, True])
def test_moment_resume_preserves_certification_diagnostics(tmp_path, fake_pipeline, monkeypatch, spot_check):
    def certify(*args, **kwargs):
        return True, [{"passed": True, "certificate_scope": "finite_true_block_grid",
                       "basis": "physical_numeric_blocks", "ell": 0}]
    monkeypatch.setattr(certification_mod, "certify_dual_from_sdpb_output", certify)
    monkeypatch.setattr(certification_mod, "_is_normalized_moment_problem", lambda problem: True)
    params = {"c": "8/7", "delta_ext": "2.0", "lam": 19,
              "identity_modes": ["include"], "directions": ["lower"], "spot_check": spot_check}
    first = moment_nu1_over_nu0(_ctx(tmp_path, params))
    assert first["include"]["lower_certificate_scope"] == ["finite_true_block_grid"]
    assert first["include"]["lower"] is None
    fake_pipeline["run"].clear()
    resumed = moment_nu1_over_nu0(_ctx(tmp_path, params))
    assert resumed["include"] == first["include"]
    assert fake_pipeline["run"] == []


@pytest.mark.parametrize('name,value', [
    ('adaptive_cert_extra_twists',1.5), ('num_threads',True), ('cert_extra_spins',-1),
])
def test_moment_task_rejects_invalid_controls_before_prepare(tmp_path, fake_pipeline, name, value):
    params = {'c':'8/7', 'delta_ext':'2', name:value}
    with pytest.raises(ValueError, match=name):
        moment_nu1_over_nu0(_ctx(tmp_path, params))
    assert fake_pipeline['prepare'] == 0
    assert fake_pipeline['run'] == []


def test_moment_task_integer_strings_are_normalized(tmp_path, fake_pipeline):
    params = {'c':'8/7', 'delta_ext':'2', 'lam':'19', 'dps':'40',
              'num_threads':'1', 'precision':'128', 'moment_cert_q_order':'80',
              'cert_n_test_points':'12', 'identity_modes':['include'], 'directions':['lower']}
    result = moment_nu1_over_nu0(_ctx(tmp_path, params))
    assert result['lam'] == 19 and result['dps'] == 40
    kwargs = fake_pipeline['certify_kwargs'][0]
    assert kwargs['moment_cert_q_order'] == 80
    assert kwargs['n_test_points'] == 12 and kwargs['num_threads'] == 1


@pytest.mark.parametrize('name,value', [('directions','lower'), ('identity_modes',['included'])])
def test_invalid_solve_selections_are_rejected(tmp_path, fake_pipeline, name, value):
    params = {'c':'8/7', 'delta_ext':'2'}
    params[name] = value
    with pytest.raises(ValueError, match=name):
        moment_nu1_over_nu0(_ctx(tmp_path, params))


@pytest.mark.parametrize('schema', [1, 2, 3])
def test_worker_checks_completed_moment_settings(tmp_path, fake_pipeline, monkeypatch, schema):
    """A completed result must pass the same resolved-settings check as a journal."""
    import json
    from dataclasses import replace
    from virasoro_bootstrap.hpc import tasks, worker, submit
    from virasoro_bootstrap.hpc.provenance import settings_digest
    from tests.integration.test_hpc_parameter_reuse import _old_settings

    params = {'c': '8', 'delta_ext': '1', 'lam': 19,
              'identity_modes': ['include'], 'directions': ['lower']}
    path = tmp_path / 'specs/0000.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'task': 'moment_nu1_over_nu0', 'params': params,
                                'output_dir': 'tasks/0000'}))
    builder = tasks._moment_nu1_over_nu0_spec
    monkeypatch.setattr(tasks, '_moment_nu1_over_nu0_spec',
                        lambda *a, **kw: replace(builder(*a, **kw), large_spin_lmin=26))
    assert worker.run_spec(path) == 0
    assert worker.run_spec(path) == 0
    assert len(fake_pipeline['run']) == 1
    if schema != 3:
        old = _old_settings(schema, 'moment_nu1_over_nu0', params)
        for filename in ('progress.json', 'result.json'):
            saved_path = tmp_path / 'tasks/0000' / filename
            doc = json.loads(saved_path.read_text())
            doc.update(execution_settings=old, execution_hash=settings_digest(old))
            saved_path.write_text(json.dumps(doc))
    monkeypatch.setattr(tasks, '_moment_nu1_over_nu0_spec', builder)
    assert submit._pending_indices(tmp_path, 1) == [0]
    assert worker.run_spec(path) == 0
    assert len(fake_pipeline['run']) == 2
    assert submit._pending_indices(tmp_path, 1) == []
