"""Reuse follows effective parameters across defaults changes and schema upgrades."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from virasoro_bootstrap import defaults
from virasoro_bootstrap.hpc import submit, tasks, worker
from virasoro_bootstrap.hpc.provenance import execution_matches, execution_settings, settings_digest


def _spec(root, task_name="scalar_gap_bisection", **params):
    path = root / "specs/0000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task": task_name, "params": {"c": "8", **params},
                               "output_dir": "tasks/0000"}))
    return path


def _old_settings(schema, task_name, params):
    config = defaults.load_default_params()
    flags = tasks.TASK_SDPB_DEFAULTS.get(task_name)
    if schema == 1:
        constants = {name: value for name, value in vars(tasks).items()
                     if name.lstrip("_").startswith("DEFAULT_")
                     and isinstance(value, (str, int, float, bool, dict, tuple, list, type(None)))}
        saved = {"schema": 1, "code": "ignored-code-fingerprint", "defaults": config,
                 "presets": tasks.LAMBDA_PRESETS, "task_defaults": constants}
    else:
        saved = {"schema": 2, "profiles": {
            key: defaults._profile_from_snapshot(config, int(key.removeprefix("lambda")))
            for key in ("lambda7", "lambda11", "lambda15", "lambda19")},
            "sdpb_defaults": None if flags is None else getattr(tasks, flags)}
    return json.loads(json.dumps(saved))


def _change_profile(monkeypatch, lam, **changes):
    config = defaults.load_default_params()
    config[f"lambda{lam}"].update(changes)
    monkeypatch.setattr(defaults, "load_default_params", lambda: config)
    monkeypatch.setattr(tasks, "LAMBDA_PRESETS", tasks._presets_from_snapshot(config))


@pytest.fixture
def probe(monkeypatch):
    calls = []

    def run(ctx):
        if "probe" in ctx.journal.data:
            return ctx.journal.data["probe"]
        profile = ctx.profile(int(ctx.params.get("lam", 7)))
        result = {"q_order": ctx.params.get("q_order", profile["q_order"])}
        calls.append(result)
        ctx.journal.update(probe=result)
        return result

    monkeypatch.setitem(tasks.TASKS, "scalar_gap_bisection", run)
    return calls


@pytest.mark.parametrize("schema", [1, 2, 3])
@pytest.mark.parametrize("overridden", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_defaults_changes_compare_effective_parameters(tmp_path, monkeypatch, probe, schema, overridden, partial):
    params = {"q_order": 24} if overridden else {}
    spec = _spec(tmp_path, **params)  # omitted lam must resolve to this task's default
    assert worker.run_spec(spec) == 0
    out = tmp_path / "tasks/0000"
    if schema != 3:
        old = _old_settings(schema, "scalar_gap_bisection", params)
        for filename in ("result.json", "progress.json"):
            path = out / filename
            saved = json.loads(path.read_text())
            saved.update(execution_settings=old, execution_hash=settings_digest(old))
            path.write_text(json.dumps(saved))
    if partial:
        (out / "result.json").unlink()
    _change_profile(monkeypatch, 7, q_order=42)
    assert submit._pending_indices(tmp_path, 1) == ([0] if partial or not overridden else [])
    assert worker.run_spec(spec) == 0
    assert len(probe) == (1 if overridden else 2)
    result = json.loads((out / "result.json").read_text())["result"]
    assert result["q_order"] == (24 if overridden else 42)
    assert bool(list(out.glob("progress.execution-*.json"))) is (not overridden)


@pytest.mark.parametrize("override", [{"--precision": 2048}, {"--precision=2048": True}, {"--precision": False}])
@pytest.mark.parametrize("schema", [1, 2, 3])
def test_overridden_solver_defaults_are_ignored(monkeypatch, override, schema):
    p = {"lam": 7, "sdpb_options": override}
    saved = execution_settings("scalar_gap_bisection", p) if schema == 3 else _old_settings(schema, "scalar_gap_bisection", p)
    monkeypatch.setattr(tasks, "DEFAULT_FEASIBILITY_SDPB_OPTIONS",
                        {**tasks.DEFAULT_FEASIBILITY_SDPB_OPTIONS, "--precision": 1024})
    current = execution_settings("scalar_gap_bisection", p)
    assert execution_matches(saved, current)
    assert not execution_matches(saved, execution_settings("scalar_gap_bisection", {"lam": 7}))


def test_records_imported_preset_not_later_file_contents(tmp_path, monkeypatch, probe):
    config = defaults.load_default_params()
    config["lambda7"]["q_order"] = 42
    monkeypatch.setattr(defaults, "load_default_params", lambda: config)
    spec = _spec(tmp_path)
    assert worker.run_spec(spec) == 0
    saved = json.loads((tmp_path / "tasks/0000/result.json").read_text())
    assert saved["result"]["q_order"] == 40
    assert saved["execution_settings"]["effective"]["q_order"] == 40
    # A fresh process imports the updated preset; the old result must be pending.
    monkeypatch.setattr(tasks, "LAMBDA_PRESETS", tasks._presets_from_snapshot(config))
    assert submit._pending_indices(tmp_path, 1) == [0]
    assert worker.run_spec(spec) == 0
    assert probe == [{"q_order": 40}, {"q_order": 42}]


def test_worker_executes_the_snapshot_it_checked(tmp_path, monkeypatch):
    p = {"delta_ext": "1", "lam": 19}
    spec = _spec(tmp_path, "moment_nu1_over_nu0", **p)
    config = defaults.load_default_params()
    original_open = worker._open_journal

    def change_after_check(*args, **kwargs):
        config["lambda19"]["q_order"] = 82
        monkeypatch.setattr(defaults, "load_default_params", lambda: config)
        monkeypatch.setattr(tasks, "_DEFAULT_MOMENT_SDPB_OPTIONS", {"--maxIterations": 5})
        return original_open(*args, **kwargs)

    def run(ctx):
        return {"q_order": ctx.profile(19, moment=True)["q_order"],
                "flags": tasks._merged_sdpb_options(tasks._DEFAULT_MOMENT_SDPB_OPTIONS,
                                                     None, execution=ctx.execution)}

    monkeypatch.setattr(worker, "_open_journal", change_after_check)
    monkeypatch.setitem(tasks.TASKS, "moment_nu1_over_nu0", run)
    assert worker.run_spec(spec) == 0
    saved = json.loads((tmp_path / "tasks/0000/result.json").read_text())
    assert saved["result"]["q_order"] == 80
    assert saved["result"]["flags"]["--maxIterations"] == 1000
    assert submit._pending_indices(tmp_path, 1) == [0]


def test_omitted_lam_and_unused_profile_fields_do_not_invalidate(monkeypatch):
    saved = execution_settings("scalar_gap_bisection", {})
    _change_profile(monkeypatch, 19, q_order=82)
    _change_profile(monkeypatch, 7, cert_extra_spins=100, sdpb_precision=2048,
                    spin_list=[0, 2], global_pole_selection="unused", _large_spin_npoles_note="Edited prose.")
    assert execution_matches(saved, execution_settings("scalar_gap_bisection", {}))


@pytest.mark.parametrize("task_name,p", [
    ("scalar_gap_bisection", {"tower": False}),
    ("scalar_ope_max_scan", {"tower": False}),
    ("isolated_scalar_four_point_ope_bounds", {"four_point_large_spin": False}),
    ("isolated_scalar_four_point_fixed_ope", {"four_point_large_spin": False}),
    ("modular_scalar_gap_bisection", {}),
    ("isolated_relevant_scalar_delta_phi_intervals", {}),
    ("isolated_relevant_scalar_degeneracy_intervals", {}),
])
def test_unused_tower_defaults_do_not_invalidate(monkeypatch, task_name, p):
    saved = execution_settings(task_name, p)
    _change_profile(monkeypatch, saved["effective"]["lam"], large_spin_npoles=16, large_spin_double_twists=30)
    monkeypatch.setattr(tasks, "DEFAULT_LARGE_SPIN_POLE_SELECTION", "unused")
    assert execution_matches(saved, execution_settings(task_name, p))


@pytest.mark.parametrize("task_name,p", [
    ("moment_nu1_over_nu0", {}),
    ("moment_recertify", {"moment_cert_q_order": 60}),
])
def test_moment_overrides_and_resolved_floor(monkeypatch, task_name, p):
    p = {"c": "8", "delta_ext": "1", "lam": 19, "q_order": 24,
         "precision": 2048, "cert_extra_spins": 12, **p}
    saved = execution_settings(task_name, p)
    _change_profile(monkeypatch, 19, q_order=82, sdpb_precision=4096, cert_extra_spins=100)
    assert execution_matches(saved, execution_settings(task_name, p))
    builder = tasks._moment_nu1_over_nu0_spec
    monkeypatch.setattr(tasks, "_moment_nu1_over_nu0_spec", lambda *a, **kw: replace(builder(*a, **kw), large_spin_lmin=26))
    assert not execution_matches(saved, execution_settings(task_name, p))


def test_unknown_or_malformed_settings_are_not_adopted():
    current = execution_settings("scalar_gap_bisection", {})
    assert execution_matches(None, current)
    for saved in ({"schema": 99}, {"schema": 1}, {"schema": 2}, "broken"):
        assert not execution_matches(saved, current)


def test_all_builtin_tasks_have_a_default_profile():
    from virasoro_bootstrap.hpc.provenance import _DEFAULT_LAM
    assert set(_DEFAULT_LAM) == set(tasks.TASKS)


@pytest.mark.parametrize('task_name', list(tasks.TASKS))
@pytest.mark.parametrize('schema', [1, 2])
def test_matching_legacy_settings_are_reused_for_every_task(task_name, schema):
    p = {'c': '8', 'delta_ext': '1'}
    if task_name == 'moment_recertify':
        p['moment_cert_q_order'] = 60
    assert execution_matches(_old_settings(schema, task_name, p), execution_settings(task_name, p))


def test_default_orders_and_selector_help_match_task_signatures():
    from virasoro_bootstrap.hpc.params import task_signature
    from virasoro_bootstrap.hpc.provenance import _DEFAULT_LAM
    for name, lam in _DEFAULT_LAM.items():
        signature = task_signature(name)
        assert signature.params['lam'].default == str(lam)
        if name in {'scalar_gap_bisection', 'isolated_scalar_four_point_ope_bounds',
                    'isolated_scalar_four_point_fixed_ope'}:
            assert signature.params['pole_selection'].default == '"contribution_optimal"'
