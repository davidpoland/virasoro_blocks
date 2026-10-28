"""moment_recertify task: rechecks certification against an existing sdpb_out
at a different moment_cert_q_order, without re-solving SDPB.

Every expensive call (manifest rebuild, certification) is
monkeypatched to a cheap fake -- these tests check the task's own logic
(sdpb_out discovery, journaling, param wiring), not the certification
numerics, which certification.py's own tests cover.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import virasoro_bootstrap.certification as certification_mod
import virasoro_bootstrap.identical_scalar as identical_scalar_mod
import virasoro_bootstrap.moments as moments_mod
import virasoro_bootstrap.manifest as manifest_mod
import virasoro_bootstrap.hpc.tasks as tasks_mod
from virasoro_bootstrap.hpc.journal import Journal
from virasoro_bootstrap.hpc.tasks import TaskContext, TaskInterrupted, moment_recertify


from virasoro_blocks._conventions import scalar_G_spin0_convention


class _FakeProblem:
    def __init__(self, direction, identity_mode, spec):
        self.direction = direction
        self.identity_mode = identity_mode
        self.spec = spec
        self.scalar_G_spin0 = scalar_G_spin0_convention()
        self.problem_kind = f"normalized_moment_{direction}"
        self.pmp_metadata = {"moment": {
            "identity_mode": identity_mode,
            "numerator": moments_mod.MomentWeight.monomial(Delta=1).to_json(),
            "denominator": moments_mod.MomentWeight.one().to_json(),
        }}


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = {"load": [], "certify": [], "specs": []}
    original_spec = tasks_mod._moment_nu1_over_nu0_spec
    def capture_spec(*args, **kwargs):
        spec = original_spec(*args, **kwargs)
        calls["specs"].append(spec)
        return spec
    def fake_load(path):
        calls["load"].append(path)
        identity_mode, direction = path.parent.name.split("_")
        return _FakeProblem(direction, identity_mode, calls["specs"][-1])
    def fake_certify(problem, sdpb_out_dir, **kwargs):
        calls["certify"].append(
            (problem.identity_mode, problem.direction, str(sdpb_out_dir), kwargs.get("moment_cert_q_order"))
        )
        return True, [{"passed": True}, {"passed": True}]
    monkeypatch.setattr(tasks_mod, "_moment_nu1_over_nu0_spec", capture_spec)
    monkeypatch.setattr(manifest_mod, "build_identical_scalar_problem_from_manifest", fake_load)
    monkeypatch.setattr(certification_mod, "certify_dual_from_sdpb_output", fake_certify)
    def unexpected_prepare(*args, **kwargs):
        pytest.fail("recertification must load saved blocks, not prepare new ones")
    monkeypatch.setattr(identical_scalar_mod, "prepare_identical_scalar_gap_build", unexpected_prepare)
    return calls


def _ctx(tmp_path: Path, params: dict, *, should_stop=lambda: False) -> TaskContext:
    return TaskContext(
        params=params, output_dir=tmp_path,
        journal=Journal(tmp_path / "progress.json"), should_stop=should_stop,
    )


def _make_source_dirs(tmp_path: Path, keys=("include_lower", "include_upper", "exclude_lower", "exclude_upper")) -> Path:
    source = tmp_path / "source"
    for key in keys:
        (source / key / "sdpb_out").mkdir(parents=True)
    return source


def _base_params(source: Path, **overrides) -> dict:
    params = {
        "c": "8/7", "delta_ext": "2.0", "lam": 19,
        "source_output_dir": str(source), "moment_cert_q_order": 160,
    }
    params.update(overrides)
    return params


def test_recertify_all_four_against_existing_sdpb_out(tmp_path, fake_pipeline):
    source = _make_source_dirs(tmp_path)
    ctx = _ctx(tmp_path, _base_params(source))
    result = moment_recertify(ctx)

    assert len(fake_pipeline["load"]) == 4
    assert len(fake_pipeline["certify"]) == 4
    assert all(q == 160 for (_, _, _, q) in fake_pipeline["certify"])
    for mode in ("include", "exclude"):
        for direction in ("lower", "upper"):
            entry = result[mode][direction]
            assert entry["certified"] is True
            assert entry["n_pass"] == 2 and entry["n_total"] == 2
            assert "details" in entry  # spot_check defaults True for this task


def test_recertify_keeps_finite_grid_pass_separate_from_global_certification(
    tmp_path, fake_pipeline, monkeypatch,
):
    source = _make_source_dirs(tmp_path, keys=("include_lower",))

    def finite_grid_certify(problem, sdpb_out_dir, **kwargs):
        return True, [{
            "passed": True,
            "basis": "physical_numeric_blocks",
            "certificate_scope": "finite_true_block_grid",
        }]

    monkeypatch.setattr(
        certification_mod, "certify_dual_from_sdpb_output", finite_grid_certify,
    )
    monkeypatch.setattr(
        certification_mod, "_is_normalized_moment_problem", lambda problem: True,
    )
    result = moment_recertify(_ctx(
        tmp_path,
        _base_params(source, identity_modes=["include"], directions=["lower"]),
    ))
    entry = result["include"]["lower"]
    assert entry["finite_grid_passed"] is True
    assert entry["certified"] is False
    assert entry["certificate_scope"] == ["finite_true_block_grid"]
    loads = len(fake_pipeline['load'])
    repeated = moment_recertify(_ctx(
        tmp_path,
        _base_params(source, identity_modes=["include"], directions=["lower"]),
    ))
    assert len(fake_pipeline['load']) == loads
    assert repeated['include']['lower'] == entry


def test_recertify_uses_original_solve_params_not_defaults(tmp_path, fake_pipeline, monkeypatch):
    source = _make_source_dirs(tmp_path)
    ctx = _ctx(tmp_path, _base_params(
        source, q_order=60, npoles=20,  # must match the ORIGINAL solve, not certification target
        identity_modes=["include"], directions=["lower"],
    ))
    result = moment_recertify(ctx)
    assert result["q_order"] == 60 and result["npoles"] == 20
    assert result["moment_cert_q_order"] == 160  # the diagnostic knob, independent of the above
    assert fake_pipeline["specs"][0].q_order == 60
    assert fake_pipeline["specs"][0].npoles == 20


def test_recertify_missing_sdpb_out_records_error_not_crash(tmp_path, fake_pipeline):
    source = tmp_path / "source"  # no subdirs created at all
    ctx = _ctx(tmp_path, _base_params(source, identity_modes=["include"], directions=["lower"]))
    result = moment_recertify(ctx)
    assert "error" in result["include"]["lower"]
    assert fake_pipeline["certify"] == []  # never attempted a certify call


def test_recertify_journal_resume_skips_completed(tmp_path, fake_pipeline):
    source = _make_source_dirs(tmp_path)
    params = _base_params(source)
    moment_recertify(_ctx(tmp_path, params))
    assert len(fake_pipeline["certify"]) == 4

    fake_pipeline["certify"].clear()
    fake_pipeline["load"].clear()
    result = moment_recertify(_ctx(tmp_path, params))
    assert fake_pipeline["certify"] == []
    assert result["include"]["lower"]["certified"] is True
    assert fake_pipeline["load"] == []
    ctx = _ctx(tmp_path, params)
    valid = dict(ctx.journal.data['solved'])
    for invalid in (None, {}, {'certified': True},
                    {**valid['include_lower'], 'certified': 'false'},
                    {**valid['include_lower'], 'finite_grid_passed': False},
                    {**valid['include_lower'], 'n_pass': True},
                    {**valid['include_lower'], 'n_total': 0},
                    {**valid['include_lower'], 'certificate_scope': ['finite_true_block_grid']},
                    {**valid['include_lower'], 'certificate_scope': 'global'}):
        ctx.journal.update(solved={**valid, 'include_lower': invalid})
        fake_pipeline['certify'].clear()
        retried = moment_recertify(_ctx(tmp_path, params))
        assert [(mode, direction) for mode, direction, _, _ in fake_pipeline['certify']] == [('include', 'lower')]
        assert retried['include']['lower']['certified'] is True


def test_recertify_interrupted_then_resumes(tmp_path, fake_pipeline):
    stop = {"flag": False}
    source = _make_source_dirs(tmp_path, keys=("include_lower", "include_upper"))
    params = _base_params(source, identity_modes=["include"], directions=["lower", "upper"])

    real_certify = certification_mod.certify_dual_from_sdpb_output

    def certify_then_stop(problem, sdpb_out_dir, **kwargs):
        result = real_certify(problem, sdpb_out_dir, **kwargs)
        if problem.direction == "lower":
            stop["flag"] = True
        return result

    import virasoro_bootstrap.certification as cm
    cm.certify_dual_from_sdpb_output = certify_then_stop
    try:
        with pytest.raises(TaskInterrupted):
            moment_recertify(_ctx(tmp_path, params, should_stop=lambda: stop["flag"]))
    finally:
        cm.certify_dual_from_sdpb_output = real_certify

    assert len(fake_pipeline["certify"]) == 1

    stop["flag"] = False
    fake_pipeline["certify"].clear()
    result = moment_recertify(_ctx(tmp_path, params, should_stop=lambda: stop["flag"]))
    assert len(fake_pipeline["certify"]) == 1  # only the remaining solve ran
    assert result["include"]["lower"]["certified"] is True
    assert result["include"]["upper"]["certified"] is True


def test_recertify_spot_check_false_omits_details(tmp_path, fake_pipeline):
    source = _make_source_dirs(tmp_path, keys=("include_lower",))
    ctx = _ctx(tmp_path, _base_params(
        source, identity_modes=["include"], directions=["lower"], spot_check=False,
    ))
    result = moment_recertify(ctx)
    assert "details" not in result["include"]["lower"]


def test_recertify_manifest_failure_can_be_retried(tmp_path, fake_pipeline, monkeypatch):
    loader = manifest_mod.build_identical_scalar_problem_from_manifest
    def boom(path):
        raise ValueError("block hash mismatch")
    monkeypatch.setattr(manifest_mod, "build_identical_scalar_problem_from_manifest", boom)
    source = _make_source_dirs(tmp_path)
    params = _base_params(source)
    result = moment_recertify(_ctx(tmp_path, params))
    assert "hash mismatch" in result["include"]["lower"]["error"]
    assert fake_pipeline["certify"] == []
    monkeypatch.setattr(manifest_mod, "build_identical_scalar_problem_from_manifest", loader)
    result = moment_recertify(_ctx(tmp_path, params))
    assert len(fake_pipeline["certify"]) == 4
    assert result["include"]["lower"]["certified"] is True


def test_recertify_requires_moment_cert_q_order(tmp_path, fake_pipeline):
    source = _make_source_dirs(tmp_path)
    params = _base_params(source)
    del params["moment_cert_q_order"]
    with pytest.raises(KeyError):
        moment_recertify(_ctx(tmp_path, params))


def test_recertify_missing_output_can_be_retried(tmp_path, fake_pipeline):
    source = tmp_path / "source"
    params = _base_params(source, identity_modes=["include"], directions=["lower"])
    first = moment_recertify(_ctx(tmp_path, params))
    assert "error" in first["include"]["lower"]
    _make_source_dirs(tmp_path, keys=("include_lower",))
    second = moment_recertify(_ctx(tmp_path, params))
    assert second["include"]["lower"]["certified"] is True
    assert len(fake_pipeline["certify"]) == 1


def test_recertify_worker_retries_after_source_is_repaired(tmp_path, fake_pipeline):
    import json
    from virasoro_bootstrap.hpc import worker

    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    (run_dir / "specs").mkdir(parents=True)
    spec = run_dir / "specs" / "0000.json"
    spec.write_text(json.dumps({
        "task": "moment_recertify", "params": _base_params(source),
        "output_dir": "tasks/0000",
    }))
    assert worker.run_spec(spec) == 1
    assert fake_pipeline["certify"] == []
    _make_source_dirs(tmp_path)
    assert worker.run_spec(spec) == 0
    assert len(fake_pipeline["certify"]) == 4
    document = json.loads((run_dir / "tasks" / "0000" / "result.json").read_text())
    assert "failed" not in document
    assert document["result"]["include"]["lower"]["certified"] is True


@pytest.mark.parametrize('name,value', [('moment_cert_q_order',80.5), ('dps',0)])
def test_recertify_rejects_invalid_controls_before_loading(tmp_path, fake_pipeline, name, value):
    params = _base_params(_make_source_dirs(tmp_path), **{name:value})
    with pytest.raises(ValueError, match=name):
        moment_recertify(_ctx(tmp_path, params))
    assert fake_pipeline['load'] == []
    assert fake_pipeline['certify'] == []


def test_recertify_honors_physical_grid_overrides(tmp_path, fake_pipeline, monkeypatch):
    captured = []
    def certify(problem, out_dir, **kwargs):
        captured.append(kwargs)
        return True, [{'passed':True}]
    monkeypatch.setattr(certification_mod, 'certify_dual_from_sdpb_output', certify)
    params = _base_params(_make_source_dirs(tmp_path),
        identity_modes=['include'], directions=['lower'],
        cert_n_test_points='240', cert_delta_range='40', cert_log_near_points='12',
        physical_moment_blocks=False, num_threads='1', moment_cert_q_order='160')
    result = moment_recertify(_ctx(tmp_path, params))
    assert result['moment_cert_q_order'] == 160
    assert captured[0]['n_test_points'] == 240
    assert captured[0]['delta_range'] == 40
    assert captured[0]['log_near_points'] == 12
    assert captured[0]['physical_moment_blocks'] is False
    assert captured[0]['num_threads'] == 1


@pytest.mark.parametrize('name,value', [('directions',['lowre'])])
def test_invalid_solve_selections_are_rejected(tmp_path, fake_pipeline, name, value):
    params = _base_params(_make_source_dirs(tmp_path))
    params[name] = value
    with pytest.raises(ValueError, match=name):
        moment_recertify(_ctx(tmp_path, params))
