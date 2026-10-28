"""SDPB termination statuses in the task layer, without invoking SDPB.

``--maxRuntime`` is the clean pre-wall exit the worker requests, so it must
interrupt (and requeue) the task. ``maxIterations`` and SDPB's step-size
terminations say nothing about feasibility, so they must never be scored as an
ALLOWED or EXCLUDED verdict.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from virasoro_bootstrap import identical_scalar
from virasoro_bootstrap.hpc.journal import Journal
from virasoro_bootstrap.hpc.tasks import (
    _INCONCLUSIVE_SDPB_STATUSES, TaskContext, TaskInterrupted, _checked_sdpb_status,
    scalar_gap_bisection, scalar_only_gap_probe,
)

INCONCLUSIVE = ("max_iterations", "primal_step_too_small", "dual_step_too_small")


def _ctx(tmp_path: Path, params: dict) -> TaskContext:
    return TaskContext(params=params, output_dir=tmp_path, journal=Journal(tmp_path / "progress.json"))


def test_max_runtime_status_or_log_interrupts(tmp_path):
    with pytest.raises(TaskInterrupted, match="maxRuntime"):
        _checked_sdpb_status(SimpleNamespace(sdpb_status="max_runtime"), "x")
    log = tmp_path / "sdpb.log"
    log.write_text("...\n-----maxRuntime exceeded-----\n")
    with pytest.raises(TaskInterrupted, match="maxRuntime"):
        _checked_sdpb_status(SimpleNamespace(sdpb_status=None, sdpb_log_path=log), "x")
    log.write_text("...\n-----maxIterations exceeded-----\n")
    assert _checked_sdpb_status(SimpleNamespace(sdpb_status=None, sdpb_log_path=log), "x") == "max_iterations"


def test_non_verdict_statuses_are_inconclusive():
    assert {"max_runtime", *INCONCLUSIVE} <= _INCONCLUSIVE_SDPB_STATUSES
    assert not {"primal_feasible", "dual_feasible"} & _INCONCLUSIVE_SDPB_STATUSES


@pytest.fixture
def probe_solver(monkeypatch):
    state = {"status": None}
    monkeypatch.setattr(identical_scalar, "build_identical_scalar_gap_problem", lambda spec: object())
    monkeypatch.setattr(
        identical_scalar, "run_identical_scalar_gap_problem",
        lambda problem, output_dir, **kw: SimpleNamespace(
            sdpb_status=state["status"], sdpb_summary=None, sdpb_log_path=None),
    )
    return state


@pytest.mark.parametrize("status", INCONCLUSIVE)
def test_scalar_only_probe_never_scores_inconclusive_status(tmp_path, probe_solver, status):
    probe_solver["status"] = status
    result = scalar_only_gap_probe(_ctx(tmp_path, {"c": "2", "delta": "0.5", "lam": 7}))
    assert result["reason"] == "inconclusive" and "allowed" not in result
    assert result["sdpb_status"] == status


def test_scalar_only_probe_max_runtime_interrupts(tmp_path, probe_solver):
    probe_solver["status"] = "max_runtime"
    with pytest.raises(TaskInterrupted, match="maxRuntime"):
        scalar_only_gap_probe(_ctx(tmp_path, {"c": "2", "delta": "0.5", "lam": 7}))


class _Problem:
    pmp_build = SimpleNamespace(doc={
        "objective": [0, 0],
        "PositiveMatrixWithPrefactorArray": [{"polynomials": [[[[0]]]], "samplePoints": [0]}],
    })


@pytest.mark.parametrize("status", [*INCONCLUSIVE, "max_runtime"])
def test_gap_bisection_never_excludes_on_inconclusive_status(tmp_path, monkeypatch, status):
    monkeypatch.setattr(identical_scalar, "build_identical_scalar_gap_problem",
                        lambda spec, prepared=None: _Problem())
    monkeypatch.setattr(
        identical_scalar, "run_identical_scalar_gap_problem",
        lambda problem, output_dir, **kw: SimpleNamespace(
            sdpb_status=status, sdpb_summary=None, checkpoint_dir=None,
            sdpb_log_path=None, sdpb_out_dir=None),
    )
    ctx = _ctx(tmp_path, {"c": "2", "lam": 7, "spins": [0, 2], "gap_lo": "0.5", "gap_hi": "2.0",
                          "tol": "0.2", "hotstart": False, "large_spin_double_twists": 3})
    if status == "max_runtime":
        with pytest.raises(TaskInterrupted, match="maxRuntime"):
            scalar_gap_bisection(ctx)
    else:
        result = scalar_gap_bisection(ctx)
        assert result["reason"] == "solve_error" and result["delta_gap"] is None
        assert status in result["error"]
    probes = ctx.journal.data.get("probes", {})
    assert all(entry.get("allowed") is not False for entry in probes.values()), probes
