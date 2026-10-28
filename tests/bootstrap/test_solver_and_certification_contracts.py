"""Contracts of the solver status, the dual-vector loader, the certification details, and spec validation."""

from __future__ import annotations

from fractions import Fraction
import json
from types import SimpleNamespace

import mpmath as mp
import pytest

from virasoro_bootstrap import (
    IdenticalScalarGapSpec,
    certify_dual_functional,
    load_y_vector,
    reconstruct_full_y,
    worst_certification_detail,
)
from virasoro_bootstrap import certification as C
from virasoro_bootstrap import identical_scalar as S

pytestmark = pytest.mark.bootstrap


def _spec(**changes):
    base = dict(c=8, h_ext=Fraction(1, 16), delta_gap=1, lam=3, spin_list=(0, 2), q_order=10, dps=40,
                npoles=4, num_threads=1)
    base.update(changes)
    return IdenticalScalarGapSpec(**base)


def _constant_channel(values, *, ell=0, delta_min=0, twist=None):
    """A channel whose F-vector is ``values`` at every point."""
    block = SimpleNamespace(evaluate=lambda coordinate, point: [[mp.mpf(v)] for v in values])
    return SimpleNamespace(ell=ell, delta_min=delta_min, twist=twist, prefactored=block)


# ---------------------------------------------------------------------------
# load_y_vector(full=True) feeds certify_dual_functional
# ---------------------------------------------------------------------------

def _solver_output(tmp_path, normalization, y_printed):
    sdp, out = tmp_path / "sdp", tmp_path / "sdpb_out"
    sdp.mkdir()
    out.mkdir()
    (sdp / "control.json").write_text("{}")
    (sdp / "normalization.json").write_text(json.dumps({"normalization": normalization}))
    (out / "y.txt").write_text(f"{len(y_printed)} 1\n" + "".join(f"{v}\n" for v in y_printed))
    return out


def test_full_dual_vector_route_certifies(tmp_path):
    # n = (2, 1, 0): SDPB eliminates component 0 and prints (y1, y2); n . y = 1 restores y0 = (1 - y1) / 2
    out = _solver_output(tmp_path, ["2", "1", "0"], ["3", "-0.5"])
    pairs = [(0, 0), (1, 0), (2, 0)]
    with mp.workdps(30):
        printed = load_y_vector(out)
        assert len(printed) == 2
        with pytest.raises(ValueError, match=r"load_y_vector\(sdpb_out_dir, full=True\)"):
            certify_dual_functional(printed, [_constant_channel([1, 1, 1])], pairs)
        full = load_y_vector(out, full=True)
        assert full == [mp.mpf(-1), mp.mpf(3), mp.mpf("-0.5")]
        assert reconstruct_full_y(printed, out) == full
        # y . F = -1 + 3 - 0.5 > 0 on this channel, and the y2 < 0 channel fails
        ok, details = certify_dual_functional(full, [_constant_channel([1, 1, 1]), _constant_channel([0, 0, 1], ell=2)],
                                              pairs, n_test_points=2, delta_range=1)
    assert not ok and [d["passed"] for d in details] == [True, False]


def test_full_dual_vector_refuses_what_it_cannot_restore(tmp_path):
    out = _solver_output(tmp_path, ["2", "1", "0"], ["3"])
    with pytest.raises(ValueError, match="expected 2"):
        load_y_vector(out, full=True)
    (tmp_path / "sdp" / "normalization.json").unlink()
    with pytest.raises(ValueError, match="normalization.json"):
        load_y_vector(out, full=True)


# ---------------------------------------------------------------------------
# SDPB status and summary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reason, status", [
    ("found primal-dual optimal solution", "optimal"),
    ("maxIterations exceeded", "max_iterations"),
    ("maxRuntime exceeded", "max_runtime"),
    ("found dual feasible solution", "dual_feasible"),
])
def test_sdpb_outcome_names_every_terminate_reason(tmp_path, reason, status):
    log = tmp_path / "sdpb.log"
    log.write_text(f"primalErrorThreshold = 1e-30\niter ...\n-----{reason}-----\nprimalObjective = 1\n")
    assert S._sdpb_outcome(0, log) == (status, f"-----{reason}-----")


def test_sdpb_failure_summary_carries_the_cause(tmp_path):
    log = tmp_path / "sdpb.log"
    # a runner failure need not say "error"; SDPB's parameter echo is not one
    log.write_text("dualErrorThreshold = 1e-30\ndocker: Cannot connect to the Docker daemon at "
                   "unix:///var/run/docker.sock. Is the docker daemon running?\n")
    status, summary = S._sdpb_outcome(1, log)
    assert status == "solver_error"
    assert summary.startswith("SDPB exited with code 1: ") and "Cannot connect to the Docker daemon" in summary
    log.write_text("Error: block_data_3.json: unexpected end of file\nmpirun noticed that process rank 0 exited\n")
    assert S._sdpb_outcome(2, log)[1] == "SDPB exited with code 2: Error: block_data_3.json: unexpected end of file"
    log.write_text("iter 1\n")
    assert S._sdpb_outcome(0, log)[0] is None
    assert S._sdpb_outcome(0, None) == (None, None)


# ---------------------------------------------------------------------------
# Certification details
# ---------------------------------------------------------------------------

def test_tower_details_report_spin_and_dimension():
    twist = Fraction(5, 2)
    # value L - 13 along the tower L >= 12: negative at L = 12, the minimum
    block = SimpleNamespace(evaluate=lambda coordinate, L: [[mp.mpf(L) - 13]])
    tower = SimpleNamespace(ell=12, delta_min=12, twist=twist, prefactored=block)
    with mp.workdps(30):
        ok, (detail,) = certify_dual_functional([1], [tower], [(0, 0)], n_test_points=3, delta_range=2)
    assert not ok and detail["variable"] == "L"
    assert detail["min_spin"] == 12 and detail["min_delta"] == 14.5
    assert detail["first_negative_spin"] == 12 and detail["first_negative_delta"] == 14.5
    assert detail["min_value"] == -1 and detail["min_value_text"] == "-1.0"


def test_worst_detail_handles_strings_and_nonfinite_minima():
    details = [
        {"passed": True, "min_value": 0.5, "min_value_text": "0.5"},
        {"passed": False, "min_value": "-1.0e-500", "min_value_text": "-1.0e-500"},
        {"passed": False, "min_value": -2.0, "min_value_text": "-2.0"},
        {"passed": False, "min_value": "nan", "min_value_text": "nan"},
    ]
    assert worst_certification_detail(details) is details[3]
    assert worst_certification_detail(details[:3]) is details[2]
    assert worst_certification_detail([]) is None


@pytest.mark.parametrize("kind, result", [
    ("gap", "dual_feasible (exclusion) result"),
    ("ope_max", "OPE-coefficient bound"),
    ("normalized_moment_upper", "moment bound"),
])
def test_failure_warning_names_the_problem_and_the_tower_twist(kind, result):
    details = [
        {"ell": 12, "twist": 2.5, "twist_label": "5/2", "passed": False},
        {"ell": 4, "twist": None, "passed": False},
        {"ell": 2, "twist": None, "passed": True},
    ]
    with pytest.warns(UserWarning) as record:
        C._warn_certification_failed(SimpleNamespace(problem_kind=kind), details)
    message = str(record[0].message)
    assert "tower twist 5/2, L >= 12" in message and "spin 4" in message and "spin 2" not in message
    assert result in message and f"this {kind} problem" in message


def test_skipped_certification_returns_a_failed_detail(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "y.txt").write_text("3 1\n1\n1\n1\n")
    problem = SimpleNamespace(spec=SimpleNamespace(dps=30), F_pairs=[(0, 0), (1, 0)], spin_channels=[],
                              pmp_build=None)
    ok, details = C.certify_dual_from_sdpb_output(problem, out, q_gate="off", warn=False)
    assert not ok and details[0]["reason"] == "dual_vector_length_mismatch"
    assert worst_certification_detail(details) is details[0]


def test_one_spin_override_is_not_a_twist_floor(monkeypatch):
    from virasoro_bootstrap.controls import global_control

    calls = []
    monkeypatch.setattr(global_control, "build_global_scalar_spin_channel", lambda **kw: calls.append(kw))

    def extra_edges(channels, **spec_changes):
        calls.clear()
        problem = SimpleNamespace(spec=_spec(**spec_changes), spin_channels=channels,
                                  resolved_method="global_control")
        C._build_extra_spin_channels(problem, 1)
        return [(kw["ell"], kw["delta_min"]) for kw in calls]

    fixed = lambda ell, delta: SimpleNamespace(ell=ell, delta_min=delta)  # noqa: E731
    # delta_min_by_spin = {2: 3} constrains spin 2 alone: spin 4 is checked from its unitarity bound
    assert extra_edges([fixed(0, 1), fixed(2, 3)]) == [(4, 4)]
    # a twist gap shared by two spins is the problem's convention, capped at the lowest tower twist
    assert extra_edges([fixed(0, 1), fixed(2, 3), fixed(4, 5)]) == [(6, 7)]
    assert extra_edges([fixed(0, 1), fixed(2, 3), fixed(4, 5)], spin_list=(0, 2, 4), large_spin_lmin=6,
                       large_spin_twists=("0.5", "2")) == [(6, Fraction(13, 2))]


# ---------------------------------------------------------------------------
# Spec validation and the prepared-build key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("changes, message", [
    ({"pole_selection": "bogus"}, "pole_selection must be one of"),
    ({"large_spin_pole_selection": "bogus"}, "large_spin_pole_selection must be one of"),
    ({"regularize": "bogus"}, "regularize must be"),
    ({"sample_via": "bogus"}, "sample_via must be"),
    ({"large_spin_npoles": -3}, "large_spin_npoles must be a non-negative integer"),
    ({"large_spin_lmin": 4, "large_spin_twists": ("0.25", "1/4")}, "twice"),
])
def test_spec_refuses_invalid_choices_at_construction(changes, message):
    with pytest.raises(ValueError, match=message):
        _spec(**changes)


def test_non_integer_chiral_norm_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="positive integer N.*non-integer --chiral-norm"):
        S._validate_gap_spec_common(_spec(chiral_norm="16.5"))


def test_reuse_key_resolves_the_default_normalization_and_names_mismatches():
    def key(spec):
        spin_list, _coordinate, c, h_ext, prefactor = S._validate_gap_spec_common(spec)
        return S._prepared_reuse_key(spec, spin_list=spin_list, c=c, h_ext=h_ext, prefactor_exponent=prefactor)

    # a manifest round trip records chiral_norm None as 16
    assert key(_spec()) == key(_spec(chiral_norm=16))
    mismatch = S._reuse_key_mismatch(key(_spec()), key(_spec(q_order=12, chiral_norm=4)))
    assert "q_order (prepared 10, requested 12)" in mismatch and "chiral_norm (prepared 16, requested 4)" in mismatch
