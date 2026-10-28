"""Tests for the q-order convergence gate (virasoro_blocks.qgate).

The gate estimates the q-truncation accuracy of the chiral block series by
differencing evaluator builds at three q_orders (the recursion is triangular
in k, so the differences are exactly the tail windows) and extrapolating the
window ratio geometrically. These tests calibrate the estimate against ground
truth (a q_order + 16 reference), check the loud triggers (starved q_order,
non-geometric large-c regime), exact singular-c probes, caching, and the
CLI / bootstrap wiring.
"""

import subprocess
import sys
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.derivs import chiral_z_series_evaluator
from virasoro_blocks.qgate import (
    QConvergenceError,
    gate_q_convergence,
    q_tail_report,
)

H_GEN = "0.075"  # generic external chiral weight (Delta_ext = 0.15)


def test_custom_q_probes_accept_exact_rational_labels():
    with mp.workdps(30):
        options = dict(c=32, h_externals=['.1']*4, q_order=8, n_terms=1)
        expected = q_tail_report(**options, h_probes=['7/10'])
        actual = q_tail_report(**options, h_probes=[Fraction(7, 10)])
        assert actual == expected


def test_q_gate_nudges_a_genuine_probe_pole():
    with mp.workdps(30):
        report = q_tail_report(c=32, h_externals=['.1','.2','.3','.4'],
                               q_order=8, n_terms=1, h_probes=[0])
        assert report['status'] != 'unavailable'
        assert Fraction(report['per_probe'][0]['h']) == Fraction('0.0587')
        assert mp.mp.dps == 30


def test_q_gate_rejects_an_empty_center_sequence_with_a_validation_error():
    with pytest.raises(ValueError, match='center'):
        q_tail_report(c=32, h_externals=['.1']*4, q_order=8, n_terms=1, center=[])


def test_q_reports_preserve_native_charge_recognition_before_input_guards():
    from virasoro_blocks.qgate import point_q_tail_report, super_point_q_tail_report, super_q_tail_report

    with mp.workdps(30):
        native, exact = mp.mpf('.7'), Fraction(7, 10)
        exchange = Fraction(1, 10)+Fraction(1, 10**70)
        options = dict(h_externals=['.1','.2','.3','.4'], q_order=16)
        for report in (
            lambda charge: q_tail_report(c=charge, **options, n_terms=1, h_probes=[exchange]),
            lambda charge: point_q_tail_report(c=charge, **options, h=exchange, z='.5'),
        ):
            assert report(native) == report(exact)
        for family, external in (('ns', Fraction(1, 10)), ('ramond', Fraction(3, 80))):
            options = dict(family=family, h_externals=[external]*4, q_order=8, sector='even', branch='++')
            for report in (
                lambda charge: super_q_tail_report(c=charge, **options, n_terms=1),
                lambda charge: super_point_q_tail_report(c=charge, **options, h='.4', z='.5'),
            ):
                assert report(native) == report(exact)
        assert mp.mp.dps == 30


@pytest.mark.parametrize("point", [False, True])
@pytest.mark.parametrize("order", [4, 8])
def test_super_gate_rejects_unknown_family(point, order):
    from virasoro_blocks.qgate import super_point_q_tail_report, super_q_tail_report

    kwargs = dict(family="typo", c=2, h_externals=[".125"] * 4, q_order=order, sector="even")
    with pytest.raises(ValueError, match="family"):
        if point:
            super_point_q_tail_report(**kwargs, h=".3", z=".5")
        else:
            super_q_tail_report(**kwargs, n_terms=2)


def test_super_table_gate_propagates_sampler_validation_errors(monkeypatch):
    from virasoro_blocks import superconformal_blocks as blocks
    from virasoro_blocks.qgate import super_q_tail_report

    def invalid_sampler(*args, **kwargs):
        raise ValueError("invalid sampler input")

    monkeypatch.setattr(blocks, "numeric_chiral_table", invalid_sampler)
    with pytest.raises(ValueError, match="invalid sampler input"):
        super_q_tail_report(family="ns", c=2, h_externals=[".125"] * 4, q_order=8, n_terms=2)


def test_super_point_gate_normalizes_ns_family_alias(monkeypatch):
    from virasoro_blocks import recursion_ns
    from virasoro_blocks.qgate import super_point_q_tail_report

    calls = []
    def ns_series(*args, **kwargs):
        calls.append(kwargs)
        return [mp.mpf(1)] + [mp.mpf(0)] * 16

    monkeypatch.setattr(recursion_ns, "H_q_series_ns", ns_series)
    report = super_point_q_tail_report(family="NS", c=2, h_externals=[".125"] * 4,
                                      h=".3", z=".5", q_order=8, sector="even")
    assert calls
    assert report["status"] == "ok"


def test_super_table_gate_still_retries_genuine_poles(monkeypatch):
    from virasoro_blocks import superconformal_blocks as blocks
    from virasoro_blocks.qgate import super_q_tail_report

    weights = []
    def sampler(family, h, *args, **kwargs):
        weights.append(h)
        if len(weights) == 1:
            raise blocks.SuperconformalPoleError("test pole", family=family, exchanged=h, pole=h)
        return [[mp.mpf(1)], [mp.mpf(1)]]

    monkeypatch.setattr(blocks, "numeric_chiral_table", sampler)
    report = super_q_tail_report(family="ns", c=2, h_externals=[".125"] * 4,
                                q_order=8, n_terms=2)
    assert weights[1] != weights[0]
    assert len(report["per_probe"]) == 3
    assert report["status"] == "ok"


@pytest.mark.parametrize("center", [None, ".3"])
def test_gate_preserves_tiny_exact_c_shifts(center):
    from fractions import Fraction
    from virasoro_blocks.numeric_input import parse_number

    c = parse_number(".7") + Fraction(1, 10**68)
    def report():
        return q_tail_report(c=c, h_externals=[mp.mpf(".125")] * 4,
                             q_order=16, n_terms=2, center=center)
    with mp.workdps(35):
        low = report()
    assert isinstance(parse_number(low["probe_c"]), Fraction)
    assert parse_number(low["probe_c"]) == c
    with mp.workdps(100):
        high = report()
    assert low["status"] == high["status"]
    assert low["digits"] == pytest.approx(high["digits"], abs=1e-10)


def test_gate_cached_reports_do_not_share_mutable_probe_data(monkeypatch):
    import virasoro_blocks.qgate as gate

    monkeypatch.setattr(gate, "_REPORT_CACHE", {})
    with mp.workdps(35):
        kwargs = dict(c=2, h_externals=[mp.mpf(".125")] * 4, q_order=16, n_terms=2)
        first = q_tail_report(**kwargs)
        expected = first["per_probe"][0]["digits"]
        first["per_probe"][0]["digits"] = -12345
        second = q_tail_report(**kwargs)
        assert second["per_probe"][0]["digits"] == expected
        second["per_probe"].clear()
        assert q_tail_report(**kwargs)["per_probe"]


@pytest.mark.parametrize("family", ["ns", "ramond"])
def test_super_table_gate_records_the_exact_charge(family):
    from virasoro_blocks.qgate import super_q_tail_report
    from virasoro_blocks.numeric_input import parse_number

    c = parse_number(".7") + parse_number("1e-60")
    with mp.workdps(30):
        report = super_q_tail_report(family=family, c=c, h_externals=[parse_number(".125")] * 4,
                                     q_order=8, n_terms=1)
    assert parse_number(report["probe_c"]) == c


def test_zero_truncated_sum_with_nonzero_terms_does_not_pass_the_gate():
    from virasoro_blocks.qgate import _window_report, act_on_q_report

    terms = list(map(mp.mpf, [1, 0, 0, 0, 0, -2, 0, 1, 0]))
    report = _window_report(terms, 2, q_order=8, per_order=1, probe_c=2)
    assert report["status"] == "unconverged"
    with pytest.raises(QConvergenceError, match="zero"):
        act_on_q_report(report, mode="strict")


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), True, "invalid"])
def test_gate_refuses_invalid_accuracy_thresholds(threshold):
    from virasoro_blocks.qgate import act_on_q_report

    with pytest.raises(ValueError, match="min_digits"):
        act_on_q_report({"status": "ok", "digits": 1, "q_order": 8}, mode="strict", min_digits=threshold)


@pytest.mark.parametrize("report", [{"status": "ok", "digits": float("nan")},
                                   {"status": "ok", "digits": None},
                                   {"status": "typo", "digits": 100}])
def test_strict_gate_refuses_invalid_reports(report):
    from virasoro_blocks.qgate import act_on_q_report

    with pytest.raises(ValueError, match="report"):
        act_on_q_report({**report, "q_order": 8}, mode="strict")


def test_strict_gate_still_accepts_zero_measured_tail():
    from virasoro_blocks.qgate import act_on_q_report

    act_on_q_report({"status": "ok", "digits": float("inf"), "q_order": 8}, mode="strict")


def _true_digits(c, h_ext, q_order, n_terms, h_probes):
    """Ground-truth digits: compare against a q_order + 16 reference build."""
    worst = mp.mpf(0)
    for h in h_probes:
        ev = chiral_z_series_evaluator(list(h_ext), n_terms, c=c, q_order=q_order)
        ref = chiral_z_series_evaluator(
            list(h_ext), n_terms, c=c, q_order=q_order + 16
        )
        F = [mp.mpc(v) for v in ev(h)]
        R = [mp.mpc(v) for v in ref(h)]
        max_r = max(abs(v) for v in R)
        for i in range(n_terms):
            scale = max(abs(R[i]), max_r * mp.mpf("1e-8"))
            worst = max(worst, abs(F[i] - R[i]) / scale)
    return float(-mp.log10(max(worst, mp.mpf("1e-300"))))


@pytest.mark.parametrize(
    "c_str, q_order, n_terms",
    [
        ("1.1", 12, 7),
        ("30", 30, 12),
        ("300", 16, 7),
    ],
)
def test_gate_estimate_tracks_ground_truth(c_str, q_order, n_terms):
    """The gate's digit estimate matches the actual truncation error to
    within ~2 digits across small, moderate, and large central charge."""
    mp.mp.dps = 80
    c = mp.mpf(c_str)
    h_ext = [mp.mpf(H_GEN)] * 4
    rep = q_tail_report(c=c, h_externals=h_ext, q_order=q_order, n_terms=n_terms)
    assert rep["status"] == "ok"
    probes = [mp.mpf(p) for p in ("0.1371", "1.6371", "7.1371")]
    true = _true_digits(c, h_ext, q_order, n_terms, probes)
    assert abs(rep["digits"] - true) < 2.0, (rep["digits"], true)


def test_gate_estimate_tracks_truth_large_externals():
    """Large external dimensions starve q_order (the twist-gap runaway
    regime); the gate must track that too."""
    mp.mp.dps = 80
    c = mp.mpf("8.7")
    h_ext = [mp.mpf("6.0")] * 4
    for q_order, n_terms in ((16, 7), (40, 7)):
        rep = q_tail_report(c=c, h_externals=h_ext, q_order=q_order, n_terms=n_terms)
        assert rep["status"] == "ok"
        probes = [mp.mpf(p) for p in ("0.1371", "1.6371", "7.1371")]
        true = _true_digits(c, h_ext, q_order, n_terms, probes)
        assert abs(rep["digits"] - true) < 2.0, (q_order, rep["digits"], true)


def test_gate_flags_nongeometric_regime_as_unconverged():
    """At very large c the H_k grow before decaying; a low q_order is not
    merely inaccurate, it is outside the geometric regime entirely. The gate
    must say so rather than extrapolate garbage (true error there is ~10^3)."""
    mp.mp.dps = 80
    rep = q_tail_report(
        c=mp.mpf("3000"), h_externals=[mp.mpf(H_GEN)] * 4, q_order=16, n_terms=7
    )
    assert rep["status"] == "unconverged"
    with pytest.raises(QConvergenceError, match="geometric regime"):
        gate_q_convergence(
            c=mp.mpf("3000"), h_externals=[mp.mpf(H_GEN)] * 4,
            q_order=16, n_terms=7, mode="strict",
        )


def test_gate_healthy_config_passes_silently(capsys):
    """A production-shaped config passes without output in warn mode."""
    mp.mp.dps = 60
    rep = gate_q_convergence(
        c=mp.mpf("8.7"), h_externals=[mp.mpf(H_GEN)] * 4,
        q_order=30, n_terms=8, mode="warn",
    )
    assert rep["status"] == "ok"
    assert rep["digits"] > 20
    assert capsys.readouterr().err == ""


def test_gate_warn_mode_prints_and_returns(capsys):
    mp.mp.dps = 60
    rep = gate_q_convergence(
        c=mp.mpf("300"), h_externals=[mp.mpf(H_GEN)] * 4,
        q_order=12, n_terms=8, mode="warn", label="unit test",
    )
    err = capsys.readouterr().err
    assert "q-order gate [unit test]" in err
    assert rep["status"] in ("ok", "unconverged")
    if rep["status"] == "ok":
        assert rep["digits"] < 20
        assert rep["suggested_q_order"] is not None
        assert rep["suggested_q_order"] > 12


def test_gate_probes_singular_c_exactly():
    """At a recognized singular locus the gate probes c0 itself, at the working precision: the evaluators are built
    there exactly (merged poles of the Laurent table), so there is no shifted charge and no precision tripling.  The
    tail it measures agrees with the shifted probe it replaced (c + 10^-(dps//2) at 3x precision)."""
    mp.mp.dps = 60
    rep = q_tail_report(
        c=mp.mpf("0.5"), h_externals=[mp.mpf("0.0625")] * 4,
        q_order=20, n_terms=7,
    )
    assert rep["status"] == "ok"
    assert mp.mpf(rep["probe_c"]) == mp.mpf("0.5")
    assert rep["digits"] > 15  # small c converges fast


def test_gate_coordinate_families():
    """All three chiral coordinate families are probeable; digits agree to a
    few digits (same q-tail physics, different geometry factors)."""
    mp.mp.dps = 60
    vals = {}
    for coord in ("zzb", "yyb", "qqb"):
        rep = q_tail_report(
            c=mp.mpf("8.7"), h_externals=[mp.mpf(H_GEN)] * 4,
            q_order=20, n_terms=6, coordinate=coord,
        )
        assert rep["status"] == "ok", coord
        vals[coord] = rep["digits"]
    spread = max(vals.values()) - min(vals.values())
    assert spread < 5.0, vals


def test_cli_q_gate_strict_errors_on_starved_order(tmp_path):
    cmd = [
        sys.executable, "-m", "virasoro_blocks",
        "table", "--family", "virasoro", "--basis", "scalar_G",
        "--central-charge", "300", "--spin-ranges", "0", "--lambda", "4",
        "--delta-ext", "0.15", "0.15", "0.15", "0.15",
        "--order", "10", "--precision", "60", "--npoles", "6",
        "--q-gate", "strict", "-o", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "q-order gate" in result.stderr


def test_cli_q_gate_warn_still_writes_output(tmp_path):
    cmd = [
        sys.executable, "-m", "virasoro_blocks",
        "table", "--family", "virasoro", "--basis", "scalar_G",
        "--central-charge", "300", "--spin-ranges", "0", "--lambda", "4",
        "--delta-ext", "0.15", "0.15", "0.15", "0.15",
        "--order", "10", "--precision", "60", "--npoles", "6",
        "-o", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "q-order gate" in result.stderr
    assert (tmp_path / "derivs_0.json").exists()


def test_prepare_identical_scalar_gap_build_warns_on_starved_q(capsys):
    """The bootstrap prepare path surfaces the gate warning before the
    (expensive) block build."""
    from virasoro_bootstrap import prepare_identical_scalar_gap_build
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec

    mp.mp.dps = 40
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("300"), h_ext=mp.mpf(H_GEN), delta_gap=mp.mpf("0.9"),
        lam=2, spin_list=(0,), q_order=10, dps=40, npoles=6,
    )
    prepare_identical_scalar_gap_build(spec)
    assert "q-order gate [identical-scalar build]" in capsys.readouterr().err
