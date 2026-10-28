"""Torus one-point accuracy, precision, singular charges, and option contracts.

Check independent reference values, relative imaginary-part guards,
anchors, pole counts, and evaluation truncation.
"""

from __future__ import annotations

import json

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main
from virasoro_blocks.api import build_table_request
from virasoro_blocks.modular import TORUS_COORDINATE
from virasoro_blocks.requests import BlockRequest, NumericalPolicy, RequestValidationError
from virasoro_blocks.table_adapters import table_adapter_for
from virasoro_blocks.torus_onept import (
    TorusOnePtTableError,
    _coerce_real_coeffs,
    build_torus_onept_state,
    torus_onept_delta_table,
    torus_onept_table_values,
)


def _onept_request(c, d, *, lam, q_order, npoles, precision, gap=None, scalar_shift=None):
    return table_adapter_for("torus-onepoint").request(
        central_charge=mp.mpf(c), torus_onepoint_ext=mp.mpf(d), lambda_order=lam, q_order=q_order,
        npoles=npoles, precision=precision, gap=gap, torus_scalar_shift=scalar_shift,
    )


def _true_digits(table, c, d, spin, lam, q_order, precision):
    """Digits the table reproduces the block to at every Delta >= shift -- between its nodes and far past the last
    one, at points of its own choosing -- against a much deeper, much more precise recursion (independent of the
    table's own verification reference)."""
    xs = sorted(mp.re(x) for x in table.sample_points_x)
    kept = [(m, n) for m in range(lam + 1) for n in range(min(m, lam - m) + 1)]
    worst = None
    with mp.workdps(2 * precision + 30):
        truth = build_torus_onept_state(q_order + 10, d_ext=mp.mpf(d), c=mp.mpf(c))
        between = [(xs[i] + xs[i + 1]) / 2 for i in range(0, len(xs) - 1, 2)]
        for x in between + [3 * xs[-1], 40 * xs[-1], 3000 * xs[-1]]:
            D = x + table.shift
            ref = torus_onept_table_values(truth, spin, lam, D)
            got = table.evaluate(TORUS_COORDINATE, D)
            scale = max(abs(ref[m][n]) for m, n in kept)
            err = max(abs(got[m][n] - ref[m][n]) / max(abs(ref[m][n]), scale * mp.mpf(10) ** -precision)
                      for m, n in kept)
            digits = -mp.log10(err)
            worst = digits if worst is None else min(worst, digits)
    return float(worst)


@pytest.mark.parametrize(
    "c, d, q_order, precision, limit",
    [
        ("32.7", "5.7", 14, 30, "fit_digits"),         # order reduction (npoles) binds
        ("10", "0.8", 3, 30, "q_truncation_digits"),   # a short --order binds
    ],
)
def test_onept_table_reports_verified_digits_that_match_the_truth(c, d, q_order, precision, limit, capsys):
    """T1: the table is built at the requested precision and says how many digits it delivers -- within about a
    digit of an independent deep, high-precision truth -- and which of npoles / precision / order limits it.
    (Before, it recorded nothing: this c = 32.7 table delivers 14 of 30 digits.)"""
    lam, spin = 4, 1
    with mp.workdps(precision):
        request = _onept_request(c, d, lam=lam, q_order=q_order, npoles=8, precision=precision)
    table = build_table_request(request, spin=spin)
    meta = table.metadata
    verified = table.verified_digits
    assert verified is not None and verified < precision
    assert meta[limit] == pytest.approx(verified, abs=1.0)
    assert abs(_true_digits(table, c, d, spin, lam, q_order, precision) - verified) < 1.5
    assert "reproduces the block to" in capsys.readouterr().err


def test_onept_table_measures_the_recursions_precision_loss():
    """T1: a large external makes the recursion cancel (10 of 20 digits here before the level table was certified from
    its own balls, 2026-09-29; the table now keeps its digits); the precision part says what remains, and it agrees
    with the kernel's error against a deep high-precision kernel."""
    c, d, spin, lam, q_order, precision = "32.7", "100", 0, 0, 20, 20
    with mp.workdps(precision):
        request = _onept_request(c, d, lam=lam, q_order=q_order, npoles=16, precision=precision,
                                 gap=mp.mpf("0.2"))
    table = build_table_request(request, spin=spin)
    lost = table.metadata["precision_digits"]
    assert precision - 5 < lost <= precision + 1
    with mp.workdps(precision):
        kernel = build_torus_onept_state(q_order, d_ext=mp.mpf(d), c=mp.mpf(c))
        own = torus_onept_table_values(kernel, spin, lam, mp.mpf("0.5"), guard_imag=False)[0][0]
    with mp.workdps(3 * precision):
        deep = build_torus_onept_state(q_order, d_ext=mp.mpf(d), c=mp.mpf(c))
        exact = torus_onept_table_values(deep, spin, lam, mp.mpf("0.5"))[0][0]
        measured = -mp.log10(abs(own - exact) / abs(exact))
    assert measured > lost - 1.5


def test_onept_table_records_verified_range_and_node_values(tmp_path):
    """T2 + node values: the verified range spans the check points (between the nodes and out to 1024 x the last);
    the node values are written and round-trip -- the block's for the highest-degree (m + n = lambda) entries, each
    lower entry's own interpolant (as on the sphere); npoles is the kept count (T9)."""
    from virasoro_blocks.serialization import serialized_table_from_json
    from virasoro_blocks.torus_onept import torus_onept_check_points_x

    rc = main([
        "table", "--surface", "torus", "--family", "torus-onepoint", "--central-charge", "9.4",
        "--torus-onept-ext", "0.83", "--spin-ranges", "2", "--lambda", "3", "--order", "10",
        "--npoles", "5", "--precision", "30", "-o", str(tmp_path),
    ])
    assert rc == 0
    doc = json.loads((tmp_path / "derivs_2.json").read_text())
    with mp.workdps(30):
        nodes = [mp.mpf(x) for x in doc["sample_points_x"]]
        lo, hi = (mp.mpf(x) for x in doc["verified_delta_range"])
        assert lo == mp.mpf(doc["variable_minus_x"]) == 2
        assert abs(hi - (lo + max(torus_onept_check_points_x(nodes)))) < mp.mpf("1e-20")
        assert hi - lo >= 1000 * max(nodes)
        # a conjugate pair is never split: the 5-pole cap keeps 4 here (it recorded 5)
        assert doc["npoles"] == len(doc["pole_list_x"]) == 4
        assert doc["npoles_requested"] == 5
        for key in ("verified_digits", "fit_digits", "precision_digits", "q_truncation_digits"):
            assert float(doc[key]) > 0
        back = serialized_table_from_json(doc)
        values = back.sample_block_values[TORUS_COORDINATE]
        assert len(values) == len(nodes)
        kernel = build_torus_onept_state(10, d_ext=mp.mpf("0.83"), c=mp.mpf("9.4"))
        direct = torus_onept_table_values(kernel, 2, 3, nodes[3] + 2)
        own = back.evaluate(TORUS_COORDINATE, nodes[3] + 2)
        for m in range(4):
            for n in range(min(m, 3 - m) + 1):
                expected = direct[m][n] if m + n == 3 else own[m][n]
                assert abs(values[3][m][n] - expected) <= mp.mpf("1e-25") * abs(expected)


def test_onept_adapter_builds_at_the_request_precision():
    """The adapter builds at the request's precision even when ambient precision is lower."""
    request = _onept_request("7/3", "0.8", lam=3, q_order=10, npoles=8, precision=40)
    adapter = table_adapter_for("torus-onepoint")
    with mp.workdps(15):
        kernel = adapter.build_kernel(request)
        table = build_table_request(request, spin=1, kernel=kernel)
    with mp.workdps(40):
        reference = build_table_request(request, spin=1)
        a = table.evaluate(TORUS_COORDINATE, mp.mpf("2.3"))
        b = reference.evaluate(TORUS_COORDINATE, mp.mpf("2.3"))
        assert table.pole_list_x == reference.pole_list_x
        assert abs(a[2][1] - b[2][1]) <= mp.mpf("1e-35") * abs(b[2][1])


def test_reusing_onepoint_kernels_checks_inputs_and_verifies_requested_order():
    from fractions import Fraction
    from virasoro_blocks.torus_onept import torus_onept_point_value

    adapter = table_adapter_for('torus-onepoint')
    request = adapter.request(central_charge=4, torus_onepoint_ext='.7', lambda_order=0,
                              q_order=1, npoles=2, precision=30, gap='.2')
    with mp.workdps(30):
        kernel = build_torus_onept_state(4, c=4, d_ext='.7')
        table = adapter.build_table(request, spin=1, kernel=kernel)
        # Order one with both poles retained is an exact rational fit.
        # Report its interpolation accuracy separately from q-series truncation.
        assert table.metadata['fit_digits'] > 25
        assert table.metadata['precision_digits'] > 25
        assert table.metadata['q_truncation_digits'] < 10
        with pytest.raises(ValueError, match='supplied kernel'):
            torus_onept_point_value(4+Fraction(1, 10**70), '.7', 1, 1, 0, 0, 0,
                                    q_order=1, kernel=kernel)
        reference = build_torus_onept_state(5, c=4, d_ext='.8')
        with pytest.raises(ValueError, match='supplied reference'):
            adapter.build_table(request, spin=1, kernel=kernel, reference=reference)
    # A Laurent state's guard precision does not change the precision it
    # returns. A lower-precision cache must be rebuilt for this request.
    with mp.workdps(15):
        low = build_torus_onept_state(1, c=26, d_ext='.7', work_dps=40)
        low_reference = build_torus_onept_state(5, c=26, d_ext='.7', work_dps=40)
    got, _info = torus_onept_point_value(26, '.7', 1, 1, 0, 0, 0, q_order=1, precision=30, kernel=low)
    request = adapter.request(central_charge=26, torus_onepoint_ext='.7', lambda_order=0,
                              q_order=1, npoles=2, precision=30, gap='.2')
    table = adapter.build_table(request, spin=1, kernel=low, reference=low_reference)
    with mp.workdps(80):
        q = mp.exp(-2*mp.pi)
        expected = q**(Fraction(-1, 12))*(1-Fraction(21, 200)*q)**2
        assert abs(got/expected-1) < mp.mpf('1e-28')
        delta = Fraction(6, 5)
        h, hbar = mp.convert(Fraction(11, 10)), mp.mpf('.1')
        expected = 2*q**(delta-Fraction(25, 12))*(1-Fraction(21, 100)*q/(2*h))*(1-Fraction(21, 100)*q/(2*hbar))
        assert abs(table.evaluate(TORUS_COORDINATE, delta)[0][0]/expected-1) < mp.mpf('1e-25')


def test_imaginary_guard_is_relative_to_half_the_digits():
    """Drop imaginary rounding noise below the relative guard; reject comparable parts."""
    with mp.workdps(60):
        noisy = [mp.mpc(1), mp.mpc("2.5", "1e-47"), mp.mpc("1e-40", "1e-50")]
        assert _coerce_real_coeffs(noisy, "test") == [1, mp.mpf("2.5"), mp.mpf("1e-40")]
        with pytest.raises(ValueError, match="imaginary part"):
            _coerce_real_coeffs([mp.mpc("2.5", "1e-20")], "test")
        # the case that was refused: complex Kac data (c = 2) at a large external, 48 of 60 digits clean
        kernel = build_torus_onept_state(24, d_ext=mp.mpf("5.7"), c=mp.mpf(2))
        assert len(torus_onept_table_values(kernel, 0, 0, mp.mpf(80))) == 1


def test_singular_c_onept_tables_and_values_use_the_laurent_kernel(tmp_path):
    """Singular charges use the exact Laurent kernel, including c = 1 with positive poles."""
    from virasoro_blocks.evaluation import evaluate_block_request
    from virasoro_blocks.torus_onept import (
        build_torus_onept_contour_kernels,
        torus_onept_f_coeffs_singular,
    )

    for c in ("25", "26", "1"):
        precision, cap = (65, 20) if c == "1" else (25, 8)
        rc = main([
            "table", "--surface", "torus", "--torus-onept-ext", "0.8", "--central-charge", c,
            "--spin-ranges", "1", "--lambda", "2", "--order", "8", "--npoles", str(cap),
            "--precision", str(precision),
            "-o", str(tmp_path / c),
        ])
        assert rc == 0
        doc = json.loads((tmp_path / c / "derivs_1.json").read_text())
        if c == "1":
            assert float(doc["fit_digits"]) > 40
            # Both sides of positive poles are checked: at order 8 the
            # deeper reference differs by more than on the node interval.
            assert 8 < float(doc["verified_digits"]) < 11
        else:
            assert float(doc["verified_digits"]) > 10
    with mp.workdps(30):
        assert issubclass(TorusOnePtTableError, ValueError)
        request = BlockRequest(
            operation="evaluate", surface="torus", family="torus-onepoint", coordinate="vvb",
            central_charge=mp.mpf(25), torus_onepoint_ext=mp.mpf("0.8"), exchange_delta=mp.mpf("2.3"),
            exchange_spin=0, point={"v": 0, "vbar": 0}, policy=NumericalPolicy(precision=30, order=6),
        )
        result = evaluate_block_request(request)
        assert result.metadata["regularization_policy"] == "laurent"
        # independent: the discrete Cauchy average around c = 25
        h = mp.mpf("1.15")
        contour = build_torus_onept_contour_kernels(6, d_ext=mp.mpf("0.8"), c0=mp.mpf(25))
        f = torus_onept_f_coeffs_singular(contour, h)
        q = mp.exp(-2 * mp.pi)
        chiral = q ** (h - 1) * sum(fn * q ** n for n, fn in enumerate(f))
        assert abs(result.value - chiral ** 2) < mp.mpf("1e-15") * abs(chiral ** 2)


def _table(args, tmp_path):
    return main([
        "table", "--surface", "torus", "--central-charge", "10", "--lambda", "2", "--precision", "25",
        "-o", str(tmp_path), *args,
    ])


_ONEPT = ["--family", "torus-onepoint", "--torus-onept-ext", "0.8", "--order", "8", "--npoles", "4"]


def test_gap_anchors_every_onepoint_spin_and_excludes_the_scalar_shift(tmp_path):
    """T6/T7: --family torus-onepoint --gap G is honored (every spin at |j| + G); with --torus-scalar-shift it is
    refused, as is a scalar shift without spin 0, and the request refuses both anchors at once."""
    assert _table([*_ONEPT, "--spin-ranges", "0", "2", "--gap", "0.3"], tmp_path) == 0
    shifts = [json.loads((tmp_path / f"derivs_{j}.json").read_text())["variable_minus_x"] for j in (0, 2)]
    assert [mp.mpf(s) for s in shifts] == [mp.mpf("0.3"), mp.mpf("2.3")]
    for extra in (["--spin-ranges", "0", "--gap", "0.3", "--torus-scalar-shift", "0.5"],
                  ["--spin-ranges", "1", "--torus-scalar-shift", "0.5"]):
        with pytest.raises(SystemExit):
            _table([*_ONEPT, *extra], tmp_path)
    with pytest.raises(RequestValidationError, match="both set the spin-0 anchor"):
        _onept_request("10", "0.8", lam=2, q_order=8, npoles=4, precision=25, gap=mp.mpf("0.3"),
                       scalar_shift=mp.mpf("0.5"))


@pytest.mark.parametrize("flag", [
    ["--pole-selection", "nearest"], ["--sample-via", "numeric"], ["--shared-sample-points"],
    ["--regularize", "on"], ["--num-threads", "2"], ["--super-sector", "odd"], ["--large-spin-twists", "2"],
])
def test_torus_tables_refuse_flags_they_do_not_use(flag, tmp_path, capsys):
    """T8: an explicit flag no torus table reads is refused, not silently ignored."""
    with pytest.raises(SystemExit):
        _table([*_ONEPT, "--spin-ranges", "1", *flag], tmp_path)
    assert "does not use " + flag[0] in capsys.readouterr().err


def test_character_tables_refuse_onepoint_only_flags_and_q_gate_is_honoured(tmp_path, capsys):
    """T8: character tables refuse --q-gate and --torus-scalar-shift; one-point tables honor --q-gate."""
    for extra in (["--q-gate", "warn"], ["--torus-scalar-shift", "0.5"]):
        with pytest.raises(SystemExit):
            _table(["--spin-ranges", "0", *extra], tmp_path)
    low_order = ["--family", "torus-onepoint", "--torus-onept-ext", "0.8", "--order", "3", "--npoles", "4",
                 "--spin-ranges", "1"]
    assert _table(low_order, tmp_path) == 0
    assert "q-gate" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _table([*low_order, "--q-gate", "strict"], tmp_path)


def test_evaluate_onepoint_reports_and_validates_its_truncation(capsys):
    """T9: evaluate records the order and the digits its truncation leaves (measured against a deeper
    reference), warns when they fall short of --precision, refuses --order 0 and the table-only anchor."""
    from virasoro_blocks.cli.evaluate import main as evaluate_main

    base = ["--surface", "torus", "--family", "torus-onepoint", "--coordinate", "vvb", "--central-charge", "10",
            "--torus-onept-ext", "0.8", "--point", "v=0", "--point", "vbar=0", "--exchange-delta", "2.3",
            "--exchange-spin", "1", "--precision", "30"]
    assert evaluate_main([*base, "--order", "3"]) == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["order"] == 3
    assert 5 < doc["q_truncation_digits"] < 20
    assert "--order 3 truncates" in captured.err
    assert evaluate_main([*base, "--order", "14"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["q_truncation_digits"] > 30 and "truncates" not in captured.err
    with pytest.raises(SystemExit):
        evaluate_main([*base, "--order", "0"])
    with pytest.raises(SystemExit):
        evaluate_main([*base, "--order", "5", "--torus-scalar-shift", "0.5"])


def test_evaluate_onepoint_is_verified_at_large_external():
    """T1 for evaluate: at a large external the level table cancels; it is certified from its own balls
    (2026-09-29), so at the working precision 20 alone the value holds ~19 digits (recorded as verified_digits, the
    precision left as given), and with target_digits the working precision is raised until it holds all 20."""
    from virasoro_blocks.evaluation import evaluate_block_request
    from virasoro_blocks.torus_onept import _onept_point

    c, d, delta = mp.mpf("32.7"), mp.mpf(50), mp.mpf("0.4")

    def value(precision, target=None):
        request = BlockRequest(
            operation="evaluate", surface="torus", family="torus-onepoint", coordinate="vvb",
            central_charge=c, torus_onepoint_ext=d, exchange_delta=delta, exchange_spin=0,
            point={"v": 0, "vbar": 0}, policy=NumericalPolicy(precision=precision, order=20, target_digits=target),
        )
        return evaluate_block_request(request)

    plain = value(20)
    assert plain.metadata["precision_actual"] == 20 and plain.metadata["verified_digits"] >= 18
    low = value(20, target=20)
    assert low.metadata["verified_digits"] >= 20
    with mp.workdps(20):
        naive = _onept_point(build_torus_onept_state(20, d_ext=d, c=c), delta / 2, delta / 2, 0, 0, 0, 20)
    with mp.workdps(60):
        high = value(60).value
        assert abs(low.value - high) < mp.mpf("1e-19") * abs(high)
        assert abs(naive - high) < mp.mpf("1e-18") * abs(high)


def _worst_digits(got, ref, lam):
    return min(float(-mp.log10(abs(got[m][n] - ref[m][n]) / abs(ref[m][n])))
               for m in range(lam + 1) for n in range(min(m, lam - m) + 1))


def test_onept_table_keeps_the_blocks_large_delta_power():
    """Entry (m, n) of a one-point table grows like Delta^(m + n) times the damping, so its numerator has degree
    (kept poles) + m + n, each on its own node set (as on the sphere).  One degree for every entry (15 at 5 kept
    poles) put the documented --lambda 3 --npoles 6 spin-1 table at 2.7e3 x the block at Delta = 51 and 3e13 x at
    501; now it holds 7 digits out to Delta = 5000, and the node values rebuild the same function."""
    from virasoro_blocks.interpolation import lagrange_matrix

    c, d, spin, lam, q_order = mp.mpf(10), mp.mpf("0.2"), 1, 3, 10
    with mp.workdps(30):
        kernel = build_torus_onept_state(q_order, d_ext=d, c=c)
        table = torus_onept_delta_table(c, spin, lam, d_ext=d, q_order=q_order, npoles=6, kernel=kernel)
        kept = len(table.pole_list_x)
        rows = table.numerators[TORUS_COORDINATE]
        for m in range(lam + 1):
            for n in range(min(m, lam - m) + 1):
                assert rows[m][n].degree == kept + m + n
        # the growth itself: R_mn = e^{2 pi Delta} * entry scales as Delta^(m + n)
        big = [torus_onept_table_values(kernel, spin, lam, D) for D in (mp.mpf(2000), mp.mpf(4000))]
        for m in range(lam + 1):
            for n in range(min(m, lam - m) + 1):
                ratio = abs(big[1][m][n] / big[0][m][n]) * mp.exp(2 * mp.pi * 2000)
                assert abs(mp.log(ratio, 2) - (m + n)) < 0.01
        nodes = [mp.re(x) for x in table.sample_points_x]
        values = table.sample_block_values[TORUS_COORDINATE]
        pref = [table._prefactor(x) for x in nodes]
        for x in (mp.mpf(10), mp.mpf(50), mp.mpf(200), mp.mpf(500), mp.mpf(5000)):
            D = table.shift + x
            direct = torus_onept_table_values(kernel, spin, lam, D)
            assert _worst_digits(table.evaluate(TORUS_COORDINATE, D), direct, lam) > 6.5
            # a consumer's rebuild from the node values (the Lagrange basis by products at the writer nodes)
            weights = [mp.mpf(w.mid().str(40, radius=False)) for w in lagrange_matrix(nodes, [x]).entries()]
            rebuilt = [[table._prefactor(x) * mp.fsum(wk * values[k][m][n] / pref[k] for k, wk in enumerate(weights))
                        for n in range(lam + 1)] for m in range(lam + 1)]
            assert _worst_digits(rebuilt, direct, lam) > 6.5


def test_onept_verified_digits_cover_delta_far_past_the_nodes(capsys):
    """The recorded verified_digits hold at every Delta a consumer reads: the check runs out to 1024 x the last
    node.  (The documented table recorded 13 digits, checked on [1, 4.4] only, while it was 2.7e3 x the block at
    Delta = 51.)"""
    c, d, spin, lam, q_order, precision = "10", "0.2", 1, 3, 10, 30
    with mp.workdps(precision):
        request = _onept_request(c, d, lam=lam, q_order=q_order, npoles=6, precision=precision)
    table = build_table_request(request, spin=spin)
    lo, hi = table.metadata["verified_delta_range"]
    assert hi - lo > 1000
    with mp.workdps(2 * precision):
        truth = build_torus_onept_state(q_order + 10, d_ext=mp.mpf(d), c=mp.mpf(c))
        delivered = min(_worst_digits(table.evaluate(TORUS_COORDINATE, lo + x),
                                      torus_onept_table_values(truth, spin, lam, lo + x), lam)
                        for x in (mp.mpf(50), mp.mpf(500), mp.mpf(2000)))
    assert table.verified_digits <= delivered + 0.5
    assert abs(_true_digits(table, c, d, spin, lam, q_order, precision) - table.verified_digits) < 1.0
    assert f"over x = Delta - shift in [0, {mp.nstr(hi - lo, 3)}]" in capsys.readouterr().err


def test_torus_character_json_records_precision_actual(tmp_path):
    """Character tables record precision_actual like every other table (it was missing)."""
    assert _table(["--spin-ranges", "0", "2"], tmp_path) == 0
    for j in (0, 2):
        doc = json.loads((tmp_path / f"derivs_{j}.json").read_text())
        assert doc["precision_input"] == doc["precision_actual"] == 25
