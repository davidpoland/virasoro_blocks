"""Positive-pole splitting for torus one-point interpolated families."""

import json
import subprocess
import sys
from collections import Counter

import mpmath as mp
import pytest

from virasoro_blocks.modular import TORUS_COORDINATE
from virasoro_blocks.torus_onept import (
    _onept_positive_principal_parts,
    build_torus_onept_state,
    torus_onept_delta_table,
    torus_onept_table_values,
)


def entries(lam):
    return [(m, n) for m in range(lam+1) for n in range(min(m, lam-m)+1)]


@pytest.mark.parametrize("c,spin,shift", [
    ("0.73", 0, "0"), ("1", 0, "0"), ("0.5", 2, "2"),
    ("-0.7", 1, "1.2"), ("1", 0, "0.5"),
])
def test_finite_order_fit_on_both_sides_of_poles_and_in_tail(c, spin, shift):
    """Keeping all remaining poles gives the finite-order block throughout its domain."""
    with mp.workdps(85):
        lam, order = 3, 6
        shift = mp.mpf(shift)
        kernel = build_torus_onept_state(order, c=c, d_ext="0.83")
        table = torus_onept_delta_table(c, spin, lam, shift=shift, d_ext="0.83",
                                       q_order=order, npoles=100, kernel=kernel)
        positive = sorted({mp.re(p) for p in table.pole_list_x if mp.im(p) == 0 and mp.re(p) > 0})
        assert positive
        xs = [mp.mpf(x) for x in ("0.17", "1.4", "100", "1000")]
        for i, pole in enumerate(positive):
            gap = min(mp.mpf("0.0001"), pole/10,
                      (pole-positive[i-1])/10 if i else pole/10)
            xs.extend((pole-gap, pole+gap))
        for x in xs:
            expected = torus_onept_table_values(kernel, spin, lam, shift+x)
            got = table.evaluate(TORUS_COORDINATE, shift+x)
            for m, n in entries(lam):
                assert abs(got[m][n]/expected[m][n]-1) < mp.mpf("1e-45")
        for m, n in entries(lam):
            assert table.numerators[TORUS_COORDINATE][m][n].degree == len(table.pole_list_x)+m+n


def test_colliding_c_one_poles_are_forced_even_with_zero_cap():
    with mp.workdps(65):
        zero = torus_onept_delta_table(1, 0, 2, d_ext="0.83", q_order=4, npoles=0)
        full = torus_onept_delta_table(1, 0, 2, d_ext="0.83", q_order=4, npoles=100)
        expected = Counter({mp.mpf("0.5"): 4, mp.mpf(2): 4, mp.mpf("4.5"): 4})
        assert Counter(mp.re(p) for p in zero.pole_list_x if mp.re(p) > 0) == expected
        assert Counter(mp.re(p) for p in full.pole_list_x if mp.re(p) > 0) == expected
        assert len(zero.sample_points_x) == 3  # only the regular remainder is sampled
        assert len(zero.pole_list_x) == 12


def test_principal_coefficients_match_independent_cauchy_extraction():
    """Contour samples of the full block recover all four orders at a merged c=1 pole."""
    with mp.workdps(85):
        lam, order, loc, radius, count = 2, 4, mp.mpf("0.5"), mp.mpf("0.001"), 32
        kernel = build_torus_onept_state(order, c=1, d_ext="0.83")
        _pool, _forced, record = _onept_positive_principal_parts(kernel, 0, lam, order, 0, entries(lam))
        base, parts = record
        expected = dict(parts)[loc]
        samples = []
        for k in range(count):
            z = radius * mp.exp(2j*mp.pi*k/count)
            values = torus_onept_table_values(kernel, 0, lam, loc+z, guard_imag=False)
            samples.append((z, [[v/base**(loc+z) for v in row] for row in values]))
        for entry in entries(lam):
            m, n = entry
            for a, coefficient in enumerate(expected[entry], 1):
                extracted = mp.fsum(values[m][n]*z**a for z, values in samples)/count
                assert abs(extracted-coefficient) < mp.mpf("1e-65") * max(1, abs(coefficient))


def test_deeper_kernel_does_not_add_poles_beyond_requested_order():
    with mp.workdps(70):
        deep = build_torus_onept_state(8, c=1, d_ext="0.83")
        fresh = torus_onept_delta_table(1, 1, 2, d_ext="0.83", q_order=4, npoles=20)
        reused = torus_onept_delta_table(1, 1, 2, d_ext="0.83", q_order=4, npoles=20, kernel=deep)
        assert fresh.pole_list_x == reused.pole_list_x
        for delta in ("1.17", "5.1", "100"):
            a, b = (t.evaluate(TORUS_COORDINATE, mp.mpf(delta)) for t in (fresh, reused))
            for m, n in entries(2):
                assert abs(a[m][n]/b[m][n]-1) < mp.mpf("1e-60")


@pytest.mark.parametrize("external", [0, 1])
def test_removable_poles_do_not_survive_splitting(external):
    with mp.workdps(50):
        kernel = build_torus_onept_state(4, c=1, d_ext=external)
        table = torus_onept_delta_table(1, 0, 2, d_ext=external, q_order=4, npoles=0, kernel=kernel)
        assert not table.pole_list_x
        expected = torus_onept_table_values(kernel, 0, 2, mp.mpf("0.7"))
        got = table.evaluate(TORUS_COORDINATE, mp.mpf("0.7"))
        for m, n in entries(2):
            assert abs(got[m][n]/expected[m][n]-1) < mp.mpf("1e-45")


def test_node_at_positive_pole_is_moved_before_sampling(monkeypatch):
    import virasoro_blocks.interpolation as interpolation

    original = interpolation.get_sample_points

    def colliding(prefactor, count):
        nodes = list(original(prefactor, count))
        nodes[0] = mp.mpf("0.5")
        return nodes

    monkeypatch.setattr(interpolation, "get_sample_points", colliding)
    with mp.workdps(65):
        kernel = build_torus_onept_state(4, c=1, d_ext="0.83")
        table = torus_onept_delta_table(1, 0, 2, d_ext="0.83", q_order=4, npoles=100, kernel=kernel)
        assert mp.mpf("0.5") not in table.sample_points_x
        expected = torus_onept_table_values(kernel, 0, 2, mp.mpf("1.3"))
        got = table.evaluate(TORUS_COORDINATE, mp.mpf("1.3"))
        for m, n in entries(2):
            assert abs(got[m][n]/expected[m][n]-1) < mp.mpf("1e-45")


def test_writer_round_trip_preserves_per_entry_fits_and_forced_poles():
    from virasoro_blocks.serialization import serialized_table_from_json
    from virasoro_blocks.writers import write_torus_json

    with mp.workdps(80):
        kernel = build_torus_onept_state(6, c="0.5", d_ext="0.83")
        table = torus_onept_delta_table("0.5", 1, 3, d_ext="0.83", q_order=6, npoles=2, kernel=kernel)
        doc = json.loads(write_torus_json(table, c="0.5", spin=1, basis="torus_onept_G",
                                         onept_d_ext="0.83", q_order=6, npoles=2))
        assert doc["npoles_requested"] == 2
        assert doc["npoles"] == len(table.pole_list_x) > 2
        back = serialized_table_from_json(doc)
        for x, values in zip(back.sample_points_x, back.sample_block_values[TORUS_COORDINATE]):
            own = back.evaluate(TORUS_COORDINATE, x+1)
            direct = torus_onept_table_values(kernel, 1, 3, x+1)
            for m, n in entries(3):
                assert abs(values[m][n]/own[m][n]-1) < mp.mpf("1e-55")
                if m+n == 3:
                    assert abs(values[m][n]/direct[m][n]-1) < mp.mpf("1e-65")


def test_verification_near_poles_and_precision_escalation():
    from virasoro_blocks.table_adapters import table_adapter_for
    from virasoro_blocks.torus_onept import torus_onept_check_points_x

    adapter = table_adapter_for("torus-onepoint")
    request = adapter.request(central_charge=1, torus_onepoint_ext=".83", lambda_order=1,
                              q_order=12, npoles=50, precision=30, target_digits=12)
    table = adapter.build_table(request, spin=1)
    assert table.working_dps > 30
    assert table.verified_digits >= 12
    assert table.metadata["fit_digits"] > 25
    assert table.metadata["q_truncation_digits"] == pytest.approx(table.verified_digits, abs=0.1)
    with mp.workdps(table.working_dps):
        positive = sorted({mp.re(p) for p in table.pole_list_x if mp.re(p) > 0})
        points = torus_onept_check_points_x(table.sample_points_x, positive)
        for pole in (positive[0], positive[-1]):
            assert any(0 < pole-x < mp.mpf("0.001") for x in points)
            assert any(0 < x-pole < mp.mpf("0.001") for x in points)
        assert max(points) >= 1024 * positive[-1]


def test_verification_probes_both_sides_of_nearby_and_near_anchor_poles():
    from virasoro_blocks.torus_onept import torus_onept_check_points_x

    with mp.workdps(80):
        poles = [mp.mpf("1e-30"), mp.mpf("0.5"), mp.mpf("0.5")+mp.mpf("1e-20")]
        points = torus_onept_check_points_x([mp.mpf("0.1"), mp.mpf(1)], poles)
        for i, pole in enumerate(poles):
            gap = min(pole, poles[i]-poles[i-1] if i else pole,
                      poles[i+1]-poles[i] if i+1 < len(poles) else 1)
            assert any(0 < pole-x < gap/100 for x in points)
            assert any(0 < x-pole < gap/100 for x in points)


def test_small_principal_coefficients_keep_their_genuine_pole_orders():
    with mp.workdps(65):
        kernel = build_torus_onept_state(12, c=1, d_ext=".83")
        _pool, forced, record = _onept_positive_principal_parts(kernel, 0, 0, 12, 0, [(0, 0)])
        loc = mp.mpf("60.5")
        row = dict(record[1])[loc][0, 0]
        assert Counter(forced)[loc] == 4
        assert 0 < abs(row[-1]) < mp.mpf("1e-60")


def test_remainder_accuracy_improves_with_more_kept_poles():
    with mp.workdps(85):
        kernel = build_torus_onept_state(8, c=".73", d_ext=".83")
        delta = mp.mpf("2.37")
        expected = torus_onept_table_values(kernel, 2, 2, delta)
        errors = []
        for cap in (0, 2, 8):
            table = torus_onept_delta_table(".73", 2, 2, d_ext=".83", q_order=8, npoles=cap, kernel=kernel)
            got = table.evaluate(TORUS_COORDINATE, delta)
            errors.append(max(abs(got[m][n]/expected[m][n]-1) for m, n in entries(2)))
        assert errors[2] < errors[1] < errors[0]
        assert errors[2] < mp.mpf("1e-12")


def _table_cli(output, *args):
    return subprocess.run(
        [sys.executable, "-m", "virasoro_blocks", "table", "--surface", "torus",
         "--torus-onept-ext", ".83", "--output-template", "positive_{spin}.json", "-o", str(output), *args],
        cwd=output, capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("c,anchors,cap,shifts", [
    (".73", (), None, ("0", "2")),
    (".5", ("--gap", ".3"), 100, (".3", "2.3")),
    ("1", ("--torus-scalar-shift", ".5"), 100, (".5", "2")),
    ("-.7", ("--gap", ".2"), 100, (".2", "2.2")),
])
def test_cli_positive_poles_and_anchor_options_match_direct_blocks(c, anchors, cap, shifts, tmp_path):
    from virasoro_blocks.serialization import serialized_table_from_json

    options = ["--central-charge", c, "--spin-ranges", "0", "2", "--lambda", "2",
               "--order", "6", "--precision", "85", "--q-gate", "off", *anchors]
    if cap is not None:
        options += ["--family", "torus-onepoint", "--npoles", str(cap)]
    result = _table_cli(tmp_path, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    with mp.workdps(90):
        kernel = build_torus_onept_state(6, c=c, d_ext=".83")
        for spin, shift in zip((0, 2), shifts):
            doc = json.loads((tmp_path / f"positive_{spin}.json").read_text())
            table = serialized_table_from_json(doc)
            assert mp.mpf(table.shift) == mp.mpf(shift)
            if cap is None:
                assert "npoles_requested" not in doc and doc["npoles_chosen"] >= 0
            else:
                assert doc["npoles_requested"] == cap and "npoles_chosen" not in doc
            assert doc["npoles"] == len(table.pole_list_x)
            assert any(mp.im(p) == 0 and mp.re(p) > 0 for p in table.pole_list_x)
            assert float(doc["fit_digits"]) > 40
            assert doc["regularization_policy"] == ("laurent" if c in {"1", ".5"} else "direct")
            for key in ("precision_digits", "q_truncation_digits", "verified_digits", "verified_delta_range"):
                assert key in doc
            for offset in (".17", "1.4", "100"):
                delta = mp.mpf(shift)+mp.mpf(offset)
                expected = torus_onept_table_values(kernel, spin, 2, delta)
                got = table.evaluate(TORUS_COORDINATE, delta)
                for m, n in entries(2):
                    assert abs(got[m][n]/expected[m][n]-1) < mp.mpf("1e-40")


def test_cli_zero_cap_retains_merged_poles_and_correct_near_pole_values(tmp_path):
    from virasoro_blocks.serialization import serialized_table_from_json

    result = _table_cli(tmp_path, "--central-charge", "1", "--spin-ranges", "0", "--lambda", "2",
                        "--order", "4", "--npoles", "0", "--precision", "70", "--q-gate", "off")
    assert result.returncode == 0, result.stdout + result.stderr
    doc = json.loads((tmp_path / "positive_0.json").read_text())
    with mp.workdps(80):
        table = serialized_table_from_json(doc)
        assert doc["npoles_requested"] == 0
        assert doc["npoles"] == 12
        assert Counter(mp.re(p) for p in table.pole_list_x) == Counter(
            {mp.mpf(".5"): 4, mp.mpf(2): 4, mp.mpf("4.5"): 4})
        kernel = build_torus_onept_state(4, c=1, d_ext=".83")
        for sign in (-1, 1):
            delta = mp.mpf(".5")+sign*mp.mpf("1e-12")
            expected = torus_onept_table_values(kernel, 0, 2, delta)
            got = table.evaluate(TORUS_COORDINATE, delta)
            for m, n in entries(2):
                assert abs(got[m][n]/expected[m][n]-1) < mp.mpf("1e-25")


def test_cli_target_digits_escalates_and_serializes_achieved_precision(tmp_path):
    result = _table_cli(tmp_path, "--central-charge", "1", "--spin-ranges", "1", "--lambda", "1",
                        "--order", "8", "--npoles", "50", "--precision", "15",
                        "--target-digits", "6", "--max-precision", "70", "--q-gate", "off")
    assert result.returncode == 0, result.stdout + result.stderr
    doc = json.loads((tmp_path / "positive_1.json").read_text())
    assert doc["target_digits"] == 6
    assert doc["precision_input"] == 15
    assert 15 < doc["precision_actual"] <= 70
    assert float(doc["verified_digits"]) >= 6
    assert float(doc["fit_digits"]) > 12


def test_cli_strict_q_gate_refuses_an_underresolved_positive_pole_table(tmp_path):
    result = _table_cli(tmp_path, "--central-charge", "1", "--spin-ranges", "0", "--lambda", "1",
                        "--order", "4", "--npoles", "100", "--precision", "70", "--q-gate", "strict")
    assert result.returncode == 2
    assert "q-gate (strict)" in result.stderr
    assert "raise --order" in result.stderr
    assert not list(tmp_path.glob("*.json"))


def test_omitted_npoles_chooses_the_pole_count_from_the_measured_digits():
    # as for the global tables: few remainder poles where they suffice, more where the kept poles limit the table
    from virasoro_blocks.table_adapters import table_adapter_for

    adapter = table_adapter_for("torus-onepoint")

    def build(d_ext, npoles):
        request = adapter.request(central_charge="10", torus_onepoint_ext=d_ext, lambda_order=2, q_order=30,
                                  npoles=npoles, precision=60)
        return adapter.build_table(request, spin=0)

    light = build("1", None)
    assert light.npoles_chosen == 4 and light.verified_digits > 50
    heavy, heavy_four = build("6", None), build("6", 4)
    assert heavy_four.npoles_chosen is None
    assert heavy.npoles_chosen > 4 and heavy.verified_digits > heavy_four.verified_digits + 5
