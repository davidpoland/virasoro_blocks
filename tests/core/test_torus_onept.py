"""Tests for the torus one-point block kernel and tables.

The decisive check is the Gram-matrix oracle (`tests.oracles.torus_trace`):
the recursion must reproduce the direct inverse-Gram Verma trace, which shares
no formulas with the kernel beyond the Virasoro algebra itself. The residue
convention it pins down: R_{rs} = A_rs * P[d; d_rs] * P[d; d_rs + rs]
(CCY 1703.09805 eq. 4.27 -- two distinct null-state fusion factors, not a
square; getting this wrong reproduces neither the oracle nor the level-1
closed form f_1 = d(d-1)/(2h)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.oracles.torus_trace import (  # noqa: E402
    torus_onept_f_coeffs_oracle,
)
from virasoro_blocks.modular import TORUS_COORDINATE, tau2_power_table
from virasoro_blocks.torus_onept import (
    build_torus_onept_kernel,
    torus_onept_delta_table,
    torus_onept_f_coeffs,
    torus_onept_F,
    torus_onept_table_values,
)

_I = mp.mpc(0, 1)


@pytest.mark.parametrize(
    "c_str, h_str, d_str",
    [
        ("32.7", "1.37", "0.83"),  # real b (c > 25)
        ("9.4", "2.11", "1.3"),    # complex Kac data (1 < c < 25)
        ("0.73", "1.37", "0.83"),  # positive real Kac poles (c < 1)
    ],
)
def test_kernel_matches_gram_oracle(c_str, h_str, d_str):
    with mp.workdps(45):
        c, h, d = mp.mpf(c_str), mp.mpf(h_str), mp.mpf(d_str)
        n_max = 4
        oracle = torus_onept_f_coeffs_oracle(h, c, d, n_max)
        kernel = build_torus_onept_kernel(n_max, d_ext=d, c=c)
        recur = torus_onept_f_coeffs(kernel, h)
        for a, b in zip(oracle, recur):
            denom = max(abs(mp.mpc(a)), mp.mpf("1e-30"))
            assert abs(mp.mpc(a) - mp.mpc(b)) / denom < mp.mpf("1e-35")


def test_level_one_closed_form():
    """f_1 = d(d-1)/(2h), independent of c."""
    with mp.workdps(40):
        for c_str in ("32.7", "5.5", "80"):
            c = mp.mpf(c_str)
            h, d = mp.mpf("0.71"), mp.mpf("1.9")
            kernel = build_torus_onept_kernel(1, d_ext=d, c=c)
            f = torus_onept_f_coeffs(kernel, h)
            expected = d * (d - 1) / (2 * h)
            assert abs(mp.mpc(f[1]) - expected) < mp.mpf("1e-30")


def test_explicit_b_keeps_stored_offset_at_lower_kernel_precision():
    from virasoro_blocks.params import c_from_b, c_origin_of_b

    with mp.workdps(150):
        real_b = mp.sqrt(mp.mpf(2)/3) + mp.mpf('1e-70')
        complex_b = mp.mpc(real_b, 0)
    for b in (real_b, complex_b):
        assert c_origin_of_b(b) is None
        with mp.workdps(220):
            # The oracle uses the mathematical charge of the stored b;
            # there is no source c to recover or singular locus to snap to.
            charge = c_from_b(b).real
            expected = torus_onept_f_coeffs_oracle(mp.mpf('1.37'), charge, mp.mpf('.7'), 4)
        with mp.workdps(30):
            kernel = build_torus_onept_kernel(4, d_ext='.7', b=b)
            got = torus_onept_f_coeffs(kernel, '1.37')
            assert all(abs(a/e-1) < mp.mpf('1e-28') for a, e in zip(got, expected))


def test_onepoint_direct_controls_are_validated_before_numerical_construction(monkeypatch):
    import virasoro_blocks.torus_onept as onept

    kernel = build_torus_onept_kernel(0, d_ext='.7', c=4)
    def unexpected(*args, **kwargs):
        pytest.fail('invalid controls reached numerical construction')
    monkeypatch.setattr(onept, '_rs_pairs', unexpected)
    monkeypatch.setattr(onept, '_torus_laurent_levels', unexpected)
    monkeypatch.setattr(onept, '_chiral_v_coeffs_flint', unexpected)
    monkeypatch.setattr(onept, '_onept_point', unexpected)
    calls = (
        ('q_order', lambda: build_torus_onept_kernel(.5, d_ext='.7', c=4)),
        ('q_order', lambda: onept.build_torus_onept_laurent_state(.5, d_ext='.7', c0=26)),
        ('work_dps', lambda: onept.build_torus_onept_state(0, d_ext='.7', c=4, work_dps=mp.mp.dps+.5)),
        ('n_points', lambda: onept.build_torus_onept_contour_kernels(0, d_ext='.7', c0=26, n_points=4.5)),
        ('lam', lambda: onept.torus_onept_chiral_v_coeffs(kernel, 1, -.5)),
        ('spin', lambda: torus_onept_table_values(kernel, 1.5, 0, 3)),
        ('q_order', lambda: torus_onept_table_values(kernel, 0, 0, 3, q_order=.5)),
        ('npoles', lambda: torus_onept_delta_table(4, 0, 0, d_ext='.7', q_order=0, npoles=-.5)),
        ('precision', lambda: onept.torus_onept_point_value(4, '.7', 1, 1, 0, 0, 0, q_order=0, precision=30.5)),
    )
    for name, call in calls:
        with pytest.raises(ValueError, match=name):
            call()


def test_onepoint_table_keeps_exact_anchor_and_rejects_different_kernel_inputs():
    import json
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_components
    from virasoro_blocks.serialization import serialized_table_from_json
    from virasoro_blocks.torus_onept import build_torus_onept_state
    from virasoro_blocks.writers import write_torus_json

    tiny = Fraction(1, 10**70)
    with mp.workdps(30):
        kernel = build_torus_onept_state(1, d_ext='.7', c=4)
        with pytest.raises(ValueError, match='unitarity bound'):
            torus_onept_delta_table(4, 1, 0, d_ext='.7', q_order=1, npoles=2, shift=1-tiny, kernel=kernel)
        table = torus_onept_delta_table(4, 1, 0, d_ext='.7', q_order=1, npoles=2, shift=1+tiny, kernel=kernel)
        assert literal_components(table.shift) == (1+tiny, 0)
        assert any(abs(p/mp.convert(-tiny)-1) < mp.mpf('1e-28') for p in table.pole_list_x)
        doc = json.loads(write_torus_json(table, c=4, spin=1, basis='torus_onept_G', onept_d_ext='.7', q_order=1))
        assert literal_components(serialized_table_from_json(doc).shift) == (1+tiny, 0)
        for c, d in ((4+tiny, '.7'), (4, Fraction(7, 10)+tiny)):
            with pytest.raises(ValueError, match='supplied kernel'):
                torus_onept_delta_table(c, 1, 0, d_ext=d, q_order=1, npoles=2, kernel=kernel)


def test_identity_external_reduces_to_character():
    with mp.workdps(40):
        kernel = build_torus_onept_kernel(6, d_ext=mp.mpf(0), c=mp.mpf("14.3"))
        f = torus_onept_f_coeffs(kernel, mp.mpf("1.9"))
        assert abs(mp.mpc(f[0]) - 1) == 0
        for coeff in f[1:]:
            assert abs(mp.mpc(coeff)) < mp.mpf("1e-30")


def test_large_c_limit_is_global_2F1():
    """CCY eq. 4.28: f -> 2F1(d, 1-d; 2h; q/(q-1)) as c -> infinity."""
    with mp.workdps(40):
        c, h, d = mp.mpf("1e18"), mp.mpf("1.37"), mp.mpf("0.83")
        kernel = build_torus_onept_kernel(5, d_ext=d, c=c)
        f = torus_onept_f_coeffs(kernel, h)
        ref = mp.taylor(lambda q: mp.hyp2f1(d, 1 - d, 2 * h, q / (q - 1)), 0, 5)
        for a, b in zip(f, ref):
            denom = max(abs(mp.mpc(b)), mp.mpf("1e-30"))
            # 1/c corrections at c = 1e18 leave ~1e-16 relative residue
            assert abs(mp.mpc(a) - mp.mpc(b)) / denom < mp.mpf("1e-12")


def test_singular_c_guard():
    with mp.workdps(40):
        with pytest.raises(ValueError, match="singular locus"):
            build_torus_onept_kernel(4, d_ext=mp.mpf("0.8"), c=mp.mpf(26))


def test_reduced_table_matches_direct_product():
    """Numeric (m, n) table == tau_2^{d+1/2} q^hhat f * (antichiral) directly."""
    with mp.workdps(50):
        c, d, j = mp.mpf("32.7"), mp.mpf("0.83"), 2
        Delta, lam, q_order = mp.mpf("3.41"), 10, 20
        kernel = build_torus_onept_kernel(q_order, d_ext=d, c=c)
        T = torus_onept_table_values(kernel, j, lam, Delta, symmetrize=False)

        def direct(v, vbar):
            tau = _I * (1 + v) / (1 - v)
            taubar = -_I * (1 + vbar) / (1 - vbar)
            tau2 = (tau - taubar) / (2 * _I)
            q = mp.exp(2 * _I * mp.pi * tau)
            qbar = mp.exp(-2 * _I * mp.pi * taubar)
            h = (Delta + j) / 2
            hbar = (Delta - j) / 2
            hhat = h - (c - 1) / 24
            hbarhat = hbar - (c - 1) / 24

            def eta_factor(qv):
                prod = mp.mpf(1)
                for n in range(1, q_order + 1):
                    prod *= 1 - qv**n
                return prod

            return (
                tau2 ** (d + mp.mpf(1) / 2)
                * q**hhat * torus_onept_F(kernel, h, q) * eta_factor(q)
                * qbar**hbarhat
                * torus_onept_F(kernel, hbar, qbar) * eta_factor(qbar)
            )

        # small |v| so the lam-truncation tail sits far below the value
        v0, vbar0 = mp.mpf("0.015"), mp.mpf("-0.01")
        ref = direct(v0, vbar0)
        got = sum(
            mp.mpc(T[m][n]) * v0**m * vbar0**n
            for m in range(lam + 1)
            for n in range(lam + 1)
        )
        assert abs(got - ref) / abs(ref) < mp.mpf("1e-14")


@pytest.mark.parametrize(
    "c_str, j, shift_str",
    [
        ("32.7", 0, "0.4"),
        ("32.7", 2, None),
        ("9.4", 2, None),
    ],
)
def test_delta_table_reconstruction(c_str, j, shift_str):
    with mp.workdps(50):
        c, d = mp.mpf(c_str), mp.mpf("0.83")
        lam, q_order = 6, 16
        kernel = build_torus_onept_kernel(q_order, d_ext=d, c=c)
        kwargs = {} if shift_str is None else {"shift": mp.mpf(shift_str)}
        table = torus_onept_delta_table(
            c, j, lam, d_ext=d, q_order=q_order, npoles=14, kernel=kernel,
            **kwargs,
        )
        base_delta = mp.mpf(shift_str) if shift_str else mp.mpf(j)
        worst = mp.mpf(0)
        for offset in ("0.17", "1.4", "4.2"):
            Delta = mp.mpf(offset) + base_delta
            direct = torus_onept_table_values(kernel, j, lam, Delta)
            recon = table.evaluate(TORUS_COORDINATE, Delta)
            for m in range(lam + 1):
                for n in range(min(m, lam - m) + 1):
                    denom = max(abs(mp.mpc(direct[m][n])), mp.mpf("1e-40"))
                    worst = max(
                        worst,
                        abs(mp.mpc(recon[m][n]) - mp.mpc(direct[m][n])) / denom,
                    )
        # strip cases are unkept-complex-pole limited; well below usable
        # tolerance and convergent in npoles
        assert worst < mp.mpf("1e-10")


@pytest.mark.parametrize("j", [0, 1])
def test_anchor_pole_at_unitarity_bound_is_kept_in_the_prefactor(j):
    """Retain the hbar=0 pole when the anchor equals spin.

    The lowest node is shifted off the pole, including the double pole at
    spin zero. The table agrees near the pole; an anchor below spin is rejected.
    """
    with mp.workdps(40):
        c, d, lam, q_order = mp.mpf("32.7"), mp.mpf("0.8"), 3, 8
        kernel = build_torus_onept_kernel(q_order, d_ext=d, c=c)
        table = torus_onept_delta_table(c, j, lam, d_ext=d, q_order=q_order, npoles=10, kernel=kernel)
        assert table.shift == j
        assert sum(1 for p in table.pole_list_x if p == 0) == (2 if j == 0 else 1)
        assert 0 < table.sample_points_x[0] < mp.mpf("1e-5")
        for x in ("1e-9", "1e-4", "0.05"):
            direct = torus_onept_table_values(kernel, j, lam, j + mp.mpf(x))
            recon = table.evaluate(TORUS_COORDINATE, j + mp.mpf(x))
            for m in range(lam + 1):
                for n in range(min(m, lam - m) + 1):
                    assert abs(recon[m][n] - direct[m][n]) < mp.mpf("1e-15") * abs(direct[m][n])
        # (the pole is genuine: f_1 = d (d - 1) / (2 h) has a nonzero residue at h = 0)
        assert abs(torus_onept_table_values(kernel, j, lam, j + mp.mpf("1e-9"))[0][0]) > 1e6
        with pytest.raises(ValueError, match="unitarity bound"):
            torus_onept_delta_table(c, j, lam, d_ext=d, q_order=q_order, npoles=10, kernel=kernel,
                                    shift=j - mp.mpf("0.1"))


def test_tau2_power_table_general_exponent():
    with mp.workdps(40):
        a = mp.mpf("2.35")
        lam = 10
        T = tau2_power_table(lam, a)
        v0, vbar0 = mp.mpf("0.008"), mp.mpf("-0.006")
        ref = (1 + v0 / (1 - v0) + vbar0 / (1 - vbar0)) ** a
        got = sum(
            T[i][j] * v0**i * vbar0**j
            for i in range(lam + 1)
            for j in range(lam + 1)
        )
        assert abs(got - ref) / abs(ref) < mp.mpf("1e-18")
        T[0][0] = -1
        assert tau2_power_table(lam, a)[0][0] == 1


def test_torus_coefficient_caches_distinguish_bit_precisions_with_the_same_dps(monkeypatch):
    from fractions import Fraction
    from flint import ctx
    from virasoro_blocks import modular, torus_onept

    caches = []
    for module, names in ((modular, ('_CHIRAL_COEFF_CACHE', '_SQRT_TAU2_CACHE')),
                          (torus_onept, ('_QPOW_CACHE', '_FLINT_STATIC_CACHE'))):
        for name in names:
            cache = {}
            monkeypatch.setattr(module, name, cache)
            caches.append(cache)
    def static(field):
        return [[torus_onept._mp_scalar(matrix[i, j], field)
                 for i in range(matrix.nrows()) for j in range(matrix.ncols())]
                for matrix in torus_onept._flint_static(3, 2, field)]
    calls = (lambda: modular._chiral_coeff_polys(4),
             lambda: modular.tau2_power_table(3, Fraction(1, 3)),
             lambda: torus_onept._qpow_matrix(3, 2),
             lambda: static('arb'), lambda: static('acb'))
    saved_bits = ctx.prec
    try:
        for call in calls:
            for cache in caches:
                cache.clear()
            with mp.workprec(201):
                assert mp.mp.dps == 60
                ctx.prec = torus_onept._flint_prec()
                low = call()
            with mp.workprec(203):
                assert mp.mp.dps == 60
                ctx.prec = torus_onept._flint_prec()
                reused = call()
                for cache in caches:
                    cache.clear()
                assert reused == call()
                assert mp.mp.prec == 203
            with mp.workprec(201):
                assert call() == low
    finally:
        ctx.prec = saved_bits


def test_tau2_exponents_do_not_round_onto_an_integer_or_share_its_cache():
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_components

    tiny = Fraction(1, 10**70)
    with mp.workdps(150):
        stored = mp.mpf(1)+mp.mpf('1e-70')
        complex_stored = mp.mpc(1, mp.mpf('1e-70'))
    with mp.workdps(30):
        assert tau2_power_table(1, 1)[1][1] == 0
        for value in (Fraction(1)+tiny, Fraction(-1)+tiny, tiny, stored, complex_stored):
            real, imag = literal_components(value)
            table = tau2_power_table(2, value)
            # Independent coefficients: a(a-1), a(a+1)/2, and a^2(a-1)/2.
            # Near a=-1 the axis coefficient cancels O(1) terms; near zero
            # the (1,2) coefficient cancels O(a) terms to give O(a^2).
            checks = [
                (table[1][1], real*(real-1)-imag*imag, imag*(2*real-1)),
                (table[0][2], (real*(real+1)-imag*imag)/2, imag*(2*real+1)/2),
                (table[1][2], (real*real*(real-1)-imag*imag*(3*real-1))/2,
                             imag*(3*real*real-2*real-imag*imag)/2),
            ]
            with mp.workdps(180):
                for got, re, im in checks:
                    for actual, expected in ((mp.re(got), re), (mp.im(got), im)):
                        if expected == 0:
                            assert actual == 0
                        else:
                            target = mp.mpf(expected.numerator)/expected.denominator
                            assert abs(actual/target-1) < mp.mpf('1e-28')


def test_onept_writer_round_trip():
    import json

    from virasoro_blocks.serialization import serialized_table_from_json
    from virasoro_blocks.writers import write_torus_json

    with mp.workdps(45):
        c, d, j = mp.mpf("9.4"), mp.mpf("0.83"), 2
        lam, q_order = 5, 12
        kernel = build_torus_onept_kernel(q_order, d_ext=d, c=c)
        table = torus_onept_delta_table(
            c, j, lam, d_ext=d, q_order=q_order, npoles=10, kernel=kernel
        )
        doc = json.loads(
            write_torus_json(
                table, c=c, spin=j, basis="torus_onept_G", onept_d_ext=d,
                q_order=q_order, npoles=10,
            )
        )
        assert doc["basis"] == "torus_onept_G"
        assert doc["order"] == q_order
        # the conjugate-pair-preserving cut may keep fewer than npoles
        assert 0 < len(doc["pole_list_x"]) <= 10
        assert len(doc["pole_list_x"]) == len(table.pole_list_x)
        back = serialized_table_from_json(doc)
        Delta = mp.mpf("4.3")
        a = table.evaluate(TORUS_COORDINATE, Delta)
        b = back.evaluate(TORUS_COORDINATE, Delta)
        for m in range(lam + 1):
            for n in range(min(m, lam - m) + 1):
                denom = max(abs(mp.mpc(a[m][n])), mp.mpf("1e-40"))
                assert abs(mp.mpc(a[m][n]) - mp.mpc(b[m][n])) / denom < mp.mpf(
                    "1e-30"
                )


def test_onept_cli_round_trip(tmp_path):
    import json

    from virasoro_blocks.__main__ import main
    from virasoro_blocks.serialization import serialized_table_from_json

    rc = main(
        [
            "--surface", "torus",
            "--central-charge", "10",
            "--torus-onept-ext", "0.8",
            "--spin-ranges", "0:1:1",
            "--lambda", "4",
            "--order", "10",
            "--npoles", "8",
            "--precision", "40",
            "--torus-scalar-shift", "0.5",
            "-o", str(tmp_path),
        ]
    )
    assert rc == 0
    for spin in (0, 1):
        doc = json.loads((tmp_path / f"derivs_{spin}.json").read_text())
        assert doc["basis"] == "torus_onept_G"
        assert doc["surface"] == "torus"
        with mp.workdps(40):
            table = serialized_table_from_json(doc)
            assert 0 < len(table.pole_list_x) <= 8


def test_onept_cli_spin0_anchor_zero_builds_and_negative_is_refused(tmp_path):
    """Spin 0 at anchor 0 keeps its double Kac pole at Delta = 0 in the prefactor (the first node is lifted off it),
    as the library does; an anchor below 0 is under the unitarity bound and refused."""
    from virasoro_blocks.__main__ import main

    base = ["--surface", "torus", "--central-charge", "10", "--torus-onept-ext", "0.8", "--spin-ranges", "0",
            "--lambda", "2", "--order", "8", "--npoles", "6", "--precision", "30"]
    main(base + ["-o", str(tmp_path / "zero")])
    assert (tmp_path / "zero" / "derivs_0.json").exists()
    with pytest.raises(SystemExit):
        main(base + ["--torus-scalar-shift", "-0.1", "-o", str(tmp_path / "neg")])


def test_contour_regularized_singular_c_matches_oracle():
    """Discrete Cauchy average in the c-plane recovers the block at the
    singular loci where the raw kernel loses everything (c = 26 was rel err
    1e+51 at level 5) or is outright rejected."""
    from virasoro_blocks.torus_onept import (
        build_torus_onept_contour_kernels,
        torus_onept_f_coeffs_singular,
    )

    with mp.workdps(50):
        n_max = 4
        for c0_str, h_str, d_str, tol_str in [
            ("26", "0.61", "2.7", "1e-30"),      # b^2 = 2/3 dual locus
            ("0.5", "0.37", "0.71", "1e-25"),    # Ising central charge
            ("25", "1.9", "0.83", "1e-30"),      # b = 1 boundary
        ]:
            c0, h, d = mp.mpf(c0_str), mp.mpf(h_str), mp.mpf(d_str)
            oracle = torus_onept_f_coeffs_oracle(h, c0, d, n_max)
            contour = build_torus_onept_contour_kernels(
                n_max, d_ext=d, c0=c0
            )
            reg = torus_onept_f_coeffs_singular(contour, h)
            for a, b in zip(oracle, reg):
                denom = max(abs(mp.mpc(a)), mp.mpf("1e-30"))
                assert abs(mp.mpc(a) - mp.mpc(b)) / denom < mp.mpf(tol_str)


def test_quotient_oracle_reduces_to_plain_at_generic_weights():
    from tests.oracles.torus_trace import (
        torus_onept_F_coeffs_oracle,
        torus_onept_F_coeffs_oracle_irreducible,
    )

    with mp.workdps(45):
        h, c, d = mp.mpf("1.37"), mp.mpf("32.7"), mp.mpf("0.83")
        plain = torus_onept_F_coeffs_oracle(h, c, d, 4)
        quotient = torus_onept_F_coeffs_oracle_irreducible(h, c, d, 4)
        for a, b in zip(plain, quotient):
            assert abs(a - b) < mp.mpf("1e-35")


def test_ising_sigma_module_epsilon_trace_is_euler_product():
    """Free-fermion closed form: only the sigma module contributes to
    <epsilon>_tau (C_{1 eps 1} = C_{eps eps eps} = 0), and
    Z <epsilon> = (1/2)|eta|^2 with C_{sigma eps sigma} = 1/2 forces the
    normalized irreducible-module trace to equal Euler's prod(1 - q^n):
    pentagonal-number coefficients. A two-sided check of the quotient-trace
    oracle (raw Virasoro algebra) against free fermions."""
    from tests.oracles.torus_trace import (
        torus_onept_F_coeffs_oracle_irreducible,
    )

    with mp.workdps(45):
        T = torus_onept_F_coeffs_oracle_irreducible(
            mp.mpf(1) / 16, mp.mpf(1) / 2, mp.mpf(1) / 2, 7
        )
        euler = [1, -1, -1, 0, 0, 1, 0, 1]
        for got, expected in zip(T, euler):
            assert abs(got - expected) < mp.mpf("1e-30")


# ---------------------------------------------------------------------------
# Laurent (exact) singular-c kernel, tables, and point blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "c0_str, h_str, d_str",
    [("26", "0.61", "2.7"), ("0.5", "0.37", "0.71"),
     ("25", "1.9", "0.83"), ("1", "1.23", "0.6")],
)
def test_laurent_kernel_matches_oracle(c0_str, h_str, d_str):
    from virasoro_blocks.torus_onept import (
        build_torus_onept_laurent_state, torus_onept_f_coeffs_laurent,
    )

    with mp.workdps(50):
        c0, h, d = mp.mpf(c0_str), mp.mpf(h_str), mp.mpf(d_str)
        n_max = 5
        oracle = torus_onept_f_coeffs_oracle(h, c0, d, n_max)
        state = build_torus_onept_laurent_state(n_max, d_ext=d, c0=c0)
        laur = torus_onept_f_coeffs_laurent(state, h)
        for a, b in zip(oracle, laur):
            denom = max(abs(mp.mpc(a)), mp.mpf("1e-30"))
            assert abs(mp.mpc(a) - mp.mpc(b)) / denom < mp.mpf("1e-40")


def test_laurent_state_keeps_its_precision_when_caller_lowers_it():
    from fractions import Fraction
    from virasoro_blocks.laurent_levels import eval_t0
    from virasoro_blocks.torus_onept import (
        build_torus_onept_laurent_state, torus_onept_f_coeffs_laurent,
    )

    h = Fraction(13, 10)
    with mp.workdps(30):
        state = build_torus_onept_laurent_state(4, d_ext=100, c0=26)
        target = +mp.eps
    with mp.workdps(15):
        found = torus_onept_f_coeffs_laurent(state, h)
        assert mp.mp.dps == 15
    # The actual coefficient balls must meet the state's target even when
    # the caller has lowered its ambient precision since building it.
    with mp.workdps(state.work_dps):
        radii = []
        values, negatives = eval_t0(state.levels, h, sector=0, seed=[1, 0, 0, 0, 0],
                                     couples=lambda _sector, _pole: True, radii=radii)
        assert not negatives
        assert max((radius + mp.eps / 2 * abs(value)) / max(1, abs(value))
                   for value, radius in zip(values, radii)) <= target
    # Real-output conversion also retains the promised digits. The
    # inverse-Gram trace is independent of the Laurent recursion.
    with mp.workdps(100):
        expected = torus_onept_f_coeffs_oracle(mp.mpf(13) / 10, mp.mpf(26), mp.mpf(100), 4)
        assert all(isinstance(value, mp.mpf) for value in found)
        assert all(abs(value - truth) <= mp.mpf('1e-28') * max(1, abs(truth))
                   for value, truth in zip(found, expected))


def test_laurent_beats_contour_conditioning():
    """At low dps the Laurent path holds full precision where the contour
    average loses ~half its digits (the reason the sphere pipeline retired
    its analogous contour scorer)."""
    from virasoro_blocks.torus_onept import (
        build_torus_onept_laurent_state, torus_onept_f_coeffs_laurent,
        build_torus_onept_contour_kernels, torus_onept_f_coeffs_singular,
    )

    with mp.workdps(30):
        c0, h, d, n_max = mp.mpf("1"), mp.mpf("1.23"), mp.mpf("0.6"), 5
        oracle = torus_onept_f_coeffs_oracle(h, c0, d, n_max)
        st = build_torus_onept_laurent_state(n_max, d_ext=d, c0=c0)
        lerr = max(
            abs(mp.mpc(a) - mp.mpc(b)) / max(abs(mp.mpc(a)), mp.mpf("1e-40"))
            for a, b in zip(oracle, torus_onept_f_coeffs_laurent(st, h))
        )
        cont = build_torus_onept_contour_kernels(n_max, d_ext=d, c0=c0)
        cerr = max(
            abs(mp.mpc(a) - mp.mpc(b)) / max(abs(mp.mpc(a)), mp.mpf("1e-40"))
            for a, b in zip(oracle, torus_onept_f_coeffs_singular(cont, h))
        )
        assert lerr < mp.mpf("1e-25")
        assert cerr > lerr * mp.mpf("1e6")


def test_laurent_identity_external_at_singular_c():
    from virasoro_blocks.torus_onept import (
        build_torus_onept_laurent_state, torus_onept_f_coeffs_laurent,
    )

    with mp.workdps(40):
        st = build_torus_onept_laurent_state(6, d_ext=mp.mpf(0), c0=mp.mpf("0.5"))
        f = torus_onept_f_coeffs_laurent(st, mp.mpf("1.9"))
        assert abs(mp.mpc(f[0]) - 1) == 0
        for coeff in f[1:]:
            assert abs(mp.mpc(coeff)) < mp.mpf("1e-30")


@pytest.mark.parametrize("c0_str, j, shift", [("26", 2, None), ("50", 2, None), ("33", 3, None)])
def test_singular_c_dual_family_tables(c0_str, j, shift):
    from virasoro_blocks.torus_onept import (
        build_torus_onept_laurent_state, torus_onept_delta_table,
        torus_onept_table_values,
    )

    with mp.workdps(50):
        c0, d, lam, qo = mp.mpf(c0_str), mp.mpf("0.83"), 4, 16
        state = build_torus_onept_laurent_state(qo, d_ext=d, c0=c0)
        kw = {} if shift is None else {"shift": mp.mpf(shift)}
        # auto-detected singular routing (no kernel passed)
        tab = torus_onept_delta_table(c0, j, lam, d_ext=d, q_order=qo, npoles=14, **kw)
        base = mp.mpf(shift) if shift else mp.mpf(j)
        worst = mp.mpf(0)
        for off in ("0.17", "1.4", "4.2"):
            Delta = mp.mpf(off) + base
            direct = torus_onept_table_values(state, j, lam, Delta)
            recon = tab.evaluate(TORUS_COORDINATE, Delta)
            for m in range(lam + 1):
                for n in range(min(m, lam - m) + 1):
                    den = max(abs(mp.mpc(direct[m][n])), mp.mpf("1e-40"))
                    worst = max(worst, abs(mp.mpc(recon[m][n]) - mp.mpc(direct[m][n])) / den)
        assert worst < mp.mpf("1e-14")


def test_c_below_one_table_splits_positive_poles():
    from virasoro_blocks.torus_onept import build_torus_onept_state

    with mp.workdps(70):
        kernel = build_torus_onept_state(12, c="0.5", d_ext="0.83")
        table = torus_onept_delta_table("0.5", 2, 4, d_ext="0.83", q_order=12, npoles=14, kernel=kernel)
        assert any(mp.re(p) > 0 for p in table.pole_list_x)
        for delta in ("2.17", "3.4", "6.2", "1000"):
            expected = torus_onept_table_values(kernel, 2, 4, mp.mpf(delta))
            got = table.evaluate(TORUS_COORDINATE, mp.mpf(delta))
            for m in range(5):
                for n in range(min(m, 4-m) + 1):
                    assert abs(got[m][n]/expected[m][n]-1) < mp.mpf("1e-35")


def test_phase2b_generator_matches_oracle_and_ising():
    from virasoro_blocks.torus_onept import irreducible_onept_F_series, kac_weight
    from tests.oracles.torus_trace import (
        torus_onept_F_coeffs_oracle_irreducible,
    )

    with mp.workdps(40):
        # (an external the module admits: epsilon on sigma; d = 1/4 does not decouple from sigma's nulls, and there
        # the greedy-subset trace depended on the subset)
        a = irreducible_onept_F_series(mp.mpf(1) / 16, mp.mpf("0.5"), mp.mpf("0.5"), 6)
        b = torus_onept_F_coeffs_oracle_irreducible(
            mp.mpf(1) / 16, mp.mpf("0.5"), mp.mpf("0.5"), 6
        )
        for x, y in zip(a, b):
            assert abs(mp.mpc(x) - mp.mpc(y)) < mp.mpf("1e-35")
        # Ising free-fermion identity: Tr_L(sigma) epsilon = prod(1 - q^n)
        T = irreducible_onept_F_series(mp.mpf(1) / 16, mp.mpf(1) / 2, mp.mpf(1) / 2, 7)
        for got, e in zip(T, [1, -1, -1, 0, 0, 1, 0, 1]):
            assert abs(mp.mpc(got) - e) < mp.mpf("1e-30")
        assert abs(kac_weight(1, 2, 3, 4) - mp.mpf(1) / 16) < mp.mpf("1e-35")


def test_phase2b_degenerate_point_block_vs_direct():
    from virasoro_blocks.torus_onept import (
        torus_onept_degenerate_point_table, kac_weight, irreducible_onept_F_series,
    )

    with mp.workdps(40):
        # Taylor truncation ~ 0.02^(lam+1), q-series truncation ~ (2e-3)^(nmax+1): both far below 1e-14
        r, s, p, q, lam, dext, nmax = 1, 2, 3, 4, 8, mp.mpf("0.5"), 8
        tab = torus_onept_degenerate_point_table(
            r, s, p, q, lam, d_ext=dext, n_max=nmax, spin=0
        )
        c = 1 - 6 * mp.mpf((p - q) ** 2) / (p * q)
        h = kac_weight(r, s, p, q)
        Fq = irreducible_onept_F_series(h, c, dext, nmax)
        I = mp.mpc(0, 1)

        def direct(v, vb):
            tau = I * (1 + v) / (1 - v)
            taub = -I * (1 + vb) / (1 - vb)
            t2 = (tau - taub) / (2 * I)
            qv = mp.exp(2 * I * mp.pi * tau)
            qbv = mp.exp(-2 * I * mp.pi * taub)
            hh = h - (c - 1) / 24

            def eta_prod(qq):
                pr = mp.mpf(1)
                for n in range(1, nmax + 1):
                    pr *= 1 - qq**n
                return pr

            def series(qq):
                return sum(mp.mpc(Fq[n]) * qq**n for n in range(nmax + 1))

            return (
                t2 ** (dext + mp.mpf(1) / 2)
                * qv**hh * eta_prod(qv) * series(qv)
                * qbv**hh * eta_prod(qbv) * series(qbv)
            )

        v0, vb0 = mp.mpf("0.02"), mp.mpf("-0.015")
        ref = direct(v0, vb0)
        got = sum(
            mp.mpc(tab[m][n]) * v0**m * vb0**n
            for m in range(lam + 1)
            for n in range(lam + 1)
        )
        assert abs(got - ref) / abs(ref) < mp.mpf("1e-14")


def test_phase2b_rejects_outside_kac_table():
    from virasoro_blocks.torus_onept import torus_onept_degenerate_point_table

    with mp.workdps(30):
        with pytest.raises(ValueError, match="Kac table"):
            torus_onept_degenerate_point_table(
                3, 2, 3, 4, 4, d_ext=mp.mpf("0.25"), n_max=6
            )
