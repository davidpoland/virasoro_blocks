"""Tests for the single-knob precision controls."""

from __future__ import annotations

import mpmath as mp
import pytest
from fractions import Fraction

from virasoro_blocks.composer import G_block
from virasoro_blocks.interpolation import (
    DampedRational,
    build_delta_interpolant,
    get_sample_points,
    scalar_block_delta_prefactor,
    scalar_block_delta_value,
)
from virasoro_blocks.precision import iteration_cap, set_precision, tolerance


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def test_set_precision_updates_mp_dps():
    saved = mp.mp.dps
    try:
        set_precision(73)
        assert mp.mp.dps == 73
    finally:
        mp.mp.dps = saved


def test_set_precision_rejects_invalid_values():
    with pytest.raises(ValueError):
        set_precision(2)
    with pytest.raises(ValueError):
        set_precision(15.5)  # not an integer


def test_tolerance_scales_with_dps():
    saved = mp.mp.dps
    try:
        mp.mp.dps = 30
        tol_30 = tolerance(guard=10)
        mp.mp.dps = 60
        tol_60 = tolerance(guard=10)
        # 30 more digits of precision -> tolerance shrinks by 10^30.
        assert _close(tol_60 * mp.mpf(10) ** 30, tol_30, tol=mp.mpf("1e-50"))
        # And the magnitude is what the formula says.
        mp.mp.dps = 50
        assert _close(tolerance(guard=10), mp.mpf("1e-40"), tol=mp.mpf("1e-50"))
        assert _close(tolerance(guard=5), mp.mpf("1e-45"), tol=mp.mpf("1e-55"))
    finally:
        mp.mp.dps = saved


def test_tolerance_rejects_negative_guard():
    with pytest.raises(ValueError):
        tolerance(guard=-1)


def test_low_precision_preserves_material_complex_arithmetic():
    from flint import ctx
    from virasoro_blocks.derivs import _assemble_block_grid
    from virasoro_blocks.params import arithmetic_mode
    from virasoro_blocks.polyderivs import _is_negative_one_phase
    from virasoro_blocks.precision import taylor_coefficients

    saved_bits = ctx.prec
    try:
        for dps in (3, 5, 7, 10, 40):
            with mp.workdps(dps):
                value = mp.mpc(1, '.01')
                full = _assemble_block_grid([value], [mp.mpf(1)], 0, 0, sigma=0)
                selected = _assemble_block_grid([value], [mp.mpf(1)], 0, 0,
                                                sigma=0, entries=[(0, 0)])
                assert full[0][0] == selected[0][0] == value
                assert taylor_coefficients(lambda x: value*x, 0, 1)[1] == value
                externals = [mp.mpf('.125')]*4
                assert arithmetic_mode(mp.mpc(30, '.01'), externals) == 'generic_complex'
                assert arithmetic_mode(30, [value, *externals[1:]]) == 'generic_complex'
                for phase in (1, mp.j, -mp.j):
                    assert not _is_negative_one_phase(phase)
                assert _is_negative_one_phase(-1)
                # Numerical real output still permits negligible residue.
                noise = mp.mpf(10)**-max(7, dps)
                rounded = _assemble_block_grid([mp.mpc(1, noise)], [mp.mpf(1)],
                                               0, 0, sigma=0)[0][0]
                assert rounded == 1
    finally:
        ctx.prec = saved_bits


def test_iteration_cap_scales_with_dps():
    saved = mp.mp.dps
    try:
        mp.mp.dps = 30
        cap_30 = iteration_cap(base=30, per_dps=4)
        assert cap_30 == 30 + 4 * 30
        mp.mp.dps = 100
        cap_100 = iteration_cap(base=30, per_dps=4)
        assert cap_100 == 30 + 4 * 100
        # Default args used by find_bohr_sommerfeld_roots: should still be
        # plenty for dps = 100 bisection (need ~3.32 * 100 = 332 iterations).
        assert iteration_cap() >= 332
    finally:
        mp.mp.dps = saved


def test_precision_helpers_share_discrete_control_validation():
    with mp.workdps(30):
        for invalid in (True, False, float('inf'), float('nan'), Fraction(10) + Fraction(1, 10**70)):
            for call in (set_precision, lambda v: tolerance(guard=v),
                         lambda v: iteration_cap(base=v), lambda v: iteration_cap(per_dps=v)):
                with pytest.raises(ValueError):
                    call(invalid)
                assert mp.mp.dps == 30
        for integral in (10.0, Fraction(10), mp.mpf(10)):
            set_precision(integral)
            assert mp.mp.dps == 10
            assert tolerance(guard=integral) == 1
            assert iteration_cap(base=integral, per_dps=integral) == 110


def test_damped_rational_conjugate_check_tightens_with_dps():
    """At higher dps, the conjugate-pair tolerance shrinks, so a pair that
    would be accepted at low dps gets rejected at high dps."""
    saved = mp.mp.dps
    try:
        mp.mp.dps = 20
        p = mp.mpc("-0.5", "0.3")
        partner_with_noise = mp.mpc("-0.5", "-0.3") + mp.mpc(0, "1e-12")
        # At dps=20, guard=10 -> tol = 1e-10, and the 1e-12 noise is below it.
        DampedRational(
            constant=1,
            poles=(p, partner_with_noise),
            base=mp.e ** (-mp.pi),
        )
        mp.mp.dps = 60
        # At dps=60, guard=10 -> tol = 1e-50, and 1e-12 noise is above it.
        with pytest.raises(ValueError):
            DampedRational(
                constant=1,
                poles=(p, partner_with_noise),
                base=mp.e ** (-mp.pi),
            )
    finally:
        mp.mp.dps = saved


def test_interpolation_unitary_low_c_works_at_higher_dps():
    """The interpolation cross-check at c=10 holds at higher precision.
    Tolerance scales with dps, so the agreement bound tightens accordingly."""
    saved = mp.mp.dps
    try:
        set_precision(70)
        ell = 2
        c = mp.mpf("10")
        h_ext = [mp.mpf("0.15")] * 4
        q_order = 3

        z = mp.mpf(1) / 2 + mp.mpf("0.03")
        zbar = mp.mpf(1) / 2 + mp.mpf("0.02")

        raw_prefactor = scalar_block_delta_prefactor(
            ell, z, zbar, h_ext, c=c, q_order=q_order
        )
        shift = raw_prefactor.rightmost_real_pole + mp.mpf("0.5")
        prefactor = scalar_block_delta_prefactor(
            ell, z, zbar, h_ext, shift=shift, c=c, q_order=q_order
        )
        evaluator = lambda Delta: scalar_block_delta_value(
            Delta, ell, z, zbar, h_ext, c=c, q_order=q_order
        )
        sample_xs = get_sample_points(prefactor, prefactor.pole_degree + 1)
        sample_deltas = [x + shift for x in sample_xs]
        interpolant = build_delta_interpolant(
            sample_deltas, evaluator, prefactor, shift=shift
        )

        test_delta = (sample_deltas[len(sample_deltas) // 2]
                      + sample_deltas[len(sample_deltas) // 2 + 1]) / 2
        truth = G_block(test_delta, ell, z, zbar, h_ext, c=c, q_order=q_order)
        approx = interpolant.evaluate(test_delta)
        # At dps=70 we get ~20 more digits of agreement than at dps=50 (1e-20
        # in the original test).
        assert _close(approx, truth, tol=mp.mpf("1e-40"))
    finally:
        mp.mp.dps = saved


@pytest.mark.parametrize("initial,expected", [(4300, 1_000_000), (0, 0), (2_000_000, 2_000_000)])
def test_precision_import_does_not_reduce_integer_string_limit(initial, expected):
    import subprocess
    import sys

    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("Python has no integer-string conversion limit")
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            f"sys.set_int_max_str_digits({initial}); "
            "import virasoro_blocks.precision; "
            "print(sys.get_int_max_str_digits())"
        )], capture_output=True, text=True, check=True,
    )
    assert int(result.stdout.strip()) == expected
