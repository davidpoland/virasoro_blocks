"""The flint coordinate geometry (nome and prefactor series) carries the full working precision."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.coordinates import z_of_y


@pytest.mark.parametrize("coordinate", ["z", "y"])
def test_flint_geometry_coefficients_carry_full_working_precision(coordinate):
    """The coordinate composition cancels (y coefficients are O(1) combinations of z coefficients growing like 2^n),
    so the geometry needs guard bits: at 30 terms every nome and prefactor coefficient must agree with a 100-digit
    build to ~100 ulp (it lost 13 digits in y without them). Binary-exact exponents keep the inputs identical."""
    from virasoro_blocks.derivs import _flint_geometry_series_coeffs

    kwargs = (dict(z_of_u=z_of_y, u_symmetric=mp.mpf(0)) if coordinate == "y"
              else dict(z_of_u=lambda u: u, u_symmetric=mp.mpf(1) / 2))
    exponents = (mp.mpf("0.25"), mp.mpf("-0.125"), mp.mpf("0.375"), mp.mpf("1.5"))

    def build(dps):
        with mp.workdps(dps):
            return _flint_geometry_series_coeffs(30, *exponents, **kwargs)

    low, high = build(50), build(100)
    with mp.workdps(100):
        for got, ref in zip(low, high):
            worst = max(abs(g - r) / abs(r) for g, r in zip(got, ref))
            assert worst < mp.mpf("1e-48"), (coordinate, mp.nstr(worst, 3))


def test_zzb_block_at_negative_z_above_c_25_is_not_zero():
    """At real z < 0 the zzb geometry's balls straddle the cut (zero midpoints, radius ~3); they were accepted as an
    exact zero series and the block came out 0. It must agree with the qqb coordinate there."""
    from virasoro_blocks.derivs import evaluate_block

    mp.mp.dps = 30
    args = (mp.mpf("-0.5"), mp.mpf("-0.5"), mp.mpf("1.3"), mp.mpf("1.3"), [mp.mpf("0.3")] * 4)
    zzb = mp.mpc(evaluate_block(*args, c=mp.mpf(30), q_order=12, coord="zzb"))
    qqb = mp.mpc(evaluate_block(*args, c=mp.mpf(30), q_order=12, coord="qqb"))
    assert abs(zzb) > 0 and abs(zzb - qqb) < mp.mpf(10) ** -25 * abs(qqb)


@pytest.mark.parametrize("coord", ["zzb", "qqb"])
def test_block_next_to_z_one_carries_the_working_precision(coord):
    """The theta series is summed until its terms are negligible: a fixed count ignored |q0|, which approaches 1 as
    z -> 1, and a block at z = 0.999999 lost 20 of 40 digits."""
    from virasoro_blocks.derivs import evaluate_block

    def block(dps):
        with mp.workdps(dps):
            z = mp.mpf("0.999999")
            return evaluate_block(z, z, mp.mpf("1.3"), mp.mpf("1.3"), [mp.mpf(1) / 16] * 4, c=mp.mpf(8), q_order=40,
                                  coord=coord, m_max=4, n_max=0)

    low, high = block(40), block(100)
    with mp.workdps(100):
        for m in range(5):
            a, b = mp.mpc(low[m][0]), mp.mpc(high[m][0])
            assert abs(a - b) < mp.mpf(10) ** -33 * abs(b), (coord, m)


def test_batched_real_evaluation_declines_a_complex_geometry():
    """``evaluate.many`` returns real coefficients, so for a geometry that is not real (here real z > 1) it must
    decline instead of dropping an O(1) imaginary part."""
    from virasoro_blocks.derivs import chiral_z_series_evaluator

    mp.mp.dps = 30
    evaluate = chiral_z_series_evaluator([mp.mpf(1) / 16] * 4, 4, c=mp.mpf(30), q_order=12, z_shift=None)
    assert evaluate.many([mp.mpf("1.3")]) is not None           # the real geometry at z = 1/2 batches
    from virasoro_blocks.derivs import _chiral_series_evaluator

    shifted = _chiral_series_evaluator([mp.mpf(1) / 16] * 4, 4, c=mp.mpf(30), q_order=12, z_of_u=lambda u: u,
                                       u_symmetric=mp.mpf(2), coord_key="z@2")
    assert shifted.many([mp.mpf("1.3")]) is None
