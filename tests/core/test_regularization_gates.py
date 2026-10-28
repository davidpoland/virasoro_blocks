"""Accuracy gates for singular-charge blocks, including tiny values.

Scalar values use the merged-pole H series. The symbolic-table
epsilon-extrapolation gate must also resolve small blocks relative
to their magnitude.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks import regularization as R
from virasoro_blocks.recursion import F_block

SIGMA = [mp.mpf(1) / 16] * 4
Z = mp.mpf("0.9296875")          # exactly representable


def _probe_index(eps, eps0):
    return int(mp.nint(mp.log(abs(eps0 / eps), 2)))


def test_grid_gate_is_relative_to_the_grid_and_keeps_a_zero_entry():
    """A grid of 1e-50 entries with 10^-(dps/3) probe noise, one entry vanishing by symmetry (noise only), one
    exactly zero: every entry comes back to the grid's scale and nothing escalates to the ceiling."""
    c0 = mp.mpf(1) / 2
    with mp.workdps(30):
        truth = mp.mpf(10) ** -50

        def build(c_val):
            eps = c_val - c0
            noise = mp.mpf(10) ** (-mp.mp.dps / 3) * (-1) ** _probe_index(eps, mp.mpf(10) ** -15)

            def sampler(deltas, entries=None):
                return [[[truth * (1 + eps) + noise, 3 * truth * (1 - eps) - noise],
                         [noise, mp.mpf(0)]] for _ in deltas], False
            return sampler

        (grid,) = R.regularized_numeric_grids(build, [mp.mpf(2)], c=c0, n_points=5)
        assert abs(grid[0][0] - truth) < mp.mpf(10) ** -28 * truth
        assert abs(grid[0][1] - 3 * truth) < mp.mpf(10) ** -28 * truth
        assert abs(grid[1][0]) < mp.mpf(10) ** -28 * truth
        assert grid[1][1] == 0


def test_large_finite_charge_probes_keep_their_offsets_and_respect_the_precision_ceiling():
    with mp.workdps(30):
        c0 = mp.mpf('1e400')
        offsets = []

        def build(c_val):
            difference = c_val-c0
            assert mp.im(difference) == 0
            offset = mp.re(difference)
            offsets.append(offset)

            def sampler(deltas, entries=None):
                return [[[3 + offset]] for _ in deltas], False
            return sampler

        (grid,) = R.regularized_numeric_grids(build, [2], c=c0, n_points=5)
        assert abs(grid[0][0]-3) < mp.mpf('1e-28')
        assert len(offsets) == 5 and all(offset > 0 for offset in offsets)
        assert all(left > right for left, right in zip(offsets, offsets[1:]))
        assert mp.mp.dps == 30
        with pytest.raises(ValueError, match='insufficient work_dps'):
            R.regularized_numeric_grids(build, [2], c=c0, n_points=5, max_work_dps=200)
        assert len(offsets) == 5 and mp.mp.dps == 30


def test_kac_cluster_confirmation_survives_large_finite_coordinates():
    with mp.workdps(80):
        for hv in (mp.mpc('1e400', '2e400'), mp.mpc('1.7e308', '1.7e308'),
                   mp.mpc('8e307', '8e307')):
            scale = abs(hv)
            near = hv + scale*mp.mpf('1e-25')
            far = hv + scale*mp.mpf('1e-15')
            limits = [(0, hv), (4, -hv), (1, near), (2, far), (3, hv.conjugate())]
            assert R._cluster_members(iter(limits), hv, 20) == [0, 1]


def test_ising_block_below_one_carries_its_digits():
    """Ising sigma block at h = 300, q_order = 60 (3.2e-239 at z = 0.93): 57 of 60 digits before (q_order = 70:
    33; q_order = 100 at 100 digits: the wrong sign), the full 60 now."""
    with mp.workdps(60):
        ref = mp.mpf("3.221688982801300509080581228031650986031992857888803583256472702025524260250705e-239")
        value = F_block(mp.mpf(300), Z, SIGMA, c=Fraction(1, 2), q_order=60)
        assert abs(value - ref) < mp.mpf(10) ** -59 * ref


def test_ising_kac_weight_limit_carries_its_digits():
    """h = 50 = h_{13,1} at c = 1/2 (its residue vanishes for sigma externals), q_order = 14: 24 of 30 digits
    before (the combine accepted a 1e-24 residual of regularizer noise)."""
    with mp.workdps(60):
        ref = F_block(mp.mpf(50), Z, SIGMA, c=Fraction(1, 2), q_order=14)
    with mp.workdps(30):
        value = F_block(mp.mpf(50), Z, SIGMA, c=Fraction(1, 2), q_order=14)
    assert abs(value - ref) < mp.mpf(10) ** -29 * abs(ref)


def test_ising_kac_weight_regularizer_noise_is_not_a_genuine_pole():
    """The same weight at q_order = 40 raised 'genuine pole' at 30 digits (and at 45 and 60 for z = 1/2)."""
    with mp.workdps(30):
        ref = mp.mpf("1.87427380212760710165827683760305124682272097e-40")
        value = F_block(mp.mpf(50), Z, SIGMA, c=Fraction(1, 2), q_order=40)
        assert abs(value - ref) < mp.mpf(10) ** -29 * ref
