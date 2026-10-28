"""Singular-charge blocks on and near Kac weights, and exact pole orders.

At a Kac weight, the default evaluator uses the irreducible-module block.
``kac_limit="c_limit"`` selects the fixed-weight central-charge limit,
which can differ from the irreducible block. Away from Kac weights,
evaluation uses the continuous merged block and retains supplied offsets.
Table pole multiplicities come from the recursion's exact orders.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.point import _point_grid
from virasoro_blocks.recursion import F_block, block_prefactor
from virasoro_blocks.regularization import F_block_regularized

SIGMA = [mp.mpf(1) / 16] * 4


@pytest.mark.parametrize("h", ["0.2500001", "0.2501", "0.25000000000000000000000000001", "0.7"])
def test_c1_twist_block_is_H_equals_one_off_the_kac_weights(h):
    """For c=1 and h_ext=1/16, H(q)=1 off Kac weights, including nearby points."""
    mp.mp.dps = 40
    h = mp.mpf(h)
    for z in (mp.mpf("0.5"), mp.mpf("0.99")):
        value = F_block(h, z, SIGMA, c=1, q_order=30)
        assert abs(value / block_prefactor(h, z, SIGMA, c=1) - 1) < mp.mpf("1e-34")


@pytest.mark.parametrize("h", ["0.25", "0", "1"])
def test_c1_twist_block_on_a_kac_weight_is_the_irreducible_block(h):
    """On a c = 1 Kac weight the block is the irreducible-module block (H_q_series_kac_irreducible, the default
    kac_limit): it keeps the null-submodule piece H = 1 (the Fock-module block) lacks.  kac_limit="c_limit" is the
    c -> 1 limit at fixed h (the t^0 coefficients of the Laurent recursion), equal to it at h = 0 and, for these
    twist externals, through level 6 at h = 1/4 (3.3e-8 apart at level 7)."""
    from fractions import Fraction

    from virasoro_blocks.recursion import H_q_series_kac_irreducible, H_q_series_laurent, q_from_z

    mp.mp.dps = 40
    exact = Fraction(h)
    h = mp.mpf(h)
    irr = H_q_series_kac_irreducible(exact, [Fraction(1, 16)] * 4, c0=1, q_order=30)
    lim = H_q_series_laurent(h, SIGMA, c0=mp.mpf(1), q_order=30)
    for z in (mp.mpf("0.5"), mp.mpf("0.99")):
        q = q_from_z(z)
        for H, kw in ((irr, {}), (lim, {"kac_limit": "c_limit"})):
            expected = block_prefactor(h, z, SIGMA, c=1) * sum(Hk * q ** k for k, Hk in enumerate(H))
            value = F_block(h, z, SIGMA, c=1, q_order=30, **kw)
            assert abs(value / expected - 1) < mp.mpf("1e-34")
            assert abs(value / block_prefactor(h, z, SIGMA, c=1) - 1) > mp.mpf("1e-20")
    differs = max(abs(a - b) for a, b in zip(irr, lim))
    assert (differs < mp.mpf("1e-30")) if exact == 0 else (differs > mp.mpf("1e-12"))


def test_kac_weight_with_a_surviving_residue_raises():
    """Ising sigma x sigma has no sigma: h = 1/16 is a genuine pole of the c = 1/2 block, not a finite limit."""
    mp.mp.dps = 30
    with pytest.raises(ValueError, match="genuine pole"):
        F_block_regularized(mp.mpf(1) / 16, mp.mpf("0.5"), SIGMA, c=mp.mpf(1) / 2, q_order=20)


def _grid(c, delta):
    grid, _m, _n = _point_grid(
        "zzb", basis="scalar_G", ell=2, point_value=mp.mpf(delta), h_externals=[mp.mpf("0.125")] * 4,
        hbar_externals=None, lam=3, c=mp.mpf(c), q_order=30, chiral_norm=None, z_shift=None, zbar_shift=None,
        center=None, center_bar=None, parity=None, parity_external=None, exchanged_parity="1",
    )
    return grid


@pytest.mark.parametrize("c", [1, -2])
def test_point_grid_on_a_kac_weight_is_the_fixed_weight_c_limit(c):
    """Delta = 2, spin 2 (hbar = 0, a Kac weight): the point-table on it is the c -> c0 limit at fixed Delta (the
    bare block just off c0), 4e-6 (c = 1) and 2e-13 (c = -2) away from the limit in Delta the tables carry; a weight
    near it (2 + 1e-30) retains its supplied nonzero offset and is the continuous
    block, as are the points 2 +- 1e-15."""
    from virasoro_blocks.derivs import G_zzb_table

    mp.mp.dps = 40
    on = _grid(c, 2)
    inside = _grid(c, mp.mpf(2) + mp.mpf("1e-30"))
    plus, minus = _grid(c, mp.mpf(2) + mp.mpf("1e-15")), _grid(c, mp.mpf(2) - mp.mpf("1e-15"))
    with mp.workdps(140):
        off = G_zzb_table(mp.mpf(2), 2, [mp.mpf("0.125")] * 4, 3, 3, c=mp.mpf(c) + mp.mpf(10) ** -60, q_order=30)
    for (m, n), value in on.items():
        ref = mp.mpc(off[m][n])
        assert abs(value - ref) <= mp.mpf("1e-28") * abs(ref)
        assert abs(inside[(m, n)] - plus[(m, n)]) <= mp.mpf("1e-12") * abs(ref)
        assert abs(plus[(m, n)] - minus[(m, n)]) <= mp.mpf("1e-12") * abs(ref)
    assert abs(inside[(0, 0)] - on[(0, 0)]) > mp.mpf("1e-16") * abs(on[(0, 0)])


def test_laurent_block_at_c_minus_two_reads_a_wide_enough_window():
    """Fixed-weight Laurent coefficients agree with the merged series at c=-2."""
    from virasoro_blocks.recursion import H_q_series_laurent
    from virasoro_blocks.regularization import H_q_series_regularized

    mp.mp.dps = 80
    h = mp.mpf("0.4006898")
    laurent = H_q_series_laurent(h, SIGMA, c0=mp.mpf(-2), q_order=30)
    merged = H_q_series_regularized(h, SIGMA, c=-2, q_order=30)
    assert max(abs(a - b) / max(abs(a), 1) for a, b in zip(laurent, merged)) < mp.mpf("1e-70")
