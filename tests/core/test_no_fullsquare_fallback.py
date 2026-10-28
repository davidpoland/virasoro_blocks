"""Requested entries remain restricted across sampler fallback paths.

Cover complex charges and dimensions, Kac poles, folded coordinates,
and the default derivative triangle in symbolic interpolation.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.derivs import G_zzb_table
from virasoro_blocks.polyderivs import (
    G_zzb_poly_table,
    _NumericBlockSampler,
    interpolate_delta_table,
)

EXT = [mp.mpf("0.1")] * 4


@pytest.mark.parametrize("coordinate", ["zzb", "yyb", "qqb", "xt", "ws", "qqb_ws"])
def test_complex_c_sampler_is_entry_restricted(coordinate):
    """Complex c forces the batched-arb path to bail; the per-Delta acb fallback
    must still fill only the requested entries (never the full square)."""
    mp.mp.dps = 40
    c = mp.mpc("2.5", "0.3")
    s = _NumericBlockSampler(
        coordinate, EXT, 2, 3, 3, b=None, c=c, q_order=40,
        chiral_norm=None, z_shift=None,
    )
    entries = [(0, 0), (1, 1), (3, 0)]
    tables, is_arb = s([mp.mpf("3.1")], entries=entries)
    assert is_arb is False  # complex -> acb, not arb
    filled = {(m, n) for m in range(4) for n in range(4)
              if tables[0][m][n] is not None}
    assert filled == set(entries)


def test_complex_c_fallback_values_match_full_square():
    """The entry-restricted complex fallback agrees with the full-square numeric
    reference on the requested entries."""
    mp.mp.dps = 50
    c = mp.mpc("2.5", "0.3")
    s = _NumericBlockSampler(
        "zzb", EXT, 2, 3, 3, b=None, c=c, q_order=40,
        chiral_norm=None, z_shift=None,
    )
    entries = [(0, 0), (2, 1), (3, 3)]
    tables, _ = s([mp.mpf("3.1")], entries=entries)
    ref = G_zzb_table(mp.mpf("3.1"), 2, EXT, 3, 3, c=c, q_order=40)
    for (m, n) in entries:
        assert abs(mp.mpc(tables[0][m][n]) - mp.mpc(ref[m][n])) < mp.mpf(10) ** -45


def test_offdiagonal_complex_c_sampler_is_entry_restricted():
    """Off-diagonal numeric sampling (independent z/zbar centers) keeps the
    same sparse-entry contract in the complex fallback path."""
    mp.mp.dps = 40
    c = mp.mpc("2.5", "0.3")
    s = _NumericBlockSampler(
        "zzb", EXT, 2, 3, 3, b=None, c=c, q_order=40,
        chiral_norm=None, z_shift=None, center=mp.mpf("0.37"), center_bar=mp.mpf("0.55"),
    )
    entries = [(0, 0), (1, 2), (3, 0)]
    tables, is_arb = s([mp.mpf("3.1")], entries=entries)
    assert is_arb is False
    filled = {(m, n) for m in range(4) for n in range(4) if tables[0][m][n] is not None}
    assert filled == set(entries)


def test_triangle_table_interpolation_skips_corner():
    """Interpolating an order-Lambda triangle table (the default build) without an
    explicit kept set must not sample the discarded m+n>Lambda corner."""
    mp.mp.dps = 50
    from virasoro_blocks.params import b_from_c
    b = b_from_c(mp.mpf("10"))
    tri = G_zzb_poly_table(2, EXT, 2, 2, b=b, q_order=8)          # default = triangle
    interp = interpolate_delta_table(tri, npoles=6)
    # The corner (2,2), m+n=4 > Lambda=2, is a zero placeholder and must not be
    # interpolated (no numeric sample taken there); it reconstructs to zero.
    val = interp.evaluate(mp.mpf("3.3"))
    assert abs(mp.mpc(val[2][2])) == 0
    # A kept entry still reconstructs the block.
    full = G_zzb_poly_table(2, EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    assert abs(mp.mpc(val[1][1]) - mp.mpc(full.evaluate(mp.mpf("3.3"))[1][1])) < mp.mpf(10) ** -8
