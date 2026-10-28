"""Off-center (z != 1/2) coverage for the chiral_F block path.

A chiral half F(h, z) has a single expansion point and no zbar, so only the
holomorphic `center` applies (no off-diagonal). Covers the Virasoro symbolic
table, the numeric sampler + interpolation, and the global (SL(2)) chiral path,
each against an independent pointwise oracle.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.polyderivs import (
    F_z_poly_table,
    _NumericChiralSampler,
    interpolate_delta_table,
)
from virasoro_blocks.recursion import F_block
from virasoro_blocks.global_blocks import (
    _GlobalBlockSamplerChiral,
    global_chiral_F_interpolated_table,
)

EXT = [mp.mpf("0.1"), mp.mpf("0.15"), mp.mpf("0.2"), mp.mpf("0.25")]
C = mp.mpf("2.5")
QORDER = 50
Z0 = mp.mpf("0.3")


def test_symbolic_chiral_table_offcenter_matches_F_block():
    """High dps: the off-center symbolic chiral table value matches F_block at z0
    (low dps is the documented P(h)/D(h) ill-conditioning, not off-center).
    Both sides share q_order, so a shorter q-series than QORDER suffices; dps
    120 already leaves the comparison far below the 1e-30 bound."""
    mp.mp.dps = 120
    q_order = 30
    tab = F_z_poly_table(EXT, 3, c=C, q_order=q_order, center=Z0)
    for h in (mp.mpf("0.7"), mp.mpf("1.3")):
        got = tab.evaluate(h)[0][0]
        assert abs(mp.mpc(got) - mp.mpc(F_block(h, Z0, EXT, c=C, q_order=q_order))) < mp.mpf(10) ** -30


def test_numeric_chiral_sampler_offcenter_matches_F_block():
    mp.mp.dps = 45
    s = _NumericChiralSampler("z", EXT, 3, b=None, c=C, q_order=QORDER,
                              chiral_norm=None, z_shift=None, center=Z0)
    tables, _ = s([mp.mpf("0.7"), mp.mpf("1.3")])
    for tab, h in zip(tables, (mp.mpf("0.7"), mp.mpf("1.3"))):
        assert abs(mp.mpc(tab[0][0]) - mp.mpc(F_block(h, Z0, EXT, c=C, q_order=QORDER))) < mp.mpf(10) ** -40


def test_numeric_chiral_sampler_center_none_is_symmetric():
    mp.mp.dps = 45
    s = _NumericChiralSampler("z", EXT, 3, b=None, c=C, q_order=QORDER,
                              chiral_norm=None, z_shift=None)
    assert s.center is None and s._u_center == mp.mpf(1) / 2
    tables, _ = s([mp.mpf("0.7")])
    assert abs(mp.mpc(tables[0][0][0]) - mp.mpc(F_block(mp.mpf("0.7"), mp.mpf("0.5"), EXT, c=C, q_order=QORDER))) < mp.mpf(10) ** -40


def test_chiral_interpolation_offcenter_roundtrip():
    mp.mp.dps = 45
    exact = F_z_poly_table(EXT, 4, c=C, q_order=QORDER, center=Z0)
    exact.numeric_sampler = _NumericChiralSampler(
        "z", EXT, exact.m_max, b=None, c=C, q_order=QORDER,
        chiral_norm=None, z_shift=None, center=Z0,
    )
    interp = interpolate_delta_table(exact, npoles=8)
    h = mp.mpf("1.1")
    assert abs(mp.mpc(interp.evaluate(h)[0][0]) - mp.mpc(F_block(h, Z0, EXT, c=C, q_order=QORDER))) < mp.mpf(10) ** -6


def test_global_chiral_offcenter_matches_2F1():
    mp.mp.dps = 40
    N = 4

    def k(h, z):
        h1, h2, h3, h4 = EXT
        return z ** h * mp.hyp2f1(h + h2 - h1, h + h3 - h4, 2 * h, z) / N ** h

    s = _GlobalBlockSamplerChiral("q", EXT, N, None, 3, center=Z0)
    assert abs(mp.mpc(s(mp.mpf("0.9"))[0][0]) - k(mp.mpf("0.9"), Z0)) < mp.mpf(10) ** -30
    tab = global_chiral_F_interpolated_table(
        "q", EXT, 2, chiral_norm=N, npoles=10, shift=mp.mpf("0.5"), center=Z0,
    )
    for h in (mp.mpf("0.9"), mp.mpf("1.6")):
        assert abs(mp.mpc(tab.evaluate(h)[0][0]) - k(h, Z0)) < mp.mpf(10) ** -6
