"""Pointwise / off-symmetric block evaluation via derivs.evaluate_block."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.coordinates import q_of_z, z_of_q
from virasoro_blocks.derivs import chiral_q_taylor, evaluate_block

EXT = [mp.mpf("0.1")] * 4
C = mp.mpf("2.5")
H = mp.mpf("0.7")
HBAR = mp.mpf("0.4")
QORDER = 20  # every check compares evaluations at this same q_order


def test_symmetric_value_matches_ordered_chiral_product():
    mp.mp.dps = 40
    half = mp.mpf(1) / 2
    val = evaluate_block(half, half, H, HBAR, EXT, c=C, q_order=QORDER, coord="qqb")
    ch = chiral_q_taylor(H, EXT, 1, c=C, q_order=QORDER)[0]
    chb = chiral_q_taylor(HBAR, EXT, 1, c=C, q_order=QORDER)[0]
    assert abs(val - ch * chb) < mp.mpf(10) ** (-35)


@pytest.mark.parametrize("coord", ["zzb", "yyb", "qqb"])
def test_offsymmetric_value_is_coordinate_independent(coord):
    mp.mp.dps = 40
    z, zbar = mp.mpf("0.37"), mp.mpf("0.62")
    ref = evaluate_block(z, zbar, H, HBAR, EXT, c=C, q_order=QORDER, coord="qqb")
    got = evaluate_block(z, zbar, H, HBAR, EXT, c=C, q_order=QORDER, coord=coord)
    assert abs(got - ref) < mp.mpf(10) ** (-35)


def test_zzb_grid_is_z_derivatives():
    mp.mp.dps = 40
    z, zbar = mp.mpf("0.37"), mp.mpf("0.62")
    grid = evaluate_block(
        z, zbar, H, HBAR, EXT, c=C, q_order=QORDER, coord="zzb", m_max=2, n_max=1
    )
    f = lambda zz: evaluate_block(zz, zbar, H, HBAR, EXT, c=C, q_order=QORDER, coord="zzb")
    assert abs(grid[0][0] - f(z)) < mp.mpf(10) ** (-35)
    assert abs(grid[1][0] - mp.diff(f, z)) < mp.mpf(10) ** (-30)
    assert abs(grid[2][0] - mp.diff(f, z, 2) / 2) < mp.mpf(10) ** (-30)


def test_qqb_grid_is_nome_derivatives():
    mp.mp.dps = 40
    z, zbar = mp.mpf("0.37"), mp.mpf("0.62")
    grid = evaluate_block(
        z, zbar, H, HBAR, EXT, c=C, q_order=QORDER, coord="qqb", m_max=1, n_max=0
    )
    fq = lambda qq: evaluate_block(
        z_of_q(qq), zbar, H, HBAR, EXT, c=C, q_order=QORDER, coord="qqb"
    )
    assert abs(grid[1][0] - mp.diff(fq, q_of_z(z))) < mp.mpf(10) ** (-30)


def test_complex_lorentzian_z_neq_zbar():
    mp.mp.dps = 40
    z = mp.mpc("0.3", "0.15")
    zbar = mp.mpc("0.55", "-0.2")
    val = evaluate_block(z, zbar, H, HBAR, EXT, c=C, q_order=QORDER, coord="qqb")
    # Chiral factorization F = f(h,z) * f(hbar,zbar) implies
    #   F(z,zbar;h,hbar)^2 = F(z,z;h,h) * F(zbar,zbar;hbar,hbar),
    # which must hold for independent complex z, zbar (Lorentzian kinematics).
    hh = evaluate_block(z, z, H, H, EXT, c=C, q_order=QORDER, coord="qqb")
    bb = evaluate_block(zbar, zbar, HBAR, HBAR, EXT, c=C, q_order=QORDER, coord="qqb")
    assert abs(val * val - hh * bb) < mp.mpf(10) ** (-30)


def test_folded_coord_rejected():
    with pytest.raises(ValueError):
        evaluate_block(0.4, 0.4, H, HBAR, EXT, c=C, q_order=QORDER, coord="qqb_ws")
