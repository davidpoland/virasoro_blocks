"""Pointwise / off-diagonal global-block evaluation via global_evaluate_block."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.coordinates import q_of_z, z_of_q
from virasoro_blocks.derivs import _global_chiral_block, global_chiral_q_taylor
from virasoro_blocks.global_blocks import global_evaluate_block

H_EXT = [mp.mpf("0.1")] * 4
HBAR_EXT = [mp.mpf("0.12"), mp.mpf("0.09"), mp.mpf("0.11"), mp.mpf("0.08")]
H = mp.mpf("0.9")
HBAR = mp.mpf("0.6")


def test_symmetric_value_matches_ordered_chiral_product():
    mp.mp.dps = 40
    half = mp.mpf(1) / 2
    val = global_evaluate_block(half, half, H, HBAR, H_EXT, coord="qqb")
    ch = global_chiral_q_taylor(H, H_EXT, 1)[0]
    chb = global_chiral_q_taylor(HBAR, H_EXT, 1)[0]
    assert abs(val - ch * chb) < mp.mpf(10) ** (-35)


@pytest.mark.parametrize("coord", ["zzb", "yyb", "qqb"])
def test_offdiagonal_value_is_coordinate_independent(coord):
    mp.mp.dps = 40
    z, zbar = mp.mpf("0.37"), mp.mpf("0.62")
    ref = global_evaluate_block(z, zbar, H, HBAR, H_EXT, coord="qqb")
    got = global_evaluate_block(z, zbar, H, HBAR, H_EXT, coord=coord)
    assert abs(got - ref) < mp.mpf(10) ** (-35)


def test_spinning_externals_match_manual_ordered_product():
    mp.mp.dps = 40
    z, zbar = mp.mpf("0.39"), mp.mpf("0.58")
    got = global_evaluate_block(
        z,
        zbar,
        H,
        HBAR,
        H_EXT,
        hbar_externals=HBAR_EXT,
        coord="qqb",
        z_shift=mp.mpf("0.03"),
        zbar_shift=mp.mpf("-0.02"),
    )
    left = _global_chiral_block(H, z, H_EXT, z_shift=mp.mpf("0.03"))
    right = _global_chiral_block(HBAR, zbar, HBAR_EXT, z_shift=mp.mpf("-0.02"))
    assert abs(got - left * right) < mp.mpf(10) ** (-32)


def test_zzb_grid_is_z_derivatives():
    mp.mp.dps = 40
    z, zbar = mp.mpf("0.37"), mp.mpf("0.62")
    grid = global_evaluate_block(z, zbar, H, HBAR, H_EXT, coord="zzb", m_max=2, n_max=1)
    f = lambda zz: global_evaluate_block(zz, zbar, H, HBAR, H_EXT, coord="zzb")
    assert abs(grid[0][0] - f(z)) < mp.mpf(10) ** (-35)
    assert abs(grid[1][0] - mp.diff(f, z)) < mp.mpf(10) ** (-30)
    assert abs(grid[2][0] - mp.diff(f, z, 2) / 2) < mp.mpf(10) ** (-30)


def test_qqb_grid_is_nome_derivatives():
    mp.mp.dps = 40
    z, zbar = mp.mpf("0.37"), mp.mpf("0.62")
    grid = global_evaluate_block(z, zbar, H, HBAR, H_EXT, coord="qqb", m_max=1, n_max=0)
    fq = lambda qq: global_evaluate_block(z_of_q(qq), zbar, H, HBAR, H_EXT, coord="qqb")
    assert abs(grid[1][0] - mp.diff(fq, q_of_z(z))) < mp.mpf(10) ** (-30)


def test_complex_lorentzian_z_neq_zbar_factorization():
    mp.mp.dps = 40
    z = mp.mpc("0.3", "0.15")
    zbar = mp.mpc("0.55", "-0.2")
    val = global_evaluate_block(z, zbar, H, HBAR, H_EXT, coord="qqb")
    hh = global_evaluate_block(z, z, H, H, H_EXT, coord="qqb")
    bb = global_evaluate_block(zbar, zbar, HBAR, HBAR, H_EXT, coord="qqb")
    assert abs(val * val - hh * bb) < mp.mpf(10) ** (-30)


def test_folded_coord_rejected():
    with pytest.raises(ValueError):
        global_evaluate_block(0.4, 0.4, H, HBAR, H_EXT, coord="qqb_ws")
