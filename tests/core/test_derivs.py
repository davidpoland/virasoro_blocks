"""Validate the (z, zbar)-derivative tables for B and G at z = zbar = 1/2."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.composer import B_block, G_block
from virasoro_blocks.coordinates import (
    q_symmetric,
    y_of_z,
    y_symmetric,
    z_of_q,
    z_of_y,
)
from virasoro_blocks.derivs import (
    B_qqb_table,
    B_yyb_table,
    B_zzb_table,
    G_qqb_table,
    G_qqb_table_many,
    G_qqb_ws_radial_table,
    G_qqb_ws_table,
    G_qqb_ws_table_many,
    G_ws_radial_table,
    G_ws_table,
    G_ws_table_many,
    G_xt_radial_table,
    G_xt_table,
    G_xt_table_many,
    G_yyb_table,
    G_yyb_table_many,
    G_zzb_table,
    G_zzb_table_many,
    chiral_q_taylor_fast,
    chiral_q_taylor,
    chiral_y_taylor,
    chiral_y_taylor_fast,
    chiral_z_taylor,
    chiral_z_taylor_fast,
    global_G_zzb_table,
    global_chiral_q_taylor,
    global_chiral_y_taylor,
    global_chiral_z_taylor,
)
import virasoro_blocks.derivs as _derivs_mod
from virasoro_blocks.params import b_from_c
from virasoro_blocks.recursion import DEFAULT_CHIRAL_NORM, F_block


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def test_chiral_z_taylor_zeroth_term_equals_F_at_half():
    """The 0-th Taylor coefficient at z = 1/2 is F(h, 1/2)."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    series = chiral_z_taylor(h, d, 4, b=b, q_order=10)
    F_at_half = F_block(h, mp.mpf(1) / 2, d, b=b, q_order=10)
    assert _close(series[0], F_at_half, tol=mp.mpf("1e-25"))


def test_chiral_z_taylor_reconstructs_F_near_half():
    """Truncated Taylor reproduces F(h, z) near z = 1/2."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    n_terms = 8
    series = chiral_z_taylor(h, d, n_terms, b=b, q_order=14)
    z = mp.mpf(1) / 2 + mp.mpf("0.03")
    delta = z - mp.mpf(1) / 2
    approx = sum(series[k] * delta**k for k in range(n_terms))
    exact = F_block(h, z, d, b=b, q_order=14)
    # Convergence radius is 1/2 (branch point at z = 1); ratio = 0.06,
    # truncation error ~ 0.06^8 ~ 3e-10.
    assert _close(approx, exact, tol=mp.mpf("1e-8"))


def test_chiral_z_taylor_threads_z_shift_to_f_block():
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    z_shift = mp.mpf("-0.5")
    n_terms = 5

    series = chiral_z_taylor(h, d, n_terms, b=b, q_order=12, z_shift=z_shift)
    direct = mp.taylor(
        lambda z: F_block(h, z, d, b=b, q_order=12, z_shift=z_shift),
        mp.mpf(1) / 2,
        n_terms - 1,
    )

    for got, want in zip(series, direct):
        assert _close(got, want, tol=mp.mpf("1e-25"))


def test_B_zzb_table_zeroth_entry_equals_B_at_half():
    """The (0, 0) entry equals B at the symmetric point."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h = mp.mpf("1.4")
    hbar = mp.mpf("1.1")

    table = B_zzb_table(h, hbar, h_ext, hbar_ext, 3, b=b, q_order=10)
    B_val = B_block(
        h,
        hbar,
        mp.mpf(1) / 2,
        mp.mpf(1) / 2,
        h_ext,
        hbar_ext,
        b=b,
        q_order=10,
    )
    assert _close(table[0][0], B_val, tol=mp.mpf("1e-25"))


def test_B_zzb_table_factorizes():
    """table[m][n] = chiral[m] * antichiral[n]."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h = mp.mpf("1.4")
    hbar = mp.mpf("1.1")

    table = B_zzb_table(h, hbar, h_ext, hbar_ext, 3, b=b, q_order=10)
    c_h = chiral_z_taylor(h, h_ext, 4, b=b, q_order=10)
    c_hbar = chiral_z_taylor(hbar, hbar_ext, 4, b=b, q_order=10)
    for m in range(4):
        for n in range(4):
            assert _close(
                table[m][n], c_h[m] * c_hbar[n], tol=mp.mpf("1e-30")
            )


def test_B_zzb_table_reconstructs_B_near_half():
    """Truncated 2D Taylor reproduces B near (z, zbar) = (1/2, 1/2)."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2")] * 4
    hbar_ext = [mp.mpf("0.2")] * 4
    h = mp.mpf("1.2")
    hbar = mp.mpf("1.0")

    n = 6
    table = B_zzb_table(h, hbar, h_ext, hbar_ext, n, b=b, q_order=12)
    z = mp.mpf(1) / 2 + mp.mpf("0.03")
    zbar = mp.mpf(1) / 2 - mp.mpf("0.02")
    dz = z - mp.mpf(1) / 2
    dzb = zbar - mp.mpf(1) / 2
    approx = mp.mpc(0)
    for m in range(n + 1):
        for k in range(n + 1):
            approx += table[m][k] * dz**m * dzb**k
    exact = B_block(h, hbar, z, zbar, h_ext, hbar_ext, b=b, q_order=12)
    assert _close(approx, exact, tol=mp.mpf("1e-7"))


def test_G_zzb_table_ell_zero():
    """For ell = 0, G_table[m][n] = c_m(h) c_n(h) with h = Delta/2 (the block B(h, h))."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    Delta = mp.mpf("2.8")

    table = G_zzb_table(Delta, 0, h_ext, 3, b=b, q_order=10)
    c_h = chiral_z_taylor(Delta / 2, h_ext, 4, b=b, q_order=10)
    for m in range(4):
        for n in range(4):
            assert _close(table[m][n], c_h[m] * c_h[n], tol=mp.mpf("1e-30"))


def test_G_zzb_table_parity_symmetry():
    """G_zzb_table[m][n] = G_zzb_table[n][m] for scalar externals at every ell (G is z <-> zbar even)."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("3.1")

    for ell in (0, 1, 2, 3):
        table = G_zzb_table(Delta, ell, h_ext, 4, b=b, q_order=10)
        for m in range(5):
            for n in range(5):
                assert _close(table[m][n], table[n][m], tol=mp.mpf("1e-30"))


def test_G_zzb_table_zeroth_entry_equals_G_at_half():
    """table[0][0] = G at the symmetric point."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    Delta = mp.mpf("2.8")

    for ell in (0, 1, 2):
        table = G_zzb_table(Delta, ell, h_ext, 2, b=b, q_order=10)
        G_val = G_block(
            Delta,
            ell,
            mp.mpf(1) / 2,
            mp.mpf(1) / 2,
            h_ext,
            b=b,
            q_order=10,
        )
        assert _close(table[0][0], G_val, tol=mp.mpf("1e-25"))


def test_global_G_zzb_table_zeroth_entry_equals_global_formula():
    mp.mp.dps = 40
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    Delta = mp.mpf("2.8")
    ell = 2
    z = mp.mpf(1) / 2
    h = (Delta + ell) / 2
    hbar = (Delta - ell) / 2

    table = global_G_zzb_table(Delta, ell, h_ext, 2)
    F_h = (
        mp.power(DEFAULT_CHIRAL_NORM, -h)
        * z**h
        * mp.hyp2f1(h + h_ext[1] - h_ext[0], h + h_ext[2] - h_ext[3], 2 * h, z)
    )
    F_hbar = (
        mp.power(DEFAULT_CHIRAL_NORM, -hbar)
        * z**hbar
        * mp.hyp2f1(
            hbar + h_ext[1] - h_ext[0],
            hbar + h_ext[2] - h_ext[3],
            2 * hbar,
            z,
        )
    )
    expected = 2 * F_h * F_hbar
    assert _close(table[0][0], expected, tol=mp.mpf("1e-25"))


def test_global_G_zzb_table_matches_large_c_limit():
    mp.mp.dps = 50
    c = mp.mpf("200")
    b = b_from_c(c)
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    Delta = mp.mpf("2.8")
    ell = 2

    virasoro = G_zzb_table(Delta, ell, h_ext, 2, b=b, q_order=24)
    global_table = global_G_zzb_table(Delta, ell, h_ext, 2)
    rel = abs((virasoro[0][0] - global_table[0][0]) / virasoro[0][0])
    assert rel < mp.mpf("3e-3")


def test_B_zzb_table_rectangular_shape():
    """m_max != n_max yields a rectangular table."""
    mp.mp.dps = 30
    b = b_from_c(mp.mpf("1.7"))
    table = B_zzb_table(
        mp.mpf("1.0"),
        mp.mpf("1.0"),
        [mp.mpf("0.1")] * 4,
        [mp.mpf("0.1")] * 4,
        3,
        n_max=5,
        b=b,
        q_order=8,
    )
    assert len(table) == 4
    for row in table:
        assert len(row) == 6


# yyb / qqb sanity


def test_chiral_y_taylor_zeroth_term_equals_F_at_y_symmetric():
    """The k=0 entry of chiral_y_taylor is F(h, z(y_*))."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    series = chiral_y_taylor(h, d, 3, b=b, q_order=10)
    F_at_y_symmetric = F_block(h, z_of_y(y_symmetric()), d, b=b, q_order=10)
    assert _close(series[0], F_at_y_symmetric, tol=mp.mpf("1e-25"))


def test_chiral_q_taylor_zeroth_term_equals_F_at_q_symmetric():
    """The k=0 entry of chiral_q_taylor is F(h, z(q_*))."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    series = chiral_q_taylor(h, d, 3, b=b, q_order=10)
    F_at_q_symmetric = F_block(h, z_of_q(q_symmetric()), d, b=b, q_order=10)
    assert _close(series[0], F_at_q_symmetric, tol=mp.mpf("1e-25"))


def test_chiral_y_taylor_reconstructs_F():
    """Truncated y-series of F reproduces F(h, z(y)) near y = y_symmetric."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    n_terms = 8
    series = chiral_y_taylor(h, d, n_terms, b=b, q_order=14)
    y = y_symmetric() + mp.mpf("0.01")
    delta = y - y_symmetric()
    approx = sum(series[k] * delta**k for k in range(n_terms))
    exact = F_block(h, z_of_y(y), d, b=b, q_order=14)
    assert _close(approx, exact, tol=mp.mpf("1e-12"))


def test_chiral_q_taylor_reconstructs_F():
    """Truncated q-series of F reproduces F(h, z(q)) near q = q_symmetric."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    n_terms = 8
    series = chiral_q_taylor(h, d, n_terms, b=b, q_order=14)
    q = q_symmetric() + mp.mpf("0.005")
    delta = q - q_symmetric()
    approx = sum(series[k] * delta**k for k in range(n_terms))
    exact = F_block(h, z_of_q(q), d, b=b, q_order=14)
    assert _close(approx, exact, tol=mp.mpf("1e-11"))


def test_B_yyb_table_factorizes():
    """table[m][n] = chiral_y[m] * chiral_y[n] in their respective h, hbar."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h, hbar = mp.mpf("1.4"), mp.mpf("1.1")

    table = B_yyb_table(h, hbar, h_ext, hbar_ext, 3, b=b, q_order=10)
    c_h = chiral_y_taylor(h, h_ext, 4, b=b, q_order=10)
    c_hbar = chiral_y_taylor(hbar, hbar_ext, 4, b=b, q_order=10)
    for m in range(4):
        for n in range(4):
            assert _close(table[m][n], c_h[m] * c_hbar[n], tol=mp.mpf("1e-30"))


def test_B_qqb_table_factorizes():
    """table[m][n] = chiral_q[m] * chiral_q[n] in their respective h, hbar."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h, hbar = mp.mpf("1.4"), mp.mpf("1.1")

    table = B_qqb_table(h, hbar, h_ext, hbar_ext, 3, b=b, q_order=10)
    c_h = chiral_q_taylor(h, h_ext, 4, b=b, q_order=10)
    c_hbar = chiral_q_taylor(hbar, hbar_ext, 4, b=b, q_order=10)
    for m in range(4):
        for n in range(4):
            assert _close(table[m][n], c_h[m] * c_hbar[n], tol=mp.mpf("1e-30"))


def test_B_yyb_table_reconstructs_B_near_y_symmetric():
    """2D y-Taylor reproduces B(z(y), z(ybar)) near y = ybar = y_symmetric."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2")] * 4
    hbar_ext = [mp.mpf("0.2")] * 4
    h = mp.mpf("1.2")
    hbar = mp.mpf("1.0")
    n = 6

    table = B_yyb_table(h, hbar, h_ext, hbar_ext, n, b=b, q_order=12)
    y_star = y_symmetric()
    y = y_star + mp.mpf("0.01")
    ybar = y_star - mp.mpf("0.005")
    dy = y - y_star
    dybar = ybar - y_star
    approx = mp.mpc(0)
    for m in range(n + 1):
        for k in range(n + 1):
            approx += table[m][k] * dy**m * dybar**k
    exact = B_block(h, hbar, z_of_y(y), z_of_y(ybar), h_ext, hbar_ext, b=b, q_order=12)
    assert _close(approx, exact, tol=mp.mpf("1e-10"))


def test_B_qqb_table_reconstructs_B_near_q_symmetric():
    """2D q-Taylor reproduces B(z(q), z(qbar)) near q = qbar = q_symmetric."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2")] * 4
    hbar_ext = [mp.mpf("0.2")] * 4
    h = mp.mpf("1.2")
    hbar = mp.mpf("1.0")
    n = 6

    table = B_qqb_table(h, hbar, h_ext, hbar_ext, n, b=b, q_order=12)
    q_star = q_symmetric()
    q = q_star + mp.mpf("0.002")
    qbar = q_star - mp.mpf("0.001")
    dq = q - q_star
    dqbar = qbar - q_star
    approx = mp.mpc(0)
    for m in range(n + 1):
        for k in range(n + 1):
            approx += table[m][k] * dq**m * dqbar**k
    exact = B_block(h, hbar, z_of_q(q), z_of_q(qbar), h_ext, hbar_ext, b=b, q_order=12)
    assert _close(approx, exact, tol=mp.mpf("1e-10"))


def test_G_yyb_table_parity_symmetry():
    """G_yyb[m][n] = G_yyb[n][m] at every ell."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("3.1")
    for ell in (0, 1, 2, 3):
        table = G_yyb_table(Delta, ell, h_ext, 4, b=b, q_order=10)
        for m in range(5):
            for n in range(5):
                assert _close(table[m][n], table[n][m], tol=mp.mpf("1e-30"))


def test_G_qqb_table_zeroth_entry_equals_G_at_q_symmetric():
    """The (0, 0) entry equals G_block(Delta, ell; z(q_*), z(q_*))."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    Delta = mp.mpf("2.8")
    for ell in (0, 1, 2):
        table = G_qqb_table(Delta, ell, h_ext, 2, b=b, q_order=10)
        z_star = z_of_q(q_symmetric())
        G_val = G_block(Delta, ell, z_star, z_star, h_ext, b=b, q_order=10)
        assert _close(table[0][0], G_val, tol=mp.mpf("1e-25"))


def test_yyb_qqb_table_validate_orders():
    h_ext = [mp.mpf("0.1")] * 4
    with pytest.raises(ValueError):
        B_yyb_table(0, 0, h_ext, h_ext, -1, c=mp.mpf("1.7"), q_order=4)
    with pytest.raises(ValueError):
        B_qqb_table(0, 0, h_ext, h_ext, -1, c=mp.mpf("1.7"), q_order=4)
    with pytest.raises(ValueError):
        G_qqb_table(mp.mpf("2.0"), 1.5, h_ext, 2, c=mp.mpf("1.7"), q_order=4)


# parity-folded coordinates: ws, xt, qqb_ws


def test_G_ws_table_even_ell_reconstructs_G_via_y():
    """For even ell, G(z, zbar) = sum table[a][b] w^a s^b, with
    w = (y + ybar)/2, s = ((y - ybar)/2)^2.

    Tolerance is set by the truncation: with n_max = 2 we drop O(s^3), and
    s ~ ((y - ybar)/2)^2 ~ 3e-4 here, so the residual sits a few times 1e-9.
    """
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.7")
    ell = 2  # even

    m_max = 4
    n_max = 2
    table = G_ws_table(Delta, ell, h_ext, m_max, n_max, b=b, q_order=12)

    y_val = mp.mpf("0.06")
    ybar_val = mp.mpf("-0.04")
    z = z_of_y(y_val)
    zbar = z_of_y(ybar_val)
    w_val = (y_val + ybar_val) / 2
    s_val = ((y_val - ybar_val) / 2) ** 2

    approx = mp.mpc(0)
    for a in range(m_max + 1):
        for bi in range(n_max + 1):
            approx += table[a][bi] * w_val**a * s_val**bi
    exact = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=12)
    assert _close(approx, exact, tol=mp.mpf("1e-7"))


def test_G_ws_table_odd_ell_reconstructs_G_without_r_factor():
    """For odd ell, G is z <-> zbar even: G(z, zbar) = sum table[a][b] w^a s^b, with no r = (y - ybar)/2 stripped."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("3.0")
    ell = 1  # odd

    m_max = 4
    n_max = 2
    table = G_ws_table(Delta, ell, h_ext, m_max, n_max, b=b, q_order=12)

    y_val = mp.mpf("0.06")
    ybar_val = mp.mpf("-0.04")
    z = z_of_y(y_val)
    zbar = z_of_y(ybar_val)
    w_val = (y_val + ybar_val) / 2
    r_val = (y_val - ybar_val) / 2
    s_val = r_val**2

    inner = mp.mpc(0)
    for a in range(m_max + 1):
        for bi in range(n_max + 1):
            inner += table[a][bi] * w_val**a * s_val**bi
    approx = inner
    exact = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=12)
    assert _close(approx, exact, tol=mp.mpf("1e-10"))


def test_G_ws_table_zeroth_even_ell_equals_G_at_half():
    """For even ell, table[0][0] = G at the symmetric point."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.6")
    for ell in (0, 2, 4):
        table = G_ws_table(Delta, ell, h_ext, 1, 1, b=b, q_order=10)
        G_val = G_block(
            Delta, ell, mp.mpf(1) / 2, mp.mpf(1) / 2, h_ext, b=b, q_order=10
        )
        assert _close(table[0][0], G_val, tol=mp.mpf("1e-25"))


def test_G_xt_table_reconstructs_G_via_z_fold():
    """For even ell, G(z, zbar) = sum table[a][b] (x - y_*)^a t^b with
    x = (z + zbar)/2 - 1/2, t = ((z - zbar)/2)^2."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.7")
    ell = 2

    m_max = 4
    n_max = 2
    table = G_xt_table(Delta, ell, h_ext, m_max, n_max, b=b, q_order=12)

    z = mp.mpf(1) / 2 + mp.mpf("0.02")
    zbar = mp.mpf(1) / 2 - mp.mpf("0.015")
    x_minus = (z + zbar) / 2 - mp.mpf(1) / 2
    t_val = ((z - zbar) / 2) ** 2

    approx = mp.mpc(0)
    for a in range(m_max + 1):
        for bi in range(n_max + 1):
            approx += table[a][bi] * x_minus**a * t_val**bi
    exact = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=12)
    assert _close(approx, exact, tol=mp.mpf("1e-10"))


def test_yyb_ws_xt_tables_match_same_physical_block():
    """yyb, ws, and xt expansions reproduce the same zzb block at fixed z, zbar."""
    mp.mp.dps = 60
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.7")
    ell = 2

    z = mp.mpf("0.53")
    zbar = mp.mpf("0.49")
    dz = z - mp.mpf(1) / 2
    dzbar = zbar - mp.mpf(1) / 2
    y_val = y_of_z(z)
    ybar_val = y_of_z(zbar)

    zzb_table = G_zzb_table(Delta, ell, h_ext, 6, b=b, q_order=14)
    yyb_table = G_yyb_table(Delta, ell, h_ext, 6, b=b, q_order=14)
    ws_table = G_ws_table(Delta, ell, h_ext, 6, 3, b=b, q_order=14)
    xt_table = G_xt_table(Delta, ell, h_ext, 6, 3, b=b, q_order=14)

    zzb_approx = mp.mpc(0)
    yyb_approx = mp.mpc(0)
    ws_approx = mp.mpc(0)
    xt_approx = mp.mpc(0)

    for m in range(7):
        for n in range(7):
            zzb_approx += zzb_table[m][n] * dz**m * dzbar**n
            yyb_approx += yyb_table[m][n] * y_val**m * ybar_val**n
    w_val = (y_val + ybar_val) / 2
    s_val = ((y_val - ybar_val) / 2) ** 2
    x_val = (z + zbar) / 2 - mp.mpf(1) / 2
    t_val = ((z - zbar) / 2) ** 2
    for a in range(7):
        for bi in range(4):
            ws_approx += ws_table[a][bi] * w_val**a * s_val**bi
            xt_approx += xt_table[a][bi] * x_val**a * t_val**bi

    exact = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=14)
    assert _close(zzb_approx, exact, tol=mp.mpf("5e-10"))
    assert _close(yyb_approx, exact, tol=mp.mpf("5e-10"))
    assert _close(ws_approx, exact, tol=mp.mpf("5e-10"))
    assert _close(xt_approx, exact, tol=mp.mpf("5e-10"))


def test_G_qqb_ws_table_reconstructs_G_via_q():
    """G_qqb_ws[a][b] reconstructs G(z(q), z(qbar)) for even ell."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.7")
    ell = 2

    m_max = 4
    n_max = 2
    table = G_qqb_ws_table(Delta, ell, h_ext, m_max, n_max, b=b, q_order=12)

    q_star = q_symmetric()
    q_val = q_star + mp.mpf("0.002")
    qbar_val = q_star - mp.mpf("0.001")
    w_q = (q_val + qbar_val) / 2 - q_star
    s_q = ((q_val - qbar_val) / 2) ** 2

    z = z_of_q(q_val)
    zbar = z_of_q(qbar_val)

    approx = mp.mpc(0)
    for a in range(m_max + 1):
        for bi in range(n_max + 1):
            approx += table[a][bi] * w_q**a * s_q**bi
    exact = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=12)
    assert _close(approx, exact, tol=mp.mpf("1e-10"))


def test_G_ws_table_validate_inputs():
    h_ext = [mp.mpf("0.1")] * 4
    with pytest.raises(ValueError):
        G_xt_table(mp.mpf("2.0"), 1.5, h_ext, 2, c=mp.mpf("1.7"), q_order=4)
    with pytest.raises(ValueError):
        G_qqb_ws_table(mp.mpf("2.0"), 0, h_ext, -1, c=mp.mpf("1.7"), q_order=4)


def test_derivs_validate_inputs():
    with pytest.raises(ValueError):
        chiral_z_taylor(mp.mpf("1"), [mp.mpf("0.1")] * 4, 0, c=mp.mpf("1.5"), q_order=4)
    with pytest.raises(ValueError):
        chiral_z_taylor(mp.mpf("1"), [mp.mpf("0.1")] * 3, 2, c=mp.mpf("1.5"), q_order=4)
    with pytest.raises(ValueError):
        chiral_z_taylor(mp.mpf("1"), [mp.mpf("0.1")] * 4, 2, q_order=4)
    with pytest.raises(ValueError):
        B_zzb_table(
            mp.mpf("1"),
            mp.mpf("1"),
            [mp.mpf("0.1")] * 4,
            [mp.mpf("0.1")] * 4,
            -1,
            c=mp.mpf("1.5"),
            q_order=4,
        )
    with pytest.raises(ValueError):
        G_zzb_table(
            mp.mpf("2"), 1.5, [mp.mpf("0.1")] * 4, 2, c=mp.mpf("1.5"), q_order=4
        )


# Radial-only slices.


def test_G_ws_radial_matches_n0_column_of_ws_table():
    """G_ws_radial_table is the n = 0 column of G_ws_table."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.7")
    for ell in (0, 1, 2, 3):
        full = G_ws_table(Delta, ell, h_ext, 4, 0, b=b, q_order=10)
        radial = G_ws_radial_table(Delta, ell, h_ext, 4, b=b, q_order=10)
        assert len(radial) == 5
        for m, (a, b_) in enumerate(zip(radial, [row[0] for row in full])):
            assert _close(a, b_, tol=mp.mpf("1e-35"))


def test_G_xt_radial_matches_n0_column():
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.7")
    for ell in (0, 2):
        radial = G_xt_radial_table(Delta, ell, h_ext, 4, b=b, q_order=10)
        full = G_xt_table(Delta, ell, h_ext, 4, 0, b=b, q_order=10)
        for a, b_ in zip(radial, [row[0] for row in full]):
            assert _close(a, b_, tol=mp.mpf("1e-35"))


def test_G_qqb_ws_radial_matches_n0_column():
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.7")
    radial = G_qqb_ws_radial_table(Delta, 2, h_ext, 4, b=b, q_order=10)
    full = G_qqb_ws_table(Delta, 2, h_ext, 4, 0, b=b, q_order=10)
    for a, b_ in zip(radial, [row[0] for row in full]):
        assert _close(a, b_, tol=mp.mpf("1e-35"))


def test_G_ws_radial_even_ell_reconstructs_G_on_diagonal():
    """For even ell, G(z(y), z(y)) = sum table[m] w^m along y = ybar."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("2.7")
    ell = 2

    m_max = 6
    radial = G_ws_radial_table(Delta, ell, h_ext, m_max, b=b, q_order=12)

    y = mp.mpf("0.02")
    z = z_of_y(y)
    w_val = y
    approx = mp.mpc(0)
    for m, c in enumerate(radial):
        approx += c * w_val**m
    exact = G_block(Delta, ell, z, z, h_ext, b=b, q_order=12)
    # Truncation-limited: keep through w^6, drop w^7. Coefficients grow
    # roughly geometrically near the symmetric point, so the residual sits a
    # few times 1e-9 with offset 0.02.
    assert _close(approx, exact, tol=mp.mpf("1e-7"))


def test_G_ws_radial_odd_ell_nonzero_on_diagonal():
    """For odd ell, G(z, z) is non-zero (G is z <-> zbar even) and the radial table starts with G(1/2, 1/2)."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("3.0")
    ell = 1

    radial = G_ws_radial_table(Delta, ell, h_ext, 4, b=b, q_order=10)
    z = mp.mpf(1) / 2 + mp.mpf("0.02")
    assert abs(G_block(Delta, ell, z, z, h_ext, b=b, q_order=10)) > mp.mpf("1e-10")
    half = mp.mpf(1) / 2
    assert _close(radial[0], G_block(Delta, ell, half, half, h_ext, b=b, q_order=10), tol=mp.mpf("1e-25"))


def test_G_radial_table_validates_inputs():
    h_ext = [mp.mpf("0.1")] * 4
    with pytest.raises(ValueError):
        G_xt_radial_table(mp.mpf("2.0"), 1.5, h_ext, 2, c=mp.mpf("1.7"), q_order=4)
    with pytest.raises(ValueError):
        G_qqb_ws_radial_table(mp.mpf("2.0"), 0, h_ext, -1, c=mp.mpf("1.7"), q_order=4)


def _max_rel(a, b):
    worst = mp.mpf(0)
    for x, y in zip(a, b):
        worst = max(
            worst,
            abs(mp.mpc(x) - mp.mpc(y)) / (abs(mp.mpc(x)) + mp.mpf("1e-200")),
        )
    return worst


def test_chiral_z_taylor_fast_matches_mp_taylor():
    """The analytic flint-series chiral block equals the mp.taylor path to
    working precision across c and h."""
    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    z_shift = -2 * mp.mpf("0.0625")
    for c in ("1.5", "10"):
        for h in ("0.1", "9.2"):
            ref = chiral_z_taylor(
                mp.mpf(h), h_ext, 12, c=mp.mpf(c), q_order=30, z_shift=z_shift
            )
            fast = chiral_z_taylor_fast(
                mp.mpf(h), h_ext, 12, c=mp.mpf(c), q_order=30, z_shift=z_shift
            )
            assert _max_rel(ref, fast) < mp.mpf("1e-65"), (c, h)


def test_chiral_y_taylor_fast_matches_mp_taylor():
    """The analytic flint-series chiral block in the y coordinate equals the
    mp.taylor path to working precision across c and h (parity with the z
    fast evaluator)."""
    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    z_shift = -2 * mp.mpf("0.0625")
    for c in ("1.5", "10"):
        for h in ("0.1", "9.2"):
            ref = chiral_y_taylor(
                mp.mpf(h), h_ext, 12, c=mp.mpf(c), q_order=30, z_shift=z_shift
            )
            fast = chiral_y_taylor_fast(
                mp.mpf(h), h_ext, 12, c=mp.mpf(c), q_order=30, z_shift=z_shift
            )
            assert _max_rel(ref, fast) < mp.mpf("1e-65"), (c, h)


def test_chiral_q_taylor_fast_matches_mp_taylor():
    """The analytic flint-series chiral block in the q coordinate equals the
    mp.taylor path to working precision across c and h."""
    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    z_shift = -2 * mp.mpf("0.0625")
    for c in ("1.5", "10"):
        for h in ("0.1", "9.2"):
            ref = chiral_q_taylor(
                mp.mpf(h), h_ext, 12, c=mp.mpf(c), q_order=30, z_shift=z_shift
            )
            fast = chiral_q_taylor_fast(
                mp.mpf(h), h_ext, 12, c=mp.mpf(c), q_order=30, z_shift=z_shift
            )
            assert _max_rel(ref, fast) < mp.mpf("1e-65"), (c, h)


@pytest.mark.parametrize(
    ("single", "many", "ell"),
    [
        (G_zzb_table, G_zzb_table_many, 0),
        (G_yyb_table, G_yyb_table_many, 2),
        (G_qqb_table, G_qqb_table_many, 0),
        (G_ws_table, G_ws_table_many, 2),
        (G_xt_table, G_xt_table_many, 0),
        (G_qqb_ws_table, G_qqb_ws_table_many, 2),
    ],
)
def test_G_table_many_matches_per_delta(single, many, ell):
    """Batched G-table builders equal the per-Delta builders across coordinates
    (one ell per coordinate, alternating even spins)."""
    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    z_shift = -2 * mp.mpf("0.0625")
    deltas = [mp.mpf(x) for x in ("0.3", "7.1")]
    ref = [
        single(d, ell, h_ext, 6, 6, c=mp.mpf("2"), q_order=20, z_shift=z_shift)
        for d in deltas
    ]
    batched = many(
        deltas, ell, h_ext, 6, 6, c=mp.mpf("2"), q_order=20, z_shift=z_shift
    )
    worst = mp.mpf(0)
    for table_ref, table_bat in zip(ref, batched):
        for row_ref, row_bat in zip(table_ref, table_bat):
            worst = max(worst, _max_rel(row_ref, row_bat))
    assert worst < mp.mpf("1e-65"), (single.__name__, ell)


@pytest.mark.parametrize(
    ("single", "many"),
    [
        (G_ws_table, G_ws_table_many),
        (G_xt_table, G_xt_table_many),
        (G_qqb_ws_table, G_qqb_ws_table_many),
    ],
)
def test_folded_table_many_odd_ell_and_complex_fallback(single, many):
    """The fused arb folded builder matches per-Delta on the odd-ell r-strip
    fold, and the complex-external path falls back to the generic builder."""
    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    z_shift = -2 * mp.mpf("0.0625")
    deltas = [mp.mpf(x) for x in ("1.7", "8.4")]
    # Odd ell exercises the r-stripped (smoothed) fold branch.
    ref = [
        single(d, 3, h_ext, 5, 5, c=mp.mpf("2"), q_order=20, z_shift=z_shift)
        for d in deltas
    ]
    batched = many(deltas, 3, h_ext, 5, 5, c=mp.mpf("2"), q_order=20, z_shift=z_shift)
    worst = mp.mpf(0)
    for table_ref, table_bat in zip(ref, batched):
        for row_ref, row_bat in zip(table_ref, table_bat):
            worst = max(worst, _max_rel(row_ref, row_bat))
    assert worst < mp.mpf("1e-65"), (single.__name__, "odd")

    # Complex externals route the fused builder through its acb branch.
    h_complex = [mp.mpc(0, "0.1")] + [mp.mpf("0.0625")] * 3
    ref_c = [
        single(d, 0, h_complex, 4, 4, c=mp.mpf("2"), q_order=18, z_shift=z_shift)
        for d in deltas
    ]
    batched_c = many(
        deltas, 0, h_complex, 4, 4, c=mp.mpf("2"), q_order=18, z_shift=z_shift
    )
    worst_c = mp.mpf(0)
    for table_ref, table_bat in zip(ref_c, batched_c):
        for row_ref, row_bat in zip(table_ref, table_bat):
            worst_c = max(worst_c, _max_rel(row_ref, row_bat))
    assert worst_c < mp.mpf("1e-65"), (single.__name__, "complex")


@pytest.mark.parametrize(
    ("single", "many"),
    [
        (G_ws_table, G_ws_table_many),
        (G_xt_table, G_xt_table_many),
        (G_qqb_ws_table, G_qqb_ws_table_many),
    ],
)
def test_folded_table_many_complex_delta_acb(single, many):
    """The fused folded builder's acb branch (complex Delta, and complex c)
    matches the per-Delta builder. Guards the exponent prefactor (16 q)^h /
    N^{-h}, which must carry the full complex h -- not just its real part."""
    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    z_shift = -2 * mp.mpf("0.0625")
    deltas = [mp.mpc("1.7", "0.05"), mp.mpc("8.4", "0.3")]

    def _worst(ref, bat):
        w = mp.mpf(0)
        for tr, tb in zip(ref, bat):
            for rr, rb in zip(tr, tb):
                w = max(w, _max_rel(rr, rb))
        return w

    # Complex Delta on real Kac data (real c), even and odd ell.
    for ell in (0, 3):
        ref = [
            single(d, ell, h_ext, 5, 5, c=mp.mpf("2"), q_order=20, z_shift=z_shift)
            for d in deltas
        ]
        bat = many(deltas, ell, h_ext, 5, 5, c=mp.mpf("2"), q_order=20, z_shift=z_shift)
        assert _worst(ref, bat) < mp.mpf("1e-65"), (single.__name__, "real-c", ell)

    # Complex c (generic-complex mode) at complex Delta.
    c_complex = mp.mpc("2.0", "0.4")
    ref = [
        single(d, 0, h_ext, 4, 4, c=c_complex, q_order=18, z_shift=z_shift)
        for d in deltas
    ]
    bat = many(deltas, 0, h_ext, 4, 4, c=c_complex, q_order=18, z_shift=z_shift)
    assert _worst(ref, bat) < mp.mpf("1e-65"), (single.__name__, "complex-c")


@pytest.mark.parametrize(
    "global_taylor,z_of_u,u_symmetric",
    [(global_chiral_z_taylor, lambda u: u, lambda: mp.mpf(1) / 2),
     (global_chiral_y_taylor, z_of_y, y_symmetric),
     (global_chiral_q_taylor, z_of_q, q_symmetric)],
)
def test_global_chiral_taylor_flint_matches_mp_taylor(global_taylor, z_of_u, u_symmetric):
    """The flint acb_series global SL(2) chiral builder reproduces mp.taylor of the closed-form 2F1 block in every
    coordinate (z, y, and the nome q)."""
    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    z_shift = -2 * mp.mpf("0.0625")
    worst = mp.mpf(0)
    for h in (mp.mpf("0.3"), mp.mpc("1.7", "0.2")):  # one real, one complex
        flint_res = global_taylor(h, h_ext, 10, z_shift=z_shift)
        mp_res = mp.taylor(
            lambda u: _derivs_mod._global_chiral_block(h, z_of_u(u), h_ext, z_shift=z_shift), u_symmetric(), 9,
        )
        for a, b in zip(flint_res, mp_res):
            rel = abs(mp.mpc(a) - mp.mpc(b)) / (abs(mp.mpc(a)) + mp.mpf("1e-200"))
            worst = max(worst, rel)
    assert worst < mp.mpf("1e-55"), worst


def test_H_context_cache_is_h_independent():
    """The cached H-recursion context must not depend on h: evaluating at a
    second h after the first must still match a fresh mp.taylor block."""
    mp.mp.dps = 60
    h_ext = [mp.mpf("0.1")] * 4
    evaluate = None
    # First evaluation populates the H-context cache; the second must reuse it
    # and still agree with the independent reference for a different h.
    for h in ("0.4", "2.2"):
        ref = chiral_z_taylor(mp.mpf(h), h_ext, 8, c=mp.mpf("3"), q_order=16)
        fast = chiral_z_taylor_fast(mp.mpf(h), h_ext, 8, c=mp.mpf("3"), q_order=16)
        assert _max_rel(ref, fast) < mp.mpf("1e-45"), h


@pytest.mark.parametrize("c", ["4", "30", "1.2"])
def test_chiral_series_evaluator_many_matches_per_h(c):
    """The batched `evaluate.many` reproduces per-h `evaluate(return_arb=True)`
    for both strip (1 < c < 25) and real-nonstrip (c > 25) modes -- it only
    reorganizes the H-recursion into one matmul over all h."""
    from virasoro_blocks.derivs import chiral_z_series_evaluator
    from virasoro_blocks.polynomial import _arb_to_mpf

    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    b = b_from_c(mp.mpf(c))
    ev = chiral_z_series_evaluator(h_ext, 12, b=b, q_order=20)
    hs = [mp.mpf("0.5"), mp.mpf("1.7"), mp.mpf("3.2"), mp.mpf("5.1")]
    many = ev.many(hs)
    assert many is not None
    worst = mp.mpf(0)
    for h, coeffs in zip(hs, many):
        ref = ev(h, return_arb=True)
        for a, r in zip(coeffs, ref):
            worst = max(worst, abs(_arb_to_mpf(a) - _arb_to_mpf(r)))
    assert worst < mp.mpf("1e-70"), (c, worst)


@pytest.mark.parametrize("c", ["4", "30"])
def test_chiral_series_pole_laurent_matches_pole_data_special_case(c):
    """`pole_data` remains the (1, 0) special case of `pole_laurent`."""
    from virasoro_blocks.derivs import chiral_z_series_evaluator
    from virasoro_blocks.polyderivs import _acb_to_mpc

    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    ev = chiral_z_series_evaluator(
        h_ext,
        8,
        b=b_from_c(mp.mpf(c)),
        q_order=20,
    )
    locs = ev.kac_locations()
    assert locs is not None and len(locs) > 0
    loc = None
    for cand in locs:
        if ev.pole_data(cand) is not None:
            loc = cand
            break
    assert loc is not None

    pd = ev.pole_data(loc)
    assert pd is not None
    res0, reg0 = pd

    pl = ev.pole_laurent(loc, principal_order=1, regular_order=0)
    assert pl is not None
    principal, regular = pl
    assert len(principal) == 1
    assert len(regular) == 1
    ndig = mp.mp.dps + 5
    for i in range(8):
        assert abs(_acb_to_mpc(principal[0][i], ndig) - _acb_to_mpc(res0[i], ndig)) < mp.mpf("1e-65")
        assert abs(_acb_to_mpc(regular[0][i], ndig) - _acb_to_mpc(reg0[i], ndig)) < mp.mpf("1e-65")


def test_chiral_series_pole_laurent_higher_orders_shape_and_simple_limit():
    """At a simple Kac pole, higher principal orders beyond 1 are zero."""
    from virasoro_blocks.derivs import chiral_z_series_evaluator
    from virasoro_blocks.polyderivs import _acb_to_mpc

    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    ev = chiral_z_series_evaluator(h_ext, 6, c=mp.mpf("30"), q_order=16)
    locs = ev.kac_locations()
    assert locs is not None and len(locs) > 0
    loc = None
    for cand in locs:
        if ev.pole_data(cand) is not None:
            loc = cand
            break
    assert loc is not None

    pl = ev.pole_laurent(loc, principal_order=3, regular_order=2)
    assert pl is not None
    principal, regular = pl
    assert len(principal) == 3
    assert len(regular) == 3
    assert all(len(row) == 6 for row in principal)
    assert all(len(row) == 6 for row in regular)

    ndig = mp.mp.dps + 5
    for i in range(6):
        assert abs(_acb_to_mpc(principal[1][i], ndig)) < mp.mpf("1e-60")
        assert abs(_acb_to_mpc(principal[2][i], ndig)) < mp.mpf("1e-60")


def test_chiral_series_pole_laurent_supports_explicit_matched_cluster():
    """Anchored Laurent assembly can use an explicit matched probe cluster.

    This is the core hook option-B regularization uses: around an anchored
    location we can force a selected split-pole set to contribute to the
    principal part while everything else is treated as regular.
    """
    from virasoro_blocks.derivs import chiral_z_series_evaluator

    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    ev = chiral_z_series_evaluator(h_ext, 6, c=mp.mpf("30"), q_order=16)
    poles = ev.kac_poles()
    assert poles is not None and len(poles) >= 2
    # Anchor between two simple probe poles: no implicit match at this location,
    # but an explicit matched cluster must still produce a valid Laurent object.
    anchor = (mp.mpf(poles[0]) + mp.mpf(poles[1])) / 2
    assert ev.pole_laurent(anchor, principal_order=1, regular_order=0) is None
    forced = ev.pole_laurent(
        anchor,
        principal_order=1,
        regular_order=1,
        matched_locations=[poles[0]],
    )
    assert forced is not None
    principal, regular = forced
    assert len(principal) == 1
    assert len(regular) == 2
    assert len(principal[0]) == 6
    assert len(regular[0]) == 6


def test_assemble_block_grid_paths_agree():
    """The shared block-grid assembler agrees across its flint (full grid) and
    elementwise (entry-restricted / fallback) paths, and reproduces the defining
    outer-product combination for the ordered (sigma = 0), symmetric (c_h is
    c_hbar) and parity (complex sigma) cases."""
    from virasoro_blocks.derivs import _assemble_block_grid

    mp.mp.dps = 60
    c_h = [mp.mpf(1), mp.mpf("0.5"), mp.mpc("0.3", "0.1"), mp.mpf("-0.2")]
    c_hb = [mp.mpf("0.9"), mp.mpc("0.2", "-0.4"), mp.mpf("0.1"), mp.mpf("0.05")]
    m_max, n_max = 3, 2
    all_entries = [(m, n) for m in range(m_max + 1) for n in range(n_max + 1)]

    def reference(sigma):
        return [
            [c_h[m] * c_hb[n] + (0 if sigma == 0 else sigma) * (c_hb[m] * c_h[n])
             for n in range(n_max + 1)]
            for m in range(m_max + 1)
        ]

    for sigma in (mp.mpf(0), mp.mpf(1), mp.mpf(-1), mp.mpc("0", "1")):
        ref = reference(sigma)
        full = _assemble_block_grid(c_h, c_hb, m_max, n_max, sigma=sigma)
        restricted = _assemble_block_grid(
            c_h, c_hb, m_max, n_max, sigma=sigma, entries=all_entries
        )
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                tol = mp.mpf(10) ** -55 * (abs(ref[m][n]) + 1)
                assert abs(full[m][n] - ref[m][n]) <= tol
                assert abs(restricted[m][n] - ref[m][n]) <= tol

    # c_h is c_hbar (scalar ell = 0): grid = (1 + sigma) * outer product.
    same = _assemble_block_grid(c_h, c_h, m_max, n_max, sigma=mp.mpf(1))
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            assert abs(same[m][n] - 2 * c_h[m] * c_h[n]) <= mp.mpf(10) ** -55 * (
                abs(c_h[m] * c_h[n]) + 1
            )


def test_fold_geometry_helpers():
    """The shared fold-geometry helpers encode the (w, s) conventions."""
    from virasoro_blocks.derivs import (
        fold_parity_marker, folded_source_order, folded_entry_power,
    )

    # parity marker: scalar G is even at every ell; the projection sign sets it otherwise.
    for ell in (0, 2, 3, -3):
        assert fold_parity_marker(ell=ell) == 0
    assert fold_parity_marker(parity="+") == 0
    assert fold_parity_marker(parity="-") == 1
    with pytest.raises(ValueError):
        fold_parity_marker(parity="x")

    # source order and entry power agree with the m + 2n (+odd) convention.
    for m in range(4):
        for n in range(4):
            for odd in (0, 1):
                assert folded_source_order(m, n, odd) == m + 2 * n + odd
                assert folded_entry_power(m, n, odd) == m + 2 * n + odd


def test_coordinate_registry_consistency():
    """The canonical registry agrees with the maps derived from it."""
    from virasoro_blocks.coordinates import COORDINATES, FOLDED_BASE_2D
    from virasoro_blocks.global_blocks import _FOLDED_2D_TO_CHIRAL
    from virasoro_blocks.large_spin import (
        LARGE_SPIN_COORDINATES, _LARGE_SPIN_CHIRAL_COORD,
    )
    from virasoro_blocks.polyderivs import _COORD_CHIRAL_FAMILY

    assert FOLDED_BASE_2D["xt"] == "zzb"
    assert FOLDED_BASE_2D["ws"] == "yyb"
    assert FOLDED_BASE_2D["qqb_ws"] == "qqb"
    # global folded map covers exactly the folded_2d coordinates.
    assert _FOLDED_2D_TO_CHIRAL == {
        c: s.base_2d for c, s in COORDINATES.items() if s.kind == "folded_2d"
    }
    # large-spin map covers its supported coords, mapping to the 2d parent.
    assert _LARGE_SPIN_CHIRAL_COORD == {
        c: COORDINATES[c].base_2d for c in LARGE_SPIN_COORDINATES
    }
    # polyderivs chiral family is the (var, z_of_u, u_symmetric) projection.
    for c, spec in COORDINATES.items():
        assert _COORD_CHIRAL_FAMILY[c] == (spec.chiral_var, spec.z_of_u, spec.u_symmetric)


def test_assemble_block_batch_arb_matches_mpmath():
    """The arb-native batched assembler reproduces the mpmath single-grid path
    (and its `_fold_parity` fold) for non-folded, folded, ordered (sigma=None),
    and complex-sigma cases."""
    from flint import arb
    from virasoro_blocks.derivs import (
        assemble_block_batch, _assemble_block_grid, _fold_parity,
    )
    from virasoro_blocks.polynomial import _mpf_to_arb, _arb_to_mpf, _acb_to_mpc

    mp.mp.dps = 50
    # Real source coefficients long enough for the folded source order.
    src = 8
    c_h = [mp.mpf(1) / (k + 1) ** 2 - mp.mpf("0.01") * k for k in range(src + 1)]
    c_hb = [mp.mpf("0.7") ** k + mp.mpf("0.02") * k for k in range(src + 1)]
    ch_arb = [_mpf_to_arb(x) for x in c_h]
    chb_arb = [_mpf_to_arb(x) for x in c_hb]
    m_max, n_max = 3, 2

    def close(a, b):
        return abs(a - b) <= mp.mpf(10) ** -44 * (abs(b) + 1)

    # non-folded, sigma = -1 (odd scalar) and sigma = None (ordered).
    for sigma in (mp.mpf(-1), None):
        tables, is_arb = assemble_block_batch(
            [ch_arb], [chb_arb], m_max, n_max, sigma=sigma,
        )
        assert is_arb
        ref = _assemble_block_grid(c_h, c_hb, m_max, n_max, sigma=sigma or 0)
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                assert close(_arb_to_mpf(tables[0][m][n]), ref[m][n])

    # folded, sigma = -1, odd parity: arb fused fold vs mpmath source + fold.
    odd = 1
    tables, is_arb = assemble_block_batch(
        [ch_arb], [chb_arb], m_max, n_max, sigma=mp.mpf(-1), fold_parity=odd,
    )
    assert is_arb
    M = m_max + 2 * n_max + odd
    src_grid = _assemble_block_grid(c_h, c_hb, M, M, sigma=mp.mpf(-1))
    folded_ref = _fold_parity(src_grid, odd, m_max, n_max)
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            assert close(_arb_to_mpf(tables[0][m][n]), folded_ref[m][n])

    # complex sigma promotes real arb input to acb (mpc output).
    sigma = mp.mpc("0", "1")
    tables, is_arb = assemble_block_batch(
        [ch_arb], [chb_arb], m_max, n_max, sigma=sigma,
    )
    assert not is_arb
    ref = _assemble_block_grid(c_h, c_hb, m_max, n_max, sigma=sigma)
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            got = tables[0][m][n]
            assert abs(got - ref[m][n]) <= mp.mpf(10) ** -44 * (abs(ref[m][n]) + 1)
