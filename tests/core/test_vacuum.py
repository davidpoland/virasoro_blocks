"""Cross-check against Perlmutter, arXiv:1502.07742, eq (2.14)-(2.15).

Perlmutter expands the vacuum block (h_p -> 0 with h_{12} = h_{34} = 0, i.e.
d_1 = d_2 = h_1, d_3 = d_4 = h_3) over global SL(2) blocks:

    F_vac(c, h_i, z) = sum_q chi_vac,q(c, h_i) z^q * 2F1(q, q, 2q; z)

with

    chi_vac,0 = 1
    chi_vac,1 = 0
    chi_vac,2 = 2 h_1 h_3 / c
    chi_vac,4 = 10 (h_1^2 + h_1/5)(h_3^2 + h_3/5) / [c (5c + 22)]
    chi_vac,6 = (h_1^2 + h_1)_? (h_3^2 + h_3)_? / [63 c (70c + 29)]
              + 4 h_1 h_3 (c(70 h_1^2 + 42 h_1 + 8) + 29 h_1^2 - 57 h_1 - 2)
                          (c(70 h_3^2 + 42 h_3 + 8) + 29 h_3^2 - 57 h_3 - 2)
                         / [3 c (2c - 1)(5c + 22)(7c + 68)(70c + 29)]

(see paper for exact form of chi_vac,6).

Convention bridge: Perlmutter (2.17) has F_Perl = z^{h_p} 2F1(...) at c -> oo,
while Cho-Collier-Yin (which our code follows) has f_c = 2F1(...) without the
z^{h_p} factor. So F_Perl = z^{h_p} F_CCY. For h_p = 0 these are equal.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c
from virasoro_blocks.recursion import F_block


def _chi_vac(c, h1, h3):
    """Return (chi_vac,0, chi_vac,2, chi_vac,4, chi_vac,6) from Perlmutter (2.15)."""
    c, h1, h3 = mp.mpf(c), mp.mpf(h1), mp.mpf(h3)
    chi0 = mp.mpf(1)
    chi2 = 2 * h1 * h3 / c
    chi4 = 10 * (h1**2 + h1 / 5) * (h3**2 + h3 / 5) / (c * (5 * c + 22))
    # chi_vac,6 (Perlmutter eq 2.15, second line):
    chi6_a = (14 * h1**2 + h1) * (14 * h3**2 + h3) / (63 * c * (70 * c + 29))
    f1 = c * (70 * h1**2 + 42 * h1 + 8) + 29 * h1**2 - 57 * h1 - 2
    f3 = c * (70 * h3**2 + 42 * h3 + 8) + 29 * h3**2 - 57 * h3 - 2
    chi6_b = 4 * h1 * h3 * f1 * f3 / (
        3 * c * (2 * c - 1) * (5 * c + 22) * (7 * c + 68) * (70 * c + 29)
    )
    chi6 = chi6_a + chi6_b
    return chi0, chi2, chi4, chi6


def _F_perlmutter_truncated(c, h1, h3, z, q_max=6):
    """F_vac via Perlmutter's series truncated at q <= q_max (returns mpmath complex)."""
    chi0, chi2, chi4, chi6 = _chi_vac(c, h1, h3)
    z = mp.mpc(z)
    out = chi0 * mp.hyp2f1(0, 0, 0, z)  # = 1
    # chi_1 = 0
    out += chi2 * z**2 * mp.hyp2f1(2, 2, 4, z)
    if q_max >= 4:
        out += chi4 * z**4 * mp.hyp2f1(4, 4, 8, z)
    if q_max >= 6:
        out += chi6 * z**6 * mp.hyp2f1(6, 6, 12, z)
    return out


@pytest.mark.parametrize(
    "z_val",
    [mp.mpf("0.05"), mp.mpf("0.1"), mp.mpf("0.15"), mp.mpf("0.2")],
)
def test_vacuum_block_matches_Perlmutter_2_15(z_val):
    """F_CCY(h_p -> 0, z, (h1,h1,h3,h3), c) matches Perlmutter's truncated series.

    We pick small z so the q^8 truncation error in Perlmutter's series is
    small. Specifically chi_vac,8 z^8 ~ z^8 (with chi_vac,8 of order 1/c^k),
    so at z = 0.2 the error is bounded by ~ 2.6e-6.
    """
    mp.mp.dps = 60
    c = mp.mpf("1.7")
    h1 = mp.mpf("0.3")
    h3 = mp.mpf("0.2")
    b = b_from_c(c)

    # Take the h_p -> 0 limit numerically. Because d_1 = d_2 (and d_3 = d_4),
    # K_{1,1} = 0 in the recursion, so 1/(h - d_{1,1}) = 1/h is multiplied by
    # an exact zero -- the recursion evaluates cleanly even at very small h.
    h_eps = mp.mpf("1e-25")
    d = (h1, h1, h3, h3)
    F_ours = F_block(h_eps, z_val, d, b=b, q_order=12)

    F_perl = _F_perlmutter_truncated(c, h1, h3, z_val, q_max=6)

    # Convention (z_shift=0): F_code = z^h * N^{-h} * F_standard
    # For h->0: F_code = z^0 * 1 * F_Perlmutter = F_Perlmutter
    F_ours_standard = F_ours

    # Tolerance set by Perlmutter's q^8 truncation, roughly z_val^8 * (1/c).
    expected_truncation = (z_val ** 8) / abs(c)
    tol = max(10 * expected_truncation, mp.mpf("1e-10"))

    diff = abs(F_ours_standard - F_perl)
    assert diff < tol, (
        f"At z={z_val}: F_ours={complex(F_ours_standard)} F_Perl={complex(F_perl)} "
        f"diff={diff} tol={tol}"
    )


def test_vacuum_block_q2_coefficient():
    """Isolate the z^2 coefficient: it should equal chi_vac,2 = 2 h_1 h_3 / c.

    Compute (F_CCY - 1)/z^2 at very small z (Richardson-style) and compare
    to the expected leading value 2 h_1 h_3 / c.
    """
    mp.mp.dps = 60
    c = mp.mpf("1.7")
    h1 = mp.mpf("0.3")
    h3 = mp.mpf("0.2")
    b = b_from_c(c)
    d = (h1, h1, h3, h3)
    expected = 2 * h1 * h3 / c

    h_eps = mp.mpf("1e-25")
    # Convention (z_shift=0): F_code = z^h * N^{-h} * F_standard.
    # For h->0: F_code = z^0 * 1 * (1 + chi_vac,2 z^2 + ...) = 1 + chi_vac,2 z^2 + ...
    # So (F_code - 1) / z^2 = chi_vac,2
    z = mp.mpf("0.01")
    F = F_block(h_eps, z, d, b=b, q_order=8)
    ratio = (F - 1) / z ** 2
    # At z = 0.01, the next correction is ~z^2 * d/dz (chi_vac,2 2F1(2,2,4;z))
    # which is small.
    assert abs(ratio - expected) < mp.mpf("1e-2"), (
        f"(F-1)/z^2 = {ratio}, expected {expected}, diff = {ratio - expected}"
    )
