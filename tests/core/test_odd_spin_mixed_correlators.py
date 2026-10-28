"""Odd-spin scalar_G blocks against exactly known mixed correlators of distinct scalars.

For scalars phi_1 != phi_2 the s-channel of <phi_1 phi_2 phi_2 phi_1> and <phi_1 phi_2 phi_1 phi_2> exchanges primaries of
every integer spin.  The (h, hbar) and (hbar, h) primaries enter with the same OPE coefficients, so each pair is
p_{Delta, ell} G_{Delta, ell} with G = B(h, hbar) + B(hbar, h).  The coefficient is lambda_{12O}^2 >= 0 in
<phi_1 phi_2 phi_2 phi_1> and (-1)^ell lambda_{12O}^2 in <phi_1 phi_2 phi_1 phi_2>: the ordering sign belongs to the
coefficient, not to the block.  Both correlators below factorize, G(z, zbar) = g(z) g(zbar), and are reproduced only
when the odd-spin blocks are the symmetric sum (the antisymmetric B - B misses them at O(1e-2) or worse).

* 2d generalized free fields with global blocks: the 1d coefficients are the closed form
  (2 h_1)_n (2 h_2)_n / (n! (2 h_1 + 2 h_2 + n - 1)_n), with (-1)^n in <phi_1 phi_2 phi_1 phi_2>.
* (Ising)^2 at c = 1 with Virasoro blocks: phi_1 = sigma x 1 and phi_2 = 1 x sigma, so
  <phi_1 phi_2 phi_2 phi_1> = <sigma sigma>^2.  The s-channel module sigma x sigma decomposes into c = 1 Virasoro
  primaries of weight (2n + 1)^2 / 8, by chi_{1/16}^2 = sum_n q^{n(n+1)/2} chi^{Verma}_{1/8}.  Its chiral coefficients
  are extracted from the c = 1 blocks, and the 2d sum is checked at a point with z != zbar.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.composer import G_block, parity_projected_B_block
from virasoro_blocks.derivs import global_G_zzb_table, global_parity_projected_B_zzb_table
from virasoro_blocks.recursion import DEFAULT_CHIRAL_NORM, F_block


def _ordering(name, h1, h2):
    """External chiral weights and the stripped chiral correlator g(z) of the two orderings."""
    H = h1 + h2
    if name == "1221":
        return (h1, h2, h2, h1), (lambda z: z**H * (1 - z) ** (-2 * h2)), 1
    return (h1, h2, h1, h2), (lambda z: z**H), -1


@pytest.mark.parametrize("name", ["1221", "1212"])
def test_global_gff_mixed_correlator_needs_symmetric_odd_spin_blocks(name):
    mp.mp.dps = 40
    h1, h2, M, K = mp.mpf("0.17"), mp.mpf("0.29"), 3, 40
    H, N = h1 + h2, DEFAULT_CHIRAL_NORM
    ext, g, sign = _ordering(name, h1, h2)

    def coefficient(n):
        a = mp.rf(2 * h1, n) * mp.rf(2 * h2, n) / (mp.factorial(n) * mp.rf(2 * H + n - 1, n))
        return sign**n * a * mp.mpf(N) ** (H + n)      # the library's chiral blocks carry N^-h

    g_taylor = mp.taylor(g, mp.mpf(1) / 2, M)
    exact = [[g_taylor[m] * g_taylor[n] for n in range(M + 1)] for m in range(M + 1)]
    even = [[mp.mpf(0)] * (M + 1) for _ in range(M + 1)]
    odd, odd_antisymmetric = [row[:] for row in even], [row[:] for row in even]
    for n in range(K):
        for m in range(n + 1):
            p, ell = coefficient(n) * coefficient(m), n - m
            table = global_G_zzb_table(2 * H + n + m, ell, list(ext), M, M)
            target = odd if ell % 2 else even
            if ell % 2:
                if name == "1221":
                    assert p > 0                       # lambda^2 at odd spin too
                minus = global_parity_projected_B_zzb_table(
                    2 * H + n + m, ell, list(ext), list(ext), M, M, parity="-")
            for i in range(M + 1):
                for j in range(M + 1 - i):
                    target[i][j] += p * table[i][j]
                    if ell % 2:
                        odd_antisymmetric[i][j] += p * minus[i][j]
    scale = abs(exact[0][0])
    entries = [(i, j) for i in range(M + 1) for j in range(M + 1 - i)]
    assert max(abs(even[i][j] + odd[i][j] - exact[i][j]) for i, j in entries) < mp.mpf("1e-24") * scale
    assert max(abs(odd[i][j]) for i, j in entries) > mp.mpf("1e-2") * scale
    assert max(abs(even[i][j] + odd_antisymmetric[i][j] - exact[i][j]) for i, j in entries) > mp.mpf("1e-2") * scale


def _ising_squared_chiral_coefficients(K, *, q_order):
    """Coefficients b_k of z^(1/8) (1 - z)^(-1/8) and z^(1/8) in c = 1 blocks F_{1/8 + k} with externals 1/16."""
    s, R, points = mp.mpf(1) / 16, mp.mpf("0.4"), 48
    ext, H = [s] * 4, 2 * s

    def taylor_at_zero(f):
        values = [f(R * mp.expjpi(2 * mp.mpf(j) / points)) for j in range(points)]
        return [mp.re(sum(v * mp.expjpi(-2 * mp.mpf(j * k) / points) for j, v in enumerate(values)) / points / R**k)
                for k in range(K)]

    blocks = [taylor_at_zero(lambda z, k=k: F_block(H + k, z, ext, c=1, q_order=q_order) / z ** (H + k))
              for k in range(K)]

    def solve(target):
        b = []
        for k in range(K):
            b.append((target[k] - sum(b[n] * blocks[n][k - n] for n in range(k))) / blocks[k][0])
        return b

    b1221 = solve([mp.rf(2 * s, k) / mp.factorial(k) for k in range(K)])
    b1212 = solve([mp.mpf(1)] + [mp.mpf(0)] * (K - 1))
    return ext, H, b1221, b1212


def test_ising_squared_c1_mixed_correlator_needs_symmetric_odd_spin_blocks():
    mp.mp.dps = 30
    K, q_order = 12, 16
    ext, H, b1221, b1212 = _ising_squared_chiral_coefficients(K, q_order=q_order)

    # the chiral spectrum: c = 1 primaries at h = (2n + 1)^2 / 8, i.e. k = n (n + 1) / 2, with lambda^2 > 0 in 1221
    # and the ordering sign (-1)^k in 1212
    triangular = {n * (n + 1) // 2 for n in range(5)}
    tol = mp.mpf("1e-8")
    for k in range(K):
        if k in triangular:
            assert b1221[k] > 0.1
            assert abs(b1212[k] - (-1) ** k * b1221[k]) < tol * b1221[k]
        else:
            assert abs(b1221[k]) < tol and abs(b1212[k]) < tol

    z, zbar = mp.mpf("0.04"), mp.mpf("0.09")
    s = ext[0]
    for b, g in ((b1221, lambda x: x**H * (1 - x) ** (-2 * s)), (b1212, lambda x: x**H)):
        symmetric = antisymmetric = mp.mpf(0)
        for k in triangular:
            for l in triangular:
                if l > k:
                    continue
                p, ell = b[k] * b[l], k - l
                G = G_block(2 * H + k + l, ell, z, zbar, ext, c=1, q_order=q_order)
                symmetric += p * G
                antisymmetric += p * (parity_projected_B_block(H + k, H + l, z, zbar, ext, ext, parity="-", c=1,
                                                               q_order=q_order) if ell % 2 else G)
        exact = g(z) * g(zbar)
        assert abs(symmetric - exact) < mp.mpf("1e-18") * exact
        assert abs(antisymmetric - exact) > mp.mpf("1e-3") * exact
