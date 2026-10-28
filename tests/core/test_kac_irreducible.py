"""The irreducible-module block on a Kac weight of a singular c (``kac_limit="irreducible"``).

On a Kac weight where poles of one level merge (every ``h = m^2/4`` at c = 1, ``h = 1 - n^2/4`` at c = 25, same-level
collisions at ``b^2 = p/q``) the c -> c0 limit at fixed h is not the block of ``L(h)``: it keeps finite admixtures of
the null submodule (at c = 1 from level ``2(m + 2)`` on).  ``recursion.H_q_series_kac_irreducible`` reads the
recursion on the Kac curve of the lowest null vector instead, where the quotient module is an analytic family.

References here are independent of the recursion: the exact Verma Gram matrix (its pseudo-inverse on the quotient
by the radical is the irreducible block), and the exact Fock couplings of the c = 1 free boson, which the vertex
correlator must reproduce with no admixture.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import factorial

import flint
import mpmath as mp
import pytest

from virasoro_blocks import F_block, G_block, H_q_series_regularized
from virasoro_blocks.recursion import H_q_series_kac_irreducible, H_q_series_laurent
from virasoro_blocks.requests import BlockPoleError


def _partitions(n, largest=None):
    largest = n if largest is None else largest
    if n == 0:
        yield ()
        return
    for k in range(min(n, largest), 0, -1):
        for rest in _partitions(n - k, k):
            yield (k,) + rest


def _q(x):
    return flint.fmpq(x.numerator, x.denominator)


def _gram_block(c, h, ext, levels):
    """z-series ``B_N`` of ``F z^{-(h - h1 - h2)}`` from the Verma Gram matrix at (c, h), exactly: the pseudo-inverse
    on the quotient by the radical (the irreducible block), checking that both 3-point vectors decouple it."""
    h1, h2, h3, h4 = ext

    @lru_cache(maxsize=None)
    def lower(k, st):
        if not st or k >= st[0]:
            return {(k,) + st: F(1)}
        n, rest = st[0], st[1:]
        out = {}
        for s1, v1 in lower(k, rest).items():
            for s2, v2 in lower(n, s1).items():
                out[s2] = out.get(s2, 0) + v1 * v2
        for s1, v1 in lower(k + n, rest).items():
            out[s1] = out.get(s1, 0) + (n - k) * v1
        return out

    @lru_cache(maxsize=None)
    def raise_(m, st):
        if not st:
            return {}
        n, rest = st[0], st[1:]
        out = {}
        for s1, v1 in raise_(m, rest).items():
            for s2, v2 in lower(n, s1).items():
                out[s2] = out.get(s2, 0) + v1 * v2
        if m > n:
            for s1, v1 in raise_(m - n, rest).items():
                out[s1] = out.get(s1, 0) + (m + n) * v1
        elif m < n:
            for s1, v1 in lower(n - m, rest).items():
                out[s1] = out.get(s1, 0) + (m + n) * v1
        else:
            out[rest] = out.get(rest, 0) + 2 * m * (h + sum(rest)) + c * F(m ** 3 - m, 12)
        return out

    def inner(lam, mu):
        st = {mu: F(1)}
        for k in lam:
            new = {}
            for s1, v1 in st.items():
                for s2, v2 in raise_(k, s1).items():
                    new[s2] = new.get(s2, 0) + v1 * v2
            st = new
        return st.get((), F(0))

    def vertex(lam, hL, hR):
        out = F(1)
        for i, k in enumerate(lam):
            out *= h + k * hR - hL + sum(lam[i + 1:])
        return out

    out = []
    for N in range(levels + 1):
        basis = list(_partitions(N))
        G = flint.fmpq_mat(len(basis), len(basis), [_q(inner(a, b)) for a in basis for b in basis])
        v12 = [vertex(a, h1, h2) for a in basis]
        v43 = [vertex(a, h4, h3) for a in basis]
        R, rank = G.rref()
        pivots = [next(col for col in range(len(basis)) if R[r, col] != 0) for r in range(rank)]
        sub = flint.fmpq_mat(rank, rank, [G[a, b] for a in pivots for b in pivots])
        y = sub.solve(flint.fmpq_mat(rank, 1, [_q(v43[a]) for a in pivots]))
        for a in range(len(basis)):          # v43 in the row space, v12 orthogonal to the radical
            assert sum((G[a, pivots[b]] * y[b, 0] for b in range(rank)), flint.fmpq(0)) == _q(v43[a])
        if rank < len(basis):
            x = sub.solve(flint.fmpq_mat(rank, 1, [_q(v12[a]) for a in pivots]))
            for a in range(len(basis)):
                assert sum((G[a, pivots[b]] * x[b, 0] for b in range(rank)), flint.fmpq(0)) == _q(v12[a])
        value = sum((_q(v12[pivots[b]]) * y[b, 0] for b in range(rank)), flint.fmpq(0))
        out.append(F(int(value.p), int(value.q)))
    return out


# ----------------------------------------------------------------------------- q-series helpers (exact thetas)
def _mul(a, b, n):
    out = [0] * (n + 1)
    for i, x in enumerate(a[: n + 1]):
        if x:
            for j, y in enumerate(b[: n + 1 - i]):
                out[i + j] += x * y
    return out


def _power(a, e, n):
    """``a^e`` for a series with ``a[0] = 1`` (``e`` rational), by ``(a^e)' a = e a' a^e``."""
    out = [mp.mpf(1)] + [mp.mpf(0)] * n
    for k in range(1, n + 1):
        out[k] = sum((e * j - (k - j)) * a[j] * out[k - j] for j in range(1, min(k, len(a) - 1) + 1)) / k
    return out


def _thetas(n):
    t3, t4, t2r = [F(0)] * (n + 1), [F(0)] * (n + 1), [F(0)] * (n + 1)
    t3[0] = t4[0] = F(1)
    k = 1
    while k * k <= n:
        t3[k * k] += 2
        t4[k * k] += 2 * (-1) ** k
        k += 1
    k = 0
    while k * (k + 1) <= n:                  # theta2 / (2 q^(1/4))
        t2r[k * (k + 1)] += 1
        k += 1
    return t2r, t3, t4


def _fl(x):
    return mp.mpf(x.numerator) / x.denominator if isinstance(x, F) else mp.mpf(x)


def _prefactor_q(c, h, ext, n):
    """``P(q)`` with ``F z^{-(h - h1 - h2)} = P(q) H(q)`` (library convention, ``chiral_norm = 1``), and ``z(q)``."""
    h1, h2, h3, h4 = (_fl(x) for x in ext)
    c, h = _fl(c), _fl(h)
    sigma = (c - 1) / 24
    t2r, t3, t4 = _thetas(n)
    t2r4 = _mul(_mul(t2r, t2r, n), _mul(t2r, t2r, n), n)
    t34 = _mul(_mul(t3, t3, n), _mul(t3, t3, n), n)
    t44 = _mul(_mul(t4, t4, n), _mul(t4, t4, n), n)
    inv34 = _power([_fl(x) for x in t34], -1, n)
    z_over_16q = _mul([_fl(x) for x in t2r4], inv34, n)            # z = 16 q t2r^4 / t3^4
    one_minus_z = _mul([_fl(x) for x in t44], inv34, n)
    pref = _power(z_over_16q, -(h - sigma), n)                      # (16 q / z)^(h - sigma)
    pref = _mul(pref, _power(one_minus_z, sigma - h2 - h3, n), n)
    pref = _mul(pref, _power([_fl(x) for x in t3], (c - 1) / 2 - 4 * (h1 + h2 + h3 + h4), n), n)
    return pref, z_over_16q


def _gram_H(c, h, ext, levels):
    """The Gram block as library H coefficients (``F z^{-(h-h1-h2)} = P(q) sum_k H_k q^k``), composed in q (at 30
    more digits than the caller's: the composition cancels ``16^k`` growth)."""
    with mp.workdps(mp.mp.dps + 30):
        return [+x for x in _gram_H_at_work(c, h, ext, levels)]


def _gram_H_at_work(c, h, ext, levels):
    B = [_fl(x) for x in _gram_block(c, h, ext, levels)]
    pref, z16 = _prefactor_q(c, h, ext, levels)
    # B(z(q)) with z = 16 q z16(q): powers of 16 q carried as q^N 16^N
    total, zpow = [mp.mpf(0)] * (levels + 1), [mp.mpf(1)] + [mp.mpf(0)] * levels
    for N in range(levels + 1):
        if N:
            zpow = [0] + _mul(zpow, z16, levels)[:levels]
            zpow = [16 * x for x in zpow]
        for k in range(levels + 1):
            total[k] += B[N] * zpow[k]
    return _mul(total, _power(pref, -1, levels), levels)


def _rel(a, b):
    return max(abs(x - y) / max(1, abs(y)) for x, y in zip(a, b))


# ----------------------------------------------------------------------------- against the Verma Gram matrix
GRAM_CASES = [
    # (c, h, externals, levels, first): c = 1, h = m^2/4 for m = 0..4 (even m: any equal pair; odd m: the
    # fusion-compatible q^2/16), generic / twist / degenerate / unequal externals; c = 25 and a same-level collision at
    # b^2 = 2 (c = 28).  ``first``: the level where the c -> c0 limit at fixed h leaves the irreducible block (None:
    # it does not within ``levels`` -- the vacuum; the su(2) currents h_ext = 1, whose coupling to L(16) vanishes)
    (1, F(0), (F(25, 16),) * 4, 10, None),
    (1, F(1, 4), (F(1, 16),) * 4, 9, 6),
    (1, F(1), (F(25, 16),) * 4, 10, 8),
    (1, F(1), (F(1, 9), F(1, 9), F(1, 36), F(1, 36)), 9, 8),
    (1, F(9, 4), (F(9, 16),) * 4, 11, 10),
    (1, F(1, 4), (F(1, 9), F(1, 36), F(1, 36), F(1, 9)), 8, 6),
    (1, F(4), (F(1),) * 4, 12, None),
    (25, F(-3), (F(1, 3),) * 4, 8, 4),
    (28, F(-27, 8), (F(35, 32),) * 4, 8, 4),
]


@pytest.mark.parametrize("c, h, ext, levels, first", GRAM_CASES)
def test_irreducible_block_is_the_gram_quotient_block(c, h, ext, levels, first):
    with mp.workdps(30):
        exact = _gram_H(F(c), h, ext, levels)
        irr = H_q_series_kac_irreducible(h, ext, c0=c, q_order=levels)
        lim = H_q_series_laurent(h, ext, c0=c, q_order=levels)
        assert _rel(irr, exact) < mp.mpf(10) ** -24
        # the c -> c0 limit at fixed h: the irreducible block below ``first``, not at it
        top = levels + 1 if first is None else first
        assert _rel(lim[:top], exact[:top]) < mp.mpf(10) ** -24
        if first is not None:
            assert abs(lim[first] - exact[first]) > mp.mpf(10) ** -12


@pytest.mark.parametrize("c, h, ext, levels", [
    (F(1, 2), F(1, 16), (F(1, 16), F(1, 2), F(1, 2), F(1, 16)), 9),   # Ising, braid weight, moving externals
    (F(7, 10), F(3, 5), (F(1, 10), F(1, 10), F(3, 5), F(3, 5)), 9),   # tricritical Ising, unequal sides
    (F(-22, 5), F(-1, 5), (F(-1, 5),) * 4, 9),                         # Lee-Yang
])
def test_minimal_models_irreducible_equals_the_c_limit(c, h, ext, levels):
    with mp.workdps(30):
        exact = _gram_H(c, h, ext, levels)
        irr = H_q_series_kac_irreducible(h, ext, c0=c, q_order=levels)
        lim = H_q_series_laurent(h, ext, c0=c, q_order=levels)
        assert _rel(irr, exact) < mp.mpf(10) ** -24
        assert _rel(lim, exact) < mp.mpf(10) ** -24


# ----------------------------------------------------------------------------- c = 1 free boson: no admixture
def _counts(lam):
    out = {}
    for p in lam:
        out[p] = out.get(p, 0) + 1
    return out


def _fock_norm(lam):
    value = F(1)
    for n, k in _counts(lam).items():
        value *= F(n, 2) ** k * factorial(k)
    return value


def _lower_osc(n, lam):
    k = _counts(lam).get(n, 0)
    if not k:
        return {}
    rest = list(lam)
    rest.remove(n)
    return {tuple(rest): F(n, 2) * k}


def _virasoro_lowering(n, lam, q):
    out = {}
    for key, v in _lower_osc(n, lam).items():
        out[key] = out.get(key, 0) + 2 * q * v
    for l in range(1, n):
        for k1, v1 in _lower_osc(l, lam).items():
            for k2, v2 in _lower_osc(n - l, k1).items():
                out[k2] = out.get(k2, 0) + v1 * v2
    for l in range(1, sum(lam) + 1):
        for k1, v1 in _lower_osc(n + l, lam).items():
            key = tuple(sorted(k1 + (l,), reverse=True))
            out[key] = out.get(key, 0) + 2 * v1
    return out


def _fock_coupling(w_a, w_b, k):
    """Exact ``|<k, m| V_{w_a}(1) |-w_b/2>|^2 / <k, m|k, m>`` (``m = (w_a - w_b)/2``; the Virasoro primary ``|k, m>`` is
    the kernel of ``L_1, L_2`` at level ``k^2 - m^2`` of the Fock module of momentum ``m``; ``b = a / sqrt 2``)."""
    m = (w_a - w_b) / 2
    level = int(k * k - m * m)
    basis = list(_partitions(level))
    rows = []
    for n in (1, 2):
        if level < n:
            continue
        target = {lam: i for i, lam in enumerate(_partitions(level - n))}
        block = [[F(0)] * len(basis) for _ in target]
        for col, lam in enumerate(basis):
            for key, v in _virasoro_lowering(n, lam, m).items():
                block[target[key]][col] += v
        rows.extend(block)
    if rows:
        R, rank = flint.fmpq_mat(len(rows), len(basis), [_q(v) for row in rows for v in row]).rref()
        pivots = [next(col for col in range(len(basis)) if R[r, col] != 0) for r in range(rank)]
        free = [col for col in range(len(basis)) if col not in set(pivots)]
        assert len(free) == 1
        vec = {free[0]: flint.fmpq(1)}
        vec.update({col: -R[r, free[0]] for r, col in enumerate(pivots)})
        primary = {basis[col]: F(int(v.p), int(v.q)) for col, v in vec.items() if v != 0}
    else:
        primary = {(): F(1)}
    overlap = F(0)
    for lam, coefficient in primary.items():
        coherent = F(1)
        for n, count in _counts(lam).items():
            coherent *= (w_a / n) ** count / factorial(count)
        overlap += coefficient * coherent * _fock_norm(lam)
    return overlap ** 2 / sum(v * v * _fock_norm(lam) for lam, v in primary.items())


def _peel_vertex_correlator(w_a, w_b, top, kac_limit):
    """Coefficients of the c = 1 blocks at ``h = j^2`` in the chiral vertex correlator of the pair: neutral
    ``x^{-2h_b} (1-x)^{-w_a w_b/2}`` (externals ``h_b, h_b, h_a, h_a``) or charged ``x^{-w_a w_b/2} (1-x)^{-w_a^2/2}``
    (``h_b, h_a, h_a, h_b``, s-channel from ``h = m^2``), peeled in q with the library's blocks."""
    m = (w_a - w_b) / 2
    h_a, h_b = w_a * w_a / 4, w_b * w_b / 4
    ext = (h_b, h_b, h_a, h_a) if m == 0 else (h_b, h_a, h_a, h_b)
    expo = -w_a * w_b / 2 if m == 0 else -w_a * w_a / 2
    n = int(top * top - m * m)
    # F_h z^{h1 + h2} = (16 q)^h P0(q) H_h(q) (c = 1, P0 the h = 0 prefactor) and z^{m^2} = (16 q)^{m^2} z16^{m^2}:
    # X = z16^{m^2} (1 - z)^expo / P0 = sum_j c_j 16^{j^2 - m^2} q^{j^2 - m^2} H_j(q)
    P0, z16 = _prefactor_q(1, 0, ext, n)
    _t2r, t3, t4 = _thetas(n)
    t34 = _mul(_mul(t3, t3, n), _mul(t3, t3, n), n)
    t44 = _mul(_mul(t4, t4, n), _mul(t4, t4, n), n)
    one_minus_z = _mul([_fl(x) for x in t44], _power([_fl(x) for x in t34], -1, n), n)
    X = _mul(_mul(_power(one_minus_z, expo, n), _power(z16, m * m, n), n), _power(P0, -1, n), n)
    coeffs, spins, k = {}, [], abs(m)
    while k <= top:
        spins.append(k)
        k += 1
    res = list(X)
    for j in spins:
        off = int(j * j - m * m)
        H = H_q_series_regularized(j * j, ext, c=1, q_order=n - off, kac_limit=kac_limit)
        cj = res[off] / mp.mpf(16) ** off
        coeffs[j] = cj
        for i, hk in enumerate(H):
            res[off + i] -= cj * mp.mpf(16) ** off * hk
    return coeffs


@pytest.mark.parametrize("w_a, w_b, top", [
    (F(1, 3), F(1, 3), F(3)),          # neutral, spins 0..3 (Fock level 9)
    (F(1, 3), F(7, 3), F(3)),          # neutral, unequal windings
    (F(1, 3), F(-2, 3), F(5, 2)),      # charge 1/2, spins 1/2..5/2 (level 6)
    (F(1, 2), F(-3, 2), F(3)),         # charge 1, spins 1..3 (level 8)
])
def test_free_boson_vertex_correlator_has_no_admixture_in_irreducible_blocks(w_a, w_b, top):
    """Peeled with irreducible blocks, the vertex correlator's coefficients are the exact Fock couplings to the
    Virasoro primaries |j, m> -- so a pair rotated by beta weights them by the Wigner d^j(beta) (Legendre P_j for a
    neutral pair) with no admixture.  The c -> c0 pipeline blocks need p_k / c_k = 1 - sum kappa (kappa_{1,2} = 1/6
    neutral, kappa_{1/2,2}(1/2) = 1/9, kappa_{1,2}(1) = 9/64)."""
    with mp.workdps(40):
        m = (w_a - w_b) / 2
        irr = _peel_vertex_correlator(w_a, w_b, top, "irreducible")
        lim = _peel_vertex_correlator(w_a, w_b, top, "c_limit")
        for j, value in irr.items():
            exact = _fock_coupling(w_a, w_b, j)
            assert abs(value / _fl(exact) - 1) < mp.mpf(10) ** -25, j
        k = abs(m) + 2 if m else 3         # the first spin a two-step admixture reaches (kappa_{0,2} = 0)
        kappa = (2 * (k - 2) + 1) * ((k - 1) ** 2 - m * m) / (4 * (k - 1) * (k * k - m * m))
        assert abs(lim[k] / _fl(_fock_coupling(w_a, w_b, k)) - (1 - _fl(kappa))) < mp.mpf(10) ** -25
        assert kappa != 0


# ----------------------------------------------------------------------------- refusals and the value paths
def test_irreducible_block_refuses_externals_that_do_not_decouple_the_null_vector():
    with mp.workdps(20):
        with pytest.raises(BlockPoleError):          # m = 1 needs h_ext = q^2/16 with q odd <= 1
            H_q_series_kac_irreducible(F(1, 4), [F(25, 16)] * 4, c0=1, q_order=6)
        with pytest.raises(BlockPoleError):          # h1 != h2 on the even m = 2 weight, no fusion factor vanishes
            H_q_series_kac_irreducible(F(1), [F(2, 7), F(1, 3), F(1, 3), F(2, 7)], c0=1, q_order=6)
        with pytest.raises(ValueError):
            H_q_series_regularized(F(1), [F(25, 16)] * 4, c=1, q_order=6, kac_limit="fock")


def test_value_paths_thread_the_irreducible_kac_limit():
    """The irreducible block is the default of every value path on a Kac weight; ``kac_limit="c_limit"`` (explicit, or
    from :func:`kac_limit_context`) is the c -> c0 limit at fixed h."""
    from virasoro_blocks.recursion import DEFAULT_KAC_LIMIT, kac_limit_context, kac_limit_mode

    assert DEFAULT_KAC_LIMIT == kac_limit_mode() == "irreducible"
    with pytest.raises(ValueError):
        kac_limit_mode("fock")
    with mp.workdps(25):
        he = F(25, 16)
        ext = [he] * 4
        irr = H_q_series_kac_irreducible(F(1), ext, c0=1, q_order=12)
        lim = H_q_series_laurent(F(1), ext, c0=1, q_order=12)
        assert _rel(H_q_series_regularized(F(1), ext, c=1, q_order=12), irr) < 1e-22
        assert _rel(H_q_series_regularized(F(1), ext, c=1, q_order=12, kac_limit="c_limit"), lim) < 1e-22
        with kac_limit_context("c_limit"):
            assert kac_limit_mode() == "c_limit"
            assert _rel(H_q_series_regularized(F(1), ext, c=1, q_order=12), lim) < 1e-22
        assert kac_limit_mode() == "irreducible"
        z = mp.mpf("0.3")
        kw = dict(c=1, q_order=12, z_shift=-2 * he, chiral_norm=1)
        f_irr = F_block(F(1), z, ext, **kw)
        f_lim = F_block(F(1), z, ext, kac_limit="c_limit", **kw)
        assert f_irr == F_block(F(1), z, ext, kac_limit="irreducible", **kw)
        # the two differ by the level-8 admixture: (lim - irr)_8 z^8 times the prefactor, to leading order
        assert abs(lim[8] - irr[8]) > 1e-8
        assert abs(f_irr - f_lim) > mp.mpf(10) ** -9 and abs(f_irr - f_lim) < mp.mpf(10) ** -5
        # off a Kac weight, and at the vacuum, the option changes nothing
        for h in (F(7, 5), F(0)):
            assert F_block(h, z, ext, kac_limit="c_limit", **kw) == F_block(h, z, ext, **kw)
        g_irr = G_block(2, 0, z, z / 2, ext, c=1, q_order=12)
        g_lim = G_block(2, 0, z, z / 2, ext, c=1, q_order=12, kac_limit="c_limit")
        assert 0 < abs(g_irr - g_lim) < mp.mpf(10) ** -8
