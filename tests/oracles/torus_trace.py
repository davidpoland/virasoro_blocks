"""First-principles oracle for the torus one-point block: direct Verma trace.

Validates `virasoro_blocks.torus_onept` (the CCY 1703.09805 eq. 4.27
h-recursion) against the *definition* of the block, CCY eq. 4.26:

    [q^n] F(q, h, d, c) = sum_{|A|=|B|=n} (G_n^{-1})^{AB} rho(L_{-B} h, d, L_{-A} h)

computed by raw Virasoro algebra: PBW-basis states L_{-a_1}...L_{-a_k}|h>
(a_1 >= a_2 >= ...), the Shapovalov Gram matrix, and primary matrix elements
<L_{-A} h| V_d(1) |L_{-B} h> obtained by commuting generators through V one
at a time. No recursion input is shared with the kernel under test (only the
Virasoro commutator and [L_m, V(z)] = z^m (z d/dz + (m+1) d) V(z)), so an
agreement is a genuine two-sided check.

The bilinear (Shapovalov) pairing is used without complex conjugation, so
complex c / complex weights are fine -- the block is an analytic object.

Everything here is exact linear algebra at working precision, but the state
space grows as p(n), so keep levels <= ~6. That is plenty: every residue
family (r, s) with rs <= level is exercised.

Key derivative identity used to close the matrix-element recursion on V(1):
for L_0-eigenstates, <alpha| V(z) |beta> = z^{E_alpha - E_beta - d} <alpha|
V(1) |beta>, hence <alpha| dV(1) |beta> = (E_alpha - E_beta - d) <alpha| V(1)
|beta>.
"""

from __future__ import annotations

import mpmath as mp


def partitions(n: int, cap: int | None = None) -> list:
    """Partitions of n in non-increasing order (the PBW index set)."""
    if n == 0:
        return [()]
    if cap is None:
        cap = n
    out = []
    for first in range(min(n, cap), 0, -1):
        for rest in partitions(n - first, first):
            out.append((first,) + rest)
    return out


def apply_L(m: int, part: tuple, h, c) -> dict:
    """L_m acting on the PBW state L_{-part}|h>, as {partition: coeff}."""
    if part == ():
        if m > 0:
            return {}
        if m == 0:
            return {(): h}
        return {(-m,): mp.mpf(1)}
    a = part[0]
    rest = part[1:]
    if m < 0 and -m >= a:
        return {(-m,) + part: mp.mpf(1)}
    out: dict = {}

    def _add(d: dict, scale) -> None:
        for key, val in d.items():
            out[key] = out.get(key, mp.mpf(0)) + scale * val

    # L_m L_{-a} = L_{-a} L_m + (m + a) L_{m-a} + (c/12) m (m^2 - 1) delta_{m,a}
    for p, coef in apply_L(m, rest, h, c).items():
        for p2, coef2 in apply_L(-a, p, h, c).items():
            out[p2] = out.get(p2, mp.mpf(0)) + coef * coef2
    _add(apply_L(m - a, rest, h, c), mp.mpf(m + a))
    if m == a:
        _add({rest: mp.mpf(1)}, mp.mpf(c) / 12 * m * (m * m - 1))
    return out


def shapovalov_inner(A: tuple, B: tuple, h, c):
    """Bilinear <L_{-A} h, L_{-B} h> via successive raising operators."""
    state = {B: mp.mpf(1)}
    for a in A:  # transpose of L_{-a_1}...L_{-a_k} applies a_1 first
        new: dict = {}
        for p, coef in state.items():
            for p2, coef2 in apply_L(a, p, h, c).items():
                new[p2] = new.get(p2, mp.mpf(0)) + coef * coef2
        state = new
    return state.get((), mp.mpf(0))


def primary_matrix_elements(h, c, d_ext):
    """Memoized M(A, B) = <L_{-A} h| V_d(1) |L_{-B} h> / <h|V_d(1)|h>."""
    memo: dict = {}

    def level(part: tuple) -> int:
        return sum(part)

    def M(A: tuple, B: tuple):
        key = (A, B)
        if key in memo:
            return memo[key]
        if A == () and B == ():
            memo[key] = mp.mpf(1)
        elif B != ():
            # V L_{-b} = L_{-b} V - [L_{-b}, V];  [L_{-b}, V(1)] =
            # (1 - b) d V(1) + dV(1), and dV -> (E_A - E_B' - d) V.
            b = B[0]
            Bp = B[1:]
            acc = (b * d_ext - level(A) + level(Bp)) * M(A, Bp)
            for p, coef in apply_L(b, A, h, c).items():
                acc += coef * M(p, Bp)
            memo[key] = acc
        else:
            # mirror step removing from the bra
            a = A[0]
            Ap = A[1:]
            acc = (a * d_ext + level(Ap) - level(B)) * M(Ap, B)
            for p, coef in apply_L(a, B, h, c).items():
                acc += coef * M(Ap, p)
            memo[key] = acc
        return memo[key]

    return M


def torus_onept_F_coeffs_oracle(h, c, d_ext, n_max: int) -> list:
    """[q^n] F(q, h, d, c) for n = 0..n_max by direct inverse-Gram trace."""
    M = primary_matrix_elements(h, c, d_ext)
    out = [mp.mpf(1)]
    for n in range(1, n_max + 1):
        parts = partitions(n)
        k = len(parts)
        G = mp.matrix(k, k)
        Mm = mp.matrix(k, k)
        for i, A in enumerate(parts):
            for j, B in enumerate(parts):
                G[i, j] = shapovalov_inner(A, B, h, c)
                Mm[i, j] = M(A, B)
        Ginv = G**-1
        trace = mp.mpf(0)
        for i in range(k):
            for j in range(k):
                trace += Ginv[i, j] * Mm[j, i]
        out.append(trace)
    return out


def torus_onept_f_coeffs_oracle(h, c, d_ext, n_max: int) -> list:
    """[q^n] f = F * prod_{k>=1} (1 - q^k): the recursion object of eq. 4.27."""
    F = torus_onept_F_coeffs_oracle(h, c, d_ext, n_max)
    prod = [mp.mpf(1)] + [mp.mpf(0)] * n_max
    for k in range(1, n_max + 1):
        new = list(prod)
        for i in range(0, n_max + 1 - k):
            new[i + k] -= prod[i]
        prod = new
    out = [mp.mpf(0)] * (n_max + 1)
    for i in range(n_max + 1):
        for j in range(n_max + 1 - i):
            out[i + j] += F[i] * prod[j]
    return out


def torus_onept_F_coeffs_oracle_irreducible(
    h, c, d_ext, n_max: int, *, rank_tol=None
) -> list:
    """[q^n] of the normalized one-point trace over the IRREDUCIBLE module.

    Same inverse-Gram trace as `torus_onept_F_coeffs_oracle`, but over the
    quotient of the Verma module by its null submodule: at each level a
    maximal subset of PBW states with invertible Gram restriction is
    selected greedily (residual-pivot test), and the trace runs over that
    subset. Null states are orthogonal to everything, so any complement
    basis computes the quotient trace exactly.

    This is what a degenerate internal primary (e.g. Ising sigma at
    h = 1/16, c = 1/2) contributes to a physical one-point function; the
    Verma trace literally diverges there. For non-degenerate (h, c) every
    state is kept and this reduces to the plain oracle.
    """
    if rank_tol is None:
        rank_tol = mp.mpf(10) ** (-max(2, mp.mp.dps // 2))
    M = primary_matrix_elements(h, c, d_ext)
    out = [mp.mpf(1)]
    for n in range(1, n_max + 1):
        parts = partitions(n)
        selected: list = []
        G_rows: list = []
        Ginv = None
        for cand in parts:
            g_cand = [shapovalov_inner(A, cand, h, c) for A in selected]
            g_self = shapovalov_inner(cand, cand, h, c)
            if selected:
                proj = mp.mpf(0)
                v = Ginv * mp.matrix(g_cand)
                for i in range(len(selected)):
                    proj += g_cand[i] * v[i]
                residual = g_self - proj
                scale = max(abs(g_self), abs(proj), mp.mpf(1))
            else:
                residual = g_self
                scale = max(abs(g_self), mp.mpf(1))
            if abs(residual) <= rank_tol * scale:
                continue  # null direction: quotiented away
            selected.append(cand)
            k = len(selected)
            G = mp.matrix(k, k)
            for i, A in enumerate(selected):
                for j, B in enumerate(selected):
                    G[i, j] = shapovalov_inner(A, B, h, c)
            Ginv = G**-1
        if not selected:
            out.append(mp.mpf(0))
            continue
        k = len(selected)
        Mm = mp.matrix(k, k)
        for i, A in enumerate(selected):
            for j, B in enumerate(selected):
                Mm[i, j] = M(A, B)
        trace = mp.mpf(0)
        for i in range(k):
            for j in range(k):
                trace += Ginv[i, j] * Mm[j, i]
        out.append(trace)
    return out


if __name__ == "__main__":
    from virasoro_blocks.torus_onept import (
        build_torus_onept_kernel,
        torus_onept_f_coeffs,
    )

    mp.mp.dps = 50
    n_max = 5
    # NB c = 26 looks innocuous but is the b^2 = 2/3 singular locus (Kac
    # collisions blow up the kernel at level 5); the kernel now rejects it.
    for c, h, d_ext in [
        (mp.mpf("32.7"), mp.mpf("1.37"), mp.mpf("0.83")),
        (mp.mpf("9.4"), mp.mpf("2.11"), mp.mpf("1.3")),
        (mp.mpf("27.9"), mp.mpf("0.61"), mp.mpf("2.7")),
    ]:
        oracle = torus_onept_f_coeffs_oracle(h, c, d_ext, n_max)
        kernel = build_torus_onept_kernel(n_max, d_ext=d_ext, c=c)
        recur = torus_onept_f_coeffs(kernel, h)
        err = max(
            abs(mp.mpc(a) - mp.mpc(b)) / max(abs(mp.mpc(a)), mp.mpf("1e-30"))
            for a, b in zip(oracle, recur)
        )
        print(
            f"c={mp.nstr(c, 6)} h={mp.nstr(h, 6)} d={mp.nstr(d_ext, 6)}: "
            f"max rel err {mp.nstr(err, 3)}"
        )
