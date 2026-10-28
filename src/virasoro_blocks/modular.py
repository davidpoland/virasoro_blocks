"""Exact genus-1 (torus) reduced-character derivative tables.

The modular-bootstrap analogue of the sphere 4-point block tables. Everything
here is exact -- no recursion,
no q-truncation, no Kac poles, no Delta-interpolation.

Conventions:

    q_tau  = e^{2 pi i tau}          (modular-form convention; NOT the sphere
                                      elliptic nome q = e^{i pi tau})
    hhat    = h    - (c - 1)/24
    hbarhat = hbar - (c - 1)/24
    Delta   = h + hbar,  j = h - hbar in Z,  unitarity Delta >= |j|

Reduced partition function Zhat = sqrt(tau_2) |eta|^2 Z; the eta's cancel
against the characters, so a primary (h, hbar) contributes

    sqrt(tau_2) * q_tau^{hhat} * qbar_tau^{hbarhat}

with an extra (1 - q_tau)(1 - qbar_tau) for the vacuum. Crossing is
S-invariance, Zhat(tau) = Zhat(-1/tau).

Crossing coordinate: the disk variable

    v = (tau - i)/(tau + i),   tau(v) = i (1 + v)/(1 - v)

('vvb' in coordinate-menu terms; `x, t, w, s` are taken by the folded sphere
coordinates and by the writer's polynomial variable x = Delta - shift).
S acts as (v, vbar) -> (-v, -vbar), so crossing functionals keep m + n odd.
The antichiral map is taubar(vbar) = -i (1 + vbar)/(1 - vbar), the complex
conjugate of tau(v) on the Euclidean section vbar = v*.

Chiral building block: with 2 pi i tau(v) = -2 pi (1 + v)/(1 - v),

    q_tau^{a} = e^{-2 pi a} * exp(a g(v)),   g(v) = -4 pi v / (1 - v),

so the v-Taylor coefficients of q_tau^a are e^{-2 pi a} * C_p(a) with C_p an
exact degree-p polynomial in a (C_0 = 1). The antichiral side is identical
with a = hbarhat. sqrt(tau_2) = sqrt(1 + A(v) + A(vbar)) with
A(u) = u/(1 - u) is the one non-factorized piece; its bivariate Taylor table
is exact and shared by every entry.

The per-spin (m, n) table entry is then an exact polynomial in Delta of
degree m + n, times the prefactor e^{-2 pi (Delta - shift)}: precisely the
`SerializedDerivTable` form with base = e^{-2 pi} and an EMPTY pole list.

Tables follow the repo-wide Taylor-coefficient convention: entry (m, n) is
the series coefficient of v^m vbar^n (the 1/(m! n!) is folded in), so
products convolve without binomial weights.
"""

from __future__ import annotations

from fractions import Fraction
import mpmath as mp

from .polynomial import Polynomial
from .numeric_input import (
    ExactRational, literal_components, real_input_label,
)
from .requests import _integer_field
from .serialization import SerializedDerivTable

TORUS_COORDINATE = "vvb"
TORUS_BASIS = "torus_G"

# Chiral coefficients depend on lam and working bit precision; tau_2 powers
# also depend on the exact exponent label. Both are shared across spin sectors and
# central charges, with numerical entries computed at the ambient precision.
_SQRT_TAU2_CACHE: dict = {}
_CHIRAL_COEFF_CACHE: dict = {}


def torus_disk_point(v, vbar):
    """``(tau, taubar, tau2, q, qbar)`` at the disk point ``(v, vbar)`` (at the current precision):
    tau = i (1 + v)/(1 - v), taubar = -i (1 + vbar)/(1 - vbar), q = e^{2 pi i tau}, qbar = e^{-2 pi i taubar}."""
    one = mp.mpf(1)
    tau = mp.mpc(0, 1) * (one + v) / (one - v)
    taubar = mp.mpc(0, -1) * (one + vbar) / (one - vbar)
    tau2 = (tau - taubar) / mp.mpc(0, 2)
    q = mp.exp(2 * mp.pi * mp.mpc(0, 1) * tau)
    qbar = mp.exp(-2 * mp.pi * mp.mpc(0, 1) * taubar)
    return tau, taubar, tau2, q, qbar


def torus_exp_base() -> mp.mpf:
    """The torus damping base e^{-2 pi}: |q_tau| at the self-dual point."""
    return mp.exp(-2 * mp.pi)


def _chiral_coeff_polys(n_terms: int) -> list[list]:
    """Coefficient lists (in a) of C_p(a): exp(a g(v)) = sum_p C_p(a) v^p.

    g(v) = -4 pi v/(1 - v). From E' = a g' E with g'(v) = -4 pi sum (i+1) v^i:

        C_{p+1}(a) = a/(p+1) * sum_{i=0}^{p} (-4 pi) (i+1) C_{p-i}(a).

    Returns plain ascending mpf coefficient lists; C_p has degree exactly p.
    """
    n_terms = _integer_field("n_terms", n_terms)
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    key = (n_terms, mp.mp.prec)
    cached = _CHIRAL_COEFF_CACHE.get(key)
    if cached is not None:
        return cached
    four_pi = 4 * mp.pi
    C = [[mp.mpf(1)]]
    for p in range(n_terms - 1):
        acc = [mp.mpf(0)] * (p + 2)  # degree p+1 in a
        for i in range(p + 1):
            w = -four_pi * (i + 1) / (p + 1)
            src = C[p - i]
            # multiply src by (w * a): shift degree by one
            for k, coeff in enumerate(src):
                acc[k + 1] += w * coeff
        C.append(acc)
    _CHIRAL_COEFF_CACHE[key] = C
    return C


def _eval_coeff_polys(C: list[list], a) -> list:
    """Evaluate each C_p at a numeric a (Horner), returning [C_p(a)]."""
    out = []
    for coeffs in C:
        r = mp.mpf(0)
        for coeff in reversed(coeffs):
            r = r * a + coeff
        out.append(r)
    return out


def _tau2_power_exact(lam, real, imag):
    """Resolve sensitive coefficients using exact complex rational arithmetic.

    tau_2 = (1 - v*vbar)/((1-v)*(1-vbar)); its coefficient is
    sum_k (-1)^k binom(a,k) (a)_{m-k}/(m-k)! (a)_{n-k}/(n-k)!.
    This avoids cancellation between floating binomial terms near integers.
    """
    def multiply(left, right):
        a, b = left
        c, d = right
        return a*c-b*d, a*d+b*c

    rising = [(Fraction(1), Fraction(0))]
    choose = rising.copy()
    for k in range(1, lam+1):
        rising.append(tuple(x/k for x in multiply(rising[-1], (real+k-1, imag))))
        choose.append(tuple(x/k for x in multiply(choose[-1], (real-k+1, imag))))
    out = []
    for m in range(lam+1):
        row = []
        for n in range(lam+1):
            re = im = Fraction(0)
            for k in range(min(m, n)+1):
                term = multiply(choose[k], multiply(rising[m-k], rising[n-k]))
                re += (-1)**k*term[0]
                im += (-1)**k*term[1]
            value = mp.mpf(ExactRational(re))
            if imag != 0:
                value = mp.mpc(value, mp.mpf(ExactRational(im)))
            row.append(value)
        out.append(row)
    return out


def tau2_power_table(lam: int, exponent) -> list[list]:
    """Bivariate Taylor table of tau_2^exponent at v = vbar = 0.

    tau_2 = Im tau on the Euclidean section extends to independent (v, vbar)
    as (tau(v) - taubar(vbar))/(2i) = 1 + u with u = A(v) + A(vbar),
    A(w) = w/(1-w). The general power is the truncated binomial sum
    (1 + u)^a = sum_k binom(a, k) u^k; u has valuation 1 in each variable's
    total grading, so k <= 2*lam terms contribute. Used with a = 1/2 for the
    reduced partition function and a = (Delta_Phi + 1)/2 for the
    reduced torus one-point function.
    """
    lam = _nonnegative_order(lam)
    real, imag = literal_components(exponent)
    real = real_input_label(real, name="exponent real part")
    imag = real_input_label(imag, name="exponent imaginary part")
    key = (lam, real, imag, mp.mp.prec)
    cached = _SQRT_TAU2_CACHE.get(key)
    if cached is not None:
        return [list(row) for row in cached]
    if isinstance(real, Fraction) and isinstance(imag, Fraction):
        separation2 = (real-round(real))**2 + imag**2
        if separation2 < Fraction(1, 10000) and (separation2 != 0 or real < 0):
            out = _tau2_power_exact(lam, real, imag)
            _SQRT_TAU2_CACHE[key] = tuple(tuple(row) for row in out)
            return out
    n = lam + 1
    zero = mp.mpf(0)
    # u[a][b]: 1 on each axis (a >= 1, b = 0) and (a = 0, b >= 1), else 0.
    u = [[zero] * n for _ in range(n)]
    for k in range(1, n):
        u[k][0] = mp.mpf(1)
        u[0][k] = mp.mpf(1)
    out = [[zero] * n for _ in range(n)]
    out[0][0] = mp.mpf(1)
    # power = u^k built iteratively; binom(a, k) via the falling-factorial
    # recurrence to stay exact for non-integer a.
    power = None
    binom = mp.mpf(1)
    for k in range(1, 2 * lam + 1):
        # Subtract from the supplied components before numerical conversion:
        # a = 1 + eps must not terminate at k=2 just because eps < 10^-dps.
        factor = mp.mpf(real - (k - 1))
        if imag != 0:
            factor = mp.mpc(factor, mp.mpf(imag))
        binom = binom * factor / k
        if power is None:
            power = [row[:] for row in u]
        else:
            new = [[zero] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    coeff = power[a][b]
                    if coeff == 0:
                        continue
                    # multiply by u: shift along each axis
                    for kk in range(1, n - a):
                        new[a + kk][b] += coeff
                    for kk in range(1, n - b):
                        new[a][b + kk] += coeff
            power = new
        if binom == 0:
            break
        for a in range(n):
            row_p = power[a]
            row_o = out[a]
            for b in range(n):
                if row_p[b] != 0:
                    row_o[b] += binom * row_p[b]
    _SQRT_TAU2_CACHE[key] = tuple(tuple(row) for row in out)
    return out


def sqrt_tau2_table(lam: int) -> list[list]:
    """Bivariate Taylor table S[a][b] of sqrt(tau_2) at v = vbar = 0."""
    return tau2_power_table(lam, mp.mpf(1) / 2)


def _nonnegative_order(lam):
    lam = _integer_field("lam", lam)
    if lam < 0:
        raise ValueError("lam must be >= 0")
    return lam


def _one_minus_q_series(lam: int, C: list[list]) -> list:
    """Taylor coefficients of 1 - q_tau(v): the vacuum null-state factor."""
    base = torus_exp_base()
    q_series = [base * val for val in _eval_coeff_polys(C, mp.mpf(1))]
    out = [-coeff for coeff in q_series]
    out[0] += mp.mpf(1)
    return out[: lam + 1]


def _convolve(a: list, b: list, n_terms: int) -> list:
    out = [mp.mpf(0)] * n_terms
    for i, ai in enumerate(a[:n_terms]):
        if ai == 0:
            continue
        for j, bj in enumerate(b[: n_terms - i]):
            out[i + j] += ai * bj
    return out


def torus_vacuum_table(c, lam: int) -> list[list]:
    """Exact numeric (m, n) Taylor table of the reduced vacuum character.

    sqrt(tau_2) * q_tau^{a0} (1 - q_tau) * qbar_tau^{a0} (1 - qbar_tau) with
    a0 = -(c - 1)/24. This is the identity-vector source for the crossing
    equation. Entries are the actual table values (prefactor included);
    there is no Delta dependence.
    """
    c = real_input_label(c, name="c")
    lam = _nonnegative_order(lam)
    return _torus_vacuum_table(c, lam)


def _torus_vacuum_table(c, lam):
    n = lam + 1
    C = _chiral_coeff_polys(n)
    a0 = mp.mpf(-(c - 1) / 24)
    scale = mp.exp(-2 * mp.pi * a0)
    chi = [scale * val for val in _eval_coeff_polys(C, a0)]
    chi = _convolve(chi, _one_minus_q_series(lam, C), n)
    S = sqrt_tau2_table(lam)
    table = [[mp.mpf(0)] * n for _ in range(n)]
    for m in range(n):
        for nn in range(n):
            acc = mp.mpf(0)
            for a in range(m + 1):
                row = S[a]
                for b in range(nn + 1):
                    s_ab = row[b]
                    if s_ab == 0:
                        continue
                    acc += s_ab * chi[m - a] * chi[nn - b]
            table[m][nn] = acc
    return table


def torus_current_table(c, spin: int, lam: int) -> list[list]:
    """Exact numeric (m, n) Taylor table of the reduced character of a chiral current pair.

    A primary (h, hbar) = (j, 0) is a current: its antichiral half is the vacuum module, whose character carries
    the null-state factor (1 - qbar_tau). The pair (j, 0) + (0, j) contributes

        sqrt(tau_2) * [q^{jhat} qbar^{a0} (1 - qbar) + q^{a0} (1 - q) qbar^{jhat}],

    with jhat = j - (c - 1)/24 and a0 = -(c - 1)/24. This differs from the generic table at Delta = |j|, which is
    the Verma continuation (no null factor) and is not the character of any state.
    """
    c = real_input_label(c, name="c")
    lam = _nonnegative_order(lam)
    spin = _integer_field("spin", spin)
    if spin < 1:
        raise ValueError("a chiral current has spin j >= 1")
    n = lam + 1
    C = _chiral_coeff_polys(n)
    a0 = mp.mpf(-(c - 1) / 24)
    aj = spin + a0
    current = [mp.exp(-2 * mp.pi * aj) * val for val in _eval_coeff_polys(C, aj)]
    vacuum = _convolve([mp.exp(-2 * mp.pi * a0) * val for val in _eval_coeff_polys(C, a0)],
                       _one_minus_q_series(lam, C), n)
    S = sqrt_tau2_table(lam)
    table = [[mp.mpf(0)] * n for _ in range(n)]
    for m in range(n):
        for nn in range(n):
            acc = mp.mpf(0)
            for a in range(m + 1):
                row = S[a]
                for b in range(nn + 1):
                    s_ab = row[b]
                    if s_ab == 0:
                        continue
                    acc += s_ab * (current[m - a] * vacuum[nn - b] + vacuum[m - a] * current[nn - b])
            table[m][nn] = acc
    return table


def _majorant(coeffs, y):
    """``sum |coeff_k| y^k``: a bound on ``|C_p(a)|`` for ``|a| <= y``."""
    out = mp.mpf(0)
    for coeff in reversed(coeffs):
        out = out * y + abs(coeff)
    return out


def _null_series_level(lam: int, C: list[list], a) -> int:
    """The last q-level an irreducible character's null series needs at Taylor order ``lam`` (chiral weight
    ``a = h - (c - 1)/24``), at the working precision.

    The v^p coefficient of q^{a + n} is e^{-2 pi (a + n)} C_p(a + n), bounded by e^{-2 pi (a + n)} B_p(|a| + n) with
    B_p the coefficientwise majorant of C_p.  Past n = p / (2 pi - log 2) successive levels shrink by at least 1/2
    (B_p(y + 1) / B_p(y) <= (1 + 1/y)^p), so with the null coefficients bounded by 1 the series beyond level N is
    below twice its level-(N + 1) bound.  N is the first level where that is below 2^-prec of the size of a
    q^a (1 + q) expansion, for every p <= lam."""
    eps = mp.mpf(2) ** (-mp.mp.prec)
    y = abs(a)
    base = torus_exp_base()
    scale = [_majorant(C[p], y) + base * _majorant(C[p], y + 1) for p in range(lam + 1)]
    level = max(1, int(mp.ceil(lam / (2 * mp.pi - mp.log(2)))))
    while any(2 * mp.exp(-2 * mp.pi * (level + 1)) * _majorant(C[p], y + level + 1) > eps * scale[p]
              for p in range(lam + 1)):
        level += 1
    return level


def _character_chiral_coeffs(c, h, lam: int, C: list[list]):
    """``([v^p] of the chiral half q^{h - (c-1)/24} g(q) of the reduced character, degenerate)``: g = 1 for a Verma
    module, the irreducible module's null series (:func:`~virasoro_blocks.torus_onept.irreducible_character_levels`)
    on a Kac weight -- the vacuum at every c, and every Kac weight of a recognized singular c.

    The null series enters as the v-series of g(q(v)) (each q^n contributing e^{-2 pi n} C_p(n)) convolved with the
    q^a coefficients, the construction of :func:`torus_current_table`: there it is the finite 1 - q, for a minimal
    model it is the infinite Rocha-Caridi series, cut where it falls below the working precision
    (:func:`_null_series_level`)."""
    from .torus_onept import irreducible_character_levels

    n = lam + 1
    a0 = mp.mpf(-(c - 1) / 24)
    a = mp.mpf(h) + a0
    chi = [mp.exp(-2 * mp.pi * a) * val for val in _eval_coeff_polys(C, a)]
    levels = irreducible_character_levels(c, h, _null_series_level(lam, C, a))
    if levels is None:
        return chi, False
    if any(abs(mu) > 1 for _level, mu in levels):
        # (the Virasoro embedding posets have Moebius values 0, +-1; the truncation bound assumes it)
        raise ArithmeticError(f"irreducible character of h = {h} has a null coefficient beyond +-1")
    null = [mp.mpf(0)] * n
    for level, mu in levels:
        if level == 0:
            null[0] += mu
            continue
        weight = mp.exp(-2 * mp.pi * level)
        for p, val in enumerate(_eval_coeff_polys(C, mp.mpf(level))):
            term = weight * val
            null[p] += term if mu == 1 else -term
    return _convolve(chi, null, n), True


def torus_character_point_values(c, delta, spin: int, lam: int, *, symmetrize: bool = True):
    """``(table, irreducible)``: the exact numeric (m, n) Taylor table of the reduced character of the module
    (h, hbar) = ((Delta + j)/2, (Delta - j)/2), paired with its mirror (hbar, h) for j > 0 when ``symmetrize`` (the
    convention of :func:`torus_character_table`), and whether an irreducible (degenerate) character entered.

    A module on a Kac weight carries its irreducible character (:func:`_character_chiral_coeffs`): the vacuum
    (0, 0) is :func:`torus_vacuum_table`, a current (j, 0) is :func:`torus_current_table`, and at a recognized
    singular c (a minimal model, c = 1, ...) every Kac weight is resolved exactly.  Elsewhere the table is
    :func:`torus_character_table` evaluated at Delta.  Entries are the table values (prefactor included)."""
    c = real_input_label(c, name="c")
    delta = real_input_label(delta, name="Delta")
    lam = _nonnegative_order(lam)
    spin = _integer_field("spin", spin)
    if spin < 0:
        raise ValueError("spin labels the j >= 0 sector; pass |j|")
    n = lam + 1
    C = _chiral_coeff_polys(n)
    h, hbar = (delta + spin) / 2, (delta - spin) / 2
    u, left = _character_chiral_coeffs(c, h, lam, C)
    w, right = (u, left) if spin == 0 else _character_chiral_coeffs(c, hbar, lam, C)
    S = sqrt_tau2_table(lam)
    pair = symmetrize and spin != 0
    table = [[mp.mpf(0)] * n for _ in range(n)]
    for m in range(n):
        for nn in range(n):
            acc = mp.mpf(0)
            for a in range(m + 1):
                row = S[a]
                for b in range(nn + 1):
                    s_ab = row[b]
                    if s_ab == 0:
                        continue
                    if pair:
                        acc += s_ab * (u[m - a] * w[nn - b] + w[m - a] * u[nn - b])
                    else:
                        acc += s_ab * u[m - a] * w[nn - b]
            table[m][nn] = acc
    return table, left or right


def torus_character_point_table(c, delta, spin: int, lam: int, *, symmetrize: bool = True) -> SerializedDerivTable:
    """:func:`torus_character_point_values` as a frozen point table: degree-0 numerators, ``shift = Delta``, base
    e^{-2 pi} and no poles, as the sphere point tables are stored (evaluating it at Delta gives the values; elsewhere
    it only rescales them).  ``metadata["regularization_policy"]`` is ``"irreducible"`` when a degenerate module's
    irreducible character entered, else ``"direct"``."""
    delta_label = real_input_label(delta, name="Delta")
    values, irreducible = torus_character_point_values(c, delta_label, spin, lam, symmetrize=symmetrize)
    numerators = [[Polynomial([value], field=mp.mpf) for value in row] for row in values]
    return SerializedDerivTable(
        shift=delta_label,
        base=torus_exp_base(),
        pole_list_x=(),
        coordinates=(TORUS_COORDINATE,),
        numerators={TORUS_COORDINATE: numerators},
        sample_points_x=(mp.mpf(0),),
        metadata={"regularization_policy": "irreducible" if irreducible else "direct"},
        working_dps=mp.mp.dps,
    )


def torus_character_table(
    c,
    spin: int,
    lam: int,
    *,
    shift=None,
    symmetrize: bool = True,
) -> SerializedDerivTable:
    """Exact per-spin reduced-character derivative table, serialized form.

    Entry (m, n) reconstructs as base^x * numerator[m][n](x) with
    base = e^{-2 pi}, x = Delta - shift, and an empty pole list. The
    numerator is an exact polynomial of degree m + n (2(m + n) after
    symmetrization is still m + n: the two orientations share the degree).

    `spin` = j >= 0 labels the sector. For j > 0 and `symmetrize=True`
    (default) the table is the j + (-j) pair sum -- reality of Z pairs the
    degeneracies d_{h, hbar} = d_{hbar, h}, so the pair is the natural
    positive variable and entry (m, n) becomes N_{m,n} + N_{n,m}. For j = 0
    the table is symmetric already and `symmetrize` is a no-op.

    `shift` defaults to the unitarity bound |spin| (the repo-wide convention);
    pass a larger value (e.g. a gap) to anchor x = 0 at the positivity edge.
    The exact input label is retained and checked before numerical conversion.
    """
    lam = _nonnegative_order(lam)
    spin = _integer_field("spin", spin)
    if spin < 0:
        raise ValueError("spin labels the j >= 0 sector; pass |j|")
    c = real_input_label(c, name="c")
    if shift is None:
        shift = spin
    shift = real_input_label(shift, name="shift")
    if shift < spin:
        raise ValueError(
            f"shift {mp.nstr(shift, 8)} below the unitarity bound |j| = {spin}"
        )
    return _torus_character_table(c, spin, lam, shift=shift, symmetrize=symmetrize)


def _torus_character_table(c, spin, lam, *, shift, symmetrize):
    n = lam + 1
    C_raw = _chiral_coeff_polys(n)
    ccm1_24 = (c - 1) / 24
    # hhat(x) = x/2 + s1, hbarhat(x) = x/2 + s2 with x = Delta - shift.
    s1 = (shift + spin) / 2 - ccm1_24
    s2 = (shift - spin) / 2 - ccm1_24
    C_polys = [Polynomial(coeffs, field=mp.mpf) for coeffs in C_raw]
    half = mp.mpf(1) / 2
    U = [poly.compose_linear(half, s1) for poly in C_polys]
    V = (
        U
        if spin == 0
        else [poly.compose_linear(half, s2) for poly in C_polys]
    )

    S = sqrt_tau2_table(lam)
    K = mp.exp(2 * mp.pi * ((c - 1) / 12 - shift))

    # Chiral product cache P[p][r] = U_p * V_r as ascending coefficient lists.
    P: dict[tuple[int, int], list] = {}

    def _product(p: int, r: int) -> list:
        key = (p, r)
        got = P.get(key)
        if got is None:
            got = (U[p] * V[r]).coeffs
            P[key] = got
        return got

    rows = []
    for m in range(n):
        row = []
        for nn in range(n):
            acc = [mp.mpf(0)] * (m + nn + 1)
            for p in range(m + 1):
                s_row = S[m - p]
                for r in range(nn + 1):
                    s_ab = s_row[nn - r]
                    if s_ab == 0:
                        continue
                    prod = _product(p, r)
                    for k, coeff in enumerate(prod):
                        acc[k] += s_ab * coeff
            row.append([K * coeff for coeff in acc])
        rows.append(row)

    if symmetrize and spin != 0:
        sym_rows = []
        for m in range(n):
            row = []
            for nn in range(n):
                a, b = rows[m][nn], rows[nn][m]
                width = max(len(a), len(b))
                a = a + [mp.mpf(0)] * (width - len(a))
                b = b + [mp.mpf(0)] * (width - len(b))
                row.append([ai + bi for ai, bi in zip(a, b)])
            sym_rows.append(row)
        rows = sym_rows

    numerators = [
        [Polynomial(entry, field=mp.mpf) for entry in row] for row in rows
    ]
    return SerializedDerivTable(
        shift=shift,
        base=torus_exp_base(),
        pole_list_x=(),
        coordinates=(TORUS_COORDINATE,),
        numerators={TORUS_COORDINATE: numerators},
        working_dps=mp.mp.dps,
    )
