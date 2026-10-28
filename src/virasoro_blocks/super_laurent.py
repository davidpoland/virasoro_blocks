"""Exact N=1 (NS / Ramond) blocks at a recursion-singular central charge.

At an N=1 singular ``c0`` (rational ``beta^2``: the minimal models, ``c = 3/2``, ...) Kac poles of the NS and Ramond
recursions collide.  Every recursion quantity is carried here as a Laurent series in a local variable ``t``
(``c = c0 + sigma t^p``; ``p = 2`` at the branch points ``c0 = 3/2, 27/2`` of ``c -> beta`` and where a Ramond
momentum vanishes, else ``p = 1``), so the colliding poles cancel algebraically and the ``t^0`` coefficient at a fixed
exchanged weight is the ``c -> c0`` limit exactly: the physical (irreducible-module) block, also on a Kac weight
(the free boson + fermion at ``c = 3/2`` decomposes exactly with it).  A negative power surviving at the top is a
genuine pole of the block.  This replaces the evaluation at ``c0 + 10^-E`` and its measured guard digits.

The level table and its evaluation are the shared Laurent engine (:mod:`virasoro_blocks.laurent_levels`; NS: two
sectors, Ramond: eight states); this module supplies the N=1 pole data.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import mpmath as mp

from .laurent_levels import Tracked, WindowDeficit, balanced_product, build_levels, eval_t0, recognized_rational
from .numeric_input import _integer_control

FAMILY_NS = "ns"
FAMILY_RAMOND = "ramond"


class SuperLaurentPoleError(ValueError):
    """The N=1 block has a genuine pole at this exchanged weight of the singular c (its c -> c0 limit diverges)."""


def _laurent_inputs(family, h_ext, order):
    key = family.lower() if isinstance(family, str) else None
    if key == 'r':
        key = FAMILY_RAMOND
    if key not in (FAMILY_NS, FAMILY_RAMOND):
        raise ValueError("family must be 'ns' or 'ramond'")
    order = _integer_control('order', order)
    if len(h_ext) != 4:
        raise ValueError('h_ext must be a length-4 sequence')
    return key, order


def _laurent_selectors(sector, branch):
    sector = _integer_control('sector', sector)
    branch = _integer_control('branch', branch)
    if sector not in (0, 1):
        raise ValueError('sector must be 0 (even) or 1 (odd)')
    if branch not in range(4):
        raise ValueError('branch must be 0 (++), 1 (+-), 2 (-+), or 3 (--)')
    return sector, branch


def exact_beta_squared(c) -> Fraction | None:
    """The rational ``beta^2`` of an N=1 singular ``c`` (``chat = 2c/3 = 1 - 2 (beta - 1/beta)^2``), or None.

    Recognized at the precision ``c`` carries (as :func:`virasoro_blocks.singularity.exact_locus_c`)."""
    from .numeric_input import exact_real, exact_b_squared as rational_b_squared

    if exact_real(c) is not None:
        return rational_b_squared(c, superconformal=True)
    from .recursion_ns import beta_from_c
    from .superconformal_blocks import _candidate_superconformal_singular_c

    if isinstance(c, Fraction):
        value = mp.mpf(c.numerator) / c.denominator
    else:
        value = mp.convert(c)
        if isinstance(value, mp.mpc):
            if value.imag != 0:
                return None
            value = value.real
    bits = value._mpf_[3] if value != 0 else 1
    # a long mantissa carries the precision it was formed at; a short one (30000, 3/2) is the exact value itself,
    # tested at the working precision (at 15 digits some rational beta^2 fits every c)
    digits = max(15, int(bits * 0.30103) - 2) if bits > 64 else mp.mp.dps
    with mp.workdps(digits):
        if not _candidate_superconformal_singular_c(+value):
            return None
        beta_sq = beta_from_c(+value) ** 2
        if isinstance(beta_sq, mp.mpc):
            if abs(beta_sq.imag) > abs(beta_sq) * mp.mpf(10) ** (-(digits // 2)):
                return None
            beta_sq = beta_sq.real
        frac = Fraction(mp.nstr(beta_sq, digits)).limit_denominator(10 ** 6)
        if abs(mp.mpf(frac.numerator) / frac.denominator - beta_sq) > abs(beta_sq) * mp.mpf(10) ** (-(digits // 2)):
            return None
    return frac


def exact_c_of(beta_sq: Fraction) -> Fraction:
    """``c = (3/2) chat`` with ``chat = 1 - 2 (x + 1/x - 2)``, ``x = beta^2``."""
    return Fraction(3, 2) * (1 - 2 * (beta_sq + 1 / beta_sq - 2))


def _ns_d_const(r: int, s: int, x: Fraction) -> Fraction:
    """``Delta_rs`` at ``c0``: ``(chat - 1)/16 + (r^2 x - 2 r s + s^2 / x)/8``."""
    chat = 1 - 2 * (x + 1 / x - 2)
    return (chat - 1) / 16 + (r * r * x - 2 * r * s + Fraction(s * s) / x) / 8


def _r_d_const(r: int, s: int, x: Fraction) -> Fraction:
    """The NS Kac weight of a Ramond-external pole at ``c0`` (``b^2 = -x``)."""
    return Fraction(-(r * s - 1), 4) + Fraction(1 - r * r, 8) * (-x) + Fraction(1 - s * s, 8) / (-x)


class _Local:
    """The local variable and the beta series at one singular c0 (Tracked: with their ceilings)."""

    def __init__(self, c0: Fraction, x: Fraction, *, need_even: bool):
        from .recursion_ns import beta_from_c

        self.c0, self.x = c0, x
        branch = x in (1, -1)
        self.p = 2 if (branch or need_even) else 1
        # at c0 = 3/2 run on the c < 3/2 side, where beta is real (as Virasoro at c = 1)
        self.sigma = -1 if x == 1 else 1
        c0m = mp.mpf(c0.numerator) / c0.denominator
        # every exact constant (c0, 2/3, sqrt 2) enters as a ball containing it, so the radii bound the error
        self.c = Tracked.const(c0) + Tracked.monomial(self.p, mp.mpf(self.sigma))
        chat = self.c * Tracked.const(Fraction(2, 3))
        self.inv_root2 = Tracked.const(Fraction(1, 2)).sqrt()
        beta = ((1 - chat).sqrt() + (9 - chat).sqrt()) * (self.inv_root2 / 2)
        b0 = beta_from_c(c0m)
        if abs(mp.mpc(beta.s.coeff(0)) - b0) > abs(mp.mpc(beta.s.coeff(0)) + b0):
            beta = -beta
        self.beta = beta
        self.chat = chat


def _ns_poles(loc: _Local, h_ext, order: int):
    """The NS poles (both sectors): [HJS-a] (6.13)-(6.16), as :class:`recursion_ns.NSFamily`."""
    from .recursion_ns import enumerate_ns_poles

    beta, x = loc.beta, loc.x
    inv_beta = beta.reciprocal()
    inv_root2 = loc.inv_root2
    # exact alpha^2 at c0 for externals given as (or carrying) rationals: a fusion factor that vanishes there is
    # snapped to an exact zero constant (a degenerate external rounded to the working precision is not degenerate),
    # and the external enters as a ball containing the rational
    chat0 = 1 - 2 * (x + 1 / x - 2)
    exact_ext = [recognized_rational(d) for d in h_ext]
    alpha_sq = [8 * Tracked.const(e if e is not None else mp.convert(d)) - (loc.chat - 1) / 2
                for d, e in zip(h_ext, exact_ext)]
    alpha_sq0 = [None if e is None else 8 * e - (chat0 - 1) / 2 for e in exact_ext]
    inv_factor: dict = {}

    def A_rs(r, s):
        out = [Tracked.const(mp.mpf(-1 if (r * s - 1) % 2 else 1) / 2)]
        for p in range(1 - r, r + 1):
            for q in range(1 - s, s + 1):
                if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                    continue
                inv = inv_factor.get((p, q))
                if inv is None:
                    inv = inv_factor[(p, q)] = ((p * beta - q * inv_beta) * inv_root2).snapped(
                        p * x == q).reciprocal()
                out.append(inv)
        return balanced_product(out)

    def fusion(r, s, ia, ib, starred):
        # prod over p, q of (alpha_b - alpha_a + S)(alpha_b + alpha_a + S) / 8, S = p beta - q/beta: the (p, q) set is
        # closed under (p, q) -> (-p, -q), and each such pair is a polynomial in alpha_a^2, alpha_b^2 alone
        a_sq, b_sq, a0, b0 = alpha_sq[ia], alpha_sq[ib], alpha_sq0[ia], alpha_sq0[ib]
        target = 0 if starred else 2
        seen, factors = set(), []
        for p in range(1 - r, r, 2):
            for q in range(1 - s, s, 2):
                if (p + q - (r + s)) % 4 != target or (p, q) in seen:
                    continue
                seen.add((p, q))
                exact = a0 is not None and b0 is not None
                if (p, q) == (0, 0):
                    factors.append(((b_sq - a_sq) / 8).snapped(exact and b0 == a0))
                    continue
                seen.add((-p, -q))
                shift_sq = (p * beta - q * inv_beta).snapped(p * x == q) ** 2
                u = b_sq + shift_sq - a_sq
                zero = False
                if exact:
                    s0 = p * p * x - 2 * p * q + Fraction(q * q) / x
                    u0 = b0 + s0 - a0
                    zero = u0 * u0 - 4 * b0 * s0 == 0
                factors.append(((u * u - 4 * b_sq * shift_sq) / 64).snapped(zero))
        return balanced_product(factors)

    beta_sq, inv_beta_sq = beta * beta, inv_beta * inv_beta
    poles, cache = [], {}
    for r, s, rs in enumerate_ns_poles(order):
        # The expanded Kac formula makes d_{1,1}=0 an exact identity,
        # rather than a cancellation ball that can hide tiny input weights.
        d = ((r * r - 1) * beta_sq - 2 * (r * s - 1) + (s * s - 1) * inv_beta_sq) / 8
        A = A_rs(r, s)
        dc = _ns_d_const(r, s, x)
        d = d.snapped(dc == 0)
        for sector in (0, 1):
            starred = bool((rs % 2) ^ sector)
            K = cache.get((r, s, starred))
            if K is None:
                K = cache[(r, s, starred)] = A * fusion(r, s, 3, 2, starred) * fusion(r, s, 0, 1, starred) \
                    * mp.mpf(4) ** rs
            poles.append(dict(r=r, s=s, rs=rs, K=K, d=d, v=d + mp.mpf(rs) / 2, d_const=dc,
                              v_const=dc + Fraction(rs, 2), sector=sector, branch=0))
    poles.sort(key=lambda p: (p["rs"], p["sector"]))
    return poles


def _r_poles(loc: _Local, h_ext, order: int):
    """The Ramond-external poles (eight states): HJS (5.10)-(5.12), (6.5), as :class:`recursion_r.RFamily`."""
    from .recursion_ns import enumerate_ns_poles
    from .recursion_r import SIGN_PLUS, _N_STATES, _state_parts

    x = loc.x
    b = loc.beta * mp.mpc(0, 1)
    inv_b = b.reciprocal()
    inv_root2 = loc.inv_root2
    inv_two_root2 = inv_root2 / 2
    # gamma at c0 for rational externals, at four times the digits: a factor gamma_l -+ gamma_u + S/(2 sqrt 2) whose
    # exact (algebraic) value vanishes is snapped to an exact zero constant; the external enters as a ball
    exact_ext = [recognized_rational(d) for d in h_ext]
    gamma = [(loc.c * Tracked.const(Fraction(1, 24)) - Tracked.const(e if e is not None else mp.convert(d))).sqrt()
             for d, e in zip(h_ext, exact_ext)]
    with mp.workdps(4 * mp.mp.dps):
        c0h = mp.mpf(loc.c0.numerator) / loc.c0.denominator
        gamma0 = [None if e is None else mp.sqrt(mp.mpc(c0h / 24 - mp.mpf(e.numerator) / e.denominator))
                  for e in exact_ext]
        beta0 = mp.sqrt(mp.mpc(mp.mpf(x.numerator) / x.denominator))
        if mp.re(beta0 * mp.conj(mp.mpc(loc.beta.s.coeff(0)))) < 0:
            beta0 = -beta0
        b0 = mp.mpc(0, 1) * beta0
    inv_factor: dict = {}

    def A_rs(r, s):
        out = [Tracked.const(mp.mpc(1) / 2)]
        for p in range(1 - r, r + 1):
            for q in range(1 - s, s + 1):
                if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                    continue
                inv = inv_factor.get((p, q))
                if inv is None:
                    inv = inv_factor[(p, q)] = ((p * b + q * inv_b) * inv_root2).snapped(p * x == q).reciprocal()
                out.append(inv)
        return balanced_product(out)

    def fusion(r, s, il, iu, sign):
        lower, upper = gamma[il], gamma[iu]
        up = upper if sign == SIGN_PLUS else -upper
        exact = gamma0[il] is not None and gamma0[iu] is not None
        factors = []
        for p in range(1 - r, r, 2):
            for q in range(1 - s, s, 2):
                kind = (p + q - (r + s)) % 4
                if kind not in (0, 2):
                    continue
                shift = ((p * b + q * inv_b) * inv_two_root2).snapped(p * x == q)
                term = lower - up + shift if kind == 2 else lower + up + shift
                zero = False
                if exact:
                    with mp.workdps(4 * mp.mp.dps):
                        u0 = gamma0[iu] if sign == SIGN_PLUS else -gamma0[iu]
                        sh0 = 0 if p * x == q else (p * b0 + q / b0) / (2 * mp.sqrt(2))
                        val = gamma0[il] - u0 + sh0 if kind == 2 else gamma0[il] + u0 + sh0
                        zero = abs(val) <= mp.mpf(10) ** (-3 * mp.mp.dps // 4)
                factors.append(term.snapped(zero))
        return balanced_product(factors)

    poles, cache = [], {}
    for r, s, rs in enumerate_ns_poles(order):
        d = -Tracked.const(mp.mpf(r * s - 1) / 4) + (mp.mpf(1 - r * r) / 8) * b * b \
            + (mp.mpf(1 - s * s) / 8) * inv_b * inv_b
        A = A_rs(r, s)
        dc = _r_d_const(r, s, x)
        d = d.snapped(dc == 0)
        for state in range(_N_STATES):
            outer_state = state if not (rs % 2) else state ^ 7
            _sec, outer_branch = _state_parts(outer_state)
            sign3, sign2 = (outer_branch >> 1) & 1, outer_branch & 1
            K = cache.get((r, s, sign3, sign2))
            if K is None:
                K = A * fusion(r, s, 3, 2, sign3) * fusion(r, s, 0, 1, sign2) * mp.mpf(4) ** rs
                if rs % 2:
                    K = -K
                cache[(r, s, sign3, sign2)] = K
            _inner_sector, inner_branch = _state_parts(state)
            poles.append(dict(r=r, s=s, rs=rs, K=K, d=d, v=d + mp.mpf(rs) / 2, d_const=dc,
                              v_const=dc + Fraction(rs, 2), sector=state, branch=inner_branch))
    poles.sort(key=lambda p: (p["rs"], p["sector"], p["branch"]))
    return poles


def _couples(family):
    if family == FAMILY_NS:
        return lambda outer_sector, inner: (outer_sector ^ (inner["rs"] % 2)) == inner["sector"]
    return lambda outer_sector, inner: (outer_sector if not (inner["rs"] % 2) else outer_sector ^ 7) == inner["sector"]


def _seed(family, sector: int, order: int) -> list:
    out = [0] * (order + 1)
    if family == FAMILY_NS:
        if sector == 0:
            out[0] = 1
            n = 1
            while 4 * n * n <= order:
                out[4 * n * n] = 2
                n += 1
        return out
    if sector // 4 == 0:
        out[0] = 1
    return out


_STATES: dict = {}


def super_laurent_state(family: str, c0, h_ext: Sequence, order: int, *, half: int | None = None) -> dict:
    """The h-independent Laurent level table of the NS (``family='ns'``) or Ramond block at the singular ``c0``,
    through ``order`` half-steps (powers of ``q^(1/2)``), at the working precision, on the narrowest window that
    holds it exactly (at least ``half``).  Cached."""
    from .tseries import set_real_mode

    family, order = _laurent_inputs(family, h_ext, order)
    if half is not None:
        half = _integer_control('half', half, 1)
    x = exact_beta_squared(c0)
    if x is None:
        raise ValueError(f"c = {c0} is not an N=1 singular central charge (rational beta^2)")
    c_exact = exact_c_of(x)
    h_ext = [d if isinstance(d, Fraction) else mp.convert(d) for d in h_ext]
    key = (family, c_exact, tuple(_value_key(d) for d in h_ext), int(order), mp.mp.prec)
    hit = _STATES.get(key)
    if hit is not None and (half is None or hit["levels"].half >= half):
        return hit
    need_even = False
    if family == FAMILY_RAMOND:
        c0m = mp.mpf(c_exact.numerator) / c_exact.denominator
        # a Ramond momentum gamma = sqrt(c/24 - Delta) vanishing at c0 is analytic only in sqrt(c - c0)
        need_even = any(recognized_rational(d) == c_exact / 24 for d in h_ext)
    # NS at a locus with real beta (the minimal models and c = 3/2, on its c < 3/2 side) and real externals: every
    # series is real, and the build runs in arb (as the Virasoro table at c <= 1)
    real = family == FAMILY_NS and x > 0 and all(not isinstance(d, mp.mpc) for d in h_ext)
    saved_real = set_real_mode(real)
    try:
        local = {}

        def make_poles():
            loc = local["loc"] = _Local(c_exact, x, need_even=need_even)
            return (_ns_poles if family == FAMILY_NS else _r_poles)(loc, h_ext, int(order))

        levels = build_levels(make_poles, int(order), seeds=lambda sector, n: _seed(family, sector, n),
                              couples=_couples(family), label=lambda p: (p["r"], p["s"]), half=half,
                              window_key=(family, c_exact, int(order), need_even, mp.mp.prec))
    finally:
        set_real_mode(saved_real)
    state = dict(family=family, c0=c_exact, x=x, loc=local["loc"], levels=levels, order=int(order), real=real,
                 h_ext=h_ext, work_prec=mp.mp.prec)
    if len(_STATES) > 16:
        _STATES.clear()
    _STATES[key] = state
    return state


def super_laurent_H(state: dict, delta, *, sector: int = 0, branch: int = 0, radii: list | None = None) -> list:
    """``[H_0 .. H_order]`` (powers of ``q^(1/2)``) of the ``sector`` block (Ramond: and ``branch``, as
    :func:`recursion_r.H_q_series_r`) at the exchanged weight ``delta``: the ``t^0`` coefficients, the c -> c0 limit
    at fixed weight.  On a Kac weight of c0 the colliding poles cancel; a negative power that survives is a genuine
    pole (:class:`SuperLaurentPoleError`). Evaluation runs at the table's construction precision, or rebuilds the
    table at the caller's when that is higher (or on a wider window). Returned midpoints and optional ball radii retain the evaluation
    precision; :func:`singular_H` additionally certifies and rounds values to the caller's precision."""
    sector, branch = _laurent_selectors(sector, branch)
    delta = delta if isinstance(delta, Fraction) else mp.convert(delta)
    work_prec = max(state["work_prec"], mp.mp.prec)
    with mp.workprec(work_prec):
        if work_prec > state["work_prec"]:
            # (the precision-specific builder cache)
            state = super_laurent_state(state["family"], state["c0"], state["h_ext"], state["order"],
                                        half=state["levels"].half)
        return _super_laurent_H_at_work(state, delta, sector=sector, branch=branch, radii=radii)


def _super_laurent_H_at_work(state, delta, *, sector=0, branch=0, radii=None):
    """Read a table after the caller has resolved input and evaluation precision."""
    from .tseries import set_real_mode

    family = state["family"]
    if family == FAMILY_RAMOND:
        from .recursion_r import _state

        sector = _state(sector, 0, branch)
    delta = delta if isinstance(delta, Fraction) else mp.convert(delta)
    # a weight given as (the rounding of) a rational enters the table as a ball containing it
    exact = None if isinstance(delta, mp.mpc) else recognized_rational(delta)
    weight = exact if exact is not None else delta
    while True:
        saved_real = set_real_mode(state["real"] and not isinstance(delta, mp.mpc))
        try:
            values, negatives = eval_t0(state["levels"], weight, sector=sector,
                                        seed=_seed(family, sector, state["order"]), couples=_couples(family),
                                        radii=radii)
            break
        except WindowDeficit as exc:
            # this weight reads deeper than the build's analysis covers: the table on a window wider by that much
            if radii is not None:
                radii.clear()
            state.update(super_laurent_state(family, state["c0"],
                                             state["h_ext"], state["order"],
                                             half=state["levels"].half + exc.deficit))
        finally:
            set_real_mode(saved_real)
    if negatives:
        k, v = negatives[0]
        raise SuperLaurentPoleError(
            f"{family} block: the exchanged weight {mp.nstr(delta, 15)} is a genuine pole at the singular "
            f"c = {state['c0']} (" + (f"H_{k} keeps a t^{v} term after the colliding poles cancel" if v is not None
                                     else f"from H_{k}: a pole there at every c") + ")")
    if state["real"] and not isinstance(delta, mp.mpc):
        values = [mp.re(v) for v in values]
    return values


def exact_singular_c(c) -> Fraction | None:
    """The N=1 singular ``c0`` that ``c`` is, to the precision it carries (where the bare recursion divides by zero),
    as a Fraction; None for any other c -- a c deliberately near a locus is a different c."""
    x = exact_beta_squared(c)
    if x is None:
        return None
    c0 = exact_c_of(x)
    value = mp.convert(c) if not isinstance(c, Fraction) else None
    if value is None:
        return c0 if c == c0 else None
    value = mp.re(value)
    bits = value._mpf_[3] if value != 0 else 1
    with mp.workprec(max(53, bits + 8)):
        c0m = mp.mpf(c0.numerator) / c0.denominator
        if abs(value - c0m) > mp.mpf(2) ** (-(bits - 4)) * max(1, abs(c0m)):
            return None
    return c0


def _value_key(x):
    if isinstance(x, Fraction):
        return ("rational", x.numerator, x.denominator)
    x = mp.convert(x)
    return x._mpf_ if isinstance(x, mp.mpf) else (x.real._mpf_, x.imag._mpf_)


_SINGULAR_LOSS: dict = {}


def singular_H(family: str, delta, h_ext: Sequence, *, c0: Fraction, order: int, sector: int = 0,
               branch: int = 0) -> list:
    """``[H_0 .. H_order]`` at the N=1 singular ``c0`` (the c -> c0 limit at fixed ``delta``), at the working
    precision ``P``; raises :class:`SuperLaurentPoleError` on a genuine pole.

    The table runs in Arb with every exact input (c0, a rational external or weight, 2/3, sqrt 2) a ball containing
    it, so each t^0 coefficient's radius, plus the rounding of its midpoint to the working precision, bounds its error
    (checked against builds at 4x the precision: under 0.4 of the bound).  The radius is the rounding unit times an
    amplification set by the table, not by the precision, so one build measures how many digits it loses: the first
    build runs at ``P``. If it falls short, a rebuild adds the measured loss (verified by the same bound). The loss
    is cached per table, so later calls build once. The window likewise grows only by measured deficits. An escalation whose
    shortfall does not shrink raises."""
    family, order = _laurent_inputs(family, h_ext, order)
    sector, branch = _laurent_selectors(sector, branch)
    base = mp.mp.dps
    target = mp.eps         # certified to the working precision's bits (relative, unit floor)
    key = (family, c0, tuple(_value_key(d) for d in h_ext), int(order), mp.mp.prec)
    guard = _SINGULAR_LOSS.get(key, 0)
    ext = [d if isinstance(d, Fraction) else mp.convert(d) for d in h_ext]
    previous_worst = None
    while True:
        with mp.workdps(base + guard):
            state = super_laurent_state(family, c0, ext, order)
            rads: list = []
            values = _super_laurent_H_at_work(state, delta, sector=sector, branch=branch, radii=rads)
            worst = max(((r + mp.eps / 2 * abs(v)) / max(1, abs(v)) for v, r in zip(values, rads)), default=mp.mpf(0))
        if worst <= target:
            break
        # every digit lost: the loss is at least the precision carried, so carry that much more
        short = base + guard if not mp.isfinite(worst) else int(mp.ceil(mp.log10(worst / target)))
        if previous_worst is not None and mp.isfinite(worst) and worst >= previous_worst:
            raise ArithmeticError(f"the {family} block at c = {c0}: the Laurent table's error bound does not shrink "
                                  "with the precision")
        previous_worst = worst
        guard += max(1, short)
    _SINGULAR_LOSS[key] = max(guard, _SINGULAR_LOSS.get(key, 0))
    return [+v for v in values]


_MERGED: dict = {}


def merged_state(family: str, h_ext: Sequence, *, c0: Fraction, order: int, sector: int = 0, branch: int = 0) -> dict:
    """The NS / Ramond block at the singular ``c0`` as a function of the exchanged weight, in merged-pole form
    (:func:`laurent_levels.merged_columns`): ``H_k(delta) = seed_k + sum M[k, col] / (delta - d0)^a``.  Built at the
    working precision plus the loss of digits the table's error bound measured (:func:`singular_H`, probed once at an
    irrational weight). Coefficients and exact rational pole locations retain their ball bounds; :func:`merged_H`
    checks those bounds at each requested weight and rebuilds at more digits if needed. Cached per cutoff."""
    from flint import acb_mat, ctx as flint_ctx

    from .laurent_levels import merged_columns
    from .polynomial import _flint_prec_bits
    from .recursion import _location_ball

    family, order = _laurent_inputs(family, h_ext, order)
    sector, branch = _laurent_selectors(sector, branch)
    base = mp.mp.dps
    ext_key = tuple(_value_key(d) for d in h_ext)
    key = (family, c0, ext_key, int(order), int(sector), int(branch), mp.mp.prec)
    hit = _MERGED.get(key)
    if hit is not None:
        return hit
    # Build each requested cutoff independently. Slicing a deeper matrix leaves
    # its higher-level poles and multiplicities in the interpolation skeleton;
    # even column pruning depends on the coefficient scale at this cutoff.
    table_key = (family, c0, tuple(_value_key(d) for d in h_ext), int(order), mp.mp.prec)
    if table_key not in _SINGULAR_LOSS:
        singular_H(family, mp.sqrt(2), h_ext, c0=c0, order=order, sector=sector, branch=branch)
    guard = _SINGULAR_LOSS[table_key]
    ext = [d if isinstance(d, Fraction) else mp.convert(d) for d in h_ext]
    state_sector = sector
    if family == FAMILY_RAMOND:
        from .recursion_r import _state

        state_sector = _state(sector, 0, branch)
    with mp.workdps(base + guard):
        state = super_laurent_state(family, c0, ext, order)
        while True:
            try:
                discarded = []
                locations, M, ncols = merged_columns(state["levels"], sector=state_sector, couples=_couples(family),
                                                    as_acb=True, discarded=discarded)
                break
            except WindowDeficit as exc:
                state = super_laurent_state(family, state["c0"], ext, order,
                                            half=state["levels"].half + exc.deficit)
        flint_ctx.prec = _flint_prec_bits()
        couples = _couples(family)
        # every Kac location of the poles this sector reads (a merged pole, or a removable one with no column):
        # there the block is the c -> c0 limit at fixed weight, as for Virasoro (recursion.eval_merged_H)
        kac = {p["d_const"] for p in state["levels"].poles
               if p.get("d_const") is not None and not p["K"].is_zero() and couples(state_sector, p)}
        out = dict(family=family, c0=c0, order=int(order), sector=state_sector, locations=locations, ncols=ncols,
                   kac=kac,
                   M=acb_mat(M) if ncols else None,
                   d0=[_location_ball(d0, ("exact", e) if e is not None else ("num", d0), False)
                       for d0, _a, e in locations],
                   discarded=[(_location_ball(d0, ("exact", e) if e is not None else ("num", d0), False), a, bounds)
                              for (d0, a, e), bounds in discarded],
                   powers=[a for _d0, a, _e in locations],
                   seed=_seed(family, state_sector, int(order)), dps=base + guard, real=state["real"],
                   h_ext=ext, branch=int(branch), user_sector=int(sector))
    if len(_MERGED) > 32:
        _MERGED.clear()
    _MERGED[key] = out
    return out


def merged_pole_pool(ms: dict) -> list:
    """The merged poles as a pool for an interpolated table: each location repeated by its order."""
    order = {}
    for d0, a, exact in ms["locations"]:
        k = exact if exact is not None else mp.nstr(d0, 30)
        prev = order.get(k)
        order[k] = (d0, a if prev is None else max(prev[1], a))
    pool = []
    for d0, a in order.values():
        pool += [mp.re(mp.mpc(d0)) if mp.im(mp.mpc(d0)) == 0 else mp.mpc(d0)] * a
    return pool


def merged_laurent_at(ms: dict, delta) -> tuple:
    """The merged form of a :func:`merged_state` expanded at ``delta``: ``(finite, principal)``, ``finite[k]`` its
    regular part there (``seed_k`` plus every column located elsewhere) and ``principal[a][k]`` the coefficient of
    ``(delta' - delta)^-a`` (the columns located at ``delta``; empty off the merged poles).  Exact: the Laurent data of
    the block as a function of the weight at the singular c, read off its columns -- the limit in the weight, not the
    c -> c0 limit :func:`merged_H` takes on a Kac location."""
    from flint import acb, ctx as flint_ctx
    from mpmath.libmp import dps_to_prec

    from .derivs import _acb_to_scalar
    from .polynomial import _flint_prec_bits, _mpf_to_acb

    delta = delta if isinstance(delta, Fraction) else mp.convert(delta)
    dq = None if (isinstance(delta, mp.mpc) and delta.imag != 0) else recognized_rational(delta)
    order = ms["order"]
    work_prec = max(dps_to_prec(ms["dps"]), mp.mp.prec)
    with mp.workprec(work_prec):
        flint_ctx.prec = _flint_prec_bits()
        from .recursion import _weight_ball

        x = _weight_ball(delta, True)
        finite = [acb(v) for v in ms["seed"]]
        principal: dict = {}
        if ms["M"] is not None:
            for col, ((d0, a, exact), d0_ball) in enumerate(zip(ms["locations"], ms["d0"])):
                here = (exact is not None and dq == exact) or (exact is None and mp.mpc(d0) == mp.mpc(delta))
                if here:
                    row = principal.setdefault(a, [acb(0)] * (order + 1))
                    for k in range(order + 1):
                        row[k] += ms["M"][k, col]
                    continue
                w = (x - d0_ball) ** (-a)
                for k in range(order + 1):
                    finite[k] += ms["M"][k, col] * w
        real = ms["real"] and not isinstance(delta, mp.mpc)

        def read(values):
            out = [_acb_to_scalar(v) for v in values]
            return [+(mp.re(v) if real else mp.mpc(v)) for v in out]

        return read(finite), {a: read(row) for a, row in sorted(principal.items())}


def merged_laurent_rows(ms: dict, loc, *, principal_order: int, regular_order: int, tol=None,
                        allow_regular_point=False):
    """The Laurent data of a :func:`merged_state` at ``loc`` in the weight (:func:`poles.laurent_rows_acb`: each merged
    column is ``1 / (delta - d0)^a``), as flint balls at the state's precision, or None when no merged column sits at
    ``loc`` (to ``tol``, default :func:`poles.location_tolerance`) -- unless ``allow_regular_point``, which returns the
    Taylor rows there.  The weight's own Laurent expansion at the singular c, which the exact principal-part
    split of an interpolated table needs (not the c -> c0 limit :func:`merged_H` takes on a Kac location).  The pruned
    columns (``discarded``) are below the state's error bound and are left out."""
    from flint import acb, ctx as flint_ctx
    from mpmath.libmp import dps_to_prec

    from .poles import laurent_rows_acb, location_tolerance
    from .polynomial import _flint_prec_bits, _mpf_to_acb

    loc_c = mp.mpc(loc)
    tol = location_tolerance(loc_c) if tol is None else tol
    at_loc = [col for col, (d0, _a, _e) in enumerate(ms["locations"]) if abs(mp.mpc(d0) - loc_c) <= tol]
    if not at_loc and not allow_regular_point:
        return None
    with mp.workprec(max(dps_to_prec(ms["dps"]), mp.mp.prec)):
        flint_ctx.prec = _flint_prec_bits()
        rows = laurent_rows_acb(ms["M"], [acb(1)] * ms["ncols"], ms["d0"], ms["powers"], at_loc, _mpf_to_acb(loc_c),
                                [acb(v) for v in ms["seed"]], principal_order, regular_order)
    flint_ctx.prec = _flint_prec_bits()
    return rows


def merged_H(ms: dict, delta) -> list:
    """``[H_0 .. H_order]`` from a :func:`merged_state` at ``delta``; a weight on a Kac location (merged or
    removable) is the c -> c0 limit there (:func:`singular_H`), as for Virasoro: the merged form's limit in the
    weight is the Fock-module block (c = 3/2, h = 0, externals 3/20: H_4 3.84 against the physical 2.88)."""
    from flint import acb, acb_mat, arb, ctx as flint_ctx

    from .derivs import _acb_to_scalar
    from .polynomial import _flint_prec_bits, _mpf_to_acb

    target = mp.eps
    delta = delta if isinstance(delta, Fraction) else mp.convert(delta)
    dq = None if (isinstance(delta, mp.mpc) and delta.imag != 0) else recognized_rational(delta)
    if (dq is not None and dq in ms["kac"]) or any(exact is None and delta == d0 for d0, _a, exact in ms["locations"]):
        return singular_H(ms["family"], delta, ms["h_ext"], c0=ms["c0"], order=ms["order"],
                          sector=ms["user_sector"], branch=ms["branch"])
    previous = None
    while True:
        with mp.workdps(ms["dps"]):
            flint_ctx.prec = _flint_prec_bits()
            balls = [acb(v) for v in ms["seed"]]
            from .recursion import _weight_ball

            x = _weight_ball(delta, True)
            if ms["M"] is not None:
                w = acb_mat([[(x - d0) ** (-a)] for d0, a in zip(ms["d0"], ms["powers"])])
                col = ms["M"] * w
                balls = [o + col[k, 0] for k, o in enumerate(balls)]
            for d0, a, bounds in ms["discarded"]:
                weight = ((x - d0) ** (-a)).abs_upper()
                for k, bound in enumerate(bounds):
                    if not bound.is_zero():
                        radius = arb(0, (bound * weight).abs_upper())
                        balls[k] += acb(radius, radius)
            out = [_acb_to_scalar(v) for v in balls]
            worst = (max(((mp.mpf(ball.rad()) + mp.eps / 2 * abs(value)) / max(1, abs(value))
                          for value, ball in zip(out, balls)), default=mp.mpf(0))
                     if all(ball.is_finite() for ball in balls) else mp.inf)
            real = ms["real"] and not isinstance(delta, mp.mpc)
            out = [mp.re(v) if real else mp.mpc(v) for v in out]
        if mp.isfinite(worst) and worst <= target:
            break
        # A probe at sqrt(2) measures table loss, not the amplification next to
        # this weight's poles. Rebuild by the loss measured at the actual weight.
        if previous is not None and mp.isfinite(worst) and worst >= previous:
            raise ArithmeticError("the merged N=1 series error bound does not shrink with the precision")
        previous = worst
        extra = ms["dps"] if not mp.isfinite(worst) else int(mp.ceil(mp.log10(worst / target)))
        with mp.workdps(ms["dps"] + max(1, extra)):
            ms = merged_state(ms["family"], ms["h_ext"], c0=ms["c0"], order=ms["order"],
                              sector=ms["user_sector"], branch=ms["branch"])
    return [+v for v in out]
