"""The Laurent-in-t level table shared by the Virasoro, NS and Ramond recursions at a singular central charge.

At a recursion-singular ``c0`` every recursion quantity is a Laurent series in a local variable ``t`` (``c = c0 +
sigma t^p``), so colliding Kac poles cancel algebraically and the ``t^0`` coefficient at a fixed exchanged weight is
the c -> c0 limit.  The families differ only in their pole data (residue ``K``, pole ``d``, intermediate weight
``v = d + level``, exact ``t^0`` constants of ``d`` and ``v`` for collision snapping, a sector) and in which inner
poles feed which outer block (``couples``); this module holds what they share:

* :class:`Tracked`: a series with its *ceiling* -- the highest power it is known exactly to, through the window
  truncation of every operation (a product ``x y`` is exact up to ``min(a_x + v_y, a_y + v_x)``, a reciprocal to
  ``a - 2 v``, a square root to ``a - v/2``).  A product whose lowest power falls below the window raises
  :class:`WindowDeficit` with the powers it lacked.  Nothing about the window is guessed: a build reports exactly
  how much wider it must be.
* :func:`build_levels`: the interaction weights ``K_j / (v_o - d_j)``, the per-row truncation analysis (each entry
  keeps the positive powers a ``t^0`` downstream reads), the level fill with the batched t-convolutions, and the
  ceiling check of every entry against what is read from it.
* :func:`eval_t0`: the ``t^0`` coefficients at an exchanged weight (a Kac weight snapped exactly when the weight is a
  recognized rational), their ball radii and ceilings, and the surviving negative powers (a genuine pole).
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp

from . import tseries as _ts
from .tseries import LaurentSeries, set_window, t_conv_weighted_sum, window_half

EXACT = None            # ceiling of an exactly known series (no truncation reaches it)


class WindowDeficit(ArithmeticError):
    """A Laurent computation needed ``deficit`` more powers of window on one side."""

    def __init__(self, deficit: int, where: str = ""):
        super().__init__(f"the Laurent window lacks {deficit} powers{where}")
        self.deficit = int(deficit)


def certain_valuation(series: LaurentSeries):
    """The lowest power whose coefficient is present (None for the zero series): its ball excludes zero, or its
    midpoint is significant at the working precision (:func:`tseries._valuation_tol` of the series' scale, the rule of
    ``LaurentSeries.valuation``).  Either alone misleads: Arb radii here can be pessimistic by many orders (a genuine
    coefficient in a ball that contains zero), and the rounding an exact cancellation leaves has a tiny midpoint."""
    if series._zero is True:
        return None
    const = series._const
    if const is not _ts._NOT_CONSTANT:
        return None if const == 0 else 0
    fl = series._fl
    if fl is not None and series._coeffs is None:
        # the common case without the coefficient list: the first coefficient that is not an exact zero (flint's
        # valuation) excludes zero
        P = fl.P
        i = P.valuation()
        if i < 0:
            return None
        if abs(P[i]) > 0:
            return _ts.T_LO + i
        cc = P.coeffs()
        tol = None
        for i, c in enumerate(cc):
            if c.is_zero():
                continue
            m = abs(c)
            if m > 0:
                return _ts.T_LO + i
            if tol is None:         # (only for a ball that contains zero: is its midpoint significant?)
                tol = max(abs(x).mid() for x in cc) * _ts._ARB(_ts._valuation_tol())
            if m.mid() > tol:
                return _ts.T_LO + i
        return None
    return series.valuation()


def _min_ceiling(*values):
    out = EXACT
    for v in values:
        if v is not EXACT and (out is EXACT or v < out):
            out = v
    return out


def _cap(ceiling):
    """A computed series is exact at most up to the window's top power."""
    return _ts.T_HI if ceiling is EXACT or ceiling > _ts.T_HI else ceiling


def product_ceiling(ax, vx, ay, vy):
    """The ceiling of a product of series with ceilings ``ax``, ``ay`` and valuations ``vx``, ``vy`` (raises
    :class:`WindowDeficit` when its lowest power falls below the window)."""
    if vx is None or vy is None:
        return EXACT
    if vx + vy < _ts.T_LO:
        raise WindowDeficit(_ts.T_LO - (vx + vy), " below")
    return _cap(_min_ceiling(None if ax is EXACT else ax + vy, None if ay is EXACT else ay + vx))


class Tracked:
    """A LaurentSeries with its ceiling (see the module docstring)."""

    __slots__ = ("s", "a", "_v", "_vset")

    def __init__(self, series: LaurentSeries, ceiling=EXACT):
        self.s = series
        self.a = ceiling
        self._vset = False
        self._v = None

    @property
    def v(self):
        if not self._vset:
            self._v = certain_valuation(self.s)
            self._vset = True
        return self._v

    @classmethod
    def const(cls, value):
        # an exact rational enters as a ball containing it, so the radii downstream bound the error
        if isinstance(value, Fraction):
            return cls(LaurentSeries.rational(value), EXACT)
        return cls(LaurentSeries.constant(value), EXACT)

    @classmethod
    def monomial(cls, power: int, value=1):
        if power > _ts.T_HI:
            raise WindowDeficit(power - _ts.T_HI, " above")
        return cls(LaurentSeries.monomial(power, value), EXACT)

    @staticmethod
    def _wrap(other):
        return other if isinstance(other, Tracked) else Tracked.const(other)

    def is_zero(self) -> bool:
        return self.s.is_zero()

    def __add__(self, other):
        if not isinstance(other, Tracked):
            return Tracked(self.s + other, self.a)
        return Tracked(self.s + other.s, _min_ceiling(self.a, other.a))

    __radd__ = __add__

    def __neg__(self):
        return Tracked(-self.s, self.a)

    def __sub__(self, other):
        if not isinstance(other, Tracked):
            return Tracked(self.s - other, self.a)
        return Tracked(self.s - other.s, _min_ceiling(self.a, other.a))

    def __rsub__(self, other):
        return Tracked._wrap(other) - self

    def __mul__(self, other):
        if not isinstance(other, Tracked):
            return Tracked(self.s * other, self.a)
        a = product_ceiling(self.a, self.v, other.a, other.v)
        return Tracked(self.s * other.s, a)

    __rmul__ = __mul__

    def __pow__(self, n: int):
        out = Tracked.const(1)
        for _ in range(int(n)):
            out = out * self
        return out

    def reciprocal(self):
        v = self.v
        if v is None:
            raise ZeroDivisionError("reciprocal of a zero series")
        if -v < _ts.T_LO:
            raise WindowDeficit(_ts.T_LO + v, " below")
        # 1/x at t^(-v + k) reads x up to t^(v + k): exact up to a - 2 v
        a = EXACT if (self.a is EXACT and v == 0 and self.s._const is not _ts._NOT_CONSTANT) else _cap(
            None if self.a is EXACT else self.a - 2 * v)
        # The ball may establish a small but nonzero leading coefficient. The
        # series' relative structural-zero heuristic must not discard it again.
        return Tracked(self.s.reciprocal(valuation=v), a)

    def __truediv__(self, other):
        if not isinstance(other, Tracked):
            return Tracked(self.s / other, self.a)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return Tracked._wrap(other) * self.reciprocal()

    def sqrt(self):
        v = self.v
        if v is None:
            return Tracked(LaurentSeries.zero(), EXACT)
        # sqrt(x) at t^(v/2 + k) reads x up to t^(v + k): exact up to a - v/2
        return Tracked(self.s.sqrt(valuation=v), _cap(None if self.a is EXACT else self.a - v // 2))

    def mirrored(self):
        """This series at ``-t`` (:meth:`LaurentSeries.mirrored`), with its ceiling."""
        return Tracked(self.s.mirrored(), self.a)

    def snapped(self, exact_zero: bool):
        """With its ``t^0`` coefficient set to an exact zero when that is exactly zero (a collision the rounding left
        at 1e-dps).  Set in place in the coefficient list: subtracting the coefficient read back through mpmath
        leaves the bits flint carries past mpmath's precision."""
        if not exact_zero:
            return self
        return Tracked(zero_t0(self.s), self.a)


def zero_t0(series: LaurentSeries) -> LaurentSeries:
    """``series`` with its ``t^0`` coefficient replaced by an exact zero."""
    fl = series._fl
    if fl is not None and series._coeffs is None:
        cc = list(fl.P.coeffs())
        if _ts._OFF < len(cc):
            cc[_ts._OFF] = (_ts._ARB if fl.real else _ts._ACB)(0)
        _ts._FLINT_CTX.prec = fl.prec
        _ts._FLINT_CTX.cap = _ts._CAP
        ctor = _ts._ARB_SERIES if fl.real else _ts._ACB_SERIES
        return LaurentSeries._from_flint(_ts._FlintLaurent(ctor(cc), fl.prec, fl.real))
    coeffs = list(series.coeffs)
    coeffs[-_ts.T_LO] = _ts._MPC_ZERO
    return LaurentSeries._wrap(coeffs)


def balanced_product(factors):
    factors = list(factors)
    if not factors:
        return Tracked.const(1)
    while len(factors) > 1:
        nxt = []
        it = iter(factors)
        for a in it:
            b = next(it, None)
            nxt.append(a if b is None else a * b)
        factors = nxt
    return factors[0]


def recognized_rational(value):
    """``value`` as the rational it is the rounding of, or None: the candidate of
    :func:`virasoro_blocks.torus_onept._recognized_fraction` (denominator <= 10^6), kept only when ``value`` is within
    a few units of its own last place of it.  That candidate alone matches to half the digits, and at 25 digits a
    fraction with a 10^6 denominator is that close to almost any real (a generic weight moved by 1e-12 and was then
    taken as exact); the rounding of a rational, a float 0.7 or an mpf 3/80 included, is within half a unit."""
    from .torus_onept import _recognized_fraction

    frac = _recognized_fraction(value)
    if frac is None or isinstance(value, (Fraction, int, str)):
        return frac
    v = mp.convert(value)
    v = v.real if isinstance(v, mp.mpc) else v
    if v == 0:
        return frac
    bits = max(53, v._mpf_[3])
    with mp.workprec(bits + 32):
        # (a few roundings of the arithmetic that formed it: eight units of its last place)
        if abs(mp.mpf(frac.numerator) / frac.denominator - v) > abs(v) * mp.mpf(2) ** (3 - bits):
            return None
    return frac


def _neg(v) -> int:
    return 0 if v is None else max(0, -v)


class LevelTable:
    """``poles`` (dicts with Tracked ``K``, ``d``, ``v``, exact constants ``d_const`` / ``v_const`` or None, ``rs``,
    ``sector``, and labels), the filled ``table`` (series), its ``ceilings`` and ``vals``, the ``need`` per entry, the
    window ``half``, and the mirror involution ``partner`` (or None; :func:`build_levels`)."""

    partner = None

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _interaction_rows(poles, order, couples, inv_cache, label, partner=None):
    """Per outer pole ``o``: ``(j, rs_j, kd series, kd ceiling, kd valuation)`` with ``kd = K_j / (v_o - d_j)``.  With
    ``partner`` (:func:`build_levels`' ``mirror``), an outer pole that mirrors an earlier one gets the valuation-only
    row ``(partner_j, rs_j, None, ceiling, valuation)`` of its partner's (``kd[o', j'] = kd[o, j](-t)``)."""
    rows = []
    for o, po in enumerate(poles):
        if partner is not None and partner[o] < o:
            rows.append(sorted(((partner[j], rs, None, a, v) for j, rs, _kd, a, v in rows[partner[o]]),
                               key=lambda e: (e[1], e[0])))
            continue
        limit = order - po["rs"]
        row = []
        for j, pi in enumerate(poles):
            if pi["rs"] > limit:
                break
            if pi["K"].is_zero() or not couples(po["sector"], pi):
                continue
            key = (label(po), label(pi))
            inv = inv_cache.get(key) if inv_cache is not None else None
            if inv is None:
                exact = (po.get("v_const") is not None and pi.get("d_const") is not None
                         and po["v_const"] == pi["d_const"])
                inv = (po["v"] - pi["d"]).snapped(exact).reciprocal()
                if inv_cache is not None:
                    inv_cache[key] = inv
            kd = pi["K"] * inv
            row.append((j, pi["rs"], kd.s, kd.a, kd.v))
        rows.append(row)
    return rows


def _top_negativity(p):
    """The most negative power of the top weight ``K / (delta - d)`` over every delta: a Kac weight ``delta =
    d(c0)`` divides by ``d - d(c0)``."""
    vk = p["K"].v
    if vk is None:
        return 0
    vd = certain_valuation(zero_t0(p["d"].s))
    return max(0, (0 if vd is None else vd) - vk, -vk)


def _need(poles, rows, order):
    """The positive powers each entry must hold: ``T[j][k]`` is read at the top (at most ``_top_negativity``) and by
    each consumer ``o`` at level ``k + rs_j`` through ``kd[o, j]``, so it needs ``max(top_j, neg(kd) +
    need[o][k + rs_j])``."""
    final = [_top_negativity(p) for p in poles]
    consumers = [[] for _ in poles]
    for o, row in enumerate(rows):
        for j, _rs, _kd, _a, vkd in row:
            consumers[j].append((o, _neg(vkd)))
    need = [[final[j]] * (order - p["rs"] + 1) for j, p in enumerate(poles)]
    for k in range(order, -1, -1):
        for j, p in enumerate(poles):
            if k > order - p["rs"]:
                continue
            best = final[j]
            level = k + p["rs"]
            for o, vneg in consumers[j]:
                if level <= order - poles[o]["rs"]:
                    best = max(best, vneg + need[o][level])
            need[j][k] = best
    return need


def _fill(poles, rows, order, seeds, need, partner=None):
    """The level fill: ``T[i][k] = seed_k + sum_j kd[i, j] T[j][k - rs_j]``, each entry truncated to its need, with its
    ceiling (exact constants for the seeds).

    Each weight's and each entry's support is read once (:func:`tseries.support_of`) and every sum is formed only
    through the entry's need (:func:`tseries.conv_supports`): the products' Python bookkeeping, not their
    arithmetic, was most of a fill.  With ``partner`` an entry whose pole mirrors an earlier one is its partner's
    at ``-t`` (exact), formed after each level's own entries."""
    table = [[LaurentSeries.constant(v) for v in seeds(p["sector"], order - p["rs"])] for p in poles]
    ceil = [[EXACT] * len(row) for row in table]
    vals = [[certain_valuation(e) for e in row] for row in table]
    cctx = _ts.conv_context()
    support_of = _ts.support_of
    # an entry's constant value (a seed), else its support: the products' right factors
    right = [[_constant_or_support(e, cctx) for e in row] for row in table]
    rows = [None if partner is not None and partner[o] < o else
            [(j, rs, kd, akd, vkd, support_of(kd, cctx)) for j, rs, kd, akd, vkd in row] for o, row in enumerate(rows)]
    t_lo, t_hi = _ts.T_LO, _ts.T_HI
    mirrors = [] if partner is None else [i for i in range(len(poles)) if partner[i] < i]
    own = range(len(poles)) if partner is None else [i for i in range(len(poles)) if partner[i] >= i]
    for k in range(1, order + 1):
        for i in own:
            po = poles[i]
            if k > order - po["rs"]:
                continue
            consts, pairs = [], []
            a = ceil[i][k]
            for j, rs, kd, akd, vkd, ksup in rows[i]:
                if rs > k:
                    break
                kk = k - rs
                vin = vals[j][kk]
                if vin is None:
                    continue
                # (product_ceiling and _min_ceiling, inline)
                if vkd is not None:
                    if vkd + vin < t_lo:
                        raise WindowDeficit(t_lo - (vkd + vin), " below")
                    ain = ceil[j][kk]
                    pa = None if akd is EXACT else akd + vin
                    if ain is not EXACT and (pa is None or ain + vkd < pa):
                        pa = ain + vkd
                    if pa is None or pa > t_hi:
                        pa = t_hi
                    if a is EXACT or pa < a:
                        a = pa
                rhs = right[j][kk]
                if rhs.__class__ is tuple:
                    if ksup[0] is not None:
                        pairs.append((ksup, rhs))
                elif rhs == 1:
                    consts.append(kd)
                elif rhs != 0:
                    consts.append(kd * rhs)
            if consts or pairs:
                top = None if need is None else need[i][k]
                entry = table[i][k] + _ts.conv_supports(consts, pairs, top, cctx)
                if need is not None:
                    entry = entry.truncated_above(top)
                    a = _min_ceiling(a, top)
                table[i][k] = entry
                ceil[i][k] = _cap(a)
                vals[i][k] = v = certain_valuation(entry)
                right[i][k] = _constant_or_support(entry, cctx) if v is not None else 0
        for i in mirrors:
            if k > order - poles[i]["rs"]:
                continue
            j = partner[i]
            table[i][k] = entry = table[j][k].mirrored()
            ceil[i][k] = ceil[j][k]
            vals[i][k] = v = vals[j][k]
            right[i][k] = _constant_or_support(entry, cctx) if v is not None else 0
    return table, ceil, vals


def _constant_or_support(entry, cctx):
    """A table entry as the right factor of a product: its exact constant value (the seeds), or its support."""
    const = entry._cheap_constant_value() if isinstance(entry, LaurentSeries) else entry
    if const is not _ts._NOT_CONSTANT:
        return const
    sup = _ts.support_of(entry, cctx)
    return 0 if sup[0] is None else sup


_MEASURED_HALF: dict = {}


def _narrowed_poles(poles, from_half: int):
    """The poles built at window ``from_half``, on the current (narrower or equal) window: exact for the powers kept
    (:meth:`LaurentSeries.narrowed`), each ceiling capped at the new window's top."""
    if from_half == _ts.T_HI:
        return poles
    out = []
    for p in poles:
        q = dict(p)
        for name in ("K", "d", "v"):
            t = p[name]
            q[name] = Tracked(t.s.narrowed(from_half), t.a if t.a is EXACT else min(t.a, _ts.T_HI))
        out.append(q)
    return out


def _seeded_half(window_key, window_seeds) -> int:
    """The window measured for ``window_key``, else the widest measured for a key starting with one of
    ``window_seeds`` (the same table at another precision, a sibling table): a starting point, from which the build
    still grows only by measured deficits."""
    hit = _MEASURED_HALF.get(window_key)
    if hit is not None or not window_seeds:
        return hit or 0
    return max((v for k, v in _MEASURED_HALF.items() if any(k[:len(p)] == p for p in window_seeds)), default=0)


def build_levels(make_poles, order: int, *, seeds, couples, label, inv_cache_for=None, trim: bool = True,
                 half: int | None = None, window_key=None, window_seeds=(), mirror=None) -> LevelTable:
    """Build the level table at the narrowest window that holds it exactly.

    ``make_poles()`` returns the poles at the current window (Tracked data); ``seeds(sector, length)`` the seed
    coefficients; ``couples(outer_sector, inner_pole)`` the coupling rule; ``label(pole)`` the key of the
    externals-independent reciprocals ``1 / (v_o - d_j)``, cached in ``inv_cache_for(half)`` when given.

    The window searched starts at ``half`` (default: 1 for a trimmed table, the current window for an untrimmed one,
    whose consumer reads it at its own window) and grows only by measured deficits: a product below the window, an
    entry whose truncation analysis needs more powers than the window holds, or an entry whose ceiling falls short of
    what is read from it.  The poles -- the costly part -- are built once, at the wider of that and the current
    window, and narrowed (exactly) to each window tried; they are rebuilt only when a search passes their window.
    ``window_key`` (the locus, order and precision: the window does not depend on the externals) remembers the window
    a build measured, so the next build with that key starts there; a key not yet measured starts at the widest
    window measured under ``window_seeds`` (key prefixes).  ``mirror(poles)`` (optional): the involution ``partner``
    of a recursion invariant under ``t -> -t`` with the poles' labels swapped (``K``, ``d`` of ``partner[j]`` are
    ``j``'s at ``-t``; ``partner[j] = j`` for a self-mirrored pole): then so is every entry, and the entries and
    interaction rows of the later pole of each pair are its partner's at ``-t`` instead of a second computation.
    Restores the caller's window; the table records its own (``half``)."""
    saved = window_half()
    target = int(half) if half is not None else (1 if trim else saved)
    if window_key is not None:
        target = max(target, _seeded_half(window_key, window_seeds))
    pole_half, wide = max(saved, target), None
    try:
        while True:
            if wide is None or pole_half < target:
                pole_half = max(pole_half, target)
                set_window(pole_half)
                try:
                    wide = make_poles()
                except WindowDeficit as exc:
                    pole_half += exc.deficit
                    wide = None
                    continue
            set_window(target)
            try:
                poles = _narrowed_poles(wide, pole_half)
                partner = None if mirror is None else mirror(poles)
                rows = _interaction_rows(poles, order, couples,
                                         None if inv_cache_for is None else inv_cache_for(target), label, partner)
                need = _need(poles, rows, order) if trim else None
                widest = max((max(r) for r in need if r), default=0) if need is not None else 0
                if widest > target:
                    raise WindowDeficit(widest - target, " above (the entries' needed powers)")
                table, ceil, vals = _fill(poles, rows, order, seeds, need, partner)
            except WindowDeficit as exc:
                target += exc.deficit
                continue
            # every entry exact up to what is read from it, and the top's products (weight depth plus the entry's
            # own negative depth, at a Kac weight the deepest) inside the window
            short = 0
            for i, row in enumerate(table):
                top = _top_negativity(poles[i])
                for k in range(len(row)):
                    # (an untrimmed table has no read analysis: its consumer sizes its own window and reads)
                    a = ceil[i][k]
                    if need is not None and a is not EXACT and a < need[i][k]:
                        short = max(short, need[i][k] - a)
                    v = vals[i][k]
                    if v is not None and top - v > target:
                        short = max(short, top - v - target)
            if short:
                target += short
                continue
            if window_key is not None:
                _MEASURED_HALF[window_key] = max(target, _MEASURED_HALF.get(window_key, 0))
            # (the interaction rows -- one series per coupled pair, ~3/4 of a table's memory -- are read only by the
            # fill: a cached table does not keep them)
            return LevelTable(poles=poles, table=table, ceilings=ceil, vals=vals, need=need,
                              order=int(order), half=target, partner=partner)
    finally:
        set_window(saved)


def read_valuation(acc: LaurentSeries, full) -> int | None:
    """:func:`certain_valuation` of a ``t^0`` read ``acc`` formed only through ``t^0`` (the reads need no positive
    power), as the whole sum would give it: the structural-zero test of a ball containing zero is relative to the
    series' scale, which the positive powers set.  The truncated scale is smaller, so it can only add negative powers;
    a negative valuation decided by that test (its ball contains zero) is settled on the whole sum (``full()``)."""
    v = certain_valuation(acc)
    if v is None or v >= 0:
        return v
    fl = acc._fl
    if fl is not None and acc._coeffs is None:
        c = fl.P.coeffs()[v - _ts.T_LO]
        if not c.contains(0):
            return v
    return certain_valuation(full())


def levels_real(levels: LevelTable) -> bool:
    """Whether every series of the table (residues, locations, entries) is real-backed, so a real weight's
    ``t^0`` reads run in arb (:func:`tseries.set_real_mode`): outside the build the batch paths default to acb,
    3-4x the cost, and every evaluation converted its weights to acb rows.  Memoized on the table."""
    real = getattr(levels, "_real", None)
    if real is None:
        def ok(x):
            if x._fl is not None and x._coeffs is None:
                return x._fl.real
            return all(mp.im(c) == 0 for c in x._get_coeffs())
        real = (all(ok(p["K"].s) and ok(p["d"].s) for p in levels.poles)
                and all(ok(e) for row in levels.table for e in row))
        levels._real = real
    return real


def real_weight(delta) -> bool:
    """A real exchanged weight (a rational, or an mpmath number with zero imaginary part)."""
    if isinstance(delta, (Fraction, int)):
        return True
    if isinstance(delta, LaurentSeries):
        return False
    return mp.im(mp.convert(delta)) == 0


def eval_t0(levels: LevelTable, delta, *, sector, seed, couples, radii: list | None = None):
    """``(values, negatives)``: the ``t^0`` coefficients ``H_k = seed_k + sum_j K_j / (delta - d_j) T[j][k - rs_j]``
    for k = 0..order, and the ``(k, valuation)`` of every ``H_k`` that keeps a negative power (a genuine pole, which
    the caller reports; valuation None: ``delta`` is a pole's location at every c).  Runs on the table's window.  A weight on a Kac weight of c0 (``delta`` a recognized rational
    equal to ``d_const``) has its collision snapped exactly.  Raises :class:`WindowDeficit` when a ``t^0`` reads past
    a ceiling (the caller rebuilds wider by that much).  ``radii`` collects each ``t^0``'s ball radius."""
    poles, table, order = levels.poles, levels.table, levels.order
    saved = window_half()
    set_window(levels.half)
    saved_real = _ts.set_real_mode(True) if levels_real(levels) and real_weight(delta) else None
    try:
        if isinstance(delta, Fraction):
            dq = delta                          # (exact: enters as a ball containing it)
        else:
            delta = mp.convert(delta)
            dq = None if (isinstance(delta, mp.mpc) and delta.imag != 0) else recognized_rational(delta)
        top, everywhere = [], None
        # a mirrored table (partner): the term of a pole's mirror is the pole's own at -t (its weight K / (delta - d)
        # too: delta is t-independent), so each pair is formed once and summed as f(t) + f(-t)
        partner = getattr(levels, "partner", None)
        # (one constant series for every pole: built per pole, its flint conversion was a quarter of an evaluation)
        cdelta = Tracked.const(delta)
        for j, p in enumerate(poles):
            if p["K"].is_zero() or not couples(sector, p):
                continue
            if partner is not None and partner[j] < j:
                continue
            exact = dq is not None and p.get("d_const") is not None and dq == p["d_const"]
            gap = (cdelta - p["d"]).snapped(exact)
            if gap.is_zero():
                # the weight is this pole's location at every c (d_{1,1} = 0 for NS): a pole that no cancellation
                # can remove, reported as a genuine one from its level on
                everywhere = p["rs"] if everywhere is None else min(everywhere, p["rs"])
                continue
            w = p["K"] * gap.reciprocal()
            top.append((j, p["rs"], w, partner is not None and partner[j] != j))
        out, negatives = [], []
        for k in range(order + 1):
            ws, hs, wp, hp = [], [], [], []
            a = EXACT
            for j, rs, w, paired in top:
                if rs > k:
                    break
                vin = levels.vals[j][k - rs]
                if vin is None:
                    continue
                a = _min_ceiling(a, product_ceiling(w.a, w.v, levels.ceilings[j][k - rs], vin))
                (wp if paired else ws).append(w.s)
                (hp if paired else hs).append(table[j][k - rs])
            # (only t^0 and the negative powers are read)
            def read(upto, ws=ws, hs=hs, wp=wp, hp=hp):
                acc = t_conv_weighted_sum(ws, hs, upto) if ws else LaurentSeries.zero()
                if wp:
                    acc = acc + t_conv_weighted_sum(wp, hp, upto).symmetrized()
                return acc
            acc = read(0)
            if a is not EXACT and a < 0:
                raise WindowDeficit(-a, f" above (H_{k})")
            # A tiny deliberate external shift can leave a pole far below
            # the finite coefficient's scale. Keep it when its ball excludes
            # zero, just as the tracked intermediate products do.
            v = read_valuation(acc, lambda: read(None))
            ws = ws or wp
            if everywhere is not None and k >= everywhere:
                negatives.append((k, None))
            elif v is not None and v < 0:
                negatives.append((k, v))
            out.append(acc.coeff(0) + seed[k])
            if radii is not None:
                radii.append(_t0_radius(acc) if ws else mp.mpf(0))
        return out, negatives
    finally:
        set_window(saved)
        if saved_real is not None:
            _ts.set_real_mode(saved_real)


def _t0_radius(series: LaurentSeries):
    fl = series._fl
    if fl is None:
        return mp.mpf(0)
    cc = fl.P.coeffs()
    idx = _ts._OFF
    return mp.mpf(cc[idx].rad()) if idx < len(cc) else mp.mpf(0)


def merged_columns(levels: LevelTable, *, sector, couples, structural_tol=None, as_acb=False, discarded=None):
    """The block AT c0 as a function of the exchanged weight, from the level table: ``H_k(delta) = seed_k + sum_col
    M[k][col] / (delta - d0_col)^{a_col}``, the merged poles (each Kac location ``d0`` -- exact when the poles carry
    ``d_const`` -- with the orders its colliding poles leave after every cancellation).

    Since ``K / (delta - d) = sum_j K delta_d^j / (delta - d0)^(j + 1)`` with ``delta_d = d - d0`` of positive
    valuation, column ``(d0, j + 1)`` collects ``[t^0](K delta_d^j T[k - rs])`` over the poles at ``d0`` (``j`` while
    that ``t^0`` can be nonzero).  Each read is checked against the ceilings; a read past one raises
    :class:`WindowDeficit`.  Returns ``(locations, M, ncols)``: ``locations[col] = (d0, a, exact)`` (``exact`` the
    rational ``d0`` or None) and ``M`` a list of rows of mpmath numbers (or flint balls with ``as_acb=True``,
    retaining the error bounds through the column sums). With balls, ``discarded`` optionally collects
    ``(location, coefficient absolute bounds)`` for pruned columns, whose uncertainty still enters evaluations.
    Runs on the table's window."""
    from flint import acb

    poles, table, order = levels.poles, levels.table, levels.order
    saved = window_half()
    set_window(levels.half)
    try:
        cols: dict = {}
        locations: list = []
        entries: list = []
        for j, p in enumerate(poles):
            if p["K"].is_zero() or not couples(sector, p):
                continue
            exact = p.get("d_const")
            if exact is not None:
                d0 = mp.mpf(exact.numerator) / exact.denominator
                key = ("exact", exact)
                delta = Tracked(zero_t0(p["d"].s - LaurentSeries.constant(d0)), p["d"].a)
            else:
                d0 = mp.mpc(p["d"].s.coeff(0))
                key = ("num", mp.nstr(d0, max(10, mp.mp.dps // 2)))
                delta = p["d"] - Tracked.const(d0)
            row = table[j]
            vals = levels.vals[j]
            v_rows = [v for v in vals if v is not None]
            if not v_rows:
                continue
            val_h = min(v_rows)
            E, power = p["K"], 0
            while E.v is not None and E.v <= -val_h:
                for l, entry in enumerate(row):
                    vin = vals[l]
                    if vin is None:
                        continue
                    a = product_ceiling(E.a, E.v, levels.ceilings[j][l], vin)
                    if a is not EXACT and a < 0:
                        raise WindowDeficit(-a, " above (a merged column)")
                    product = E.s * entry
                    value = product.coeff(0)
                    ball = acb(product._as_flint(_ts._flint_prec_bits()).P[_ts._OFF]) if as_acb else None
                    if value != 0 or (as_acb and not ball.is_zero()):
                        col = cols.get((key, power + 1))
                        if col is None:
                            col = cols[(key, power + 1)] = len(locations)
                            locations.append((d0, power + 1, exact))
                        entries.append((p["rs"] + l, col, value, ball))
                if delta.v is None:
                    break
                E, power = E * delta, power + 1
        ncols = len(locations)
        M = [[mp.mpc(0)] * ncols for _ in range(order + 1)]
        balls = [[acb(0)] * ncols for _ in range(order + 1)] if as_acb else None
        for k, col, value, ball in entries:
            if k <= order:
                M[k][col] += mp.mpc(value)
                if as_acb:
                    balls[k][col] += ball
        # a location's order is its highest column with an entry above the structural-zero tolerance of the
        # location's largest entry (the rule of the Laurent valuations): higher columns are cancellation residue
        tol = _ts._valuation_tol() if structural_tol is None else structural_tol
        keep = []
        by_key: dict = {}
        for (key, a), col in cols.items():
            by_key.setdefault(key, []).append((a, col))
        for key, cs in by_key.items():
            scale = max((abs(M[k][col]) for _a, col in cs for k in range(order + 1)), default=mp.mpf(0))
            for a, col in cs:
                if any(abs(M[k][col]) > tol * scale or (as_acb and not balls[k][col].contains(0))
                       for k in range(order + 1)):
                    keep.append(col)
                elif as_acb and discarded is not None:
                    discarded.append((locations[col], [balls[k][col].abs_upper() for k in range(order + 1)]))
        keep.sort()
        locations = [locations[c] for c in keep]
        M = [[row[c] for c in keep] for row in (balls if as_acb else M)]
        return locations, M, len(keep)
    finally:
        set_window(saved)
