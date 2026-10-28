"""Truncated Laurent series in t, for exact regularization at singular c.

At a recursion-singular central charge c0 the Zamolodchikov h-recursion blows
up: the residue factor A_rs has b-poles and the Kac weights d_rs resonate.
Setting c = c0 + sigma t^p cures this (p = 2 at c0 = 1, 25, else 1;
sigma = -1 at c0 = 1, else +1; see params.laurent_t_power and laurent_t_sq_sign).
b is analytic in the chosen t even at the square-root branch points c = 1
and c = 25, so A_rs, d_rs and
the fusion polynomials become *Laurent* series in t whose negative powers
cancel when the recursion terms are summed. The physical block is the t^0
coefficient.

`LaurentSeries` carries coefficients on a fixed power window [T_LO, T_HI].
Structural zeros -- e.g. a resonant factor m*b + n/b that vanishes at t = 0 --
are detected by magnitude: a coefficient below 10^(-dps/2) of the series scale
is treated as zero when locating the valuation. The whole computation runs at
elevated precision, so genuine O(1) coefficients and structural ~10^(-dps)
zeros are separated by a wide margin.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partialmethod

import mpmath as mp
from .numeric_input import _integer_control

# Power window. A resonant A_rs factor contributes one simple t-pole, and the
# deepest A_rs pole reachable at the q-orders used here is a handful of powers;
# the symmetric window leaves ample headroom on both sides.
T_LO = -12
T_HI = 12
_WIDTH = T_HI - T_LO + 1
_MPC_ZERO = mp.mpc(0)
_NOT_CONSTANT = object()

# Fixed-window flint offset: the native handle stores coefficients of powers
# T_LO .. T_HI in slots 0 .. _WIDTH-1, i.e. value = t^T_LO * P. _OFF is the
# index of power 0, and the carry applied after a (base-2*T_LO) multiply.
_OFF = -T_LO

# The resting flint series cap (constructions, add, scale). The products
# (_window_product, reciprocal, u_convolve) set their own cap to the length the
# window keeps and restore this one.
_CAP = _WIDTH + _OFF


def window_half() -> int:
    """Current Laurent window half-width (powers run t^[-half, +half])."""
    return T_HI


def set_window(half: int) -> None:
    """Resize the symmetric Laurent window to powers t^[-half, +half].

    Updates every derived layout constant. Series built before/after a change
    must not be mixed, so callers set the window before constructing an
    evaluator's coefficients and keep it fixed for that whole computation.
    """
    global T_LO, T_HI, _WIDTH, _OFF, _CAP
    half = _integer_control('window half-width', half, 1)
    T_LO = -half
    T_HI = half
    _WIDTH = 2 * half + 1
    _OFF = half
    _CAP = _WIDTH + _OFF
    cache = globals().get("_SCALAR_ROW_CACHE")
    if cache is not None:
        cache.clear()


def _valuation_tol() -> mp.mpf:
    """Relative threshold separating structural zeros from genuine terms."""
    return mp.mpf(10) ** (-(mp.mp.dps // 2))


def _flint_prec_bits() -> int:
    """Working bit-precision for the flint backend (matches polynomial.py)."""
    return int(mp.mp.dps * 3.33) + 20


# Flint symbols bound once at import: the hot ``_FlintLaurent`` methods read
# these module globals instead of re-running ``from flint import ...`` (an
# 8M-call import-machinery hotspot) on every multiply.
from flint import (  # noqa: E402
    acb as _ACB,
    acb_mat as _ACB_MAT,
    acb_series as _ACB_SERIES,
    arb as _ARB,
    arb_mat as _ARB_MAT,
    arb_series as _ARB_SERIES,
    ctx as _FLINT_CTX,
)

# When True, _FlintLaurent backs all-real series with arb_series (real arithmetic,
# ~3-4x cheaper mul and inv than acb) and only promotes to acb on a genuinely
# complex operand. _REAL_MODE is the "decide once" switch (mirroring
# derivs._use_arb): the scorer entry point (regularization._build_series_adaptive)
# and recursion.H_q_series_laurent_precompute assert -- from the realness of c0,
# the externals and the geometry/sigma scalars -- that the WHOLE t-series build is
# real (b(c0) real at c0 >= 25, or purely imaginary at c0 <= 1 and carried as the
# real beta = -i b), so from_mpc takes the arb backing WITHOUT a per-construction
# realness scan and the batch contractions (t_conv_weighted_sum / u_convolve) run
# in arb. Off (per-construction scan) is the safe default, kept whenever c or an
# external is complex.
_USE_ARB_LAURENT = True
_REAL_MODE = False

_SCALAR_ROW_CACHE_MAXSIZE = 512
_SCALAR_ROW_CACHE = OrderedDict()


def set_real_mode(flag: bool) -> bool:
    """Set the global real-mode switch; return the previous value.

    Callers (the scorer entry point, the Laurent H-table build) decide once whether the whole t-series build
    is real and wrap the build in ``set_real_mode(True)`` / restore. Returns the
    prior value so the caller can restore it in a ``finally``.
    """
    global _REAL_MODE
    prev = _REAL_MODE
    _REAL_MODE = bool(flag)
    return prev


def _acb_from_mpc(value):
    """Convert an mpmath scalar to a flint acb at the current precision."""
    z = value if isinstance(value, mp.mpc) else mp.mpc(value)
    return _ACB(mp.mpf(z.real), mp.mpf(z.imag))


def _acb_to_mpc(c):
    """Convert a flint acb coefficient back to mp.mpc (midpoint, no radius)."""
    return mp.make_mpc((c.real._mpf_, c.imag._mpf_))


def _scalar_row_cache_key(x, real):
    """Cache key for scalar coefficient rows at the current precision."""
    if isinstance(x, (mp.mpf, mp.mpc)):
        if isinstance(x, mp.mpc) and x.imag == 0:
            x = mp.mpf(x.real)
    if isinstance(x, mp.mpf):
        return ("mpf", real, mp.mp.prec, _WIDTH, _OFF, x._mpf_)
    if isinstance(x, mp.mpc):
        return ("mpc", real, mp.mp.prec, _WIDTH, _OFF, x._mpc_)
    if isinstance(x, int):
        return ("int", real, mp.mp.prec, _WIDTH, _OFF, x)
    if isinstance(x, float):
        return ("mpf", real, mp.mp.prec, _WIDTH, _OFF, mp.mpf(x)._mpf_)
    if isinstance(x, complex):
        return ("mpc", real, mp.mp.prec, _WIDTH, _OFF, mp.mpc(x)._mpc_)
    return None


_ARB_ZERO_MPF = (0, 0, 0, 0)  # mpmath _mpf_ for 0.0


def _arb_to_mpc(a):
    """Convert a flint arb coefficient to mp.mpc (real, zero imaginary)."""
    return mp.make_mpc((a._mpf_, _ARB_ZERO_MPF))


def _arb_from_mpf(value):
    """Convert a real scalar to flint arb at current precision.

    The hot Laurent-score paths sometimes hand us a Flint ``arb`` interval
    directly; accept that as-is instead of round-tripping through ``mp.mpf``,
    which can reject nonzero-width intervals.
    """
    if isinstance(value, _ARB):
        return value
    if isinstance(value, _ACB):
        return value.real
    return _ARB(value)


def _is_real_mpc(x) -> bool:
    return (x.imag == 0) if isinstance(x, mp.mpc) else (mp.mpc(x).imag == 0)


def _to_acb_series(P, is_real):
    """Return an acb_series view of P (identity if already acb)."""
    if not is_real:
        return P
    acb_series, acb = _ACB_SERIES, _ACB
    return acb_series([acb(c) for c in P.coeffs()])


def _acb_mag(c):
    """arb magnitude |c| of an acb/arb coefficient.

    Returns an ``arb`` (not a Python float): at the inflated working
    precision of the regularization path (dps up to 16x base) a float would
    underflow to 0.0 both for the magnitudes themselves and for the
    structural-zero tolerance, silently misjudging is_zero/valuation. The arb
    comparison stays exact at any dps -- the acb_series multiply radius
    (~10**-dps) sits far below the tolerance (~10**(-dps/2)), so the scan is
    always decidable. (The hot is_zero/_valuation_index scans inline ``abs(c)``
    directly to skip this call's overhead; kept for callers that want a name.)
    """
    return abs(c)


class _FlintLaurent:
    """Native flint backing for a LaurentSeries on the fixed window [T_LO, T_HI].

    Represents value = sum_i P[i] * t^(T_LO + i) where P is an ``acb_series``
    whose slot i is the coefficient of t^(T_LO + i). Anchoring every handle to
    the same base T_LO makes addition/subtraction/negation exact 1:1 with the
    mpmath window (no shifting), and makes the structural-zero is_zero/valuation
    a direct tolerance scan of P -- so the hot sparsity gates never have to
    round-trip back to mpmath. Multiplication carries the base back to T_LO by
    slicing the raw convolution (flint has no series shift operator).
    """

    __slots__ = ("P", "prec", "real")

    def __init__(self, P, prec, real=False):
        self.P = P
        self.prec = prec
        # ``real`` => P is an arb_series (real backing); else an acb_series.
        self.real = real

    @classmethod
    def from_mpc(cls, coeffs, prec) -> "_FlintLaurent":
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = prec
        flint_ctx.cap = _CAP
        if _USE_ARB_LAURENT and (_REAL_MODE or all(_is_real_mpc(c) for c in coeffs)):
            arb_series = _ARB_SERIES
            return cls(
                arb_series([_arb_from_mpf(
                    c.real if isinstance(c, mp.mpc) else c) for c in coeffs]),
                prec, real=True,
            )
        acb_series = _ACB_SERIES
        return cls(acb_series([_acb_from_mpc(c) for c in coeffs]), prec)

    def acb_coeffs(self):
        """Raw coefficient list as acb objects (converts an arb backing).

        Used by the batch acb matmul paths (t_conv_weighted_sum / u_convolve),
        which assemble acb_series/acb_mat and so need acb-typed coefficients even
        when this handle is real-backed.
        """
        cc = self.P.coeffs()
        if not self.real:
            return cc
        acb = _ACB
        return [acb(c) for c in cc]

    def to_mpc(self):
        """Materialize the canonical length-_WIDTH list of mp.mpc coefficients."""
        cc = self.P.coeffs()
        out = [mp.mpc(0)] * _WIDTH
        conv = _arb_to_mpc if self.real else _acb_to_mpc
        for i, c in enumerate(cc):
            if i >= _WIDTH:
                break
            out[i] = conv(c)
        return out

    # --- structural-zero readers (tolerance-based, native) ---

    def _valuation_index(self):
        """Lowest slot index above the structural-zero tolerance, or None.

        Returns None for a numerically zero series. The scan runs entirely in
        arb so it stays exact at the inflated regularization working precision
        (where a Python-float threshold would underflow to 0.0).
        """
        cc = self.P.coeffs()
        if not cc:
            return None
        mags = [abs(c) for c in cc]  # arb magnitudes
        sc = mags[0]
        for m in mags[1:]:
            if m > sc:
                sc = m
        if not (sc > 0):  # all-zero (or radius-dominated) series
            return None
        thresh = sc * (_ARB(10) ** (-(mp.mp.dps // 2)))
        for i, m in enumerate(mags):
            if m > thresh:
                return i
        return None

    def is_zero(self) -> bool:
        # Structurally zero iff every coefficient has zero magnitude -- the same
        # ``not (sc > 0)`` test _valuation_index makes, but short-circuiting on
        # the first genuinely nonzero coefficient instead of building the full
        # magnitude list and max scan. Valuation 0 dominates the mul/reciprocal
        # chains, so a nonzero series returns on its first coefficient; only the
        # genuinely all-zero sparse u-row placeholders scan to the end (the same
        # cost as before). ``abs(c) > 0`` is the arb comparison, so a midpoint-0
        # radius-only coefficient is still treated as zero (consistent).
        for c in self.P.coeffs():
            if abs(c) > 0:
                return False
        return True

    def valuation_power(self):
        i0 = self._valuation_index()
        return None if i0 is None else T_LO + i0

    def coeff(self, power):
        cc = self.P.coeffs()
        idx = power - T_LO
        if 0 <= idx < len(cc):
            return (_arb_to_mpc if self.real else _acb_to_mpc)(cc[idx])
        return mp.mpc(0)

    # --- arithmetic (all results re-anchored to base T_LO) ---

    def _binop_operands(self, other):
        """Return (P_self, P_other, real, series_ctor) promoting to a common type.

        Same-type (both real or both complex) stays native; a mixed pair promotes
        the real operand to acb so the convolution/addition runs in acb.
        """
        if self.real == other.real:
            ctor = _ARB_SERIES if self.real else _ACB_SERIES
            return self.P, other.P, self.real, ctor
        return (_to_acb_series(self.P, self.real),
                _to_acb_series(other.P, other.real), False, _ACB_SERIES)

    def neg(self) -> "_FlintLaurent":
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = self.prec
        flint_ctx.cap = _CAP
        return _FlintLaurent(-self.P, self.prec, self.real)

    def add(self, other) -> "_FlintLaurent":
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = self.prec
        flint_ctx.cap = _CAP
        ps, po, real, _ = self._binop_operands(other)
        return _FlintLaurent(ps + po, self.prec, real)

    def sub(self, other) -> "_FlintLaurent":
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = self.prec
        flint_ctx.cap = _CAP
        ps, po, real, _ = self._binop_operands(other)
        return _FlintLaurent(ps - po, self.prec, real)

    def mul(self, other) -> "_FlintLaurent":
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = self.prec
        ps, po, real, ctor = self._binop_operands(other)
        ca = ps.coeffs()
        cb = po.coeffs()
        la, ha = _support(ca)
        lb, hb = _support(cb)
        return _FlintLaurent(ctor(_window_product(ca, la, ha, cb, lb, hb, ctor)), self.prec, real)

    def scale(self, factor) -> "_FlintLaurent":
        """Multiply by a scalar: value -> factor * value (no series convolution)."""
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = self.prec
        flint_ctx.cap = _CAP
        if self.real and _is_real_mpc(factor):
            f = factor.real if isinstance(factor, mp.mpc) else factor
            return _FlintLaurent(self.P * _arb_from_mpf(f), self.prec, True)
        P = _to_acb_series(self.P, self.real)
        return _FlintLaurent(P * _acb_from_mpc(mp.mpc(factor)), self.prec)

    def add_scalar(self, value) -> "_FlintLaurent":
        """Add a scalar: value -> value + scalar (offsets the power-0 term)."""
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = self.prec
        flint_ctx.cap = _CAP
        if self.real and _is_real_mpc(value):
            arb_series, arb = _ARB_SERIES, _ARB
            v = value.real if isinstance(value, mp.mpc) else value
            add = arb_series([arb(0)] * _OFF + [_arb_from_mpf(v)])
            return _FlintLaurent(self.P + add, self.prec, True)
        acb_series, acb = _ACB_SERIES, _ACB
        add = acb_series([acb(0)] * _OFF + [_acb_from_mpc(mp.mpc(value))])
        return _FlintLaurent(_to_acb_series(self.P, self.real) + add, self.prec)

    def reciprocal(self, *, valuation=None) -> "_FlintLaurent":
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = self.prec
        i0 = self._valuation_index() if valuation is None else valuation + _OFF
        if i0 is None:
            raise ZeroDivisionError("reciprocal of a zero LaurentSeries")
        cc = self.P.coeffs()
        ctor = _ARB_SERIES if self.real else _ACB_SERIES
        zero = _arb_from_mpf(0) if self.real else _acb_from_mpc(0)
        # Q = value / t^(T_LO+i0) is valuation 0; 1/value = t^-(T_LO+i0) * Q^-1.
        # In the fixed window, P_r[base_j + k] = Qinv[k] with base_j = 2*_OFF-i0,
        # so only the first _WIDTH - base_j terms of Q^-1 are kept (inverted to that length).
        base_j = 2 * _OFF - i0
        flint_ctx.cap = max(1, _WIDTH - base_j)
        qi = ctor(cc[i0:]).inv().coeffs()
        flint_ctx.cap = _CAP
        return _FlintLaurent(ctor([zero] * base_j + qi), self.prec, self.real)

    def sqrt(self, *, valuation=None) -> "_FlintLaurent":
        """The principal square root (of the leading coefficient), in balls: ``t^(v/2) sqrt(Q)`` with ``Q`` the
        valuation-0 part, kept to the window.  A real series with a negative leading coefficient goes complex."""
        flint_ctx = _FLINT_CTX
        flint_ctx.prec = self.prec
        i0 = self._valuation_index() if valuation is None else valuation + _OFF
        if i0 is None:
            return self
        v = T_LO + i0
        if v % 2 != 0:
            raise ValueError("sqrt of an odd-valuation LaurentSeries")
        base_j = v // 2 + _OFF
        if base_j < 0:
            raise ValueError("sqrt underflows the Laurent window")
        cc = self.P.coeffs()
        real = self.real and cc[i0] > 0
        ctor = _ARB_SERIES if real else _ACB_SERIES
        lead = cc[i0:] if real or not self.real else [_ACB(c) for c in cc[i0:]]
        flint_ctx.cap = max(1, _WIDTH - base_j)
        qs = ctor(lead).sqrt().coeffs()
        flint_ctx.cap = _CAP
        zero = _ARB(0) if real else _ACB(0)
        return _FlintLaurent(ctor([zero] * base_j + qs), self.prec, real)


def _support(cc):
    """``(lo, hi)``: the slots outside ``[lo, hi)`` of a coefficient list are exact zeros."""
    lo = 0
    hi = len(cc)
    while lo < hi and cc[lo].is_zero():
        lo += 1
    while hi > lo and cc[hi - 1].is_zero():
        hi -= 1
    return lo, hi


def _window_product(ca, la, ha, cb, lb, hb, ctor):
    """The base-T_LO window (at most _WIDTH slots) of the product of two base-T_LO coefficient rows.

    Only the supports ``ca[la:ha]``, ``cb[lb:hb]`` are multiplied, and only to the length the window keeps: a
    window row is typically half empty (valuation >= 0), and the full-width product took twice as long. Slot i
    times slot j lands in slot i + j - _OFF.
    """
    start = la + lb - _OFF
    n = min(ha - la + hb - lb - 1, _WIDTH - start)
    if la >= ha or lb >= hb or n <= 0:
        return []
    flint_ctx = _FLINT_CTX
    flint_ctx.cap = n
    # (flint multiplies the operands whole whatever the cap: only their first n coefficients reach the n kept)
    prod = (ctor(ca[la:min(ha, la + n)]) * ctor(cb[lb:min(hb, lb + n)])).coeffs()
    flint_ctx.cap = _CAP
    if start > 0 and prod:
        return [(_ARB if ctor is _ARB_SERIES else _ACB)(0)] * start + prod
    return prod[-start:] if start < 0 else prod


class _LaurentCoefficients(list):
    """Fixed-width public coefficients that invalidate their owner's cached data."""

    __slots__ = ("_owner",)

    def __init__(self, values, owner):
        import weakref

        super().__init__(values)
        self._owner = weakref.ref(owner)

    def _mutate(self, operation, *args, **kwargs):
        original = self.copy()
        try:
            result = operation(self, *args, **kwargs)
            if len(self) != len(original):
                raise ValueError("Laurent coefficients must retain their fixed window length")
        except BaseException:
            list.__setitem__(self, slice(None), original)
            raise
        owner = self._owner()
        if owner is not None and owner._coeffs is self:
            owner._invalidate_cache()
        return result

    __setitem__ = partialmethod(_mutate, list.__setitem__)
    __delitem__ = partialmethod(_mutate, list.__delitem__)
    append = partialmethod(_mutate, list.append)
    extend = partialmethod(_mutate, list.extend)
    insert = partialmethod(_mutate, list.insert)
    pop = partialmethod(_mutate, list.pop)
    remove = partialmethod(_mutate, list.remove)
    clear = partialmethod(_mutate, list.clear)
    reverse = partialmethod(_mutate, list.reverse)
    sort = partialmethod(_mutate, list.sort)
    __iadd__ = partialmethod(_mutate, list.__iadd__)
    __imul__ = partialmethod(_mutate, list.__imul__)

    def __reduce__(self):
        # A copied/pickled series must bind its coefficient view to its own
        # caches on first access, rather than keep the original owner's link.
        return list, (list(self),)


class LaurentSeries:
    """A Laurent series sum_{k=T_LO}^{T_HI} coeffs[k - T_LO] * t^k.

    Carries one of two backing stores (at least one is always present):
    ``_coeffs`` is the canonical list of _WIDTH mp.mpc coefficients, and
    ``_fl`` is an optional native :class:`_FlintLaurent` handle. Arithmetic
    keeps the whole chain behind flint (returning flint-state results without
    materializing mpmath) and only converts back when a caller reads the
    coefficients. The structural-zero gates is_zero/valuation run natively on
    the flint handle so the hot recursion never forces a round-trip.
    Public coefficient edits invalidate cached handles and structural flags;
    changing the fixed window length through the coefficient list is rejected.
    """

    # ``_row`` caches the (length-_WIDTH coefficient row, support) that _coeff_row builds
    # (at precision ``_rowprec``): weights and H_table entries are fed to
    # t_conv_weighted_sum repeatedly (the same object across every k / n_terms /
    # pole), so memoizing the row avoids re-running ``.coeffs()`` + padding on each.
    # ``_zero`` memoizes the structural-zero state of an mpmath-backed series
    # (True/False, or None = not yet computed). zero() pre-sets True so the sparse
    # all-zero u-row placeholders short-circuit is_zero without an O(_WIDTH)
    # coefficient scan (the dominant churn). In-place mutators build via _zeros()
    # (flag None) so a mutated zero never carries a stale True.
    __slots__ = ("_coeffs", "_fl", "_row", "_rowprec", "_supser", "_zero", "_const", "__weakref__")

    def __init__(self, coeffs):
        c = [mp.mpc(x) for x in coeffs]
        if len(c) != _WIDTH:
            raise ValueError(f"coeffs must have length {_WIDTH}")
        self._coeffs = c
        self._fl = None
        self._row = None
        self._rowprec = -1
        self._supser = None
        self._zero = None
        self._const = _NOT_CONSTANT

    @property
    def coeffs(self):
        """Mutable fixed-width coefficients; edits invalidate cached arithmetic."""
        values = self._get_coeffs()
        if not isinstance(values, _LaurentCoefficients):
            values = _LaurentCoefficients(values, self)
            self._coeffs = values
        return values

    def _get_coeffs(self):
        if self._coeffs is None:
            self._coeffs = self._fl.to_mpc()
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        value = list(value)
        if len(value) != _WIDTH:
            raise ValueError(f"coeffs must have length {_WIDTH}")
        self._coeffs = value
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._fl = None
        self._row = None
        self._rowprec = -1
        self._supser = None
        self._zero = None
        self._const = _NOT_CONSTANT

    def __getstate__(self):
        # Native flint handles are not picklable. Preserve coefficient values
        # and rebuild the derived flags/handles, as for Polynomial snapshots.
        return {"coeffs": self._get_coeffs().copy()}

    def __setstate__(self, state):
        self.coeffs = state["coeffs"]

    @classmethod
    def _wrap(cls, mpc_coeffs) -> "LaurentSeries":
        """Fast constructor for a list of `_WIDTH` mp.mpc values already in the
        right type and length -- skips the per-coefficient mp.mpc() conversion
        that `__init__` performs. Only call from arithmetic that produces mpc
        results of the correct length."""
        s = cls.__new__(cls)
        s._coeffs = mpc_coeffs
        s._fl = None
        s._row = None
        s._rowprec = -1
        s._supser = None
        s._zero = None
        s._const = _NOT_CONSTANT
        return s

    @classmethod
    def _from_flint(cls, fl: "_FlintLaurent") -> "LaurentSeries":
        """Wrap a native flint handle as a flint-state series (coeffs lazy)."""
        s = cls.__new__(cls)
        s._coeffs = None
        s._fl = fl
        s._row = None
        s._rowprec = -1
        s._supser = None
        s._zero = None
        s._const = _NOT_CONSTANT
        return s

    def truncated_above(self, power: int) -> "LaurentSeries":
        """This series with every coefficient of ``t^k``, ``k > power``, set to zero (the powers no later product's
        ``t^0`` can reach: an exact truncation for that use, and a narrower support for the products)."""
        idx = int(power) - T_LO + 1
        if idx >= _WIDTH:
            return self
        if idx <= 0 or self._zero is True:
            return LaurentSeries.zero()
        if self._const is not _NOT_CONSTANT:
            return self if power >= 0 else LaurentSeries.zero()
        fl = self._fl
        if fl is not None:
            cc = fl.P.coeffs()
            if len(cc) <= idx:
                return self
            _FLINT_CTX.prec = fl.prec
            _FLINT_CTX.cap = _CAP
            series = _ARB_SERIES if fl.real else _ACB_SERIES
            return LaurentSeries._from_flint(_FlintLaurent(series(list(cc[:idx])), fl.prec, real=fl.real))
        coeffs = list(self._coeffs)
        return LaurentSeries._wrap(coeffs[:idx] + [type(coeffs[0])(0)] * (len(coeffs) - idx))

    def mirrored(self) -> "LaurentSeries":
        """This series at ``-t``: every odd power's coefficient negated (exact, balls included).  At the branch points
        ``c0 = 1, 25`` the Kac data of ``(s, r)`` are those of ``(r, s)`` at ``-t`` (``b^2(-t) = 1 / b^2(t)``)."""
        if self._zero is True or self._const is not _NOT_CONSTANT:
            return self
        odd = T_LO % 2
        fl = self._fl
        if fl is not None:
            cc = fl.P.coeffs()
            _FLINT_CTX.prec = fl.prec
            _FLINT_CTX.cap = _CAP
            series = _ARB_SERIES if fl.real else _ACB_SERIES
            return LaurentSeries._from_flint(_FlintLaurent(
                series([-c if (i + odd) % 2 else c for i, c in enumerate(cc)]), fl.prec, real=fl.real))
        return LaurentSeries._wrap([-c if (i + odd) % 2 else c for i, c in enumerate(self._coeffs)])

    def symmetrized(self) -> "LaurentSeries":
        """``f(t) + f(-t)``: the even coefficients doubled, the odd ones exact zeros (they cancel exactly)."""
        if self._zero is True:
            return self
        if self._const is not _NOT_CONSTANT:
            return LaurentSeries.constant(2 * self._const)
        odd = T_LO % 2
        fl = self._fl
        if fl is not None:
            cc = fl.P.coeffs()
            _FLINT_CTX.prec = fl.prec
            _FLINT_CTX.cap = _CAP
            series = _ARB_SERIES if fl.real else _ACB_SERIES
            zero = (_ARB if fl.real else _ACB)(0)
            return LaurentSeries._from_flint(_FlintLaurent(
                series([zero if (i + odd) % 2 else 2 * c for i, c in enumerate(cc)]), fl.prec, real=fl.real))
        return LaurentSeries._wrap([_MPC_ZERO if (i + odd) % 2 else 2 * c for i, c in enumerate(self._coeffs)])

    def narrowed(self, from_half: int) -> "LaurentSeries":
        """This series, built at window half-width ``from_half`` >= the current one, on the current window.

        Exact: a narrower window only drops the outer powers, and the powers it keeps are those a build at the
        narrower window computes (at least as accurately: the wide build kept more cross terms).  Lets one table
        built at the widest window a computation needs serve every narrower one."""
        from_half = int(from_half)
        if from_half == T_HI:
            return self
        if from_half < T_HI:
            raise ValueError(f"cannot widen a series built at half-width {from_half} to {T_HI}")
        if self._zero is True:
            return LaurentSeries.zero()
        if self._const is not _NOT_CONSTANT:
            return LaurentSeries.constant(self._const)
        off = from_half - T_HI
        fl = self._fl
        if fl is not None:
            cc = fl.P.coeffs()[off: off + _WIDTH]
            _FLINT_CTX.prec = fl.prec
            _FLINT_CTX.cap = _CAP
            series = _ARB_SERIES if fl.real else _ACB_SERIES
            return LaurentSeries._from_flint(_FlintLaurent(series(list(cc)), fl.prec, real=fl.real))
        return LaurentSeries._wrap(list(self._coeffs[off: off + _WIDTH]))

    def _as_flint(self, prec: int) -> "_FlintLaurent":
        """Return a native handle at `prec`, building/caching from coeffs if needed."""
        fl = self._fl
        if fl is not None and fl.prec == prec:
            return fl
        fl = _FlintLaurent.from_mpc(self._get_coeffs(), prec)
        if self._coeffs is not None:
            self._fl = fl
        return fl

    def _flint_binop(self, other: "LaurentSeries", name: str) -> "LaurentSeries":
        """The flint-backed binary op ``name`` (add / sub / mul) as a LaurentSeries."""
        prec = _flint_prec_bits()
        res = getattr(self._as_flint(prec), name)(other._as_flint(prec))
        return LaurentSeries._from_flint(res)

    @classmethod
    def _zeros(cls) -> "LaurentSeries":
        # Fresh all-zero series WITHOUT the memoized-zero flag, for callers that
        # immediately mutate slots in place (constant/monomial/reciprocal/sqrt):
        # the flag would otherwise be a stale True after the mutation.
        return cls._wrap([_MPC_ZERO] * _WIDTH)

    @classmethod
    def zero(cls) -> "LaurentSeries":
        # ``_wrap`` of a fresh list of references to the cached _MPC_ZERO avoids
        # __init__'s per-coefficient ``mp.mpc()`` re-conversion (this is called
        # hundreds of thousands of times as the accumulator seed in the series
        # row contractions). The list is fresh per call, so a later in-place
        # mutation never touches another series' coeffs, and _MPC_ZERO itself is
        # never mutated (slots are reassigned, not mutated). Flagged zero so the
        # sparse u-row placeholders answer is_zero() without a coefficient scan.
        s = cls._wrap([_MPC_ZERO] * _WIDTH)
        s._zero = True
        s._const = _MPC_ZERO
        return s

    @classmethod
    def constant(cls, value) -> "LaurentSeries":
        s = cls._zeros()
        v = mp.mpc(value)
        s._get_coeffs()[-T_LO] = v
        s._zero = (v == 0)
        s._const = v
        return s

    @classmethod
    def rational(cls, value) -> "LaurentSeries":
        """The constant ``value`` (a ``Fraction``) as a ball that contains it.  A rational rounded to a midpoint
        carries its rounding outside every ball radius downstream, so a radius would not bound the error of a
        result built from it; a dyadic one is exact either way."""
        den = value.denominator
        if den & (den - 1) == 0:
            return cls.constant(mp.mpf(value.numerator) / den)
        prec = _flint_prec_bits()
        _FLINT_CTX.prec = prec
        _FLINT_CTX.cap = _CAP
        coeffs = [_ARB(0)] * (_OFF + 1)
        coeffs[_OFF] = _ARB(value.numerator) / den
        if _USE_ARB_LAURENT:
            return cls._from_flint(_FlintLaurent(_ARB_SERIES(coeffs), prec, real=True))
        return cls._from_flint(_FlintLaurent(_ACB_SERIES([_ACB(c) for c in coeffs]), prec))

    @classmethod
    def monomial(cls, power: int, value=1) -> "LaurentSeries":
        if power < T_LO or power > T_HI:
            raise ValueError(f"power {power} outside window [{T_LO}, {T_HI}]")
        s = cls._zeros()
        v = mp.mpc(value)
        s._get_coeffs()[power - T_LO] = v
        s._zero = (v == 0)
        if power == 0:
            s._const = v
        return s

    @staticmethod
    def _promote(value) -> "LaurentSeries":
        if isinstance(value, LaurentSeries):
            return value
        return LaurentSeries.constant(value)

    def coeff(self, power: int):
        if self._coeffs is None and self._fl is not None:
            if power < T_LO or power > T_HI:
                return mp.mpc(0)
            return self._fl.coeff(power)
        if power < T_LO or power > T_HI:
            return mp.mpc(0)
        return self._get_coeffs()[power - T_LO]

    def scale(self) -> mp.mpf:
        return max((abs(c) for c in self._get_coeffs()), default=mp.mpf(0))

    def is_zero(self) -> bool:
        if self._zero is not None:
            return self._zero
        if self._coeffs is None and self._fl is not None:
            z = self._fl.is_zero()
            self._zero = z
            return z
        z = not any(self._get_coeffs())
        self._zero = z
        return z

    def valuation(self):
        """Lowest power with a coefficient above the structural-zero tolerance.

        Returns None for a numerically zero series.
        """
        if self._coeffs is None and self._fl is not None:
            return self._fl.valuation_power()
        sc = self.scale()
        if sc == 0:
            return None
        thresh = sc * _valuation_tol()
        for i, c in enumerate(self._get_coeffs()):
            if abs(c) > thresh:
                return T_LO + i
        return None

    # --- arithmetic ---

    def _scalar_op(self, value, kind: str):
        """Scalar +/-/* fast path, avoiding promotion of `value` to a full
        Laurent series and its flint conversion (the dominant cost in the
        Kac/Laurent build, where every ``m*b`` / ``b + scalar`` otherwise
        converts a 24-zeros constant series). Operates on the cached flint
        handle when present, else on the mpc coefficients. Returns a
        LaurentSeries, or ``None`` to fall back to the general path."""
        prec = _flint_prec_bits()
        fl = self._fl
        if fl is not None and fl.prec == prec:
            if kind == "mul":
                return LaurentSeries._from_flint(fl.scale(value))
            if kind == "add":
                return LaurentSeries._from_flint(fl.add_scalar(value))
            if kind == "sub":
                return LaurentSeries._from_flint(fl.add_scalar(-value))
        coeffs = self._coeffs
        if coeffs is None:
            return None
        if kind == "mul":
            v = mp.mpc(value)
            return LaurentSeries._wrap([c * v for c in coeffs])
        v = mp.mpc(value)
        out = list(coeffs)
        if kind == "add":
            out[-T_LO] = out[-T_LO] + v
        else:  # sub
            out[-T_LO] = out[-T_LO] - v
        return LaurentSeries._wrap(out)

    def _is_cheap_zero(self) -> bool:
        """True if this is the exact (structural) zero series, checked *cheaply*.

        Only consults an already-materialized mpmath coefficient list (the state
        of every ``LaurentSeries.zero()`` accumulator); a flint-backed handle
        returns False rather than paying the arb tolerance scan, so the fast
        path below never adds cost to the non-zero flint chain. Exact-zero only
        (a numerically-tiny-but-nonzero coefficient is *not* fast-pathed), so no
        precision is lost.
        """
        c = self._coeffs
        if c is None:
            return False
        # Identity compare against the cached _MPC_ZERO -- a pointer check, no
        # bool(mpc)/mpc_is_nonzero per coefficient (that test was itself the
        # dominant call-count hotspot). Only the zero() / _u_from_numeric zeros
        # are _MPC_ZERO-backed; a computed all-zero series is simply not
        # fast-pathed (correctness is unaffected -- it falls to the flint op).
        for x in c:
            if x is not _MPC_ZERO:
                return False
        return True

    def _cheap_constant_value(self):
        """Return the exact t^0 coefficient if this is cheaply known constant.

        Only constructors that know they produced a scalar set this tag. A
        flint-backed or computed result may also be constant, but proving that
        in the hot multiply path costs more than it saves.
        """
        return self._const

    def __add__(self, other) -> "LaurentSeries":
        if not isinstance(other, LaurentSeries):
            res = self._scalar_op(other, "add")
            if res is not None:
                return res
        other = self._promote(other)
        # Structural-zero fast path: skip the flint conversion + add for the
        # ``acc = LaurentSeries.zero(); acc = acc + term`` accumulator pattern
        # that pervades the series row contractions. Returns the other operand
        # (arithmetic results are treated immutably).
        if self._is_cheap_zero():
            return other
        if other._is_cheap_zero():
            return self
        return self._flint_binop(other, "add")

    __radd__ = __add__

    def __neg__(self) -> "LaurentSeries":
        return LaurentSeries._from_flint(self._as_flint(_flint_prec_bits()).neg())

    def __sub__(self, other) -> "LaurentSeries":
        if not isinstance(other, LaurentSeries):
            res = self._scalar_op(other, "sub")
            if res is not None:
                return res
        other = self._promote(other)
        if other._is_cheap_zero():
            return self
        return self._flint_binop(other, "sub")

    def __rsub__(self, other) -> "LaurentSeries":
        return self._promote(other).__sub__(self)

    def __mul__(self, other) -> "LaurentSeries":
        if not isinstance(other, LaurentSeries):
            res = self._scalar_op(other, "mul")
            if res is not None:
                return res
        other = self._promote(other)
        # Structural-zero fast path: a zero operand (very common -- the u-series
        # rows are sparse, so most ``a[j] * b[i-j]`` convolution terms have a
        # zero factor) short-circuits to zero without a flint mul.
        if self._is_cheap_zero():
            return self
        if other._is_cheap_zero():
            return other
        self_const = self._cheap_constant_value()
        if self_const is not _NOT_CONSTANT:
            if self_const == 1:
                return other
            if self_const == 0:
                return LaurentSeries.zero()
            res = other._scalar_op(self_const, "mul")
            if res is not None:
                return res
        other_const = other._cheap_constant_value()
        if other_const is not _NOT_CONSTANT:
            if other_const == 1:
                return self
            if other_const == 0:
                return LaurentSeries.zero()
            res = self._scalar_op(other_const, "mul")
            if res is not None:
                return res
        return self._flint_binop(other, "mul")

    __rmul__ = __mul__

    def __pow__(self, n: int) -> "LaurentSeries":
        if not isinstance(n, int) or n < 0:
            raise ValueError("LaurentSeries supports only non-negative integer powers")
        result = LaurentSeries.constant(1)
        for _ in range(n):
            result = result * self
        return result

    def reciprocal(self, *, valuation=None) -> "LaurentSeries":
        """Invert, optionally retaining a leading power established by ball bounds."""
        return LaurentSeries._from_flint(self._as_flint(_flint_prec_bits()).reciprocal(valuation=valuation))

    def __truediv__(self, other) -> "LaurentSeries":
        if not isinstance(other, LaurentSeries):
            # Division by a scalar is a scale by 1/scalar -- no reciprocal of a
            # promoted constant series, no flint conversion of it.
            res = self._scalar_op(mp.mpc(1) / mp.mpc(other), "mul")
            if res is not None:
                return res
        return self * self._promote(other).reciprocal()

    def __rtruediv__(self, other) -> "LaurentSeries":
        return self._promote(other) * self.reciprocal()

    def sqrt(self, *, valuation=None) -> "LaurentSeries":
        """The principal square root, in flint balls (an mpmath recursion here dropped every radius it was given)."""
        if valuation is None and self.valuation() is None:
            return LaurentSeries.zero()
        return LaurentSeries._from_flint(self._as_flint(_flint_prec_bits()).sqrt(valuation=valuation))


def _real_path() -> bool:
    """True iff the batch flint paths should run in arb (real) instead of acb."""
    return _USE_ARB_LAURENT and _REAL_MODE


def _urow_to_blocks(rows, prec, n, real):
    """The per-u-power coefficient lists (length-_WIDTH-ish) of a u-row.

    In real mode the handles are arb-backed, so ``.P.coeffs()`` returns arb
    coefficients with no conversion; otherwise acb. A structurally-zero u-power
    maps to ``None`` (no flint conversion): the packed product never reads its
    slots, so skipping it is bit-identical to filling them with zeros.
    """
    out = []
    for i in range(n):
        r = rows[i]
        if r.is_zero():
            out.append(None)
        else:
            fl = r._as_flint(prec)
            out.append(fl.P.coeffs() if real else fl.acb_coeffs())
    return out


def u_convolve(a_rows, b_rows):
    """Bivariate (u, t) convolution of two u-rows, as a single flint multiply.

    ``a_rows`` / ``b_rows`` are equal-length lists of :class:`LaurentSeries`
    (coefficients of u^0, u^1, ...). Returns the list ``out`` with
    ``out[k] = sum_{i+j=k} a_rows[i] * b_rows[j]`` (each a t-series product),
    re-anchored to the canonical window -- identical to the naive
    ``out[k] = sum_j a[j]*b[k-j]`` loop but computed by packing both rows into one
    ``acb_series`` (Kronecker substitution: the u-blocks are spaced so their t-products cannot overlap), multiplying
    once, and slicing each u-block back out.
    """
    n = len(a_rows)
    if n == 0:
        return []
    prec = _flint_prec_bits()
    flint_ctx = _FLINT_CTX
    flint_ctx.prec = prec
    real = _real_path()
    SER = _ARB_SERIES if real else _ACB_SERIES
    a_blocks = _urow_to_blocks(a_rows, prec, n, real)
    b_blocks = _urow_to_blocks(b_rows, prec, n, real)
    # A structurally-zero operand (every u-block zero) makes the product zero;
    # skip the pack and the flint multiply entirely.
    if all(blk is None for blk in a_blocks) or all(blk is None for blk in b_blocks):
        return [LaurentSeries.zero() for _ in range(n)]
    # Pack only each side's common t-support (the blocks are typically half empty), at the stride that support
    # needs: a block product spans wa + wb - 1 slots, so a stride of wa + wb - 1 leaves no carry between u-powers.
    a_supp = [_support(blk) for blk in a_blocks if blk is not None]
    b_supp = [_support(blk) for blk in b_blocks if blk is not None]
    la = min(lo for lo, _hi in a_supp)
    ha = max(hi for _lo, hi in a_supp)
    lb = min(lo for lo, _hi in b_supp)
    hb = max(hi for _lo, hi in b_supp)
    if la >= ha or lb >= hb:
        return [LaurentSeries.zero() for _ in range(n)]
    S = (ha - la) + (hb - lb) - 1
    total = n * S
    zero = _ARB(0) if real else _ACB(0)
    A = [zero] * total
    B = [zero] * total
    for i in range(n):
        base = i * S
        ai = a_blocks[i]
        if ai is not None:
            seg = ai[la:ha]
            A[base:base + len(seg)] = seg
        bi = b_blocks[i]
        if bi is not None:
            seg = bi[lb:hb]
            B[base:base + len(seg)] = seg
    # Set the packed cap here: building the per-block flint handles above may
    # have routed through _FlintLaurent.from_mpc (for any mpc-backed input not
    # yet flint-cached), which resets the global flint cap to _CAP.
    flint_ctx.cap = total
    C = (SER(A) * SER(B)).coeffs()
    flint_ctx.cap = _CAP
    lenC = len(C)
    # Packed slot k*S + m holds t-slot m + la + lb - _OFF of u-power k.
    shift = la + lb - _OFF
    m_lo = max(0, -shift)
    m_hi = min(S, _WIDTH - shift)
    prefix = [zero] * max(0, shift)
    out = []
    for k in range(n):
        base = k * S
        if base + m_lo >= lenC or m_lo >= m_hi:
            out.append(LaurentSeries.zero())
            continue
        block = prefix + C[base + m_lo:base + m_hi]
        out.append(LaurentSeries._from_flint(_FlintLaurent(SER(block), prec, real)))
    return out


class StackedURows:
    """The t-coefficient rows of many u-rows of t-series, stacked once into one flint matrix for repeated weighted
    sums (:func:`stacked_weighted_sum`).  ``urows[k]`` is the u-row of item ``k`` (a length-``n_terms`` list of
    :class:`LaurentSeries`); ``None`` rows are items that never contribute.  Tied to the current window, precision
    and field."""

    def __init__(self, urows, n_terms):
        self.n_terms = _integer_control('n_terms', n_terms)
        self.n_rows = len(urows)
        if any(urow is not None and len(urow) != self.n_terms for urow in urows):
            raise ValueError('each u-row must have length n_terms')
        prec = _flint_prec_bits()
        _FLINT_CTX.prec = prec
        real = _real_path()
        zero0 = _ARB(0) if real else _ACB(0)
        self.prec, self.real = prec, real
        lo, hi = _WIDTH, 0
        rows = []
        for urow in urows:
            if urow is None:
                rows.append(None)
                continue
            coeff = [_coeff_row(x, prec, zero0, real) for x in urow]
            for _r, rlo, rhi in coeff:
                if rlo < rhi:
                    lo, hi = min(lo, rlo), max(hi, rhi)
            rows.append(coeff)
        self.lo, self.hi = lo, hi
        width = max(0, hi - lo)
        self.width = width
        blank = [zero0] * (width * self.n_terms)
        mat = []
        for coeff in rows:
            if coeff is None:
                mat.append(blank)
                continue
            line = []
            for r, _rlo, _rhi in coeff:
                line.extend(r[lo:hi])
            mat.append(line)
        self.matrix = (_ARB_MAT if real else _ACB_MAT)(mat) if mat and width else None


def stacked_weighted_sum(weights, stacked):
    """``out[i] = sum_k weights[k] * urows[k][i]`` for t-series ``weights`` (``None`` or zero: item skipped) and the
    :class:`StackedURows` ``urows``: ONE flint matrix product for every u-component, then the anti-diagonal sums
    re-anchored to the window (identical to the per-product window truncation).  Returns the u-row."""
    if len(weights) != stacked.n_rows:
        raise ValueError('weights must have one entry per stacked u-row')
    n_terms = stacked.n_terms
    prec, real = stacked.prec, stacked.real
    _FLINT_CTX.prec = prec
    SER = _ARB_SERIES if real else _ACB_SERIES
    MAT = _ARB_MAT if real else _ACB_MAT
    zero0 = _ARB(0) if real else _ACB(0)
    if stacked.matrix is None:
        return [LaurentSeries.zero() for _ in range(n_terms)]
    wrows = [None if w is None else _coeff_row(w, prec, zero0, real) for w in weights]
    live = [r for r in wrows if r is not None and r[1] < r[2]]
    if not live:
        return [LaurentSeries.zero() for _ in range(n_terms)]
    lw = min(r[1] for r in live)
    hw = max(r[2] for r in live)
    nw = hw - lw
    blank = [zero0] * nw
    X = MAT([blank if r is None or r[1] >= r[2] else r[0][lw:hw] for r in wrows])
    G = list((X.transpose() * stacked.matrix).entries())      # (nw x width*n_terms), row-major
    total_cols = stacked.width * n_terms

    nh = stacked.width
    lh = stacked.lo
    shift = _OFF - lw - lh
    out = []
    s_lo = max(0, -shift)
    s_hi = min(_WIDTH, nw + nh - 1 - shift)
    step = total_cols - 1       # G[a, base + target - a] for consecutive a: an arithmetic progression in G
    for i in range(n_terms):
        base = i * nh
        row = [zero0] * _WIDTH
        for s_ in range(s_lo, s_hi):
            target = s_ + shift
            a0 = max(0, target - nh + 1)
            a1 = min(nw - 1, target)
            first = a0 * total_cols + base + target - a0
            row[s_] = sum(G[first:first + (a1 - a0) * step + 1:step], zero0)
        out.append(LaurentSeries._from_flint(_FlintLaurent(SER(row), prec, real)))
    return out


def numeric_linear_map(weights, series):
    """``out[a] = sum_b weights[a][b] * series[b]`` for numeric ``weights`` (t-independent scalars, mpmath or flint)
    and :class:`LaurentSeries` ``series``: ONE flint matrix product ``weights @ [coefficient rows]`` instead of a
    scalar-times-series multiply and add per term.  Returns the list of :class:`LaurentSeries`."""
    if any(len(line) != len(series) for line in weights):
        raise ValueError('each weight row must have one entry per series')
    n_out = len(weights)
    if n_out == 0:
        return []
    prec = _flint_prec_bits()
    _FLINT_CTX.prec = prec
    real = _real_path()
    SER = _ARB_SERIES if real else _ACB_SERIES
    MAT = _ARB_MAT if real else _ACB_MAT
    zero0 = _ARB(0) if real else _ACB(0)
    rows = [_coeff_row(x, prec, zero0, real) for x in series]
    lo = min((r[1] for r in rows if r[1] < r[2]), default=_WIDTH)
    hi = max((r[2] for r in rows if r[1] < r[2]), default=0)
    if lo >= hi:
        return [LaurentSeries.zero() for _ in range(n_out)]
    scalar = _flint_scalar(real)
    out = MAT([[scalar(w) for w in line] for line in weights]) * MAT([r[0][lo:hi] for r in rows])
    pad_lo, pad_hi = [zero0] * lo, [zero0] * (_WIDTH - hi)
    cols = hi - lo
    return [LaurentSeries._from_flint(_FlintLaurent(SER(pad_lo + [out[a, m] for m in range(cols)] + pad_hi),
                                                    prec, real)) for a in range(n_out)]


def _flint_scalar(real):
    """Converter of a numeric scalar (mpmath or flint) to the flint field of the batch paths."""
    def to_arb(x):
        if isinstance(x, _ARB):
            return x
        if isinstance(x, _ACB):
            return x.real
        return _arb_from_mpf(x.real if isinstance(x, mp.mpc) else mp.mpf(x))

    def to_acb(x):
        if isinstance(x, _ACB):
            return x
        if isinstance(x, _ARB):
            return _ACB(x)
        return _acb_from_mpc(x if isinstance(x, mp.mpc) else mp.mpc(x))

    return to_arb if real else to_acb


def u_numeric_convolve_sum(terms, n_terms):
    """``sum_r t^(s_r) E_r * H_r`` for numeric u-series ``E_r`` (t-independent) and u-rows ``H_r`` of t-series.

    ``terms`` is a list of ``(E, H)`` or ``(E, H, s)``: ``E`` a length-``n_terms`` list of scalars (mpmath or
    flint), ``H`` a length-``n_terms`` list of :class:`LaurentSeries`, ``s >= 0`` a power of t multiplying the
    product (default 0; powers past the window are dropped, as the window product drops them).  The u-convolution
    with a t-independent factor is the lower triangular Toeplitz matrix of ``E`` acting on ``H``'s stacked
    t-coefficient rows, so the whole sum is ONE flint matrix product ``[T_1 | T_2 | ...] @ [H_1; H_2; ...]`` --
    exact, and ``n_terms`` times cheaper than packing each product as a full bivariate (u, t) convolution.  Returns
    the u-row of :class:`LaurentSeries`.
    """
    n = int(n_terms)
    if not terms:
        return [LaurentSeries.zero() for _ in range(n)]
    prec = _flint_prec_bits()
    flint_ctx = _FLINT_CTX
    flint_ctx.prec = prec
    real = _real_path()
    SER = _ARB_SERIES if real else _ACB_SERIES
    MAT = _ARB_MAT if real else _ACB_MAT
    zero0 = _ARB(0) if real else _ACB(0)

    scalar = _flint_scalar(real)
    left = []           # n rows of the concatenated Toeplitz blocks
    right = []          # the stacked coefficient rows
    lo, hi = _WIDTH, 0
    blocks = []
    for term in terms:
        E, H = term[0], term[1]
        shift = int(term[2]) if len(term) > 2 else 0
        e = [scalar(E[i]) for i in range(n)]
        rows = []
        for j in range(n):
            row, rlo, rhi = _coeff_row(H[j], prec, zero0, real)
            if shift:
                row = ([zero0] * shift + row)[:_WIDTH]
                rlo, rhi = min(rlo + shift, _WIDTH), min(rhi + shift, _WIDTH)
            rows.append((row, rlo, rhi))
            if rlo < rhi:
                lo, hi = min(lo, rlo), max(hi, rhi)
        blocks.append((e, rows))
    if lo >= hi:
        return [LaurentSeries.zero() for _ in range(n)]
    for e, rows in blocks:
        right.extend(row[lo:hi] for row, _rlo, _rhi in rows)
    for k in range(n):
        line = []
        for e, _rows in blocks:
            line.extend(e[k - j] if j <= k else zero0 for j in range(n))
        left.append(line)
    out = MAT(left) * MAT(right)
    pad_lo = [zero0] * lo
    pad_hi = [zero0] * (_WIDTH - hi)
    cols = hi - lo
    result = []
    for k in range(n):
        row = pad_lo + [out[k, m] for m in range(cols)] + pad_hi
        result.append(LaurentSeries._from_flint(_FlintLaurent(SER(row), prec, real)))
    return result


def _coeff_row(x, prec, zero0, real):
    """``(row, lo, hi)``: a length-_WIDTH coefficient row for a LaurentSeries or a scalar, and its
    :func:`_support` (cached with the row).

    Native type is arb in real mode (``real`` True), else acb. A scalar
    (mp.mpc / number) is the t^0 constant, so its only nonzero coefficient sits
    at slot _OFF -- this lets the matmul primitives mix t-dependent series and
    pure-number factors (e.g. q-power coefficients that happen to be
    t-independent)."""
    if isinstance(x, LaurentSeries):
        row_key = (prec, _WIDTH, _OFF, real)
        if x._row is not None and x._rowprec == row_key:
            return x._row
        fl = x._as_flint(prec)
        c = fl.P.coeffs() if real else fl.acb_coeffs()
        if len(c) < _WIDTH:
            row = list(c) + [zero0] * (_WIDTH - len(c))
        else:
            row = list(c) if len(c) == _WIDTH else list(c[:_WIDTH])
        row = (row, *_support(row))
        x._row = row
        x._rowprec = row_key
        return row
    key = _scalar_row_cache_key(x, real)
    if key is not None:
        hit = _SCALAR_ROW_CACHE.get(key)
        if hit is not None:
            return hit
    row = [zero0] * _WIDTH
    # A pre-converted scalar (callers that hoist the conversion out of the hot
    # loop pass these) goes straight in; otherwise convert from mpmath.
    if real:
        if isinstance(x, _ARB):
            row[_OFF] = x                       # already arb
        elif isinstance(x, _ACB):
            row[_OFF] = x.real                  # pre-converted acb scalar -> arb part
        else:
            row[_OFF] = _arb_from_mpf(x.real if isinstance(x, mp.mpc) else x)
    else:
        row[_OFF] = x if isinstance(x, _ACB) else \
            _acb_from_mpc(x if isinstance(x, mp.mpc) else mp.mpc(x))
    row = (row, *_support(row))
    if key is not None:
        _SCALAR_ROW_CACHE[key] = row
        if len(_SCALAR_ROW_CACHE) > _SCALAR_ROW_CACHE_MAXSIZE:
            _SCALAR_ROW_CACHE.popitem(last=False)
    return row


def _support_series(x, prec, zero0, real, SER):
    """``(series, lo, hi, key)``: the exact-zero-trimmed support ``[lo, hi)`` of ``x``'s window row (:func:`_coeff_row`)
    as one flint series (None for the zero series), cached on a LaurentSeries so a table entry or weight read by many
    products is converted once.  Only the support series is kept, not the full row (2/3 of a cached table's memory
    went to the rows and their copies)."""
    key = (prec, _WIDTH, _OFF, real)
    if isinstance(x, LaurentSeries):
        cached = x._supser
        if cached is not None and cached[3] == key:
            return cached
        if x._row is not None and x._rowprec == key:
            row, lo, hi = x._row
        else:
            fl = x._as_flint(prec)
            row = fl.P.coeffs() if real else fl.acb_coeffs()
            if len(row) > _WIDTH:
                row = row[:_WIDTH]
            lo, hi = _support(row)
        # (a flint series built under cap n is known only to O(x^n), and so is any product with it: built at the
        # window's cap, it bounds no product the window keeps)
        _FLINT_CTX.cap = _CAP
        x._supser = out = (SER(row[lo:hi]) if lo < hi else None, lo, hi, key)
        return out
    row, lo, hi = _coeff_row(x, prec, zero0, real)
    _FLINT_CTX.cap = _CAP
    return (SER(row[lo:hi]) if lo < hi else None), lo, hi, key


def _conv_support_pairs(pairs, top_slot, prec, real, SER, zero0):
    """``sum`` of the products of ``pairs`` -- two :func:`_support_series` supports ``(series, lo, hi, key)`` each --
    through slot ``top_slot`` (window slots ``0 .. top_slot``): one truncated product per pair (``mullow``
    of the supports, to the slots kept), summed per starting slot and placed once."""
    groups = {}
    ctx = _FLINT_CTX
    off = _OFF
    for (sw, lw, hw, _kw), (sh, lh, hh, _kh) in pairs:
        start = lw + lh - off
        # (every product from one start slot is formed to the same length -- flint truncates a sum to its shorter
        # operand's O(x^n) -- and a length past the product's own costs nothing)
        n = top_slot + 1 - start
        if n <= 0 or start + hw - lw + hh - lh - 1 <= 0:
            continue
        ctx.cap = n
        # flint multiplies the operands whole whatever the cap (27 x 18 coefficients to 13: 9.3 us, cut to 13 x 13
        # first: 3.9 us), so an operand longer than the product kept is cut to it (a copy under the cap)
        if hw - lw > n:
            sw = SER(sw)
        if hh - lh > n:
            sh = SER(sh)
        prod = sw * sh
        acc = groups.get(start)
        groups[start] = prod if acc is None else acc + prod
    ctx.cap = _CAP
    out = [zero0] * _WIDTH
    for start, ser in groups.items():
        cc = ser.coeffs()
        j0 = -start if start < 0 else 0
        for j in range(j0, min(len(cc), top_slot + 1 - start)):
            out[start + j] = out[start + j] + cc[j]
    return LaurentSeries._from_flint(_FlintLaurent(SER(out), prec, real))


def _t_conv_pairs(ws, hs, top_slot, prec, real, SER, zero0):
    """:func:`_conv_support_pairs` of the series pairs ``ws[i], hs[i]``."""
    pairs = []
    for w, h in zip(ws, hs):
        sw = _support_series(w, prec, zero0, real, SER)
        if sw[0] is None:
            continue
        sh = _support_series(h, prec, zero0, real, SER)
        if sh[0] is not None:
            pairs.append((sw, sh))
    return _conv_support_pairs(pairs, top_slot, prec, real, SER, zero0)


def conv_context():
    """``(prec, real, series type, zero)``: the flint working precision and arithmetic mode of the current
    computation, for :func:`support_of` / :func:`conv_supports` (a level fill reads them once)."""
    real = _real_path()
    return (_flint_prec_bits(), real, _ARB_SERIES if real else _ACB_SERIES, _ARB(0) if real else _ACB(0))


def support_of(x, cctx):
    """``(series, lo, hi, key)``: ``x``'s exact-zero-trimmed support as one flint series (``series`` None for the zero
    series), cached on ``x``."""
    prec, real, SER, zero0 = cctx
    _FLINT_CTX.prec = prec
    return _support_series(x, prec, zero0, real, SER)


def conv_supports(consts, pairs, top, cctx):
    """``sum(consts) + sum_i a_i b_i`` through the power ``top`` (None: the window's top), the powers above it exact
    zeros: :func:`t_conv_weighted_sum` with the supports of the products' factors read once by the caller
    (:func:`support_of`).  ``consts`` are series added whole (a product with a constant factor, formed by the
    caller); ``pairs`` are ``(support_of(a), support_of(b))`` with both supports nonempty."""
    prec, real, SER, zero0 = cctx
    _FLINT_CTX.prec = prec
    top_slot = _WIDTH - 1 if top is None else min(_WIDTH - 1, int(top) - T_LO)
    if top_slot < 0:
        return LaurentSeries.zero()
    acc = _conv_support_pairs(pairs, top_slot, prec, real, SER, zero0) if pairs else None
    if consts:
        const_acc = consts[0]
        for term in consts[1:]:
            const_acc = const_acc + term
        if top is not None:
            const_acc = const_acc.truncated_above(top)
        _FLINT_CTX.prec = prec
        acc = const_acc if acc is None else acc + const_acc
    return LaurentSeries.zero() if acc is None else acc


def t_conv_weighted_sum(ws, hs, top=None):
    """Sum of t-series products ``sum_idx ws[idx] * hs[idx]``, as one matmul.

    ``top`` (default the window's top power): the highest power of ``t`` the caller reads; the powers above it come
    back as exact zeros and are never formed (the level fill keeps each entry's ``need`` positive powers, the
    ``t^0`` readers none -- most of a full-window product).

    ``ws`` / ``hs`` are equal-length lists of :class:`LaurentSeries` (or scalars,
    treated as t^0 constants). The result equals ``sum_idx ws[idx]*hs[idx]`` (each
    a width-_WIDTH t-series convolution) but is computed as the anti-diagonal sums
    of ``G = Wt @ H`` where ``W[idx][p] = ws[idx]`` coefficient p and likewise
    ``H`` -- i.e. the whole weighted sum over ``idx`` collapses into a single
    ``acb_mat`` product instead of ``len(ws)`` per-coefficient LaurentSeries
    multiplies. (A packed single-multiply variant was tried but is far slower: it
    computes all n^2 u-block cross-terms to use one, dominating when n is large.)
    Returns a LaurentSeries (the zero series for empty inputs).
    """
    n = len(ws)
    if len(hs) != n:
        raise ValueError('ws and hs must have equal lengths')
    if n == 0:
        return LaurentSeries.zero()
    prec = _flint_prec_bits()
    flint_ctx = _FLINT_CTX
    flint_ctx.prec = prec
    real = _real_path()
    SER = _ARB_SERIES if real else _ACB_SERIES
    MAT = _ARB_MAT if real else _ACB_MAT
    zero0 = _ARB(0) if real else _ACB(0)
    # Peel off terms where the right factor is a known t^0 constant. In the
    # H-table fill these are overwhelmingly H_0 = 1 rows; including them in the
    # batched matrix multiply wastes width^2 work for a product that is just a
    # scalar scaling of the left factor.
    const_acc = None
    if n > 1:
        ws_mat = []
        hs_mat = []
        for w, h in zip(ws, hs):
            h_const = (
                h._cheap_constant_value()
                if isinstance(h, LaurentSeries)
                else h
            )
            if h_const is not _NOT_CONSTANT:
                if h_const == 1:
                    term = w
                elif h_const == 0:
                    continue
                else:
                    term = w * h_const
                const_acc = term if const_acc is None else const_acc + term
            else:
                ws_mat.append(w)
                hs_mat.append(h)
        if const_acc is not None:
            ws = ws_mat
            hs = hs_mat
            n = len(ws)
            if not isinstance(const_acc, LaurentSeries):
                const_acc = LaurentSeries.constant(const_acc)
            if top is not None:
                const_acc = const_acc.truncated_above(top)
            if n == 0:
                return const_acc
    top_slot = _WIDTH - 1 if top is None else min(_WIDTH - 1, int(top) - T_LO)
    if top_slot < 0:
        return LaurentSeries.zero()
    if n == 1 or top is not None:
        # one truncated product per pair: no matmul over the padded common support (and no anti-diagonal re-anchor);
        # a truncated sum keeps a fraction of the padded product (q 40 fill: 12%), which the matmul cannot skip
        acc = _t_conv_pairs(ws, hs, top_slot, prec, real, SER, zero0)
        flint_ctx.prec = prec
        return acc if const_acc is None else acc + const_acc
    W_rows = [_coeff_row(w, prec, zero0, real) for w in ws]
    H_rows = [_coeff_row(h, prec, zero0, real) for h in hs]
    # Only the common support of each side enters the matmul (the rows are typically half empty).
    lw = min(r[1] for r in W_rows)
    hw = max(r[2] for r in W_rows)
    lh = min(r[1] for r in H_rows)
    hh = max(r[2] for r in H_rows)
    shift = _OFF - lw - lh
    # (G[p, q] lands in slot p + q - shift: a column past the top slot on either side reaches only powers above top)
    nw = min(hw - lw, top_slot + shift + 1)
    nh = min(hh - lh, top_slot + shift + 1)
    if nw <= 0 or nh <= 0:
        return LaurentSeries.zero() if const_acc is None else const_acc
    G = MAT([r[0][lw:lw + nw] for r in W_rows]).transpose() * MAT([r[0][lh:lh + nh] for r in H_rows])
    # Re-anchor to base T_LO: G[p,q] holds the coefficient of t^(2*T_LO+lw+lh+p+q), so out[s] (coeff of
    # t^(T_LO+s)) = sum_{p+q = s+_OFF-lw-lh} G[p,q]. Read G's entries once into a flat row-major list --
    # acb_mat.__getitem__ per element is a hot-loop cost; flat-list indexing avoids it.
    Gf = G.entries()
    out = [zero0] * _WIDTH
    # anti-diagonal ``target`` = the flat entries target + p (nh - 1), p in [p0, p1]: one strided slice, summed in C
    # (the same additions in the same order as an explicit loop over p)
    step = nh - 1
    for s in range(max(0, -shift), min(top_slot + 1, nw + nh - 1 - shift)):
        target = s + shift
        p0 = max(0, target - nh + 1)
        p1 = min(nw - 1, target)
        if step:
            out[s] = sum(Gf[target + p0 * step: target + p1 * step + 1: step], zero0)
        else:
            out[s] = zero0 + Gf[target]
    acc = LaurentSeries._from_flint(_FlintLaurent(SER(out), prec, real))
    return acc if const_acc is None else acc + const_acc
