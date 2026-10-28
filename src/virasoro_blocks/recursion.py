"""Zamolodchikov h-recursion for the sphere 4-point Virasoro conformal block.

Following arXiv:1703.09805 eqs (2.19)-(2.20):

    F(h, z; h_i; c)
      = (16 q)^{h - (c-1)/24}
        z^{(c-1)/24 + z_shift}
        (1 - z)^{(c-1)/24 - h2 - h3}
        theta_3(q)^{(c-1)/2 - 4 (h1+h2+h3+h4)}
        H(c, h; h_i; q)
        / N^h

with z_shift = 0 by default (block ~ z^h / N^h at small z).
For the physical crossing equation, pass z_shift = -h1 - h2
(where h_i are the holomorphic weights) to include the kinematic
factor z^{h-h1-h2}.

The h-recursion (eq 2.20):

    H(c, h; h_i; q) = 1
        + sum_{r,s >= 1} (16 q)^{r s} A^c_{r,s} / (h - d_{r,s})
            * P^c_{r,s}[h1, h2] * P^c_{r,s}[h4, h3]
            * H(c, h -> d_{r,s} + r s; h_i; q).

The shift inside the recursive H is `h -> d_{r,s} + r s`, NOT `h -> h + r s`.

Implementation: we compute H truncated at q^N. Let
    K_{r,s} = 16^{r s} A^c_{r,s} P^c_{r,s}[h1, h2] P^c_{r,s}[h4, h3]
    v_{r,s} = d_{r,s} + r s
Then in q-expansion the k-th coefficient of H at argument h_arg is

    H_k(h_arg) = delta_{k,0}
              + sum_{(r,s): r s <= k}
                    K_{r,s} / (h_arg - d_{r,s}) * H_{k - r s}(v_{r,s}).

The set of arguments that ever appears is {h_top} union {v_{r,s} : r s <= N}.
We build a table H_table[h_arg][k] bottom-up in k, so the cost is O(N^2 . D(N))
where D(N) is the number of (r,s) pairs with r s <= N (roughly N log N).

Nome convention: q(z) = exp(-pi K(1-z) / K(z)).

Library normalization convention:
    the returned chiral block is

        F(h, z) = (16 q / N)^h * z^{sigma + z_shift} * (1-z)^{sigma - h2 - h3}
                  * (16q)^{-sigma} * theta_3^{...} * H(q)

    where sigma = (c-1)/24 and N is the user-selectable `chiral_norm`
    (default 16). The h-dependent exponential base is 16 q(z) / N (z-
    independent). With the default N = 16 the base at z = 1/2 is
    q(1/2) = e^{-pi} ~ 0.043.

    With z_shift = 0 (default), the block behaves as z^h / N^h at small z.
    With z_shift = -h1 - h2, the block behaves as z^{h-h1-h2} / N^h at
    small z, which is the normalization needed for crossing symmetry.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from typing import Sequence

import mpmath as mp

from .numeric_input import _integer_control, _numeric_input_values
from .params import (
    A_rs,
    arithmetic_mode,
    b_from_c,
    b_series,
    beta_series,
    c_from_b,
    d_rs,
    fusion_polynomial,
    fusion_polynomial_via_squares,
    imaginary_b_locus,
)
from . import poles as _poles
from .polynomial import _mpf_to_arb, _arb_to_mpf, _mpf_to_acb, _acb_to_mpc, _flint_prec_bits
from .tseries import LaurentSeries, set_real_mode, t_conv_weighted_sum


@dataclass
class PairTable:
    """A list of (r, s, rs, K, d, v) pair dicts plus arithmetic-mode metadata.

    `mode` classifies the recursion arithmetic for this (c, externals)
    combination as one of `"real_b"`, `"strip_twin"`, `"imag_b"`, or
    `"generic_complex"`. The Kac data in `pairs` is stored in the narrowest
    field consistent with the mode (mp.mpf for real_b/imag_b, mp.mpc for
    strip_twin/generic_complex).

    When `mode == "strip_twin"`, the pairs (r, s) and (s, r) are exact
    complex-conjugate twins (b lies on the unit circle and the externals are
    real). `twin_of[i]` is the index of pair i's twin (i itself for diagonals
    r == s), `is_canonical[i]` flags whether i has r <= s, and
    `canonical_indices` is the list of canonical indices in pair-order. In
    every other mode the twin metadata is the trivial identity (every pair
    is canonical and is its own twin).

    `is_strip_symmetric` is retained as a derived attribute equal to
    `mode == "strip_twin"` for downstream code that still dispatches on it.

    For backwards compatibility PairTable iterates and indexes like a list of
    pairs so existing call sites (`for p in pairs`, `enumerate(pairs)`,
    `pairs[idx]`, `len(pairs)`) work unchanged.
    """

    pairs: list
    canonical_indices: list
    twin_of: list
    is_canonical: list
    mode: str
    interaction_weights: dict | None = None
    # the same weights as acb (set with ``interaction_weights`` by _pair_interaction_weights_flint), so the fill
    # builds its matrix without converting every weight back from mpmath
    interaction_weights_flint: dict | None = None
    # {flint precision: (K balls, d balls)} of the numeric Kac data (_kac_balls), reused by the certified H fill
    kac_balls: dict | None = None
    # whether the Kac data were formed from the exact c behind b (_b_origin), or from b as given (no recorded c, or
    # an exactly singular c, where the data of c do not exist)
    kac_from_c: bool = False

    @property
    def is_strip_symmetric(self) -> bool:
        return self.mode == "strip_twin"

    def __iter__(self):
        return iter(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)


# Block normalization N in F = N^{-h} (16 q)^h * ...; the damped-rational base is
# 16q/N. N=16 gives base q(1/2)=e^{-pi}~0.043, which matches the physical
# large-weight decay of OPE^2*block (VMFT, arXiv:1811.05710) and places the
# Bohr-Sommerfeld sample points where the contributions live -> better-
# conditioned bootstrap SDPs. (Pass
# chiral_norm=4 for the self-dual base 4q~0.173 / Hogervorst-Rychkov radial.)
DEFAULT_CHIRAL_NORM = mp.mpf(16)
_PAIR_TABLE_CACHE_MAXSIZE = 64
_PAIR_TABLE_CACHE: OrderedDict = OrderedDict()
_H_CONTEXT_CACHE_MAXSIZE = 64
_H_CONTEXT_CACHE: OrderedDict = OrderedDict()
# a c=1 word class cycles through ~40 external tuples; an LRU smaller than the working set misses on
# every call (447 rebuilds for 41 keys), so keep a few hundred (each state is ~100 kB at q_order 34)
_H_LAURENT_PRECOMP_CACHE_MAXSIZE = 256
_H_LAURENT_PRECOMP_CACHE: OrderedDict = OrderedDict()


def _resolve_chiral_norm(chiral_norm):
    """Return mp.mpf(chiral_norm) or the module default if None."""
    if chiral_norm is None:
        return DEFAULT_CHIRAL_NORM
    from .numeric_input import _evaluation_number

    message = "chiral_norm must be > 0, finite, and real"
    try:
        value = _evaluation_number(chiral_norm)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if isinstance(chiral_norm, bool) or mp.im(value) != 0 or not mp.isfinite(value) or value <= 0:
        raise ValueError(message)
    return value


def _maybe_real_scalar(x):
    """Collapse exact-zero-imaginary `mpc` scalars to `mpf`."""
    if isinstance(x, mp.mpc) and x.imag == 0:
        return x.real
    return x


def _scalar_cache_key(value):
    """Hashable cache key for supported numeric scalars, or None."""
    if isinstance(value, (mp.mpf, mp.mpc)):
        value = _maybe_real_scalar(value)
    if isinstance(value, mp.mpf):
        return ("mpf", value._mpf_)
    if isinstance(value, mp.mpc):
        return ("mpc", value._mpc_)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("mpf", mp.mpf(value)._mpf_)
    if isinstance(value, complex):
        return ("mpc", mp.mpc(value)._mpc_)
    if isinstance(value, Fraction):
        return ("frac", value.numerator, value.denominator)
    return None


def _pair_table_cache_key(q_order: int, b, h_ext) -> tuple | None:
    """Return a precision-aware cache key for numeric pair tables."""
    b_key = _scalar_cache_key(b)
    if b_key is None:
        return None
    h_ext_keys = []
    for hi in h_ext:
        hi_key = _scalar_cache_key(hi)
        if hi_key is None:
            return None
        h_ext_keys.append(hi_key)
    # the exact c behind b (the Kac data are built from it), so a b shared by two nearby c never shares a table
    origin = _b_origin(b)
    # a pinned arithmetic mode (_pin_mode) can differ from the one this precision alone would give
    pins = _MODE_PIN.get()
    pinned = pins.get(_mode_pin_key(b, h_ext)) if pins else None
    return (mp.mp.prec, q_order, b_key, tuple(h_ext_keys), None if origin is None else _scalar_cache_key(origin),
            pinned if pinned is not None and pinned != _classify_unpinned(b, h_ext) else None)


def _accumulator_field(pairs) -> type:
    """Return `mp.mpf` if every K, d, v in `pairs` is mpf; else `mp.mpc`.

    Used by the H-table recursion in `H_q_series` and the non-strip branch
    of `_eval_H_at` to skip the mp.mpc detour when the Kac data are real
    (real_b and imag_b modes). Strip mode pairs are mpc-typed by
    construction even with real externals (b on the unit circle), so this
    naturally returns mpc there.
    """
    for p in pairs:
        if not isinstance(p["K"], mp.mpf):
            return mp.mpc
        if not isinstance(p["d"], mp.mpf):
            return mp.mpc
    return mp.mpf


def _q_from_z_flint(z):
    """Flint-accelerated q(z) = exp(-pi K(1-z) / K(z)).

    acb.elliptic_k and the AGM (~7x mpmath's at dps 80). Returns mp.mpf for real z < 1, mp.mpc otherwise.
    """
    from flint import arb, acb, ctx as flint_ctx
    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits
    ndigits = mp.mp.dps + 5

    is_real = isinstance(z, mp.mpf)
    z_f = _mpf_to_acb(z)

    pi_f = arb.pi()
    K_f = acb.elliptic_k(z_f)
    # K(1 - z) = pi / (2 agm(1, sqrt z)): forming 1 - z first would cost
    # log10(1/|z|) digits at small |z| (and give q = 0 once |z| < 10^-dps)
    Kp_f = acb(pi_f) / (2 * z_f.sqrt().agm())
    q_f = (-acb(pi_f) * Kp_f / K_f).exp()

    if isinstance(z, mp.mpc) and z.imag != 0 and q_f.imag.contains(0):
        # A tiny Im z leaves Im q below the rounding of q (on the negative real axis its sign selects the branch of
        # q^h): q is real on the real segment, so Im q = Re(q'(z)) Im z to first order, with
        # d log q / dz = pi^2 / (4 z (1 - z) K(z)^2) (Legendre's relation).
        dq = q_f * acb(pi_f) ** 2 / (4 * z_f * (1 - z_f) * K_f ** 2)
        q_f = acb(q_f.real, dq.real * _mpf_to_arb(z.imag))

    if is_real and z < 1:
        return mp.mpf(q_f.real.str(ndigits, radius=False))
    # a real z > 1 is on the cut of K(z): q is complex there, as on mpmath's path
    return mp.mpc(
        mp.mpf(q_f.real.str(ndigits, radius=False)),
        mp.mpf(q_f.imag.str(ndigits, radius=False)),
    )


def _q_cache_key(z):
    """Fast key for the supplied point, before numerical conversion.

    Uses the raw ``_mpf_`` integer tuple (sign, mantissa, exponent, bitcount)
    which is unique at the current working precision and never requires string
    conversion.  For mpc, concatenate both halves.  The working precision
    ``mp.mp.prec`` is appended so that the same mathematical value computed at
    different precisions never shares a cache entry.
    """
    prec = mp.mp.prec
    if isinstance(z, mp.mpf):
        return (z._mpf_, prec)
    if isinstance(z, mp.mpc):
        return (z.real._mpf_, z.imag._mpf_, prec)
    if isinstance(z, Fraction):
        return ("rational", z.numerator, z.denominator, prec)
    if isinstance(z, complex):
        return ("complex", z.real.hex(), z.imag.hex(), prec)
    raise TypeError(f"unsupported nome point {type(z).__name__}")


# Module-level cache for q_from_z results.  At any one working precision the
# oracle calls taylor_coefficients at a fixed grid of ~8–25 points that repeat across
# all probe Deltas (same u_symmetric / n_terms).  The cache converts
# O(n_probes * n_taylor_pts) elliptic-K evaluations into O(n_taylor_pts).
_q_from_z_cache: dict = {}


def _q_from_z_uncached(z):
    if isinstance(z, Fraction):
        from .numeric_input import literal_number

        z = literal_number(z)
    if not isinstance(z, (mp.mpf, mp.mpc)):
        z = _maybe_real_scalar(mp.convert(z))
    # K(1-z) diverges at z=0, and K(z) at z=1.
    return mp.mpf(0) if z == 0 else mp.mpf(1) if z == 1 else _q_from_z_flint(z)


def q_from_z(z):
    """Elliptic nome q(z) = exp(-pi K(1-z) / K(z)).

    Evaluated in flint (:func:`_q_from_z_flint`) at every precision.

    Results are memoised by the exact input and ``mp.mp.prec`` so repeated evaluations at
    the same point within a computation (e.g. oracle Richardson-extrapolation
    grid shared across all probe Deltas) hit the in-process cache instead of
    recomputing two elliptic integrals.

    A real z < 1 gives an mp.mpf (q is real there), so the chiral block path stays in
    mp.mpf for real z; a real z > 1 (on the cut of K) and a complex z give mp.mpc.
    Exact real literals retain their value until conversion.
    """
    if isinstance(z, (mp.mpf, mp.mpc)):
        z = z.real if isinstance(z, mp.mpc) and z.imag == 0 else z
    elif not isinstance(z, Fraction):
        from .numeric_input import literal_number

        z = literal_number(z)
        if not isinstance(z, (Fraction, complex, mp.mpf, mp.mpc)):
            z = mp.convert(z)
        if isinstance(z, mp.mpc) and z.imag == 0:
            z = z.real
    # memoised at every precision (a block family evaluates many channels at one point)
    key = _q_cache_key(z)
    cached = _q_from_z_cache.get(key)
    if cached is not None:
        return cached
    result = _q_from_z_uncached(z)
    if len(_q_from_z_cache) > 200000:
        _q_from_z_cache.clear()
    _q_from_z_cache[key] = result
    return result


def _enumerate_rs_pairs(q_order: int):
    """Yield (r, s) pairs with r, s >= 1 and r*s <= q_order, in r*s order."""
    for n in range(1, q_order + 1):
        for r in range(1, n + 1):
            if n % r == 0:
                yield (r, n // r, n)  # (r, s, r*s)


# The arithmetic modes pinned by a certification ladder, {(b, externals) value key: mode}.  A ladder's escalated
# passes rebuild the recursion at more bits from the SAME b; classified afresh there, a strip-real b rounded at the
# working precision is no longer unit-modulus to the extra bits and read as generic complex (c = 3/2, dps 30: an
# mpc H_q_series), while the regularizer's probes rely on the mode the working precision gives.  So a ladder
# classifies once, at the working precision (:func:`_pin_mode`), and every pass under it uses that mode.
_MODE_PIN: contextvars.ContextVar = contextvars.ContextVar("virasoro_blocks_mode_pin", default=None)


def _mode_pin_key(b, h_ext):
    def part(x):
        z = x if isinstance(x, mp.mpc) else mp.mpc(mp.convert(x))
        return (z.real._mpf_, z.imag._mpf_)
    try:
        return (part(b), tuple(part(x) for x in h_ext))
    except (TypeError, ValueError):
        return None


@contextlib.contextmanager
def _pin_mode(b, h_ext):
    """Classify ``(b, h_ext)`` once, at the current (working) precision, for every recursion build inside."""
    if isinstance(b, LaurentSeries) or b is None:
        yield
        return
    key = _mode_pin_key(b, h_ext)
    if key is None:
        yield
        return
    pins = dict(_MODE_PIN.get() or {})
    pins.setdefault(key, _classify_mode(b, h_ext))
    token = _MODE_PIN.set(pins)
    try:
        yield
    finally:
        _MODE_PIN.reset(token)


def _classify_mode(b, h_ext) -> str:
    """Determine the recursion arithmetic mode from b and the externals.

    Wraps `arithmetic_mode(c, d)` by first computing c from b for non-
    LaurentSeries inputs; LaurentSeries b (regularization path) routes to
    generic_complex.

    A known source charge decides whether the parameter is mathematically
    real; rounding artifacts in a derived b do not change that choice.
    Nonzero imaginary input components and independent b values off both
    axes use generic_complex. The interval check for real inputs is capped
    at 50 dps; rounding onto an endpoint only selects the safe generic path.
    """
    if isinstance(b, LaurentSeries):
        return "generic_complex"
    pins = _MODE_PIN.get()
    if pins:
        pinned = pins.get(_mode_pin_key(b, h_ext))
        if pinned is not None:
            return pinned
    return _classify_unpinned(b, h_ext)


def _classify_unpinned(b, h_ext) -> str:
    if isinstance(b, LaurentSeries):
        return "generic_complex"
    # Supplied components are inputs, rather than computed imaginary noise.
    # In particular imag_b must not replace a near-imaginary independent b
    # by i*Im(b), nor may strip mode impose conjugacy on an independent b.
    from .params import c_origin_of_b

    origin = c_origin_of_b(b)
    if any(isinstance(h, (complex, mp.mpc)) and h.imag != 0 for h in h_ext):
        return "generic_complex"
    if origin is not None:
        if isinstance(origin, (complex, mp.mpc)) and origin.imag != 0:
            return "generic_complex"
    else:
        numeric_b = b if isinstance(b, (complex, mp.mpf, mp.mpc)) else mp.convert(b)
        if isinstance(numeric_b, (complex, mp.mpc)) and numeric_b.real != 0 and numeric_b.imag != 0:
            return "generic_complex"
    with mp.workdps(min(mp.mp.dps, 50)):
        return arithmetic_mode(origin if origin is not None else c_from_b(b), h_ext)


def _detect_strip_symmetric(b, h_ext) -> bool:
    """Back-compat shim: True iff `_classify_mode(b, d) == "strip_twin"`."""
    return _classify_mode(b, h_ext) == "strip_twin"


def _build_pair_indexing(pairs, strip_symmetric: bool):
    """Return (canonical_indices, twin_of, is_canonical) for the given pair list.

    In non-symmetric mode every pair is canonical and is its own twin.
    In strip mode, canonical = (r <= s) and twin = the (s, r) pair (or self
    for r == s).
    """
    if not strip_symmetric:
        n = len(pairs)
        return list(range(n)), list(range(n)), [True] * n

    # Build a map from (r, s) -> pair index for the twin lookup.
    by_rs = {(p["r"], p["s"]): idx for idx, p in enumerate(pairs)}
    twin_of = []
    is_canonical = []
    canonical_indices = []
    for idx, p in enumerate(pairs):
        r, s = p["r"], p["s"]
        twin_idx = by_rs[(s, r)]  # always present for the same q_order: rs == sr
        twin_of.append(twin_idx)
        canonical = r <= s
        is_canonical.append(canonical)
        if canonical:
            canonical_indices.append(idx)
    return canonical_indices, twin_of, is_canonical


def _kac_input_values(values):
    """Numeric source labels for ball evaluation, preserving rationals and stored mpmath bits."""
    from .numeric_input import parse_number

    return tuple(value if isinstance(value, (Fraction, mp.mpf)) else _maybe_real_scalar(mp.convert(value))
                 for value in (parse_number(value) for value in values))


def _numeric_kac_inputs(b, h_ext):
    """``(b, externals)`` as numeric source labels for the flint Kac builder, or ``None`` for Laurent data.

    Rational labels enter ball arithmetic directly. Converting them to mpmath first would discard their source
    before the adaptive fill increases precision, producing bounds for the rounded input instead.
    """
    if isinstance(b, LaurentSeries) or any(isinstance(x, LaurentSeries) for x in h_ext):
        return None
    try:
        b = b if isinstance(b, (mp.mpf, mp.mpc, Fraction)) else mp.convert(b)
        ext = _kac_input_values(h_ext)
    except (TypeError, ValueError):
        return None
    if not isinstance(b, (mp.mpf, mp.mpc, Fraction)) or not all(isinstance(x, (mp.mpf, mp.mpc, Fraction)) for x in ext):
        return None
    return b, ext


def _is_real_scalar(x) -> bool:
    return isinstance(x, (mp.mpf, Fraction)) or (isinstance(x, mp.mpc) and x.imag == 0)


def _b_ball_from_c(c, b_ref):
    """A ball (current flint precision) for the Liouville b of the exact central charge ``c``, on the branch of
    ``b_ref`` (b, -b, 1/b and -1/b give the same c).

    ``b^2`` is a root of ``x^2 - w x + 1`` with ``w = (c - 1)/6 - 2``: the root of larger modulus is formed without
    cancellation (``(w +- sqrt(w^2 - 4))/2`` with the sign of the larger sum) and the other as its inverse.  The
    square root is taken of ``-b^2`` (times ``i``) when ``Re(b^2) < 0``, away from the branch cut, so the c < 1 b is
    exactly imaginary and a real c > 25 gives an exactly real b.
    """
    from flint import acb

    c_f = _mpf_to_acb(c)
    w = (c_f - 1) / 6 - 2
    disc = (w * w - 4).sqrt()
    s_plus, s_minus = w + disc, w - disc
    s = s_plus if s_plus.abs_upper() >= s_minus.abs_upper() else s_minus
    target = mp.mpc(b_ref) ** 2
    roots = (s / 2, 2 / s)
    b_sq = min(roots, key=lambda x: abs(_acb_to_mpc(x) - target))
    if mp.re(target) < 0:
        b_f = acb(0, 1) * (-b_sq).sqrt()
    else:
        b_f = b_sq.sqrt()
    if abs(_acb_to_mpc(b_f) + b_ref) < abs(_acb_to_mpc(b_f) - b_ref):
        b_f = -b_f
    return b_f


def _b_origin(b):
    """The exact c that ``b`` was formed from by ``params.b_from_c`` (the Kac data are then built from a ball of b
    around that c), or None: a b given directly is exact as given."""
    from .params import c_origin_of_b

    return c_origin_of_b(b)


class _SingularKacData(Exception):
    """The Kac data of this exact c do not exist (a factor ``m b + n/b`` of ``A_{r,s}`` vanishes)."""


class _UnresolvedKacData(Exception):
    """A Kac-data enclosure needs more precision to separate a nonzero factor or gap."""


def _fusion_zero_checker(b, source, c):
    """Prove a paired fusion factor vanishes, using rational complex component arithmetic."""
    from .numeric_input import literal_components, _zero_on_quadratic_root

    add = lambda a, z: (a[0]+z[0], a[1]+z[1])
    scale = lambda a, n: (n*a[0], n*a[1])
    mul = lambda a, z: (a[0]*z[0]-a[1]*z[1], a[0]*z[1]+a[1]*z[0])
    if c is not None:
        cr, ci = literal_components(c)
        total = ((cr-13)/6, ci/6)
        q_squared = add(total, (2, 0))
        hint = b**2
    else:
        br, bi = literal_components(b)
        squared = (br*br-bi*bi, 2*br*bi)
        norm = squared[0]**2+squared[1]**2
        inverse = (squared[0]/norm, -squared[1]/norm)
        q_squared = add(add(squared, inverse), (2, 0))

    def zero(p, q, left, right):
        difference = scale(add(source[right], scale(source[left], -1)), 4)
        lam_squared = add(q_squared, scale(source[left], -4))
        if c is None:
            shift = add(add(scale(squared, p*p), scale(inverse, q*q)), (2*p*q, 0))
            u = add(difference, shift)
            return add(mul(u, u), scale(mul(lam_squared, shift), -4)) == (0, 0)
        # shift² = a*t+v with t²-total*t+1=0; reduce the
        # squared fusion factor to linear*t+constant before checking its root.
        a = p*p-q*q
        v = add(scale(total, q*q), (2*p*q, 0))
        u = add(difference, v)
        linear = add(add(scale(total, a*a), scale(u, 2*a)), scale(lam_squared, -4*a))
        constant = add(add(mul(u, u), (-a*a, 0)), scale(mul(lam_squared, v), -4))
        return _zero_on_quadratic_root(linear, constant, total, hint)

    return zero


def _kac_balls(q_order: int, b, h_ext, mode: str, c=None):
    """Flint balls ``(K, d)`` of every ``(r, s)`` with ``r s <= q_order`` (in ``_enumerate_rs_pairs`` order), at the
    current flint precision, from ``b`` and the externals taken as exact (a dyadic mpf converts exactly).

    ``K_{r,s} = 16^{rs} A_{r,s} P_{r,s}[h1, h2] P_{r,s}[h4, h3]``, the fusion polynomials in the manifestly
    ``(p, q) <-> (-p, -q)``-paired squares form (``params.fusion_polynomial_via_squares``), and ``d_{r,s}`` in the
    cancellation-free product form

        d_{r,s} = ((b + 1/b)^2 - (r b + s/b)^2) / 4 = ((1-r) b + (1-s)/b) ((1+r) b + (1+s)/b) / 4,

    so ``d_{1,1} = 0`` exactly (the difference of squares left ``-3.9e-32`` at dps 30 for c < 1, and an exchanged
    ``h`` near 0 lost ``log10(1/h)`` digits against it).  When ``h1 == h2`` (``h4 == h3``) the ``(0, 0)`` factor
    of ``P_{r,s}[h1, h2]`` (``r, s`` odd) is an exact zero, so structurally absent poles stay exact zeros.  In
    strip mode (b on the unit circle, real externals) the ``(s, r)`` data are the conjugates of ``(r, s)``; for
    c < 1 (``imag_b``) b is taken as exactly ``i Im(b)``.  acb is used uniformly (the real modes read real parts),
    so real and complex inputs round alike.  Every ball's radius covers its own rounding, so the H-table fill that
    consumes these (:func:`_certified_h_table`) certifies the Kac data too.  With ``c`` (the exact central charge
    b was formed from, :func:`_b_origin`) b itself is the ball :func:`_b_ball_from_c`, so the data are those of
    that c rather than of its rounded b.
    """
    from flint import acb

    labels = list(_enumerate_rs_pairs(q_order))
    if c is not None:
        b_f = _b_ball_from_c(c, b)
    elif mode == "imag_b":
        numeric_b = mp.convert(b)
        y = numeric_b.imag if isinstance(numeric_b, mp.mpc) else numeric_b
        b_f = acb(0, _mpf_to_arb(y))
    else:
        b_f = _mpf_to_acb(b)
    one = acb(1)
    inv_b = one / b_f
    Q = b_f + inv_b
    Q_sq = Q * Q
    b_sq = b_f * b_f
    inv_b_sq = inv_b * inv_b
    lam_sq = [Q_sq - 4 * _mpf_to_acb(x) for x in h_ext]
    # lam_L^2 - lam_R^2 = 4 (h_R - h_L): an exact zero for equal externals (the (0,0) factor of P_{r,s}[h, h])
    from .numeric_input import literal_components

    source = tuple(literal_components(value) for value in h_ext)
    diff_12 = acb(0) if source[0] == source[1] else 4 * (_mpf_to_acb(h_ext[1]) - _mpf_to_acb(h_ext[0]))
    diff_43 = acb(0) if source[3] == source[2] else 4 * (_mpf_to_acb(h_ext[2]) - _mpf_to_acb(h_ext[3]))
    top = max((max(r, s) for r, s, _rs in labels), default=0) + 1
    m_b = {m: m * b_f for m in range(-top, top + 1)}
    n_ib = {n: n * inv_b for n in range(-top, top + 1)}
    sixteen = acb(16)
    fusion_zero = None
    weight_matches = None

    def fusion(lam_L_sq, diff, r, s, left, right):
        nonlocal fusion_zero

        acc = one
        for p in range(1 - r, r, 2):
            for q in range(1 - s, s, 2):
                if p < 0 or (p == 0 and q < 0):
                    continue            # (p, q) and (-p, -q) share one factor
                if p == 0 and q == 0:
                    if diff.is_zero():
                        return None     # an exact zero
                    acc = acc * (diff / 4)
                    continue
                shift_sq = (p * p) * b_sq + 2 * p * q + (q * q) * inv_b_sq
                u = diff + shift_sq
                factor = (u * u - 4 * lam_L_sq * shift_sq) / 16
                if factor.contains(0):
                    if fusion_zero is None:
                        fusion_zero = _fusion_zero_checker(b, source, c)
                    if fusion_zero(p, q, left, right):
                        return None
                acc = acc * factor
        return acc

    strip = mode == "strip_twin"
    index = {(r, s): i for i, (r, s, _rs) in enumerate(labels)}
    K = [None] * len(labels)
    d = [None] * len(labels)
    for i, (r, s, rs) in enumerate(labels):
        if strip and r > s:
            continue
        d_i = (m_b[1 - r] + n_ib[1 - s]) * (m_b[1 + r] + n_ib[1 + s]) / 4
        if d_i.contains(0) and not d_i.is_zero():
            if weight_matches is None:
                weight_matches = _poles._kac_weight_checker(b, c=c)
            if weight_matches(0, r, s):
                d_i = acb(0)
        P12 = fusion(lam_sq[0], diff_12, r, s, 0, 1)
        P43 = fusion(lam_sq[3], diff_43, r, s, 3, 2) if P12 is not None else None
        if P12 is None or P43 is None:
            K_i = acb(0)
        else:
            # A_{r,s} = (1/2) prod_{(m,n) != (0,0),(r,s)} 1 / (m b + n/b): one division for the whole product
            prod = one
            for m in range(1 - r, r + 1):
                bm = m_b[m]
                for n in range(1 - s, s + 1):
                    if (m == 0 and n == 0) or (m == r and n == s):
                        continue
                    prod = prod * (bm + n_ib[n])
            if prod.contains(0):
                # Containing zero asks for precision unless a source identity
                # proves m*b²+n=0 on this branch.
                relation = _poles._kac_relation_checker(b, c=c)
                for m in range(1-r, r+1):
                    for n in range(1-s, s+1):
                        if (m, n) in ((0, 0), (r, s)):
                            continue
                        if relation(m, 0, (n, 0)):
                            if c is not None:
                                raise _SingularKacData(r, s)
                            raise ZeroDivisionError(f"A_{{{r},{s}}} divides by zero: b^2 is rational at this c")
                raise _UnresolvedKacData
            K_i = (sixteen ** rs) * P12 * P43 / (2 * prod)
        K[i], d[i] = K_i, d_i
        if strip and r != s:
            j = index[(s, r)]
            K[j], d[j] = K_i.conjugate(), d_i.conjugate()
    return K, d


def _kac_collides(q_order: int, d, collision_check=None) -> bool:
    """Scan possible shifted Kac collisions; prove source equality when a checker is supplied.

    An unresolved candidate requests more precision. Without a checker this remains a numerical diagnostic.
    """
    import numpy as np

    rs = np.array([rs for _r, _s, rs in _enumerate_rs_pairs(q_order)], dtype=float)
    if not len(rs):
        return False
    d64 = np.array([complex(float(x.real.mid()), float(x.imag.mid())) for x in d])
    scale = max(1.0, float(np.abs(d64).max()))
    close = np.abs((d64 + rs)[:, None] - d64[None, :]) <= 1e-8 * scale
    for i, j in zip(*np.nonzero(close)):
        if (d[i] + int(rs[i]) - d[j]).contains(0):
            if collision_check is None:
                return True
            labels = list(_enumerate_rs_pairs(q_order))
            outer = dict(zip(('r', 's', 'rs'), labels[i]))
            inner = dict(zip(('r', 's', 'rs'), labels[j]))
            if collision_check(outer, inner):
                return True
            raise _UnresolvedKacData
    return False


def _pairs_from_kac_balls(q_order: int, K, d, real: bool) -> list:
    """The pair dicts (K, d, v at the working precision) from the Kac balls' midpoints."""
    pairs = []
    for (r, s, rs), k_ball, d_ball in zip(_enumerate_rs_pairs(q_order), K, d):
        if real:
            K_val, d_val = +_arb_to_mpf(k_ball.real), +_arb_to_mpf(d_ball.real)
        else:
            K_val, d_val = +_acb_to_mpc(k_ball), +_acb_to_mpc(d_ball)
        pairs.append(dict(r=r, s=s, rs=rs, K=K_val, d=d_val, v=d_val + rs))
    return pairs


def _build_numeric_pair(r: int, s: int, rs: int, b, d1, d2, d3, d4, sixteen):
    """One recursion datum from the mpmath formulas (the LaurentSeries-b path of the regularizer)."""
    d_val = d_rs(r, s, b)
    K_val = (
        sixteen**rs
        * A_rs(r, s, b)
        * fusion_polynomial(r, s, b, d1, d2)
        * fusion_polynomial(r, s, b, d4, d3)
    )
    return dict(r=r, s=s, rs=rs, K=K_val, d=d_val, v=d_val + rs)


def _conjugate_pair(source, *, r: int, s: int, rs: int):
    """Materialize the strip twin of a canonical pair by conjugation."""
    return dict(
        r=r,
        s=s,
        rs=rs,
        K=mp.conj(source["K"]),
        d=mp.conj(source["d"]),
        v=mp.conj(source["v"]),
    )


def _pair_interaction_weights_flint(pair_table: PairTable):
    """Flint-acb version of the K / (v_outer - d_inner) weight matrix.

    The mpmath path is ``canonical x pairs`` complex divisions (~36k at
    Lambda=11/q_order=60), which dominate the one-time H-context build. Compute
    them in flint acb instead (uniformly, even for real Kac data, so the result
    matches the strip/generic path bit-for-bit -- same acb-uniform rule as the
    Kac construction). The structural markers are kept exact in mpmath: a weight
    is ``0`` where K_inner == 0 and ``None`` where v_outer == d_inner (the
    pole-coincidence sentinel the H-recursion checks for). Returns the
    ``{outer_idx: [weight per inner]}`` dict, or ``None`` for non-numeric Kac
    data (LaurentSeries).
    """
    pairs = pair_table.pairs
    if any(not isinstance(p["K"], (mp.mpf, mp.mpc)) for p in pairs):
        return None
    from flint import acb, ctx as flint_ctx

    flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
    real_out = _accumulator_field(pairs) is mp.mpf
    zero = mp.mpf(0) if real_out else mp.mpc(0)

    K_mp = [p["K"] for p in pairs]
    d_mp = [p["d"] for p in pairs]
    K_is_zero = [k == 0 for k in K_mp]
    K_acb = [_mpf_to_acb(k) for k in K_mp]
    d_acb = [_mpf_to_acb(d) for d in d_mp]
    n = len(pairs)
    d_to_indices = {}
    for j, dj in enumerate(d_mp):
        d_to_indices.setdefault(dj, []).append(j)

    weights, flint_weights = {}, {}
    acb_zero = acb(0)
    for idx in pair_table.canonical_indices:
        v_mp = pairs[idx]["v"]
        v_acb = _mpf_to_acb(v_mp)
        row = [zero] * n
        flint_row = [acb_zero] * n
        pole_cols = d_to_indices.get(v_mp, ())
        for j in pole_cols:
            row[j] = flint_row[j] = None
        for j in range(n):
            if K_is_zero[j]:
                continue
            if row[j] is None:
                continue
            w = K_acb[j] / (v_acb - d_acb[j])
            flint_row[j] = w
            row[j] = _arb_to_mpf(w.real) if real_out else _acb_to_mpc(w)
        weights[idx] = row
        flint_weights[idx] = flint_row
    pair_table.interaction_weights_flint = flint_weights
    return weights


def _pair_interaction_weights(pair_table: PairTable):
    """Lazy cache of K / (v_outer - d_inner) weights for the pair table."""
    if pair_table.interaction_weights is None:
        weights = _pair_interaction_weights_flint(pair_table)
        if weights is None:
            field = _accumulator_field(pair_table.pairs)
            weights = {
                idx: _fixed_pole_weights(pair_table.pairs[idx]["v"], pair_table.pairs, field(0))
                for idx in pair_table.canonical_indices
            }
        pair_table.interaction_weights = weights
    return pair_table.interaction_weights


def _build_rs_data_uncached(q_order: int, b, h_ext) -> PairTable:
    """Precompute K_{r,s}, d_{r,s}, v_{r,s} for all (r,s) with r*s <= q_order.

    Returns a PairTable whose `.pairs` carries dicts with keys 'r', 's', 'rs',
    'K', 'd', 'v', and whose `.mode` field records which arithmetic regime
    was used. The mode is one of `"real_b"` (c > 25, real externals),
    `"strip_twin"` (1 < c < 25, real externals; b on the unit circle, K's in
    conjugate twins), `"imag_b"` (c < 1, real externals; b on the imaginary
    axis, K and d kept in mp.mpf), or `"generic_complex"` (everything else,
    including the singular endpoints c in {1, 25}).

    Numeric data come from :func:`_kac_balls`, starting at the H fill's first precision and increasing until the
    residues and weights are certified. They are kept as ``kac_balls``; only then are their midpoints rounded
    into ``pairs``. This preserves small nonzero residues in symbolic pole selection. A LaurentSeries b keeps
    the mpmath formulas.
    """
    mode = _classify_mode(b, h_ext)
    numeric = _numeric_kac_inputs(b, h_ext)
    if numeric is None:
        return _build_rs_data_mpmath(q_order, b, h_ext, mode)

    from flint import ctx as flint_ctx

    b_num, ext = numeric
    work = mp.mp.prec
    prec, previous = _h_fill_first_prec(q_order), None
    c_exact = _b_origin(b_num)
    try:
        while True:
            flint_ctx.prec = prec
            try:
                K, d = _kac_balls(q_order, b_num, ext, mode, c=c_exact)
                if c_exact is not None and _kac_collides(
                        q_order, d, _poles._kac_collision_checker(b_num, c=c_exact)):
                    raise _SingularKacData
                deficit = _relative_deficit((*K, *d), work)
            except _SingularKacData:
                # The bare recursion's singular-c probes intentionally use
                # b as stored. Only a proved source collision selects this
                # fallback; an unresolved enclosure must keep its source c.
                c_exact = None
                continue
            except _UnresolvedKacData:
                deficit = work
            if deficit == 0:
                break
            step = _poles.next_certification_prec(prec, deficit, previous, work)
            if step is None:
                raise RecursionPrecisionError(f"Kac data cannot be certified to {mp.mp.dps} digits "
                                              f"at {prec} bits (q_order {q_order})")
            prec, previous = step
    finally:
        flint_ctx.prec = _flint_prec_bits()
    real = mode == "imag_b" or (_is_real_scalar(b_num) and all(_is_real_scalar(x) for x in ext))
    pairs = _pairs_from_kac_balls(q_order, K, d, real)
    canonical_indices, twin_of, is_canonical = _build_pair_indexing(pairs, mode == "strip_twin")
    return PairTable(
        pairs=pairs,
        canonical_indices=canonical_indices,
        twin_of=twin_of,
        is_canonical=is_canonical,
        mode=mode,
        kac_balls={prec: (K, d)},
        kac_from_c=c_exact is not None,
    )


def _build_rs_data_mpmath(q_order: int, b, h_ext, mode: str) -> PairTable:
    """The mpmath-formula pair table (a LaurentSeries b: the regularizer's exact Laurent analysis)."""
    h1, h2, h3, h4 = h_ext
    sixteen = mp.mpf(16)
    pairs = []
    strip_symmetric = mode == "strip_twin"
    if strip_symmetric:
        canonical_pairs = {}
        for r, s, rs in _enumerate_rs_pairs(q_order):
            if r > s:
                continue
            canonical_pairs[(r, s)] = _build_numeric_pair(
                r, s, rs, b, h1, h2, h3, h4, sixteen
            )
        for r, s, rs in _enumerate_rs_pairs(q_order):
            if r <= s:
                pairs.append(canonical_pairs[(r, s)])
            else:
                pairs.append(
                    _conjugate_pair(canonical_pairs[(s, r)], r=r, s=s, rs=rs)
                )
    else:
        for r, s, rs in _enumerate_rs_pairs(q_order):
            pairs.append(_build_numeric_pair(r, s, rs, b, h1, h2, h3, h4, sixteen))

    canonical_indices, twin_of, is_canonical = _build_pair_indexing(
        pairs, strip_symmetric
    )
    return PairTable(
        pairs=pairs,
        canonical_indices=canonical_indices,
        twin_of=twin_of,
        is_canonical=is_canonical,
        mode=mode,
    )


def _build_rs_data(q_order: int, b, h_ext) -> PairTable:
    """Precision-aware cached wrapper around `_build_rs_data_uncached`."""
    cache_key = _pair_table_cache_key(q_order, b, h_ext)
    if cache_key is not None:
        cached = _PAIR_TABLE_CACHE.get(cache_key)
        if cached is not None:
            _PAIR_TABLE_CACHE.move_to_end(cache_key)
            return cached

    pair_table = _build_rs_data_uncached(q_order, b, h_ext)

    if cache_key is not None:
        _PAIR_TABLE_CACHE[cache_key] = pair_table
        _PAIR_TABLE_CACHE.move_to_end(cache_key)
        if len(_PAIR_TABLE_CACHE) > _PAIR_TABLE_CACHE_MAXSIZE:
            _PAIR_TABLE_CACHE.popitem(last=False)
    return pair_table


def _fixed_pole_weights(eval_arg, pairs, zero):
    """Precompute K / (eval_arg - d) for a fixed evaluation argument.

    The q-recursion revisits the same denominator against many different inner
    H-values. Caching the quotient once per evaluation point cuts repeated
    high-precision divisions without changing the arithmetic.

    Thin alias for :func:`virasoro_blocks.poles.fixed_pole_weights`: the
    operation is common to every block family (see that module), so it lives
    there and the Virasoro path delegates rather than keeping a copy.
    """
    return _poles.fixed_pole_weights(eval_arg, pairs, zero)


def _horner_H_q_series(H_coeffs, q, *, coeffs_real: bool | None = None):
    """Evaluate sum_k H_k q^k in the narrowest compatible field."""
    if coeffs_real is None:
        coeffs_real = all(isinstance(coeff, mp.mpf) for coeff in H_coeffs)
    q_real = isinstance(q, mp.mpf)
    H_val = mp.mpf(0) if (coeffs_real and q_real) else mp.mpc(0)
    for k in range(len(H_coeffs) - 1, -1, -1):
        H_val = H_val * q + H_coeffs[k]
    return H_val


def _build_H_arb_context(pairs, H_table, q_order, balls=None):
    """Pre-convert the real non-strip recursion data to flint arb.

    Returns a dict of arb arrays (K, d, rs, H_table) reused by
    :func:`_eval_H_at_arb_real` for every sample Delta, or ``None`` when the
    arb fast path does not apply (any complex K/d/H, or the strip-symmetric
    mode).  Built once per H-context (so once per
    ``(q_order, b, h_ext, precision)``), it moves the per-Delta accumulation off
    pure-Python mpmath bignums onto compiled arb arithmetic.

    The precomputed ``M`` (arb_mat, shape (q_order+1, n_pairs)) encodes the
    shifted H-table rows so that :func:`_eval_H_at_arb_real` evaluates the full
    recursion as a single arb_mat @ arb_vec product in compiled C.

    ``balls`` (from :func:`_certified_h_table`: ``K``, ``d`` and the ``H`` rows as arb) replaces the exact
    conversion of the mpmath values, so the context carries the certified radii (a midpoint read as exact would
    claim digits the fill does not have).
    """
    for p in pairs:
        if not isinstance(p["K"], mp.mpf) or not isinstance(p["d"], mp.mpf):
            return None
    for idx in range(len(pairs)):
        row = H_table[idx]
        if row is None or any(not isinstance(v, mp.mpf) for v in row):
            return None
    from flint import arb, arb_mat, ctx as flint_ctx

    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    n_pairs = len(pairs)

    if balls is not None:
        K_arb, d_arb, H_arb = balls["K"], balls["d"], balls["H"]
        K_zero = [k.is_zero() for k in K_arb]
    else:
        K_arb = [_mpf_to_arb(p["K"]) for p in pairs]
        d_arb = [_mpf_to_arb(p["d"]) for p in pairs]
        K_zero = [p["K"] == 0 for p in pairs]
        H_arb = [[_mpf_to_arb(v) for v in H_table[idx]] for idx in range(n_pairs)]
    rs_list = [p["rs"] for p in pairs]

    # M[k, idx] = H[idx][k - rs[idx]] for k >= rs[idx], else 0.
    # out[k] = sum_idx w[idx] * M[k, idx] (one arb_mat matmul per eval call).
    zero = arb(0)
    M_rows = [[zero] * n_pairs for _ in range(q_order + 1)]
    for idx, (rs, k_z, H_row) in enumerate(zip(rs_list, K_zero, H_arb)):
        if k_z:
            continue
        for i, h_val in enumerate(H_row):
            k = rs + i
            if k > q_order:
                break
            M_rows[k][idx] = h_val
    M = arb_mat(M_rows)

    return {
        "mode": "real_nonstrip",
        "prec": prec_bits,
        "rs": rs_list,
        "K_zero": K_zero,
        "K": K_arb,
        "d": d_arb,
        "H": H_arb,
        "M": M,
        "n_pairs": n_pairs,
    }


def _build_H_acb_nonstrip_context(pairs, H_table, q_order, balls=None):
    """Pre-convert the complex non-strip recursion data to flint acb.

    The generic-complex mode (complex c / complex externals) carries complex
    Kac data, so the real arb context (:func:`_build_H_arb_context`) does not
    apply. This builds the acb analogue: the K/d pole data and shifted H-table
    ``M`` as ``acb`` -- so the same matmul evaluator
    (:func:`_eval_H_at_acb_nonstrip_complex`) can drive every sample h. Built
    once per H-context; returns ``None`` if any H-table row is missing (a
    pole-coincidence weight forces the scalar fill).

    Real Kac data is handled by :func:`_build_H_arb_context` instead; this path
    is only taken when that one declines (complex K / d / H).  ``balls`` as in
    :func:`_build_H_arb_context` (acb).
    """
    for idx in range(len(pairs)):
        if H_table[idx] is None:
            return None
    from flint import acb, acb_mat, ctx as flint_ctx

    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    n_pairs = len(pairs)

    if balls is not None:
        K_acb = [acb(k) for k in balls["K"]]
        d_acb = [acb(x) for x in balls["d"]]
        H_rows = [[acb(v) for v in row] for row in balls["H"]]
        K_zero = [k.is_zero() for k in K_acb]
    else:
        K_acb = [_mpf_to_acb(p["K"]) for p in pairs]
        d_acb = [_mpf_to_acb(p["d"]) for p in pairs]
        H_rows = [[_mpf_to_acb(v) for v in row] for row in H_table]
        K_zero = [p["K"] == 0 for p in pairs]
    rs_list = [p["rs"] for p in pairs]

    # M[k, idx] = H[idx][k - rs[idx]] for k >= rs[idx], else 0, so
    # out[k] = sum_idx w[idx] * M[k, idx] is one acb_mat matmul per eval.
    zero = acb(0)
    M_rows = [[zero] * n_pairs for _ in range(q_order + 1)]
    for idx in range(n_pairs):
        if K_zero[idx]:
            continue
        rs = rs_list[idx]
        for i, v in enumerate(H_rows[idx]):
            k = rs + i
            if k > q_order:
                break
            M_rows[k][idx] = v
    M_acb = acb_mat(M_rows)

    return {
        "mode": "complex_nonstrip",
        "prec": prec_bits,
        "K_zero": K_zero,
        "K_acb": K_acb,
        "d_acb": d_acb,
        "M_acb": M_acb,
        "n_pairs": n_pairs,
    }


def _build_H_acb_strip_context(pair_table, pairs, H_table, q_order, balls=None):
    """Pre-convert the strip-symmetric recursion data to flint acb.

    The 1 < c < 25 ("strip") regime carries complex Kac data (b on the unit
    circle) but a real block at real h.  The mpc accumulation is ~10x slower
    than acb at dps=80, so this pre-builds acb arrays over the *canonical*
    (r <= s) pairs -- the strip-real evaluator sums those with the r==s / 2*Re
    twin weighting, matching the mpmath path.  Returns ``None`` if any
    canonical H-table row is missing (a pole-coincidence weight).

    The precomputed ``M_re`` / ``M_im`` (arb_mat, shape (q_order+1, n_entries))
    store scale * Re(Hrow[k-rs]) and scale * Im(Hrow[k-rs]) respectively, so
    :func:`_eval_H_at_acb_strip_real` evaluates out[k] = sum_j Re(w_j *
    Hrow_j[k-rs_j]) via two arb_mat @ arb_vec products in compiled C instead of
    a Python double-loop over k and j.  ``balls`` as in :func:`_build_H_arb_context` (acb; the canonical ``H``
    rows, the full ``K`` / ``d`` lists).
    """
    canonical_indices = pair_table.canonical_indices
    for idx in canonical_indices:
        if H_table[idx] is None:
            return None
    from flint import arb, acb, arb_mat, acb_mat, ctx as flint_ctx

    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits

    if balls is not None:
        K_full = list(balls["K"])
        d_full = list(balls["d"])
        H_rows = balls["H"]
        K_zero_full = [k.is_zero() for k in K_full]
    else:
        K_full = [_mpf_to_acb(p["K"]) for p in pairs]
        d_full = [_mpf_to_acb(p["d"]) for p in pairs]
        H_rows = [None if row is None else [_mpf_to_acb(v) for v in row] for row in H_table]
        K_zero_full = [p["K"] == 0 for p in pairs]

    entries = []
    for idx in canonical_indices:
        p = pairs[idx]
        entries.append((
            p["rs"],
            p["r"] == p["s"],
            K_zero_full[idx],
            K_full[idx],
            d_full[idx],
            H_rows[idx],
        ))

    # Full-pair acb data (including non-canonical rows via conjugation) for
    # generic complex-h strip evaluation.
    is_canonical = pair_table.is_canonical
    twin_of = pair_table.twin_of
    n_pairs_full = len(pairs)
    M_full_rows = [[acb(0)] * n_pairs_full for _ in range(q_order + 1)]
    for idx, p in enumerate(pairs):
        if is_canonical[idx]:
            Hrow = H_rows[idx]
        else:
            Hrow = [v.conjugate() for v in H_rows[twin_of[idx]]]
        rs = p["rs"]
        for i, h_val in enumerate(Hrow):
            k = rs + i
            if k > q_order:
                break
            M_full_rows[k][idx] = h_val
    M_full = acb_mat(M_full_rows)

    # Precompute M_re[k, j] = scale_j * Re(Hrow_j[k - rs_j]) and
    #            M_im[k, j] = scale_j * Im(Hrow_j[k - rs_j]) (0 for k < rs_j).
    # out[k] = (M_re @ Re_w_vec - M_im @ Im_w_vec)[k] avoids the Python k-loop.
    n_entries = len(entries)
    zero = arb(0)
    M_re_rows = [[zero] * n_entries for _ in range(q_order + 1)]
    M_im_rows = [[zero] * n_entries for _ in range(q_order + 1)]
    two = arb(2)
    for j, (rs, diag, k_zero, _K, _d, Hrow) in enumerate(entries):
        if k_zero:
            continue
        scale = arb(1) if diag else two
        for i, h_val in enumerate(Hrow):
            k = rs + i
            if k > q_order:
                break
            M_re_rows[k][j] = scale * h_val.real
            M_im_rows[k][j] = scale * h_val.imag
    M_re = arb_mat(M_re_rows)
    M_im = arb_mat(M_im_rows)

    return {
        "mode": "strip_real",
        "prec": prec_bits,
        "entries": entries,
        "M_re": M_re,
        "M_im": M_im,
        "n_entries": n_entries,
        "K_full": K_full,
        "d_full": d_full,
        "K_zero_full": K_zero_full,
        "M_full": M_full,
        "n_pairs_full": n_pairs_full,
    }


def _eval_H_at_acb_strip_real(h_typed, acb_ctx, q_order, *, return_arb=False):
    """acb port of the strip-symmetric real-h ``_eval_H_at`` branch.

    Returns ``[H_0, ..., H_{q_order}]`` as mp.mpf (or as arb when
    ``return_arb=True``), or ``None`` to fall back to the mpmath path (a
    Kac-pole hit makes a weight non-finite).

    The inner double-loop over k and j is replaced by two arb_mat matmuls:
      out[k] = Re(sum_j w_j * Hrow_j[k-rs_j])
             = (M_re @ Re_w_vec - M_im @ Im_w_vec)[k]
    where M_re / M_im are precomputed once in the context.
    """
    from flint import arb, acb, arb_mat, ctx as flint_ctx

    flint_ctx.prec = acb_ctx["prec"]
    entries = acb_ctx["entries"]
    M_re = acb_ctx["M_re"]
    M_im = acb_ctx["M_im"]
    n_entries = acb_ctx["n_entries"]
    h_acb = acb(_mpf_to_arb(h_typed))

    # Compute per-entry weights; bail if h lands on a Kac pole.
    zero = arb(0)
    Re_w = []
    Im_w = []
    for (_rs, _diag, k_zero, K, d, _Hrow) in entries:
        if k_zero:
            Re_w.append(zero)
            Im_w.append(zero)
            continue
        w = K / (h_acb - d)
        if not w.is_finite():
            return None
        Re_w.append(w.real)
        Im_w.append(w.imag)

    # Column vectors (n_entries × 1) for the matmul.
    Re_w_vec = arb_mat([[v] for v in Re_w])
    Im_w_vec = arb_mat([[v] for v in Im_w])

    # out[k] = (M_re @ Re_w_vec - M_im @ Im_w_vec)[k, 0]  for k = 1..q_order.
    result = M_re * Re_w_vec - M_im * Im_w_vec  # (q_order+1, 1)

    out = [arb(1)] + [result[k, 0] for k in range(1, q_order + 1)]
    if return_arb:
        return out
    return [_arb_to_mpf(o) for o in out]


def _eval_H_at_acb_strip_complex(h_typed, acb_ctx, q_order, *, return_acb=False):
    """acb-matmul strip evaluator for generic complex h.

    Returns ``[H_0, ..., H_q]`` as mpmath complex scalars (or as native acb
    when ``return_acb=True``), or ``None`` if h lands on a Kac pole (non-finite
    weight), matching scalar fallback semantics.
    """
    from flint import acb, acb_mat, ctx as flint_ctx

    flint_ctx.prec = acb_ctx["prec"]
    K = acb_ctx["K_full"]
    d = acb_ctx["d_full"]
    K_zero = acb_ctx["K_zero_full"]
    M = acb_ctx["M_full"]
    n_pairs = acb_ctx["n_pairs_full"]

    h_acb = _mpf_to_acb(h_typed)
    zero = acb(0)
    w_data = []
    for idx in range(n_pairs):
        if K_zero[idx]:
            w_data.append([zero])
            continue
        w = K[idx] / (h_acb - d[idx])
        if not w.is_finite():
            return None
        w_data.append([w])
    w_vec = acb_mat(w_data)
    result = M * w_vec
    if return_acb:
        return [acb(1)] + [result[k, 0] for k in range(1, q_order + 1)]
    return [mp.mpf(1)] + [_acb_to_mpc(result[k, 0]) for k in range(1, q_order + 1)]


def _eval_H_at_acb_nonstrip_complex(h_typed, arb_ctx, q_order, *, return_acb=False):
    """acb-matmul non-strip evaluator for generic complex h on real Kac data.

    The real non-strip context (:func:`_build_H_arb_context`) holds the K/d
    pole data and the shifted H-table ``M`` as flint ``arb``. For a complex
    exchanged ``h`` (e.g. an eps-shifted sample Delta near a Kac pole) only the
    pole weights ``w[idx] = K[idx] / (h - d[idx])`` become complex, so we
    promote the (Delta-independent) arb data to ``acb`` once -- cached on the
    context -- and evaluate the recursion as a single acb_mat @ acb_vec product,
    replacing the per-h mpmath Python loop.

    Returns ``[H_0, ..., H_q]`` as mpmath complex scalars (or native acb when
    ``return_acb=True``), or ``None`` if ``h`` lands on a Kac pole (non-finite
    weight), matching the scalar fallback semantics.
    """
    from flint import acb, acb_mat, ctx as flint_ctx

    flint_ctx.prec = arb_ctx["prec"]
    M_acb = arb_ctx.get("M_acb")
    if M_acb is None:
        # Promote the real arb pole data + shifted H-table to acb once; reused
        # for every subsequent complex-h sample sharing this context.
        M = arb_ctx["M"]
        K_acb = [acb(v) for v in arb_ctx["K"]]
        d_acb = [acb(v) for v in arb_ctx["d"]]
        M_acb = acb_mat(
            [[acb(M[r, col]) for col in range(M.ncols())] for r in range(M.nrows())]
        )
        arb_ctx["K_acb"] = K_acb
        arb_ctx["d_acb"] = d_acb
        arb_ctx["M_acb"] = M_acb
    else:
        K_acb = arb_ctx["K_acb"]
        d_acb = arb_ctx["d_acb"]

    K_zero = arb_ctx["K_zero"]
    n_pairs = arb_ctx["n_pairs"]
    h_acb = _mpf_to_acb(h_typed)
    zero = acb(0)
    w_data = []
    for idx in range(n_pairs):
        if K_zero[idx]:
            w_data.append([zero])
            continue
        w = K_acb[idx] / (h_acb - d_acb[idx])
        if not w.is_finite():
            return None
        w_data.append([w])
    w_vec = acb_mat(w_data)
    result = M_acb * w_vec
    if return_acb:
        return [acb(1)] + [result[k, 0] for k in range(1, q_order + 1)]
    return [mp.mpf(1)] + [_acb_to_mpc(result[k, 0]) for k in range(1, q_order + 1)]


def _eval_H_at_arb_real(h_typed, arb_ctx, q_order, *, return_arb=False):
    """arb port of the real non-strip ``_eval_H_at`` branch.

    Returns ``[H_0, ..., H_{q_order}]`` as mp.mpf, or ``None`` to signal the
    caller to fall back to the mpmath path (e.g. ``h`` lands on a Kac pole, so
    a weight is non-finite -- the same configuration the mpmath path rejects).

    Uses the precomputed M matrix (arb_mat) from the context so the inner
    double-loop becomes a single arb_mat @ arb_vec product in compiled C:
      out[k] = sum_idx w[idx] * M[k, idx] = (M @ w_vec)[k].
    """
    from flint import arb, arb_mat, ctx as flint_ctx

    flint_ctx.prec = arb_ctx["prec"]
    K = arb_ctx["K"]
    d = arb_ctx["d"]
    K_zero = arb_ctx["K_zero"]
    M = arb_ctx["M"]
    n_pairs = arb_ctx["n_pairs"]
    h_arb = _mpf_to_arb(h_typed)

    # Compute weight vector; bail if h lands on a Kac pole.
    zero = arb(0)
    w_data = []
    for idx in range(n_pairs):
        if K_zero[idx]:
            w_data.append([zero])
            continue
        w = K[idx] / (h_arb - d[idx])
        if not w.is_finite():
            return None
        w_data.append([w])
    w_vec = arb_mat(w_data)  # (n_pairs, 1)

    result = M * w_vec  # (q_order+1, 1)
    out = [arb(1)] + [result[k, 0] for k in range(1, q_order + 1)]
    if return_arb:
        return out
    return [_arb_to_mpf(o) for o in out]


def _eval_H_at_many(
    h_args,
    pairs,
    H_table,
    q_order: int,
    *,
    strip_symmetric: bool = False,
    canonical_indices=None,
    is_canonical=None,
    twin_of=None,
    arb_ctx=None,
):
    """Batch evaluate ``_eval_H_at`` for multiple h values.

    Returns a list aligned with ``h_args``. Entries are either
    ``[H_0, ..., H_q]`` or ``None`` (for non-finite weights at a Kac pole),
    matching the single-point helpers' fallback semantics.
    """
    if not h_args:
        return []

    # Fast batched flint path for real h where we have prebuilt matrix context.
    if arb_ctx is not None and arb_ctx.get("mode") in ("real_nonstrip", "strip_real"):
        typed = []
        real_idx = []
        out = [None] * len(h_args)
        for i, h_arg in enumerate(h_args):
            h_typed = (
                _maybe_real_scalar(h_arg)
                if isinstance(h_arg, (mp.mpf, mp.mpc))
                else _maybe_real_scalar(mp.mpc(h_arg))
            )
            typed.append(h_typed)
            if isinstance(h_typed, mp.mpf):
                real_idx.append(i)

        # The strip_real batch handles complex h too (the regular weights
        # d +/- ell in numeric pole ranking are complex), so enter it whenever
        # there is anything to evaluate -- not only when some h is real. The
        # real_nonstrip batch only covers real h (complex ones fall through to
        # the scalar fill below).
        if real_idx or (arb_ctx.get("mode") == "strip_real" and h_args):
            from flint import arb, acb, arb_mat, ctx as flint_ctx

            flint_ctx.prec = arb_ctx["prec"]
            n_cols = len(real_idx)

            if arb_ctx["mode"] == "real_nonstrip":
                K = arb_ctx["K"]
                d = arb_ctx["d"]
                K_zero = arb_ctx["K_zero"]
                M = arb_ctx["M"]
                n_pairs = arb_ctx["n_pairs"]

                zero = arb(0)
                finite_col = [True] * n_cols
                W_rows = [[zero] * n_cols for _ in range(n_pairs)]
                for col, src in enumerate(real_idx):
                    h_arb = _mpf_to_arb(typed[src])
                    for row in range(n_pairs):
                        if K_zero[row]:
                            continue
                        w = K[row] / (h_arb - d[row])
                        if not w.is_finite():
                            finite_col[col] = False
                            break
                        W_rows[row][col] = w

                W = arb_mat(W_rows)
                R = M * W
                for col, src in enumerate(real_idx):
                    if not finite_col[col]:
                        continue
                    out[src] = [mp.mpf(1)] + [
                        _arb_to_mpf(R[k, col])
                        for k in range(1, q_order + 1)
                    ]

            elif arb_ctx["mode"] == "strip_real":
                from flint import acb_mat

                K = arb_ctx["K_full"]
                d = arb_ctx["d_full"]
                K_zero = arb_ctx["K_zero_full"]
                M = arb_ctx["M_full"]
                n_pairs = arb_ctx["n_pairs_full"]

                all_idx = list(range(len(h_args)))
                n_cols_all = len(all_idx)
                finite_col = [True] * n_cols_all
                W_rows = [[acb(0)] * n_cols_all for _ in range(n_pairs)]
                for col, src in enumerate(all_idx):
                    h_t = typed[src]
                    if isinstance(h_t, mp.mpc):
                        h_acb = _mpf_to_acb(h_t)
                    else:
                        h_acb = acb(_mpf_to_arb(h_t))
                    for row in range(n_pairs):
                        if K_zero[row]:
                            continue
                        w = K[row] / (h_acb - d[row])
                        if not w.is_finite():
                            finite_col[col] = False
                            break
                        W_rows[row][col] = w

                W = acb_mat(W_rows)
                R = M * W
                for col, src in enumerate(all_idx):
                    if not finite_col[col]:
                        continue
                    out[src] = [mp.mpf(1)] + [
                        _acb_to_mpc(R[k, col])
                        for k in range(1, q_order + 1)
                    ]

            # Fill any non-real or failed columns through the scalar path.
            for i in range(len(h_args)):
                if out[i] is None:
                    out[i] = _eval_H_at(
                        h_args[i],
                        pairs,
                        H_table,
                        q_order,
                        strip_symmetric=strip_symmetric,
                        canonical_indices=canonical_indices,
                        is_canonical=is_canonical,
                        twin_of=twin_of,
                        arb_ctx=arb_ctx,
                    )
            return out

    # No prebuilt real/strip context (or nothing to batch): scalar path per h.
    return [
        _eval_H_at(
            h_arg,
            pairs,
            H_table,
            q_order,
            strip_symmetric=strip_symmetric,
            canonical_indices=canonical_indices,
            is_canonical=is_canonical,
            twin_of=twin_of,
            arb_ctx=arb_ctx,
        )
        for h_arg in h_args
    ]


def _eval_H_at(
    h_arg,
    pairs,
    H_table,
    q_order: int,
    *,
    strip_symmetric: bool = False,
    canonical_indices=None,
    is_canonical=None,
    twin_of=None,
    arb_ctx=None,
) -> list:
    """Compute [H_0(h_arg), ..., H_{q_order}(h_arg)] using cached H_table.

    H_table is indexed by pair_index (in `pairs`) -> list of H_k values
    already filled up to the order we need. In strip-symmetric mode the table
    stores only canonical rows (r <= s); non-canonical rows are reconstructed
    by conjugation on lookup.
    """
    if strip_symmetric:
        if canonical_indices is None or is_canonical is None or twin_of is None:
            raise ValueError("strip-symmetric evaluation requires pair metadata")

        def lookup_h(pair_idx: int, k: int):
            if is_canonical[pair_idx]:
                return H_table[pair_idx][k]
            return mp.conj(H_table[twin_of[pair_idx]][k])

        h_typed = (
            _maybe_real_scalar(h_arg)
            if isinstance(h_arg, (mp.mpf, mp.mpc))
            else _maybe_real_scalar(mp.mpc(h_arg))
        )
        if (
            isinstance(h_typed, mp.mpf)
            and arb_ctx is not None
            and arb_ctx.get("mode") == "strip_real"
        ):
            strip_out = _eval_H_at_acb_strip_real(h_typed, arb_ctx, q_order)
            if strip_out is not None:
                return strip_out
        if (
            isinstance(h_typed, mp.mpc)
            and arb_ctx is not None
            and arb_ctx.get("mode") == "strip_real"
        ):
            strip_out = _eval_H_at_acb_strip_complex(h_typed, arb_ctx, q_order)
            if strip_out is not None:
                return strip_out
        top_weights = _fixed_pole_weights(h_typed, pairs, mp.mpc(0))
        if isinstance(h_typed, mp.mpf):
            out = [mp.mpf(0)] * (q_order + 1)
            out[0] = mp.mpf(1)
            for k in range(1, q_order + 1):
                acc = mp.mpf(0)
                for idx in canonical_indices:
                    p = pairs[idx]
                    if p["rs"] > k:
                        break
                    inner = lookup_h(idx, k - p["rs"])
                    if inner == 0:
                        continue
                    weight = top_weights[idx]
                    if weight is None:
                        term = p["K"] / (h_typed - p["d"]) * inner
                    else:
                        term = weight * inner
                    if p["r"] == p["s"]:
                        acc += mp.re(term)
                    else:
                        acc += 2 * mp.re(term)
                out[k] = acc
            return out

        h_arg = mp.mpc(h_typed)
        top_weights = _fixed_pole_weights(h_arg, pairs, mp.mpc(0))
        out = [mp.mpc(0)] * (q_order + 1)
        out[0] = mp.mpc(1)
        for k in range(1, q_order + 1):
            acc = mp.mpc(0)
            for idx, p in enumerate(pairs):
                if p["rs"] > k:
                    break
                inner = lookup_h(idx, k - p["rs"])
                if inner == 0:
                    continue
                weight = top_weights[idx]
                if weight is None:
                    acc += p["K"] / (h_arg - p["d"]) * inner
                    continue
                if weight == 0:
                    continue
                acc += weight * inner
            out[k] = acc
        return out

    # Non-strip: real_b / imag_b / generic_complex. When every K and d is
    # mpf, the H-table accumulator stays in mp.mpf if h_arg is also real;
    # otherwise (and for generic_complex) we keep the mp.mpc default.
    pairs_field = _accumulator_field(pairs)
    h_typed = (
        _maybe_real_scalar(h_arg)
        if isinstance(h_arg, (mp.mpf, mp.mpc))
        else _maybe_real_scalar(mp.mpc(h_arg))
    )
    if pairs_field is mp.mpf and isinstance(h_typed, mp.mpf):
        if arb_ctx is not None and arb_ctx.get("mode") == "real_nonstrip":
            arb_out = _eval_H_at_arb_real(h_typed, arb_ctx, q_order)
            if arb_out is not None:
                return arb_out
        top_weights = _fixed_pole_weights(h_typed, pairs, mp.mpf(0))
        out = [mp.mpf(0)] * (q_order + 1)
        out[0] = mp.mpf(1)
        for k in range(1, q_order + 1):
            acc = mp.mpf(0)
            for idx, p in enumerate(pairs):
                if p["rs"] > k:
                    break
                inner = H_table[idx][k - p["rs"]]
                if inner == 0:
                    continue
                weight = top_weights[idx]
                if weight is None:
                    acc += p["K"] / (h_typed - p["d"]) * inner
                    continue
                if weight == 0:
                    continue
                acc += weight * inner
            out[k] = acc
        return out

    # Complex h (real Kac data) or complex Kac data: stay in flint via the acb
    # non-strip evaluator (compiled acb_mat product) instead of the mpmath
    # k-loop below.
    if arb_ctx is not None and arb_ctx.get("mode") in (
        "real_nonstrip", "complex_nonstrip"
    ):
        acb_out = _eval_H_at_acb_nonstrip_complex(h_typed, arb_ctx, q_order)
        if acb_out is not None:
            return acb_out

    h_arg = mp.mpc(h_typed)
    top_weights = _fixed_pole_weights(h_arg, pairs, mp.mpc(0))
    out = [mp.mpc(0)] * (q_order + 1)
    out[0] = mp.mpc(1)
    for k in range(1, q_order + 1):
        acc = mp.mpc(0)
        for idx, p in enumerate(pairs):
            if p["rs"] > k:
                break  # pairs are sorted by rs, ascending
            # K = 0 pairs contribute the zero function; skipping them also
            # avoids a spurious 0/0 when h_arg lands exactly on the pole d
            # (a degenerate / minimal-model exchanged operator).
            inner = H_table[idx][k - p["rs"]]
            if inner == 0:
                continue
            weight = top_weights[idx]
            if weight is None:
                acc += p["K"] / (h_arg - p["d"]) * inner
                continue
            if weight == 0:
                continue
            acc += weight * inner
        out[k] = acc
    return out


def _fill_h_table_scalar(pair_table, interaction_weights, q_order, field):
    """Scalar mpmath fill of H_table[idx][k] (the Zamolodchikov q-recursion).

    H_table[idx][k] = H_k at the intermediate weight v of canonical pair idx,
    filled bottom-up in k from H_table[idx][0] = 1. Used as the fallback when
    the vectorized acb path is inapplicable (a v_outer == d_inner
    pole-coincidence weight). Returns the H_table list.

    The loop itself is the family-generic one in
    :func:`virasoro_blocks.poles.fill_level_table`; what is Virasoro-specific
    and stays here is the seed (``H = 1 + ...``, i.e. a constant 1 rather than a
    series) and the strip-symmetric ``lookup`` that reconstructs non-canonical
    rows by conjugation from their ``r <= s`` twin.
    """
    pairs = pair_table.pairs
    is_canonical = pair_table.is_canonical
    twin_of = pair_table.twin_of

    seed = [field(0)] * (q_order + 1)
    seed[0] = field(1)

    def lookup(table, idx, k):
        if is_canonical[idx]:
            return table[idx][k]
        return mp.conj(table[twin_of[idx]][k])

    return _poles.fill_level_table(
        pairs,
        interaction_weights,
        q_order,
        {0: seed},
        outer_indices=pair_table.canonical_indices,
        lookup=lookup,
    )


class RecursionPrecisionError(ValueError):
    """A recursion level table (the H table at the shifted Kac weights) could not be certified to the working
    precision: more guard bits did not shrink the measured loss."""


# The first guard of the certified H fill, in bits per q-order: the balls' bound grows by 1.5 - 2.1 bits per q-order
# at light externals and 1 < c < 25 (124 bits at c = 8, h_ext = 1/16, q_order 80, whatever the precision), up to ~3.8
# at c = 100 or h_ext = 1, which take a second pass.  Deliberately a fixed function of (q_order, precision): a guard
# remembered from an earlier build would make the last bits of a table depend on what ran before it.
_H_FILL_GUARD_RATE = 2


def _h_fill_first_prec(q_order: int) -> int:
    """First flint precision of the certified H fill (:data:`_H_FILL_GUARD_RATE`)."""
    return _flint_prec_bits() + _H_FILL_GUARD_RATE * int(q_order) + 16


class _KacCollision(Exception):
    """``v_i - d_j`` is a ball around 0 at this precision (a pole collision, or one closer than it resolves)."""

    def __init__(self, outer, inner):
        super().__init__(outer, inner)
        self.outer, self.inner = outer, inner


def _fill_h_balls(pair_table, K, d, q_order: int, real: bool):
    """One fill of the Zamolodchikov H table at the current flint precision, from the Kac balls.

    For each level k the recursion
        H[outer][k] = sum_inner W[outer][inner] * H[inner][k - rs_inner]
    is one matrix-vector product: every H[inner][k - rs_inner] is from a level < k, and
    W[outer][inner] = K_inner / (v_outer - d_inner) (v = d + rs) is fixed.  Returns the canonical rows as balls
    (arb when ``real``, else acb; non-canonical rows are the conjugates of their strip twins), entry 0 the exact 1.
    """
    from flint import arb, acb, arb_mat, acb_mat

    pairs = pair_table.pairs
    n = len(pairs)
    canonical = pair_table.canonical_indices
    is_canonical = pair_table.is_canonical
    twin_of = pair_table.twin_of
    rs = [p["rs"] for p in pairs]
    if real:
        # the real modes run the per-level matvec in arb (~4x faster than acb at production size)
        K = [k.real for k in K]
        d = [x.real for x in d]
        zero, one, mat = arb(0), arb(1), arb_mat
    else:
        zero, one, mat = acb(0), acb(1), acb_mat
    live = [j for j in range(n) if not K[j].is_zero()]
    rows = []
    for i in canonical:
        v = d[i] + rs[i]
        row = [zero] * n
        for j in live:
            if rs[j] > q_order - rs[i]:
                break               # row i is read only up to level q_order - rs_i
            gap = v - d[j]
            if gap.contains(0):
                raise _KacCollision(i, j)
            row[j] = K[j] / gap
        rows.append(row)

    H = [None] * n
    for i in canonical:
        row = [zero] * (q_order - rs[i] + 1)
        row[0] = one
        H[i] = row
    # pairs (and canonical rows) are in ascending rs, so level k reads the inner-column prefix rs_j <= k and feeds
    # the outer-row prefix rs_i <= q_order - k.  The levels are taken in a few groups sharing one leading block of W
    # (the rows the group's first level feeds x the columns its last level reads): ~4x fewer products than the full
    # W at every level, without converting a new block per level.
    n_upto = [0] * (q_order + 1)
    c_upto = [0] * (q_order + 1)
    for j in range(n):
        if rs[j] <= q_order:
            n_upto[rs[j]] += 1
    for i in canonical:
        c_upto[rs[i]] += 1
    for m in range(1, q_order + 1):
        n_upto[m] += n_upto[m - 1]
        c_upto[m] += c_upto[m - 1]
    groups = 8
    bounds = sorted({1 + (q_order * g) // groups for g in range(groups)} | {q_order + 1})
    for lo, hi in zip(bounds, bounds[1:]):
        n_block_rows, n_block_cols = c_upto[q_order - lo], n_upto[hi - 1]
        if n_block_rows == 0 or n_block_cols == 0:
            continue
        block = mat([row[:n_block_cols] for row in rows[:n_block_rows]])
        for k in range(lo, hi):
            n_rows = c_upto[q_order - k]
            if n_rows == 0:
                continue
            column = []
            for j in range(n_block_cols):
                if rs[j] > k:
                    column.append([zero])
                elif is_canonical[j]:
                    column.append([H[j][k - rs[j]]])
                else:
                    twin = H[twin_of[j]][k - rs[j]]
                    column.append([twin if real else twin.conjugate()])
            product = block * mat(column)
            for pos in range(n_rows):
                H[canonical[pos]][k] = product[pos, 0]
    return H


def _h_table_deficit(H, canonical, work: int, input_bits: int = 0):
    """``(bits, (index, k))``: how far the least accurate entry falls short of ``work`` bits (0 when all are
    certified) and where it sits.  Each row ``H(v_i)`` is a q-series and is held to its own scale
    (:func:`_power_series_deficit`): an entry that cancels to zero is certified by a radius small against the row,
    not by relative bits it cannot have."""
    worst, where = 0, None
    for i in canonical:
        short, k = _power_series_deficit(H[i], work, input_bits)
        if short > worst:
            worst, where = short, (i, k)
    return worst, where


def _certified_h_table(pair_table, b, h_ext, q_order: int):
    """Return the H table and its certified balls: ``(H_table, balls)``.

    Cancellation can cost roughly one digit per q-order for heavy externals
    or near c = 1 and c = 25. Build the Kac data and the table in ball
    arithmetic, widening the guard until every entry meets the working
    precision relative to its row's scale (:func:`_h_table_deficit`).
    This includes regularization probes near singular c. If additional bits
    do not shrink the deficit, raise :class:`RecursionPrecisionError`
    (:func:`poles.next_certification_prec`).

    ``balls`` holds ``K``, ``d`` (every pair), and the canonical ``H`` rows
    with their radii at ``_flint_prec_bits()`` precision for flint evaluation.
    ``H_table`` holds the midpoints at the working precision.
    """
    from flint import ctx as flint_ctx

    b_num, ext = _numeric_kac_inputs(b, h_ext)
    c_exact = _b_origin(b_num) if pair_table.kac_from_c else None
    real = _accumulator_field(pair_table.pairs) is mp.mpf
    canonical = pair_table.canonical_indices
    work = mp.mp.prec
    prec = _h_fill_first_prec(q_order)
    previous = None
    collision_check = _poles._kac_collision_checker(b_num, c=c_exact)
    input_bits = _lazy(lambda: _kac_input_bits(b_num, ext))
    try:
        while True:
            flint_ctx.prec = prec
            cached = (pair_table.kac_balls or {}).get(prec)
            if cached is not None:
                K, d = cached
            else:
                try:
                    K, d = _kac_balls(q_order, b_num, ext, pair_table.mode, c=c_exact)
                except _SingularKacData:
                    raise ZeroDivisionError("the Kac data of this c do not exist (b^2 is rational)") from None
                except _UnresolvedKacData:
                    step = _poles.next_certification_prec(prec, work, None, work)
                    if step is None:
                        raise RecursionPrecisionError(f"Kac data do not resolve at {prec} bits "
                                                      f"(q_order {q_order})") from None
                    prec = step[0]
                    continue
            try:
                H = _fill_h_balls(pair_table, K, d, q_order, real)
            except _KacCollision as hit:
                outer, inner = pair_table.pairs[hit.outer], pair_table.pairs[hit.inner]
                if collision_check(outer, inner):
                    raise ZeroDivisionError(
                        f"Kac-pole collision v_{{{outer['r']},{outer['s']}}} = d_{{{inner['r']},{inner['s']}}} "
                        "the bare recursion is singular at this c"
                    ) from None
                step = _poles.next_certification_prec(prec, work, None, work)
                if step is None:
                    raise RecursionPrecisionError(f"Kac-pole near-collision does not resolve at {prec} bits "
                                                  f"(q_order {q_order})") from None
                prec = step[0]
                continue
            deficit, where = _h_table_deficit(H, canonical, work, input_bits)
            if deficit == 0:
                break
            step = _poles.next_certification_prec(prec, deficit, previous, work)
            if step is None:
                p = pair_table.pairs[where[0]]
                raise RecursionPrecisionError(
                    f"the H-recursion table cannot be certified to {mp.mp.dps} digits: its entry H_{where[1]} at the "
                    f"shifted Kac weight v_{{{p['r']},{p['s']}}} stays {deficit} bits short at {prec} bits of flint "
                    f"precision (q_order {q_order}, c = {mp.nstr(c_from_b(b_num), 12)}); the recursion cancels "
                    "beyond what guard bits recover here (a Kac-pole collision or a vanishing entry)"
                )
            prec, previous = step
        flint_ctx.prec = _flint_prec_bits()
        if real:
            K = [k.real for k in K]
            d = [x.real for x in d]
            H_table = [None if row is None else [+_arb_to_mpf(v) for v in row] for row in H]
        else:
            H_table = [None if row is None else [+_acb_to_mpc(v) for v in row] for row in H]
        balls = {
            "K": [+k for k in K],
            "d": [+x for x in d],
            "H": [None if row is None else [+v for v in row] for row in H],
            # the bits beyond the working precision the fill needed: the cancellation depth of the Kac data here
            "guard_bits": prec - work,
        }
        return H_table, balls
    finally:
        flint_ctx.prec = _flint_prec_bits()


def _build_H_context_uncached(q_order: int, b, h_ext):
    """Build the h-independent H-recursion context.

    Returns ``(pair_table, H_table, arb_ctx)``.  ``H_table[idx][k]`` is ``H_k``
    evaluated at the intermediate Kac weight of pair ``idx``; it depends only on
    the Kac data (K, d, v) and the externals -- not on the exchanged ``h`` -- so
    it is shared across every ``h`` (e.g. all sample Deltas of an interpolation)
    via :func:`_build_H_context`.  ``arb_ctx`` is the pre-converted flint form
    for the fast evaluators (or ``None``), built from the certified balls of
    :func:`_certified_h_table` so its radii are honest.
    """
    pair_table = _build_rs_data(q_order, b, h_ext)
    pairs = pair_table.pairs
    field = _accumulator_field(pairs)

    balls = None
    if _numeric_kac_inputs(b, h_ext) is not None:
        H_table, balls = _certified_h_table(pair_table, b, h_ext, q_order)
    else:
        # a LaurentSeries b: the scalar fill with its collision sentinel
        H_table = _fill_h_table_scalar(pair_table, _pair_interaction_weights(pair_table), q_order, field)

    if pair_table.is_strip_symmetric:
        arb_ctx = _build_H_acb_strip_context(pair_table, pairs, H_table, q_order, balls=balls)
    else:
        arb_ctx = _build_H_arb_context(pairs, H_table, q_order, balls=balls)
        if arb_ctx is None:
            # Complex Kac data (generic-complex mode): use the acb context so
            # the recursion stays in flint instead of the mpmath Python loop.
            arb_ctx = _build_H_acb_nonstrip_context(pairs, H_table, q_order, balls=balls)
    if arb_ctx is not None and balls is not None:
        arb_ctx["guard_bits"] = balls["guard_bits"]
    return pair_table, H_table, arb_ctx


def _b_at_working_precision(b):
    """A b formed at a higher precision is re-formed at this one, so one precision shares one table."""
    if isinstance(b, mp.mpf):
        bits = b._mpf_[3]
    elif isinstance(b, mp.mpc):
        bits = max(b._mpc_[0][3], b._mpc_[1][3])
    else:
        return b
    if bits <= mp.mp.prec:
        return b
    from .params import c_origin_of_b

    origin = c_origin_of_b(b)
    if origin is not None:
        return b_from_c(origin) if isinstance(b, mp.mpc) else b_from_c(origin).real
    return +b


def _build_H_context(q_order: int, b, h_ext):
    """Precision-aware cached wrapper around :func:`_build_H_context_uncached`."""
    b = _b_at_working_precision(b)
    cache_key = _pair_table_cache_key(q_order, b, h_ext)
    if cache_key is not None:
        cached = _H_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            _H_CONTEXT_CACHE.move_to_end(cache_key)
            return cached

    context = _build_H_context_uncached(q_order, b, h_ext)

    if cache_key is not None:
        _H_CONTEXT_CACHE[cache_key] = context
        _H_CONTEXT_CACHE.move_to_end(cache_key)
        if len(_H_CONTEXT_CACHE) > _H_CONTEXT_CACHE_MAXSIZE:
            _H_CONTEXT_CACHE.popitem(last=False)
    return context


def _H_laurent_precompute_cache_key(h_ext, c0, q_order: int, work_dps: int) -> tuple | None:
    c_key = _scalar_cache_key(c0)
    if c_key is None:
        return None
    ext_keys = []
    for hi in h_ext:
        hi_key = _scalar_cache_key(hi)
        if hi_key is None:
            return None
        ext_keys.append(hi_key)
    # no window: one table, built at the widest window asked for, serves every narrower one (narrowed_state)
    return (
        mp.mp.prec, int(work_dps), int(q_order),
        c_key, tuple(ext_keys),
    )


def _chiral_residue_series(pair_table: PairTable, H_table, q_order: int):
    """Return per-(r, s) residue q-series ``K_{r,s} * H(v_{r,s})``.

    For each surviving chiral pole (``K != 0``), returns a dict entry keyed by
    ``(r, s)`` with fields:

    - ``d``: the pole position ``d_{r,s}``
    - ``rs``: the pair level ``r*s``
    - ``series``: length ``q_order + 1`` list where
      ``series[k] = K_{r,s} * H_{k-rs}(v_{r,s})`` for ``k >= rs`` and ``0``
      otherwise.

    In strip-symmetric mode this uses the same canonical/twin-conjugation
    lookup as the recursion evaluator, so non-canonical entries are exact
    conjugates of their canonical twins.
    """
    pairs = pair_table.pairs
    is_canonical = pair_table.is_canonical
    twin_of = pair_table.twin_of
    strip_symmetric = pair_table.is_strip_symmetric

    if strip_symmetric:
        def lookup_h(pair_idx: int, k: int):
            if is_canonical[pair_idx]:
                return H_table[pair_idx][k]
            return mp.conj(H_table[twin_of[pair_idx]][k])
    else:
        def lookup_h(pair_idx: int, k: int):
            return H_table[pair_idx][k]

    residues = {}
    for idx, pair in enumerate(pairs):
        K = pair["K"]
        if K == 0:
            continue
        rs = pair["rs"]
        zero = K * 0
        series = [zero] * (q_order + 1)
        for k in range(rs, q_order + 1):
            series[k] = K * lookup_h(idx, k - rs)
        residues[(pair["r"], pair["s"])] = {
            "d": pair["d"],
            "rs": rs,
            "series": series,
        }
    return residues


def H_q_series(
    h,
    h_ext: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
) -> list:
    """Return [H_0, H_1, ..., H_{q_order}] where H(c, h, h_i, q) = sum_k H_k q^k.

    Pass either `b` or `c`, not both. `b` is the Liouville parameter
    (c = 1 + 6 (b + 1/b)^2).

    The (16)^{r s} factors of (16 q)^{r s} are absorbed into the returned H_k
    coefficients, so the caller multiplies by plain q (not 16 q) to evaluate.

    Every coefficient is certified to the working precision (:func:`_H_coefficient_balls`).
    A genuine internal Kac pole raises ``ZeroDivisionError``; a finite offset asks for more precision.
    """
    q_order = _integer_control('q_order', q_order)
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    if b is None:
        b = b_from_c(c)
    if len(h_ext) != 4:
        raise ValueError("h_ext must be a length-4 sequence (h1, h2, h3, h4)")

    h_typed, = _kac_input_values((h,))
    h_ext_typed = _kac_input_values(h_ext)
    # Demote b to mp.mpf when it's mathematically real, so the downstream Kac
    # data builders (d_rs, A_rs, fusion_polynomial) propagate mpf through.
    # `b_from_c(real_c)` returns mp.mpc(real, 0) at every c, so without this
    # step the real_b mode would keep its pair data in mp.mpc unnecessarily.
    if isinstance(b, mp.mpc):
        b = _maybe_real_scalar(b)
    certified = _H_coefficient_balls(h_typed, h_ext_typed, b, q_order)
    if certified is not None:
        balls, real = certified
        if real:
            return [+_arb_to_mpf(value) for value in balls]
        return [mp.mpf(1)] + [+_acb_to_mpc(value) for value in balls[1:]]

    # no flint evaluator applies (h on a Kac pole, or a LaurentSeries b): the mpmath recursion
    pair_table, H_table, arb_ctx = _build_H_context(q_order, b, h_ext_typed)
    return _eval_H_at(
        h_typed,
        pair_table.pairs,
        H_table,
        q_order,
        strip_symmetric=pair_table.is_strip_symmetric,
        canonical_indices=pair_table.canonical_indices,
        is_canonical=pair_table.is_canonical,
        twin_of=pair_table.twin_of,
        arb_ctx=arb_ctx,
    )


def _eval_H_balls(h_typed, arb_ctx, q_order: int):
    """``([H_0(h), ..., H_q(h)] as flint balls, real)`` from a flint context, or None when none applies (no
    context, or an unresolved pole denominator: a non-finite weight). The coefficient certification distinguishes
    an exact pole from a finite offset. The same dispatch as :func:`_eval_H_at`."""
    if arb_ctx is None:
        return None
    mode = arb_ctx.get("mode")
    real_h = _is_real_scalar(h_typed)
    if mode == "strip_real":
        if real_h:
            return _pair_or_none(_eval_H_at_acb_strip_real(h_typed, arb_ctx, q_order, return_arb=True), True)
        return _pair_or_none(_eval_H_at_acb_strip_complex(h_typed, arb_ctx, q_order, return_acb=True), False)
    if mode == "real_nonstrip" and real_h:
        return _pair_or_none(_eval_H_at_arb_real(h_typed, arb_ctx, q_order, return_arb=True), True)
    if mode in ("real_nonstrip", "complex_nonstrip"):
        return _pair_or_none(_eval_H_at_acb_nonstrip_complex(h_typed, arb_ctx, q_order, return_acb=True), False)
    return None


def _pair_or_none(values, real):
    return None if values is None else (values, real)


def _relative_deficit(values, work: int) -> int:
    """Bits by which the least accurate of ``values`` (flint balls: independent scalars such as Kac data or one
    block value, each held to its own magnitude) falls short of ``work`` relative bits; exact zeros are exact, a
    ball around an exact-zero midpoint certifies nothing.  The coefficients of a power series are held to the
    series' scale instead (:func:`_power_series_deficit`)."""
    worst = min((value.rel_accuracy_bits() for value in values), default=work)
    if worst < -(1 << 62):
        return 2 * work
    return max(work - worst, 0)


_LOG2 = math.log(2)


def _ball_radius_arb(value):
    """The radius of an arb, or the larger component radius of an acb (as ``rel_accuracy_bits`` reads it)."""
    if hasattr(value, "imag"):
        re_rad, im_rad = value.real.rad(), value.imag.rad()
        return re_rad if re_rad >= im_rad else im_rad
    return value.rad()


def _input_bits(values) -> int:
    """The bits of the exact inputs ``values`` (source labels: Fractions, ints, binary mpf / mpc), summed: a binary
    label counts its odd mantissa, a rational its numerator and denominator.  It is the resolution the inputs carry
    (two of them differ by at least about ``2^-bits``), measured from the inputs, not a guard constant."""
    from .numeric_input import literal_components

    total = 0
    for value in values:
        if value is None:
            continue
        for part in literal_components(value):
            num, den = abs(part.numerator), part.denominator
            if num == 0:
                continue
            if den & (den - 1) == 0:
                total += (num >> ((num & -num).bit_length() - 1)).bit_length()
            else:
                total += num.bit_length() + den.bit_length()
    return total


def _lazy(compute):
    """A zero-argument callable computing ``compute()`` once, on first call."""
    cell = []

    def value():
        if not cell:
            cell.append(compute())
        return cell[0]
    return value


def _kac_input_bits(b, values) -> int:
    """:func:`_input_bits` of ``values`` and of the exact c that ``b`` came from (``b`` itself when given directly)."""
    origin = _b_origin(b)
    return _input_bits((*values, b if origin is None else origin))


def _power_series_deficit(values, work: int, input_bits: int = 0):
    """``(bits, k)``: by how much the coefficients ``values = [a_0, a_1, ...]`` (flint balls) of a power series in
    a variable with ``|x| <= 1`` (q, or q^(1/2)) fall short of certification, and the first worst index (None when
    all are certified).

    A coefficient is certified when it carries ``work`` relative bits, or when its radius is below
    ``2^-(work + input_bits)`` of the series' scale up to its order, ``max_{j <= k} |a_j|``.  The scale is the
    whole-vector scale of the regularization gates and the Laurent rows (``poles._laurent_deficit``), taken as a
    running maximum so a small early coefficient is never measured against a later, larger one: the radius bounds
    the term's error by ``2^-work`` of the largest term ``|a_j x^j|, j <= k``, at any ``|x| <= 1``.  This is what
    certifies a coefficient that cancels -- to exactly zero by its exact inputs (c = 4, h = 0, h_ext = 1/16: H_2 = 0;
    a ball around 0 has no relative bit, so the relative test alone escalated to its cap and raised), or to
    something tiny.  ``input_bits`` (:func:`_input_bits`) is the inputs' own resolution: a coefficient made small by
    the inputs themselves (h_2 - h_1 = 10^-200 makes H_1 ~ 10^-200) is resolved down to it, so it keeps its relative
    digits.  A non-finite ball, or a radius with no scale at all, certifies nothing (``2 work``).  ``input_bits``
    may be a callable, called only when some coefficient misses the relative test (the hot path reads one
    ``rel_accuracy_bits`` per coefficient, as before)."""
    worst, where = 0, None
    scale, scanned, target = None, 0, None
    for k, value in enumerate(values):
        if not value.is_finite():
            return 2 * work, k
        relative = value.rel_accuracy_bits()
        if relative >= work:
            continue
        rad = _ball_radius_arb(value)
        if rad.is_zero():
            continue
        for prior in values[scanned:k + 1]:          # the running scale, extended only when it is needed
            size = prior.abs_lower()
            if scale is None or size > scale:
                scale = size
        scanned = k + 1
        short = work - relative if relative > -(1 << 62) else 2 * work
        if not scale.is_zero():
            if target is None:
                target = work + int(input_bits() if callable(input_bits) else input_bits)
            short = min(short, int(math.ceil(float((rad / scale).log()) / _LOG2)) + target)
        if short > worst:
            worst, where = short, k
    return worst, where


def _zero_centred(values):
    """Certified balls with every component that contains 0 re-centered on 0 (radius ``|mid| + rad``): a coefficient
    certified against its series' scale whose ball holds 0 is an exact (structural) zero at the working precision,
    and reads as exactly 0, not as the rounding residue of its cancellation."""
    from flint import acb, arb

    out = []
    for value in values:
        if isinstance(value, acb):
            re, im = value.real, value.imag
            if (re.contains(0) and not re.is_zero()) or (im.contains(0) and not im.is_zero()):
                re = arb(0, re.abs_upper()) if re.contains(0) and not re.is_zero() else re
                im = arb(0, im.abs_upper()) if im.contains(0) and not im.is_zero() else im
                value = acb(re, im)
        elif value.contains(0) and not value.is_zero():
            value = arb(0, value.abs_upper())
        out.append(value)
    return out


def _H_coefficient_balls(h_typed, h_ext_typed, b, q_order: int):
    """:func:`_H_coefficient_balls_pinned` with the arithmetic mode classified once, here (:func:`_pin_mode`)."""
    with _pin_mode(b, h_ext_typed):
        return _H_coefficient_balls_pinned(h_typed, h_ext_typed, b, q_order)


def _H_coefficient_balls_pinned(h_typed, h_ext_typed, b, q_order: int):
    """``(balls, real)``: the coefficients ``H_k(h)`` as flint balls, each certified to the working precision against
    the series' scale (:func:`_power_series_deficit`; a certified ball holding 0 comes back centered on 0,
    :func:`_zero_centred`).

    The certified table (:func:`_certified_h_table`) still meets a canceling top-level sum ``sum_j K_j/(h - d_j)
    H(v_j)`` -- next to a singular c the colliding poles' rows are huge and cancel (c = 0.7 as a binary number, dps
    80: H_24 came out 3.4 against the true 3.37 after the table fix, -8e51 before it) -- so the sum's own balls are
    measured and the whole evaluation is repeated at the working precision plus the deficit until each coefficient
    is certified; :class:`RecursionPrecisionError` when more bits stop helping. Exact active Kac poles raise
    ``ZeroDivisionError``; None when no flint context applies (a LaurentSeries b).
    """

    work = mp.mp.prec
    input_bits = _lazy(lambda: _kac_input_bits(b, (h_typed, *h_ext_typed)))
    extra, previous, real, where = 0, None, None, None
    while True:
        with mp.workprec(work + extra):
            pair_table, _H_table, arb_ctx = _build_H_context(q_order, b, h_ext_typed)
            result = _eval_H_balls(h_typed, arb_ctx, q_order)
        if result is None:
            if arb_ctx is None:
                return None
            # A denominator ball containing zero does not prove that the
            # source input is on a pole. Resolve a finite offset before the
            # mpmath fallback can divide by a rounded pole location.
            K = arb_ctx.get('K_full', arb_ctx.get('K', arb_ctx.get('K_acb')))
            origin = _b_origin(b) if pair_table.kac_from_c else None
            matches = _poles._kac_weight_checker(b, c=origin)
            for index, pair in enumerate(pair_table.pairs):
                if K[index].contains(0):
                    continue
                if matches(h_typed, pair['r'], pair['s']):
                    raise ZeroDivisionError(f"internal weight h = {mp.nstr(h_typed, 12)} sits on the "
                                            f"Kac pole d_{{{pair['r']},{pair['s']}}}")
            deficit = work
        else:
            if real is None:
                real = result[1]
            elif real and not result[1]:
                # Preserve the initial field if an escalated strip context
                # reads as complex; the imaginary bounds must contain zero.
                if not all(v.imag.contains(0) for v in result[0]):
                    raise RecursionPrecisionError("H_q_series: the escalated evaluation of a real series is not real")
                result = ([v.real for v in result[0]], True)
            deficit, where = _power_series_deficit(result[0], work, input_bits)
        if deficit == 0:
            return _zero_centred(result[0]), result[1]
        step = _poles.next_certification_prec(work + extra, deficit, previous, work)
        if step is None:
            which = "a coefficient" if result is None or where is None else f"the coefficient H_{where}"
            raise RecursionPrecisionError(
                f"H_q_series cannot be certified to {mp.mp.dps} digits at h = {mp.nstr(h_typed, 12)}: {which} "
                f"stays {deficit} bits short of the series' scale with {extra} guard bits (q_order {q_order})"
            )
        extra, previous = step[0] - work, step[1]


def _theta3(q):
    return mp.jtheta(3, 0, q)


def block_prefactor(h, z, h_ext: Sequence, *, b=None, c=None, chiral_norm=None, z_shift=None):
    """Return F(h, z) = block_prefactor(h, z) * H(q(z)).

    The block z-exponent is controlled by `z_shift` (default 0):
        F(h, z) ~ z^{h + z_shift} / N^h  as z -> 0.
    The h-independent amplitude factor has z^{sigma + z_shift}, and
    the h-dependent exponential factor is (16q/N)^h.

    For the physical crossing equation, pass z_shift = -h1 - h2 to include
    the kinematic factor z^{h - h1 - h2} in the block.
    Default N = `chiral_norm` = 16.
    """
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    from .numeric_input import parse_number

    h, z = parse_number(h), parse_number(z)
    h_ext = tuple(parse_number(value) for value in h_ext)
    c = parse_number(c)
    chiral_norm, z_shift = parse_number(chiral_norm), parse_number(z_shift)
    from .conditioning import endpoint_loss, report

    report(endpoint_loss(z), what="block_prefactor")
    return _block_prefactor_at_work(
        h, z, h_ext, b=b, c=c, chiral_norm=chiral_norm, z_shift=z_shift,
    )


def _block_prefactor_at_work(h, z, h_ext, *, b, c, chiral_norm, z_shift):
    """Prefactor arithmetic after resolving the original physical point."""
    if b is None:
        b = b_from_c(c)
    else:
        from .params import _c_input_of_b

        c = _c_input_of_b(b)
    N = _resolve_chiral_norm(chiral_norm)
    # Keep real scalars in mp.mpf so the prefactor stays in real arithmetic
    # for the standard case (real c, real externals, real h, real z in (0, 1));
    # any complex input promotes the whole expression to mp.mpc as before.
    # (mp.convert also takes a Fraction, the exact-input convention of precision.convert_exact)
    c = _maybe_real_scalar(mp.convert(c))
    z = _maybe_real_scalar(mp.convert(z))
    h = _maybe_real_scalar(mp.convert(h))
    h1, h2, h3, h4 = _numeric_input_values(h_ext)
    sigma = (c - 1) / 24
    q = q_from_z(z)
    z_exp = sigma + (mp.mpf(0) if z_shift is None else _maybe_real_scalar(mp.convert(z_shift)))
    # The h-dependent exponential factor is (16 q / N)^h: fold the N^{-h} division
    # into the (16 q)^h base so the prefactor carries a single h-power instead of
    # (16 q)^{h-sigma} and a separate N^h (which double-counts as a redundant
    # transcendental). (16 q)^{h-sigma}/N^h = (16 q)^{-sigma} * (16 q / N)^h.
    theta_exp = (c - 1) / 2 - 4 * (h1 + h2 + h3 + h4)
    # each power x^e carries the relative rounding of e log x: (16 q / N)^h at h = 300 lost 1.2 of 60 digits, so the
    # powers are formed with that many guard digits and the product rounded to the caller's precision
    with mp.workdps(10):
        lost = sum(abs(mp.mpc(e)) * abs(mp.log(mp.mpc(x))) for e, x in (
            (sigma, 16 * q), (h, 16 * q / N), (z_exp, z), (sigma - h2 - h3, 1 - z), (theta_exp, _theta3(q))))
    guard = int(mp.ceil(mp.log10(1 + lost)))
    with mp.workdps(mp.mp.dps + guard):
        if guard:
            q = q_from_z(z)
        q16 = 16 * q
        value = (
            q16 ** (-sigma)
            * (q16 / N) ** h
            * z ** z_exp
            * (1 - z) ** (sigma - h2 - h3)
            * _theta3(q) ** theta_exp
        )
    return +value


def F_block(
    h, z, h_ext: Sequence, *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, regularize: str = "auto", kac_limit: str | None = None,
):
    """Sphere 4-point Virasoro block F(h, z; h_i; c), numerically.

    Uses the Zamolodchikov h-recursion in the q-expansion form, truncated at
    q^{q_order}. Pass exactly one of `b` or `c`. `chiral_norm` selects the
    N in the rescaling (default 16) and `z_shift` the z-power: the value is

        F(h, z) = N^{-h} z^{h1 + h2 + z_shift} F_std(h, z),   F_std = z^{h-h1-h2} (1 + O(z)),

    so the default z_shift = 0 behaves as z^h / N^h at small z, and z_shift = -h1-h2 gives
    N^{-h} times the standard block F_std (16^{-h} F_std at the default N).

    `regularize` controls the bare-vs-regularized dispatch at recursion-
    singular `c` (`c = 1`, `c = 25`, all minimal models, and `c >= 25` dual-
    family loci with rational b² = p/q). Without this guard the bare
    recursion silently returns ~10^88 numerical garbage from near-zero
    `1/(h - d_{r,s})` divisions inside `H_q_series`.

    - `"auto"` (default): if `c` matches a recognized singular locus
      (`virasoro_blocks.singularity.recognized_singular_locus(c) is not
      None`), route to `F_block_regularized`. Coverage spans c=1, c=25,
      every minimal model M(p, q) including Ising c=1/2, the b² = -1/q
      loci c = -2, -7, -25/2, ..., and dual-family loci c=50, c=33, c=28,
      c=26, c=38.5, ... (the c=13 + 6·(p/q + q/p) family for
      q <= max(16, q_order)). At any other c the bare path runs unchanged.
    - `"on"`: always route to `F_block_regularized`. Use this if you
      suspect a singular c outside the recognized list.
    - `"off"`: use the bare path, refusing recognized singular charges with
      a ValueError rather than running their singular recursion.

    `kac_limit` selects the block ON a Kac weight of a recognized singular c
    (regularized path only): `"irreducible"` (the default, `kac_limit_mode`)
    the block of the irreducible module L(h)
    (`recursion.H_q_series_kac_irreducible`); `"c_limit"` the c -> c0 limit
    at fixed h. They differ where Kac poles of one level merge -- every Kac
    weight at c = 1 and c = 25.

    The singular-locus check uses `c` directly when provided, so passing
    `c` (rather than `b`) gives an *exact* match against the dual-family
    loci with no tolerance involved. A `b` obtained from `b_from_c` keeps
    its source charge and is rebuilt at the evaluation precision. An
    independently supplied `b` retains its numerical value; its charge
    is derived with `c_from_b`. The callers in `composer.py`
    (`B_block`, `G_block`, `parity_*_B_block`) thread the user-provided
    `c` through `_resolve_bc` so the chain calls hit the exact path.
    """
    from .numeric_input import parse_number

    q_order = _integer_control('q_order', q_order)
    h = parse_number(h)
    z = parse_number(z)
    from .conditioning import endpoint_loss, report

    report(endpoint_loss(z), what="F_block")
    h_ext = [parse_number(value) for value in h_ext]
    if regularize not in ("auto", "on", "off"):
        raise ValueError("regularize must be 'auto', 'on', or 'off'")
    kac_limit = kac_limit_mode(kac_limit)
    if b is None and c is None:
        raise ValueError("specify exactly one of b, c (or both, with composer threading)")
    if b is not None and c is None:
        from .params import c_origin_of_b

        origin = c_origin_of_b(b)
        if origin is not None:
            b, c = None, origin
    if c is not None:
        from .singularity import snap_float_c

        # a Fraction (precision.convert_exact's convention) has no mpc(); a float 0.7 means 7/10
        c = snap_float_c(c)
    if regularize != "off":
        from .singularity import recognized_singular_locus
        # Prefer user/caller-provided c for the auto-check (exact match).
        # Only derive c from b as a fallback (incurs roundoff at the dual-
        # family loci; the tolerance in positive_rational_b2_pq covers it).
        c_check = c if c is not None else c_from_b(b)
        if regularize == "on" or recognized_singular_locus(c_check, q_order=q_order) is not None:
            from .regularization import F_block_regularized
            return F_block_regularized(
                h, z, h_ext, b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
                z_shift=z_shift, kac_limit=kac_limit,
            )
    if regularize == "off":
        # the bare recursion is singular at a recognized locus: refuse it here rather than divide by zero inside
        from .singularity import reject_singular_c
        reject_singular_c(c if c is not None else c_from_b(b), b=b, q_order=q_order, h_externals=h_ext,
                          where="F_block(regularize='off')")
    if b is None:
        b = b_from_c(c)
    # (h and the externals as parsed: an exact label stays exact, an mpmath value is a number at its precision)
    return _bare_F_block(h, z, h_ext, b=b, c=c, q_order=q_order,
                         chiral_norm=chiral_norm, z_shift=z_shift)


def _bare_F_block(h, z, h_ext, *, b, c, q_order, chiral_norm, z_shift):
    """Evaluate an already-dispatched bare block at its resolved input precision."""
    z = mp.convert(z)
    if z == 0 or z == 1:
        raise ValueError(
            f"F_block: z = {z} is an OPE point (the block is evaluated for 0 < |z| < 1, q_from_z maps z = 1 to |q| = 1)"
        )
    h_source = h
    h = mp.convert(h)
    # at a recognized singular c the Kac weights collide and the genuine/removable split is the regularizer's job
    # (as in evaluation._check_virasoro_pole); the bare "off" path keeps its legacy behavior there
    from .singularity import recognized_singular_locus

    kac_hit = _exact_kac_hit(h_source, b, q_order, c=c)
    if kac_hit is not None and recognized_singular_locus(c if c is not None else c_from_b(b)) is None:
        from .polyblock import chiral_denominator_roots
        from .requests import BlockPoleError

        genuine = chiral_denominator_roots(h_ext, b=b, q_order=q_order)
        if any(_near_pole(kac_hit, pole) for pole in genuine):
            raise BlockPoleError(family="virasoro", basis="F_block", exchanged=h, pole=kac_hit)
        # The certified series skips structurally zero residues before
        # dividing by h-d. Evaluate that analytic continuation at h itself:
        # a symmetric displacement can cross other genuine Kac poles when
        # the charge is close to a collision, and need not approximate this
        # removable limit even under doubled precision.

    # an exact label reaches the certified value, which resolves its offset from a pole; a rounded mpmath h (complex
    # decimal text is parsed at the working precision) cannot carry an offset below it
    from .conditioning import _carries_extra_bits

    if (isinstance(h_source, (mp.mpf, mp.mpc)) and not _carries_extra_bits(h_source) and kac_hit is None
            and len(_kac_candidate_indices(h_source, b, q_order))):
        from .conditioning import pole_loss, report
        from .polyblock import chiral_denominator_roots

        report(pole_loss(h_source, chiral_denominator_roots(h_ext, b=b, q_order=q_order)), what="F_block",
               source=h_source)
    # near (not on) a singular locus the colliding poles cancel pairwise: the certified value measures that loss in
    # its own balls (c = 7/10 + 1e-45, 1 + 1e-45, 25 + 1e-45 at q_order 60, dps 60: 61+ digits), so no second guard
    return _certified_block_value(h_source, z, h_ext, b, q_order, chiral_norm, z_shift)


def _certified_block_value(h, z, h_ext, b, q_order, chiral_norm, z_shift):
    """:func:`_certified_block_value_pinned` with the arithmetic mode classified once, here (:func:`_pin_mode`)."""
    b_typed = _maybe_real_scalar(b) if isinstance(b, mp.mpc) else b
    h_ext_typed = _kac_input_values(h_ext)
    with _pin_mode(b_typed, h_ext_typed):
        return _certified_block_value_pinned(h, z, h_ext_typed, b, q_order, chiral_norm, z_shift)


def _certified_block_value_pinned(h, z, h_ext, b, q_order, chiral_norm, z_shift):
    """``block_prefactor * H(q(z))`` with the sum ``H(q) = sum_k H_k q^k`` certified to the working precision.

    The coefficients are certified (:func:`_H_coefficient_balls`); their sum cancels on its own at z near 1 with
    heavy externals, so it is formed in ball arithmetic (q with its rounding radius) and the evaluation repeated with
    the measured deficit as guard bits until it is certified.
    """
    from flint import arb, acb, ctx as flint_ctx

    work = mp.mp.prec
    h_typed, = _kac_input_values((h,))
    h_ext_typed = _kac_input_values(h_ext)
    b_typed = _maybe_real_scalar(b) if isinstance(b, mp.mpc) else b
    extra, previous = 0, None
    while True:
        with mp.workprec(work + extra):
            q = q_from_z(z)
            certified = _H_coefficient_balls(h_typed, h_ext_typed, b_typed, q_order)
            if certified is None:
                # no flint evaluator (h on a Kac pole): the mpmath recursion, as H_q_series returns it
                H_coeffs = H_q_series(h, h_ext, b=b, q_order=q_order)
                value = (block_prefactor(h, z, h_ext, b=b, chiral_norm=chiral_norm, z_shift=z_shift)
                         * _horner_H_q_series(H_coeffs, q))
                break
            balls, real = certified
            flint_ctx.prec = _flint_prec_bits()
            ulp = mp.mpf(2) ** (-mp.mp.prec)
            if real and isinstance(q, mp.mpf):
                q_ball = arb(_mpf_to_arb(q).mid(), _mpf_to_arb(abs(q) * ulp))
                total = arb(0)
            else:
                q_c = mp.mpc(q)
                rad = _mpf_to_arb(abs(q_c) * ulp)
                q_ball = acb(arb(_mpf_to_arb(q_c.real).mid(), rad), arb(_mpf_to_arb(q_c.imag).mid(), rad))
                balls = [acb(value) for value in balls]
                total = acb(0)
            for coefficient in reversed(balls):
                total = total * q_ball + coefficient
            deficit = _relative_deficit([total], work)
            if deficit == 0:
                H_value = _arb_to_mpf(total) if isinstance(total, arb) else _acb_to_mpc(total)
                value = block_prefactor(h, z, h_ext, b=b, chiral_norm=chiral_norm, z_shift=z_shift) * H_value
                break
        step = _poles.next_certification_prec(work + extra, deficit, previous, work)
        if step is None:
            raise RecursionPrecisionError(
                f"F_block cannot be certified to {mp.mp.dps} digits at h = {mp.nstr(h, 12)}, z = {mp.nstr(z, 12)}: "
                f"H(q) stays {deficit} bits short with {extra} guard bits (q_order {q_order})"
            )
        extra, previous = step[0] - work, step[1]
    flint_ctx.prec = _flint_prec_bits()
    return +value


def _kac_tolerance(pole):
    """Roundoff at the working precision around a Kac location ``pole``: ``10^-(dps - 10)`` relative (unit floor),
    the tolerance every Kac-location recognition shares (:func:`_near_pole`, :func:`merged_kac_hit`)."""
    return mp.mpf(10) ** (-(mp.mp.dps - 10)) * max(1, abs(pole))


def _near_pole(value, pole):
    """Whether ``value`` is the Kac location ``pole``: exactly when both are exact rationals, else to roundoff at the
    working precision (:func:`_kac_tolerance`; an mpmath weight is a number at its precision, not a label)."""
    if isinstance(value, (int, Fraction)) and isinstance(pole, (int, Fraction)):
        return value == pole
    pole = mp.mpc(pole)
    return abs(mp.mpc(value) - pole) <= _kac_tolerance(pole)


_KAC_WEIGHT_CACHE: dict = {}


def _kac_weights(b, q_order):
    """``(exact(i), complex128 array)`` of the Kac weights ``d_{r,s}``, ``r s <= q_order``, cached per
    (b, q_order, precision): F_block's pole check must not cost a Kac-table build per call, and the exact
    (multiprecision) weight is formed only for a candidate the float64 scan leaves."""
    import numpy as np

    b = mp.convert(b)
    key = (b._mpc_ if isinstance(b, mp.mpc) else b._mpf_, int(q_order), mp.mp.prec)
    hit = _KAC_WEIGHT_CACHE.get(key)
    if hit is None:
        labels = [(r, s) for r in range(1, int(q_order) + 1) for s in range(1, int(q_order) // r + 1)]
        r_arr = np.array([r for r, _s in labels], dtype=float)
        s_arr = np.array([s for _r, s in labels], dtype=float)
        b64 = complex(b)
        Q64 = b64 + 1 / b64
        approx = (Q64 * Q64 - (r_arr * b64 + s_arr / b64) ** 2) / 4

        def exact(i, _b=b, _labels=labels):
            # only for the few candidates the float64 scan leaves (h near a pole)
            r, s = _labels[i]
            b_work = +_b          # at the working precision (exact identities such as d_{1,1} = 0 survive)
            Q = b_work + 1 / b_work
            return (Q * Q - (r * b_work + s / b_work) ** 2) / 4

        hit = (exact, approx)
        if len(_KAC_WEIGHT_CACHE) > 256:
            _KAC_WEIGHT_CACHE.clear()
        _KAC_WEIGHT_CACHE[key] = hit
    return hit


def _kac_candidate_indices(h, b, q_order):
    """Coarse candidates for the guarded, full-precision Kac comparison."""
    import numpy as np

    _exact, approx = _kac_weights(b, q_order)
    if not len(approx):
        return ()
    h64 = complex(h)
    if not np.isfinite(h64):
        return ()
    return np.nonzero(np.abs(approx - h64) <= 1e-8 * max(abs(h64), 1.0))[0]


def _exact_kac_hit(h, b, q_order, *, c=None):
    """The Kac weight ``d_{r,s}`` (``r s <= q_order``, at the working precision) that ``h`` sits on, or None.

    An exact rational ``h`` at an exact rational ``c`` is compared exactly with the rational Kac weights (every
    ``d_{r,s}`` when ``b^2`` is rational, else ``d_{r,r} = (1 - r^2)(c - 1)/24``; the others are irrational).
    Otherwise a float64 scan leaves candidates within relative distance 1e-8, and those are compared to roundoff
    at the working precision (:func:`_near_pole`).
    """
    from .numeric_input import exact_real

    exact, _approx = _kac_weights(b, q_order)
    h_exact = exact_real(h) if isinstance(h, (int, Fraction)) else None
    c_exact = exact_real(c) if isinstance(c, (int, Fraction)) else None
    if h_exact is not None and c_exact is not None:
        label = _rational_kac_weights(c_exact, int(q_order)).get(h_exact)
        return None if label is None else exact(label)
    for i in _kac_candidate_indices(h, b, q_order):
        d = exact(int(i))
        if _near_pole(h, d):
            return d
    return None


@lru_cache(maxsize=64)
def _rational_kac_weights(c: Fraction, q_order: int) -> dict:
    """``{d_{r,s}: index into _kac_weights' labels}`` for the rational Kac weights at the exact charge ``c``."""
    from .numeric_input import exact_b_squared

    labels = [(r, s) for r in range(1, q_order + 1) for s in range(1, q_order // r + 1)]
    x = exact_b_squared(c)
    out = {}
    for i, (r, s) in enumerate(labels):
        if x is not None:
            d = ((1 - r * r) * x + Fraction(1 - s * s) / x + 2 - 2 * r * s) / 4
        elif r == s:
            d = Fraction((1 - r * r) * (c - 1), 24)
        else:
            continue
        out.setdefault(d, i)
    return out


def _fixed_h_block_evaluator(
    h,
    h_ext: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Return a callable z -> F_block(h, z, h_ext; c) with cached H coefficients.

    The bare recursion: raises ValueError at a recognized singular c (see
    `singularity.reject_singular_c`); `regularization._fixed_h_block_evaluator_regularized`
    is the singular-c counterpart.
    """
    q_order = _integer_control('q_order', q_order)
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    from .singularity import reject_singular_c, snap_float_c

    if c is not None:
        c = snap_float_c(c)
    reject_singular_c(c, b=b, q_order=q_order, h_externals=h_ext, where="_fixed_h_block_evaluator")
    if b is None:
        b = b_from_c(c)

    H_coeffs = H_q_series(h, h_ext, b=b, q_order=q_order)
    coeffs_real = all(isinstance(coeff, mp.mpf) for coeff in H_coeffs)

    def evaluate(z):
        q = q_from_z(z)
        H_val = _horner_H_q_series(H_coeffs, q, coeffs_real=coeffs_real)
        pref = block_prefactor(h, z, h_ext, b=b, chiral_norm=chiral_norm, z_shift=z_shift)
        return pref * H_val

    return evaluate


def _rounded_rational_label(value):
    """An mpf within roundoff at the working precision (eight units) of a rational (denominator <= 10^6) as that
    rational, else ``value``: a degenerate external rounded to the working precision stays degenerate (the Z3
    parafermion's 1/15 at c = 4/5), while a value that differs from the rational by more (a 53-bit 0.1 at 30
    digits) is the number it is."""
    if not isinstance(value, mp.mpf) or value == 0:
        return value
    from .torus_onept import _recognized_fraction

    rational = _recognized_fraction(value)
    if rational is None:
        return value
    prec = mp.mp.prec
    with mp.workprec(prec + 32):
        close = abs(mp.mpf(rational.numerator) / rational.denominator - value) <= abs(value) * mp.mpf(2) ** (3 - prec)
    return rational if close else value


def H_q_series_laurent_precompute(
    h_ext: Sequence,
    *,
    c0,
    q_order: int,
    work_dps: int | None = None,
    trim: bool = True,
    half: int | None = None,
) -> dict:
    """Precompute the h-independent parts of H_q_series_laurent.

    The table is built on the narrowest window that holds it exactly (at least the current one, or ``half``), and
    records it as ``window_half``: a consumer that reads its series runs on that window (or narrows the state to its
    own with :func:`_narrowed_state`).
    ``work_dps`` is a minimum (default the working precision).

    Returns a state dict for :func:`H_q_series_laurent_eval`. Separating the
    build from the per-h eval lets callers amortise the expensive b-series /
    K-table / H-table construction when H coefficients are needed at many
    different h values for fixed (c0, h_ext, q_order) -- notably the oracle,
    which evaluates at 2 h values per probe Delta across 4 probe Deltas.

    The series variable is ``t`` with ``c = c0 + params.laurent_t_sq_sign(c0) t^p``,
    ``p = params.laurent_t_power(c0)``: at the branch point
    ``c0 = 1`` the table is built on the ``c < 1`` side, where it is real. A caller that combines the table with
    its own ``c - c0`` dependence must use the same sign; the t^0 coefficients and valuations do not depend on it.
    """
    q_order = _integer_control('q_order', q_order)
    if half is not None:
        half = _integer_control('half', half)
    if len(h_ext) != 4:
        raise ValueError("h_ext must be a length-4 sequence (h1, h2, h3, h4)")

    c0, = _kac_input_values((c0,))
    h_ext = tuple(_rounded_rational_label(h) for h in _kac_input_values(h_ext))
    base_dps = mp.mp.dps
    base_prec = mp.mp.prec
    if work_dps is None:
        work_dps = base_dps
    work_dps = _integer_control('work_dps', work_dps, base_dps)

    from mpmath.libmp import dps_to_prec

    work_prec = max(base_prec, dps_to_prec(work_dps))
    with mp.workprec(work_prec):
        return _H_q_series_laurent_precompute_resolved(
            h_ext, c0=c0, q_order=q_order, work_dps=work_dps, work_prec=work_prec, trim=trim, half=half,
        )


def _H_q_series_laurent_precompute_resolved(h_ext, *, c0, q_order, work_dps, work_prec, trim, half):
    from . import tseries as _ts

    cache_key = _H_laurent_precompute_cache_key(h_ext, c0, q_order, work_dps)
    if cache_key is not None:
        cache_key = cache_key + (bool(trim),)
        cached = _H_LAURENT_PRECOMP_CACHE.get(cache_key)
        if cached is not None and cached["window_half"] >= max(_ts.window_half(), half or 0):
            _H_LAURENT_PRECOMP_CACHE.move_to_end(cache_key)
            return cached

    from .singularity import minimal_model_pq

    b_sq_pq = minimal_model_pq(c0)
    phase_timing = bool(os.environ.get("VB_LAURENT_PRECOMPUTE_TIMING"))
    phase_times = {}
    phase_t0 = time.perf_counter()

    def _mark_phase(name):
        nonlocal phase_t0
        if not phase_timing:
            return
        now = time.perf_counter()
        phase_times[name] = phase_times.get(name, 0.0) + (now - phase_t0)
        phase_t0 = now

    # At c0 <= 1, b = i beta with beta real: carrying beta keeps every series real, so the whole build runs in
    # arb (3-4x cheaper than acb) with the batch contractions on their real path. m b + n/b = i (m beta - n/beta),
    # so A_rs takes the sign of params.A_rs_imag_b.
    imag_b = imaginary_b_locus(c0)
    # real Kac data -- b(c0) real (c0 >= 25) or purely imaginary (c0 <= 1, carried as beta) -- and real externals:
    # every series is real, so the build runs in arb.  Decided here from the data, not by the caller, because the
    # one cached table serves every caller (the scorer, the identity vector, the merged sampler).
    real_b = (not imag_b) and _real_scalar(c0) and mp.re(mp.convert(c0)) >= 25
    real_build = (imag_b or real_b) and all(_real_scalar(h) for h in h_ext)
    saved_real = set_real_mode(True) if real_build else None
    try:
        with mp.workprec(work_prec):
            levels = _laurent_levels(h_ext, c0, q_order, trim=trim, half=half)
        _mark_phase("levels")
        pairs = [dict(p, K=p["K"].s, d=p["d"].s, v=p["v"].s) for p in levels.poles]
        H_table = levels.table
    finally:
        if saved_real is not None:
            set_real_mode(saved_real)

    if phase_timing:
        parts = " ".join(f"{k}={v:.3f}s" for k, v in phase_times.items())
        print(
            f"H_q_series_laurent_precompute q={q_order} pairs={len(pairs)} {parts}",
            file=sys.stderr,
        )

    state = {
        "c0": c0,
        "h_ext": h_ext,
        "trim": trim,
        "q_order": q_order,
        "work_dps": work_dps,
        "work_prec": work_prec,
        "b_sq_pq": b_sq_pq,
        "pairs": pairs,
        "H_table": H_table,
        # the table's own window (the engine widens past the caller's by measured deficits only)
        "window_half": levels.half,
        "levels": levels,
        "views": {},
    }
    if cache_key is not None:
        _H_LAURENT_PRECOMP_CACHE[cache_key] = state
        _H_LAURENT_PRECOMP_CACHE.move_to_end(cache_key)
        if len(_H_LAURENT_PRECOMP_CACHE) > _H_LAURENT_PRECOMP_CACHE_MAXSIZE:
            _H_LAURENT_PRECOMP_CACHE.popitem(last=False)
    return state


def _narrowed_state(state: dict) -> dict:
    """``state`` (a cached precompute) on the current Laurent window: itself at the window it was built at, else its
    series narrowed to the current one (:meth:`~virasoro_blocks.tseries.LaurentSeries.narrowed`, exact), memoized
    per window.  The measuring scorer pass, the tight scorer build and the identity vector of one singular-c block
    all read the same table (it was built three times, once at twice the working precision)."""
    from . import tseries as _ts

    half = _ts.window_half()
    built = state["window_half"]
    if half == built:
        return dict(state)
    view = state["views"].get(half)
    if view is None:
        with mp.workdps(state["work_dps"]):
            pairs = []
            for p in state["pairs"]:
                q = dict(p)
                for name in ("K", "d", "v"):
                    q[name] = p[name].narrowed(built)
                pairs.append(q)
            H_table = [None if row is None else [x.narrowed(built) for x in row] for row in state["H_table"]]
        view = dict(state, pairs=pairs, H_table=H_table, window_half=half, views={})
        state["views"][half] = view
    return dict(view)


def _real_scalar(value) -> bool:
    try:
        return mp.im(mp.convert(value)) == 0
    except (TypeError, ValueError):
        return False


_LAURENT_B_CACHE: "OrderedDict" = None


def _laurent_b_cache(c0, work_dps: int) -> dict:
    """The externals-independent series of a Laurent table build at ``c0`` -- the A_rs products and the reciprocal
    Kac denominators ``1 / (v_outer - d_inner)`` -- shared by the builds of every externals set (and q_order) at one
    precision, window and arithmetic mode: they were ~45% of a build, rebuilt per externals set and per guard
    precision."""
    from collections import OrderedDict

    from . import tseries as _ts

    global _LAURENT_B_CACHE
    if _LAURENT_B_CACHE is None:
        _LAURENT_B_CACHE = OrderedDict()
    key = (mp.mp.prec, int(work_dps), _scalar_cache_key(c0), _ts.window_half(), _ts._real_path())
    entry = _LAURENT_B_CACHE.get(key)
    if entry is None:
        entry = {"A": {}, "factor_inv": {}, "inv_den": {}}
        _LAURENT_B_CACHE[key] = entry
        # (one entry per precision and window: a word build alternates a few q-orders, each at its working and its
        # certified precision -- 4 entries thrashed, rebuilding them per table; an entry is ~9 MB at q_order 40)
        if len(_LAURENT_B_CACHE) > 16:
            _LAURENT_B_CACHE.popitem(last=False)
    else:
        _LAURENT_B_CACHE.move_to_end(key)
    return entry


def _tracked_b_squared(c0, exact=None):
    """``b^2`` and ``b`` (at an imaginary-b locus ``beta = -i b``) as Tracked series in the local variable ``t``
    (``c = c0 + sigma t^p``, :func:`params.laurent_t_power` / :func:`params.laurent_t_sq_sign`), on the branch of
    ``b_from_c(c0)``: :func:`params.b_series` / :func:`params.beta_series` with their ceilings.  With the locus's
    rational ``exact`` b^2, ``(c0 - 1) / 6 = 2 + b^2 + 1/b^2`` enters as a ball containing it."""
    from .laurent_levels import Tracked
    from .params import laurent_t_power, laurent_t_sq_sign

    c0m = mp.convert(c0)
    imag_b = imaginary_b_locus(c0)
    sign = laurent_t_sq_sign(c0m) if imag_b else 1
    # (the t^p coefficient only sets the scale of t: its rounding leaves every t^0 coefficient unchanged)
    u0 = Tracked.const(2 + exact + 1 / exact) if exact is not None else Tracked.const((mp.mpc(c0m) - 1) / 6)
    u = u0 + Tracked.monomial(laurent_t_power(c0m), mp.mpf(sign) / 6)
    w = u - 2
    disc = (w * w - 4).sqrt()
    b0 = b_from_c(mp.mpc(c0m))
    target = b0 * b0
    b_sq = min([(w + disc) / 2, (w - disc) / 2], key=lambda x: abs(mp.mpc(x.s.coeff(0)) - target))
    if imag_b:
        beta = (-b_sq).sqrt()
        if mp.re(mp.mpc(beta.s.coeff(0))) < 0:
            beta = -beta
        return b_sq, beta, True
    b = b_sq.sqrt()
    if abs(mp.mpc(b.s.coeff(0)) - b0) > abs(mp.mpc(b.s.coeff(0)) + b0):
        b = -b
    return b_sq, b, False


class _KacLaurentParts:
    """The Kac data of the singular ``c0`` as Tracked series, shared by every residue built on them (the sphere's
    ``16^rs A_rs P_12 P_43``, the torus one-point ``A_rs P[d; d_rs] P[d; d_rs + rs]``): ``b^2``, ``Q^2``, ``d_rs``
    with its exact ``t^0`` constant (the locus's rational ``b^2``), ``A_rs`` and the fusion polynomial.  ``shared``
    caches the ``A_rs`` across externals."""

    def __init__(self, c0, q_order: int, shared):
        from .singularity import exact_b_squared

        self.x = x = exact_b_squared(c0, q_order=q_order)
        b_sq, lin, imag_b = _tracked_b_squared(c0, x)
        if imag_b:
            self._lin_m, self._lin_n, self._A_sign = lin, -lin.reciprocal(), -1
        else:
            self._lin_m, self._lin_n, self._A_sign = lin, lin.reciprocal(), 1
        self.b_sq = b_sq
        self.inv_b_sq = b_sq.reciprocal()
        self.Q_sq = b_sq + 2 + self.inv_b_sq
        self._A_cache = shared.setdefault("A_tracked", {})
        self._inv_factor = shared.setdefault("factor_inv_tracked", {})
        # side key -> {(r, s): P_{r, s}, (p, q, "factor"): its factor}, least recently used first
        self._fusion = shared.setdefault("fusion_tracked", OrderedDict())

    def lam_sq(self, h):
        """``Q^2 - 4 h`` (an exact rational -- a degenerate weight -- enters as a ball containing it)."""
        from .laurent_levels import Tracked

        return self.Q_sq - 4 * Tracked.const(h if isinstance(h, Fraction) else mp.convert(h))

    def d(self, r, s_):
        # Combine the coefficients before ball arithmetic. In particular
        # d_{1,1} is identically zero, with no cancellation radius to swamp
        # a small but nonzero exchanged weight.
        return ((1 - r * r) * self.b_sq + 2 * (1 - r * s_) + (1 - s_ * s_) * self.inv_b_sq) / 4

    def d_const(self, r, s_):
        x = self.x
        return None if x is None else ((1 - r * r) * x + (1 - s_ * s_) / x + 2 * (1 - r * s_)) / 4

    def A_rs(self, r, s_):
        from .laurent_levels import Tracked, balanced_product

        hit = self._A_cache.get((r, s_))
        if hit is not None:
            return hit
        x = self.x
        factors = [Tracked.const(mp.mpf(self._A_sign) / 2)]
        for m in range(1 - r, r + 1):
            for n in range(1 - s_, s_ + 1):
                if (m, n) == (0, 0) or (m, n) == (r, s_):
                    continue
                inv = self._inv_factor.get((m, n))
                if inv is None:
                    # m b + n / b vanishes at c0 exactly when m b^2 + n = 0 (imaginary b: m beta - n / beta)
                    zero = x is not None and m * x + n == 0
                    inv = self._inv_factor[(m, n)] = (m * self._lin_m + n * self._lin_n).snapped(zero).reciprocal()
                factors.append(inv)
        self._A_cache[(r, s_)] = out = balanced_product(factors)
        return out

    def fusion(self, r, s_, lam_L, lam_R, key=None):
        """:func:`params.fusion_polynomial_via_squares`, on Tracked series.  ``key`` (hashable, naming the side
        ``(lam_L, lam_R)`` exactly) caches the polynomial and its factors in ``shared``: a factor ``(p, q)`` belongs to
        every ``P_{r, s}`` with ``|p| < r``, ``|q| < s`` of its parity, and a side recurs across the externals sets of
        a word build (the same products of the same factors, so the same balls)."""
        from .laurent_levels import balanced_product

        side = None
        if key is not None:
            side = self._fusion.get(key)
            if side is None:
                side = self._fusion[key] = {}
                if len(self._fusion) > _FUSION_SIDES_MAXSIZE:
                    self._fusion.popitem(last=False)
            else:
                self._fusion.move_to_end(key)
            hit = side.get((r, s_))
            if hit is not None:
                return hit
        factors, seen = [], set()
        for p_ in range(1 - r, r, 2):
            for q_ in range(1 - s_, s_, 2):
                if (p_, q_) in seen:
                    continue
                factor = None if side is None else side.get((p_, q_, "factor"))
                if factor is None:
                    if p_ == 0 and q_ == 0:
                        factor = (lam_L - lam_R) * (mp.mpf(1) / 4)
                    else:
                        shift_sq = (p_ * p_) * self.b_sq + 2 * p_ * q_ + (q_ * q_) * self.inv_b_sq
                        u = lam_L + shift_sq - lam_R
                        factor = (u * u - 4 * lam_L * shift_sq) * (mp.mpf(1) / 16)
                    if side is not None:
                        side[(p_, q_, "factor")] = factor
                factors.append(factor)
                seen.add((p_, q_))
                seen.add((-p_, -q_))
        out = balanced_product(factors)
        if side is not None:
            side[(r, s_)] = out
        return out


#: the fusion-polynomial sides one Kac-data cache entry keeps (~0.5 MB per side at q_order 34)
_FUSION_SIDES_MAXSIZE = 64


def _virasoro_laurent_poles(h_ext, c0, q_order: int, shared, *, mirror: bool = False):
    """The Virasoro Kac poles at the singular ``c0`` as Tracked series: ``K = 16^rs A_rs P_12 P_43``, ``d = d_rs``,
    ``v = d + rs``, with the exact ``t^0`` constants of ``d`` and ``v`` (the locus's rational ``b^2``) by which a
    collision is snapped exactly.  ``shared`` caches the externals-independent ``A_rs`` across externals.
    ``mirror`` (a branch point, :func:`_rs_mirror`): ``(r, s)`` with ``r > s`` is ``(s, r)`` at ``-t``."""
    kac = _KacLaurentParts(c0, q_order, shared)
    lam_sq = [kac.lam_sq(h) for h in h_ext]
    from .numeric_input import literal_components

    ext_parts = tuple(literal_components(h) for h in h_ext)
    same_sides = ext_parts[0] == ext_parts[3] and ext_parts[1] == ext_parts[2]
    poles = []
    index = {}
    for r, s_, rs in _enumerate_rs_pairs(q_order):
        index[(r, s_)] = len(poles)
        if mirror and r > s_:
            q = poles[index[(s_, r)]]
            d = q["d"].mirrored()
            poles.append(dict(r=r, s=s_, rs=rs, K=q["K"].mirrored(), d=d, v=q["v"].mirrored(), d_const=q["d_const"],
                              v_const=q["v_const"], sector=0))
            continue
        d_const = kac.d_const(r, s_)
        d = kac.d(r, s_).snapped(d_const == 0)
        P12 = kac.fusion(r, s_, lam_sq[0], lam_sq[1], key=("side", ext_parts[0], ext_parts[1]))
        P43 = P12 if same_sides else kac.fusion(r, s_, lam_sq[3], lam_sq[2], key=("side", ext_parts[3], ext_parts[2]))
        K = kac.A_rs(r, s_) * P12 * P43 * mp.mpf(16) ** rs
        poles.append(dict(r=r, s=s_, rs=rs, K=K, d=d, v=d + rs, d_const=d_const,
                          v_const=None if d_const is None else d_const + rs, sector=0))
    return poles


def _laurent_levels(h_ext, c0, q_order, *, trim=True, half=None):
    """The Virasoro Laurent level table at the singular ``c0`` (:func:`laurent_levels.build_levels`, a single
    sector, ``H = 1 + ...``), on the narrowest window that holds it exactly (at least ``half``, default the current
    one)."""
    from . import tseries as _ts
    from .laurent_levels import build_levels

    cacheable = _scalar_cache_key(c0) is not None

    from .params import laurent_t_power

    # at the branch points c0 = 1, 25 (t^2 = +-(c - c0)) b^2(-t) = 1 / b^2(t): the (s, r) data are the (r, s) data at
    # -t, and so is every table entry -- half the table is a sign flip of the other half
    mirror = laurent_t_power(c0) == 2

    def make_poles():
        return _virasoro_laurent_poles(h_ext, c0, q_order, _laurent_b_cache(c0, mp.mp.dps) if cacheable else {},
                                       mirror=mirror)

    def inv_cache_for(_half):
        return _laurent_b_cache(c0, mp.mp.dps).setdefault("inv_den_tracked", {}) if cacheable else None

    return build_levels(make_poles, int(q_order), seeds=lambda _sector, n: [1] + [0] * n,
                        couples=lambda _sector, _pole: True, label=lambda p: (p["r"], p["s"]),
                        inv_cache_for=inv_cache_for, trim=trim, half=_ts.window_half() if half is None else half,
                        window_key=("virasoro", _scalar_cache_key(c0), int(q_order), bool(trim), mp.mp.prec)
                        if cacheable else None,
                        # (a certification's rebuild at a higher precision starts at the window measured below it)
                        window_seeds=(("virasoro", _scalar_cache_key(c0), int(q_order), bool(trim)),),
                        mirror=_rs_mirror if mirror else None)


def _rs_mirror(poles):
    """The involution ``(r, s) <-> (s, r)`` on a pole list (index of each pole's mirror)."""
    index = {(p["r"], p["s"]): j for j, p in enumerate(poles)}
    return [index[(p["s"], p["r"])] for p in poles]


#: the digits a certified fixed-h or Kac-curve value at a singular ``c0`` needed past the working precision, per
#: ``(c0, q_order, working precision)``: the largest loss measured on any table of that family (the loss is an
#: amplification set by the recursion at ``c0`` and ``q_order``, shared by the externals and weights within a few
#: digits).  A later certification starts there, so the many tables of a word build are built once, at one shared
#: precision (whose externals-independent series are then shared too), instead of once at the working precision and
#: again at each weight's own few-digit shortfall; a shortfall past it still escalates by its measured amount.
_LAURENT_FAMILY_LOSS: dict = {}


def _limb_filled_dps(dps: int) -> int:
    """The largest precision (decimal digits) whose flint working bits (:func:`tseries._flint_prec_bits`) fill the
    same number of 64-bit limbs as ``dps``'s: a certification that must raise its precision to ``dps`` (a measured
    requirement) gets the rest of the limb for free -- Arb's cost is per limb, and a table build's is set by the
    series bookkeeping, not the digits (q 32 at c = 1: 1.96 s at 36 digits, 2.14 s at 51) -- so the tables of one
    family, whose losses differ by a few digits, meet at one precision instead of one build per digit."""
    bits = int(dps * 3.33) + 20
    top = -(-bits // 64) * 64
    out = int(dps)
    while int((out + 1) * 3.33) + 20 <= top:
        out += 1
    return out


def _laurent_family_key(c0, q_order: int):
    # (one key for the spellings of one locus: Fraction(1), 1, mpf(1))
    c_val, = _kac_input_values((c0,))
    if isinstance(c_val, mp.mpf) and mp.isint(c_val):
        c_val = Fraction(int(c_val))
    c_key = _scalar_cache_key(c_val)
    return None if c_key is None else (c_key, int(q_order), mp.mp.prec)


def _laurent_family_loss(c0, q_order: int) -> int:
    """The family's measured loss; a family not yet measured starts at the loss measured at the nearest lower
    ``q_order`` (same locus and precision): the amplification grows with the order, so that one is a lower bound
    that spares the first table of each order a build at the bare working precision."""
    key = _laurent_family_key(c0, q_order)
    if key is None:
        return 0
    hit = _LAURENT_FAMILY_LOSS.get(key)
    if hit is not None:
        return hit
    lower = [(k[1], v) for k, v in _LAURENT_FAMILY_LOSS.items() if k[0] == key[0] and k[2] == key[2] and k[1] < key[1]]
    return max(lower)[1] if lower else 0


def _record_laurent_family_loss(c0, q_order: int, extra: int) -> None:
    key = _laurent_family_key(c0, q_order)
    if key is not None and extra > _LAURENT_FAMILY_LOSS.get(key, 0):
        _LAURENT_FAMILY_LOSS[key] = int(extra)


def H_q_series_laurent_eval(h, state: dict, *, finite_part: bool = False, radii: list | None = None) -> list:
    """Evaluate H coefficients for h using precomputed state.

    Takes a state dict from :func:`H_q_series_laurent_precompute` and performs only the cheap h-dependent final step
    (:func:`laurent_levels.eval_t0`, on the table's window): summing K_{r,s}/(h - d_{r,s}) * H_table terms and reading
    off the t^0 coefficient for each k.  An ``h`` on a Kac weight of c0 (a recognized rational equal to an exact
    ``d_{r,s}(c0)``) has its collision snapped exactly: the rounding of ``h - d_{r,s}(c0)`` read as a genuine
    ``t^-1`` term (c = 7/10: h = 3/80, 7/16, 1/10 were refused as poles where the block is finite).  Raises
    :class:`~virasoro_blocks.laurent_levels.WindowDeficit` when ``h`` reads past the table's window.  ``radii``
    collects each coefficient's ball radius; add the returned midpoint's rounding allowance to bound its error
    (an exact rational ``h`` enters as a ball).
    Without ``radii``, the returned values are verified to the caller's working
    precision by their coefficient bounds. Supplying ``radii`` reads the
    current table's values and bounds for an enclosing certification loop.
    """
    from .laurent_levels import eval_t0

    source_state = state
    h, = _kac_input_values((h,))
    # (the rounding of an mpmath 0 is the vacuum, as eval_t0 recognizes it)
    vacuum = h == 0
    if not isinstance(h, Fraction) and _real_scalar(h):
        from .laurent_levels import recognized_rational

        vacuum = recognized_rational(h) == 0
    from mpmath.libmp import dps_to_prec

    work_dps = caller_dps = mp.mp.dps
    work_prec = max(mp.mp.prec, dps_to_prec(work_dps))
    family = _laurent_family_key(source_state["c0"], source_state["q_order"])
    # Reuse the measured table loss across queries, but not from the vacuum.
    share_guard = radii is None and not vacuum and work_prec <= source_state["work_prec"]
    certified = source_state.get("certified_eval_state") if share_guard else None
    if certified is not None and certified["window_half"] >= state["window_half"]:
        state = certified
    if work_prec > state["work_prec"]:
        with mp.workprec(work_prec):
            state = H_q_series_laurent_precompute(
                state["h_ext"], c0=state["c0"], q_order=state["q_order"],
                work_dps=work_dps, trim=state["trim"], half=state["window_half"],
            )

    target, previous = +mp.eps, None
    q_order = state["q_order"]
    b_sq_pq = state["b_sq_pq"]
    c0 = state["c0"]
    pole_confirmed = False
    while True:
        bounds = [] if radii is None else radii
        with mp.workprec(state["work_prec"]):
            values, negatives = eval_t0(state["levels"], h, sector=0,
                                        seed=[1] + [0] * q_order, couples=lambda _sector, _pole: True, radii=bounds)
            if radii is not None:
                break
            if negatives and not finite_part:
                if pole_confirmed:
                    break
                # A surviving negative power is read from the table's valuations at this precision, where the
                # rounding residue of an exact cancellation reads the same way (c0 = 1, h = 0, h_ext = 49/36:
                # "H_42 keeps a t^-8 pole" at 30 digits, none at 40).  Decide by one build at doubled precision,
                # or at the measured shortfall of the t^0 coefficients when that is larger.
                worst = max((radius / max(1, abs(value)) for radius, value in zip(bounds, values)),
                            default=mp.mpf(0))
                loss = (state["work_dps"] if not mp.isfinite(worst) or worst <= 0
                        else int(mp.ceil(mp.log10(worst / target))))
                work_dps = max(2 * state["work_dps"], state["work_dps"] + max(1, loss))
                with mp.workdps(work_dps):
                    state = H_q_series_laurent_precompute(
                        state["h_ext"], c0=c0, q_order=q_order,
                        work_dps=work_dps, trim=state["trim"], half=state["window_half"],
                    )
                pole_confirmed, previous = True, None
                continue
            worst = max((radius / max(1, abs(value)) for radius, value in zip(bounds, values)), default=mp.mpf(0))
            if worst <= target:
                if not pole_confirmed and not vacuum and state["work_dps"] > caller_dps and family is not None:
                    # (a loss measured here, keyed at the caller's precision; the doubled pole confirmation is no
                    # measurement)
                    _LAURENT_FAMILY_LOSS[family] = max(_LAURENT_FAMILY_LOSS.get(family, 0),
                                                       state["work_dps"] - caller_dps)
                break
            if previous is not None and (not mp.isfinite(worst) or worst >= previous):
                raise ArithmeticError(f"H_q_series_laurent at c0 = {c0}: the coefficient error bound does not shrink")
            loss = state["work_dps"] if not mp.isfinite(worst) else int(mp.ceil(mp.log10(worst / target)))
            previous = worst
            # the measured shortfall, or the family's largest measured loss when that is more (one shared rebuild)
            work_dps = max(_limb_filled_dps(state["work_dps"] + max(1, loss)),
                           caller_dps + (_LAURENT_FAMILY_LOSS.get(family, 0) if family is not None else 0))
            with mp.workdps(work_dps):
                state = H_q_series_laurent_precompute(
                    state["h_ext"], c0=c0, q_order=q_order,
                    work_dps=work_dps, trim=state["trim"], half=state["window_half"],
                )
            if share_guard:
                source_state["certified_eval_state"] = state
    if negatives and not finite_part:
        k, valuation = negatives[0]
        detail = "; the channel is genuinely singular there."
        if b_sq_pq is not None:
            p_mm, q_mm = b_sq_pq
            detail = (
                f"; at the minimal model M({p_mm}, {q_mm}) this is a "
                "genuine divergence: a degenerate (null) state resonates "
                "with the exchanged weight and does not decouple, because "
                "the external dimension is not a degenerate field of this "
                "model (the fusion-polynomial residue does not vanish). "
                "The block is finite only as a c -> c0 limit, or for "
                "degenerate externals. This is physics, not a "
                "regularization failure."
            )
        where = f"keeps a t^{valuation} pole" if valuation is not None else "has a pole at every c, here"
        raise ValueError(f"H_q_series_laurent: H_{k} {where} at c0 = {c0}{detail}")
    return [+v for v in values]


_MERGED_CONTEXT_CACHE: "OrderedDict" = None


def _exact_to_mp(x):
    """An mpmath number from x (an exact ``Fraction`` / int from a table builder included)."""
    from fractions import Fraction

    if isinstance(x, Fraction):
        return mp.mpf(x.numerator) / x.denominator
    return mp.convert(x)


def build_merged_H_context(q_order: int, c0, h_ext, *, truncate: bool = True) -> dict:
    """The H-recursion at a recursion-singular ``c0`` as merged poles, for the chiral evaluators.

    At ``c0`` colliding Kac poles merge, so the exact ``H_k(h)`` (the t^0 part of the Laurent-in-t recursion,
    :func:`H_q_series_laurent_precompute`) is

        H_k(h) = sum over merged locations d0 and orders a of  M[k, (d0, a)] / (h - d0)^a .

    With ``d_p = d0 + delta_p`` (``delta_p = O(t)``), ``K_p / (h - d_p) = K_p sum_j delta_p^j / (h - d0)^(j+1)``, a
    finite sum in the truncated window, so ``M[k, (d0, j + 1)] = sum_p [t^0](K_p delta_p^j H_table[p][k - rs_p])``
    over the pairs at ``d0``: one small matrix product per pair, from the one H-table build every other singular-c
    step reads.  A residue with built-in cancellation (``K_p`` vanishing at ``t = 0``, a degenerate external) enters
    through its own valuation, exactly.  A channel that keeps a negative power of ``t`` (genuinely singular, a
    non-degenerate external at a minimal model) raises, as :func:`H_q_series_laurent_eval` does.

    Returns a context in mode ``"merged"``: ``M`` ((q_order + 1) x n_cols, arb or acb), the column locations ``d0``
    and orders ``power``, and ``real``; the evaluators form ``1 / (h - d0)^power`` per column where the generic
    context forms ``K / (h - d)``.
    """
    from collections import OrderedDict
    from flint import acb, acb_mat, arb, arb_mat, ctx as flint_ctx

    from . import tseries as _ts

    q_order = _integer_control('q_order', q_order)
    global _MERGED_CONTEXT_CACHE
    if _MERGED_CONTEXT_CACHE is None:
        _MERGED_CONTEXT_CACHE = OrderedDict()
    c0 = _exact_to_mp(c0)
    # an exact rational external stays exact: the table takes it as a ball containing it, so its radii bound the error
    h_ext = [h if isinstance(h, Fraction) else mp.convert(h) for h in h_ext]
    key = _H_laurent_precompute_cache_key(h_ext, c0, q_order, mp.mp.dps)
    if key is not None and key in _MERGED_CONTEXT_CACHE:
        return _MERGED_CONTEXT_CACHE[key]
    if key is not None and truncate:
        # the recursion through q^q' is the first q' + 1 rows of a deeper one (H_k only reads lower orders), so a
        # cached deeper context serves it truncated -- e.g. the q-order gate's windows q, q - 4, q - 8
        prec_k, dps_k, _q, c_k, ext_k = key
        deeper = [(k[2], v) for k, v in _MERGED_CONTEXT_CACHE.items()
                  if k[0] == prec_k and k[1] == dps_k and k[3] == c_k and k[4] == ext_k and k[2] > q_order]
        if deeper:
            from flint import acb_mat, arb_mat

            _qd, ctx_d = min(deeper, key=lambda t: t[0])
            rows = ctx_d["M"].tolist()[: q_order + 1]
            # (the pole list is the deeper order's: a skeleton builds its own, merged_skeleton_rational)
            ctx = dict(ctx_d, M=(arb_mat if ctx_d["real"] else acb_mat)(rows), poles=None)
            return ctx

    from .laurent_levels import Tracked, WindowDeficit, product_ceiling

    saved = _ts.window_half()
    half = None
    try:
      while True:
        state = H_q_series_laurent_precompute(h_ext, c0=c0, q_order=q_order, half=half)
        # the columns are read off the table on its own window (the engine sized it from what is read)
        _ts.set_window(state["window_half"])
        levels = state["levels"]
        # a negative power that survives at a generic weight is a genuinely singular channel: the same error
        # Only the structural-pole check is needed here. The merged evaluator
        # verifies the coefficient balls itself; certifying this discarded
        # probe would build an extra table whose coefficients are never used.
        probe = mp.mpf("1.41421356237309504880")
        try:
            H_q_series_laurent_eval(probe, state, radii=[])
        except ValueError:
            # a negative power read at this precision may be the rounding residue of an exact cancellation (c0 = 1,
            # h_ext = 16/9, q_order 50: "H_50 keeps a t^-4 pole" at 30 digits): the default evaluation confirms it
            # by one build at doubled precision and raises only for a power that survives
            H_q_series_laurent_eval(probe, state)
        prec = _flint_prec_bits()
        flint_ctx.prec = prec
        t_lo = _ts.T_LO
        b_sq_pq = state["b_sq_pq"]
        with mp.workdps(state["work_dps"]):
            from .singularity import exact_b_squared

            beta = exact_b_squared(c0, q_order=q_order)
            columns = {}            # (location key, order) -> column index
            locations = []          # per column: d0 (mpc), order
            members = {}            # location key -> (d0, [(r, s) of the pairs there])
            exact_value = {}        # location key -> the exact rational Kac limit (or None)
            column_keys = []        # per column: its location key
            entries = []            # (k, column, value) contributions
            real = True
            deficit = 0
            for idx, (p, rows) in enumerate(zip(state["pairs"], state["H_table"])):
                K = p["K"]
                if rows is None or K.is_zero():
                    continue
                if beta is not None:
                    # every locus has a rational b^2: the Kac limit is exactly rational
                    r_, s_ = int(p["r"]), int(p["s"])
                    d_num = mp.mpc(p["d"].coeff(0))
                    tol_d = mp.mpf(10) ** (-(mp.mp.dps // 2)) * max(1, abs(d_num))
                    for rr, ss in ((r_, s_), (s_, r_)):         # (the series may carry b or 1/b: r <-> s)
                        d_exact = ((1 - rr * rr) * beta + 2 * (1 - rr * ss) + (1 - ss * ss) / beta) / 4
                        d0 = mp.mpf(d_exact.numerator) / d_exact.denominator
                        if abs(d_num - d0) <= tol_d:
                            break
                    else:
                        raise ValueError(f"merged H context: the Kac limit of ({r_}, {s_}) at c0 = {c0} is "
                                         f"{mp.nstr(d_num, 12)}, not the rational one")
                    loc_key = ("exact", d_exact)
                else:
                    d0 = mp.mpc(p["d"].coeff(0))
                    loc_key = ("num", mp.nstr(d0, max(10, mp.mp.dps // 2)))
                members.setdefault(loc_key, (d0, []))[1].append((int(p["r"]), int(p["s"])))
                exact_value[loc_key] = loc_key[1] if loc_key[0] == "exact" else None
                # d - d0: at an exact location the t^0 of d IS d0, so an exact zero (subtracting the rounded d0
                # left a 1e-dps t^0 that the expansion in delta dropped, outside every radius)
                from .laurent_levels import zero_t0

                delta = zero_t0(p["d"]) if loc_key[0] == "exact" else p["d"] - LaurentSeries.constant(d0)
                val_h = min((r.valuation() for r in rows if not r.is_zero()), default=None)
                if val_h is None:
                    continue
                # E_j = K delta^j while [t^0](E_j H) can be nonzero: val(E_j) <= -val_h; carried with their
                # ceilings, and checked against what is read (E_j up to t^{-val_h}, the rows down to t^{-i_lo})
                tracked = levels.poles[idx]
                delta_t = Tracked(delta, tracked["d"].a)
                powers, E_t = [], tracked["K"]
                while not E_t.s.is_zero() and E_t.s.valuation() <= -val_h:
                    powers.append(E_t)
                    if delta.is_zero():
                        break
                    E_t = E_t * delta_t
                if not powers:
                    continue
                i_lo = min(e.s.valuation() for e in powers)
                i_hi = -val_h
                for e in powers:
                    if e.a is not None and e.a < i_hi:
                        deficit = max(deficit, i_hi - e.a)
                for a_row in levels.ceilings[idx]:
                    if a_row is not None and a_row < -i_lo:
                        deficit = max(deficit, -i_lo - a_row)
                powers = [e.s for e in powers]
                fl_e = [e._as_flint(prec) for e in powers]
                fl_h = [r._as_flint(prec) for r in rows]
                is_real = all(f.real for f in fl_e) and all(f.real for f in fl_h)
                real = real and is_real
                mat = arb_mat if is_real else acb_mat

                def at_power(f, power, is_real=is_real):
                    cc = f.P.coeffs()
                    i = power - t_lo
                    v = cc[i] if 0 <= i < len(cc) else arb(0)
                    return v if is_real else acb(v)

                E_mat = mat([[at_power(f, i) for i in range(i_lo, i_hi + 1)] for f in fl_e])   # E_j at power i
                H_rev = mat([[at_power(f, -i) for f in fl_h] for i in range(i_lo, i_hi + 1)])  # row k' at power -i
                C = (E_mat * H_rev).tolist()                                    # C[j][k'] = [t^0](E_j H_k')
                for j in range(len(powers)):
                    col = columns.get((loc_key, j + 1))
                    if col is None:
                        col = columns[(loc_key, j + 1)] = len(locations)
                        locations.append((d0, j + 1))
                        column_keys.append(loc_key)
                    for k_rel, value in enumerate(C[j]):
                        k = p["rs"] + k_rel
                        if k <= q_order:
                            entries.append((k, col, value))
            n_cols = len(locations)
            ring = arb if real else acb
            M_rows = [[ring(0)] * n_cols for _ in range(q_order + 1)]
            for k, col, value in entries:
                M_rows[k][col] = M_rows[k][col] + (value if real else acb(value))
            if not real and all(x.imag.is_zero() for row in M_rows for x in row) and all(
                    mp.im(d0) == 0 for d0, _a in locations):
                # complex-backed series of a real block (c0 >= 25 carries b as acb): exactly real, so the arb path
                real = True
                M_rows = [[x.real for x in row] for row in M_rows]
            mat = arb_mat if real else acb_mat
            # the pole order at each location: its highest column with a nonzero entry (structural zeros by
            # magnitude, 10^(-dps/2) of the location's largest entry, as the Laurent valuations decide them)
            zero_tol = mp.mpf(10) ** (-(mp.mp.dps // 2))
            col_scale = [max((abs(_acb_to_mpc(x if not real else acb(x))) for x in col), default=mp.mpf(0))
                         for col in zip(*M_rows)] if M_rows and n_cols else []
            loc_scale = {}
            for key_c, sc in zip(column_keys, col_scale):
                loc_scale[key_c] = max(loc_scale.get(key_c, mp.mpf(0)), sc)
            pole_order = {}
            for key_c, (d0, a), sc in zip(column_keys, locations, col_scale):
                if sc > zero_tol * loc_scale[key_c]:
                    pole_order[key_c] = max(pole_order.get(key_c, 0), a)
            ctx = {
                "mode": "merged",
                "poles": [(members[k][0], order, sorted(members[k][1], key=lambda rs: (rs[0] * rs[1], rs)), exact_value[k])
                          for k, order in pole_order.items()],
                "prec": prec,
                "real": real,
                "M": mat(M_rows),
                # (an exact location as a ball containing it)
                "d0": [_location_ball(d0, key_c, real) for (d0, _a), key_c in zip(locations, column_keys)],
                "power": [a for _d0, a in locations],
                "n_cols": n_cols,
                # per column, the key of its merged location in "poles" (the order of key k is pole_order[k])
                "col_keys": list(column_keys),
                "pole_keys": list(pole_order),
                # every Kac location of the pairs (a merged pole, or a removable one with no column): there the block
                # is the fixed-h t^0 of the same table (eval_merged_H), the c -> c0 limit at fixed h
                "kac": [d0 for d0, _labels in members.values()],
                "kac_exact": [(key[1], value[0]) for key, value in members.items() if key[0] == "exact"],
                "c0": c0,
                "h_ext": list(h_ext),
            }
        if deficit:
            # a column reads past a ceiling: the table on a window wider by exactly that much
            half = state["window_half"] + deficit
            continue
        break
    finally:
        _ts.set_window(saved)
    if key is not None:
        _MERGED_CONTEXT_CACHE[key] = ctx
        if len(_MERGED_CONTEXT_CACHE) > 16:
            _MERGED_CONTEXT_CACHE.popitem(last=False)
    return ctx


def _location_ball(d0, loc_key, real: bool):
    """A merged pole location as a flint ball: the exact rational Kac limit as a ball containing it, else ``d0``."""
    from flint import acb, arb

    if loc_key[0] == "exact":
        x = loc_key[1]
        ball = arb(x.numerator) / x.denominator
        return ball if real else acb(ball)
    return _mpf_to_arb(mp.re(d0)) if real else _mpf_to_acb(d0)


def _weight_ball(h, complex_out: bool):
    """The exchanged weight as a flint ball (an exact rational as a ball containing it)."""
    from flint import acb, arb

    if isinstance(h, Fraction):
        ball = arb(h.numerator) / h.denominator
        return acb(ball) if complex_out else ball
    return _mpf_to_acb(mp.mpc(h)) if complex_out else _mpf_to_arb(mp.re(mp.mpc(h)) if isinstance(h, mp.mpc) else mp.mpf(h))


def merged_skeleton_rational(h_ext, *, c0, q_order: int):
    """The chiral skeleton of H at a recursion-singular ``c0``, exactly: an ``HRational`` whose denominator roots are the
    merged pole locations of :func:`build_merged_H_context`, each repeated by its order (after every cancellation, so
    removable poles and collision surpluses are already gone), labeled by the Kac pairs that merge there.  Numerators
    are not formed (a skeleton reads only the denominator and degrees)."""
    from .polyblock import HRational
    from .polynomial import Polynomial

    from .singularity import merged_singular_dps

    q_order = _integer_control('q_order', q_order)
    global _MERGED_CONTEXT_CACHE
    c0 = _exact_to_mp(c0)
    # Rational labels must match the evaluator's Laurent context and retain
    # exact cancellations; converting them here also duplicated its H-table.
    h_ext = [h if isinstance(h, Fraction) else _exact_to_mp(h) for h in h_ext]
    # the pole structure is read from the one table, at the precision the merged evaluators were entered at; the
    # locations are exact rationals, converted at the precision of the caller
    table_dps = merged_singular_dps() or mp.mp.dps
    with mp.workdps(table_dps):
        ctx = build_merged_H_context(q_order, +c0, [+h for h in h_ext])
        if ctx.get("poles") is None:
            key = _H_laurent_precompute_cache_key([+h for h in h_ext], +c0, q_order, mp.mp.dps)
            if _MERGED_CONTEXT_CACHE is not None and key in _MERGED_CONTEXT_CACHE:
                del _MERGED_CONTEXT_CACHE[key]
            ctx = build_merged_H_context(q_order, +c0, [+h for h in h_ext], truncate=False)
    roots, provenance = [], []
    for d0, order, labels, exact in sorted(ctx["poles"], key=lambda t: (mp.re(t[0]), mp.im(t[0]))):
        if exact is not None:
            d0 = mp.mpf(exact.numerator) / exact.denominator
        d0 = mp.mpf(mp.re(d0)) if mp.im(d0) == 0 else mp.mpc(d0)
        for i in range(order):
            roots.append(d0)
            provenance.append((labels[min(i, len(labels) - 1)],))
    denominator = Polynomial([mp.mpf(1)])
    for r in roots:
        denominator = denominator * Polynomial([-r, mp.mpf(1)])
    return HRational(numerators=[], denominator_roots=roots, denominator=denominator,
                     pairs=[lab[0] for lab in provenance], root_provenance=provenance)


def merged_H_rational(h_ext, *, c0, q_order: int):
    """H at a recursion-singular ``c0`` as an exact rational function of ``h`` (an ``HRational`` with numerators), from
    the merged poles of :func:`build_merged_H_context` at the current precision: the denominator is that of
    :func:`merged_skeleton_rational` (each merged location at its order), and ``N_k = D(h) H_k(h)`` with
    ``H_k(h) = sum M[k, (d0, a)] / (h - d0)^a``, i.e. one product of ``M`` with the deflated denominators
    ``D(h) / (h - d0)^a`` (a column past its location's order is a structural zero and dropped).  The symbolic
    table builders read it in place of :func:`~virasoro_blocks.polyblock.H_q_series_rational`, which divides by
    zero at ``c0``."""
    from flint import acb_mat, acb_poly, arb_mat, arb_poly

    from .polyblock import HRational
    from .polynomial import Polynomial, _acb_to_mpc, _arb_to_mpf

    q_order = _integer_control('q_order', q_order)
    global _MERGED_CONTEXT_CACHE
    c0 = _exact_to_mp(c0)
    # Rational labels must match the evaluator's Laurent context and retain
    # exact cancellations; converting them here also duplicated its H-table.
    h_ext = [h if isinstance(h, Fraction) else _exact_to_mp(h) for h in h_ext]
    ctx = build_merged_H_context(q_order, +c0, [+h for h in h_ext])
    if ctx.get("poles") is None:
        key = _H_laurent_precompute_cache_key([+h for h in h_ext], +c0, q_order, mp.mp.dps)
        if _MERGED_CONTEXT_CACHE is not None and key in _MERGED_CONTEXT_CACHE:
            del _MERGED_CONTEXT_CACHE[key]
        ctx = build_merged_H_context(q_order, +c0, [+h for h in h_ext], truncate=False)
    flint_ctx_prec = ctx["prec"]
    from flint import ctx as flint_ctx

    flint_ctx.prec = flint_ctx_prec
    locs = []                                   # (key, exact-or-numeric d0, order, labels)
    for key_c, (d0, order, labels, exact) in zip(ctx["pole_keys"], ctx["poles"]):
        if exact is not None:
            d0 = mp.mpf(exact.numerator) / exact.denominator
        d0 = mp.mpf(mp.re(d0)) if mp.im(d0) == 0 else mp.mpc(d0)
        locs.append((key_c, d0, order, labels))
    locs.sort(key=lambda t: (mp.re(t[1]), mp.im(t[1])))
    real = ctx["real"] and all(isinstance(d0, mp.mpf) for _k, d0, _o, _l in locs)
    POLY, MAT = (arb_poly, arb_mat) if real else (acb_poly, acb_mat)
    conv = (lambda x: _mpf_to_arb(x)) if real else (lambda x: _mpf_to_acb(mp.mpc(x)))
    lin = [POLY([-conv(d0), conv(mp.mpf(1))]) for _k, d0, _o, _l in locs]
    powers = [lin[i] ** locs[i][2] for i in range(len(locs))]
    # prefix / suffix products: D / (h - d0_i)^order_i without division
    prefix = [POLY([conv(mp.mpf(1))])]
    for f in powers:
        prefix.append(prefix[-1] * f)
    suffix = [POLY([conv(mp.mpf(1))])]
    for f in reversed(powers):
        suffix.append(suffix[-1] * f)
    suffix.reverse()
    D = prefix[-1]
    deg = D.degree()
    index = {key_c: i for i, (key_c, _d0, _o, _l) in enumerate(locs)}
    q_rows = []
    for key_c, a in zip(ctx["col_keys"], ctx["power"]):
        i = index.get(key_c)
        if i is None or a > locs[i][2]:
            q_rows.append([conv(mp.mpf(0))] * (deg + 1))          # (a structural zero: no pole there)
            continue
        poly = prefix[i] * suffix[i + 1] * lin[i] ** (locs[i][2] - a)
        cc = poly.coeffs()
        q_rows.append(list(cc) + [conv(mp.mpf(0))] * (deg + 1 - len(cc)))
    M = ctx["M"] if (real or not ctx["real"]) else acb_mat(ctx["M"])
    if real and not ctx["real"]:
        M = arb_mat([[x.real for x in row] for row in ctx["M"].tolist()])
    N = (M * MAT(q_rows)).tolist() if q_rows else [[conv(mp.mpf(0))] * (deg + 1) for _ in range(q_order + 1)]
    to_mp = _arb_to_mpf if real else _acb_to_mpc
    numerators = [Polynomial([to_mp(x) for x in D.coeffs()])]       # H_0 = 1
    for k in range(1, q_order + 1):
        numerators.append(Polynomial([to_mp(x) for x in N[k]]))
    roots, provenance = [], []
    for _k, d0, order, labels in locs:
        for i in range(order):
            roots.append(d0)
            provenance.append((labels[min(i, len(labels) - 1)],))
    return HRational(numerators=numerators, denominator_roots=roots,
                     denominator=Polynomial([to_mp(x) for x in D.coeffs()]),
                     pairs=[lab[0] for lab in provenance], root_provenance=provenance)


def merged_kac_hit(h, ctx) -> bool:
    """Whether ``h`` sits on a Kac location of the merged context (to the working precision: a degenerate weight
    rounded to the caller's digits)."""
    return _merged_kac_location(h, ctx) is not None


def _merged_kac_location(h, ctx):
    """The Kac location selected by the existing value-path snapping tolerance."""
    kac = ctx.get("kac")
    if not kac:
        return None
    from .numeric_input import exact_real

    exact = exact_real(h)
    if exact is not None:
        return next((d0 for rational, d0 in ctx.get("kac_exact", ()) if exact == rational), None)
    hh = mp.mpc(_exact_to_mp(h))
    for d0 in kac:
        # Zero is an exact binary value. A nonzero native weight is not its
        # rounding, however small its mantissa or magnitude may be.
        if d0 == 0:
            if hh == 0:
                return d0
        elif _near_pole(hh, d0):
            return d0
    return None


def c_limit_shortfall(h, ctx, q_order: int) -> int:
    """The digits by which the c -> c0 limit at the Kac location of ``h`` (:func:`merged_kac_hit`) falls short of
    the working precision, measured by the balls of its t^0 coefficients as :func:`_c_limit_balls` bounds them.

    For a limit that read a surviving negative power of t: 0 means the table is certified at this precision and the
    pole genuine; otherwise the structural zeros were not resolved, and a build at that many more digits decides."""
    from .laurent_levels import WindowDeficit

    location = _merged_kac_location(h, ctx)
    half = None
    while True:
        state = H_q_series_laurent_precompute(ctx["h_ext"], c0=ctx["c0"], q_order=q_order, half=half)
        radii: list = []
        try:
            H = H_q_series_laurent_eval(location, state, finite_part=True, radii=radii)
            break
        except WindowDeficit as exc:
            half = state["window_half"] + exc.deficit
    worst = max(((r + mp.eps / 2 * abs(v)) / max(1, abs(v)) for v, r in zip(H, radii)), default=mp.mpf(0))
    if worst <= mp.eps:
        return 0
    return mp.mp.dps if not mp.isfinite(worst) else int(mp.ceil(mp.log10(worst / mp.eps)))


def _c_limit_balls(h, ctx, q_order: int, complex_out: bool):
    """``[H_0 .. H_q_order]`` on a Kac location: the fixed-h t^0 coefficients of the same Laurent table (the c -> c0
    limit at fixed h), as flint balls, or None where a negative power survives (a genuine pole)."""
    from flint import acb, arb

    from .laurent_levels import WindowDeficit

    half = None
    while True:
        state = H_q_series_laurent_precompute(ctx["h_ext"], c0=ctx["c0"], q_order=q_order, half=half)
        radii: list = []
        try:
            H = H_q_series_laurent_eval(h, state, radii=radii)
            break
        except WindowDeficit as exc:
            half = state["window_half"] + exc.deficit
        except ValueError:
            return None
    # the balls keep the table's radii, plus the rounding of their midpoints to the working precision (half a unit,
    # mp.eps / 2): the error bound of the c -> c0 limit
    radii = [r + mp.eps / 2 * abs(v) for v, r in zip(H, radii)]
    use_acb = complex_out or not ctx["real"] or any(mp.im(mp.mpc(v)) != 0 for v in H)
    if use_acb:
        return [_mpf_to_acb(mp.mpc(v)) + acb(arb(0, _mpf_to_arb(r)), arb(0, _mpf_to_arb(r))) for v, r in zip(H, radii)]
    return [arb(_mpf_to_arb(mp.re(mp.mpc(v))), _mpf_to_arb(r)) for v, r in zip(H, radii)]


#: the values of the ``kac_limit`` option: which block a value path returns ON a Kac weight of a singular c
KAC_LIMITS = ("irreducible", "c_limit")

#: the block every value path returns on a Kac weight of a singular c when no ``kac_limit`` is passed: the block of
#: the irreducible module L(h), the one whose OPE coefficients a compact CFT at c0 has (the free-boson vertex
#: correlator peels into its exact Fock couplings with it)
DEFAULT_KAC_LIMIT = "irreducible"

_KAC_LIMIT = contextvars.ContextVar("vb_kac_limit", default=DEFAULT_KAC_LIMIT)


def kac_limit_mode(kac_limit=None) -> str:
    """``kac_limit`` validated: ``"irreducible"`` (the block of the irreducible module L(h),
    :func:`H_q_series_kac_irreducible`) or ``"c_limit"`` (the c -> c0 limit at fixed h).  ``None`` is the active
    default: :data:`DEFAULT_KAC_LIMIT`, or what an enclosing :func:`kac_limit_context` (the CLI's ``--kac-limit``)
    selects.  The two differ only where Kac poles of one level merge (every Kac weight at c = 1 and c = 25)."""
    if kac_limit is None:
        return _KAC_LIMIT.get()
    if kac_limit not in KAC_LIMITS:
        raise ValueError(f"kac_limit must be one of {KAC_LIMITS}, not {kac_limit!r}")
    return kac_limit


@contextlib.contextmanager
def kac_limit_context(kac_limit):
    """Make ``kac_limit`` the default of every value path inside (the paths that take no ``kac_limit`` argument --
    the chiral series evaluators of the tables, the peels -- read it here).  ``None`` keeps the active one."""
    token = _KAC_LIMIT.set(kac_limit_mode(kac_limit))
    try:
        yield
    finally:
        _KAC_LIMIT.reset(token)


def set_kac_limit(kac_limit) -> None:
    """Set the process default of :func:`kac_limit_mode` (a CLI run, a pool worker's initializer)."""
    _KAC_LIMIT.set(kac_limit_mode(kac_limit))


def kac_limit_lru_cache(maxsize=128):
    """``functools.lru_cache`` keyed on the active :func:`kac_limit_mode` as well as the arguments: for a function
    whose result contains singular-c blocks (c = 1) but takes no ``kac_limit`` argument.  A plain ``lru_cache`` there
    returns the block of whichever mode filled it first.  ``cache_clear`` / ``cache_info`` are the underlying cache's."""
    import functools

    def decorate(function):
        @lru_cache(maxsize=maxsize)
        def cached(_kac_limit, *args, **kwargs):
            return function(*args, **kwargs)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return cached(kac_limit_mode(), *args, **kwargs)

        wrapper.cache_clear = cached.cache_clear
        wrapper.cache_info = cached.cache_info
        return wrapper

    return decorate


def kac_weight_label(h, c0, q_order: int):
    """The exact rational Kac weight ``d_{r,s}(c0)`` (``r s <= q_order``) that ``h`` denotes at the singular ``c0``
    with a rational ``b^2`` -- ``h`` exact, or within the shared roundoff tolerance of one (:func:`_near_pole`) --
    or None."""
    from .singularity import exact_b_squared

    try:
        x = exact_b_squared(c0, q_order=q_order)
    except (TypeError, ValueError):
        return None
    if x is None:
        return None
    exact = _exact_label(h) if isinstance(h, (Fraction, int)) else None
    seen = set()
    for r, s_, _rs in _enumerate_rs_pairs(q_order):
        d = ((1 - r * r) * x + (1 - s_ * s_) / x + 2 * (1 - r * s_)) / 4
        if d in seen:
            continue
        seen.add(d)
        if exact is not None:
            if d == exact:
                return d
        elif _near_pole(h, mp.mpf(d.numerator) / d.denominator):
            return d
    return None


def H_q_series_at_kac(h, h_ext: Sequence, *, c0, q_order: int, finite_part: bool = False, kac_limit=None) -> list:
    """:func:`H_q_series_laurent` (the fixed-h ``t^0`` of the Laurent recursion at the singular ``c0``) with
    ``kac_limit`` honored: on a Kac weight of ``c0`` under ``"irreducible"`` (the default) the block of the
    irreducible module (:func:`H_q_series_kac_irreducible`); elsewhere, and under ``"c_limit"``, the fixed-h limit.
    ``finite_part=True`` (a peel's column for a channel whose coefficient must vanish) falls back to the fixed-h
    finite part where the irreducible block does not exist (a null vector of L(h) these externals do not
    decouple)."""
    from .requests import BlockPoleError

    if kac_limit_mode(kac_limit) == "irreducible":
        h0 = kac_weight_label(h, c0, q_order)
        if h0 is not None:
            try:
                return H_q_series_kac_irreducible(h0, h_ext, c0=c0, q_order=q_order)
            except BlockPoleError:
                if not finite_part:
                    raise
    return H_q_series_laurent(h, h_ext, c0=c0, q_order=q_order, finite_part=finite_part)


def _exact_label(h):
    """The rational an external or exchanged weight denotes (a Fraction / int, or the rounding of a rational), or
    None."""
    from .laurent_levels import recognized_rational

    if isinstance(h, Fraction):
        return h
    if isinstance(h, int):
        return Fraction(h)
    try:
        return recognized_rational(h)
    except (TypeError, ValueError):
        return None


def _same_weight(a, b) -> bool:
    ea, eb = _exact_label(a), _exact_label(b)
    if ea is not None and eb is not None:
        return ea == eb
    from .numeric_input import literal_components

    return literal_components(a) == literal_components(b)


def _kac_side_motion(x: Fraction, r0: int, s0: int, hL, hR):
    """How one side ``(hL, hR)`` (``P12 = P(h1, h2)`` or ``P43 = P(h4, h3)``) of the 3-point couplings decouples the
    ``(r0, s0)`` null vector identically along its Kac curve: ``None`` when it already does (the ``(0, 0)`` factor
    ``lam_L - lam_R`` of equal weights, present for odd ``r0, s0``), ``(p, q, C)`` when the fusion factor ``(p, q)``
    vanishes at ``c0`` (``C = lam_L0 + S0 - lam_R0``, ``C^2 = 4 lam_L0 S0``) and the right weight must move with ``c``
    to keep it zero, ``False`` when no factor vanishes at ``c0`` (the irreducible block does not exist)."""
    if r0 % 2 and s0 % 2 and _same_weight(hL, hR):
        return None
    eL, eR = _exact_label(hL), _exact_label(hR)
    if eL is None or eR is None:
        # a weight that is no recognizable rational: no fusion factor vanishes for a generic one (the module does not
        # appear); one that vanishes to roundoff needs the exact weights to move along the curve
        def num(f):
            return mp.mpf(f.numerator) / f.denominator

        Q0 = num(x + 2 + 1 / x)
        lamL, lamR = (Q0 - 4 * mp.mpc(_exact_to_mp(v)) for v in (hL, hR))
        for p in range(1 - r0, r0, 2):
            for q in range(1 - s0, s0, 2):
                if (p, q) == (0, 0) or p * x + q == 0:
                    continue
                S0 = num(p * p * x + 2 * p * q + q * q / x)
                C = lamL + S0 - lamR
                if abs(C * C - 4 * lamL * S0) <= _kac_tolerance(max(abs(C * C), abs(4 * lamL * S0), 1)):
                    raise ValueError("the irreducible block on a Kac weight needs exact (rational) external weights "
                                     "to move along the Kac curve; pass them as Fractions")
        return False
    Q0 = x + 2 + 1 / x
    lamL, lamR = Q0 - 4 * eL, Q0 - 4 * eR
    for p in range(1 - r0, r0, 2):
        for q in range(1 - s0, s0, 2):
            if (p, q) == (0, 0) or p * x + q == 0:
                continue
            S0 = p * p * x + 2 * p * q + q * q / x
            C = lamL + S0 - lamR
            if C * C == 4 * lamL * S0:
                return (p, q, C)
    return False


def _kac_curve_plan(h0: Fraction, c0, h_ext, q_order: int):
    """The Kac curve through the singular weight ``h0`` of ``c0`` on which the irreducible block is read:
    ``{"rs0": (r0, s0), "motion": {(0, 1): ..., (3, 2): ...}}`` (:func:`_kac_side_motion`), or None when ``h0`` has
    no null vector through ``q_order`` (the Verma block is the irreducible one there).  ``(r0, s0)`` is a pole of the
    lowest level at ``h0`` whose null vector both sides decouple; raises :class:`~virasoro_blocks.requests.
    BlockPoleError` when none does (the module L(h0) does not appear for these externals)."""
    from .requests import BlockPoleError
    from .singularity import exact_b_squared

    x = exact_b_squared(c0, q_order=q_order)
    if x is None:
        raise ValueError(f"the irreducible block needs a singular c with a rational b^2 (c = {c0})")

    def d_const(r, s_):
        return ((1 - r * r) * x + (1 - s_ * s_) / x + 2 * (1 - r * s_)) / 4

    colliding = sorted((rs, r, s_) for r, s_, rs in _enumerate_rs_pairs(q_order) if d_const(r, s_) == h0)
    if not colliding:
        return None
    lowest = colliding[0][0]
    for rs, r, s_ in colliding:
        if rs != lowest:
            break
        m12 = _kac_side_motion(x, r, s_, h_ext[0], h_ext[1])
        m43 = _kac_side_motion(x, r, s_, h_ext[3], h_ext[2])
        if m12 is not False and m43 is not False:
            return {"rs0": (r, s_), "motion": {(0, 1): m12, (3, 2): m43}}
    raise BlockPoleError(
        f"no irreducible block (the channel is genuinely singular, a genuine pole): h = {h0} is a Kac weight of c = "
        f"{mp.nstr(_exact_to_mp(c0), 12)} whose level-{lowest} "
        f"null vector these externals do not decouple (the fusion polynomial does not vanish), so L({h0}) does not "
        "appear in this channel", family="virasoro", exchanged=h0, pole=h0)


def _kac_curve_poles(h_ext, c0, q_order: int, shared, motion):
    """:func:`_virasoro_laurent_poles` with the externals moved along ``c`` so the curve's fusion factors vanish
    identically (:func:`_kac_side_motion`): on a moved side the left momentum is held fixed, ``lam_L = lam_L0``, and
    ``lam_R(t) = lam_L0 + S(t) -+ 2 sqrt|lam_L0| sigma(t)`` with ``S = (p b + q/b)^2``, ``sigma`` its real square root
    (``p beta - q/beta`` at an imaginary-b locus), the sign set by ``lam_R(0) = lam_R0``.  Every pole whose fusion
    polynomial on a moved side contains the factor ``(p, q)`` gets an exactly zero residue."""
    from .laurent_levels import Tracked

    kac = _KacLaurentParts(c0, q_order, shared)
    x = kac.x
    Q0 = x + 2 + 1 / x
    lam = [kac.lam_sq(h) for h in h_ext]
    from .numeric_input import literal_components

    parts = [literal_components(h) for h in h_ext]
    # the fusion caches' side keys: an unmoved side is the fixed-h table's own side
    side_key = {(0, 1): ("side", parts[0], parts[1]), (3, 2): ("side", parts[3], parts[2])}
    zero_pq = []
    for (L, R), mv in motion.items():
        if mv is None:
            continue
        side_key[(L, R)] = ("moved", parts[L], parts[R], mv)
        p, q, C = mv
        lamL0 = Q0 - 4 * _exact_label(h_ext[L])
        sigma = p * kac._lin_m + q * kac._lin_n
        S = p * p * kac.b_sq + 2 * p * q + q * q * kac.inv_b_sq
        sig0 = mp.re(mp.mpc(sigma.s.coeff(0)))
        sign = -2 if (C > 0) == (sig0 > 0) else 2
        lam[L] = Tracked.const(lamL0)
        lam[R] = Tracked.const(lamL0) + S + Tracked.const(abs(lamL0)).sqrt() * sigma * sign
        zero_pq.append({(p, q), (-p, -q)})
    poles = []
    for r, s_, rs in _enumerate_rs_pairs(q_order):
        d_const = kac.d_const(r, s_)
        d = kac.d(r, s_).snapped(d_const == 0)
        factors = {(a, b_) for a in range(1 - r, r, 2) for b_ in range(1 - s_, s_, 2)}
        if any(factors & pq for pq in zero_pq):
            K = Tracked(LaurentSeries.zero())
        else:
            K = kac.A_rs(r, s_) * kac.fusion(r, s_, lam[0], lam[1], key=side_key[(0, 1)]) \
                * kac.fusion(r, s_, lam[3], lam[2], key=side_key[(3, 2)]) * mp.mpf(16) ** rs
        poles.append(dict(r=r, s=s_, rs=rs, K=K, d=d, v=d + rs, d_const=d_const,
                          v_const=None if d_const is None else d_const + rs, sector=0))
    return poles


_KAC_CURVE_LEVELS: "OrderedDict" = None


def _kac_curve_levels(h_ext, c0, q_order: int, plan, *, half=None):
    """The Laurent level table the curve of ``plan`` is read on: the cached table of
    :func:`H_q_series_laurent_precompute` when no external moves, else one built with the moved externals
    (:func:`_kac_curve_poles`, cached)."""
    from collections import OrderedDict

    from . import tseries as _ts
    from .laurent_levels import build_levels

    motion = plan["motion"]
    if all(mv is None for mv in motion.values()):
        return H_q_series_laurent_precompute(h_ext, c0=c0, q_order=q_order, half=half)["levels"]
    global _KAC_CURVE_LEVELS
    if _KAC_CURVE_LEVELS is None:
        _KAC_CURVE_LEVELS = OrderedDict()
    key = (_scalar_cache_key(c0), tuple(_scalar_cache_key(_exact_label(h) if _exact_label(h) is not None else h)
                                        for h in h_ext),
           int(q_order), mp.mp.prec, tuple(sorted((k, v) for k, v in motion.items())), _ts.window_half())
    hit = _KAC_CURVE_LEVELS.get(key)
    if hit is not None and hit.half >= (half or 0):
        _KAC_CURVE_LEVELS.move_to_end(key)
        return hit
    imag_b = imaginary_b_locus(c0)
    real_b = (not imag_b) and _real_scalar(c0) and mp.re(mp.convert(c0)) >= 25
    saved_real = set_real_mode(True) if (imag_b or real_b) and all(_real_scalar(h) for h in h_ext) else None
    cacheable = _scalar_cache_key(c0) is not None
    try:
        # The curve moves only the externals: A_rs, its inverse factors and the Kac denominators 1 / (v_o - d_j)
        # are those of the fixed-h table, shared through the same cache (they were rebuilt for every curve table);
        # the window a curve build measures is remembered as the fixed-h table's is.
        levels = build_levels(
            lambda: _kac_curve_poles(h_ext, c0, q_order, _laurent_b_cache(c0, mp.mp.dps) if cacheable else {},
                                     motion),
            int(q_order), seeds=lambda _sector, n: [1] + [0] * n, couples=lambda _sector, _pole: True,
            label=lambda p: (p["r"], p["s"]),
            inv_cache_for=(lambda _half: _laurent_b_cache(c0, mp.mp.dps).setdefault("inv_den_tracked", {}))
            if cacheable else None,
            trim=True, half=_ts.window_half() if half is None else half,
            window_key=("virasoro-curve", _scalar_cache_key(c0), int(q_order), mp.mp.prec) if cacheable else None,
            # (a curve's first build starts at the window measured for the curve, or the fixed-h table, at any
            # precision: the same recursion with moved externals; the window search spent two extra fills here)
            window_seeds=(("virasoro-curve", _scalar_cache_key(c0), int(q_order)),
                          ("virasoro", _scalar_cache_key(c0), int(q_order), True)))
    finally:
        if saved_real is not None:
            set_real_mode(saved_real)
    _KAC_CURVE_LEVELS[key] = levels
    if len(_KAC_CURVE_LEVELS) > 8:
        _KAC_CURVE_LEVELS.popitem(last=False)
    return levels


def _kac_curve_t0(levels, rs0, h0: Fraction, *, radii: list):
    """``(values, negatives)``: the ``t^0`` coefficients of ``H_k = 1 + sum_j K_j / (d_{r0 s0} - d_j) T[j][k - rs_j]``
    -- the recursion evaluated ON the Kac curve ``h = d_{r0 s0}(c)``, whose own residue vanishes identically -- and
    the ``(k, valuation)`` of any surviving negative power.  ``radii`` collects the balls' radii.  Raises
    :class:`~virasoro_blocks.laurent_levels.WindowDeficit` when a read passes a ceiling."""
    from . import tseries as _ts
    from .laurent_levels import (EXACT, WindowDeficit, _min_ceiling, _t0_radius, levels_real, product_ceiling,
                                 read_valuation)
    from .tseries import t_conv_weighted_sum

    poles, table, order = levels.poles, levels.table, levels.order
    saved = _ts.window_half()
    _ts.set_window(levels.half)
    # (a real table's reads in arb, as its build: outside a build the batch paths default to acb)
    saved_real = set_real_mode(True) if levels_real(levels) else None
    try:
        j0 = next(j for j, p in enumerate(poles) if (p["r"], p["s"]) == tuple(rs0))
        if not poles[j0]["K"].is_zero():
            raise ArithmeticError(f"the Kac curve ({rs0}) keeps its own residue: its null vector is not decoupled")
        delta = poles[j0]["d"]
        top, everywhere = [], None
        for j, p in enumerate(poles):
            if j == j0 or p["K"].is_zero():
                continue
            gap = (delta - p["d"]).snapped(p.get("d_const") == h0)
            if gap.is_zero():
                everywhere = p["rs"] if everywhere is None else min(everywhere, p["rs"])
                continue
            top.append((p["rs"], j, p["K"] * gap.reciprocal()))
        top.sort(key=lambda item: item[0])
        out, negatives = [], []
        for k in range(order + 1):
            ws, hs, a = [], [], EXACT
            for rs, j, w in top:
                if rs > k:
                    break
                vin = levels.vals[j][k - rs]
                if vin is None:
                    continue
                a = _min_ceiling(a, product_ceiling(w.a, w.v, levels.ceilings[j][k - rs], vin))
                ws.append(w.s)
                hs.append(table[j][k - rs])
            # (only t^0 and the negative powers are read; read_valuation settles a pole on the whole sum)
            acc = t_conv_weighted_sum(ws, hs, 0) if ws else LaurentSeries.zero()
            if a is not EXACT and a < 0:
                raise WindowDeficit(-a, f" above (H_{k} on the Kac curve)")
            v = read_valuation(acc, lambda ws=ws, hs=hs: t_conv_weighted_sum(ws, hs))
            if everywhere is not None and k >= everywhere:
                negatives.append((k, None))
            elif v is not None and v < 0:
                negatives.append((k, v))
            out.append(acc.coeff(0) + (1 if k == 0 else 0))
            radii.append(_t0_radius(acc) if ws else mp.mpf(0))
        return out, negatives
    finally:
        _ts.set_window(saved)
        if saved_real is not None:
            set_real_mode(saved_real)


def _kac_irreducible_raw(h0: Fraction, h_ext, c0, q_order: int):
    """``(values, radii, negatives)`` of the irreducible block's H on the Kac location ``h0`` at the working
    precision (no certification), or the fixed-h ``t^0`` (the Verma block) when ``h0`` has no null vector through
    ``q_order``."""
    from .laurent_levels import WindowDeficit

    plan = _kac_curve_plan(h0, c0, h_ext, q_order)
    half = None
    while True:
        radii: list = []
        if plan is None:
            state = H_q_series_laurent_precompute(h_ext, c0=c0, q_order=q_order, half=half)
            try:
                values = H_q_series_laurent_eval(h0, state, finite_part=True, radii=radii)
                return values, radii, []
            except WindowDeficit as exc:
                half = state["window_half"] + exc.deficit
                continue
        levels = _kac_curve_levels(h_ext, c0, q_order, plan, half=half)
        try:
            values, negatives = _kac_curve_t0(levels, plan["rs0"], h0, radii=radii)
            return values, radii, negatives
        except WindowDeficit as exc:
            half = levels.half + exc.deficit


def H_q_series_kac_irreducible(h, h_ext: Sequence, *, c0, q_order: int) -> list:
    """``[H_0 .. H_q_order]`` of the block of the IRREDUCIBLE module ``L(h)`` on a Kac weight ``h`` of the singular
    ``c0`` (c = 1, 25, the minimal models, the rational-b^2 loci), certified to the working precision.

    Construction (exact).  Let ``(r0, s0)`` be a Kac pole of the lowest level at ``h`` (``d_{r0 s0}(c0) = h``).  On the
    curve ``h = d_{r0 s0}(c)`` the Verma module has the singular vector ``chi_{r0 s0}(c)``, and when both 3-point
    functionals annihilate it identically in ``c`` (equal weights on a side and odd ``r0, s0``: the ``(0, 0)`` fusion
    factor; otherwise the externals move with ``c`` so that the fusion factor that vanishes at ``c0`` stays zero,
    :func:`_kac_side_motion`) the recursion's own term ``K_{r0 s0}`` vanishes identically and ``H`` on the curve is
    the block of the quotient ``Q(c) = V / V(d + r0 s0)``.  ``Q(c)`` is an analytic family whose Gram matrix at
    ``c0`` is that of ``V / V_{c0}(h + r0 s0)``: ``L(h)`` itself when the embedding diagram is a chain (c = 1, 25,
    every rational ``b^2 > 0``, the boundary weights of the minimal models), so the quotient block is analytic in
    ``t`` and its ``t^0`` is the irreducible block; at a braid weight of a minimal model one more composition factor
    ``L(w')`` survives, at Jantzen depth one, and the functionals annihilate it at ``c0``, so it contributes
    ``O(t)``.  Every ``t^0`` is checked against the exact Verma Gram pseudo-inverse in
    ``tests/core/test_kac_irreducible.py``.

    It differs from the c -> c0 limit at fixed ``h`` (:func:`H_q_series_laurent`) where two poles of one level merge
    at ``h`` -- every Kac weight at c = 1 and 25 (the pairs ``(r, s), (s, r)``) and same-level collisions at
    ``b^2 = p/q``: at c = 1, ``h = m^2/4``, from level ``2(m + 2)`` on (unless degenerate externals cut the fusion),
    by the ``kappa`` admixtures of
    ``L((m+4)^2/4), L((m+8)^2/4), ...`` (c = 1, 25/16 externals, h = 1: 3.35e-5 at level 8).  At the minimal models
    the two agree.  Raises :class:`~virasoro_blocks.requests.BlockPoleError` when the externals do not decouple the
    null vectors of ``L(h)`` (that module does not appear in the channel)."""
    from .requests import BlockPoleError

    q_order = _integer_control('q_order', q_order)
    h0 = _exact_label(h)
    if h0 is None:
        raise ValueError("the irreducible block needs the Kac weight h exactly (a Fraction)")
    h_ext = [x if isinstance(x, Fraction) else mp.convert(x) for x in h_ext]
    c0 = _exact_to_mp(c0) if not isinstance(c0, Fraction) else c0
    # (started at the family's largest measured loss: the curve tables of one externals set, and the fixed-h tables
    # of the same family, are then built once, at one precision)
    base, extra, confirmed, previous = mp.mp.dps, _laurent_family_loss(c0, q_order), False, None
    while True:
        with mp.workdps(base + extra):
            values, radii, negatives = _kac_irreducible_raw(h0, h_ext, c0, q_order)
            worst = max(((r + mp.eps / 2 * abs(v)) / max(1, abs(v)) for v, r in zip(values, radii)),
                        default=mp.mpf(0))
        if negatives:
            if confirmed:
                k, v = negatives[0]
                raise BlockPoleError(
                    f"no irreducible block (the channel is genuinely singular): H_{k} on the Kac curve of h = {h0} "
                    f"keeps a t^{v} pole at c = "
                    f"{mp.nstr(_exact_to_mp(c0), 12)} (a null vector of L({h0}) does not decouple for these "
                    "externals)", family="virasoro", exchanged=h0, pole=h0)
            # a negative power read at this precision may be the rounding residue of an exact cancellation:
            # decided by one build at doubled precision
            confirmed, extra = True, extra + base
            continue
        if worst <= mp.eps:
            if not confirmed:
                # (a loss measured here; the doubled pole confirmation is no measurement)
                _record_laurent_family_loss(c0, q_order, extra)
            return [+v for v in values]
        loss = base + extra if not mp.isfinite(worst) else int(mp.ceil(mp.log10(worst / mp.eps)))
        if previous is not None and mp.isfinite(worst) and worst >= previous:
            raise ArithmeticError(f"H_q_series_kac_irreducible at c0 = {c0}: the error bound does not shrink")
        previous = worst
        extra = max(_limb_filled_dps(base + extra + max(1, loss)) - base, _laurent_family_loss(c0, q_order))


def _irreducible_balls(h0, ctx, q_order: int, complex_out: bool):
    """:func:`_c_limit_balls` for the irreducible block (:func:`H_q_series_kac_irreducible`) on the Kac location
    ``h0`` (exact), or None where a negative power survives (uncertain at this precision, or no irreducible
    block)."""
    from flint import acb, arb

    values, radii, negatives = _kac_irreducible_raw(h0, ctx["h_ext"], ctx["c0"], q_order)
    if negatives:
        return None
    radii = [r + mp.eps / 2 * abs(v) for v, r in zip(values, radii)]
    use_acb = complex_out or not ctx["real"] or any(mp.im(mp.mpc(v)) != 0 for v in values)
    if use_acb:
        return [_mpf_to_acb(mp.mpc(v)) + acb(arb(0, _mpf_to_arb(r)), arb(0, _mpf_to_arb(r)))
                for v, r in zip(values, radii)]
    return [arb(_mpf_to_arb(mp.re(mp.mpc(v))), _mpf_to_arb(r)) for v, r in zip(values, radii)]


def _exact_kac_location(location, ctx):
    """The exact rational of a Kac location of the merged context (``ctx["kac_exact"]``), or None."""
    for rational, d0 in ctx.get("kac_exact", ()):
        if d0 == location:
            return rational
    return None


def eval_merged_H(h, ctx, q_order: int, *, complex_out: bool = False, kac_limit: str | None = None):
    """``[H_0 .. H_q_order]`` at ``h`` from a :func:`build_merged_H_context` (flint), or None on a genuine pole.

    Off the Kac locations the merged form is the block at c0.  On one -- a weight where Kac poles collide -- the
    limits of the block differ.  ``kac_limit="irreducible"`` (the default, :func:`kac_limit_mode`) returns the block
    of the irreducible Virasoro module (:func:`H_q_series_kac_irreducible`); ``kac_limit="c_limit"`` the c -> c0 limit
    at fixed h, the fixed-h t^0 coefficients of the same table, which differs from it where poles of one level merge
    (every Kac weight at c = 1 and 25: by admixtures of the null submodule's composition factors).  (The limit in h
    of the merged form is the Fock-module block at c = 1: it carries the would-be null descendants, and peeling the
    free-boson vertex correlator with it put the h = 4 primary's coefficient 25% off its exact 1/15000.)"""
    from flint import acb, acb_mat, arb, arb_mat, ctx as flint_ctx

    kac_limit = kac_limit_mode(kac_limit)
    flint_ctx.prec = ctx["prec"]
    kac_location = _merged_kac_location(h, ctx)
    if kac_location is not None:
        if kac_limit == "irreducible":
            h0 = _exact_kac_location(kac_location, ctx)
            if h0 is None:
                raise ValueError("the irreducible block needs an exact rational Kac location")
            out = _irreducible_balls(h0, ctx, q_order, complex_out)
            flint_ctx.prec = ctx["prec"]
            return out
        # Once classified as an exact Kac value, evaluate at that location.
        # Keeping the small original offset disagrees with the classification
        # and can amplify cancellation residue in the fixed-h Laurent limit.
        return _c_limit_balls(kac_location, ctx, q_order, complex_out)
    use_acb = complex_out or not ctx["real"] or (isinstance(h, mp.mpc) and h.imag != 0)
    hh = _weight_ball(h, use_acb)
    if use_acb:
        d0 = [acb(x) for x in ctx["d0"]]
        M = ctx["M"] if not ctx["real"] else acb_mat(ctx["M"])
    else:
        d0 = ctx["d0"]
        M = ctx["M"]
    col = []
    for x, a in zip(d0, ctx["power"]):
        u = 1 / (hh - x)
        if not u.is_finite():
            return None
        col.append([u ** a])
    w = (acb_mat if use_acb else arb_mat)(col) if col else None
    one = acb(1) if use_acb else arb(1)
    if w is None:
        return [one] + [0 * one] * q_order
    out = M * w
    return [one] + [out[k, 0] for k in range(1, q_order + 1)]


def laurent_window_half(q_order: int, c0=None) -> int:
    """The Laurent window half-width a chiral H-series build through ``q^q_order`` needs.

    The deepest negative power is about ``0.3 q_order + 1`` in
    ``t = sqrt(c - c0)``. Products contributing to t^0 also require positive
    guard powers, so the window extends beyond that depth. Away from the
    branch points c0 = 1, 25, the variable is ``t = c - c0``
    (:func:`~virasoro_blocks.params.laurent_t_power`), halving the powers.
    """
    from .params import laurent_t_power

    q_order = _integer_control('q_order', q_order)
    half = q_order * 4 // 10 + 6
    if c0 is not None and laurent_t_power(c0) == 1:
        half = (half + 1) // 2 + 1
    return half


def H_q_series_laurent(
    h,
    h_ext: Sequence,
    *,
    c0,
    q_order: int,
    work_dps: int | None = None,
    finite_part: bool = False,
) -> list:
    """[H_0, ..., H_{q_order}] at a recursion-singular c0, computed exactly.

    The Zamolodchikov h-recursion is singular at c0 = 1, c0 = 25 and the
    minimal-model / rational loci: A_rs has b-poles and the Kac weights d_rs
    resonate. Setting c = c0 + sigma t^p (p = 2 at c0 = 1, 25, else 1;
    sigma = -1 at c0 = 1, see
    params.laurent_t_sq_sign) and carrying every recursion quantity as a
    Laurent series in t cures this -- b(c) is analytic in t, so each K_{r,s}
    is a Laurent series whose negative powers cancel against the other
    recursion terms. The physical block is the t^0 coefficient.

    If a negative power survives the cancellation, the channel is genuinely
    singular at c0 (not a removable recursion artifact) and ValueError is
    raised. Pass `c0` exactly equal to the target central charge.

    `work_dps` (default: the working precision) is the minimum build precision; the evaluation raises it only by
    the loss its balls measure (:func:`H_q_series_laurent_eval`).

    For repeated calls at different h values with fixed (c0, h_ext, q_order),
    prefer :func:`H_q_series_laurent_precompute` + :func:`H_q_series_laurent_eval`
    to avoid recomputing the expensive b-series / H-table build each time.

    ``finite_part=True`` returns the t^0 coefficients even when a negative power survives: not a block, but a
    column with leading coefficient 1 that a fit of a correlator to blocks can carry for a channel whose
    coefficient must vanish.
    """
    from .laurent_levels import WindowDeficit

    half = None
    # a family whose tables measured a loss is certified at the working precision plus that loss: built once there
    # (H_q_series_laurent_eval escalates further only by a larger measured shortfall)
    loss = _laurent_family_loss(c0, q_order)
    if loss:
        work_dps = max(mp.mp.dps if work_dps is None else int(work_dps), mp.mp.dps + loss)
    while True:
        state = H_q_series_laurent_precompute(h_ext, c0=c0, q_order=q_order, work_dps=work_dps, half=half)
        try:
            return H_q_series_laurent_eval(h, state, finite_part=finite_part)
        except WindowDeficit as exc:
            # this weight reads deeper than the table's analysis covers: rebuild wider by exactly that much
            half = state["window_half"] + exc.deficit
