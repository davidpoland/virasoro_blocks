"""Recursion-singularity diagnostics and dispatch guards at a target c.

This module holds the singular-locus recognizers and guards used by the
evaluation paths (`recognized_singular_locus`, `reject_singular_c`,
`near_singular_compute`, `snap_float_c`, `unstripped_collisions`), and a
read-only report. The report inspects the universal recursion data up to a fixed q-order and
reports where the bare Zamolodchikov h-recursion becomes structurally dangerous:

- near-zero factors in A_rs(r, s, b), which make the recursion coefficient blow up
- collided Kac poles d_{r,s}(c), which force multiple (r, s) terms to share the
  same h-pole

For the recognized exact singular loci (c = 1, c = 25, the full minimal-model
series c = 1 - 6(p-q)^2/(pq), its p = 1 rational extensions, and the positive-rational c > 25 branch) the
report is constructed from exact integer/rational families rather than from a
tolerance-based numeric scan. The exact regularization that consumes these
classifications lives in `recursion.H_q_series_laurent` and `regularization`.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt

import mpmath as mp

from .params import b_from_c, d_rs


@dataclass(frozen=True)
class SingularFactor:
    pair: tuple[int, int]
    factor: tuple[int, int]
    magnitude: mp.mpf


@dataclass(frozen=True)
class PoleCollision:
    pairs: tuple[tuple[int, int], ...]
    pole: mp.mpc


@dataclass(frozen=True)
class RecursionSingularityReport:
    c: mp.mpc
    b: mp.mpc
    q_order: int
    tolerance: mp.mpf
    a_pole_factors: tuple[SingularFactor, ...]
    pole_collisions: tuple[PoleCollision, ...]

    @property
    def first_a_pole_order(self):
        if not self.a_pole_factors:
            return None
        return min(r * s for r, s in (item.pair for item in self.a_pole_factors))

    @property
    def first_collision_order(self):
        if not self.pole_collisions:
            return None
        return min(min(r * s for r, s in group.pairs) for group in self.pole_collisions)


def _c_is_exact_one(c) -> bool:
    return abs(mp.mpc(c) - 1) == 0


def _c_is_exact_twenty_five(c) -> bool:
    return abs(mp.mpc(c) - 25) == 0


# Hand-curated nicknames for minimal models a physicist will recognize at a
# glance. Anything not in this dict falls through to the generic M(p, q)
# label (plus the unitary index m when applicable). Higher unitary minimal
# models (m >= 5) don't have universally agreed-on common names and are
# omitted on purpose.
_MINIMAL_MODEL_NICKNAMES = {
    (2, 5): "Yang-Lee",           # non-unitary; c = -22/5
    (3, 4): "Ising",              # c = 1/2, m = 3
    (4, 5): "tricritical Ising",  # c = 7/10, m = 4
}


_NEAR_LOCUS_DEFAULT_TOL = "1e-10"


def _minimal_model_label(p: int, q: int, *, with_unitary: bool = True) -> str:
    """Format a minimal-model M(p, q) label, adding nicknames / unitary index."""
    label = f"minimal model M({p},{q})"
    nickname = _MINIMAL_MODEL_NICKNAMES.get((p, q))
    if nickname is not None:
        label += f" ({nickname})"
    elif with_unitary and q == p + 1:
        label += f" (unitary, m = {p})"
    return label


def _iter_minimal_model_candidates(c_real, *, max_n: int, abs_tolerance=None):
    """Yield the nearest canonical models on either side of c_real in each
    n = q - p family, with 1 < p < q <= max_n. Bounding the labels avoids
    warning about arbitrarily large models near c = 1.

    Implementation note (inverse-solve). The minimal-model relation
    c = 1 - 6 (p - q)^2 / (p q) is a quadratic in p once n = q - p is fixed:
    p^2 + n p - 6 n^2 / (1 - c) = 0, so the (real-valued) positive solution is

        p = n * (sqrt((25 - c) / (1 - c)) - 1) / 2.

    The model charge is monotone in p for fixed n, so only the nearest
    allowed coprime p below and above this
    inverse can minimize the charge distance. Rounding p alone is insufficient:
    the charge is nonlinear, and the nearest integer can fail the label bounds
    or coprimality condition. Integer searches bridge these gaps without
    enumerating all O(max_n**2) models.

    Integer square roots give exact floors of the inverse for rational c;
    tolerance endpoints need no numerical rounding allowance. Only real
    c < 1 has minimal-model solutions; the caller checks this domain.
    """
    if max_n < 3:
        return

    # Outside the bounded catalog the closest endpoint is known. This also
    # keeps a charge arbitrarily close to one an O(1) warning lookup.
    largest_odd_q = max_n if max_n % 2 else max_n - 1
    if c_real <= 1 - Fraction(6 * (largest_odd_q - 2)**2, 2 * largest_odd_q):
        yield 2, largest_odd_q
        return
    if c_real >= 1 - Fraction(6, (max_n - 1) * max_n):
        yield max_n - 1, max_n
        return

    def inverse_ratio(c):
        ratio = (25 - c) / (1 - c)
        return ratio.numerator, ratio.denominator

    def inverse_floor(ratio, n):
        a, b = ratio
        return (isqrt(a * n * n // b) - n) // 2

    central_ratio = inverse_ratio(c_real)
    lower_ratio = upper_ratio = low_c = None
    low_complement_num = low_complement_den = None
    if abs_tolerance is not None:
        # Invert the tolerance window once. Almost every family has no integer
        # in this window at the CLI's small default tolerance; skip it before
        # constructing or converting any model charge.
        low_c, high_c = c_real - abs_tolerance, c_real + abs_tolerance
        low_complement = 1 - low_c
        low_complement_num, low_complement_den = low_complement.numerator, low_complement.denominator
        lower_ratio = inverse_ratio(low_c)
        if high_c < 1:
            upper_ratio = inverse_ratio(high_c)
    for n in range(1, max_n - 1):
        min_p = 2
        if lower_ratio is not None:
            floor_p = inverse_floor(lower_ratio, n)
            # ceil(p*) = floor(p*) unless the quadratic vanishes exactly.
            at_endpoint = floor_p * (floor_p + n) * low_complement_num == 6 * n*n * low_complement_den
            min_p = max(2, floor_p if at_endpoint else floor_p + 1)
        max_p = min(max_n - n, inverse_floor(upper_ratio, n)) if upper_ratio is not None else max_n - n
        if min_p > max_p:
            continue
        floor_p = inverse_floor(central_ratio, n)
        lower = min(floor_p, max_p)
        while lower >= min_p and gcd(lower, n) != 1:
            lower -= 1
        upper = max(floor_p + 1, min_p)
        while upper <= max_p and gcd(upper, n) != 1:
            upper += 1
        if lower >= min_p:
            yield lower, lower + n
        if upper <= max_p and upper != lower:
            yield upper, upper + n


def _minimal_model_search_safe(c, *, max_n: int, abs_tolerance=None):
    """Run `_iter_minimal_model_candidates` if c is real and < 1, else empty.

    Encapsulates the domain check that minimal models exist only at real c < 1.
    """
    from .numeric_input import literal_components

    c_real, imaginary = literal_components(c)
    if imaginary != 0:
        return iter(())
    if c_real >= 1:
        return iter(())
    return _iter_minimal_model_candidates(c_real, max_n=max_n, abs_tolerance=abs_tolerance)


def nearby_singular_locus(
    c, *, abs_tolerance=_NEAR_LOCUS_DEFAULT_TOL, max_n: int = 10000,
):
    """If `c` is within `abs_tolerance` of (but not exactly at) c = 1,
    c = 25, or a minimal model, return `(label, distance)` for the closest.

    Returns None if c is exactly at a singular locus (the caller should use
    `recognized_singular_locus` for that case) or if no locus is within the
    tolerance. Imaginary part of c contributes to the distance; complex c far
    from the real axis is therefore not flagged.

    Used by the CLI's `--regularize auto` mode to nudge users who appear to be
    floating-point-close to a singular locus -- typically the result of a
    typo or a non-rational c that came out of an upstream optimization.
    """
    from .numeric_input import _integer_control, literal_components

    max_n = _integer_control('max_n', max_n)
    source, tolerance, _specified = _recognizer_input(c, abs_tolerance, tolerance_name='abs_tolerance')
    if tolerance is None:
        raise ValueError('abs_tolerance must be a finite nonnegative real number')
    real, imaginary = literal_components(source)
    return _nearby_singular_locus_numeric(real, imaginary, tolerance, max_n)


def _nearby_singular_locus_numeric(real, imaginary, abs_tolerance, max_n):
    from .numeric_input import parse_number

    real, imaginary, abs_tolerance = Fraction(real), Fraction(imaginary), Fraction(abs_tolerance)
    if abs(imaginary) > abs_tolerance:
        return None
    tolerance_squared = abs_tolerance**2
    best_label = None
    best_squared = None

    def distance_squared(target):
        # Subtract the labels before converting: a written shift smaller than
        # the caller's precision must not disappear or become target roundoff.
        return (real - target)**2 + imaginary**2

    # Universal loci c = 1 and c = 25.
    for c_target, label in (
        (Fraction(1), "c = 1"),
        (Fraction(25), "c = 25"),
    ):
        d = distance_squared(c_target)
        if d == 0:
            return None  # exact match
        if d <= tolerance_squared and (best_squared is None or d < best_squared):
            best_label = label
            best_squared = d

    # Minimal-model family via inverse-solve.
    for p, q in _minimal_model_search_safe(real, max_n=max_n, abs_tolerance=abs_tolerance):
        d = distance_squared(1 - Fraction(6 * (p - q)**2, p * q))
        if d == 0:
            return None  # exact match
        if d > tolerance_squared:
            continue
        if best_squared is None or d < best_squared:
            best_label = _minimal_model_label(p, q, with_unitary=False)
            best_squared = d

    if best_label is None:
        return None
    return (best_label, mp.sqrt(mp.mpf(parse_number(best_squared))))


class SnappedCentralChargeWarning(UserWarning):
    """A binary64 central charge within float rounding of a singular locus was taken to mean that locus."""


_SNAP_WARNED: set = set()


def _snap_target(c, *, family: str = "virasoro"):
    """The exact locus (a Fraction) a Python float ``c`` rounds from, or None (see :func:`snap_float_c`); no warning."""
    if not isinstance(c, float):
        return None
    with mp.workdps(max(mp.mp.dps, 40)):
        value = mp.mpf(c)
        tol = mp.mpf(2) ** -50 * max(abs(value), 1)
        candidates = [Fraction(1), Fraction(25)]
        if value < 1:
            # M(p, p + n): p = n (sqrt((25 - c) / (1 - c)) - 1) / 2 in each family n; the lowest n within reach
            half = (mp.sqrt((25 - value) / (1 - value)) - 1) / 2
            for n in range(1, _MINIMAL_MODEL_MAX_N + 1):
                p_int = int(mp.nint(n * half))
                if p_int + n > _MINIMAL_MODEL_MAX_N:
                    break
                if p_int < 2 or gcd(p_int, p_int + n) != 1:
                    continue
                model = 1 - Fraction(6 * n * n, p_int * (p_int + n))
                if abs(value - mp.mpf(model.numerator) / model.denominator) <= tol:
                    candidates.append(model)
                    break
            q_neg = negative_unit_b2_q(value, tolerance=mp.mpf(10) ** -8)
            if q_neg is not None:
                candidates.append(1 - Fraction(6 * (q_neg - 1) ** 2, q_neg))
        if value > 25:
            pq = positive_rational_b2_pq(value, tolerance=tol * 10)
            if pq is not None:
                p_int, q_int = pq
                candidates.append(13 + 6 * (Fraction(p_int, q_int) + Fraction(q_int, p_int)))
        if family == "ns":
            candidates += [Fraction(3, 2) - Fraction(12, m * (m + 2)) for m in range(3, 256)] + [Fraction(3, 2)]
        best = None
        for exact in candidates:
            distance = abs(value - mp.mpf(exact.numerator) / exact.denominator)
            if distance <= tol and (best is None or distance < best[1]):
                best = (exact, distance)
    return None if best is None else best[0]


def snap_float_c(c, *, family: str = "virasoro"):
    """A Python float ``c`` within float rounding (``2^-50 max(|c|, 1)``, a few double ulps) of a recognized singular
    locus is that locus, as a ``Fraction``: ``6/7``, ``25/28`` or ``-68/7`` typed as floats are the minimal models
    they round from, with a :class:`SnappedCentralChargeWarning` (once per value) where the float's own decimal
    spelling is not the locus.  Any other float is its shortest round-trip decimal (0.7 is 7/10, 0.73 is 73/100).
    Strings, Decimal and Fraction values are exact and an mpf/mpc is its value: none of those is snapped.  Loci:
    c = 1, 25, the minimal models M(p, q) and b^2 = -1/q, the rational b^2 > 0 duals; with ``family="ns"`` also the
    N=1 minimal models ``3/2 - 12/(m(m+2))``.
    """
    from .numeric_input import literal_number, parse_number

    if not isinstance(c, float):
        return parse_number(c)
    spelled = literal_number(c)
    if _BARE_PROBES.get():
        return spelled    # a regularizer's probe is deliberate, never a float typo
    exact = _snap_target(c, family=family)
    if exact is None:
        return spelled
    if spelled == exact:
        return spelled
    if (c, exact) not in _SNAP_WARNED:
        _SNAP_WARNED.add((c, exact))
        import warnings

        warnings.warn(
            f"c = {c!r} (a binary64 value) is taken as the singular central charge {exact}; pass an mpf or "
            "a Fraction to evaluate a nearby c", SnappedCentralChargeWarning, stacklevel=3,
        )
    return parse_number(exact)


_IN_NEAR_GUARD: contextvars.ContextVar[bool] = contextvars.ContextVar("_IN_NEAR_GUARD", default=False)
_NEAR_GUARD_CACHE: dict = {}


def _value_key(c):
    """A cheap exact cache key for a scalar (mp.nstr at dps + 5 digits cost ~25 us per F_block call)."""
    if isinstance(c, Fraction):
        return ('rational', c.numerator, c.denominator)
    value = mp.convert(c)
    return value._mpc_ if isinstance(value, mp.mpc) else value._mpf_


def _measured_recursion_loss(c, h_externals, q_order) -> int:
    """The bits the certified bare recursion needs at this c beyond its own baseline: the H table's escalation plus
    the top-level sum's deficit at an irrational probe weight (both read from its balls, ``recursion.
    _certified_h_table`` / ``_power_series_deficit``).  0 where it is well-conditioned (a generic c); positive near a
    singular locus or at large |c| (7/10 + 1e-16: 111 bits; 1e4: 235), where the chiral objects built from the same
    recursion lose digits too.  It decides whether :func:`near_singular_compute` verifies; its size is not a guard
    (it bounds the objects' loss 2-4x pessimistically).  A recursion that cannot certify itself here counts as
    lossy."""
    if q_order is None or int(q_order) < 1:
        return 0
    try:
        key = ("loss", _value_key(c), tuple(_value_key(h) for h in h_externals), int(q_order), mp.mp.prec)
    except (TypeError, ValueError):
        return 1
    hit = _NEAR_GUARD_CACHE.get(key)
    if hit is not None:
        return hit      # F_block calls this per evaluation: one dict lookup when warm
    from .params import b_from_c
    from .recursion import (_build_H_context, _eval_H_balls, _h_fill_first_prec, _kac_input_bits, _kac_input_values,
                            _maybe_real_scalar, _pin_mode, _power_series_deficit)

    try:
        b = b_from_c(c)
        b_typed = _maybe_real_scalar(b) if isinstance(b, mp.mpc) else b
        ext = _kac_input_values(h_externals)
        with _pin_mode(b_typed, ext):
            _pairs, _table, ctx = _build_H_context(int(q_order), b_typed, ext)
            if ctx is None:
                loss = 1
            else:
                loss = max(0, int(ctx.get("guard_bits", 0)) - (_h_fill_first_prec(int(q_order)) - mp.mp.prec))
                probe = _eval_H_balls(mp.sqrt(2), ctx, int(q_order))
                loss += 1 if probe is None else _power_series_deficit(
                    probe[0], mp.mp.prec, _kac_input_bits(b_typed, ext))[0]
    except (ValueError, ZeroDivisionError, ArithmeticError):
        loss = 1
    _near_guard_cache_put(key, loss)
    return loss


def _near_guard_cache_put(key, value):
    if len(_NEAR_GUARD_CACHE) > 4096:
        _NEAR_GUARD_CACHE.clear()
    _NEAR_GUARD_CACHE[key] = value


def _flat_numbers(value, out):
    if isinstance(value, (list, tuple)):
        for item in value:
            _flat_numbers(item, out)
    elif isinstance(value, (mp.mpf, mp.mpc, int, float, complex)):
        out.append(mp.mpc(value))
    return out


def _agreement_digits(low, high, base_dps):
    """The digits ``low`` agrees with ``high`` to: entrywise relative, except for entries more than ``base_dps``
    digits below the largest, which sit at the largest's roundoff level at the working precision and are compared
    at that absolute level; None when the shapes differ."""
    a, b = _flat_numbers(low, []), _flat_numbers(high, [])
    if len(a) != len(b) or any(not mp.isfinite(x) for x in (*a, *b)):
        return None
    scale = max((abs(x) for x in b), default=mp.mpf(0))
    floor = scale * mp.mpf(10) ** (-base_dps)
    worst = mp.mpf(0)
    for x, y in zip(a, b):
        ref = max(abs(y), floor)
        if ref == 0:
            if x != y:
                return mp.mpf(0)
            continue
        worst = max(worst, abs(x - y) / ref)
    return mp.inf if worst == 0 else -mp.log10(worst)


def near_singular_compute(c, site, compute, *, h_externals, q_order, fingerprint=None, return_guard=False,
                          cache_context=()):
    """Run ``compute()`` at a VERIFIED guard precision where the recursion is ill-conditioned (near, not on, a
    singular locus, or at large |c|).

    There the colliding Kac poles (or the large residues) cancel and the chiral objects built from the bare recursion
    lose digits: F_block_chiral_h at 7/10 + 1e-16, dps 30, 14; at 1 + 1e-14, 42; at c = 1e5, 52.  The loss depends on
    the quantity, so it is measured on the quantity itself.  Whether to verify is the certified recursion's own
    measured loss at this c (:func:`_measured_recursion_loss`: 0 at a generic c, where ``compute`` runs once).  The
    first call per (c, dps, externals, q_order, ``site``, ``cache_context``) then runs :func:`verified_guard_compute`
    (``fingerprint(result)`` gives the numbers compared, default the result itself); the guard is cached and later
    calls run once at it.  c is never changed, only the precision it is evaluated at.  With ``return_guard`` the
    result is ``(value, extra_digits)`` (0 when no guard applies), for results evaluated later.
    ``cache_context`` must include the remaining inputs affecting the guarded
    quantity, such as its point, coordinate transform, normalization and shifts.
    """
    def done(value, extra=0):
        return (value, extra) if return_guard else value

    # the regularizers' deliberate probes (c + t^2 next to the locus) manage their own precision: no guard there
    if _IN_NEAR_GUARD.get() or _BARE_PROBES.get() or exact_locus_c(c, q_order=q_order) is not None:
        return done(compute())
    if not _measured_recursion_loss(c, h_externals, q_order):
        return done(compute())
    from .numeric_input import parse_number

    context_key = tuple(value if value is None or callable(value) else _value_key(parse_number(value))
                        for value in cache_context)
    key = ("site", site, _value_key(c), mp.mp.prec, tuple(_value_key(h) for h in h_externals), int(q_order), context_key)
    result, guard = verified_guard_compute(
        key, compute, fingerprint=fingerprint,
        what=f"near-singular c = {mp.nstr(mp.convert(c), 20)}: {site}",
        hint="evaluate at the singular c (regularized) or further from it")
    return done(result, guard)


def verified_guard_compute(key, compute, *, fingerprint=None, what: str, hint: str):
    """``(compute() at a verified guard precision, guard digits)``, the guard measured on the result itself.

    The first call per ``key`` (which must include ``mp.mp.prec``) runs ``compute`` at the working precision ``P``
    and at twice it; their agreement ``a`` (``fingerprint(result)`` the numbers compared, default the result) measures
    the digits it loses, ``L = P - a``, the same at any precision.  The doubled run carries ``2P - L`` digits: when that
    is at least ``P + 3`` (the agreement the result is accepted at) it is returned, else the precision doubles again
    and the next pair measures ``L``.  The guard ``L + 3`` is cached, so later calls run once at it.  A ``ValueError``
    at the lower precision only (rounding read as a structural singularity) is outrun the same way; one at both is
    raised.  The doubling stops at 16 times the working precision.
    """
    base = mp.mp.dps
    fingerprint = fingerprint or (lambda value: value)

    def run(extra):
        token = _IN_NEAR_GUARD.set(True)
        try:
            with mp.workdps(base + extra):
                result = compute()
                sample = fingerprint(result)
                if any(not mp.isfinite(x) for x in _flat_numbers(sample, [])):
                    raise ValueError(f"{what}: non-finite numerical result; {hint}")
                return result, sample
        finally:
            _IN_NEAR_GUARD.reset(token)

    cached = _NEAR_GUARD_CACHE.get(key)
    if cached is not None:
        return run(cached)[0], cached
    cap = 16 * base
    guard = 0
    try:
        low = run(guard)
    except ValueError:
        low = None
    while True:
        high_guard = 2 * (base + guard) - base
        if base + high_guard > cap:
            raise ValueError(f"{what} did not stabilize below {cap} digits; {hint}")
        try:
            high = run(high_guard)
        except ValueError:
            if low is None:
                raise
            raise ValueError(f"{what}: the evaluation raised at {base + high_guard} digits only; {hint}") from None
        if low is not None:
            agreement = _agreement_digits(low[1], high[1], base)
            if agreement is not None and agreement > 0:
                loss = max(0, base + guard - agreement) if mp.isfinite(agreement) else 0
                if base + high_guard - loss >= base + 3:
                    result_guard = int(mp.ceil(loss)) + 3
                    _near_guard_cache_put(key, result_guard)
                    return high[0], high_guard
        guard, low = high_guard, high


def recognized_singular_locus(c, *, q_order: int | None = None) -> str | None:
    """Describe a recognized Virasoro-singular locus, or return ``None``.

    Exact real inputs (including decimal text, Decimal, Fraction, and Python
    floats interpreted by their shortest decimal spelling) are classified by
    exact rational arithmetic in ``b**2``. There is no proximity tolerance on
    this path: a nonzero rational offset is retained.  (The value paths snap a
    Python float within its own rounding of a locus to that locus before they
    get here; see :func:`snap_float_c`.)  The recognized families are
    ``c = 1``, ``c = 25``, the minimal models ``c = 1 - 6(p-q)**2/(p*q)``,
    their ``p = 1`` rational extensions, and ``c > 25`` with positive rational
    ``b**2``.  Loci whose singular factors first appear beyond any practical
    recursion order are not reported: minimal models with ``q - p`` above
    ``_MINIMAL_MODEL_MAX_N``, and ``b**2 = p/q`` with ``q`` above
    ``max(16, q_order)``.

    Already-created mpmath values retain the lower-level numerical recognition
    convention. That fallback checks the two boundary charges, then inverts
    the minimal-model and rational-b-squared relations with rounding
    tolerances (see the individual recognizers). Consequently a sufficiently
    small mpf perturbation may be treated as roundoff; callers that intend an
    exact offset should supply an exact real label instead.

    The numerical positive-b-squared search uses the same denominator cutoff.
    The result identifies a locus;
    whether its structural singularities occur at the requested q-order is
    checked separately by the recursion guards.

    Labels include ``"c = 1"``, ``"minimal model M(3,4) (Ising)"``, and
    ``"b² = 1/6"`` (the dual locus at ``c = 50``).
    """
    entry = _recognized_locus_entry(c, q_order)
    return None if entry is None else entry[0]


def _optional_q_order(q_order):
    from .numeric_input import _integer_control

    return None if q_order is None else _integer_control('q_order', q_order)


def _recognized_locus_entry(c, q_order):
    """Cached ``(label, (eps, p, q))`` of the recognized locus (b² = eps·p/q), or None."""
    from .numeric_input import exact_real, exact_b_squared as rational_b_squared, parse_number, _explicit_nonreal_input

    q_order = _optional_q_order(q_order)
    c = parse_number(c)
    if _explicit_nonreal_input(c):
        return None
    max_q = _DUAL_LOCUS_MAX_Q if q_order is None else max(_DUAL_LOCUS_MAX_Q, int(q_order))
    exact = exact_real(c)
    if exact is not None:
        # the cutoffs of the numerical recognizers below (minimal_model_pq's max_n, the dual family's max_q): a
        # dual locus b² = p/q first divides by zero at q_order ~ q, so b² = 1/100 (c = 613.06) is not singular below
        beta = rational_b_squared(exact)
        if beta is None:
            return None
        if exact == 1:
            return "c = 1", (-1, 1, 1)
        if exact == 25:
            return "c = 25", (1, 1, 1)
        p, q = sorted((abs(beta.numerator), beta.denominator))
        if beta < 0:
            if p > 1 and q - p > _MINIMAL_MODEL_MAX_N:
                return None
            label = _minimal_model_label(p, q) if p > 1 else f"b² = -1/{q}"
            return label, (-1, p, q)
        if q > max_q:
            return None
        return f"b² = {p}/{q}", (1, p, q)
    if isinstance(c, (float, complex, str, Fraction)):
        c = mp.convert(c)   # cacheable, and a Fraction has no mpc()
    key = _locus_cache_key(c)
    if key is not None:
        key = key + (max_q,)
        if key in _LOCUS_CACHE:
            return _LOCUS_CACHE[key]
    result = _recognized_singular_locus_uncached(c, max_q=max_q)
    if key is not None:
        if len(_LOCUS_CACHE) > 4096:
            _LOCUS_CACHE.clear()
        _LOCUS_CACHE[key] = result
    return result


#: default dual-family (c > 25, b² = p/q) denominator cutoff of `recognized_singular_locus`
_DUAL_LOCUS_MAX_Q = 16

#: the minimal-model cutoff ``q - p <= max_n`` of `minimal_model_pq` (and of `recognized_singular_locus`)
_MINIMAL_MODEL_MAX_N = 10000


#: ``recognized_singular_locus`` is asked on every block evaluation, almost always for one ``c``
_LOCUS_CACHE: dict = {}


def _locus_cache_key(c):
    import mpmath as _mp
    from fractions import Fraction as _F

    if isinstance(c, (int, _F)):
        return ("exact", c, _mp.mp.prec)
    if isinstance(c, _mp.mpf):
        return ("mpf", c._mpf_, _mp.mp.prec)
    if isinstance(c, _mp.mpc):
        return ("mpc", c._mpc_, _mp.mp.prec)
    return None


def _recognized_singular_locus_uncached(c, *, max_q: int = 16):
    """``(label, (eps, p, q))`` with b² = eps·p/q at the recognized locus, or None."""
    c = mp.convert(c)   # a Fraction (the exact-input convention of precision.convert_exact) has no mpc()
    if isinstance(c, mp.mpc) and abs(c.imag) <= mp.mpf(10) ** (-(mp.mp.dps - 10)) * max(abs(c), 1):
        # an imaginary part below the recognition tolerance is roundoff: c = 1/2 + 1e-40 i at dps 30 is the Ising
        # c, where the bare recursion divided by zero because every real-axis check below required im(c) == 0
        c = c.real
    if _c_is_exact_one(c):
        return "c = 1", (-1, 1, 1)
    if _c_is_exact_twenty_five(c):
        return "c = 25", (1, 1, 1)
    pq = minimal_model_pq(c)
    if pq is not None:
        return _minimal_model_label(*pq), (-1, *pq)
    q_unit = negative_unit_b2_q(c)
    if q_unit is not None:
        return f"b² = -1/{q_unit}", (-1, 1, q_unit)
    pq_b2 = positive_rational_b2_pq(c, max_q=max_q)
    if pq_b2 is not None:
        return f"b² = {pq_b2[0]}/{pq_b2[1]}", (1, *pq_b2)
    return None


#: (eps, p, q, q_order) -> the singular structures of the bare recursion, see `_bare_singular_structures`
_HITS_CACHE: dict = {}


def _bare_singular_structures(eps: int, p: int, q: int, q_order: int):
    """``(a_pairs, collisions)``: where the bare recursion divides by zero at b² = eps·p/q, up to `q_order`.

    Exact integer check of the two denominators of the Zamolodchikov recursion
    (``H(h) = 1 + sum (16q)^{rs} K_{rs} H(d_{rs} + rs) / (h - d_{rs})``):

    - ``a_pairs``: pairs ``(r, s)``, ``rs <= q_order``, whose A_rs has a
      vanishing factor ``m b + n/b`` (``eps p m + q n = 0``) in its product range;
    - ``collisions``: ``((r, s), (r', s'))`` with a shifted weight on a pole,
      ``d_{r,s} + rs = d_{r',s'}``, and ``rs + r's' <= q_order`` (the inner
      ``H(d_{rs} + rs)`` then divides by zero). Since ``d_{r,s} + rs = d_{r,-s}``
      and ``d`` depends on ``(r b + s/b)^2 ∝ (eps p r + q s)^2``, this is
      ``|eps p r - q s| = |eps p r' + q s'|``.

    Plain collisions ``d_{r,s} = d_{r',s'}`` only merge poles (a repeated root
    of the symbolic denominator) and are harmless.
    """
    key = (eps, p, q, int(q_order))
    hit = _HITS_CACHE.get(key)
    if hit is not None:
        return hit
    pairs = [(r, n // r) for n in range(1, q_order + 1) for r in range(1, n + 1) if n % r == 0]
    a_pairs = []
    for r, s in pairs:           # A_rs zeros: (m, n) = (q t, -eps p t)
        for t in range(-(r + s), r + s + 1):
            m, n = q * t, -eps * p * t
            if t != 0 and 1 - r <= m <= r and 1 - s <= n <= s and (m, n) != (r, s):
                a_pairs.append((r, s))
                break
    by_label: dict = {}
    for r, s in pairs:
        by_label.setdefault(abs(eps * p * r + q * s), []).append((r, s))
    collisions = []
    for r, s in pairs:
        for other in by_label.get(abs(eps * p * r - q * s), ()):
            if r * s + other[0] * other[1] <= q_order:
                collisions.append(((r, s), other))
    result = (tuple(a_pairs), tuple(collisions))
    if len(_HITS_CACHE) > 4096:
        _HITS_CACHE.clear()
    _HITS_CACHE[key] = result
    return result


def bare_recursion_singular_order(eps: int, p: int, q: int, q_order: int) -> int | None:
    """Lowest truncation order <= `q_order` at which the bare recursion divides by zero at b² = eps·p/q.

    Structural (external-independent) bound from `_bare_singular_structures`;
    None when neither an A_rs zero nor a shifted-weight collision occurs up to
    `q_order` -- then the bare recursion is exact at this c even though it is a
    recognized locus (e.g. Ising below q_order 7).

    ``eps`` is -1 or +1 and ``p``, ``q`` are positive integers. Equivalent
    ratios are reduced before enumerating the vanishing factors.
    """
    from .numeric_input import _integer_control

    eps = _integer_control('eps', eps, -1)
    if eps not in (-1, 1):
        raise ValueError('eps must be -1 or +1')
    p, q = _integer_control('p', p, 1), _integer_control('q', q, 1)
    q_order = _integer_control('q_order', q_order)
    common = gcd(p, q)
    p, q = p // common, q // common
    a_pairs, collisions = _bare_singular_structures(eps, p, q, q_order)
    orders = [r * s for r, s in a_pairs] + [
        r * s + r2 * s2 for (r, s), (r2, s2) in collisions
    ]
    return min(orders) if orders else None


def _fusion_vanishes(r: int, s: int, b_sq, h_left, h_right, tol) -> bool:
    """Whether the fusion polynomial P_{r,s}[h_left, h_right] has an (exactly) vanishing factor.

    Factor-by-factor, in the (p, q) <-> (-p, -q) paired squares form of
    `params.fusion_polynomial_via_squares`, each factor tested against its own
    scale -- so a residue killed by the externals (e.g. the (0, 0) factor
    lam_L^2 - lam_R^2 of identical pairs) is recognized as zero, not as roundoff.
    """
    Q_sq = b_sq + 2 + 1 / b_sq
    lam_L_sq = Q_sq - 4 * h_left
    lam_R_sq = Q_sq - 4 * h_right
    seen = set()
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if (p, q) in seen:
                continue
            seen.add((p, q))
            seen.add((-p, -q))
            if p == 0 and q == 0:
                if abs(lam_L_sq - lam_R_sq) <= tol * (abs(lam_L_sq) + abs(lam_R_sq)):
                    return True
                continue
            shift_sq = p * p * b_sq + 2 * p * q + q * q / b_sq
            u = lam_L_sq + shift_sq - lam_R_sq
            v_sq = 4 * lam_L_sq * shift_sq
            if abs(u * u - v_sq) <= tol * (abs(u) ** 2 + abs(v_sq)):
                return True
    return False


def _bare_recursion_is_singular(family, q_order: int, externals) -> bool:
    """Whether the bare recursion at this locus meets a singular term with a nonzero residue.

    With ``externals=None`` the structural bound (any singular term at all).
    Otherwise a structure is live only if the residues it multiplies are
    nonzero for these externals: an A_rs zero needs K_{rs} != 0, a shifted
    collision needs K_{rs} K_{r's'} != 0 (K_{rs} = 16^rs A_rs P[h1, h2] P[h4, h3]).
    E.g. at c = 50 (b² = 1/6) with pairwise-identical externals every structure
    up to q_order 7 has a vanishing residue, and the bare value is exact.
    """
    eps, p, q = family
    a_pairs, collisions = _bare_singular_structures(eps, p, q, int(q_order))
    if not a_pairs and not collisions:
        return False
    if externals is None:
        return True
    try:
        h1, h2, h3, h4 = (mp.convert(x) for x in externals)
    except (TypeError, ValueError):
        return True
    b_sq = mp.mpf(eps * p) / q
    tol = mp.mpf(10) ** (-(mp.mp.dps - 10))
    zero_cache: dict = {}

    def residue_zero(pair):
        if pair not in zero_cache:
            r, s = pair
            zero_cache[pair] = _fusion_vanishes(r, s, b_sq, h1, h2, tol) or _fusion_vanishes(
                r, s, b_sq, h4, h3, tol
            )
        return zero_cache[pair]

    if any(not residue_zero(pair) for pair in a_pairs):
        return True
    return any(not residue_zero(one) and not residue_zero(two) for one, two in collisions)


def negative_unit_b2_q(c, *, tolerance=None):
    """Return the integer q >= 2 if c = 1 - 6 (q-1)^2 / q, i.e. b^2 = -1/q, else None.

    This is the p = 1 end of the minimal-model formula (c = -2, -7, -25/2, ...),
    which `minimal_model_pq` rejects because M(1, q) is not a minimal model; the
    bare recursion is nevertheless singular there (see
    `recognized_singular_locus`). O(1): q + 1/q = (13 - c)/6 is solved for the
    root q >= 1. Exact real labels are classified exactly by default, as in
    `minimal_model_pq`; native mpmath values use tolerance
    `10**(-(mp.mp.dps - 10))` on the inverse-solve's integer q. An explicit
    positive tolerance opts into approximate matching; zero is exact.
    """
    from .numeric_input import exact_b_squared as rational_b_squared

    source, tolerance, specified = _recognizer_input(c, tolerance)
    if specified is not None and (tolerance is None or tolerance == 0):
        beta = rational_b_squared(specified)
        return beta.denominator if beta is not None and beta.numerator == -1 and beta.denominator >= 2 else None
    return _negative_unit_b2_q_numeric(source, tolerance)


def _negative_unit_b2_q_numeric(source, tolerance):
    c_val = mp.mpc(mp.convert(source))
    if abs(mp.im(c_val)) != 0:
        return None
    c_real = mp.re(c_val)
    if c_real > -2 + mp.mpf("0.5"):   # q = 2 gives the largest such c, -2
        return None
    if tolerance is None:
        tolerance = mp.mpf(10) ** (-(mp.mp.dps - 10))
    s = (13 - c_real) / 6
    q_candidate = (s + mp.sqrt(s * s - 4)) / 2
    q_round = mp.nint(q_candidate)
    if q_round < 2 or abs(q_candidate - q_round) > tolerance:
        return None
    return int(q_round)


class SingularCentralChargeError(ValueError, ZeroDivisionError):
    """The bare recursion was asked for a c where it is singular (see `reject_singular_c`).

    A ValueError (bad input for this route) that is also a ZeroDivisionError, so
    callers that caught the bare recursion's division by zero to fall back on a
    regularizer keep working.
    """


#: set while a regularizer probes the bare recursion at c + eps (see `bare_recursion_probes`)
_BARE_PROBES: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "virasoro_blocks_bare_recursion_probes", default=False
)


#: set while a caller wants every Kac root of the probe builds (see `unstripped_collisions`)
_UNSTRIPPED_COLLISIONS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "virasoro_blocks_unstripped_collisions", default=False
)


@contextmanager
def bare_recursion_probes():
    """Suspend `reject_singular_c` inside the block (for the c-regularizers).

    The epsilon-extrapolation in `regularization` evaluates the bare builders at
    probe charges ``c + eps`` (or ``c + t**2``). With the default ladders ``eps``
    sits far above the recognition tolerance ``10**-(dps - 10)``, but a caller
    may pass ``work_dps`` as low as the base precision, where the smallest probe
    can fall inside it; the probes are deliberate, so the guard stands down.
    """
    token = _BARE_PROBES.set(True)
    try:
        yield
    finally:
        _BARE_PROBES.reset(token)


def exact_b_squared(c, *, q_order: int | None = None):
    """The rational ``b^2`` (a ``Fraction``) of the singular locus ``c`` denotes, or None (see :func:`exact_locus_c`)."""
    from fractions import Fraction

    from .numeric_input import exact_real, exact_b_squared as rational_b_squared

    q_order = _optional_q_order(q_order)
    if exact_real(c) is not None:
        return rational_b_squared(c)
    exact = exact_locus_c(c, q_order=q_order)
    if exact is None:
        return None
    with mp.workdps(max(30, mp.mp.dps)):
        from .params import b_from_c

        beta = b_from_c(exact) ** 2
        beta = mp.re(beta) if abs(mp.im(beta)) < mp.mpf(10) ** -20 * max(1, abs(beta)) else beta
        if isinstance(beta, mp.mpc):
            return None
        frac = Fraction(mp.nstr(beta, 25)).limit_denominator(10 ** 6)
    return frac if abs(mp.mpf(frac.numerator) / frac.denominator - beta) < mp.mpf(10) ** -15 * max(1, abs(beta)) else None


def exact_locus_c(c, *, q_order: int | None = None):
    """``c`` as the exact singular locus it denotes, at the current precision, or None if it is not one.

    Every recognized locus has a rational ``b^2 = beta`` (c = 1: -1, c = 25: 1, the minimal models -p/q, the duals
    p/q), and then ``c = 13 + 6 (beta + 1/beta)`` exactly.  Recognition runs at the precision ``c`` itself carries (its
    mantissa), so a value built at the caller's precision is still recognized inside a builder that raised the working
    precision (the symbolic table builders do), where comparing at the raised precision misses it."""
    from fractions import Fraction

    from .numeric_input import exact_real, exact_b_squared as rational_b_squared, parse_number, _explicit_nonreal_input

    q_order = _optional_q_order(q_order)
    c = parse_number(c)
    if _explicit_nonreal_input(c):
        return None
    specified = exact_real(c)
    if specified is not None:
        if rational_b_squared(specified) is None:
            return None
        exact = specified
    else:
        z = mp.mpc(mp.convert(c))
        if z.imag != 0 and recognized_singular_locus(z, q_order=q_order) is None:
            return None             # (an imaginary part below the recognition tolerance is roundoff)
        x = mp.mpf(z.real)
        bits = x._mpf_[3] if x != 0 else 1
        digits = max(15, int(bits * 0.30103) - 2)
        with mp.workdps(digits):
            if recognized_singular_locus(+x, q_order=q_order) is None:
                return None
            from .params import b_from_c

            beta = b_from_c(+x) ** 2
            beta = mp.re(beta) if abs(mp.im(beta)) < abs(beta) * mp.mpf(10) ** (-(digits // 2)) else beta
            if isinstance(beta, mp.mpc):
                return None
            frac = Fraction(mp.nstr(beta, digits)).limit_denominator(10 ** 6)
            if abs(mp.mpf(frac.numerator) / frac.denominator - beta) > abs(beta) * mp.mpf(10) ** (-(digits // 2)):
                return None
        exact = 13 + 6 * (frac + 1 / frac)
        # The locus search proposes a candidate with a tolerance wider than roundoff. An mpf is that locus only to
        # roundoff at the precision it is used at (its own, or the working precision when that is lower): a
        # resolvable offset (0.50000001 at 15 digits) is a nearby c, evaluated by the near-singular path.
        used_bits = max(53, min(bits, mp.mp.prec))
        with mp.workprec(bits + 32):
            candidate = mp.mpf(exact.numerator) / exact.denominator
            if abs(x - candidate) > max(1, abs(x)) * mp.mpf(2) ** (3 - used_bits):
                return None
    return mp.mpf(exact.numerator) / exact.denominator


_MERGED_EVALUATORS = contextvars.ContextVar("vb_merged_singular_evaluators", default=False)


@contextmanager
def merged_singular_evaluators():
    """Build the chiral evaluators inside AT a recognized singular c, exactly (no probes, no extrapolation).

    Their H coefficients are the merged-pole form of the Laurent-in-t recursion
    (:func:`virasoro_blocks.recursion.build_merged_H_context`); the geometry (nome, prefactor) is analytic at c0 and
    is the generic evaluator's.  Outside this block a singular c is refused as before."""
    token = _MERGED_EVALUATORS.set(mp.mp.dps)
    try:
        yield
    finally:
        _MERGED_EVALUATORS.reset(token)


def merged_singular_active() -> bool:
    return bool(_MERGED_EVALUATORS.get())


def merged_singular_dps():
    """The working precision ``merged_singular_evaluators`` was entered at (the one Laurent table is built there; a
    builder that raises the precision inside -- the symbolic skeleton -- reads it rather than building another)."""
    value = _MERGED_EVALUATORS.get()
    return None if value is False else int(value)


@contextmanager
def unstripped_collisions():
    """Keep every colliding Kac root in the builds inside (``polyblock._strip_common_chiral_factors`` stands down).

    A regularizer's probe charges ``c + t**2`` split colliding Kac weights by O(t**2), below the builders'
    coincidence tolerance, so each probe divides the collision surplus out as removable.  At a minimal model it is
    (the regularized tables rely on it), but where the colliding residues diverge like ``t**-2`` the surplus is a
    genuine higher-order pole -- Delta = -1/4 at c = -2 (scalar G) is order 6, from the labels (2,1), (4,2), (6,3)
    on each side -- and a pole-order oracle built on stripped probes counted 2 and capped the measurement there.
    """
    token = _UNSTRIPPED_COLLISIONS.set(True)
    try:
        yield
    finally:
        _UNSTRIPPED_COLLISIONS.reset(token)


def bare_singular_locus(
    c=None, *, b=None, q_order: int | None = None, h_externals=None,
    allow_minimal_models: bool = False,
):
    """``(label, exact_c)`` if the bare recursion meets a live singular term at this c, else None.

    At a `recognized_singular_locus` the bare Zamolodchikov recursion divides by
    (near-)zero Kac data and returns values 10^39-10^100 off, or raises
    ZeroDivisionError -- once `q_order` reaches a singular term whose residue
    is nonzero for `h_externals` (`_bare_recursion_is_singular`). Below that the
    bare value is exact and None is returned (e.g. Ising at q_order < 7, or
    c = 50 with pairwise-identical externals at q_order <= 7). Without
    `h_externals` the check is structural (any singular term up to `q_order`);
    without `q_order`, any recognized locus counts. Only `b` given: c is derived
    from it (the recognizers' tolerances absorb the round-trip roundoff).
    ``exact_c`` is the locus value rebuilt from its exact b² = eps·p/q at the
    working precision (what the exact-c Laurent regularizer needs).

    None inside `bare_recursion_probes`, for non-numeric c (e.g. a Laurent
    series), and -- with ``allow_minimal_models=True`` -- at the c < 1 minimal
    models M(p, q) (the symbolic tables, whose direct build there is a
    pole-multiplicity oracle; see `polyderivs.chiral_symbolic_useries`).
    """
    q_order = _optional_q_order(q_order)
    if _BARE_PROBES.get():
        return None
    if c is None:
        if b is None:
            return None
        from .params import _c_input_of_b

        try:
            return bare_singular_locus(
                _c_input_of_b(b), b=b, q_order=q_order,
                h_externals=h_externals, allow_minimal_models=allow_minimal_models,
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    try:
        entry = _recognized_locus_entry(c, q_order)
    except (TypeError, ValueError):
        return None
    if entry is None:
        return None
    locus, (eps, p, q) = entry
    if allow_minimal_models and eps == -1 and p >= 2:
        return None
    if q_order is not None and not _bare_recursion_is_singular((eps, p, q), int(q_order), h_externals):
        return None     # the truncated recursion meets no live singular term: the bare value is exact
    exact_c = mp.mpf(13) + 6 * eps * (mp.mpf(p) / q + mp.mpf(q) / p)
    return locus, exact_c


def reject_singular_c(
    c=None, *, b=None, q_order: int | None = None, where: str = "this builder",
    h_externals=None, allow_minimal_models: bool = False,
):
    """Raise `SingularCentralChargeError` (a ValueError) where `bare_singular_locus` fires.

    `F_block` / `G_block` / `B_block` and the per-h chiral Taylor builders route
    such c to the regularizer; the series evaluators, `evaluate_block`,
    `F_block_chiral_h` and the symbolic tables fail closed here instead.
    """
    hit = bare_singular_locus(
        c, b=b, q_order=q_order, h_externals=h_externals,
        allow_minimal_models=allow_minimal_models,
    )
    if hit is None:
        return
    locus, exact_c = hit
    raise SingularCentralChargeError(
        f"{where}: c = {mp.nstr(exact_c, 15)} is the recursion-singular locus [{locus}], where "
        "the bare Zamolodchikov recursion returns garbage or divides by zero. Use the regularized "
        "routes instead: F_block / G_block / B_block (regularize='auto'), F_block_regularized / "
        "G_block_regularized / B_block_regularized / H_q_series_regularized, or build the table "
        "at the singular c with regularization.merged_delta_table (tables) / "
        "regularization._RegularizedNumericSampler (numeric grids)."
    )


def _recognizer_input(c, tolerance, *, tolerance_name='tolerance'):
    from .numeric_input import exact_real, literal_components, parse_number, real_input_label

    if isinstance(c, bool):
        raise ValueError('c must be a finite number, not a boolean')
    source = parse_number(c)
    if not mp.isfinite(mp.convert(source)):
        raise ValueError('c must be finite')
    tolerance = None if tolerance is None else real_input_label(tolerance, name=tolerance_name)
    if tolerance is not None and tolerance < 0:
        raise ValueError(f'{tolerance_name} must be nonnegative')
    specified = exact_real(source)
    if tolerance == 0 and specified is None:
        real, imaginary = literal_components(source)
        if imaginary == 0:
            specified = Fraction(real)
    return source, tolerance, specified


def positive_rational_b2_pq(c, *, max_q: int = 200, tolerance=None):
    """Return canonical (p, q) if c comes from positive rational b^2 = p/q.

    The canonical choice uses coprime integers with 1 <= p <= q, matching the
    `|b| <= 1` branch chosen by `b_from_c` for real `c >= 25`.

    Exact real labels use rational arithmetic by default, retaining every
    nonzero offset. For existing mpmath values, the default numerical
    tolerance is ``10**(-(mp.mp.dps - 10))`` and accepts round-trip error.
    An explicit positive ``tolerance`` opts into approximate matching of
    ``c`` for any input type; zero requires exact matching.
    """
    from .numeric_input import _integer_control, exact_b_squared as rational_b_squared, _explicit_nonreal_input

    max_q = _integer_control('max_q', max_q)
    source, tolerance, specified = _recognizer_input(c, tolerance)
    if _explicit_nonreal_input(source) and (tolerance is None or tolerance == 0):
        return None
    if specified is not None and (tolerance is None or tolerance == 0):
        beta = rational_b_squared(specified)
        if beta is None or beta <= 0 or beta.denominator > max_q:
            return None
        return beta.numerator, beta.denominator
    return _positive_rational_b2_pq_numeric(source, max_q, tolerance)


def _positive_rational_b2_pq_numeric(source, max_q, tolerance):
    if tolerance is None:
        tolerance = mp.mpf(10) ** (-(mp.mp.dps - 10))
    tolerance = mp.mpf(tolerance)
    c_val = mp.mpc(mp.convert(source))
    if abs(mp.im(c_val)) > tolerance:
        return None
    c_real = mp.re(c_val)
    if c_real < 25 - tolerance:
        return None
    # Inverse-solve, O(max_q): b^2 = x in (0, 1] solves x + 1/x = (c - 13)/6, and within each
    # denominator q only p = nint(x q) can be within the (tiny) tolerance of c, since
    # c(p/q) = 13 + 6 (p/q + q/p) is monotone in p/q with gaps >> tolerance.
    s = (c_real - 13) / 6
    x = (s - mp.sqrt(max(s * s - 4, 0))) / 2
    for q in range(1, max_q + 1):
        p0 = int(mp.nint(x * q))
        for p in (p0 - 1, p0, p0 + 1):
            if p < 1 or p > q or gcd(p, q) != 1:
                continue
            target = mp.mpf(13) + mp.mpf(6) * (mp.mpf(p) / q + mp.mpf(q) / p)
            if abs(c_real - target) <= tolerance:
                return (p, q)
    return None


def minimal_model_pq(c, *, max_n: int = _MINIMAL_MODEL_MAX_N, tolerance=None):
    """Return canonical (p, q) if c = 1 - 6 (p-q)^2 / (p q) at the current
    working precision, via the inverse-solve algorithm
    (see `_iter_minimal_model_candidates`).

    The canonical choice uses coprime integers with 1 < p < q, which matches
    the branch `b = i sqrt(p/q)` returned by `b_from_c` for these values.

    `max_n` bounds n = q - p; the canonical-form (p, q) of any minimal model
    satisfies n < q, so `max_n = 10000` covers every M(p, q) with q <= 10001.

    Exact real labels are classified by rational arithmetic by default;
    equivalent decimal, Decimal, Fraction and rational spellings agree, and
    nonzero offsets are retained. Existing mpmath values retain numerical
    recognition at the current precision, with default tolerance
    ``10**(-(mp.mp.dps - 10))`` on the inverse-solve's ``p_candidate``.
    An explicit positive tolerance opts into that approximate matching for
    any input type; zero requires exact matching.
    """
    from .numeric_input import _integer_control, exact_b_squared as rational_b_squared

    max_n = _integer_control('max_n', max_n)
    source, tolerance, specified = _recognizer_input(c, tolerance)
    if specified is not None and (tolerance is None or tolerance == 0):
        beta = rational_b_squared(specified)
        if beta is None or beta >= 0:
            return None
        p, q = abs(beta.numerator), beta.denominator
        return (p, q) if p > 1 and q - p <= max_n else None
    return _minimal_model_pq_numeric(source, max_n, tolerance)


def _minimal_model_pq_numeric(source, max_n, tolerance):
    c_val = mp.mpc(mp.convert(source))
    if abs(mp.im(c_val)) != 0:
        return None
    c_real = mp.re(c_val)
    if c_real >= 1:
        return None
    if tolerance is None:
        tolerance = mp.mpf(10) ** (-(mp.mp.dps - 10))
    ratio = mp.sqrt((mp.mpf(25) - c_real) / (mp.mpf(1) - c_real))
    half_ratio_minus_one = (ratio - 1) / 2
    for n in range(1, max_n + 1):
        p_candidate = n * half_ratio_minus_one
        p_round = mp.nint(p_candidate)
        if abs(p_candidate - p_round) > tolerance:
            continue
        p_int = int(p_round)
        if p_int < 2:
            continue
        q_int = p_int + n
        if gcd(p_int, q_int) != 1:
            continue
        return (p_int, q_int)
    return None


def unitary_minimal_model_m(c, *, max_m: int = 10000):
    """Return m <= max_m if c = 1 - 6/(m(m+1)) is a unitary minimal model."""
    from .numeric_input import _integer_control

    max_m = _integer_control('max_m', max_m)
    pq = minimal_model_pq(c, max_n=1 if max_m >= 2 else 0)
    if pq is None:
        return None
    p, q = pq
    if q == p + 1 and p <= max_m:
        return p
    return None


def positive_rational_collision_families(p: int, q: int, q_order: int) -> tuple[PoleCollision, ...]:
    """Exact collided Kac-pole families for positive rational b^2 = p/q.

    With the canonical branch b = sqrt(p/q), we have

        r b + s / b = (p r + q s) / sqrt(p q),

    so d_{r,s} depends only on p r + q s.
    """
    from .numeric_input import _integer_control

    p, q_order = _integer_control('p', p, 1), _integer_control('q_order', q_order)
    q = _integer_control('q', q, p)
    if gcd(p, q) != 1:
        raise ValueError("p and q must be coprime")

    groups: dict[int, list[tuple[int, int]]] = {}
    for rs in range(1, q_order + 1):
        for r in range(1, rs + 1):
            if rs % r != 0:
                continue
            s = rs // r
            label = p * r + q * s
            groups.setdefault(label, []).append((r, s))

    b = mp.sqrt(mp.mpf(p) / q)
    families = []
    for _, pairs in sorted(groups.items()):
        if len(pairs) < 2:
            continue
        pole = d_rs(pairs[0][0], pairs[0][1], b)
        families.append(PoleCollision(pairs=tuple(sorted(pairs)), pole=mp.mpc(pole)))
    return tuple(families)


def c25_collision_families(q_order: int) -> tuple[PoleCollision, ...]:
    """Exact collided Kac-pole families at c = 25, grouped by r+s."""
    return positive_rational_collision_families(1, 1, q_order)


def c1_collision_families(q_order: int) -> tuple[PoleCollision, ...]:
    """Exact collided Kac-pole families at c = 1, grouped by |r-s|.

    At c = 1 we have b = i, so

        d_{r,s}(1) = (r - s)^2 / 4.

    Therefore the collided target-pole classes are indexed exactly by
    `delta = |r-s|`, with no tolerance-based clustering needed.
    """
    from .numeric_input import _integer_control

    q_order = _integer_control('q_order', q_order)

    groups: dict[int, list[tuple[int, int]]] = {}
    for rs in range(1, q_order + 1):
        for r in range(1, rs + 1):
            if rs % r != 0:
                continue
            s = rs // r
            delta = abs(r - s)
            groups.setdefault(delta, []).append((r, s))

    families = []
    for delta, pairs in sorted(groups.items()):
        if len(pairs) < 2:
            continue
        families.append(
            PoleCollision(
                pairs=tuple(sorted(pairs)),
                pole=mp.mpc(mp.mpf(delta * delta) / 4),
            )
        )
    return tuple(families)


def minimal_model_collision_families(p: int, q: int, q_order: int) -> tuple[PoleCollision, ...]:
    """Exact collided Kac-pole families at c = 1 - 6 (p-q)^2 / (p q).

    With the canonical branch b = i sqrt(p/q), we have

        r b + s / b = i (p r - q s) / sqrt(p q),

    so d_{r,s} depends only on |p r - q s|.
    """
    from .numeric_input import _integer_control

    p, q_order = _integer_control('p', p, 2), _integer_control('q_order', q_order)
    q = _integer_control('q', q, p + 1)
    if gcd(p, q) != 1:
        raise ValueError("p and q must be coprime")

    return _negative_rational_collision_families(p, q, q_order)


def _negative_rational_collision_families(p: int, q: int, q_order: int) -> tuple[PoleCollision, ...]:
    """Integer collision families for negative b², including the p = 1 extensions."""
    groups: dict[int, list[tuple[int, int]]] = {}
    for rs in range(1, q_order + 1):
        for r in range(1, rs + 1):
            if rs % r != 0:
                continue
            s = rs // r
            label = abs(p * r - q * s)
            groups.setdefault(label, []).append((r, s))

    c_exact = mp.mpf(1) - mp.mpf(6) * (p - q) ** 2 / (p * q)
    b = b_from_c(c_exact)
    families = []
    for _, pairs in sorted(groups.items()):
        if len(pairs) < 2:
            continue
        pole = d_rs(pairs[0][0], pairs[0][1], b)
        families.append(PoleCollision(pairs=tuple(sorted(pairs)), pole=mp.mpc(pole)))
    return tuple(families)


def unitary_minimal_collision_families(m: int, q_order: int) -> tuple[PoleCollision, ...]:
    """Exact collided Kac-pole families at c = 1 - 6/(m(m+1)).

    For the unitary minimal series, choosing b = i sqrt(m/(m+1)) gives

        r b + s / b = i (m r - (m+1) s) / sqrt(m(m+1)),

    so d_{r,s} depends only on |m r - (m+1) s|.
    """
    from .numeric_input import _integer_control

    m = _integer_control('m', m, 2)
    return minimal_model_collision_families(m, m + 1, q_order)


def c1_A_rs_singular_factors(q_order: int) -> tuple[SingularFactor, ...]:
    """Exact A_rs singular factors at c = 1.

    At c = 1 we have b = i, so every denominator factor in A_rs becomes

        m b + n / b = i (m - n).

    Therefore the only exact zeros come from the diagonal factors m = n. The
    allowed diagonal values are the integers t in the overlap of the A_rs index
    ranges for m and n, excluding the omitted factors (0, 0) and (r, s).
    """
    from .numeric_input import _integer_control

    q_order = _integer_control('q_order', q_order)

    out: list[SingularFactor] = []
    for rs in range(1, q_order + 1):
        for r in range(1, rs + 1):
            if rs % r != 0:
                continue
            s = rs // r
            lower = max(1 - r, 1 - s)
            upper = min(r, s)
            for t in range(lower, upper + 1):
                if t == 0:
                    continue
                if (t, t) == (r, s):
                    continue
                out.append(
                    SingularFactor(
                        pair=(r, s),
                        factor=(t, t),
                        magnitude=mp.mpf(0),
                    )
                )
    return tuple(sorted(out, key=lambda item: (item.pair[0] * item.pair[1], item.pair, item.factor)))


def positive_rational_A_rs_singular_factors(p: int, q: int, q_order: int) -> tuple[SingularFactor, ...]:
    """Exact A_rs singular factors for positive rational b^2 = p/q.

    With b = sqrt(p/q), the denominator factors become

        u b + v / b = (p u + q v) / sqrt(p q).

    Therefore exact zeros occur when (u, v) = (q t, -p t), provided that the
    factor lies in the A_rs product range and is not one of the omitted terms.
    """
    from .numeric_input import _integer_control

    p, q_order = _integer_control('p', p, 1), _integer_control('q_order', q_order)
    q = _integer_control('q', q, p)
    if gcd(p, q) != 1:
        raise ValueError("p and q must be coprime")

    out: list[SingularFactor] = []
    for rs in range(1, q_order + 1):
        for r in range(1, rs + 1):
            if rs % r != 0:
                continue
            s = rs // r
            t_min = max(
                mp.ceil((1 - r) / mp.mpf(q)),
                mp.ceil(-s / mp.mpf(p)),
            )
            t_max = min(
                mp.floor(r / mp.mpf(q)),
                mp.floor((s - 1) / mp.mpf(p)),
            )
            for t in range(int(t_min), int(t_max) + 1):
                if t == 0:
                    continue
                factor = (q * t, -p * t)
                if factor == (r, s):
                    continue
                out.append(
                    SingularFactor(
                        pair=(r, s),
                        factor=factor,
                        magnitude=mp.mpf(0),
                    )
                )
    return tuple(sorted(out, key=lambda item: (item.pair[0] * item.pair[1], item.pair, item.factor)))


def minimal_model_A_rs_singular_factors(p: int, q: int, q_order: int) -> tuple[SingularFactor, ...]:
    """Exact A_rs singular factors at c = 1 - 6 (p-q)^2 / (p q).

    With b = i sqrt(p/q), the denominator factors become

        u b + v / b = i (p u - q v) / sqrt(p q).

    Therefore exact zeros occur when (u, v) = (q t, p t), provided that factor
    lies in the A_rs product range and is not one of the omitted terms.
    """
    from .numeric_input import _integer_control

    p, q_order = _integer_control('p', p, 2), _integer_control('q_order', q_order)
    q = _integer_control('q', q, p + 1)
    if gcd(p, q) != 1:
        raise ValueError("p and q must be coprime")

    return _negative_rational_A_rs_singular_factors(p, q, q_order)


def _negative_rational_A_rs_singular_factors(p: int, q: int, q_order: int) -> tuple[SingularFactor, ...]:
    """Integer A-factor zeros for negative b², including the p = 1 extensions."""
    out: list[SingularFactor] = []
    for rs in range(1, q_order + 1):
        for r in range(1, rs + 1):
            if rs % r != 0:
                continue
            s = rs // r
            t_min = max(
                mp.ceil((1 - r) / mp.mpf(q)),
                mp.ceil((1 - s) / mp.mpf(p)),
            )
            t_max = min(
                mp.floor(r / mp.mpf(q)),
                mp.floor(s / mp.mpf(p)),
            )
            for t in range(int(t_min), int(t_max) + 1):
                if t == 0:
                    continue
                factor = (q * t, p * t)
                if factor == (r, s):
                    continue
                out.append(
                    SingularFactor(
                        pair=(r, s),
                        factor=factor,
                        magnitude=mp.mpf(0),
                    )
                )
    return tuple(sorted(out, key=lambda item: (item.pair[0] * item.pair[1], item.pair, item.factor)))


def unitary_minimal_A_rs_singular_factors(m: int, q_order: int) -> tuple[SingularFactor, ...]:
    """Exact A_rs singular factors at c = 1 - 6/(m(m+1)).

    With b = i sqrt(m/(m+1)), the denominator factors become

        u b + v / b = i (m u - (m+1) v) / sqrt(m(m+1)).

    Therefore exact zeros occur when (u, v) = ((m+1) t, m t), provided that
    factor lies in the A_rs product range and is not one of the omitted terms.
    """
    from .numeric_input import _integer_control

    m = _integer_control('m', m, 2)
    return minimal_model_A_rs_singular_factors(m, m + 1, q_order)


def c25_A_rs_singular_factors(q_order: int) -> tuple[SingularFactor, ...]:
    """Exact A_rs singular factors at c = 25.

    At c = 25 we have b = 1, so every denominator factor in A_rs becomes m+n.
    Therefore the exact zeros come from anti-diagonal factors (t, -t) that lie
    in the A_rs product range and are not omitted.
    """
    return positive_rational_A_rs_singular_factors(1, 1, q_order)


def _default_tolerance(base_dps: int) -> mp.mpf:
    return mp.mpf(10) ** (-(max(base_dps, 20) // 2))


def _a_rs_zero_factors(r: int, s: int, b, tolerance) -> list[SingularFactor]:
    out: list[SingularFactor] = []
    inv_b = 1 / b
    for m in range(1 - r, r + 1):
        for n in range(1 - s, s + 1):
            if (m, n) == (0, 0) or (m, n) == (r, s):
                continue
            magnitude = abs(m * b + n * inv_b)
            if magnitude <= tolerance:
                out.append(
                    SingularFactor(
                        pair=(r, s),
                        factor=(m, n),
                        magnitude=mp.mpf(magnitude),
                    )
                )
    return out


def _exact_singularity_report(c, q_order: int, tolerance) -> RecursionSingularityReport | None:
    from .numeric_input import exact_real, exact_b_squared as rational_b_squared

    specified = exact_real(c)
    beta = rational_b_squared(specified) if specified is not None else None
    is_one = specified == 1 if specified is not None else _c_is_exact_one(c)
    is_twenty_five = specified == 25 if specified is not None else _c_is_exact_twenty_five(c)
    if is_one:
        exact_c = 1
        return RecursionSingularityReport(
            c=mp.mpc(exact_c),
            b=b_from_c(exact_c),
            q_order=q_order,
            tolerance=mp.mpf(tolerance),
            a_pole_factors=c1_A_rs_singular_factors(q_order),
            pole_collisions=c1_collision_families(q_order),
        )

    if is_twenty_five:
        exact_c = 25
        return RecursionSingularityReport(
            c=mp.mpc(exact_c),
            b=b_from_c(exact_c),
            q_order=q_order,
            tolerance=mp.mpf(tolerance),
            a_pole_factors=c25_A_rs_singular_factors(q_order),
            pole_collisions=c25_collision_families(q_order),
        )

    if specified is None:
        minimal_pq = minimal_model_pq(c)
        if minimal_pq is None and (unit_q := negative_unit_b2_q(c)) is not None:
            minimal_pq = (1, unit_q)
    elif beta is not None and beta < 0:
        minimal_pq = (abs(beta.numerator), beta.denominator)
    else:
        minimal_pq = None
    if minimal_pq is not None:
        p, q = minimal_pq
        exact_c = 1 - Fraction(6 * (p - q) ** 2, p * q)
        return RecursionSingularityReport(
            c=mp.mpc(mp.convert(exact_c)),
            b=b_from_c(exact_c),
            q_order=q_order,
            tolerance=mp.mpf(tolerance),
            a_pole_factors=_negative_rational_A_rs_singular_factors(p, q, q_order),
            pole_collisions=_negative_rational_collision_families(p, q, q_order),
        )

    if specified is None:
        positive_pq = positive_rational_b2_pq(c)
    elif beta is not None and beta > 0:
        positive_pq = (beta.numerator, beta.denominator)
    else:
        positive_pq = None
    if positive_pq is not None:
        p, q = positive_pq
        exact_c = 13 + 6 * (Fraction(p, q) + Fraction(q, p))
        return RecursionSingularityReport(
            c=mp.mpc(mp.convert(exact_c)),
            b=b_from_c(exact_c),
            q_order=q_order,
            tolerance=mp.mpf(tolerance),
            a_pole_factors=positive_rational_A_rs_singular_factors(p, q, q_order),
            pole_collisions=positive_rational_collision_families(p, q, q_order),
        )

    return None


def analyze_recursion_singularities(c, q_order: int, *, tolerance=None) -> RecursionSingularityReport:
    """Inspect the recursion data up to `q_order` for structural c-singularities.

    The report is intentionally universal: it depends only on `c` and `q_order`,
    not on the external weights or exchanged dimension. Those channel-dependent
    data decide whether the singularity is removable, but the dangerous places
    in the recursion are already visible here.

    Exact real labels follow the block engine's algebraic classification;
    deliberate offsets are retained. On the numerical scan, ``tolerance``
    controls which small factors and pole separations are reported as near
    zero. The returned ``b`` retains the charge from which it was constructed.
    """
    from .numeric_input import literal_number, _integer_control, parse_number, real_input_label

    q_order = _integer_control('q_order', q_order)
    c = parse_number(c)
    literal_number(c)  # (validates a stored native value: finite)

    base_dps = mp.mp.dps
    if tolerance is None:
        tolerance = _default_tolerance(base_dps)
    tolerance = real_input_label(tolerance, name='tolerance')
    if tolerance < 0:
        raise ValueError('tolerance must be >= 0')
    tolerance = mp.convert(tolerance)

    if (exact_report := _exact_singularity_report(c, q_order, tolerance)) is not None:
        return exact_report

    with mp.workdps(max(80, 2 * base_dps)):
        c_val = mp.mpc(mp.convert(c))
        b = b_from_c(c)

        a_poles: list[SingularFactor] = []
        poles: list[tuple[tuple[int, int], mp.mpc]] = []
        for r in range(1, q_order + 1):
            for s in range(1, q_order + 1):
                if r * s > q_order:
                    continue
                a_poles.extend(_a_rs_zero_factors(r, s, b, tolerance))
                poles.append(((r, s), mp.mpc(d_rs(r, s, b))))

        remaining = list(poles)
        collisions: list[PoleCollision] = []
        while remaining:
            pair, pole = remaining.pop(0)
            group_pairs = [pair]
            kept: list[tuple[tuple[int, int], mp.mpc]] = []
            for other_pair, other_pole in remaining:
                if abs(other_pole - pole) <= tolerance:
                    group_pairs.append(other_pair)
                else:
                    kept.append((other_pair, other_pole))
            remaining = kept
            if len(group_pairs) > 1:
                collisions.append(
                    PoleCollision(
                        pairs=tuple(sorted(group_pairs)),
                        pole=pole,
                    )
                )

    return RecursionSingularityReport(
        c=c_val,
        b=b,
        q_order=q_order,
        tolerance=mp.mpf(tolerance),
        a_pole_factors=tuple(sorted(a_poles, key=lambda item: (item.pair[0] * item.pair[1], item.pair, item.factor))),
        pole_collisions=tuple(sorted(collisions, key=lambda item: (min(r * s for r, s in item.pairs), item.pairs))),
    )
