"""Minimal univariate polynomial over mpmath.

The mpmath library has `mp.polyval` for evaluation but no polynomial type that
supports the arithmetic we need (poly * poly, synthetic division by a linear
factor). sympy.Poly works but is slow for the inner loops of the recursion;
since we only need a handful of operations and the polynomial degrees here are
moderate (up to a few hundred), a hand-rolled class is the right tradeoff.

Coefficients are stored in ascending degree order:
    p(x) = coeffs[0] + coeffs[1] * x + coeffs[2] * x^2 + ...

Coefficients are mpmath numbers in a single field (`mp.mpf` or `mp.mpc`); type
conversions happen at construction time only. A polynomial's `field` is either
auto-detected from its coefficients (mpc if any coeff is mpc; mpf otherwise) or
explicitly given. Operations between polynomials promote to the wider field
(mpc dominates mpf). Pinning real-input polynomials to mp.mpf saves a factor
of ~3 in the inner mpmath multiply at high working precision.

Arithmetic dispatches to flint (arb_poly and acb_poly) for ~90x speedup in
multiplication and ~20x in linear composition; the flint polynomials are
cached per instance and precision, and rebuilt in each worker process.
"""

from __future__ import annotations

from typing import Iterable
from fractions import Fraction

import mpmath as mp


def _detect_field(values: Iterable) -> type:
    """Return mp.mpc if any value is an mpc; else mp.mpf."""
    for v in values:
        if isinstance(v, (mp.mpc, complex)):       # a Python complex too (mp.mpf(1j) raised)
            return mp.mpc
    return mp.mpf


def _mpf_to_arb(x):
    """Convert an mpmath mpf to a flint arb exactly, without a string round-trip.

    mpmath stores an mpf as the dyadic ``(-1)^sign * man * 2^exp`` in its
    ``_mpf_`` tuple, which arb reproduces exactly from a ``(fmpz(man), fmpz(exp))``
    pair. Rationals are divided in Arb; other real inputs are converted to mpf first.
    """
    from flint import arb, fmpz
    if isinstance(x, Fraction):
        return arb(x.numerator) / arb(x.denominator)
    if isinstance(x, int):
        return arb(x)
    mpf_tuple = getattr(x, "_mpf_", None)
    if mpf_tuple is None:
        mpf_tuple = mp.mpf(mp.convert(x))._mpf_
    sign, man, exp, _bc = mpf_tuple
    if not man and exp:
        # mpmath's nan / +-inf (a zero mantissa with a nonzero exponent) would otherwise become an exact 0
        raise ValueError(f"non-finite value {x} cannot enter the computation")
    # int(): under gmpy2 mpmath mantissas are gmpy2.mpz, which fmpz does not accept
    return arb((fmpz(-int(man) if sign else int(man)), fmpz(int(exp))))


def _arb_to_mpf(a):
    """Extract the midpoint of a flint arb as an mpmath mpf, without a string round-trip.

    ``arb._mpf_`` exposes the dyadic midpoint in mpmath's ``_mpf_`` format;
    ``mp.make_mpf`` reconstructs the value with no rounding or string parsing.
    """
    return mp.make_mpf(a._mpf_)


def _acb_to_mpc(c):
    """Extract the midpoint of a flint acb as an mpmath mpc, without string round-trips.

    Uses mp.make_mpc to skip the mpc normalization step that mp.mpc(re, im) applies.
    Always returns an mp.mpc -- callers (e.g. the field dispatch in _fold_parity)
    rely on a uniform table type, so we must NOT narrow to mp.mpf when the
    imaginary part happens to vanish for a given entry.
    """
    return mp.make_mpc((c.real._mpf_, c.imag._mpf_))


def _truncate_flint_poly(poly, length):
    """Keep coefficients below length on all supported python-flint versions."""
    truncate = getattr(poly, "truncate", None)
    if truncate is not None:
        return truncate(length)
    # python-flint 0.7 has coefficient access but no arb/acb_poly.truncate.
    # Keep ball coefficients directly, without an mpmath midpoint round-trip.
    return type(poly)([poly[i] for i in range(min(length, len(poly)))])


def _trimmed_flint_degree(flint_poly, to_scalar):
    """Degree of a flint arb_poly/acb_poly after trimming leading coefficients
    whose midpoint is zero.

    flint only strips coefficients that are the exact zero ball, so an interval
    cancellation can leave a midpoint-zero (nonzero-radius) leading coefficient
    that over-sizes ``flint_poly.degree()``. ``to_scalar`` (``_arb_to_mpf`` /
    ``_acb_to_mpc``) extracts the midpoint, matching the ``== 0`` trim the
    materialized mpmath coefficient list applies, so both degrees agree.
    """
    d = flint_poly.degree()
    while d > 0 and to_scalar(flint_poly[d]) == 0:
        d -= 1
    return d if d >= 0 else 0


def _mpf_to_acb(x):
    """Convert a numeric scalar to flint acb with minimal mpmath churn.

    Fast-paths the common mpf/mpc inputs to avoid constructing a temporary
    ``mp.mpc`` on every call in hot loops.
    """
    from flint import acb
    if isinstance(x, Fraction):
        return acb(_mpf_to_arb(x))
    if isinstance(x, mp.mpf):
        if x == 0:
            return acb(0)
        return acb(_mpf_to_arb(x))
    if isinstance(x, mp.mpc):
        if x.imag == 0:
            if x.real == 0:
                return acb(0)
            return acb(_mpf_to_arb(x.real))
        return acb(_mpf_to_arb(x.real), _mpf_to_arb(x.imag))
    if isinstance(x, int):
        return acb(x)
    x_mpc = mp.mpc(mp.convert(x))      # mp.convert takes a Fraction, which mp.mpc does not
    if x_mpc.imag == 0:
        if x_mpc.real == 0:
            return acb(0)
        return acb(_mpf_to_arb(x_mpc.real))
    return acb(_mpf_to_arb(x_mpc.real), _mpf_to_arb(x_mpc.imag))


def _promote(field_a: type, field_b: type) -> type:
    """mpc dominates mpf."""
    if field_a is mp.mpc or field_b is mp.mpc:
        return mp.mpc
    return mp.mpf


def _polynomial_input(value):
    """Normalize scalar labels before detecting their field or rounding them."""
    if isinstance(value, complex):
        from mpmath.libmp import from_float

        # Python complex components are supplied binary values. The ordinary
        # mpc constructor would round them when the caller uses fewer bits.
        return mp.make_mpc((from_float(value.real), from_float(value.imag)))
    if isinstance(value, (mp.mpf, mp.mpc, int, Fraction)):
        return value
    from .numeric_input import parse_number

    return parse_number(value)


def _coefficient_number(value, field):
    """Convert one coefficient into ``field`` at the working precision."""
    if isinstance(value, field):
        return value
    if field is mp.mpc and isinstance(value, mp.mpf):
        from mpmath.libmp import fzero

        return mp.make_mpc((value._mpf_, fzero))
    return field(mp.convert(value) if isinstance(value, Fraction) else value)


class Polynomial:
    # `_arb_cache` / `_acb_cache` lazily memoize the flint arb_poly / acb_poly
    # for this (immutable) polynomial, each as a `(prec_bits, flint_poly)` pair.
    # Rebuilding the flint polynomial on every evaluation is the dominant cost
    # when the same numerator is evaluated at many points (the removable-pole
    # stripping and the per-Delta interpolation sampling, especially in the
    # strip regime 1 < c < 25 where the table is mpc-typed and the batched
    # arb_poly path is skipped). The cache is keyed on the working precision so
    # a dps change rebuilds, and it is excluded from pickling (see __getstate__)
    # so each worker process rebuilds it -- preserving the original
    # determinism-under-multiprocessing guarantee.
    # `coeffs` is materialized lazily: a polynomial assembled from a flint
    # arb_poly/acb_poly (the block-numerator and flint-mul/compose paths) stores
    # ONLY the flint cache and leaves `_coeffs = None`; the mpmath coefficient
    # list is then converted out of the cache on first access (`coeffs` property).
    # Downstream consumers that stay in flint -- the batched Vandermonde block
    # sampler uses `_get_arb()` + `degree` only -- never trigger the conversion,
    # which was a large arb->mpf churn in the profile.
    __slots__ = ("_coeffs", "_field", "_arb_cache", "_acb_cache")

    def __init__(self, coeffs: Iterable, *, field: type | None = None):
        coeffs = [_polynomial_input(c) for c in coeffs]
        if field is None:
            field = _detect_field(coeffs) if coeffs else mp.mpc
        self._field = field
        self._coeffs = [_coefficient_number(c, field) for c in coeffs] if coeffs else [field(0)]
        # Strip trailing zeros so degree is well-defined.
        while len(self._coeffs) > 1 and self._coeffs[-1] == 0:
            self._coeffs.pop()
        self._arb_cache = None
        self._acb_cache = None

    @property
    def coeffs(self) -> list:
        """An owned coefficient snapshot; modifying it does not change this polynomial.

        Lazy (arb/acb-backed) coefficients are materialized on first access.
        Construct a new polynomial to change its coefficients.
        """
        return self._get_coeffs().copy()

    def _get_coeffs(self) -> list:
        """Read internal coefficients without copying or exposing them to callers."""
        c = self._coeffs
        if c is None:
            c = self._materialize_coeffs()
            self._coeffs = c
        return c

    def _materialize_coeffs(self) -> list:
        field = self._field
        if field is mp.mpf and self._arb_cache is not None:
            coeffs = [_arb_to_mpf(a) for a in self._arb_cache[1].coeffs()]
        elif self._acb_cache is not None:
            coeffs = [_acb_to_mpc(a) for a in self._acb_cache[1].coeffs()]
        elif self._arb_cache is not None:
            coeffs = [mp.mpc(_arb_to_mpf(a)) for a in self._arb_cache[1].coeffs()]
        else:  # pragma: no cover - lazy state always carries a flint cache
            return [field(0)]
        if not coeffs:
            return [field(0)]
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    @classmethod
    def _from_coeffs_trusted(cls, coeffs: list, *, field: type) -> "Polynomial":
        """Build directly from already-normalized mpf/mpc coefficients.

        Internal fast path for flint round-trips where every coefficient is
        already in the target mpmath field.
        """
        poly = object.__new__(cls)
        poly._field = field
        poly._coeffs = coeffs if coeffs else [field(0)]
        while len(poly._coeffs) > 1 and poly._coeffs[-1] == 0:
            poly._coeffs.pop()
        poly._arb_cache = None
        poly._acb_cache = None
        return poly

    @classmethod
    def _from_arb_trusted(cls, arb_poly_val, prec_bits: int) -> "Polynomial":
        """Build a real (mpf-field) polynomial backed by a flint ``arb_poly``,
        deferring the mpf coefficient conversion until first ``coeffs`` access.

        For the common downstream path (a flint-native Vandermonde / further
        flint arithmetic, which read ``_get_arb()`` and ``degree`` only) the mpf
        list is never built, eliminating the arb->mpf materialization churn.
        """
        poly = object.__new__(cls)
        poly._field = mp.mpf
        poly._coeffs = None
        poly._arb_cache = (prec_bits, arb_poly_val)
        poly._acb_cache = None
        return poly

    @classmethod
    def _from_acb_trusted(cls, acb_poly_val, prec_bits: int) -> "Polynomial":
        """Build a complex (mpc-field) polynomial backed by a flint ``acb_poly``,
        deferring the mpc coefficient conversion until first ``coeffs`` access.

        The acb-space analogue of ``_from_arb_trusted``: the folded scalar-G
        builder accumulates its numerators as ``acb_poly`` and the downstream
        flint Vandermonde sampler reads ``_get_acb()`` + ``degree`` only, so the
        acb->mpc materialization is pure churn that this defers.
        """
        poly = object.__new__(cls)
        poly._field = mp.mpc
        poly._coeffs = None
        poly._arb_cache = None
        poly._acb_cache = (prec_bits, acb_poly_val)
        return poly

    def __getstate__(self):
        # `_field` holds the mpmath type object (`mp.mpf` / `mp.mpc`), which is
        # not picklable (`mpmath.ctx_mp_python.mpf` is not importable by name).
        # Serialize it as a string flag so Polynomials cross process boundaries
        # (e.g. ProcessPoolExecutor in the parallel channel builders). The flint
        # caches are intentionally NOT serialized -- each process rebuilds them
        # lazily, so results stay deterministic under multiprocessing.
        return {
            "coeffs": self.coeffs,
            "field": "mpc" if self._field is mp.mpc else "mpf",
        }

    def __setstate__(self, state):
        self._field = mp.mpc if state["field"] == "mpc" else mp.mpf
        self._coeffs = list(state["coeffs"])
        self._arb_cache = None
        self._acb_cache = None

    @classmethod
    def zero(cls, *, field: type = mp.mpc) -> "Polynomial":
        return cls([field(0)], field=field)

    @classmethod
    def one(cls, *, field: type = mp.mpc) -> "Polynomial":
        return cls([field(1)], field=field)

    @classmethod
    def x_minus(cls, root, *, field: type | None = None) -> "Polynomial":
        """Return (x - root), retaining native mpmath root components."""
        from mpmath.libmp import mpf_neg, mpc_neg

        root = _polynomial_input(root)
        if field is None:
            field = _detect_field([root])
        root = _coefficient_number(root, field)
        # Negation changes a sign exactly. Ordinary mpmath negation would
        # round a stored high-precision root to the caller's current context.
        negative = (mp.make_mpf(mpf_neg(root._mpf_)) if field is mp.mpf
                    else mp.make_mpc(mpc_neg(root._mpc_)))
        return cls([negative, field(1)], field=field)

    @property
    def field(self) -> type:
        return self._field

    @property
    def degree(self) -> int:
        # Lazy polynomials read the degree off the flint cache so the mpf list is
        # not built just to size a Vandermonde. flint's arb_poly/acb_poly only
        # trims coefficients that are the *exact* zero ball, so an interval-
        # arithmetic cancellation (e.g. the parity '-' projected numerator
        # p_h p_hbar - p_hbar p_h, whose leading terms cancel to a midpoint-zero ball with a
        # nonzero radius) leaves an over-sized flint degree. Trim leading
        # coefficients whose midpoint is zero so the degree matches the
        # materialized mpf list (whose `== 0` trim drops exactly those terms).
        if self._coeffs is None:
            if self._arb_cache is not None:
                return _trimmed_flint_degree(self._arb_cache[1], _arb_to_mpf)
            if self._acb_cache is not None:
                return _trimmed_flint_degree(self._acb_cache[1], _acb_to_mpc)
        return len(self._get_coeffs()) - 1

    @property
    def is_zero(self) -> bool:
        """True only for the structural zero placeholder; never materializes.

        Used as a multiply-skip optimization (see ``_hseries_mul_flint``). A
        lazy flint-backed polynomial is a computed value -- never the
        ``Polynomial.zero()`` placeholder -- so we report ``False`` without
        forcing the mpf/mpc list to materialize. Erring toward ``False`` is the
        safe direction: a false "non-zero" verdict at worst costs a redundant
        multiply by a zero-valued operand (still correct), whereas a false
        "zero" verdict would wrongly drop a term.
        """
        coeffs = self._coeffs
        if coeffs is not None:
            return len(coeffs) == 1 and coeffs[0] == 0
        return False

    def __len__(self) -> int:
        return self.degree + 1

    def __repr__(self) -> str:
        return f"Polynomial(degree={self.degree}, field={self._field.__name__})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        if self.degree != other.degree:
            return False
        return all(a == b for a, b in zip(self._get_coeffs(), other._get_coeffs()))

    # ------------------------------------------------------------------
    # Cached flint representation
    # ------------------------------------------------------------------

    def _get_arb(self, prec_bits=None):
        """Build (and per-process cache) the arb_poly for these coefficients.

        Cached as `(prec_bits, arb_poly)`: a working-precision change rebuilds,
        and the cache is dropped on pickling (see __getstate__) so each worker
        process rebuilds lazily -- deterministic under multiprocessing.
        """
        prec_bits = _flint_prec_bits() if prec_bits is None else prec_bits
        from flint import arb, arb_poly, ctx as flint_ctx
        flint_ctx.prec = prec_bits
        cached = self._arb_cache
        if cached is not None and cached[0] == prec_bits:
            return cached[1]
        ap = arb_poly([_mpf_to_arb(c) for c in self._get_coeffs()])
        self._arb_cache = (prec_bits, ap)
        return ap

    def _get_acb(self, prec_bits=None):
        """Build (and per-process cache) the acb_poly. See _get_arb.

        When an arb_poly cache is already present, promotes it to acb_poly via
        the native flint cast (no mpf round-trip).
        """
        prec_bits = _flint_prec_bits() if prec_bits is None else prec_bits
        from flint import acb_poly, ctx as flint_ctx
        flint_ctx.prec = prec_bits
        cached = self._acb_cache
        if cached is not None and cached[0] == prec_bits:
            return cached[1]
        arb_cached = self._arb_cache
        if arb_cached is not None and arb_cached[0] == prec_bits:
            ap = acb_poly(arb_cached[1])
        else:
            ap = acb_poly([_mpf_to_acb(c) for c in self._get_coeffs()])
        self._acb_cache = (prec_bits, ap)
        return ap

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Polynomial") -> "Polynomial":
        field = _promote(self._field, other._field)
        a, b = self._get_coeffs(), other._get_coeffs()
        if len(a) < len(b):
            a, b = b, a
        out = [_coefficient_number(c, field) for c in a]
        for i, c in enumerate(b):
            out[i] += c
        return Polynomial(out, field=field)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        field = _promote(self._field, other._field)
        a, b = self._get_coeffs(), other._get_coeffs()
        out = [field(0)] * max(len(a), len(b))
        for i, c in enumerate(a):
            out[i] += c
        for i, c in enumerate(b):
            out[i] -= c
        return Polynomial(out, field=field)

    def __mul__(self, other) -> "Polynomial":
        if isinstance(other, Polynomial):
            field = _promote(self._field, other._field)
            if field is mp.mpf:
                return _flint_mul_real(self, other)
            return _flint_mul_complex(self, other)
        # Scalar multiplication.
        other = _polynomial_input(other)
        scalar_field = _detect_field([other])
        field = _promote(self._field, scalar_field)
        s = _coefficient_number(other, field)
        return Polynomial([s * c for c in self._get_coeffs()], field=field)

    __rmul__ = __mul__

    def scale(self, scalar) -> "Polynomial":
        """Return scalar * self."""
        return self * scalar

    def evaluate(self, x):
        """Horner evaluation at x.

        Runs in flint arb_poly/acb_poly (~50x faster than a pure-Python mpmath
        loop), at the working precision.
        """
        x = _polynomial_input(x)
        x_field = _detect_field([x])
        field = _promote(self._field, x_field)
        if field is mp.mpf:
            return _flint_evaluate_real(self, x)
        return _flint_evaluate_complex(self, x)

    def compose_linear(self, a, b) -> "Polynomial":
        """Return p(a*x + b), the composition with the linear map x -> a*x + b."""
        a, b = _polynomial_input(a), _polynomial_input(b)
        arg_field = _detect_field([a, b])
        field = _promote(self._field, arg_field)
        if field is mp.mpf:
            return _flint_compose_linear_real(self, a, b)
        return _flint_compose_linear_complex(self, a, b)

    def quotient_by_linear(self, root) -> "Polynomial":
        """Synthetic division: return self / (x - root) assuming root is a zero.

        If `root` is not actually a root, this returns the quotient and silently
        drops the remainder (i.e. it is the correct synthetic-division formula
        but the caller must guarantee that (x - root) divides `self`).

        Numerically stable for any root magnitude. Plain forward synthetic
        division (``q[i] = c[i+1] + root*q[i+1]``, descending) grows the quotient
        coefficients like ``|root|^k``: it keeps the *high*-order coefficients
        accurate but, for ``|root| > 1``, blows up the *low*-order ones
        (catastrophic when deflating high Kac-dimension poles at ``Delta ~ 100s``
        -- it corrupted regularized minimal-model blocks by O(1) factors). The
        reverse recurrence (``q[i] = (q[i-1] - c[i]) / root``) is the mirror:
        accurate low-order, inaccurate high-order. Since the deflated polynomial
        is later evaluated at both small ``Delta`` (point/bootstrap, low-order
        coefficients dominate) and large ``Delta`` (large-spin tower, high-order
        dominate), we need *all* coefficients accurate, so for ``|root| > 1`` we
        use the composite (Wilkinson) deflation: reverse for the low-order half,
        forward for the high-order half, spliced where the two agree best. Both
        directions give the same exact quotient when ``(x-root)`` divides
        ``self``; only the rounding differs.
        """
        root = _polynomial_input(root)
        root_field = _detect_field([root])
        field = _promote(self._field, root_field)
        root = _coefficient_number(root, field)
        # self.coeffs = [c0, c1, ..., cn], dividing by (x - root) gives a degree
        # n-1 quotient.
        n = self.degree
        if n < 1:
            raise ValueError("cannot divide a constant by (x - root)")
        c = self._get_coeffs()
        if abs(root) <= 1:
            # Forward synthetic division is stable for |root| <= 1.
            q = [field(0)] * n
            q[n - 1] = field(c[n])
            for i in range(n - 2, -1, -1):
                q[i] = c[i + 1] + root * q[i + 1]
            return Polynomial(q, field=field)
        # |root| > 1: composite deflation. Compute both recurrences ...
        fwd = [field(0)] * n
        fwd[n - 1] = field(c[n])
        for i in range(n - 2, -1, -1):
            fwd[i] = c[i + 1] + root * fwd[i + 1]
        rev = [field(0)] * n
        rev[0] = -c[0] / root
        for i in range(1, n):
            rev[i] = (rev[i - 1] - c[i]) / root
        # ... and splice at the index where they agree best (both accurate
        # there): reverse below it (accurate low-order), forward at and above
        # (accurate high-order).
        split = n - 1
        best = None
        for i in range(n):
            denom = abs(fwd[i]) + abs(rev[i])
            rel = abs(fwd[i] - rev[i]) / denom if denom != 0 else mp.mpf(0)      # real, also for an mpc field
            if best is None or rel < best:
                best, split = rel, i
        q = [rev[i] if i < split else fwd[i] for i in range(n)]
        return Polynomial(q, field=field)


# ---------------------------------------------------------------------------
# Helpers shared by the flint dispatch functions
# ---------------------------------------------------------------------------

def _flint_prec_bits() -> int:
    return int(mp.mp.dps * 3.33) + 20


def _poly_from_flint(flint_poly, field: type, prec_bits: int, poly_type: str = "auto") -> Polynomial:
    """Create a Polynomial from a flint arb_poly or acb_poly.

    No caching is performed; polynomials are created fresh from flint results
    to ensure deterministic behavior with multiprocessing.

    `poly_type` is "arb", "acb", or "auto" (infer from field).
    """
    if field is mp.mpf:
        coeffs = [_arb_to_mpf(c) for c in flint_poly.coeffs()]
    else:
        coeffs = [_acb_to_mpc(c) for c in flint_poly.coeffs()]
    return Polynomial._from_coeffs_trusted(coeffs, field=field)


# ---------------------------------------------------------------------------
# Flint-backed arithmetic
# ---------------------------------------------------------------------------

def _flint_mul_real(p: "Polynomial", q: "Polynomial") -> "Polynomial":
    """Multiply two real-coefficient polynomials using cached arb_poly."""
    from flint import ctx as flint_ctx
    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    result_arb = p._get_arb(prec_bits) * q._get_arb(prec_bits)
    # Lazy mpf: the arb_poly is the cache; defer the arb->mpf materialization so
    # a flint-chained consumer (compose, further mul, sampler) never pays it.
    return Polynomial._from_arb_trusted(result_arb, prec_bits)


def _flint_mul_complex(p: "Polynomial", q: "Polynomial") -> "Polynomial":
    """Multiply complex-coefficient polynomials using cached acb_poly."""
    from flint import ctx as flint_ctx
    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    result_acb = p._get_acb(prec_bits) * q._get_acb(prec_bits)
    return Polynomial._from_acb_trusted(result_acb, prec_bits)


def _flint_compose_linear_real(p: "Polynomial", a, b) -> "Polynomial":
    """Compute p(a*x + b) using cached arb_poly and a flint Horner loop."""
    from flint import arb, arb_poly, ctx as flint_ctx
    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    ap = p._get_arb(prec_bits)
    lin = arb_poly([_mpf_to_arb(b), _mpf_to_arb(a)])
    result = arb_poly([arb(0)])
    for c in reversed(list(ap.coeffs())):
        result = result * lin + arb_poly([c])
    # Lazy mpf: defer the arb->mpf materialization (see _flint_mul_real).
    return Polynomial._from_arb_trusted(result, prec_bits)


def _flint_compose_linear_complex(p: "Polynomial", a, b) -> "Polynomial":
    """Compute p(a*x + b) for a complex-coefficient polynomial using acb_poly."""
    from flint import arb, acb, acb_poly, ctx as flint_ctx
    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits

    ap = p._get_acb(prec_bits)
    lin = acb_poly([_mpf_to_acb(b), _mpf_to_acb(a)])
    result = acb_poly([acb(arb(0))])
    for c in reversed(list(ap.coeffs())):
        # These coefficients are already acb balls. A midpoint conversion
        # through mpmath discards stored digits and interval radii before
        # cancellation, unlike the real composition path.
        result = result * lin + acb_poly([c])
    # Lazy mpc: defer the acb->mpc materialization (see _flint_mul_complex).
    return Polynomial._from_acb_trusted(result, prec_bits)


def _flint_evaluate_real(p: "Polynomial", x) -> mp.mpf:
    """Evaluate a real polynomial at a real point using a cached arb_poly."""
    from flint import arb, ctx as flint_ctx
    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    ap = p._get_arb()
    x_arb = x if isinstance(x, arb) else _mpf_to_arb(x)
    return _arb_to_mpf(ap(x_arb))


def _flint_evaluate_complex(p: "Polynomial", x) -> mp.mpc:
    """Evaluate a polynomial at a complex point using a cached acb_poly."""
    from flint import ctx as flint_ctx
    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    ap = p._get_acb()
    return _acb_to_mpc(ap(_mpf_to_acb(x)))


def build_linear_product_flint(roots, field):
    """Build P(x) = ∏(x - root_j) using flint-native operations.

    Does all n-1 multiplications in flint space without converting back to
    Polynomial after each step, reducing conversion overhead from O(n) to O(1).
    """
    roots = [_polynomial_input(root) for root in roots]
    if len(roots) == 0:
        return Polynomial.one(field=field)
    prec_bits = _flint_prec_bits()
    if field is mp.mpf:
        from flint import arb, arb_poly, ctx as flint_ctx
        flint_ctx.prec = prec_bits
        # Start with (x - roots[0])
        product = arb_poly([-_mpf_to_arb(roots[0]), arb(1)])
        for root_j in roots[1:]:
            linear = arb_poly([-_mpf_to_arb(root_j), arb(1)])
            product = product * linear
        # Convert back to Polynomial once
        return _poly_from_flint(product, mp.mpf, prec_bits, poly_type="arb")
    else:
        from flint import arb, acb, acb_poly, ctx as flint_ctx
        flint_ctx.prec = prec_bits

        # Start with (x - roots[0])
        product = acb_poly([acb(0) - _mpf_to_acb(roots[0]), acb(arb(1))])
        for root_j in roots[1:]:
            linear = acb_poly([acb(0) - _mpf_to_acb(root_j), acb(arb(1))])
            product = product * linear
        # Convert back to Polynomial once
        return _poly_from_flint(product, mp.mpc, prec_bits, poly_type="acb")


def build_product_flint(polys, field):
    """Build ∏ poly_j using flint-native operations.

    Multiplies a list of Polynomials in flint space without converting
    back after each step.
    """
    if len(polys) == 0:
        return Polynomial.one(field=field)
    if len(polys) == 1:
        return polys[0]
    prec_bits = _flint_prec_bits()
    if field is mp.mpf:
        from flint import arb_poly, ctx as flint_ctx
        flint_ctx.prec = prec_bits
        # Convert first polynomial
        product = arb_poly([_mpf_to_arb(c) for c in polys[0]._get_coeffs()])
        for poly in polys[1:]:
            factor = arb_poly([_mpf_to_arb(c) for c in poly._get_coeffs()])
            product = product * factor
        return _poly_from_flint(product, mp.mpf, prec_bits, poly_type="arb")
    else:
        from flint import arb, acb, acb_poly, ctx as flint_ctx
        flint_ctx.prec = prec_bits

        # Convert first polynomial
        product = acb_poly([_mpf_to_acb(c) for c in polys[0]._get_coeffs()])
        for poly in polys[1:]:
            factor = acb_poly([_mpf_to_acb(c) for c in poly._get_coeffs()])
            product = product * factor
        return _poly_from_flint(product, mp.mpc, prec_bits, poly_type="acb")


def build_prefix_products_flint(polys, field):
    """Build prefix products [1, p0, p0*p1, p0*p1*p2, ...] in flint space.

    Returns a list of Polynomials where result[k] = ∏_{j<k} polys[j].
    """
    if len(polys) == 0:
        return [Polynomial.one(field=field)]
    prec_bits = _flint_prec_bits()
    if field is mp.mpf:
        from flint import arb, arb_poly, ctx as flint_ctx
        flint_ctx.prec = prec_bits

        result = [Polynomial.one(field=field)]
        product = arb_poly([arb(1)])  # Start with 1
        for poly in polys:
            factor = arb_poly([_mpf_to_arb(c) for c in poly._get_coeffs()])
            product = product * factor
            result.append(_poly_from_flint(product, mp.mpf, prec_bits, poly_type="arb"))
        return result
    else:
        from flint import arb, acb, acb_poly, ctx as flint_ctx
        flint_ctx.prec = prec_bits

        result = [Polynomial.one(field=field)]
        product = acb_poly([acb(arb(1))])  # Start with 1
        for poly in polys:
            factor = acb_poly([_mpf_to_acb(c) for c in poly._get_coeffs()])
            product = product * factor
            result.append(_poly_from_flint(product, mp.mpc, prec_bits, poly_type="acb"))
        return result


def real_factor_coeffs_from_conjugate_roots(roots, *, leading_coeff=None):
    """Real coefficients (low-order first) of ``leading_coeff * prod(x - root_i)``.

    For real ``c`` and real external weights the Kac Delta-poles are complex but
    come in conjugate pairs, so the denominator/factor polynomial is
    mathematically real. Built as a product of complex ``(x - root)`` factors it
    instead carries complex coefficients whose imaginary parts cancel only
    approximately, leaving a residue that contaminates the interpolated block.
    Rebuilding directly from the conjugate-symmetric roots in real arithmetic
    makes the polynomial exactly real:

      * a real root ``r`` contributes the linear factor ``(x - Re(r))``;
      * each complex root is matched with its *closest* conjugate partner to form
        the real quadratic ``(x - r)(x - conj r) = x^2 - 2 Re(r) x + |r|^2``.

    Returns ``None`` (so callers fall back to the complex-linear product) when a
    complex root has no conjugate partner within tolerance, or when
    ``leading_coeff`` itself has a non-negligible imaginary part -- i.e. when the
    polynomial is genuinely complex (a complex-``c`` block) and must stay so.
    With ``leading_coeff=None`` the product is monic. An empty ``roots`` yields
    the constant ``[leading_coeff]`` (or ``[1]`` when monic).
    """
    from flint import ctx as flint_ctx

    roots = [_coefficient_number(_polynomial_input(root), mp.mpc) for root in roots]
    if leading_coeff is not None:
        leading_coeff = _polynomial_input(leading_coeff)
    tol = mp.mpf(10) ** (-(mp.mp.dps // 2))
    before = flint_ctx.prec
    try:
        return _real_factor_coeffs(roots, leading_coeff, tol)
    finally:
        flint_ctx.prec = before


def _real_factor_coeffs(roots, leading_coeff, tol):
    used = [False] * len(roots)
    factors: list[list] = []  # each a real coefficient list, low-order first

    for i, root in enumerate(roots):
        if used[i]:
            continue
        if root.imag == 0:
            used[i] = True
            factors.append([-mp.mpf(mp.re(root)), mp.mpf(1)])
            continue

        partner = None
        best_dist = None
        target = mp.conj(root)
        for j, other in enumerate(roots):
            if used[j] or j == i or root.imag * other.imag >= 0:
                continue
            dist = abs(other - target)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                partner = j
        if partner is None or best_dist > tol * max(abs(root), mp.mpf(1)):
            if abs(root.imag) <= tol * max(abs(root.real), mp.mpf(1)):
                used[i] = True
                factors.append([-mp.mpf(root.real), mp.mpf(1)])
                continue
            return None

        used[i] = True
        used[partner] = True
        re = mp.mpf(mp.re(root))
        im = mp.mpf(mp.im(root))
        factors.append([re * re + im * im, -2 * re, mp.mpf(1)])

    if leading_coeff is None:
        coeffs = [mp.mpf(1)]
    else:
        lead = mp.mpc(leading_coeff)
        if abs(mp.im(lead)) > tol * max(abs(mp.re(lead)), mp.mpf(1)):
            return None
        coeffs = [mp.mpf(mp.re(lead))]

    from flint import arb_poly, ctx as flint_ctx
    flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
    poly = arb_poly([_mpf_to_arb(c) for c in coeffs])
    for factor in factors:
        poly = poly * arb_poly([_mpf_to_arb(c) for c in factor])
    return [_arb_to_mpf(c) for c in poly.coeffs()]
