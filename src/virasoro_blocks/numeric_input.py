"""Exact parsing of real input literals, before numerical evaluation.

Decimal text denotes its written rational value. In particular, neither the
ambient mpmath precision nor a denominator limit participates in parsing.
"""

from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import numbers
import re

import mpmath as mp


_DECIMAL_COMPONENT = r"(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
_REAL_DECIMAL_LITERAL = re.compile(rf"[+-]?{_DECIMAL_COMPONENT}")
_COMPLEX_DECIMAL_LITERAL = re.compile(
    rf"(?:(?P<real>[+-]?{_DECIMAL_COMPONENT})(?P<imag>[+-]{_DECIMAL_COMPONENT})"
    rf"|(?P<pure>[+-]?{_DECIMAL_COMPONENT}))j"
)


def _zero_on_quadratic_root(linear, constant, total, root_hint):
    """Prove linear*x+constant=0 for the selected root of x²-total*x+1.

    All coefficients are exact rational complex component pairs. Numerical
    proximity selects a branch only after the polynomial identity is proved.
    """
    ar, ai = linear
    br, bi = constant
    if ar == ai == 0:
        return br == bi == 0
    norm = Fraction(ar*ar+ai*ai)
    xr, xi = -(br*ar+bi*ai)/norm, -(bi*ar-br*ai)/norm
    tr, ti = total
    if xr*xr-xi*xi-tr*xr+ti*xi+1 != 0 or 2*xr*xi-tr*xi-ti*xr != 0:
        return False
    hr, hi = literal_components(root_hint)
    distance = (xr-hr)**2+(xi-hi)**2
    other_distance = (tr-xr-hr)**2+(ti-xi-hi)**2
    return distance <= other_distance


class ExactRational(Fraction):
    """A Fraction also accepted by mpmath's mpf/mpc constructors.

    Rational arithmetic stays exact. Explicit mpmath conversion is the point
    where the current working precision is applied.
    """

    def __new__(cls, numerator=0, denominator=None):
        # Fraction can retain NumPy integer storage from a Rational input.
        # Normalize before Fraction performs any products or sign changes;
        # those must use unbounded Python integers, including with two args.
        if not (type(numerator) is int and
                (denominator is None or type(denominator) is int)):
            numerator = _python_rational(numerator)
            denominator = _python_rational(denominator)
        return super().__new__(cls, numerator, denominator)

    def _mpmath_(self, prec, rounding):
        from mpmath.libmp import from_rational

        return mp.make_mpf(from_rational(self.numerator, self.denominator, prec, rounding))

    def __add__(self, other):
        return _wrap(super().__add__(_operand(other)))

    def __radd__(self, other):
        return _wrap(super().__radd__(_operand(other)))

    def __sub__(self, other):
        return _wrap(super().__sub__(_operand(other)))

    def __rsub__(self, other):
        return _wrap(super().__rsub__(_operand(other)))

    def __mul__(self, other):
        return _wrap(super().__mul__(_operand(other)))

    def __rmul__(self, other):
        return _wrap(super().__rmul__(_operand(other)))

    def __truediv__(self, other):
        return _wrap(super().__truediv__(_operand(other)))

    def __rtruediv__(self, other):
        return _wrap(super().__rtruediv__(_operand(other)))

    def __pow__(self, other):
        other = _operand(other)
        if isinstance(other, (int, Fraction)) and other == int(other):
            return _wrap(super().__pow__(int(other)))
        return mp.power(mp.convert(self), mp.convert(other))

    def __rpow__(self, other):
        other = _operand(other)
        if self.denominator == 1 and isinstance(other, (int, Fraction)):
            return _wrap(Fraction(other) ** self.numerator)
        return mp.power(mp.convert(other), mp.convert(self))

    def __neg__(self):
        return ExactRational(-self.numerator, self.denominator)

    def __pos__(self):
        return self

    def __abs__(self):
        return ExactRational(abs(self.numerator), self.denominator)


class _ComplexLiteral(mp.mpc):
    """A numerically parsed complex input whose nonzero imaginary part is deliberate.

    The type survives copying and pickling through mpmath's native state
    protocol. Arithmetic produces ordinary numerical values, as with mpmath.
    """


def _complex_charge_input(value):
    """Normalize a requested complex charge without interpreting its bits as roundoff."""
    if _explicit_nonreal_input(value):
        return value
    number = value if isinstance(value, mp.mpc) else mp.mpc(value)
    if number.imag == 0:
        return literal_number(number.real)
    result = object.__new__(_ComplexLiteral)
    result._mpc_ = number._mpc_
    return result


def _explicit_nonreal_input(value):
    return isinstance(value, _ComplexLiteral) and value.imag != 0


def _operand(value):
    return ExactRational(str(value)) if isinstance(value, float) else _python_rational(value)


def _python_rational(value):
    """Retain a rational's value with unbounded integer component storage."""
    if value is None or type(value) is int:
        return value
    if isinstance(value, Fraction) or isinstance(value, numbers.Rational):
        numerator, denominator = value.numerator, value.denominator
        if type(numerator) is int and type(denominator) is int:
            return value
        return Fraction(int(numerator), int(denominator))
    return value


def _wrap(value):
    return ExactRational(value) if isinstance(value, Fraction) else value


def _exchange_weight(delta, spin):
    """Derive (Delta + spin)/2 without rounding an exchange label.

    Real labels use rational arithmetic. Existing complex mpmath inputs carry
    exact binary components; shifting by a half-integer and halving those raw
    components preserves them even under a lower ambient precision.
    """
    if isinstance(delta, mp.mpc):
        from mpmath.libmp import from_int, mpf_add, mpf_shift

        real_spin, imaginary_spin = literal_components(spin)
        twice_spin = 2 * real_spin
        if imaginary_spin == 0 and twice_spin == int(twice_spin):
            real, imag = delta._mpc_
            spin_tuple = mpf_shift(from_int(int(twice_spin)), -1)
            return mp.make_mpc((mpf_shift(mpf_add(real, spin_tuple), -1),
                                mpf_shift(imag, -1)))
    return (delta + spin) / 2


def _signed_half_integer(value):
    """Validate a signed spin from its supplied value, without snapping it."""
    import numpy as np
    from numbers import Integral

    message = "ell must be a finite real integer or half-integer"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message + ", not a boolean")
    try:
        # Spin text follows the real decimal/rational CLI syntax. Parsing
        # complex text numerically could round its real part onto the lattice.
        if isinstance(value, str) and "j" in value.lower():
            raise ValueError(message)
        if isinstance(value, Integral):
            return ExactRational(int(value))
        real, imaginary = literal_components(value)
        real = ExactRational(real)
        if imaginary != 0 or real.denominator not in (1, 2):
            raise ValueError(message)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(message) from exc
    return real


def _integer_spin(value):
    """The signed integer spin needed by the scalar G parity factor."""
    spin = _signed_half_integer(value)
    if spin.denominator != 1:
        raise ValueError("ell must be an integer (parity matters)")
    return int(spin)


def _integer_control(name, value, minimum=0):
    """Validate a discrete numerical control before conversions, cache keys, or construction."""
    from .requests import _integer_field

    value = _integer_field(name, value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _parse_real_literal(token):
    # Fraction constructs 10**exponent even when the numerator is zero.
    # Zero scientific notation needs no large integer allocation. Validate the
    # whole decimal spelling before taking this shortcut, including its exponent.
    if 'e' in token or 'E' in token:
        normalized = token.lower()
        if (_REAL_DECIMAL_LITERAL.fullmatch(normalized) is not None
                and Decimal(normalized.split('e', 1)[0]).is_zero()):
            return ExactRational(0)
    return ExactRational(token)


def parse_number(value):
    """Keep exact real inputs exact; accept the existing complex-number syntax.

    ``.7``, ``7e-1`` and ``7/10`` all denote exactly seven tenths. Python
    floats use their shortest decimal spelling; other native real scalars
    with a binary ratio retain that ratio. Mpmath values pass through.
    """
    if isinstance(value, (int, Fraction, numbers.Rational)) and not isinstance(value, bool):
        return ExactRational(value)
    if isinstance(value, float):
        return ExactRational(str(value))
    if isinstance(value, Decimal):
        return ExactRational(value)
    if not isinstance(value, bool) and isinstance(value, numbers.Real) and hasattr(value, "as_integer_ratio"):
        # NumPy's narrower/extended floats have no Python-float spelling
        # policy. Normalize their stored bits before arithmetic or caching.
        return ExactRational(*value.as_integer_ratio())
    if isinstance(value, str):
        token = value.strip()
        if "j" not in token.lower():
            parts = token.split("/")
            if len(parts) == 1:
                return _parse_real_literal(parts[0])
            if len(parts) == 2:
                return _parse_real_literal(parts[0].strip()) / _parse_real_literal(parts[1].strip())
            raise ValueError(f"invalid numeric literal {value!r}")
        normalized = token.lower().replace(' ', '')
        if normalized.startswith('(') and normalized.endswith(')'):
            normalized = normalized[1:-1]
        match = _COMPLEX_DECIMAL_LITERAL.fullmatch(normalized)
        if match is None:
            raise ValueError(f"invalid numeric literal {value!r}")
        imaginary = match['imag'] if match['real'] is not None else match['pure']
        real = match['real'] or '0'
        # Only the mantissa decides zero; nonzero complex components remain
        # numerical without allocating exact powers of ten for huge exponents.
        if Decimal(imaginary.split('e', 1)[0]).is_zero():
            return _parse_real_literal(real)
        # Parse the validated components directly. Mpmath's complex fallback
        # accepts unbalanced parentheses and backtracks excessively on long
        # malformed input. Its real parser also mishandles zero scientific
        # notation in some spellings, so construct that component directly.
        real_value = mp.mpf(0) if Decimal(real.split('e', 1)[0]).is_zero() else mp.mpf(real)
        result = _ComplexLiteral(real_value, mp.mpf(imaginary))
        if not mp.isfinite(result):
            raise ValueError("numeric inputs must be finite")
        return result
    return value


def exact_real(value):
    """The rational denoted by a real literal, or None for mpmath values."""
    value = parse_number(value)
    return Fraction(value) if isinstance(value, (int, Fraction)) else None


def rational_sqrt(value):
    """The exact nonnegative rational square root, or None."""
    from math import isqrt

    if value < 0:
        return None
    p, q = isqrt(value.numerator), isqrt(value.denominator)
    return Fraction(p, q) if p*p == value.numerator and q*q == value.denominator else None


def exact_b_squared(c, *, superconformal=False):
    """Solve the quadratic for a rational b² using integer square roots."""
    c = exact_real(c)
    if c is None:
        return None
    # Virasoro: c = 13 + 6(x + 1/x).
    # N=1: c = 15/2 - 3(x + 1/x), with x = beta².
    total = (Fraction(15, 2) - c) / 3 if superconformal else (c - 13) / 6
    root = rational_sqrt(total*total - 4)
    if root is None:
        return None
    roots = ((total + root) / 2, (total - root) / 2)
    # Match the numerical parametrizations: b_from_c uses |b| <= 1,
    # whereas beta_from_c uses |beta| >= 1. The reciprocal branch gives
    # the same c but swaps r and s in the Kac data.
    return (max if superconformal else min)(roots, key=abs)


@lru_cache(maxsize=512)
def _mpf_literal(raw):
    """The immutable rational label of stored binary components, independent of precision."""
    if not mp.isfinite(mp.make_mpf(raw)):
        raise ValueError("numeric inputs must be finite")
    sign, mantissa, exponent, _ = raw
    numerator = -int(mantissa) if sign else int(mantissa)
    return ExactRational(numerator << exponent) if exponent >= 0 else ExactRational(numerator, 1 << -exponent)


def literal_number(value):
    """Normalize a public input without guessing a floating-point value's intent.

    A Python float denotes its shortest round-trip decimal spelling: 0.7
    means 7/10, while 0.7000000000000001 keeps its last decimal place. An
    existing mpf has no source spelling and retains its supplied binary value.
    """
    if isinstance(value, mp.mpf):
        return _mpf_literal(value._mpf_)
    value = parse_number(value)
    if isinstance(value, (complex, mp.mpc)) and not mp.isfinite(value):
        raise ValueError("numeric inputs must be finite")
    return value


def literal_components(value):
    """Exact real and imaginary labels, without mixed-type equality rounding."""
    value = literal_number(value)
    if isinstance(value, complex):
        return ExactRational(Fraction.from_float(value.real)), ExactRational(Fraction.from_float(value.imag))
    if isinstance(value, mp.mpc):
        return literal_number(value.real), literal_number(value.imag)
    return value, ExactRational(0)


def _derived_label(real, imaginary, sources):
    """Materialize exact component arithmetic while retaining native input types."""
    complex_input = any(isinstance(value, (complex, mp.mpc)) for value in sources)
    if not complex_input and all(isinstance(value, (int, Fraction)) for value in sources):
        return real
    components = (real, imaginary)
    if all(value.denominator & (value.denominator - 1) == 0 for value in components):
        from mpmath.libmp import from_man_exp

        # Binary labels and integer/half-integer shifts have an exact binary
        # sum. Construct it directly; ambient mpmath arithmetic would round it.
        raw = tuple(from_man_exp(value.numerator, 1 - value.denominator.bit_length())
                    for value in components)
        result = mp.make_mpc(raw) if complex_input else mp.make_mpf(raw[0])
    else:
        # A non-dyadic rational shift: convert the exact sum once, at the working precision.
        result = mp.mpc(mp.convert(real), mp.convert(imaginary)) if complex_input else mp.convert(real)
    if complex_input and any(_explicit_nonreal_input(value) for value in sources) and result.imag != 0:
        return _complex_charge_input(result)
    return result


def _label_sum(*values):
    """Add input labels before numerical rounding, including stored native bits."""
    values = tuple(parse_number(value) for value in values)
    if all(isinstance(value, (int, Fraction)) for value in values):
        return sum(values, ExactRational(0))
    nonzero = tuple(value for value in values if value != 0)
    if len(nonzero) == 1:
        return nonzero[0]
    components = tuple(literal_components(value) for value in values)
    real = sum((parts[0] for parts in components), ExactRational(0))
    imaginary = sum((parts[1] for parts in components), ExactRational(0))
    return _derived_label(real, imaginary, values)


def _label_scale(value, factor):
    """Scale an input label by an exact real rational without rounding its bits."""
    value, factor = parse_number(value), literal_number(factor)
    real, imaginary = literal_components(value)
    return _derived_label(real * factor, imaginary * factor, (value, factor))


def real_input_label(value, *, name):
    """Validate a finite real input while retaining its supplied exact label."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, not a boolean")
    try:
        label, imaginary = literal_components(value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if imaginary != 0:
        raise ValueError(f"{name} must be real")
    if isinstance(label, numbers.Integral):
        return ExactRational(int(label))
    if isinstance(label, numbers.Rational):
        return label
    try:
        number = mp.convert(label)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not mp.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if mp.im(number) != 0:
        raise ValueError(f"{name} must be real")
    return label


def _evaluation_number(value):
    """Apply the input-label policy and retain real arithmetic when possible."""
    if isinstance(value, mp.mpf):
        if not mp.isfinite(value):
            raise ValueError("numeric inputs must be finite")
        return value
    if isinstance(value, mp.mpc):
        if not mp.isfinite(value):
            raise ValueError("numeric inputs must be finite")
        return value.real if value.imag == 0 else value
    value = mp.convert(literal_number(value))
    return value.real if isinstance(value, mp.mpc) and value.imag == 0 else value


def _numeric_input_values(values):
    """Convert input labels at the working precision; mpmath values pass through."""
    return tuple(_evaluation_number(parse_number(value)) for value in values)


@lru_cache(maxsize=128)
def _evaluation_pole_probes(poles):
    probes = []
    for pole in poles:
        try:
            probes.append(complex(pole))
        except (TypeError, ValueError, OverflowError):
            probes.append(None)
    return tuple(probes)
