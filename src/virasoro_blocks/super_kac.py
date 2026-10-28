"""Exact NS Kac membership and input guards shared by the NS/R backends."""

from fractions import Fraction
from functools import wraps
from inspect import signature

import mpmath as mp

from .numeric_input import literal_components, parse_number


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _scale(a, b):
    return a[0]*b, a[1]*b


def _mul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def _beta_sum(c):
    real, imaginary = literal_components(c)
    return Fraction(5, 2)-real/3, -imaginary/3


def exact_affine_number(delta, c, *, delta_scale, charge_scale, constant=0):
    """Resolve a small momentum square before conversion can erase its offset."""
    real, imaginary = _exact_affine_parts(delta, c, delta_scale=delta_scale,
                                          charge_scale=charge_scale, constant=constant)
    return mp.convert(real) if imaginary == 0 else mp.mpc(mp.convert(real), mp.convert(imaginary))


def _exact_affine_parts(delta, c, *, delta_scale, charge_scale, constant):
    dr, di = literal_components(delta)
    cr, ci = literal_components(c)
    real = delta_scale*dr+charge_scale*cr+constant
    imaginary = delta_scale*di+charge_scale*ci
    return real, imaginary


def exact_affine_ball(delta, c, *, delta_scale, charge_scale, constant=0):
    """The exact affine input combination, enclosed without an mpmath conversion."""
    from flint import acb
    from .polynomial import _mpf_to_arb

    real, imaginary = _exact_affine_parts(delta, c, delta_scale=delta_scale,
                                          charge_scale=charge_scale, constant=constant)
    return acb(_mpf_to_arb(real), _mpf_to_arb(imaginary))


def _zero_at_beta(linear, constant, t, beta_squared):
    """Prove linear*x+constant=0 on the selected root of x²-Tx+1."""
    from .numeric_input import _zero_on_quadratic_root

    return _zero_on_quadratic_root(linear, constant, t, beta_squared)


def ns_fusion_zero(r, s, c, h_a, h_b, *, starred, beta_squared):
    """Exact fusion zeros, pairing (p,q) with (-p,-q) before square roots."""
    t = _beta_sum(c)
    a = _add(_scale(literal_components(h_a), 8), _add(t, (-2, 0)))
    b = _add(_scale(literal_components(h_b), 8), _add(t, (-2, 0)))
    target, seen = (0 if starred else 2), set()
    for p in range(1-r, r, 2):
        for q in range(1-s, s, 2):
            if (p+q-r-s) % 4 != target or (p, q) in seen:
                continue
            seen.update(((p, q), (-p, -q)))
            u = p*p-q*q
            v = _add(_scale(t, q*q), (-2*p*q, 0))
            w = _add(_add(b, v), _scale(a, -1))
            linear = _add(_scale(t, u*u), _scale(_add(w, _scale(b, -2)), 2*u))
            constant = _add(_add(_mul(w, w), _scale(_mul(b, v), -4)), (-u*u, 0))
            if _zero_at_beta(linear, constant, t, beta_squared):
                return True
    return False


def ramond_fusion_zero(r, s, c, h_lower, h_upper, *, sign, b, gamma_lower, gamma_upper):
    """Prove the squared fusion equation, then select its Ramond sign branch."""
    t = _beta_sum(c)
    charge = _scale(literal_components(c), Fraction(1, 24))
    lower = _add(charge, _scale(literal_components(h_lower), -1))
    upper = _add(charge, _scale(literal_components(h_upper), -1))
    seen = set()
    for p in range(1-r, r, 2):
        for q in range(1-s, s, 2):
            if (p, q) in seen:
                continue
            seen.update(((p, q), (-p, -q)))
            target = (p+q-r-s) % 4
            u = Fraction(q*q-p*p, 8)
            v = _add(_scale(t, Fraction(-q*q, 8)), (Fraction(p*q, 4), 0))
            w = _add(_add(lower, upper), _scale(v, -1))
            linear = _add(_scale(t, u*u), _scale(w, -2*u))
            constant = _add(_add(_mul(w, w), _scale(_mul(lower, upper), -4)), (-u*u, 0))
            if not _zero_at_beta(linear, constant, t, -b*b):
                continue
            signed_upper = gamma_upper if sign == 0 else -gamma_upper
            combination = gamma_lower-signed_upper if target == 2 else gamma_lower+signed_upper
            shift = (p*b+q/b)/(2*mp.sqrt(2))
            scale = max(abs(combination)**2, abs(shift)**2, abs(gamma_lower)**2, abs(gamma_upper)**2)
            if scale == 0 or abs(combination**2-shift**2) <= mp.mpf(10)**(-max(5, mp.mp.dps-8))*scale:
                return True
    return False


def matches_kac_weight(h, c, r, s, pole):
    """Whether the supplied components satisfy this NS Kac relation exactly.

    With x=beta**2, x+1/x=T=5/2-c/3 and
    8h+2(rs-1)=(r**2-1)x+(s**2-1)/x. Eliminating x gives a
    quadratic with rational complex coefficients. Numerical proximity only
    selects the (r,s) branch after exact membership has been established.
    Ramond-external blocks exchange the same NS modules.
    """
    probe = mp.convert(h)
    scale = max(1, abs(probe), abs(pole))
    if abs(probe-pole) > scale*mp.mpf(".01"):
        return False
    hr, hi = literal_components(h)
    cr, ci = literal_components(c)
    tr, ti = Fraction(5, 2)-cr/3, -ci/3
    er, ei = 8*hr+2*(r*s-1), 8*hi
    a, b = r*r-1, s*s-1
    real = er*er-ei*ei-(a+b)*(tr*er-ti*ei)+a*b*(tr*tr-ti*ti)+(a-b)**2
    imaginary = 2*er*ei-(a+b)*(tr*ei+ti*er)+2*a*b*tr*ti
    if real != 0 or imaginary != 0:
        return False
    return abs(probe-pole) <= mp.mpf(10)**(-max(5, mp.mp.dps-8))*scale


_SCALARS = frozenset({"delta", "h", "hbar", "z", "zbar", "c", "chiral_norm",
                      "z_shift", "zbar_shift", "center", "center_bar", "shift"})
_WEIGHTS = frozenset({"h_ext", "hbar_ext"})
_CUTOFFS = {"q_order": 0, "ns_order": 0, "r_order": 0,
            "m_max": 0, "n_max": 0, "n_terms": 1}


def _super_cutoff(name, value, *, minimum=0):
    """Normalize a discrete cutoff before caching or numerical construction."""
    from .requests import _integer_field

    value = _integer_field(name, value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def guard_super_inputs(family=None):
    """Parse literals and normalize discrete cutoffs before Kac arithmetic.

    Exact labels decide pole membership and fusion zeros; the numerical work
    runs at the caller's working precision.
    """
    def decorate(func):
        sig = signature(func)

        @wraps(func)
        def guarded(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            for name, value in tuple(bound.arguments.items()):
                if value is None:
                    continue
                if name in _CUTOFFS:
                    bound.arguments[name] = _super_cutoff(name, value, minimum=_CUTOFFS[name])
                elif name in _WEIGHTS:
                    bound.arguments[name] = [parse_number(v) for v in value]
                elif name in _SCALARS:
                    bound.arguments[name] = parse_number(value)
            return func(*bound.args, **bound.kwargs)

        return guarded
    return decorate
