"""Univariate chiral coordinate transforms and their expansions at z = 1/2.

The 2d block is built from two chiral copies; each coordinate option pairs a
chiral transform `u(z)` with its antichiral copy `u(zbar)`. The
parity-folded coordinates `xt`, `ws`, `qqb_ws` are downstream constructions
that combine `(u, ubar)` into ((u + ubar)/2 - u_sym, ((u - ubar)/2)**2); they
share the same chiral transforms.

Transforms exposed here:

    z          - identity. Used by `zzb` and the folded `xt` coordinate.
    y          - blocks_3d radial coordinate,
                             z(y) = (1 + y)^2 / (2 (1 + y^2)), y(1/2) = 0, y -> -y under
                             crossing z -> 1 - z. Used by `yyb` and the folded `ws`
                             coordinate.
    q          - elliptic nome. q(z) = exp(-pi K(1-z)/K(z)),
                             inverse z(q) = (theta_2(0,q) / theta_3(0,q))^4.
                             Used by `qqb` and `qqb_ws`. q_from_z lives in `recursion` for
                             historical reasons and is re-exported here.

Each transform u(z) has:
  - `u_of_z(z)`: forward
  - `z_of_u(u)`: inverse
  - `COORDINATES[name].u_symmetric()`: closed-form value of u(1/2) (`z_symmetric()`, `y_symmetric()`,
    `q_symmetric()` for the three chiral variables)
  - `u_series_around_half(n_terms)`: Taylor coefficients [c_0, ..., c_{n_terms-1}]
    of u(z) around z = 1/2 (so u(z) = sum_k c_k (z - 1/2)^k near z = 1/2).
  - `z_series_around_u_symmetric(n_terms)`: Taylor coefficients of z(u) around
    u = u(1/2) (so z(u) = sum_k c_k (u - u(1/2))^k).

These series are chain-rule data for derivative-basis conversions (used by the
tests; no production path consumes them).
"""

from __future__ import annotations

import mpmath as mp

from .recursion import q_from_z
from .numeric_input import _integer_control

def z_symmetric():
    """Crossing-symmetric point z(1/2) = 1/2."""
    return mp.mpf(1) / 2

def _coerce_keep_real(x):
    """Coerce x to mp.mpf if it's mathematically real, else to mp.mpc.

    Preserves the narrowest mpmath field so coordinate transforms applied at
    real points (z = 1/2 etc.) keep returning mp.mpf, which lets the
    downstream Taylor series stay in real arithmetic.
    """
    if isinstance(x, mp.mpf):
        return x
    if isinstance(x, mp.mpc):
        if x.imag == 0:
            return x.real
        return x
    if isinstance(x, complex):
        if x.imag == 0:
            return mp.mpf(x.real)
        return mp.mpc(x)
    return mp.mpf(mp.convert(x))    # via convert: a Fraction has no mpf()

def y_of_z(z):
    """blocks_3d radial coordinate. Exact real labels use the shared decimal parser."""
    from .numeric_input import parse_number

    z = _coerce_keep_real(parse_number(z))
    return (2 * z - 1) / (1 + 2 * mp.sqrt(z * (1 - z)))

def z_of_y(y):
    """Inverse radial map. The map has poles at y = +/-i."""
    from .numeric_input import parse_number

    y = _coerce_keep_real(parse_number(y))
    # At y=i+epsilon, forming 1+y**2 can lose epsilon**2, which changes the
    # O(1) imaginary part of a result of size 1/epsilon; factoring keeps both.
    denominator = (y-1j)*(y+1j) if isinstance(y, mp.mpc) else 1+y**2
    return (1 + y) ** 2 / (2 * denominator)

def y_symmetric():
    """The crossing-symmetric point is y(1/2) = 0."""
    return mp.mpf(0)

def q_of_z(z):
    """Elliptic nome q(z) = exp(-pi K(1-z) / K(z))."""
    return q_from_z(z)

def z_of_q(q):
    """Inverse elliptic nome, z(q) = (theta_2(0, q) / theta_3(0, q))^4.

    Near the real cusps theta functions converge slowly; there the modular
    transform q -> q_dual = exp(pi^2 / log q) is used instead.
    """
    from .numeric_input import literal_number

    q = _coerce_keep_real(literal_number(q))
    if abs(q) < mp.mpf('.25'):
        return (mp.jtheta(2, 0, q) / mp.jtheta(3, 0, q))**4
    # Keep the theta backend's existing domain restriction, including its error.
    if abs(q) > mp.mp.THETA_Q_LIM:
        raise ValueError('abs(q) > THETA_Q_LIM = %f' % mp.mp.THETA_Q_LIM)
    negative = mp.re(q) < 0
    p = -q if negative else q
    transformed_log = mp.pi**2 / mp.log(p)
    if mp.re(transformed_log) >= -10:
        return (mp.jtheta(2, 0, q) / mp.jtheta(3, 0, q))**4
    q_dual = mp.exp(transformed_log)
    complement = (mp.jtheta(2, 0, q_dual) / mp.jtheta(3, 0, q_dual))**4
    # lambda(-p) = lambda(p)/(lambda(p)-1).
    return 1 - 1/complement if negative else 1 - complement

def q_symmetric():
    """q(1/2) = exp(-pi)."""
    return mp.exp(-mp.pi)

def _check_n_terms(n_terms: int) -> int:
    return _integer_control('n_terms', n_terms, 1)

def y_series_around_half(n_terms: int):
    """Taylor coefficients of y(z) around z = 1/2, length n_terms.

    y(z) = sum_{k=0}^{n_terms - 1} c_k (z - 1/2)^k + O((z - 1/2)^{n_terms}).
    """
    n_terms = _check_n_terms(n_terms)
    return mp.taylor(y_of_z, z_symmetric(), n_terms - 1)

def z_series_around_y_symmetric(n_terms: int):
    """Taylor coefficients of z(y) around y = 0, length n_terms."""
    n_terms = _check_n_terms(n_terms)
    return mp.taylor(z_of_y, y_symmetric(), n_terms - 1)

def q_series_around_half(n_terms: int):
    """Taylor coefficients of q(z) around z = 1/2, length n_terms."""
    n_terms = _check_n_terms(n_terms)
    return mp.taylor(q_of_z, z_symmetric(), n_terms - 1)

def z_series_around_q_symmetric(n_terms: int):
    """Taylor coefficients of z(q) around q = exp(-pi), length n_terms."""
    n_terms = _check_n_terms(n_terms)
    return mp.taylor(z_of_q, q_symmetric(), n_terms - 1)

# --- canonical per-coordinate registry --------------------------------------
#
# Shared definitions of each coordinate's chiral variable, non-folded 2d
# parent, and chiral transform.

from collections import namedtuple

# chiral_var:   the chiral expansion variable z / y / q.
# base_2d:      the non-folded 2d coordinate a folded / radial coordinate
#               descends from (xt -> zzb), or the coordinate itself for the
#               non-folded 2d ones; None for the chiral-only coordinates.
# z_of_u:       inverse chiral transform u -> z.
# u_of_z:       forward chiral transform z -> u (recenters the expansion at an
#               arbitrary cross ratio; see derivs.evaluate_block).
# u_symmetric:  factory returning the symmetric expansion point u(1/2).
# kind:         "non_folded_2d" | "folded_2d" | "radial" | "chiral".
CoordinateSpec = namedtuple(
    "CoordinateSpec", "chiral_var base_2d z_of_u u_of_z u_symmetric kind"
)

_IDENTITY = lambda u: u  # noqa: E731  (z(u) = u for the z coordinate)

COORDINATES = {
    "zzb":           CoordinateSpec("z", "zzb", _IDENTITY, _IDENTITY, z_symmetric, "non_folded_2d"),
    "yyb":           CoordinateSpec("y", "yyb", z_of_y,    y_of_z,    y_symmetric, "non_folded_2d"),
    "qqb":           CoordinateSpec("q", "qqb", z_of_q,    q_of_z,    q_symmetric, "non_folded_2d"),
    "xt":            CoordinateSpec("z", "zzb", _IDENTITY, _IDENTITY, z_symmetric, "folded_2d"),
    "ws":            CoordinateSpec("y", "yyb", z_of_y,    y_of_z,    y_symmetric, "folded_2d"),
    "qqb_ws":        CoordinateSpec("q", "qqb", z_of_q,    q_of_z,    q_symmetric, "folded_2d"),
    "xt_radial":     CoordinateSpec("z", "zzb", _IDENTITY, _IDENTITY, z_symmetric, "radial"),
    "ws_radial":     CoordinateSpec("y", "yyb", z_of_y,    y_of_z,    y_symmetric, "radial"),
    "qqb_ws_radial": CoordinateSpec("q", "qqb", z_of_q,    q_of_z,    q_symmetric, "radial"),
    "z":             CoordinateSpec("z", None,  _IDENTITY, _IDENTITY, z_symmetric, "chiral"),
    "y":             CoordinateSpec("y", None,  z_of_y,    y_of_z,    y_symmetric, "chiral"),
    "q":             CoordinateSpec("q", None,  z_of_q,    q_of_z,    q_symmetric, "chiral"),
}

# folded / radial 2d coordinate -> its non-folded 2d parent (xt -> zzb).
FOLDED_BASE_2D = {
    coord: spec.base_2d
    for coord, spec in COORDINATES.items()
    if spec.kind in ("folded_2d", "radial")
}
