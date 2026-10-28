"""Validate the chiral coordinate transforms and their series expansions."""

from __future__ import annotations

import mpmath as mp
import pytest
from decimal import Decimal
from fractions import Fraction

from virasoro_blocks.coordinates import (
    q_of_z,
    q_series_around_half,
    q_symmetric,
    y_of_z,
    y_series_around_half,
    y_symmetric,
    z_of_q,
    z_of_y,
    z_series_around_q_symmetric,
    z_series_around_y_symmetric,
    z_symmetric,
)


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def test_y_symmetric_point():
    """blocks_3d's y satisfies y(1/2) = 0."""
    mp.mp.dps = 50
    assert _close(y_of_z(mp.mpf(1) / 2), y_symmetric(), tol=mp.mpf("1e-45"))
    assert _close(y_symmetric(), mp.mpf(0), tol=mp.mpf("1e-49"))


def test_y_linearizes_crossing():
    """Crossing z -> 1 - z is the reflection y -> -y."""
    mp.mp.dps = 50
    for y in [mp.mpf("-0.8"), mp.mpf("-0.2"), mp.mpf("0.0"), mp.mpf("0.3"), mp.mpf("0.75")]:
        assert _close(z_of_y(-y), 1 - z_of_y(y), tol=mp.mpf("1e-45"))
    for z in [mp.mpf("0.1"), mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.7"), mp.mpf("0.9")]:
        assert _close(y_of_z(1 - z), -y_of_z(z), tol=mp.mpf("1e-45"))


def test_q_symmetric_point():
    """q(1/2) = exp(-pi)."""
    mp.mp.dps = 50
    assert _close(q_of_z(mp.mpf(1) / 2), q_symmetric(), tol=mp.mpf("1e-40"))


def test_y_round_trip():
    """z_of_y(y_of_z(z)) = z and y_of_z(z_of_y(y)) = y for several values."""
    mp.mp.dps = 50
    for z in [mp.mpf("0.1"), mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.7"), mp.mpf("0.9")]:
        assert _close(z_of_y(y_of_z(z)), z, tol=mp.mpf("1e-45"))
    for y in [mp.mpf("-0.8"), mp.mpf("-0.2"), mp.mpf("0.0"), mp.mpf("0.35"), mp.mpf("0.8")]:
        assert _close(y_of_z(z_of_y(y)), y, tol=mp.mpf("1e-45"))
    # Coordinate helpers have the same decimal-label policy as block inputs.
    for transform in (y_of_z, z_of_y):
        expected = transform(Fraction(7, 10))
        for source in (.7, '.7', '7e-1', '7/10', Decimal('.7')):
            assert transform(source) == expected
        assert transform(.7000000000000001) != expected
        # Native complex values retain their supplied binary components.
        assert transform(complex(.7, 0)) == transform(Fraction.from_float(.7))


def test_q_round_trip():
    """z_of_q(q_of_z(z)) = z for a few values inside the convergence disk.

    The theta-function form of z_of_q converges fastest for small |q|, i.e.
    z near 0. Both formulas are valid on the entire unit interval of z.
    """
    mp.mp.dps = 50
    for z in [mp.mpf("0.2"), mp.mpf("0.4"), mp.mpf("0.5"), mp.mpf("0.6"), mp.mpf("0.8")]:
        assert _close(z_of_q(q_of_z(z)), z, tol=mp.mpf("1e-40"))


def test_y_series_zeroth_term_at_symmetric_point():
    """The 0-th Taylor coefficient at z = 1/2 is y(1/2) = 0."""
    mp.mp.dps = 40
    series = y_series_around_half(5)
    assert _close(series[0], y_symmetric(), tol=mp.mpf("1e-30"))


def test_q_series_zeroth_term_at_symmetric_point():
    mp.mp.dps = 40
    series = q_series_around_half(5)
    assert _close(series[0], q_symmetric(), tol=mp.mpf("1e-30"))


def test_y_series_reconstructs_y_near_half():
    """Evaluating the truncated series approximates y(z) near z = 1/2.

    y(z) has a branch point at z = 1, so the series around z = 1/2 has
    convergence radius 1/2. At delta = 0.05 with 10 terms the residual is
    O((delta / R)^10) = O(0.1^10) ~ 1e-10.
    """
    mp.mp.dps = 40
    n_terms = 10
    series = y_series_around_half(n_terms)
    z = mp.mpf(1) / 2 + mp.mpf("0.05")
    approx = mp.mpc(0)
    delta = z - mp.mpf(1) / 2
    for k, c in enumerate(series):
        approx += c * delta**k
    exact = y_of_z(z)
    assert _close(approx, exact, tol=mp.mpf("1e-9"))


def test_q_series_reconstructs_q_near_half():
    mp.mp.dps = 40
    n_terms = 10
    series = q_series_around_half(n_terms)
    z = mp.mpf(1) / 2 + mp.mpf("0.05")
    approx = mp.mpc(0)
    delta = z - mp.mpf(1) / 2
    for k, c in enumerate(series):
        approx += c * delta**k
    exact = q_of_z(z)
    assert _close(approx, exact, tol=mp.mpf("1e-12"))


def test_inverse_series_reconstructs_z():
    """The Taylor series of z(y) around y_symmetric reproduces z near 1/2."""
    mp.mp.dps = 40
    series = z_series_around_y_symmetric(10)
    y = y_symmetric() + mp.mpf("0.03")
    approx = mp.mpc(0)
    delta = y - y_symmetric()
    for k, c in enumerate(series):
        approx += c * delta**k
    assert _close(approx, z_of_y(y), tol=mp.mpf("1e-12"))


def test_z_of_q_series_reconstructs_z():
    mp.mp.dps = 40
    series = z_series_around_q_symmetric(10)
    q = q_symmetric() + mp.mpf("0.005")
    approx = mp.mpc(0)
    delta = q - q_symmetric()
    for k, c in enumerate(series):
        approx += c * delta**k
    assert _close(approx, z_of_q(q), tol=mp.mpf("1e-12"))


def test_y_first_derivative_at_half():
    """The k=1 Taylor coefficient is y'(1/2). Compute via finite difference."""
    mp.mp.dps = 60
    series = y_series_around_half(2)
    eps = mp.mpf("1e-25")
    fd = (y_of_z(mp.mpf(1) / 2 + eps) - y_of_z(mp.mpf(1) / 2 - eps)) / (2 * eps)
    assert _close(series[1], fd, tol=mp.mpf("1e-15"))


def test_series_validates_inputs():
    with pytest.raises(ValueError):
        y_series_around_half(0)
    with pytest.raises(ValueError):
        q_series_around_half(-1)


def test_z_symmetric_value():
    """z_symmetric() = 1/2."""
    mp.mp.dps = 40
    assert _close(z_symmetric(), mp.mpf(1) / 2, tol=mp.mpf("1e-39"))
