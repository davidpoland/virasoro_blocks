"""Validate the polynomial-in-h representation against the numerical recursion."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyblock import H_q_series_rational
from virasoro_blocks.recursion import H_q_series, q_from_z


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


@pytest.mark.parametrize("q_order", [12])
def test_rational_matches_numerical_H(q_order):
    """For any h, evaluating the rational H matches the numerical recursion."""
    mp.mp.dps = 50
    c = mp.mpf("1.7")
    b = b_from_c(c)
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]

    rational = H_q_series_rational(d, b=b, q_order=q_order)

    z = mp.mpf("0.4")
    q = q_from_z(z)

    for h in [mp.mpf("0.5"), mp.mpf("1.4"), mp.mpf("3.2"), mp.mpf("10")]:
        coeffs = H_q_series(h, d, b=b, q_order=q_order)
        # Sum as a Horner-like polynomial in q.
        H_num = mp.mpc(0)
        for k in range(q_order, -1, -1):
            H_num = H_num * q + coeffs[k]

        H_rat = rational.evaluate(h, q)

        assert _close(H_num, H_rat, tol=mp.mpf("1e-30"))


def test_rational_denominator_roots():
    """The denominator factors as prod_{(r,s): rs <= q_order} (h - d_{r,s})."""
    mp.mp.dps = 60
    b = b_from_c(mp.mpf("1.7"))
    rational = H_q_series_rational(
        [mp.mpf("0.1")] * 4, b=b, q_order=6
    )
    # Each d_{r,s} is a root of the denominator polynomial. The polynomial has
    # degree ~14 here, so evaluation error scales roughly as that times machine
    # epsilon at the configured precision.
    for root in rational.denominator_roots:
        val = rational.denominator.evaluate(root)
        assert _close(val, mp.mpc(0), tol=mp.mpf("1e-45"))


def test_rational_N0_equals_D():
    """H_0(h) = 1, so N_0(h) should equal D(h)."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    rational = H_q_series_rational(
        [mp.mpf("0.1")] * 4, b=b, q_order=5
    )
    N0 = rational.numerators[0]
    D = rational.denominator
    assert N0.degree == D.degree
    for a, b_c in zip(N0.coeffs, D.coeffs):
        assert _close(a, b_c, tol=mp.mpf("1e-35"))


def test_numerator_degrees_are_bounded():
    """Each N_k has degree at most #pairs in the recursion."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    rational = H_q_series_rational(
        [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")],
        b=b,
        q_order=6,
    )
    npairs = len(rational.pairs)
    for k, N_k in enumerate(rational.numerators):
        # N_0 has degree = #pairs (it's D). For k >= 1, degree at most #pairs - 1.
        if k == 0:
            assert N_k.degree == npairs
        else:
            assert N_k.degree <= npairs - 1


