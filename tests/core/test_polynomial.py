"""Tests for the Polynomial helper."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.polynomial import Polynomial


def _close(x, y, tol=mp.mpf("1e-40")):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def test_construction_and_degree():
    p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
    assert p.degree == 2
    assert p.coeffs[0] == mp.mpc(1)
    assert p.coeffs[2] == mp.mpc(3)


def test_trailing_zeros_stripped():
    p = Polynomial([1, 2, 0, 0])  # 1 + 2x
    assert p.degree == 1


def test_addition():
    a = Polynomial([1, 2, 3])
    b = Polynomial([4, 5])
    c = a + b
    assert c.degree == 2
    assert c.coeffs == [mp.mpc(5), mp.mpc(7), mp.mpc(3)]


def test_multiplication_with_self():
    # (1 + x)^2 = 1 + 2x + x^2
    p = Polynomial([1, 1])
    sq = p * p
    assert sq.coeffs == [mp.mpc(1), mp.mpc(2), mp.mpc(1)]


def test_evaluate():
    mp.mp.dps = 40
    # p(x) = 1 + 2x + 3x^2; p(2) = 1 + 4 + 12 = 17
    p = Polynomial([1, 2, 3])
    assert _close(p.evaluate(2), 17)


@pytest.mark.parametrize("field", [mp.mpf, mp.mpc])
def test_linear_composition_preserves_stored_coefficients_before_cancellation(field):
    with mp.workdps(100):
        offset = mp.mpc(mp.pi, mp.e) if field is mp.mpc else mp.pi
        p = Polynomial([mp.mpf("1e50") + offset, -mp.mpf("1e50")], field=field)
    with mp.workdps(40):
        translated = p.compose_linear(1, 1)
        assert translated.field is field and translated.degree == 1
        assert abs(translated.coeffs[0] - offset) < mp.mpf("1e-39")
        assert translated.coeffs[1] == -mp.mpf("1e50")


def test_x_minus_root():
    p = Polynomial.x_minus(mp.mpf("0.3"))
    assert p.degree == 1
    assert _close(p.evaluate(mp.mpf("0.3")), 0)


@pytest.mark.parametrize('kind', ['real', 'complex', 'promoted'])
def test_linear_factor_retains_a_stored_precise_root(kind):
    with mp.workdps(180):
        root = mp.mpf('.5') + mp.mpf('1e-100')
        if kind == 'complex':
            root = mp.mpc(root, '1e-110')
    with mp.workdps(30):
        factor = Polynomial.x_minus(root, field=mp.mpc if kind == 'promoted' else None)
        nearby = factor.evaluate(mp.mpf('.5'))
        assert mp.mp.dps == 30
    with mp.workdps(200):
        assert factor.evaluate(root) == 0
        expected = mp.mpf('.5') - root
        assert abs(nearby - expected) < mp.mpf('1e-28') * abs(expected)


def test_synthetic_division_roundtrip():
    # (x - 2)(x - 3)(x - 5) divided by (x - 3) gives (x - 2)(x - 5) = x^2 - 7x + 10
    p = Polynomial.x_minus(2) * Polynomial.x_minus(3) * Polynomial.x_minus(5)
    q = p.quotient_by_linear(3)
    assert q.degree == 2
    # q(x) = (x-2)(x-5) = x^2 - 7x + 10
    assert _close(q.coeffs[0], 10)
    assert _close(q.coeffs[1], -7)
    assert _close(q.coeffs[2], 1)


def test_pickle_roundtrip_preserves_field_and_coeffs():
    import pickle

    mp.mp.dps = 50
    # mpc field: the type object mp.mpc is not directly picklable, so the
    # __getstate__/__setstate__ hooks must serialize the field as a flag.
    p = Polynomial([mp.mpc("1.5", "0"), mp.mpc(2, 0), mp.mpc("0.25", "-1")])
    q = pickle.loads(pickle.dumps(p))
    assert q.field is mp.mpc
    assert q.coeffs == p.coeffs
    # mpf field round-trips to mpf, not promoted to mpc.
    r = Polynomial([mp.mpf("1.5"), mp.mpf(2)])
    s = pickle.loads(pickle.dumps(r))
    assert s.field is mp.mpf
    assert s.coeffs == r.coeffs


@pytest.mark.parametrize("field", [mp.mpf, mp.mpc])
@pytest.mark.parametrize("lazy", [False, True])
def test_coefficient_snapshots_cannot_change_cached_polynomials(field, lazy):
    import pickle

    with mp.workdps(40):
        p = Polynomial([1, 2], field=field)
        if lazy:
            p = p * Polynomial.one(field=field)
        assert p.evaluate(2) == 5
        snapshot = p.coeffs
        snapshot[0] = field(7)
        snapshot.append(field(10))
        assert p.coeffs == [field(1), field(2)]
        assert p.degree == 1 and p.evaluate(2) == 5
        assert (p * 3).evaluate(2) == 15
        assert p.compose_linear(2, 1).evaluate(2) == 11
        state = p.__getstate__()
        restored = Polynomial.__new__(Polynomial)
        restored.__setstate__(state)
        state["coeffs"].clear()
        assert restored == p
        assert pickle.loads(pickle.dumps(p)) == p


def test_quotient_by_large_root_is_numerically_stable():
    """Deflating a large root from a high-degree polynomial must stay accurate
    at low working precision. Plain forward synthetic division grows the quotient
    coefficients like |root|^k and loses ~k*log10(root) digits (this corrupted
    regularized minimal-model blocks when stripping high Kac-dimension poles at
    Delta ~ 100s); the composite (Wilkinson) deflation must recover the exact
    quotient -- verified by evaluating at both small and large points."""
    mp.mp.dps = 50
    # Degree-60 polynomial with one large root (500) and a spread of moderate
    # roots, mirroring a high-Kac-pole strip.
    qfac = Polynomial.x_minus(mp.mpf("0.3"))
    for k in range(1, 60):
        qfac = qfac * Polynomial.x_minus(mp.mpf(k) / 3)
    big = mp.mpf(500)
    p = Polynomial.x_minus(big) * qfac
    q = p.quotient_by_linear(big)
    assert q.degree == qfac.degree
    # Coefficients recovered exactly (forward-only would be O(1)-wrong at low order).
    worst = max(abs(a - b) for a, b in zip(q.coeffs, qfac.coeffs))
    scale = max(abs(b) for b in qfac.coeffs)
    assert worst / scale < mp.mpf(10) ** -40
    # Accurate evaluation at a small point (low-order coeffs dominate) AND a
    # large point (high-order dominate) -- the composite must win both.
    for x in [mp.mpf("0.7"), mp.mpf("1.3"), mp.mpf(480)]:
        ref = qfac.evaluate(x)
        assert abs(q.evaluate(x) - ref) < mp.mpf(10) ** -35 * max(mp.mpf(1), abs(ref))


def test_synthetic_division_with_complex_root():
    mp.mp.dps = 50
    root = mp.mpc(mp.mpf("0.3"), mp.mpf("0.7"))
    p = Polynomial.x_minus(root) * Polynomial([1, 2, 3])
    q = p.quotient_by_linear(root)
    # Quotient should equal Polynomial([1, 2, 3]).
    for i, expected in enumerate([1, 2, 3]):
        assert _close(q.coeffs[i], expected, tol=mp.mpf("1e-40"))


def test_quotient_by_linear_with_a_complex_root_outside_the_unit_disk():
    """The composite-deflation splice compared an mpc zero with a real ratio (TypeError) when both recurrences
    vanished at an index; Lagrange interpolation with complex nodes reached it."""
    from virasoro_blocks.interpolation import lagrange_interpolation_polynomial

    mp.mp.dps = 30
    poly = lagrange_interpolation_polynomial([mp.mpc(0, 1), mp.mpf(1)], [1, 2], shift=mp.mpc(0, 1))
    assert abs(poly.evaluate(mp.mpc(0, 1) - mp.mpc(0, 1)) - 1) < mp.mpf(10) ** -25
    assert abs(poly.evaluate(mp.mpf(1) - mp.mpc(0, 1)) - 2) < mp.mpf(10) ** -25
