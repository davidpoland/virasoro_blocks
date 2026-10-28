"""Standard rational inputs work consistently across polynomial operations."""

from fractions import Fraction as F

import mpmath as mp
import pytest

from virasoro_blocks.polynomial import Polynomial


@pytest.mark.parametrize("field", [None, mp.mpf, mp.mpc])
def test_fraction_coefficients_use_current_precision(field):
    with mp.workdps(60):
        p = Polynomial([F(1, 3), F(2, 7)], field=field)
        for x in (2, F(1, 3)):
            expected = mp.convert(F(1, 3) + F(2, 7) * x)
            assert abs(p.evaluate(x) - expected) < mp.mpf('1e-58')


@pytest.mark.parametrize("complex_field", [False, True])
def test_fraction_scalar_multiplication(complex_field):
    p = Polynomial([1, 2j] if complex_field else [1, 2])
    with mp.workdps(50):
        for result in [p * F(1, 3), F(1, 3) * p]:
            assert mp.almosteq(result.evaluate(2), p.evaluate(2) / 3)


@pytest.mark.parametrize("field", [None, mp.mpf, mp.mpc])
def test_fraction_root_linear_constructor(field):
    with mp.workdps(50):
        p = Polynomial.x_minus(F(1, 3), field=field)
        assert p.evaluate(mp.mpf(1)/3) == 0


@pytest.mark.parametrize("root", [F(1, 2), F(5, 2)])
@pytest.mark.parametrize("field", [mp.mpf, mp.mpc])
def test_fraction_root_synthetic_division(root, field):
    p = Polynomial([-mp.convert(root), 1], field=field) * Polynomial([2, 3], field=field)
    quotient = p.quotient_by_linear(root)
    assert mp.almosteq(quotient.evaluate(2), 8)


def test_polynomial_scalar_labels_share_the_decimal_policy_and_detect_complex_text():
    from decimal import Decimal

    with mp.workdps(40):
        identity = Polynomial([0, 1], field=mp.mpf)
        reference = Polynomial.x_minus(F(7, 10))
        translated_reference = identity.compose_linear(1, F(7, 10))
        for label in (.7, '.7', '7/10', Decimal('.7'), F(7, 10)):
            assert Polynomial.x_minus(label) == reference
            assert Polynomial([label, 1]) == Polynomial([F(7, 10), 1])
            assert Polynomial.one(field=mp.mpf).scale(label) == Polynomial([F(7, 10)])
            assert identity.compose_linear(1, label) == translated_reference
            product = reference*Polynomial([2, 3])
            quotient = product.quotient_by_linear(label)
            assert abs(quotient.evaluate(2)-8) < mp.mpf('1e-38')
        assert Polynomial(['1+2j', 1]).field is mp.mpc
        assert identity.compose_linear(1, '1+2j').evaluate(0) == mp.mpc(1, 2)
    with mp.workdps(10):
        query = complex(1+1e-15, 1e-20)
        result = Polynomial([-1, 1]).evaluate(query)
        with mp.workdps(60):
            expected = mp.mpc(query)-1
            assert abs(result.real/expected.real-1) < mp.mpf('1e-8')
            assert abs(result.imag/expected.imag-1) < mp.mpf('1e-8')


