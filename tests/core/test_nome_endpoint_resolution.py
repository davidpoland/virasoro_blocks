"""Endpoint conditioning must resolve the original point before conversion or caching."""
from decimal import Decimal
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import literal_number
from virasoro_blocks.recursion import q_from_z


def _point(source, side, exponent):
    value = Fraction(1) + side*Fraction(1, 10**exponent)
    if source == 'fraction':
        return value
    if source == 'exact_rational':
        return literal_number(value)
    if source == 'rational_text':
        return f'{value.numerator}/{value.denominator}'
    if source == 'decimal_text':
        return '0.' + '9'*exponent if side == -1 else '1.' + '0'*(exponent - 1) + '1'
    with mp.workdps(exponent + 100):
        value = mp.convert(literal_number(value))
        return value if source == 'mpf' else mp.mpc(value, 0)


def _reference(point):
    # Independent complete elliptic integrals, with enough precision for the
    # direct complement subtraction even in the longest endpoint tests.
    with mp.workdps(500):
        point = mp.convert(literal_number(point))
        value = mp.exp(-mp.pi*mp.ellipk(1 - point)/mp.ellipk(point))
        return value.real if mp.im(point) == 0 and mp.re(point) < 1 else value


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('precision', [15, 40])
def test_complex_endpoint_preserves_the_supplied_side(precision, side):
    with mp.workdps(160):
        point = mp.mpc(1, side*mp.mpf('1e-70'))
    expected = _reference(point)
    with mp.workdps(precision):
        actual = q_from_z(point)
        assert mp.sign(actual.imag) == side
        assert abs(actual - expected) < mp.mpf(10)**(-(precision - 1))*abs(expected)


@pytest.mark.parametrize('point', [complex(1 - 2**-52, 0), complex(1, 2**-60), complex(1, -2**-60)])
def test_native_complex_points_resolve_before_low_precision_conversion(point):
    expected = _reference(point)
    with mp.workdps(8):
        actual = q_from_z(point)
        assert abs(actual - expected) < mp.mpf('1e-7')*abs(expected)


@pytest.mark.parametrize('point', ['.7', '7e-1', '7/10', 0.7, Decimal('.7'), Fraction(7, 10)])
def test_real_nome_literals_follow_the_exact_decimal_policy(point):
    expected = _reference(Fraction(7, 10))
    with mp.workdps(40):
        assert abs(q_from_z(point) - expected) < mp.mpf('1e-39')*abs(expected)


@pytest.mark.parametrize('point', [0.7000000000000001, '.7000000000000001'])
def test_deliberate_last_decimal_shift_is_not_snapped_away(point):
    expected = _reference(Fraction(7, 10) + Fraction(1, 10**16))
    with mp.workdps(40):
        actual = q_from_z(point)
        assert actual != q_from_z(0.7)
        assert abs(actual - expected) < mp.mpf('1e-39')*abs(expected)


@pytest.mark.parametrize('sign', [-1, 1])
def test_small_origin_offsets_remain_nonzero_and_keep_the_real_nome(sign):
    point = sign*Fraction(1, 10**200)
    expected = _reference(point)
    with mp.workdps(40):
        actual = q_from_z(point)
        assert isinstance(actual, mp.mpf)
        assert mp.sign(actual) == sign
        assert abs(actual - expected) < mp.mpf('1e-39')*abs(expected)


