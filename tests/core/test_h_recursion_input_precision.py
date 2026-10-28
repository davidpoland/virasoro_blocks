"""Primitive H APIs must distinguish exact external labels before rounding."""
from decimal import Decimal
from fractions import Fraction as F

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import literal_components, _numeric_input_values
from virasoro_blocks.params import b_from_c, c_origin_of_b
from virasoro_blocks.polyblock import H_q_series_rational, chiral_denominator_roots, HRational
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.recursion import H_q_series, _scalar_cache_key


def _externals(source, exponent):
    ext = (F(1, 10), F(1, 10)+F(1, 10**exponent), F(3, 10), F(2, 5))
    if source == 'text':
        return tuple(f'{value.numerator}/{value.denominator}' for value in ext)
    if source == 'fraction':
        return ext
    with mp.workdps(exponent+160):
        numeric = tuple(mp.convert(value) for value in ext)
        if source == 'mpc':
            return tuple(mp.mpc(value) for value in numeric)
        if source == 'mixed':
            return (ext[0], numeric[1], numeric[2], ext[3])
        return numeric


def _reference_h1(ext, h):
    # Independent level-one descendant coefficient, not a recursion oracle.
    with mp.workdps(500):
        parts = tuple(literal_components(value) for value in ext)
        numbers = tuple(mp.mpc(mp.convert(re), mp.convert(im)) for re, im in parts)
        return 8*(numbers[1]-numbers[0])*(numbers[2]-numbers[3])/mp.convert(h)


@pytest.mark.parametrize('source', ['fraction', 'text', 'mpf', 'mpc', 'mixed'])
@pytest.mark.parametrize('exponent', [70, 200])
@pytest.mark.parametrize('c', [32, 13])
def test_tiny_external_difference_retains_level_one_coefficient_and_pole(source, exponent, c):
    ext = _externals(source, exponent)
    expected = _reference_h1(ext, F(7, 10))
    with mp.workdps(40):
        actual = H_q_series(F(7, 10), ext, c=c, q_order=4)[1]
        assert expected != 0
        assert abs(actual-expected) < abs(expected)*mp.mpf('1e-37')
        roots = chiral_denominator_roots(ext, c=c, q_order=4)
        table = H_q_series_rational(ext, c=c, q_order=4)
        assert 0 in roots and 0 in table.denominator_roots
        h = mp.mpf('.7')
        actual = table.numerators[1].evaluate(h)/table.denominator.evaluate(h)
        assert abs(actual-expected) < abs(expected)*mp.mpf('1e-37')
        with pytest.raises(ZeroDivisionError):
            H_q_series(0, ext, c=c, q_order=4)
        assert mp.mp.dps == 40


@pytest.mark.parametrize('source', ['fraction', 'text', 'mpf', 'mpc', 'mixed'])
def test_exactly_paired_external_labels_still_remove_identity_pole(source):
    ext = _externals(source, 70)
    ext = (ext[0], ext[0], ext[2], ext[3])
    with mp.workdps(40):
        assert 0 not in chiral_denominator_roots(ext, c=32, q_order=4)
        assert 0 not in H_q_series_rational(ext, c=32, q_order=4).denominator_roots
        assert H_q_series(F(7, 10), ext, c=32, q_order=4)[1] == 0
        assert all(mp.isfinite(value) for value in H_q_series(0, ext, c=32, q_order=4))


def test_mixed_short_rational_and_binary_label_are_not_snapped_together():
    with mp.workdps(250):
        binary = mp.mpf('.1')
    ext = (F(1, 10), binary, F(3, 10), F(2, 5))
    expected = _reference_h1(ext, F(7, 10))
    with mp.workdps(40):
        assert expected != 0
        actual = H_q_series(F(7, 10), ext, c=32, q_order=1)[1]
        assert abs(actual-expected) < abs(expected)*mp.mpf('1e-37')
        assert chiral_denominator_roots(ext, c=32, q_order=1) == [0]


@pytest.mark.parametrize('precision', [40, 100])
@pytest.mark.parametrize('kind', ['decimal_shift', 'different_denominators'])
def test_conversion_guards_accuracy_even_when_the_offset_is_already_visible(precision, kind):
    if kind == 'decimal_shift':
        left, right = F(1, 10), F(1, 10)+F(1, 10**70)
    else:
        size = 10**70
        left, right = F(size-1, size), F(size-2, size-1)
    ext = (left, right, F(3, 10), F(2, 5))
    expected = _reference_h1(ext, F(7, 10))
    with mp.workdps(precision):
        actual = H_q_series(F(7, 10), ext, c=32, q_order=1)[1]
        assert abs(actual-expected) < abs(expected)*mp.mpf(10)**(-precision+3)


@pytest.mark.parametrize('source', ['fraction', 'mpc', 'mixed'])
def test_precision_changes_do_not_reuse_a_rounded_external_pair(source):
    ext = _externals(source, 200)
    with mp.workdps(500):
        reference = H_q_series(F(7, 10), ext, c=32, q_order=4)
    for precision in (40, 100, 40):
        with mp.workdps(precision):
            actual = H_q_series(F(7, 10), ext, c=32, q_order=4)
            for got, expected in zip(actual, reference):
                assert abs(got-expected) < abs(expected)*mp.mpf(10)**(-precision+3)


@pytest.mark.parametrize('label', [.7, '.7', '7/10', F(7, 10), Decimal('.7')])
def test_rational_h_and_q_queries_accept_exact_labels(label):
    with mp.workdps(40):
        table = H_q_series_rational((F(1, 10),)*4, c=32, q_order=4)
        actual = table.evaluate(label, F(1, 100))
        coefficients = H_q_series(F(7, 10), (F(1, 10),)*4, c=32, q_order=4)
        expected = sum(value*mp.mpf('.01')**k for k, value in enumerate(coefficients))
        assert abs(actual-expected) < mp.mpf('1e-37')


def test_real_complex_demotion_preserves_cache_key_and_charge_origin():
    with mp.workdps(180):
        b = b_from_c(32)
        value = mp.mpf('.1')+mp.mpf('1e-70')
        wrapped = mp.mpc(value)
    with mp.workdps(40):
        assert c_origin_of_b(b.real) == 32
        assert _scalar_cache_key(value) == _scalar_cache_key(wrapped)
        assert _numeric_input_values((wrapped,))[0]._mpf_ == value._mpf_
        # An ordinary equal binary value still shares its real cache entry.
        assert _scalar_cache_key(mp.mpc('.1')) != _scalar_cache_key(wrapped)
