"""A b made from c keeps one theory through prefactors, recursion, and jets."""

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks import composer, derivs
from virasoro_blocks import params
from virasoro_blocks.params import b_from_c, c_origin_of_b, c_from_b, _c_input_of_b
from virasoro_blocks.recursion import F_block, block_prefactor
from virasoro_blocks.regularization import F_block_regularized, H_q_series_regularized


EXT = ('.1', '.2', '.3', '.4')


def leaves(value):
    if isinstance(value, (tuple, list)):
        for child in value:
            yield from leaves(child)
    else:
        yield value


def compare(actual, expected):
    actual, expected = list(leaves(actual)), list(leaves(expected))
    assert len(actual) == len(expected)
    assert any(value != 0 for value in expected)
    with mp.workdps(100):
        for value, reference in zip(actual, expected):
            assert abs(value-reference) < mp.mpf('1e-55')*max(1, abs(reference))


@pytest.mark.parametrize('charge', [30, 8])
@pytest.mark.parametrize('path', ['prefactor', 'F', 'B', 'G', 'chiral_z', 'chiral_y', 'chiral_q',
                                 'B_table', 'G_table', 'G_many', 'G_folded', 'point_jets'])
def test_known_b_is_rebuilt_when_evaluation_precision_increases(charge, path):
    with mp.workdps(20):
        b = b_from_c(charge)
    source = c_origin_of_b(b)

    def run(**parameters):
        options = dict(q_order=3, **parameters)
        if path == 'prefactor':
            return block_prefactor('.7', '.3', EXT, **parameters)
        if path == 'F':
            return F_block('.7', '.3', EXT, **options)
        if path == 'B':
            return composer.B_block('.7', '.8', '.3', '.45', EXT, EXT, **options)
        if path == 'G':
            return composer.G_block('2.5', 1, '.3', '.45', EXT, **options)
        if path.startswith('chiral_'):
            fn = getattr(derivs, f'{path}_taylor')
            return fn('.7', EXT, 3, **options)
        if path == 'B_table':
            return derivs.B_zzb_table('.7', '.8', EXT, EXT, 1, 1, **options)
        if path == 'G_table':
            return derivs.G_zzb_table('2.5', 1, EXT, 1, 1, **options)
        if path == 'G_many':
            return derivs.G_zzb_table_many(['2.5', '3.5'], 1, EXT, 1, 1, **options)
        if path == 'G_folded':
            return derivs.G_ws_table('2.5', 1, EXT, 1, 1, **options)
        return derivs.evaluate_block('.7', '.8', '.3', '.45', EXT,
                                     m_max=1, n_max=1, coord='zzb', **options)

    with mp.workdps(60):
        reference = run(c=source)
        actual = run(b=b)
        compare(actual, reference)
        assert mp.mp.dps == 60
        compare(run(b=b), reference)  # the warmed caches use the same source


@pytest.mark.parametrize('charge', [complex(30, 1), '.3', Fraction(3, 10)])
def test_direct_block_rebuilds_complex_and_exact_rational_source_charges(charge):
    # Caller wiring is covered above; these additional source forms exercise
    # the common b reconstruction without repeating every table/jet build.
    with mp.workdps(20):
        b = b_from_c(charge)
    source = c_origin_of_b(b)
    if charge == '.3' or isinstance(charge, Fraction):
        assert source == Fraction(3, 10)
    with mp.workdps(60):
        compare(F_block('.7', '.3', EXT, b=b, q_order=3),
                F_block('.7', '.3', EXT, c=source, q_order=3))


@pytest.mark.parametrize('charge', [.5, 25, 26])
@pytest.mark.parametrize('series', [False, True])
def test_regularized_entry_points_use_the_known_singular_source_charge(charge, series):
    with mp.workdps(20):
        b = b_from_c(charge)
    source = c_origin_of_b(b)
    with mp.workdps(60):
        if series:
            expected = H_q_series_regularized('.7', EXT, c=source, q_order=4)
            actual = H_q_series_regularized('.7', EXT, b=b, q_order=4)
        else:
            expected = F_block_regularized('.7', '.3', EXT, c=source, q_order=4)
            actual = F_block_regularized('.7', '.3', EXT, b=b, q_order=4)
        compare(actual, expected)


def test_an_independently_supplied_b_keeps_its_numerical_value():
    with mp.workdps(60):
        b = mp.mpc('.37', '.12')
        assert c_origin_of_b(b) is None
        assert derivs._resolve_b(b, None) is b
        assert abs(_c_input_of_b(b)-c_from_b(b)) < 2*mp.eps*abs(c_from_b(b))


def test_independent_b_does_not_acquire_an_origin_from_an_equal_parameter():
    with mp.workdps(20):
        independent = mp.mpc(params._b_from_c(32))
    with mp.workdps(60):
        before = F_block('.7', '.3', ('.1',)*4, b=independent, q_order=2)
    with mp.workdps(20):
        known = b_from_c(32)
        assert known._mpc_ == independent._mpc_
        assert c_origin_of_b(mp.mpc(known)) is None
    assert c_origin_of_b(known) == 32
    assert c_origin_of_b(independent) is None
    with mp.workdps(60):
        assert F_block('.7', '.3', ('.1',)*4, b=independent, q_order=2) == before


@pytest.mark.parametrize('label', ['.3', '0.3', '3e-1', '3/10', .3, Fraction(3, 10)])
def test_b_parameter_keeps_the_exact_real_input_label(label):
    with mp.workdps(15):
        b = b_from_c(label)
        assert c_origin_of_b(b) == Fraction(3, 10)
        assert mp.mp.dps == 15


def test_returned_b_keeps_its_source_after_other_parameters_are_constructed():
    with mp.workdps(20):
        b = b_from_c('.3')
        for charge in (31, 32, 33):
            b_from_c(charge)
    assert c_origin_of_b(b) == Fraction(3, 10)
    with mp.workdps(60):
        expected = F_block('.7', '.3', EXT, c=Fraction(3, 10), q_order=3)
        compare(F_block('.7', '.3', EXT, b=b, q_order=3), expected)


