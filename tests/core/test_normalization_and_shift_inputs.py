"""Physical normalization and scalar shift controls follow exact label policy."""
from fractions import Fraction as F

import mpmath as mp
import pytest

from virasoro_blocks import polyderivs as pd
from virasoro_blocks.derivs import _global_chiral_block
from virasoro_blocks.recursion import F_block, _resolve_chiral_norm


@pytest.mark.parametrize('norm', [mp.inf, mp.ninf, mp.nan, float('inf'), float('nan'),
    0, -1, True, False, 1j, mp.mpc(1, '1e-70')])
def test_invalid_chiral_normalizations_fail_at_the_shared_resolver_and_direct_blocks(norm):
    with mp.workdps(40):
        for call in (lambda:_resolve_chiral_norm(norm),
                     lambda:_global_chiral_block(F(7, 10), F(3, 10), [F(1, 10)]*4, chiral_norm=norm),
                     lambda:F_block(F(7, 10), F(3, 10), [F(1, 10)]*4, c=32, q_order=4, chiral_norm=norm)):
            with pytest.raises(ValueError):
                call()
        assert mp.mp.dps == 40


@pytest.mark.parametrize('norm', [.7, '.7', '7/10', F(7, 10)])
def test_normalizations_accept_decimal_and_fraction_labels_consistently(norm):
    with mp.workdps(40):
        assert _resolve_chiral_norm(norm) == mp.mpf(7)/10
        actual = pd.F_z_poly_table([F(1, 10)]*4, 0, c=32, q_order=4, chiral_norm=norm).evaluate(F(7, 10))[0][0]
        reference = F_block(F(7, 10), F(1, 2), [F(1, 10)]*4, c=32, q_order=4, chiral_norm=F(7, 10))
        assert abs(actual-reference) < abs(reference)*mp.mpf('1e-36')


def test_normalization_keeps_the_stored_binary_value_of_mpmath_inputs():
    with mp.workdps(200):
        norm = mp.mpf('.7')
        complex_norm = mp.mpc(norm)
    with mp.workdps(40):
        assert _resolve_chiral_norm(norm)._mpf_ == norm._mpf_
        assert _resolve_chiral_norm(complex_norm)._mpf_ == norm._mpf_


@pytest.mark.parametrize('coordinate', ['zzb', 'yyb', 'qqb'])
@pytest.mark.parametrize('bar_shift', [F(1, 5), '.2', .2])
def test_equal_scalar_kinematic_shift_labels_are_accepted_without_coercion_errors(coordinate, bar_shift):
    with mp.workdps(40):
        builder = getattr(pd, 'G_'+coordinate+'_poly_table')
        kwargs = dict(c=32, q_order=4, z_shift=F(1, 5))
        actual = builder(0, [F(1, 10)]*4, 0, zbar_shift=bar_shift, **kwargs).evaluate(F(7, 5))[0][0]
        expected = builder(0, [F(1, 10)]*4, 0, **kwargs).evaluate(F(7, 5))[0][0]
        assert actual == expected


@pytest.mark.parametrize('coordinate', ['zzb', 'yyb', 'qqb'])
def test_distinct_scalar_shifts_are_rejected_even_when_the_offset_is_below_working_precision(coordinate):
    with mp.workdps(40):
        with pytest.raises(ValueError, match='zbar_shift must equal'):
            getattr(pd, 'G_'+coordinate+'_poly_table')(0, [F(1, 10)]*4, 0, c=32, q_order=4,
                z_shift=F(1, 5), zbar_shift=F(1, 5)+F(1, 10**70))
        assert mp.mp.dps == 40
