"""Original physical points must be resolved before scalar-block conversion."""
from decimal import Decimal
from fractions import Fraction as F
from functools import lru_cache

import mpmath as mp
import pytest

from virasoro_blocks.derivs import _global_chiral_block
from virasoro_blocks.global_blocks import build_global_sampler, global_evaluate_block
from virasoro_blocks.numeric_input import literal_number
from virasoro_blocks.params import b_from_c
from virasoro_blocks.recursion import F_block, block_prefactor
from virasoro_blocks.regularization import F_block_regularized, H_q_series_regularized


EXT = (F(1, 10),)*4


def _source(kind, side=-1, exponent=70):
    point = 1+side*F(1, 10**exponent)
    if kind == 'fraction':
        return point
    if kind == 'text':
        return f'{point.numerator}/{point.denominator}'
    if kind == 'decimal':
        return Decimal('0.'+'9'*exponent) if side < 0 else Decimal('1.'+'0'*(exponent-1)+'1')
    with mp.workdps(exponent+100):
        point = mp.convert(literal_number(point))
        return point if kind == 'mpf' else mp.mpc(point, 0)


@pytest.mark.parametrize('side', [-1, 1])
def test_global_complex_endpoint_keeps_the_supplied_branch(side):
    with mp.workdps(160):
        point = mp.mpc(1, side*mp.mpf('1e-70'))
        expected = -mp.log(1-point)/16
    with mp.workdps(40):
        actual = _global_chiral_block(1, point, EXT)
        assert mp.sign(mp.im(actual)) == side
        assert abs(actual-expected) < abs(expected)*mp.mpf('1e-39')


@pytest.mark.parametrize('point', [F(1, 10**200), -F(1, 10**200), F(10**70), -F(10**70)])
def test_global_small_and_large_points_match_the_closed_logarithmic_form(point):
    with mp.workdps(300):
        expected = -mp.log1p(-mp.convert(literal_number(point)))/16
    with mp.workdps(40):
        actual = _global_chiral_block(1, point, EXT)
        assert abs(actual-expected) < abs(expected)*mp.mpf('1e-39')


def _independent_prefactor(h, z, ext, c, *, norm=16, shift=0):
    # Elliptic integrals and theta functions directly, independent of the
    # production nome helper and prefactor's point-normalization guard.
    h, z, c = (mp.convert(literal_number(x)) for x in (h, z, c))
    ext = [mp.convert(literal_number(x)) for x in ext]
    q = mp.exp(-mp.pi*mp.ellipk(1-z)/mp.ellipk(z))
    sigma = (c-1)/24
    value = (16*q)**(-sigma)*(16*q/norm)**h*z**(sigma+shift)
    value *= (1-z)**(sigma-ext[1]-ext[2])
    value *= mp.jtheta(3, 0, q)**((c-1)/2-4*sum(ext))
    return value, q


@lru_cache(maxsize=None)
def _virasoro_reference(c, side, exponent=70):
    with mp.workdps(exponent+130):
        point = 1+side*F(1, 10**exponent)
        prefactor, q = _independent_prefactor(F(7, 10), point, EXT, c)
        coefficients = H_q_series_regularized(F(7, 10), EXT, c=c, q_order=4)
        return prefactor, prefactor*mp.polyval(list(reversed(coefficients)), q)


@pytest.mark.parametrize('engine', ['global', 'prefactor', 'bare', 'singular'])
def test_endpoint_guard_restores_context_when_evaluation_fails(engine):
    point = _source('fraction')
    with mp.workdps(40):
        # (z = 1 - 1e-70 is unresolvable at 40 digits: the conditioning check may refuse it first)
        with pytest.raises(ValueError, match='chiral_norm must be > 0|no correct digit'):
            if engine == 'global':
                _global_chiral_block(1, point, EXT, chiral_norm=0)
            elif engine == 'prefactor':
                block_prefactor(F(7, 10), point, EXT, c=32, chiral_norm=0)
            elif engine == 'bare':
                F_block(F(7, 10), point, EXT, c=32, q_order=4, chiral_norm=0)
            else:
                F_block_regularized(F(7, 10), point, EXT, c=F(1, 2), q_order=4, chiral_norm=0)
        assert mp.mp.dps == 40


def test_regularized_direct_float_labels_match_their_exact_decimal_values():
    with mp.workdps(40):
        actual = F_block_regularized(.7, .3, [.1, .15, .2, .25], c=F(1, 2), q_order=4)
        expected = F_block_regularized(F(7, 10), F(3, 10),
            [F(1, 10), F(15, 100), F(1, 5), F(1, 4)], c=F(1, 2), q_order=4)
        assert actual == expected


@pytest.mark.parametrize('engine', [F_block, F_block_regularized])
def test_direct_scalar_controls_keep_native_float_decimal_intent(engine):
    with mp.workdps(40):
        actual = engine(.7, .3, [.1, .15, .2, .25], c=F(1, 2), q_order=4,
            chiral_norm=.7, z_shift=.2)
        expected = engine(F(7, 10), F(3, 10),
            [F(1, 10), F(15, 100), F(1, 5), F(1, 4)], c=F(1, 2), q_order=4,
            chiral_norm=F(7, 10), z_shift=F(1, 5))
        assert actual == expected


