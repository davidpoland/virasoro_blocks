"""Point tables must honor their basis and derivative-order controls."""
from fractions import Fraction
import mpmath as mp
import numpy as np
import pytest
from virasoro_blocks.point import point_block_table


def build(**overrides):
    kwargs = dict(basis='scalar_G', ell=0, point_value=1,
                  h_externals=[mp.mpf('.1')] * 4, c=32, q_order=4, lam=0)
    kwargs.update(overrides)
    return point_block_table(['zzb'], **kwargs)


@pytest.mark.parametrize('basis', ['typo', 'global_scalar', 'ns_scalar_G', '', None])
def test_unknown_point_basis_is_rejected(basis):
    with mp.workdps(25), pytest.raises(ValueError, match='basis'):
        build(basis=basis)


@pytest.mark.parametrize('lam', [True, False, np.bool_(False), .5, float('nan'), float('inf'), -1])
def test_invalid_point_derivative_order_is_rejected(lam):
    with mp.workdps(25), pytest.raises(ValueError, match='lam'):
        build(lam=lam)


@pytest.mark.parametrize('lam', [1.0, Fraction(1), np.int64(1)])
def test_integral_point_derivative_orders_match_integer(lam):
    with mp.workdps(25):
        expected = build(lam=1)
        actual = build(lam=lam)
        assert actual.evaluate('zzb', 1) == expected.evaluate('zzb', 1)
