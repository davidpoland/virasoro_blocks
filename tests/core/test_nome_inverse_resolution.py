"""Inverse nomes preserve endpoint offsets and agree with direct theta sums."""
from fractions import Fraction
from functools import lru_cache

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.coordinates import z_of_q
from virasoro_blocks.numeric_input import literal_number


@lru_cache(maxsize=None)
def _theta_reference(q):
    # Independent direct sums.
    with mp.workdps(750):
        q = mp.convert(literal_number(q))
        return (mp.jtheta(2, 0, q)/mp.jtheta(3, 0, q))**4


# q = .99 (1 - z ~ 1e-428) is far below any working precision; .95 is the near-endpoint case.
REQUEST_NOMES = ['.95']


@pytest.mark.parametrize('q', REQUEST_NOMES)
def test_global_request_accepts_a_finite_near_endpoint_nome(q):
    with mp.workdps(750):
        expected = -mp.log1p(-_theta_reference(q))/16
    request = BlockRequest(
        operation='evaluate', family='global', basis='chiral_F', coordinate='q',
        h_ext=(0,)*4, exchange_h=1, point={'q': q},
        policy=NumericalPolicy(precision=40),
    )
    result = evaluate_block_request(request, q_gate='off')
    with mp.workdps(60):
        assert abs(result.value-expected) < mp.mpf('1e-40')*abs(expected)
    assert literal_number(result.metadata['point']['z']) != 1


@pytest.mark.parametrize('qbar', REQUEST_NOMES)
def test_full_request_converts_the_antiholomorphic_nome(qbar):
    with mp.workdps(750):
        expected = mp.log1p(-_theta_reference('.9'))*mp.log1p(-_theta_reference(qbar))/256
    request = BlockRequest(
        operation='evaluate', family='global', basis='ordered_B', coordinate='qqb',
        h_ext=(0,)*4, hbar_ext=(0,)*4, exchange_h=1, exchange_hbar=1,
        point={'q': '.9', 'qbar': qbar}, policy=NumericalPolicy(precision=40),
    )
    result = evaluate_block_request(request, q_gate='off')
    with mp.workdps(60):
        assert abs(result.value-expected) < mp.mpf('1e-40')*abs(expected)


@pytest.mark.parametrize('q', [1, -1, Fraction(1)-Fraction(1, 10**70)])
def test_inverse_nome_preserves_the_existing_theta_domain_error(q):
    before = mp.mp.dps
    with pytest.raises(ValueError, match='THETA_Q_LIM'):
        z_of_q(q)
    assert mp.mp.dps == before
