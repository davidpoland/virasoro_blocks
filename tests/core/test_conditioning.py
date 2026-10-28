"""Point values near a pole: the measured loss is reported, and target_digits escalates by it."""
import warnings

import mpmath as mp
import pytest

from virasoro_blocks.api import evaluate_block_request
from virasoro_blocks.chiral import F_block_chiral_h
from virasoro_blocks.conditioning import InsufficientPrecisionError, PrecisionLossWarning
from virasoro_blocks.params import b_from_c
from virasoro_blocks.requests import BlockRequest, NumericalPolicy
from virasoro_blocks.verified_build import TargetDigitsError

EXT = ('1/8', '1/7', '1/6', '1/5')


def _pole_21(dps):
    with mp.workdps(dps):
        b = b_from_c(8)
        Q = b + 1/b
        return (Q*Q - (2*b + 1/b)**2)/4


def test_chiral_h_warns_near_a_pole_and_refuses_when_no_digit_survives():
    with mp.workdps(30):
        block = F_block_chiral_h('1/2', EXT, c=8, q_order=12)
        pole = _pole_21(30)
        with pytest.warns(PrecisionLossWarning, match='digits lost'):
            block.evaluate(pole + mp.mpf('1e-20'))
        with pytest.raises(InsufficientPrecisionError, match='raise the precision to about'):
            block.evaluate(pole + mp.mpf('1e-31'))
        with warnings.catch_warnings():
            warnings.simplefilter('error', PrecisionLossWarning)
            block.evaluate(pole + mp.mpf('1e-5'))


def _request(h, target=None, max_precision=None):
    return BlockRequest(operation='evaluate', family='virasoro', basis='chiral_F', coordinate='z', h_ext=EXT,
                        central_charge=8, exchange_h=h, point={'z': '1/2'},
                        policy=NumericalPolicy(precision=30, order=12, target_digits=target,
                                               max_precision=max_precision))


def test_target_digits_escalates_a_point_value_by_the_measured_loss():
    with mp.workdps(200):
        h = mp.nstr(_pole_21(200) + mp.mpf('1e-28'), 120)
        reference = evaluate_block_request(BlockRequest(**{**_request(h).__dict__,
                                                           'policy': NumericalPolicy(precision=200, order=12)}),
                                           q_gate='off').value
    value = evaluate_block_request(_request(h, target=25), q_gate='off')
    assert value.metadata['precision_actual'] > 30
    with mp.workdps(60):
        assert abs(value.value/reference - 1) < mp.mpf('1e-25')
    with pytest.raises(TargetDigitsError, match='max_precision'):
        evaluate_block_request(_request(h, target=25, max_precision=40), q_gate='off')
