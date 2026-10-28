"""Direct recursion distinguishes exact Kac poles from deliberately shifted labels."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockPoleError, BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.params import b_from_c
from virasoro_blocks.recursion import F_block


EXTERNALS = ('.3', '.4', '.5', '.6')


@pytest.mark.parametrize('c,center', [(32, 0), (13, Fraction(-3, 2))])
def test_exact_genuine_kac_poles_are_still_rejected(c, center):
    with mp.workdps(40), pytest.raises(BlockPoleError):
        F_block(center, mp.mpf('.3'), EXTERNALS, c=c, q_order=4)


def test_exact_removable_vacuum_matches_its_regular_limit():
    with mp.workdps(100):
        step = Fraction(1, 10**40)
        expected = (F_block(step, mp.mpf('.3'), ('.3',)*4, c=32, q_order=4)
                    + F_block(-step, mp.mpf('.3'), ('.3',)*4, c=32, q_order=4))/2
    with mp.workdps(40):
        actual = F_block(0, mp.mpf('.3'), ('.3',)*4, c=32, q_order=4)
        assert abs(actual-expected) < mp.mpf('1e-38')*abs(expected)


@pytest.mark.parametrize('c', [Fraction(501, 1000), 8, 30])
def test_rounded_genuine_kac_weight_is_a_pole(c):
    # d_{2,1} at the working precision is the pole to roundoff, not a finite block of size 1e28
    from virasoro_blocks.params import d_rs

    with mp.workdps(30), pytest.raises(BlockPoleError):
        F_block(d_rs(2, 1, b_from_c(c)), mp.mpf(1)/2, [mp.mpf(1)/8, mp.mpf(1)/7, mp.mpf(1)/6, mp.mpf(1)/5],
                c=c, q_order=12)


