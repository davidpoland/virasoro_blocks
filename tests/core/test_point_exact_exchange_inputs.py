"""A frozen point keeps written offsets visible to the numerical evaluator."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import literal_number
from virasoro_blocks.point import point_block_table
from virasoro_blocks.requests import BlockPoleError


@pytest.mark.parametrize('label', ['.3', Fraction(3,10), 0.3])
def test_point_external_inputs_follow_decimal_and_rational_policy(label):
    with mp.workdps(40):
        table = point_block_table(['zzb'],basis='scalar_G',ell=0,point_value='.7',
                                 h_externals=[label]*4,lam=1,c='2',q_order=4)
        reference = point_block_table(['zzb'],basis='scalar_G',ell=0,point_value=Fraction(7,10),
                                     h_externals=[literal_number('.3')]*4,lam=1,c='2',q_order=4)
        assert table.numerators == reference.numerators


