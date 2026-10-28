"""Point evaluation must recognize poles beyond the first few global levels."""
import mpmath as mp
import pytest

from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.evaluation import _check_global_pole
from virasoro_blocks.requests import BlockPoleError


@pytest.mark.parametrize('weight', ['-10.5', '-100.5', '-1000000.5'])
def test_distant_genuine_pole_is_recognized(weight):
    with mp.workdps(50), pytest.raises(BlockPoleError):
        _check_global_pole(mp.mpf(weight), [mp.mpf('.1')] * 4,
                           basis='global_chiral_F')


def test_public_evaluation_reports_distant_pole():
    request = BlockRequest(operation='evaluate', family='global',
        basis='global_chiral_F', coordinate='z', exchange_h='-100.5',
        h_ext=('.1',) * 4, point={'z': '.3'},
        policy=NumericalPolicy(precision=50))
    with pytest.raises(BlockPoleError):
        evaluate_block_request(request, q_gate='off')


@pytest.mark.parametrize('weight', ['-100', '-100.500000000000000000000000000001', '-100.5+0.01j'])
def test_distant_removable_or_shifted_weight_is_not_rejected(weight):
    from virasoro_blocks.numeric_input import parse_number
    with mp.workdps(60):
        _check_global_pole(parse_number(weight), [mp.mpf('.1')] * 4,
                           basis='global_chiral_F')


def test_frozen_point_table_reports_distant_pole():
    from virasoro_blocks.point import point_block_table
    with mp.workdps(50), pytest.raises(BlockPoleError):
        point_block_table(['z'], basis='global_chiral_F', ell=0,
                          point_value='-100.5', h_externals=[mp.mpf('.1')] * 4,
                          lam=0)
