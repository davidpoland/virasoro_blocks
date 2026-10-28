"""Tower samples preserve rational and native complex spin inputs."""
from fractions import Fraction as F
import mpmath as mp
import pytest
from virasoro_blocks.large_spin import _LargeSpinTwistSampler, large_spin_twist_table
from virasoro_blocks.polynomial import _arb_to_mpf


@pytest.mark.parametrize('mode', ['global', 'virasoro'])
@pytest.mark.parametrize('value', [F(7, 2), complex(3.5, 0), complex(3.5, .2)])
def test_tower_spin_types_match_mpmath(mode, value):
    with mp.workdps(40):
        sampler = _LargeSpinTwistSampler(mode, 'zzb', [mp.mpf('.1')]*4,
            16, 0, 1, 1, mp.mpf('.5'), c=32, q_order=4)
        reference = sampler(mp.convert(value))
        actual = sampler(value)
        for m in range(2):
            for n in range(2):
                assert mp.almosteq(actual[m][n], reference[m][n])
        grids, is_arb = sampler.sample_bucket([value], [(0, 0), (1, 1)])
        for m, n in [(0, 0), (1, 1)]:
            actual = _arb_to_mpf(grids[0][m][n]) if is_arb else grids[0][m][n]
            assert mp.almosteq(actual, reference[m][n])


