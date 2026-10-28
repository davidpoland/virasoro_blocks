"""Rational tower parameters follow the same path as mpmath parameters."""
from fractions import Fraction as F
import mpmath as mp
import pytest
from virasoro_blocks.large_spin import large_spin_twist_table


@pytest.mark.parametrize('mode', ['global', 'virasoro'])
@pytest.mark.parametrize('part', ['twist', 'h_externals', 'hbar_externals', 'Lmax'])
def test_fraction_tower_parameters(mode, part):
    with mp.workdps(40):
        rational = dict(twist=F(1, 2), h_externals=[F(1, 8)]*4,
                        hbar_externals=[F(1, 7)]*4, Lmax=F(5, 2))
        numeric = {key: list(map(mp.convert, value)) if isinstance(value, list) else mp.convert(value)
                   for key, value in rational.items()}
        options = dict(mode=mode, m_max=0, n_max=1, npoles=1, q_order=4,
                       c=32, basis='ordered_B', pole_selection='nearest')
        expected = large_spin_twist_table(**numeric, **options)
        numeric[part] = rational[part]
        actual = large_spin_twist_table(**numeric, **options)
        assert actual.shift == expected.shift
        assert actual.prefactor.poles == expected.prefactor.poles
        for n in range(2):
            for x in [0, 3]:
                assert mp.almosteq(actual.numerators[0][n].evaluate(x),
                                   expected.numerators[0][n].evaluate(x))
