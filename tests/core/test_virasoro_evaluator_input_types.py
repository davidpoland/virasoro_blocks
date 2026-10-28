"""Single and batched Virasoro evaluators preserve numeric input semantics."""
from fractions import Fraction as F
import mpmath as mp
import pytest
from virasoro_blocks.derivs import chiral_z_series_evaluator
from virasoro_blocks.polynomial import _arb_to_mpf
from virasoro_blocks.singularity import merged_singular_evaluators


def make_evaluator(c):
    with merged_singular_evaluators():
        return chiral_z_series_evaluator([mp.mpf('.1')]*4, 3, c=c, q_order=4)


@pytest.mark.parametrize('c', [32, 2, 1, 25])
@pytest.mark.parametrize('value', [complex(.8, .3), complex(.8, 0), F(4, 5)])
def test_single_input_types_match_mpmath(c, value):
    with mp.workdps(40):
        ev = make_evaluator(c)
        expected = ev(mp.convert(value))
        actual = ev(value)
        assert all(mp.almosteq(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize('c', [32, 2, 1, 25])
def test_batch_iterator_and_fraction_weights(c):
    with mp.workdps(40):
        ev = make_evaluator(c)
        values = [F(4, 5), F(6, 5)]
        actual = ev.many(iter(values))
        assert actual is not None and len(actual) == len(values)
        for value, row in zip(values, actual):
            expected = ev(mp.convert(value))
            assert all(mp.almosteq(_arb_to_mpf(a), b) for a, b in zip(row, expected))


@pytest.mark.parametrize('c', [32, 2, 1, 25])
def test_batch_declines_native_complex_weights(c):
    with mp.workdps(40):
        ev = make_evaluator(c)
        assert ev.many([complex(.8, .3)]) is None
