"""Exact query labels and one-pass iterables reach every table entry intact."""
from fractions import Fraction as F
from decimal import Decimal

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks.interpolation import DampedRational
from virasoro_blocks.polyderivs import DeltaDerivTable, InterpolatedDeltaTable
from virasoro_blocks.polynomial import Polynomial


def _exact_table(*, field=mp.mpf, multiplicity=1):
    denominator = Polynomial([1], field=field)
    for _ in range(multiplicity):
        denominator *= Polynomial([F(-1, 2), 1], field=field)
    return DeltaDerivTable(field(1), denominator, [F(1, 2)]*multiplicity,
                           [[Polynomial([1], field=field)]])


def _interpolated_table(*, multiplicity=1, regulated=False):
    return InterpolatedDeltaTable(
        DampedRational(1, (F(-1, 2),)*multiplicity, 1), mp.mpf(1), (),
        [[Polynomial([1], field=mp.mpf)]],
        min_pole_distance=mp.mpf('1e-20') if regulated else None)


@pytest.mark.parametrize('kind', ['exact', 'interpolated', 'standalone'])
@pytest.mark.parametrize('label', [.7, '.7', '7/10', F(7, 10), Decimal('.7')])
def test_table_queries_share_decimal_and_fraction_input_policy(kind, label):
    with mp.workdps(40):
        if kind == 'exact':
            value = _exact_table().evaluate(label)[0][0]
        else:
            table = _interpolated_table()
            value = table.interpolant(0, 0).evaluate(label) if kind == 'standalone' else table.evaluate(label)[0][0]
        assert abs(value-5) < mp.mpf('1e-37')


@pytest.mark.parametrize('iterable', ['generator', 'list', 'numpy'])
@pytest.mark.parametrize('mixed', [False, True])
def test_batch_evaluation_consumes_each_input_once_without_losing_the_first_item(iterable, mixed):
    with mp.workdps(40):
        labels = [F(7, 10), '4/5', complex(1, .1) if mixed else mp.mpc(1)]
        table = _exact_table()
        expected = [table.evaluate(d) for d in labels]
        inputs = iter(labels) if iterable == 'generator' else np.array(labels, dtype=object) if iterable == 'numpy' else labels
        actual = table.evaluate_many(inputs)
        assert len(actual) == len(labels)
        for got, want in zip(actual, expected):
            assert abs(got[0][0]-want[0][0]) < mp.mpf('1e-37')


