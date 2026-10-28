"""Exact rational objectives use the same input conversion as bootstrap data."""
from fractions import Fraction
from importlib import import_module
import mpmath as mp
import pytest


@pytest.mark.parametrize('module,name,sign,value', [
    ('identical_scalar', 'ope_squared_bound_from_objective', -1, Fraction(1, 3)),
    ('moments', 'normalized_moment_lower_bound_from_objective', 1, Fraction(-2, 7)),
])
def test_fraction_objective_matches_rational_string(module, name, sign, value):
    function = getattr(import_module('virasoro_bootstrap.' + module), name)
    with mp.workdps(50):
        expected = function(f'{value.numerator}/{value.denominator}')
        actual = function(value)
        assert actual == expected
        assert actual == sign * mp.mpf(f'{value.numerator}/{value.denominator}')
