"""Bootstrap spin inputs share the integer contract used by block requests."""
from fractions import Fraction
import numpy as np
import pytest
from virasoro_bootstrap.identical_scalar import (
    _validate_even_spins, _validate_identical_scalar_operator,
)


@pytest.mark.parametrize('value', [np.bool_(False)])
def test_boolean_spin_list_entries_are_rejected(value):
    with pytest.raises(ValueError, match='integers'):
        _validate_even_spins((0, value))


@pytest.mark.parametrize('value', [True])
def test_boolean_operator_spins_are_rejected(value):
    with pytest.raises(ValueError, match='integer'):
        _validate_identical_scalar_operator(value, 3, name='point constraint')


@pytest.mark.parametrize('values', [np.array([4, 0, 2, 2])])
def test_spin_sequences_normalize_consistently(values):
    assert _validate_even_spins(values) == (0, 2, 4)


@pytest.mark.parametrize('spin', [Fraction(2), np.int64(2)])
def test_integral_spin_types_remain_supported(spin):
    assert _validate_even_spins((0, spin)) == (0, 2)
    assert _validate_identical_scalar_operator(spin, 3, name='point constraint')[0] == 2


@pytest.mark.parametrize('values', [iter(())])
def test_missing_or_empty_spin_sequences_are_rejected(values):
    with pytest.raises(ValueError, match='non-empty'):
        _validate_even_spins(values)
