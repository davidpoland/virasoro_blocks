"""Submission must accept the same unbounded integers as numeric task inputs."""
import pytest

from virasoro_bootstrap.hpc.params import validate_sweep


@pytest.mark.parametrize('value', [-(10**400)])
def test_unbounded_integer_real_labels_do_not_overflow_validation(value):
    assert validate_sweep({'task':'scalar_gap_bisection','params':{'c':value}}) == []


@pytest.mark.parametrize('value', [float('nan'),True])
def test_invalid_real_labels_are_still_rejected(value):
    assert any('c =' in error for error in validate_sweep(
        {'task':'scalar_gap_bisection','params':{'c':value}}))
