"""Discrete super-block choices must not silently select another block."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.superconformal_blocks import (
    SuperBlockPolicy, normalize_branch, normalize_sector, resolve_super_policy,
)
from virasoro_blocks.super_table_backends import SuperBackendOptions


@pytest.mark.parametrize('normalize', [normalize_sector, normalize_branch])
@pytest.mark.parametrize('bad', [.5, -.5, True, float('nan'), float('inf')])
def test_discrete_choices_reject_nonintegers(normalize, bad):
    with pytest.raises(ValueError):
        normalize(bad)


@pytest.mark.parametrize('constructor', [SuperBlockPolicy, resolve_super_policy, SuperBackendOptions])
@pytest.mark.parametrize('bad', [8.5, Fraction(17, 2), True, '24', 7, float('nan'), float('inf')])
def test_offset_exponent_is_not_silently_truncated(constructor, bad):
    with pytest.raises(ValueError, match='eps_exp'):
        constructor(eps_exp=bad)


@pytest.mark.parametrize('constructor', [resolve_super_policy, SuperBackendOptions])
@pytest.mark.parametrize('value', [24., Fraction(24), mp.mpf(24), 8])
def test_integral_exponents_are_normalized(value, constructor):
    policy = constructor(eps_exp=value)
    assert type(policy.eps_exp) is int
    assert policy.eps_exp == int(value)


@pytest.mark.parametrize('constructor', [resolve_super_policy, SuperBackendOptions])
def test_stored_subprecision_exponent_offsets_are_not_integers(constructor):
    with mp.workdps(150):
        exponent = mp.mpf(24) + mp.mpf('1e-100')
    with mp.workdps(15), pytest.raises(ValueError, match='eps_exp'):
        constructor(eps_exp=exponent)


def test_table_offset_default_is_not_frozen_at_construction_precision():
    with mp.workdps(30):
        options = SuperBackendOptions()
    assert options.eps_exp is None


def test_named_and_integral_choices_remain_supported():
    assert normalize_sector('odd') == normalize_sector(1.) == 1
    assert normalize_branch('-+') == normalize_branch(Fraction(2)) == 2
