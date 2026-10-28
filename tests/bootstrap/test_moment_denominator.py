"""Normalized moment bounds need a strictly positive denominator; a provably nonpositive one is refused."""

import mpmath as mp
import pytest

from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec
from virasoro_bootstrap.moments import (
    MomentTerm,
    MomentWeight,
    build_identical_scalar_normalized_moment_lower_problem as lower,
    build_identical_scalar_normalized_moment_upper_problem as upper,
)


@pytest.fixture
def spec():
    mp.mp.dps = 40
    return IdenticalScalarGapSpec(c=mp.mpf(4), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf("0.5"), lam=3,
                                  spin_list=(0, 2), q_order=10, dps=40, npoles=4, num_threads=1)


@pytest.mark.parametrize("denominator, alpha, mode", [
    (MomentWeight.one().scale(0), -1, "include"),    # the constant ratio 1 / (-1): a "lower bound" of 10 came out
    (MomentWeight.one().scale(0), 0, "exclude"),
    (-MomentWeight.one(), 0, "exclude"),
    (MomentWeight.one().minus(MomentWeight.one()), -1, "include"),       # the zero weight written as 1 - 1
    # Delta - Delta with the monomial spelled two ways ({"Delta": 1, "L": 0} is {"Delta": 1})
    (MomentWeight((MomentTerm(1, {"Delta": 1, "L": 0}), MomentTerm(-1, {"Delta": 1}))), 0, "exclude"),
])
def test_never_positive_denominators_are_refused(spec, denominator, alpha, mode):
    with pytest.raises(ValueError, match="denominator .* is never positive"):
        lower(spec, MomentWeight.one(), denominator, alpha=alpha, beta=1, identity_mode=mode)


def test_positive_denominators_build(spec):
    zero = MomentWeight.one().scale(0)
    lower(spec, zero, zero, alpha=1, beta=1)
    # a signed weight that is positive somewhere is the caller's to vouch for, so it is not refused
    lower(spec, MomentWeight.one(), MomentWeight.monomial(Delta=1).minus(MomentWeight.monomial(Delta=2)), alpha=1)
    lower(spec, MomentWeight.monomial(Delta=1), MomentWeight.one(), identity_mode="include")
    upper(spec, MomentWeight.monomial(Delta=1), MomentWeight.one(), identity_mode="exclude")
