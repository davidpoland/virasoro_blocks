"""Standalone prefactor queries preserve labels and stored numerical data."""
from fractions import Fraction as F

import mpmath as mp
import pytest
from flint import arb

from virasoro_blocks.interpolation import DampedRational
from virasoro_blocks.polynomial import _arb_to_mpf


def test_prefactor_batches_accept_mixed_real_labels_empty_generators_and_existing_balls():
    with mp.workdps(40):
        prefactor = DampedRational(1, (-1,), '.5')
        assert prefactor.evaluate_arb_batch(iter(())) == []
        labels = [F(7, 10), '7/10', .7, mp.mpc('.7'), arb(1)]
        values = prefactor.evaluate_arb_batch(iter(labels))
        assert len(values) == 5
        for value in values[:4]:
            assert abs(_arb_to_mpf(value)-prefactor.evaluate('7/10')) < mp.mpf('1e-37')
        assert values[-1].contains(F(1, 4).numerator/F(1, 4).denominator)


def test_failed_prefactor_query_restores_the_callers_precision():
    with mp.workdps(40):
        prefactor = DampedRational(1, (-F(1, 2),), 1)
        with pytest.raises(ZeroDivisionError):
            prefactor.evaluate(-F(1, 2))
        assert mp.mp.dps == 40
