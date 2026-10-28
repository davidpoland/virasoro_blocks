"""Optimal real-node placement must use the same real shift as reconstruction."""

from fractions import Fraction as F

import mpmath as mp
import pytest

from virasoro_blocks import interpolation as ip


@pytest.mark.parametrize("shift", [mp.mpc(1, 1), mp.nan, mp.inf, -mp.inf])
def test_invalid_shift_is_rejected_before_node_search(monkeypatch, shift):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid shift reached node search")
    monkeypatch.setattr(ip, "get_sample_points", unexpected)
    prefactor = ip.DampedRational(1, (), mp.exp(-mp.pi))
    with pytest.raises(ValueError, match="shift"):
        ip.build_interpolants_for_npoles(prefactor, [0], lambda x: 1, shift=shift)


def test_real_shift_places_and_reconstructs_samples_consistently(monkeypatch):
    with mp.workdps(40):
        monkeypatch.setattr(ip, "get_sample_points", lambda *a, **kw: [mp.mpf(1)])
        prefactor = ip.DampedRational(1, (), mp.exp(-mp.pi))
        shift = F(1, 3)
        numeric_shift = mp.convert(shift)
        seen = []
        def evaluate(delta):
            seen.append(delta)
            return 2 * prefactor.evaluate(delta - numeric_shift)
        result = ip.build_interpolants_for_npoles(prefactor, [0], evaluate, shift=shift)[0]
        assert seen == [1 + numeric_shift]
        assert result.shift == numeric_shift
        assert mp.almosteq(result.evaluate(2 + numeric_shift), evaluate(2 + numeric_shift))
