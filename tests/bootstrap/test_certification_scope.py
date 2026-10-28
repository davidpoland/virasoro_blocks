"""The dual-functional certifiers cover the sphere problems and refuse a modular one clearly."""

import mpmath as mp
import pytest

from virasoro_bootstrap.certification import certify_dual_from_sdpb_output, certify_dual_functional
from virasoro_bootstrap.modular_crossing import ModularGapSpec, build_modular_gap_problem


def test_modular_problems_are_refused_with_a_clear_error(tmp_path):
    with mp.workdps(40):
        problem = build_modular_gap_problem(ModularGapSpec(c=8, delta_gap=1, lam=3, spin_list=(0, 1), dps=40))
        with pytest.raises(TypeError, match="sphere four-point problems"):
            certify_dual_from_sdpb_output(problem, tmp_path)
        with pytest.raises(TypeError, match="sphere four-point problems"):
            certify_dual_functional([1], problem.spin_channels, [(0, 0)], coordinate="vvb")


def test_modular_channels_passed_as_an_iterator_are_refused_too():
    with mp.workdps(40):
        problem = build_modular_gap_problem(ModularGapSpec(c=8, delta_gap=1, lam=3, spin_list=(0, 1), dps=40))
        with pytest.raises(TypeError, match="sphere four-point problems"):
            certify_dual_functional([1], iter(problem.spin_channels), [(0, 0)], coordinate="vvb")
