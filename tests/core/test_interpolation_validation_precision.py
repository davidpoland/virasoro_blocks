"""Interpolation domain checks and explicitly distinct conjugate pole locations."""

from collections import Counter

import mpmath as mp
import pytest

from virasoro_blocks.interpolation import (
    DampedRational,
    DeltaInterpolant,
    _compute_damped_rational_moments,
    _jacobi_coefficients,
    get_sample_points_jacobi,
    solve_support_endpoint,
    support_endpoint_equation,
)
from virasoro_blocks.polynomial import Polynomial


@pytest.mark.parametrize("dps", [1, 5, 10, 15, 40])
@pytest.mark.parametrize("argument", ["base", "query", "regulator", "endpoint", "guess", "sample", "difference"])
def test_material_imaginary_inputs_are_rejected_at_low_precision(dps, argument):
    with mp.workdps(dps):
        prefactor = DampedRational(1, (-1,), ".5")
        interpolant = DeltaInterpolant(prefactor, 0, (1,), Polynomial([1]))
        with pytest.raises(ValueError, match="real"):
            if argument == "base":
                DampedRational(1, (), mp.mpc(1, ".1"))
            elif argument == "query":
                prefactor.evaluate(1j)
            elif argument == "regulator":
                prefactor.evaluate_regulated(1, mp.mpc(1, ".1"))
            elif argument == "endpoint":
                support_endpoint_equation(prefactor, mp.mpc(2, ".1"), 3)
            elif argument == "guess":
                solve_support_endpoint(prefactor, 3, initial_guess=mp.mpc(2, ".1"))
            elif argument == "sample":
                DeltaInterpolant(prefactor, 0, (1j,), Polynomial([1]))
            else:
                interpolant.signed_difference(lambda _: 1j, 1)
        assert mp.mp.dps == dps


@pytest.mark.parametrize("dps", [1, 5, 10, 40])
@pytest.mark.parametrize("poles", [(-1 + .1j,), (-1 + .1j, -.5 - .1j)])
def test_materially_unpaired_poles_are_rejected(dps, poles):
    with mp.workdps(dps), pytest.raises(ValueError, match="conjugate"):
        DampedRational(1, poles, ".5")


@pytest.mark.parametrize("dps", [5, 10, 40])
def test_valid_conjugates_still_evaluate_the_real_quadratic(dps):
    with mp.workdps(dps):
        pole = mp.mpc(-1, ".1")
        prefactor = DampedRational(1, (pole, pole.conjugate()), 1)
        assert mp.im(prefactor.poles[0]) != 0
        assert abs(prefactor.evaluate(1) * (4 + pole.imag**2) - 1) < mp.eps * 8


def _tiny_pair(real=-1):
    with mp.workdps(180):
        upper = mp.mpc(real, "1e-70")
        return (upper, upper.conjugate()), mp.exp(-1)


@pytest.mark.parametrize("selector", ["nearest", "rightmost", "real_first", "real_deep", "strongest"])
@pytest.mark.parametrize("multiplicity", [1, 2])
def test_pole_selection_cannot_duplicate_tiny_conjugate_pairs(selector, multiplicity):
    poles, base = _tiny_pair()
    with mp.workdps(40):
        prefactor = DampedRational(1, poles * multiplicity + (-2,), base)
        available = Counter(prefactor.poles)
        for cap in range(9):
            if selector == "strongest":
                reduced = prefactor.take_strongest_poles(cap, list(range(prefactor.pole_degree)))
            elif selector == "real_deep":
                reduced = prefactor.take_real_first_poles(cap, double_first=True)
            else:
                reduced = getattr(prefactor, f"take_{selector}_poles")(cap)
            selected = Counter(reduced.poles)
            assert reduced.pole_degree <= cap
            assert selected <= available
            assert selected[poles[0]] == selected[poles[1]]
            if selector.startswith("real_") and cap == 1:
                assert reduced.poles == (mp.mpc(-2),)
        assert selected == available


def test_lone_near_real_roundoff_pole_keeps_existing_canonicalization():
    with mp.workdps(40):
        prefactor = DampedRational(1, (mp.mpc(-1, "1e-70"),), 1)
        assert prefactor.poles == (mp.mpc(-1),)


def test_large_finite_poles_keep_their_ranking_multiplicities_and_conjugates():
    with mp.workdps(50):
        upper = mp.mpc('-1.7e308', '1.7e308')
        far, near = mp.mpc('-2e400'), mp.mpc('-1e400')
        for poles in ((far, near, far, near, upper, upper.conjugate()),
                      (mp.mpc(-1), upper, upper.conjugate())):
            prefactor = DampedRational(1, poles, 1)
            available = Counter(poles)
            for selector in ('nearest', 'rightmost', 'real_first', 'strongest'):
                for cap in range(len(poles) + 1):
                    if selector == 'strongest':
                        reduced = prefactor.take_strongest_poles(cap, list(range(len(poles))))
                    else:
                        reduced = getattr(prefactor, f'take_{selector}_poles')(cap)
                    selected = Counter(reduced.poles)
                    assert selected <= available and reduced.pole_degree <= cap
                    assert selected[upper] == selected[upper.conjugate()]
                    if cap == len(poles):
                        assert selected == available
            if far in available:
                assert prefactor.take_real_first_poles(1).poles == (near,)
                assert prefactor.take_real_first_poles(2).poles == (near, far)
                assert prefactor.take_real_first_poles(2, double_first=True).poles == (near, near)
        assert mp.mp.dps == 50


def test_node_search_controls_require_integer_counts_and_precisions():
    import numpy as np
    from virasoro_blocks.interpolation import (
        _make_flint_integrated_density, find_bohr_sommerfeld_roots, get_sample_points,
    )
    from virasoro_blocks.polyderivs import DeltaDerivTable, interpolate_delta_tables

    with mp.workdps(30):
        prefactor = DampedRational(1, (), '.5')
        density = _make_flint_integrated_density(prefactor, 2)
        calls = ((solve_support_endpoint, prefactor, {}),
                 (get_sample_points, prefactor, {}),
                 (get_sample_points_jacobi, prefactor, {}),
                 (find_bohr_sommerfeld_roots, density,
                  dict(n_start=0, upper_bound=2, initial_guess='.1')))
        for function, first, keywords in calls:
            for bad in (True, np.bool_(True), np.bool_(False), 1.5, '3', mp.nan):
                with pytest.raises(ValueError, match='num_sample_points.*integer'):
                    function(first, num_sample_points=bad, **keywords)
            for bad in (True, np.bool_(True), 20.5, '20', 0, -1):
                with pytest.raises(ValueError, match='work_dps'):
                    function(first, num_sample_points=1, work_dps=bad, **keywords)
        for bad in (True, np.bool_(False), -1, 3.5, '3'):
            with pytest.raises(ValueError, match='guard_dps'):
                get_sample_points_jacobi(prefactor, 1, guard_dps=bad)
        for bad in (True, np.bool_(True), -1, 1.5):
            with pytest.raises(ValueError, match='n_start'):
                find_bohr_sommerfeld_roots(density, bad, 2, upper_bound=2, initial_guess='.1')
        # Integral numerical scalars remain valid; zero extra guard digits
        # and an empty range in the lower-level root finder are permitted.
        for function in (get_sample_points, get_sample_points_jacobi):
            assert len(function(prefactor, np.int64(1), work_dps=20.0)) == 1
        assert len(get_sample_points_jacobi(prefactor, 1.0, guard_dps=0)) == 1
        assert find_bohr_sommerfeld_roots(density, 0.0, 0.0, upper_bound=2, initial_guess='.1') == []
        forced = DeltaDerivTable(mp.mpf('.5'), Polynomial([-1, 1]), [1],
                                 [[Polynomial([3, 1])]], spin=0)
        for name in ('n_sample_points', 'node_search_dps'):
            for bad in (True, np.bool_(False), 0, -1, 1.5, '3'):
                with pytest.raises(ValueError, match=name):
                    interpolate_delta_tables({'zzb': forced}, None, **{name: bad})


def test_jacobi_coefficients_reject_a_complex_endpoint_and_restore_precision():
    poles, base = _tiny_pair()
    with mp.workdps(10):
        with pytest.raises(ValueError, match="support_endpoint.*real"):
            _jacobi_coefficients(DampedRational(1, poles, base), 2+.1j, 3, 40)
        assert mp.mp.dps == 10


