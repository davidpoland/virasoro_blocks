"""Tests for the sample-point interpolation helpers."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.composer import G_block
from virasoro_blocks.coordinates import q_symmetric
from virasoro_blocks.derivs import G_zzb_table
from virasoro_blocks.interpolation import (
    DampedRational,
    DeltaInterpolant,
    build_delta_interpolant,
    build_interpolants_for_npoles,
    build_reduced_delta_polynomial,
    compute_interpolation_error_curve,
    find_bohr_sommerfeld_roots,
    get_sample_points,
    get_sample_points_jacobi,
    integrated_density,
    interpolation_error_curves_for_npoles,
    lagrange_interpolation_polynomial,
    max_abs_errors_by_npoles,
    normalize_delta_evaluator,
    normalize_delta_prefactor,
    ordered_block_delta_prefactor,
    plot_max_abs_error_by_npoles,
    ordered_block_delta_value,
    scalar_block_delta_prefactor,
    scalar_block_delta_value,
    solve_support_endpoint,
    support_endpoint_equation,
)


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def test_support_endpoint_solves_no_pole_case():
    mp.mp.dps = 50
    prefactor = DampedRational(constant=1, poles=(), base=mp.e ** (-mp.pi))
    b = solve_support_endpoint(prefactor, 5)
    expected = 10 / mp.pi
    assert _close(b, expected, tol=mp.mpf("1e-40"))
    assert _close(
        support_endpoint_equation(prefactor, b, 5),
        mp.mpf(0),
        tol=mp.mpf("1e-40"),
    )


def test_integrated_density_hits_target_values_for_sample_points():
    mp.mp.dps = 60
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.4"), mp.mpf("-1.2")),
        base=mp.e ** (-mp.pi),
    )
    sample_points = get_sample_points(prefactor, 4, work_dps=80)
    b = solve_support_endpoint(prefactor, 4, work_dps=80)
    for n, x in enumerate(sample_points):
        omega = integrated_density(prefactor, b, x)
        assert _close(omega, n + mp.mpf("0.5"), tol=mp.mpf("1e-30"))


def test_sample_points_are_increasing_and_inside_support():
    mp.mp.dps = 50
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.25"), mp.mpf("-0.75")),
        base=mp.e ** (-mp.pi),
    )
    sample_points = get_sample_points(prefactor, 5, work_dps=80)
    b = solve_support_endpoint(prefactor, 5, work_dps=80)
    assert len(sample_points) == 5
    assert all(sample_points[i] < sample_points[i + 1] for i in range(4))
    assert all(0 <= x <= b for x in sample_points)


def test_single_zero_pole_includes_zero_sample_point():
    """With boundary_offset = 0 (the strict matrix-model setting), a single
    real pole at x = 0 places an explicit sample point at x = 0."""
    mp.mp.dps = 60
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("0"), mp.mpf("-0.8")),
        base=mp.e ** (-mp.pi),
    )
    sample_points = get_sample_points(prefactor, 4, work_dps=80, boundary_offset=0)
    b = solve_support_endpoint(prefactor, 4, work_dps=80)
    assert sample_points[0] == 0
    for n, x in enumerate(sample_points[1:], start=1):
        omega = integrated_density(prefactor, b, x)
        assert _close(omega, n + mp.mpf("0.5"), tol=mp.mpf("1e-30"))


def test_genuine_near_zero_real_pole_gets_nodes():
    """c = 1 + 1e-12 has a genuine pole at x = -2.5e-13 (it reaches 0 at c = 1), 27 orders above the precision
    floor: the whole table build refused it as an 'unremoved Kac zero'. Only a roundoff image still raises."""
    with mp.workdps(40):
        pre = DampedRational(constant=1, poles=(mp.mpf("-2.5e-13"), mp.mpf("-0.8")), base=mp.e ** (-mp.pi))
        points = get_sample_points(pre, 6, work_dps=40)
        assert len(points) == 6 and all(x > 0 for x in points)


def test_near_zero_nonzero_real_pole_raises_loudly():
    """A real pole within the zero-classification band but not *exactly* at 0 is
    the roundoff image of an exact Kac zero (e.g. the identity pole at a c < 1
    minimal model) that should have been stripped by `regularize_delta_table`.
    The two consumers of the band disagree on it (delta-at-origin vs regular
    pole), which silently produces nan nodes -- so it must fail loudly instead.
    An *exactly* zero pole is unaffected (it is the legitimate boundary pole)."""
    mp.mp.dps = 80
    near_zero = DampedRational(
        constant=1,
        poles=(mp.mpf("-3.426e-81"), mp.mpf("-0.8")),
        base=mp.e ** (-mp.pi),
    )
    with pytest.raises(ValueError, match="near-zero Kac pole"):
        get_sample_points(near_zero, 6, work_dps=80)
    # An exactly-zero pole stays valid (handled as the boundary delta).
    exact_zero = DampedRational(
        constant=1,
        poles=(mp.mpf("0"), mp.mpf("-0.8")),
        base=mp.e ** (-mp.pi),
    )
    points = get_sample_points(exact_zero, 6, work_dps=80, boundary_offset=0)
    assert len(points) == 6


def test_higher_order_zero_pole_is_not_implemented_strict_boundary():
    """With boundary_offset = 0, higher multiplicity at x = 0 is unsupported
    (distinct degenerate sample points cannot be expressed)."""
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("0"), mp.mpf("0"), mp.mpf("-1")),
        base=mp.e ** (-mp.pi),
    )
    with pytest.raises(NotImplementedError):
        get_sample_points(prefactor, 4, boundary_offset=0)


def test_higher_order_zero_pole_collapsed_by_boundary_offset():
    """With boundary_offset > 0 (the default), multiplicity at x = 0 is
    collapsed into a single lifted sample point; the rest of the count
    comes from the Bohr-Sommerfeld matrix-model roots."""
    mp.mp.dps = 60
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("0"), mp.mpf("0"), mp.mpf("-1")),
        base=mp.e ** (-mp.pi),
    )
    epsilon = mp.mpf("1e-20")
    sample_points = get_sample_points(
        prefactor, 4, work_dps=80, boundary_offset=epsilon
    )
    assert len(sample_points) == 4
    assert sample_points[0] == epsilon
    assert all(sample_points[i] < sample_points[i + 1] for i in range(3))


def test_find_bohr_sommerfeld_roots_validates_inputs():
    with pytest.raises(ValueError):
        find_bohr_sommerfeld_roots(
            lambda x: x,
            -1,
            3,
            upper_bound=mp.mpf("1"),
            initial_guess=mp.mpf("0.1"),
        )
    with pytest.raises(ValueError):
        find_bohr_sommerfeld_roots(
            lambda x: x,
            4,
            3,
            upper_bound=mp.mpf("1"),
            initial_guess=mp.mpf("0.1"),
        )


def test_ordered_block_delta_value_matches_B_block():
    mp.mp.dps = 50
    Delta = mp.mpf("2.8")
    ell = -1
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.35")
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]

    direct = ordered_block_delta_value(
        Delta,
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        c=mp.mpf("1.7"),
        q_order=10,
    )
    h = (Delta + ell) / 2
    hbar = (Delta - ell) / 2
    from virasoro_blocks.composer import B_block

    expected = B_block(h, hbar, z, zbar, h_ext, hbar_ext, c=mp.mpf("1.7"), q_order=10)
    assert _close(direct, expected, tol=mp.mpf("1e-40"))


def test_ordered_block_delta_prefactor_matches_symbolic_poles():
    mp.mp.dps = 60
    ell = -1
    raw_prefactor = ordered_block_delta_prefactor(
        ell,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")],
        [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")],
        c=mp.mpf("30"),
        q_order=4,
    )
    shift = raw_prefactor.rightmost_real_pole + 1
    shifted_prefactor = ordered_block_delta_prefactor(
        ell,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")],
        [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")],
        shift=shift,
        c=mp.mpf("30"),
        q_order=4,
    )
    assert raw_prefactor.pole_degree > 0
    assert raw_prefactor.base > 0
    assert all(mp.re(pole) <= 0 for pole in shifted_prefactor.poles)


def test_half_integer_ordered_block_prefactor_interpolates_at_nodes():
    """The ordered-B interpolation helpers support half-integer exchanged spin
    on the same sample-point path used for integer spin."""
    mp.mp.dps = 60
    ell = mp.mpf("1.5")
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.35")
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    raw_prefactor = ordered_block_delta_prefactor(
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        c=mp.mpf("30"),
        q_order=4,
    )
    shift = raw_prefactor.rightmost_real_pole + 1
    prefactor = ordered_block_delta_prefactor(
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        shift=shift,
        c=mp.mpf("30"),
        q_order=4,
    )
    sample_xs = get_sample_points(prefactor.take_rightmost_poles(4), 5, work_dps=80)
    sample_deltas = [x + shift for x in sample_xs]
    evaluator = lambda Delta: ordered_block_delta_value(
        Delta,
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        c=mp.mpf("30"),
        q_order=4,
    )
    interpolant = build_delta_interpolant(sample_deltas, evaluator, prefactor.take_rightmost_poles(4), shift=shift)

    for Delta in sample_deltas:
        assert _close(interpolant.evaluate(Delta), evaluator(Delta), tol=mp.mpf("1e-28"))


def test_normalize_delta_prefactor_divides_by_delta_exponential():
    mp.mp.dps = 60
    shift = mp.mpf("2")
    prefactor = DampedRational(
        constant=mp.mpf("3"),
        poles=(mp.mpf("-0.5"), mp.mpf("-1.5")),
        base=mp.mpf("12"),
    )
    normalization_base = mp.mpf("4")
    normalized = normalize_delta_prefactor(
        prefactor,
        normalization_base,
        shift=shift,
    )

    for Delta in [mp.mpf("2.25"), mp.mpf("3.5")]:
        x = Delta - shift
        expected = prefactor.evaluate(x) / normalization_base**Delta
        assert _close(normalized.evaluate(x), expected, tol=mp.mpf("1e-40"))


def test_normalize_delta_evaluator_divides_by_delta_exponential():
    mp.mp.dps = 60
    evaluator = lambda Delta: (mp.mpf("5") ** Delta) * (Delta + 1)
    normalized = normalize_delta_evaluator(evaluator, mp.mpf("5"))
    for Delta in [mp.mpf("0"), mp.mpf("1.25"), mp.mpf("3")]:
        assert _close(normalized(Delta), Delta + 1, tol=mp.mpf("1e-40"))


def test_self_dual_prefactor_base_matches_chiral_normalization():
    """The damped-rational base at z = zbar = 1/2 equals E(1/2) = 16 q(1/2) / N,
    with N = DEFAULT_CHIRAL_NORM. With the default N = 16 that is q(1/2)."""
    from virasoro_blocks.recursion import DEFAULT_CHIRAL_NORM
    mp.mp.dps = 60
    ell = 0
    z = mp.mpf("0.5")
    zbar = mp.mpf("0.5")
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    raw_prefactor = ordered_block_delta_prefactor(
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        c=mp.mpf("30"),
        q_order=4,
    )
    shift = raw_prefactor.rightmost_real_pole + mp.mpf("1e-6")
    shifted_prefactor = ordered_block_delta_prefactor(
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        shift=shift,
        c=mp.mpf("30"),
        q_order=4,
    )
    expected_base = 16 * q_symmetric() / DEFAULT_CHIRAL_NORM
    assert _close(shifted_prefactor.base, expected_base, tol=mp.mpf("1e-40"))
    normalized = normalize_delta_prefactor(
        shifted_prefactor,
        shifted_prefactor.base / q_symmetric(),
        shift=shift,
    )
    assert _close(normalized.base, q_symmetric(), tol=mp.mpf("1e-40"))


def test_scalar_block_delta_prefactor_matches_ordered_scalar_case():
    mp.mp.dps = 60
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    shift = mp.mpf("3")
    scalar_prefactor = scalar_block_delta_prefactor(
        2,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        h_ext,
        shift=shift,
        c=mp.mpf("30"),
        q_order=4,
    )
    ordered_prefactor = ordered_block_delta_prefactor(
        2,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        h_ext,
        h_ext,
        shift=shift,
        c=mp.mpf("30"),
        q_order=4,
    )
    assert _close(scalar_prefactor.constant, ordered_prefactor.constant, tol=mp.mpf("1e-40"))
    assert _close(scalar_prefactor.base, ordered_prefactor.base, tol=mp.mpf("1e-40"))
    assert scalar_prefactor.poles == ordered_prefactor.poles


def test_get_sample_points_rejects_non_damping_base():
    prefactor = DampedRational(constant=1, poles=(mp.mpf("-0.5"),), base=mp.mpf("1.1"))
    with pytest.raises(ValueError):
        get_sample_points(prefactor, 3)


def test_get_sample_points_rejects_positive_poles():
    prefactor = DampedRational(constant=1, poles=(mp.mpf("0.1"),), base=mp.e ** (-mp.pi))
    with pytest.raises(ValueError):
        get_sample_points(prefactor, 3)


def test_lagrange_interpolation_polynomial_reproduces_samples():
    mp.mp.dps = 50
    sample_points = [mp.mpf("1.0"), mp.mpf("1.5"), mp.mpf("2.0")]
    values = [mp.mpf("2.0"), mp.mpf("3.25"), mp.mpf("5.0")]
    poly = lagrange_interpolation_polynomial(sample_points, values, shift=mp.mpf("1.0"))
    for x, y in zip(sample_points, values):
        assert _close(poly.evaluate(x - 1), y, tol=mp.mpf("1e-40"))


def test_build_reduced_delta_polynomial_matches_ordered_block_samples():
    mp.mp.dps = 60
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.5"), mp.mpf("-1.5")),
        base=mp.e ** (-mp.pi),
    )
    sample_points = [mp.mpf("2.1"), mp.mpf("2.5"), mp.mpf("2.9")]
    shift = mp.mpf("2.0")
    evaluator = lambda Delta: ordered_block_delta_value(
        Delta,
        -1,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")],
        [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")],
        c=mp.mpf("1.7"),
        q_order=10,
    )

    poly = build_reduced_delta_polynomial(sample_points, evaluator, prefactor, shift=shift)

    for Delta in sample_points:
        reconstructed = prefactor.evaluate(Delta - shift) * poly.evaluate(Delta - shift)
        assert _close(reconstructed, evaluator(Delta), tol=mp.mpf("1e-35"))


def test_scalar_block_delta_value_and_interpolant_match_samples():
    mp.mp.dps = 60
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.5"),),
        base=mp.e ** (-mp.pi),
    )
    sample_points = [mp.mpf("2.2"), mp.mpf("2.6"), mp.mpf("3.0")]
    evaluator = lambda Delta: scalar_block_delta_value(
        Delta,
        2,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")],
        c=mp.mpf("1.7"),
        q_order=10,
    )

    poly = build_reduced_delta_polynomial(sample_points, evaluator, prefactor, shift=0)

    for Delta in sample_points:
        reconstructed = prefactor.evaluate(Delta) * poly.evaluate(Delta)
        assert _close(reconstructed, evaluator(Delta), tol=mp.mpf("1e-35"))


def test_build_delta_interpolant_reconstructs_samples():
    mp.mp.dps = 60
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.5"), mp.mpf("-1.0")),
        base=mp.e ** (-mp.pi),
    )
    sample_points = [mp.mpf("2.1"), mp.mpf("2.5"), mp.mpf("2.9")]
    evaluator = lambda Delta: ordered_block_delta_value(
        Delta,
        -1,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")],
        [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")],
        c=mp.mpf("1.7"),
        q_order=10,
    )

    interpolant = build_delta_interpolant(sample_points, evaluator, prefactor, shift=2)
    assert isinstance(interpolant, DeltaInterpolant)
    for Delta in sample_points:
        assert _close(interpolant.evaluate(Delta), evaluator(Delta), tol=mp.mpf("1e-35"))


def test_derived_prefactor_interpolant_matches_ordered_block_off_grid():
    mp.mp.dps = 70
    ell = -1
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.35")
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    raw_prefactor = ordered_block_delta_prefactor(
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        c=mp.mpf("30"),
        q_order=3,
    )
    shift = raw_prefactor.rightmost_real_pole + 1
    prefactor = ordered_block_delta_prefactor(
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        shift=shift,
        c=mp.mpf("30"),
        q_order=3,
    )
    evaluator = lambda Delta: ordered_block_delta_value(
        Delta,
        ell,
        z,
        zbar,
        h_ext,
        hbar_ext,
        c=mp.mpf("30"),
        q_order=3,
    )
    sample_points = [shift + mp.mpf("0.2") + mp.mpf(k) * mp.mpf("0.08") for k in range(prefactor.pole_degree + 1)]
    interpolant = build_delta_interpolant(sample_points, evaluator, prefactor, shift=shift)
    for Delta in [shift + mp.mpf("0.25"), shift + mp.mpf("0.45"), shift + mp.mpf("0.65")]:
        assert _close(interpolant.evaluate(Delta), evaluator(Delta), tol=mp.mpf("1e-12"))


def test_compute_interpolation_error_curve_vanishes_at_sample_points():
    mp.mp.dps = 60
    prefactor = DampedRational(constant=1, poles=(mp.mpf("-0.5"),), base=mp.e ** (-mp.pi))
    sample_points = [mp.mpf("2.2"), mp.mpf("2.6"), mp.mpf("3.0")]
    evaluator = lambda Delta: scalar_block_delta_value(
        Delta,
        2,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")],
        c=mp.mpf("1.7"),
        q_order=10,
    )
    interpolant = build_delta_interpolant(sample_points, evaluator, prefactor)
    errors = compute_interpolation_error_curve(sample_points, evaluator, interpolant, mode="relative")
    for error in errors:
        assert error < mp.mpf("1e-30")


def test_compute_interpolation_difference_curve_vanishes_at_sample_points():
    mp.mp.dps = 60
    prefactor = DampedRational(constant=1, poles=(mp.mpf("-0.5"),), base=mp.e ** (-mp.pi))
    sample_points = [mp.mpf("2.2"), mp.mpf("2.6"), mp.mpf("3.0")]
    evaluator = lambda Delta: scalar_block_delta_value(
        Delta,
        2,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")],
        c=mp.mpf("1.7"),
        q_order=10,
    )
    interpolant = build_delta_interpolant(sample_points, evaluator, prefactor)
    differences = compute_interpolation_error_curve(sample_points, evaluator, interpolant, mode="difference")
    for difference in differences:
        assert abs(difference) < mp.mpf("1e-30")


def test_build_interpolants_for_npoles_tracks_requested_counts():
    mp.mp.dps = 50
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.2"), mp.mpf("-0.5"), mp.mpf("-1.0")),
        base=mp.e ** (-mp.pi),
    )
    evaluator = lambda Delta: Delta + 1
    interpolants = build_interpolants_for_npoles(
        prefactor,
        [0, 2],
        evaluator,
        num_sample_points=3,
        sample_point_kwargs={"work_dps": 60},
    )
    assert sorted(interpolants) == [0, 2]
    assert interpolants[0].prefactor.pole_degree == 0
    assert interpolants[2].prefactor.pole_degree == 2


def test_interpolation_error_curves_for_npoles_shape():
    mp.mp.dps = 50
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.2"), mp.mpf("-0.5"), mp.mpf("-1.0")),
        base=mp.e ** (-mp.pi),
    )
    Delta_values = [mp.mpf("1.2"), mp.mpf("1.5"), mp.mpf("1.8")]
    evaluator = lambda Delta: Delta**2 + 1
    curves, interpolants = interpolation_error_curves_for_npoles(
        Delta_values,
        evaluator,
        prefactor,
        [1, 2],
        num_sample_points=3,
        sample_point_kwargs={"work_dps": 60},
    )
    assert sorted(curves) == [1, 2]
    assert sorted(interpolants) == [1, 2]
    for npoles in (1, 2):
        assert len(curves[npoles]) == len(Delta_values)


def test_max_abs_errors_by_npoles_summarizes_curves():
    curves = {
        2: [mp.mpf("1e-2"), -mp.mpf("3e-3"), mp.mpf("5e-2")],
        4: [-mp.mpf("2e-4"), mp.mpf("9e-5")],
    }
    summary = max_abs_errors_by_npoles(curves)
    assert _close(summary[2], mp.mpf("5e-2"), tol=mp.mpf("1e-40"))
    assert _close(summary[4], mp.mpf("2e-4"), tol=mp.mpf("1e-40"))


def test_plot_max_abs_error_by_npoles_rejects_zero_error():
    pytest.importorskip("matplotlib.pyplot")
    with pytest.raises(ValueError):
        plot_max_abs_error_by_npoles({2: [mp.mpf("0")]})


def test_take_rightmost_poles_does_not_overshoot_npoles_for_conjugate_pairs():
    mp.mp.dps = 50
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpc("-0.1", "0.7"), mp.mpc("-0.1", "-0.7"), mp.mpf("-0.5")),
        base=mp.e ** (-mp.pi),
    )
    reduced = prefactor.take_rightmost_poles(1)
    assert reduced.pole_degree == 1
    assert _close(reduced.poles[0], mp.mpf("-0.5"), tol=mp.mpf("1e-40"))
    reduced = prefactor.take_rightmost_poles(3)
    assert reduced.pole_degree == 3


def test_interpolation_mode_matches_derivative_mode_on_scalar_block():
    """Cross-check direct evaluation, Taylor reconstruction, and interpolation.

    At a fixed (z, zbar) slightly off the symmetric point, three routes should
    reproduce the same scalar block G_{Delta, ell}(z, zbar):

        (a) direct G_block (the reference),
        (b) derivative-mode reconstruction: build G_zzb_table at z = zbar = 1/2
            for this Delta and Taylor-sum with offsets (dz, dzbar),
        (c) interpolation-mode reconstruction: build a DeltaInterpolant at
            this (z, zbar) over Bohr-Sommerfeld sample points and evaluate.

    Tolerances:
        - (c) vs (a): the interpolant is built from samples of G_block itself,
            so the residual is interpolation error of the (regular, polynomial)
            ratio inside the sample-point interval. We test at midpoints of
            adjacent sample intervals where Lebesgue constants are small.
        - (b) vs (a): truncation-limited by the dropped (m + n > m_max + n_max)
            Taylor terms. Chiral coefficients grow ~ 2^k (convergence radius 1/2),
            so for offsets ~0.03 and m_max = n_max = 6 the residual sits a few
            times 1e-9.
    """
    mp.mp.dps = 50
    ell = 2
    # c chosen large enough that b is real (b becomes complex below c ~ 1.85),
    # which keeps the prefactor poles real as the DampedRational class requires.
    c = mp.mpf("30")
    h_ext = [mp.mpf("0.15")] * 4  # scalar externals
    q_order = 3

    dz = mp.mpf("0.03")
    dzbar = mp.mpf("0.02")
    z = mp.mpf(1) / 2 + dz
    zbar = mp.mpf(1) / 2 + dzbar
    m_max = 6
    n_max = 6

    # Interpolation-mode setup: shift so all prefactor poles sit to the left of 0.
    raw_prefactor = scalar_block_delta_prefactor(
        ell, z, zbar, h_ext, c=c, q_order=q_order
    )
    shift = raw_prefactor.rightmost_real_pole + mp.mpf("0.5")
    prefactor = scalar_block_delta_prefactor(
        ell, z, zbar, h_ext, shift=shift, c=c, q_order=q_order
    )
    evaluator = lambda Delta: scalar_block_delta_value(
        Delta, ell, z, zbar, h_ext, c=c, q_order=q_order
    )
    sample_xs = get_sample_points(prefactor, prefactor.pole_degree + 1)
    sample_deltas = [x + shift for x in sample_xs]
    interpolant = build_delta_interpolant(
        sample_deltas, evaluator, prefactor, shift=shift
    )

    # Off-grid test Delta values: midpoints of adjacent sample intervals.
    test_deltas = [
        (sample_deltas[0] + sample_deltas[1]) / 2,
        (sample_deltas[len(sample_deltas) // 2]
            + sample_deltas[len(sample_deltas) // 2 + 1]) / 2,
        (sample_deltas[-2] + sample_deltas[-1]) / 2,
    ]

    for Delta in test_deltas:
        truth = G_block(Delta, ell, z, zbar, h_ext, c=c, q_order=q_order)

        G_interp = interpolant.evaluate(Delta)

        table = G_zzb_table(Delta, ell, h_ext, m_max, n_max, c=c, q_order=q_order)
        G_deriv = mp.mpc(0)
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                G_deriv += table[m][n] * dz**m * dzbar**n

        # Both reconstructions agree with the truth ...
        assert _close(G_interp, truth, tol=mp.mpf("1e-25"))
        assert _close(G_deriv, truth, tol=mp.mpf("1e-7"))
        # ... and therefore with each other.
        assert _close(G_deriv, G_interp, tol=mp.mpf("1e-7"))


# --- complex-conjugate-pair pole support (full unitary c >= 1) ---


def test_damped_rational_accepts_conjugate_pair():
    """A pair of conjugate complex poles is allowed; evaluate(x) stays real
    for real x."""
    mp.mp.dps = 50
    p = mp.mpc("-0.5", "0.3")
    prefactor = DampedRational(
        constant=1, poles=(p, mp.conj(p)), base=mp.e ** (-mp.pi)
    )
    val = prefactor.evaluate(mp.mpf("0.2"))
    assert abs(mp.im(val)) < mp.mpf("1e-40")
    # And it equals 1 / |x - p|^2 * base^x.
    x = mp.mpf("0.2")
    expected = mp.exp(-mp.pi * x) / ((x - p) * (x - mp.conj(p)))
    assert _close(val, expected, tol=mp.mpf("1e-40"))


def test_damped_rational_rejects_unpaired_complex_pole():
    """A lone non-real pole is rejected: the prefactor would not be real on the
    real axis."""
    with pytest.raises(ValueError):
        DampedRational(
            constant=1,
            poles=(mp.mpc("-0.5", "0.3"),),
            base=mp.e ** (-mp.pi),
        )


def test_damped_rational_rightmost_real_pole_uses_real_part():
    """rightmost_real_pole takes Re of all poles, ignoring imaginary parts."""
    mp.mp.dps = 50
    prefactor = DampedRational(
        constant=1,
        poles=(
            mp.mpc("-0.5", "0.3"),
            mp.mpc("-0.5", "-0.3"),
            mp.mpf("-2"),
        ),
        base=mp.e ** (-mp.pi),
    )
    assert _close(prefactor.rightmost_real_pole, mp.mpf("-0.5"), tol=mp.mpf("1e-40"))


def test_damped_rational_take_rightmost_respects_npoles_cap():
    """take_rightmost_poles(N) never exceeds N actual poles."""
    mp.mp.dps = 50
    p = mp.mpc("-0.5", "0.3")
    prefactor = DampedRational(
        constant=1,
        poles=(p, mp.conj(p), mp.mpf("-2")),
        base=mp.e ** (-mp.pi),
    )
    trimmed = prefactor.take_rightmost_poles(1)
    assert trimmed.pole_degree == 1
    assert _close(trimmed.poles[0], mp.mpf("-2"), tol=mp.mpf("1e-40"))
    # Asking for 3 returns everything.
    assert prefactor.take_rightmost_poles(3).pole_degree == 3


def _abs_set(poles):
    return sorted(round(float(abs(p)), 6) for p in poles)


def test_take_nearest_poles_picks_smallest_magnitude():
    """take_nearest_poles keeps the smallest-|pole| conjugate pairs; this
    differs from take_rightmost_poles when a rightmost pole has a large Im."""
    mp.mp.dps = 50
    near = mp.mpc("-1.0", "1.0")          # |p| ~ 1.41
    far = mp.mpc("-0.5", "40.0")          # rightmost Re, but |p| ~ 40
    prefactor = DampedRational(
        constant=1,
        poles=(near, mp.conj(near), far, mp.conj(far)),
        base=mp.e ** (-mp.pi),
    )
    nearest = prefactor.take_nearest_poles(2)
    rightmost = prefactor.take_rightmost_poles(2)
    assert nearest.pole_degree == 2
    assert rightmost.pole_degree == 2
    # nearest keeps the small-|p| pair, rightmost keeps the large-Im pair.
    assert _abs_set(nearest.poles) == _abs_set([near, mp.conj(near)])
    assert _abs_set(rightmost.poles) == _abs_set([far, mp.conj(far)])


def test_take_nearest_poles_respects_npoles_cap():
    """take_nearest_poles(N) never exceeds N actual poles."""
    mp.mp.dps = 50
    p = mp.mpc("-1.0", "1.0")
    prefactor = DampedRational(
        constant=1,
        poles=(p, mp.conj(p), mp.mpf("-9")),
        base=mp.e ** (-mp.pi),
    )
    trimmed = prefactor.take_nearest_poles(1)
    assert trimmed.pole_degree == 1
    assert _close(trimmed.poles[0], mp.mpf("-9"), tol=mp.mpf("1e-40"))


def test_take_strongest_poles_picks_largest_score():
    """take_strongest_poles keeps the poles with the largest externally
    supplied score."""
    mp.mp.dps = 50
    a = mp.mpc("-1.0", "2.0")
    b = mp.mpc("-3.0", "4.0")
    prefactor = DampedRational(
        constant=1,
        poles=(a, mp.conj(a), b, mp.conj(b)),
        base=mp.e ** (-mp.pi),
    )
    # Scores aligned with poles: the b-pair scores higher.
    scores = [mp.mpf(1), mp.mpf(1), mp.mpf(10), mp.mpf(10)]
    kept = prefactor.take_strongest_poles(2, scores)
    assert kept.pole_degree == 2
    assert _abs_set(kept.poles) == _abs_set([b, mp.conj(b)])


def test_take_strongest_poles_rejects_misaligned_scores():
    mp.mp.dps = 50
    p = mp.mpc("-1.0", "1.0")
    prefactor = DampedRational(
        constant=1, poles=(p, mp.conj(p)), base=mp.e ** (-mp.pi)
    )
    with pytest.raises(ValueError):
        prefactor.take_strongest_poles(1, [mp.mpf(1)])  # 1 score, 2 poles


def test_support_endpoint_handles_conjugate_pair_poles():
    """Support endpoint solves for a complex-conjugate-pair pole prefactor."""
    mp.mp.dps = 60
    p = mp.mpc("-0.4", "0.2")
    prefactor = DampedRational(
        constant=1, poles=(p, mp.conj(p)), base=mp.e ** (-mp.pi)
    )
    b = solve_support_endpoint(prefactor, 5, work_dps=80)
    # The equation residual at the solved b should be tiny.
    assert _close(
        support_endpoint_equation(prefactor, b, 5),
        mp.mpf(0),
        tol=mp.mpf("1e-40"),
    )
    # Compare to the no-pole baseline 10/pi; the conjugate pair pulls b
    # somewhat. Just sanity-check positivity and order of magnitude.
    assert b > mp.mpf("0.5")
    assert b < mp.mpf("10")


def test_integrated_density_is_real_for_conjugate_pair():
    """omega(x) evaluates to a real number for a conjugate-pair prefactor."""
    mp.mp.dps = 60
    p = mp.mpc("-0.4", "0.2")
    prefactor = DampedRational(
        constant=1, poles=(p, mp.conj(p)), base=mp.e ** (-mp.pi)
    )
    b = solve_support_endpoint(prefactor, 4, work_dps=80)
    for x in [mp.mpf("0.1"), mp.mpf("0.5"), mp.mpf("1.0")]:
        omega = integrated_density(prefactor, b, x)
        # omega returns mp.mpf, but verify it's not silently dropping a large
        # imaginary part.
        assert isinstance(omega, mp.mpf) or abs(mp.im(omega)) < mp.mpf("1e-40")


def test_get_sample_points_handles_conjugate_pair_poles():
    """Sample-point generator runs end-to-end with a conjugate-pair prefactor."""
    mp.mp.dps = 60
    p = mp.mpc("-0.4", "0.2")
    prefactor = DampedRational(
        constant=1, poles=(p, mp.conj(p)), base=mp.e ** (-mp.pi)
    )
    points = get_sample_points(prefactor, 5, work_dps=80)
    assert len(points) == 5
    assert all(points[i] < points[i + 1] for i in range(4))
    b = solve_support_endpoint(prefactor, 5, work_dps=80)
    assert all(0 <= x <= b for x in points)
    for n, x in enumerate(points):
        omega = integrated_density(prefactor, b, x)
        assert _close(omega, n + mp.mpf("0.5"), tol=mp.mpf("1e-25"))


def test_damped_rational_preserves_tiny_conjugate_pair():
    """An explicit tiny-imaginary conjugate pair should remain complex rather
    than being snapped to a fake real double pole."""
    mp.mp.dps = 60
    eps = mp.mpf("1e-50")
    p = mp.mpc(mp.mpf("0.5"), eps)
    prefactor = DampedRational(
        constant=1,
        poles=(p, mp.conj(p)),
        base=mp.e ** (-mp.pi),
    )
    assert len(prefactor._pair_data) == 1
    assert any(abs(mp.im(pole)) > 0 for pole in prefactor.poles)


def test_ordered_block_prefactor_unitary_low_c_has_complex_pole_pairs():
    """At c = 5 (1 < c < 25) the Liouville parameter b is complex, so the
    d_{r,s} pole locations come in conjugate pairs. The prefactor builder
    should accept this without dropping the imaginary parts."""
    mp.mp.dps = 50
    raw_prefactor = ordered_block_delta_prefactor(
        -1,
        mp.mpf("0.4"),
        mp.mpf("0.35"),
        [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")],
        [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")],
        c=mp.mpf("5"),
        q_order=3,
    )
    # At least one pole has nonzero Im (complex b territory).
    assert any(abs(mp.im(p)) > mp.mpf("1e-10") for p in raw_prefactor.poles)
    # Real on the real axis.
    sample_x = mp.mpf("2.5") - raw_prefactor.rightmost_real_pole - 1
    val = raw_prefactor.shift_x(raw_prefactor.rightmost_real_pole + 1).evaluate(sample_x)
    assert abs(mp.im(val)) < mp.mpf("1e-30")


def test_interpolation_mode_unitary_low_c_matches_direct_block():
    """Interpolation cross-check at c = 10 (unitary, b complex). The DeltaInterpolant
    built from a complex-conjugate-paired prefactor reproduces the scalar block
    at off-grid Delta values."""
    mp.mp.dps = 50
    ell = 2
    c = mp.mpf("10")
    h_ext = [mp.mpf("0.15")] * 4
    q_order = 3

    z = mp.mpf(1) / 2 + mp.mpf("0.03")
    zbar = mp.mpf(1) / 2 + mp.mpf("0.02")

    raw_prefactor = scalar_block_delta_prefactor(
        ell, z, zbar, h_ext, c=c, q_order=q_order
    )
    # Sanity: complex poles present.
    assert any(abs(mp.im(p)) > mp.mpf("1e-10") for p in raw_prefactor.poles)

    shift = raw_prefactor.rightmost_real_pole + mp.mpf("0.5")
    prefactor = scalar_block_delta_prefactor(
        ell, z, zbar, h_ext, shift=shift, c=c, q_order=q_order
    )
    evaluator = lambda Delta: scalar_block_delta_value(
        Delta, ell, z, zbar, h_ext, c=c, q_order=q_order
    )
    sample_xs = get_sample_points(prefactor, prefactor.pole_degree + 1)
    sample_deltas = [x + shift for x in sample_xs]
    interpolant = build_delta_interpolant(
        sample_deltas, evaluator, prefactor, shift=shift
    )

    test_deltas = [
        (sample_deltas[0] + sample_deltas[1]) / 2,
        (sample_deltas[len(sample_deltas) // 2]
            + sample_deltas[len(sample_deltas) // 2 + 1]) / 2,
        (sample_deltas[-2] + sample_deltas[-1]) / 2,
    ]
    for Delta in test_deltas:
        truth = G_block(Delta, ell, z, zbar, h_ext, c=c, q_order=q_order)
        approx = interpolant.evaluate(Delta)
        assert _close(approx, truth, tol=mp.mpf("1e-20"))


# ---------------------------------------------------------------------------
# Relaxed constraint: complex poles with Re(pole) > 0 are admitted; only real
# positive poles are rejected. Motivated by the strip-c bootstrap setup where
# `shift = Delta_unit` puts the surviving "rightmost" complex poles at large
# Re(x) > 0 (with |Im| >> 0), and sample-point coverage of the physical
# positivity interval [Delta_unit, ...) requires that case to succeed.
# ---------------------------------------------------------------------------


def test_support_endpoint_equation_accepts_complex_pole_with_positive_re():
    """Conjugate-pair poles with Re > 0 (and Im != 0) are admitted: on the
    real axis the pair is a smooth strictly-positive factor, not a singularity
    inside [0, b]."""
    mp.mp.dps = 60
    p = mp.mpc("3", "5")    # Re > 0, |Im| > 0
    prefactor = DampedRational(
        constant=1, poles=(p, mp.conj(p)), base=mp.e ** (-mp.pi)
    )
    b = solve_support_endpoint(prefactor, 5, work_dps=80)
    assert b > 0
    # The equation should evaluate to ~0 at the solution.
    residual = support_endpoint_equation(prefactor, b, 5)
    assert _close(residual, 0, tol=mp.mpf("1e-25"))


def test_support_endpoint_newton_solves_the_equation_to_working_precision():
    """The flint precision-ladder Newton solves the endpoint equation to working precision for real, zero, and
    conjugate-pair poles (one pair at Re > 0)."""
    mp.mp.dps = 60
    p, q = mp.mpc("-0.7", "2.5"), mp.mpc("3", "5")
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf(0), mp.mpf("-0.4"), mp.mpf("-1.2"), p, mp.conj(p), q, mp.conj(q)),
        base=mp.e ** (-mp.pi),
    )
    for n in (6, 20):
        b = solve_support_endpoint(prefactor, n, work_dps=80)
        assert b > 0
        assert abs(support_endpoint_equation(prefactor, b, n)) < mp.mpf(10) ** -55


def _mixed_prefactor():
    """Real poles (one close to 0), a zero pole, and conjugate pairs: one ordinary, one at Re > 0, one with a
    tiny imaginary part and one far out."""
    pairs = (mp.mpc("-0.7", "2.5"), mp.mpc("3", "5"), mp.mpc("-2", "0.001"), mp.mpc("40", "3000"))
    poles = (mp.mpf(0), mp.mpf("-1e-8"), mp.mpf("-0.4"), mp.mpf("-12"))
    poles += tuple(q for p in pairs for q in (p, mp.conj(p)))
    return DampedRational(constant=1, poles=poles, base=3 - 2 * mp.sqrt(2))


def test_flint_integrated_density_matches_the_acos_form():
    """The atan2 form of omega (one real atan2 per pole or pair) and its density equal `integrated_density`
    (acos of complex arguments) and its derivative across the support."""
    from virasoro_blocks.interpolation import _make_flint_integrated_density

    mp.mp.dps = 60
    prefactor = _mixed_prefactor()
    b = solve_support_endpoint(prefactor, 30, work_dps=120)
    density = _make_flint_integrated_density(prefactor, b)
    for t in ("1e-12", "1e-6", "0.01", "0.3", "0.5", "0.77", "0.999", "0.99999999"):
        x = b * mp.mpf(t)
        with mp.workdps(120):  # the acos form loses digits next to the near-zero pole and the tiny-Im pair
            reference = integrated_density(prefactor, b, x)
            slope = mp.diff(lambda y: integrated_density(prefactor, b, y), x)
        assert abs(mp.mpf(density.omega(x).mid()._mpf_) - reference) < mp.mpf(10) ** -57
        assert abs(mp.mpf(density.density(x).mid()._mpf_) / slope - 1) < mp.mpf(10) ** -50


@pytest.mark.parametrize("dps,count", [(30, 3), (60, 40), (200, 25)])
def test_bohr_sommerfeld_nodes_are_certified_roots(dps, count):
    """Every node solves omega(x) = n + 1/2 to below the working precision (checked with the independent mpmath
    `integrated_density`), for poles near 0, a zero pole (lifted) and conjugate pairs with Re > 0."""
    mp.mp.dps = dps
    prefactor = _mixed_prefactor()
    nodes = get_sample_points(prefactor, count)
    b = solve_support_endpoint(prefactor, count, work_dps=2 * dps)
    ulp = mp.mpf(2) ** -mp.mp.prec
    assert len(nodes) == count
    for n, x in enumerate(nodes[1:], start=1):
        with mp.workdps(2 * dps):
            residual = integrated_density(prefactor, b, x) - (n + mp.mpf(1) / 2)
            slope = mp.diff(lambda t: integrated_density(prefactor, b, t), x)
        assert abs(residual / slope) < ulp * x / 4


def test_get_sample_points_accepts_complex_pole_with_positive_re():
    """End-to-end: nodes can be generated for a conjugate-pair prefactor
    whose Re(pole) > 0. The returned nodes are real and lie in (0, b)."""
    mp.mp.dps = 60
    p = mp.mpc("3", "5")
    prefactor = DampedRational(
        constant=1, poles=(p, mp.conj(p)), base=mp.e ** (-mp.pi)
    )
    points = get_sample_points(prefactor, 5, work_dps=80)
    b = solve_support_endpoint(prefactor, 5, work_dps=80)
    assert len(points) == 5
    assert all(0 < x < b for x in points)
    assert all(points[i] < points[i + 1] for i in range(4))


def test_get_sample_points_still_rejects_real_positive_pole():
    """A real pole at Re > 0 lies inside [0, b] and is a genuine singularity
    of the matrix-model density; the constraint still rejects it."""
    prefactor = DampedRational(constant=1, poles=(mp.mpf("0.1"),), base=mp.e ** (-mp.pi))
    with pytest.raises(ValueError, match="real poles"):
        get_sample_points(prefactor, 3)


def test_support_endpoint_equation_still_rejects_real_positive_pole():
    prefactor = DampedRational(constant=1, poles=(mp.mpf("0.5"),), base=mp.e ** (-mp.pi))
    # The actual support endpoint b > 0; the equation should refuse regardless.
    with pytest.raises(ValueError, match="real poles"):
        support_endpoint_equation(prefactor, mp.mpf("1.0"), 3)


def test_strip_c_interpolation_at_bootstrap_shift_covers_unitarity():
    """The motivating use case: at strip c = 10 with bootstrap-natural shift
    `shift = Delta_min` (the lower endpoint of the SDPB sampling interval,
    > 0 for the ell = 0 channel), the rightmost complex poles end up at
    Re(x) > 0. The relaxed constraint lets the sample-point generator run,
    and the resulting sample points cover the small-Delta bootstrap region
    instead of clustering far above it (as they would at the default shift,
    where sample Delta in [62, 77]).

    Absolute error on the sampling range stays well below typical SDPB
    tolerances.
    """
    from virasoro_blocks.params import b_from_c
    from virasoro_blocks.polyderivs import G_zzb_poly_table, interpolate_delta_table
    from virasoro_blocks.derivs import G_zzb_table

    mp.mp.dps = 80
    b_param = b_from_c(mp.mpf("10"))
    ext = [mp.mpf("0.15")] * 4
    ell, lam, q_order, npoles = 0, 4, 20, 10
    delta_min = mp.mpf("0.5")  # bootstrap "gap" -- lower endpoint of sampling

    # Node-spread assertions below are calibrated for the N=4 base (4q); pin
    # chiral_norm=4 so the test is independent of the library default (N=16,
    # whose steeper base q deliberately concentrates the nodes).
    exact = G_zzb_poly_table(ell, ext, lam, lam, b=b_param, q_order=q_order, chiral_norm=4, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=npoles, shift=delta_min)

    sample_deltas = [
        mp.re(x) + mp.re(interp.shift) for x in interp.sample_points
    ]
    # All sample points sit in the bootstrap positivity range [Delta_min, ...).
    assert all(d >= delta_min for d in sample_deltas)
    assert min(sample_deltas) < mp.mpf("2.0")  # starts near Delta_min
    assert max(sample_deltas) > mp.mpf("10.0")  # extends well above

    # Absolute interpolation error on [Delta_min, support_endpoint] is small.
    for Delta in [mp.mpf("0.6"), mp.mpf("1.0"), mp.mpf("2.0"), mp.mpf("5.0")]:
        truth = G_zzb_table(Delta, ell, ext, lam, lam, b=b_param, q_order=q_order, chiral_norm=4)
        approx = interp.evaluate(Delta)
        abs_err = max(
            abs(mp.mpc(a) - mp.mpc(t))
            for row_a, row_t in zip(approx, truth)
            for a, t in zip(row_a, row_t)
        )
        assert abs_err < mp.mpf("1e-6"), (Delta, abs_err)


def test_shared_sample_points_per_entry_vs_shared_node_set():
    """`shared_sample_points=True` fits every entry on the single max-count node
    set; the per-entry default sizes each entry's Bohr-Sommerfeld set to its own
    degree.  Measured *per entry* (the table-wide max metric is misleading -- its
    worst entry is the max-degree one, which uses the same nodes in both):

      * in the node range both reproduce the true block (shared marginally
        better, more nodes),
      * beyond the node range the per-entry default is far more accurate -- the
        shared high-degree fit of a low-degree entry Runge-blows-up.

    Beyond-range fidelity is what certification needs (positivity on the true
    block at large Delta), so per-entry is the right default.
    """
    from virasoro_blocks.params import b_from_c
    from virasoro_blocks.polyderivs import G_zzb_poly_table, interpolate_delta_table
    from virasoro_blocks.derivs import G_zzb_table

    mp.mp.dps = 60
    b_param = b_from_c(mp.mpf("2"))
    ext = [mp.mpf("0.0625")] * 4
    ell, lam, q_order, npoles = 0, 4, 18, 8
    shift = mp.mpf("0.0")

    # Node-range / Runge assertions are calibrated for the N=4 base; pin
    # chiral_norm=4 (library default is now N=16, base q).
    exact = G_zzb_poly_table(ell, ext, lam, lam, b=b_param, q_order=q_order, chiral_norm=4, lambda_cut=None)
    per = interpolate_delta_table(
        exact, npoles=npoles, shift=shift, shared_sample_points=False
    )
    shared = interpolate_delta_table(
        exact, npoles=npoles, shift=shift, shared_sample_points=True
    )

    # Every shared entry uses one node set; the per-entry default uses several.
    shared_counts = {len(s) for row in shared.numerator_sample_points for s in row}
    per_counts = {len(s) for row in per.numerator_sample_points for s in row}
    assert len(shared_counts) == 1
    assert len(per_counts) > 1

    def worst_per_entry(interp, deltas):
        trues = [
            G_zzb_table(d, ell, ext, lam, lam, b=b_param, q_order=q_order, chiral_norm=4)
            for d in deltas
        ]
        approxs = [interp.evaluate(d) for d in deltas]
        worst = mp.mpf(0)
        for m in range(lam + 1):
            for n in range(lam + 1):
                scale = max(abs(mp.mpc(t[m][n])) for t in trues)
                scale = max(scale, mp.mpf("1e-40"))
                err = max(
                    abs(mp.mpc(a[m][n]) - mp.mpc(t[m][n]))
                    for a, t in zip(approxs, trues)
                )
                worst = max(worst, err / scale)
        return worst

    node_max = max(mp.re(mp.mpc(x)) for x in per.sample_points)
    in_range = [mp.mpf("0.5"), mp.mpf("1.5"), node_max * mp.mpf("0.7")]
    beyond = [node_max * mp.mpf("2.0"), node_max * mp.mpf("4.0")]

    # In range both are accurate.
    assert worst_per_entry(per, in_range) < mp.mpf("1e-3")
    assert worst_per_entry(shared, in_range) < mp.mpf("1e-3")

    # Beyond the node range the per-entry default is dramatically better
    # (shared Runge-blows-up by many orders of magnitude).
    per_beyond = worst_per_entry(per, beyond)
    shared_beyond = worst_per_entry(shared, beyond)
    assert shared_beyond > per_beyond * mp.mpf("1e3"), (per_beyond, shared_beyond)


# ---------------------------------------------------------------------------
# Jacobi-matrix sample-point algorithm tests
# ---------------------------------------------------------------------------


def test_jacobi_sample_points_positive_and_increasing():
    """Jacobi sample points should be positive and strictly increasing."""
    mp.mp.dps = 50
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.25"), mp.mpf("-0.75")),
        base=mp.e ** (-mp.pi),
    )
    pts = get_sample_points_jacobi(prefactor, 5, work_dps=80)
    assert len(pts) == 5
    assert all(x > 0 for x in pts)
    assert all(pts[i] < pts[i + 1] for i in range(4))


def test_jacobi_no_poles_gives_truncated_gauss_nodes():
    """With no poles, the Jacobi approach gives Gauss nodes for e^{-αx} on [0, b].

    The support endpoint for w(x) = e^{-πx} with N=5 is b = 2N/π ≈ 3.183.
    The nodes should be positive, increasing, and lie within [0, b].
    """
    mp.mp.dps = 50
    base = mp.e ** (-mp.pi)
    prefactor = DampedRational(constant=1, poles=(), base=base)
    pts = get_sample_points_jacobi(prefactor, 5, work_dps=80)
    # Support endpoint: b = 2N/π for pure exponential w(x) = e^{-πx}
    b_expected = 2 * 5 / mp.pi
    assert len(pts) == 5
    assert all(pt > 0 for pt in pts)
    assert all(pts[i] < pts[i + 1] for i in range(4))
    # All nodes should lie within the support [0, b]
    assert pts[-1] < b_expected, (float(pts[-1]), float(b_expected))


def test_jacobi_handles_zero_pole_with_boundary_offset():
    """A zero pole produces a boundary_offset fixed point + Gauss nodes."""
    mp.mp.dps = 60
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("0"), mp.mpf("-0.8")),
        base=mp.e ** (-mp.pi),
    )
    pts = get_sample_points_jacobi(prefactor, 4, work_dps=80)
    assert len(pts) == 4
    # First point is the boundary offset
    assert pts[0] > 0
    assert pts[0] < mp.mpf("1e-10")  # very small
    # Rest are increasing and positive
    assert all(pts[i] < pts[i + 1] for i in range(3))


def test_jacobi_complex_conjugate_pair_poles():
    """Complex-conjugate-pair poles should produce real positive nodes."""
    mp.mp.dps = 50
    prefactor = DampedRational(
        constant=1,
        poles=(
            mp.mpc("-0.5", "0.8"), mp.mpc("-0.5", "-0.8"),
            mp.mpf("-2.0"),
        ),
        base=mp.e ** (-mp.pi),
    )
    pts = get_sample_points_jacobi(prefactor, 8, work_dps=80)
    assert len(pts) == 8
    assert all(x > 0 for x in pts)
    assert all(pts[i] < pts[i + 1] for i in range(7))


def test_jacobi_polynomial_interpolation_exact():
    """Interpolating a degree-(N-1) polynomial times prefactor should be exact."""
    mp.mp.dps = 60
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.4"), mp.mpf("-1.2")),
        base=mp.e ** (-mp.pi),
    )
    N = 6
    # Target: prefactor * (1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120)
    def target(x):
        poly = sum(x ** k / mp.factorial(k) for k in range(N))
        return prefactor.evaluate(x) * poly

    pts = get_sample_points_jacobi(prefactor, N, work_dps=80)
    interp = build_delta_interpolant(list(pts), target, prefactor, shift=0)

    # Check at various points
    for x_test in [mp.mpf("0.3"), mp.mpf("1.0"), mp.mpf("2.5"), mp.mpf("5.0")]:
        err = abs(interp.evaluate(x_test) - target(x_test))
        assert err < mp.mpf("1e-40"), (float(x_test), float(err))


def test_jacobi_method_dispatch_via_get_sample_points():
    """get_sample_points(method='jacobi') delegates to get_sample_points_jacobi."""
    mp.mp.dps = 50
    prefactor = DampedRational(
        constant=1,
        poles=(mp.mpf("-0.4"), mp.mpf("-1.2")),
        base=mp.e ** (-mp.pi),
    )
    direct = get_sample_points_jacobi(prefactor, 5, work_dps=80)
    via_method = get_sample_points(prefactor, 5, work_dps=80, method="jacobi")
    assert len(direct) == len(via_method)
    for d, m in zip(direct, via_method):
        assert _close(d, m, tol=mp.mpf("1e-40"))


def test_jacobi_rejects_invalid_method():
    """get_sample_points raises ValueError for unknown method."""
    prefactor = DampedRational(constant=1, poles=(), base=mp.e ** (-mp.pi))
    with pytest.raises(ValueError, match="method must be"):
        get_sample_points(prefactor, 3, method="invalid")
