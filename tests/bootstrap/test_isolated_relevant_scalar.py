from __future__ import annotations

import mpmath as mp

from virasoro_bootstrap.isolated_relevant_scalar import (
    IsolatedRelevantScalarSpec,
    build_isolated_relevant_scalar_problems,
    four_point_isolated_spec,
    modular_isolated_spec,
)


def test_isolated_relevant_scalar_specs_wire_shared_delta_phi():
    spec = IsolatedRelevantScalarSpec(
        c=8,
        delta_phi=mp.mpf("0.8"),
        delta_gap=2,
        modular_lam=3,
        four_point_lam=2,
        modular_spin_list=iter((0, 1, 2)),
        four_point_spin_list=iter((0, 2)),
        modular_dps=50,
        four_point_dps=50,
        q_order=8,
        npoles=4,
    )

    modular = modular_isolated_spec(spec)
    four_point = four_point_isolated_spec(spec)

    assert modular.isolated_scalars == ((spec.delta_phi, spec.degeneracy_phi),)
    assert modular.delta_gap == spec.delta_gap
    assert four_point.h_ext == spec.delta_phi / 2
    assert four_point.delta_gap == spec.delta_gap
    assert four_point.point_constraints == ((0, spec.delta_phi),)
    assert spec.modular_spin_list == (0, 1, 2) and spec.four_point_spin_list == (0, 2)
    assert modular_isolated_spec(spec) == modular
    assert four_point_isolated_spec(spec) == four_point


def test_modular_isolated_spec_can_impose_scalar_gap_only():
    from dataclasses import replace
    import numpy as np
    import pytest

    spec = IsolatedRelevantScalarSpec(
        c=8,
        delta_phi=mp.mpf("0.8"),
        delta_gap=2,
        modular_lam=3,
        modular_spin_list=(0, 1, 2, 3),
        modular_scalar_gap_only=True,
    )

    modular = modular_isolated_spec(spec)

    assert modular.delta_gap == 0
    assert modular.delta_min_by_spin == {0: spec.delta_gap}
    assert modular.isolated_scalars == ((spec.delta_phi, spec.degeneracy_phi),)
    all_spins = modular_isolated_spec(replace(spec, modular_scalar_gap_only=False))
    assert all_spins.delta_gap == spec.delta_gap and all_spins.delta_min_by_spin is None
    for flag in ('false', np.bool_(True)):
        with pytest.raises(ValueError, match='modular_scalar_gap_only.*True or False'):
            modular_isolated_spec(replace(spec, modular_scalar_gap_only=flag))


def test_modular_isolated_spec_can_allow_free_phi_degeneracy():
    from dataclasses import replace
    import pytest

    spec = IsolatedRelevantScalarSpec(
        c=8,
        delta_phi=mp.mpf("0.8"),
        delta_gap=2,
        modular_lam=3,
        modular_scalar_gap_only=True,
        modular_phi_mode="allowed",
    )

    modular = modular_isolated_spec(spec)

    assert modular.isolated_scalars == ()
    assert modular.allowed_isolated_scalars == (spec.delta_phi,)
    assert modular.delta_min_by_spin == {0: spec.delta_gap}
    for mode in ('typo', True):
        with pytest.raises(ValueError, match="modular_phi_mode.*fixed.*allowed"):
            modular_isolated_spec(replace(spec, modular_phi_mode=mode))


def test_build_isolated_relevant_scalar_problems_smoke():
    spec = IsolatedRelevantScalarSpec(
        c=8,
        delta_phi=mp.mpf("0.8"),
        delta_gap=2,
        modular_lam=3,
        four_point_lam=1,
        modular_spin_list=iter((0, 1)),
        four_point_spin_list=iter((0,)),
        modular_dps=50,
        four_point_dps=50,
        q_order=6,
        npoles=2,
    )

    problems = build_isolated_relevant_scalar_problems(spec)

    assert problems.modular.spec.isolated_scalars == ((spec.delta_phi, 1),)
    assert len(problems.modular.spin_channels) == 2
    assert problems.four_point.spec.point_constraints == ((0, spec.delta_phi),)
    assert any(
        channel.ell == 0 and mp.mpf(channel.delta_min) == spec.delta_phi
        for channel in problems.four_point.point_channels
    )
    assert modular_isolated_spec(spec).spin_list == (0, 1)
    assert four_point_isolated_spec(spec).spin_list == (0,)


def test_four_point_spec_preserves_exact_decimal_and_rational_external_weights():
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_number

    for value, expected in [(0.7, Fraction(7, 20)), (Fraction(1, 3), Fraction(1, 6))]:
        with mp.workdps(15):
            spec = four_point_isolated_spec(IsolatedRelevantScalarSpec(c=8, delta_phi=value))
        assert literal_number(spec.h_ext) == expected
        assert spec.point_constraints == ((0, value),)


def test_four_point_spec_does_not_round_a_high_precision_shared_weight():
    from virasoro_blocks.numeric_input import literal_number

    with mp.workdps(100):
        delta_phi = mp.mpf("0.8") + mp.mpf("1e-90")
        expected = literal_number(delta_phi) / 2
    with mp.workdps(15):
        shared = IsolatedRelevantScalarSpec(c=8, delta_phi=delta_phi)
        four_point = four_point_isolated_spec(shared)
        modular = modular_isolated_spec(shared)
        assert literal_number(four_point.h_ext) == expected
        assert literal_number(modular.isolated_scalars[0][0]) == 2 * expected


def test_isolated_scalar_build_accepts_decimal_shared_parameters():
    from decimal import Decimal

    spec = IsolatedRelevantScalarSpec(
        c=Decimal("8"), delta_phi=Decimal("0.7"), delta_gap=Decimal("2"),
        modular_lam=1, four_point_lam=1, modular_spin_list=(0,),
        four_point_spin_list=(0,), modular_dps=30, four_point_dps=30,
        q_order=4, npoles=2,
    )
    problems = build_isolated_relevant_scalar_problems(spec)
    from fractions import Fraction
    assert problems.four_point.point_channels[0].delta_min == Fraction(7, 10)
