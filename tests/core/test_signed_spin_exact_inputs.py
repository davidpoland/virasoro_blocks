"""Exact signed-spin labels through ordered and parity-projected block paths."""

from decimal import Decimal
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks import derivs, polyderivs
from virasoro_blocks.composer import B_block, G_block
from virasoro_blocks.global_blocks import build_global_sampler, global_ordered_B_interpolated_table, global_scalar_G_interpolated_table
from virasoro_blocks.interpolation import ordered_block_delta_prefactor, ordered_block_delta_value
from virasoro_blocks.numeric_input import _exchange_weight, _signed_half_integer, literal_components, literal_number


HE = (".1", ".2", ".3", ".4")
HBE = (".15", ".25", ".35", ".45")


def _call(kind, spin):
    if kind == "value":
        return ordered_block_delta_value("2.5", spin, ".3", ".45", HE, HBE, c=30, q_order=2)
    if kind == "prefactor":
        return ordered_block_delta_prefactor(spin, ".3", ".45", HE, HBE, c=30, q_order=2)
    if kind == "numeric":
        return derivs.parity_projected_B_zzb_table("2.5", spin, HE, HBE, 0, 0,
                                                 c=30, q_order=2, parity="+")
    if kind == "global":
        return derivs.global_parity_projected_B_zzb_table("2.5", spin, HE, HBE, 0, 0, parity="+")
    if kind == "symbolic":
        return polyderivs.B_zzb_poly_table(spin, HE, HBE, 0, 0, c=30, q_order=2)
    if kind == "global_interp":
        return global_ordered_B_interpolated_table("zzb", spin, HE, HBE, 0, 0,
                                                   npoles=2, q_order=2, pole_selection="nearest")
    if kind == "sampler":
        return build_global_sampler("zzb", basis="global_ordered_B", ell=spin,
                                    h_externals=HE, hbar_externals=HBE, m_max=0, n_max=0)(3)
    return polyderivs.parity_projected_B_xt_poly_table(spin, HE, HBE, 0, 0,
                                                     c=30, q_order=2, parity="+")


@pytest.mark.parametrize("kind", ["value", "prefactor", "numeric", "global", "symbolic", "folded", "global_interp", "sampler"])
@pytest.mark.parametrize("spin", [".5000000000000000000000000000000000000000000000000000000000000000000001", True])
def test_public_ordered_paths_refuse_invalid_spin_without_rounding(kind, spin):
    with mp.workdps(5):
        with pytest.raises(ValueError, match="integer or half-integer"):
            _call(kind, spin)
        assert mp.mp.dps == 5


@pytest.mark.parametrize("spin", [Fraction(1, 2), Decimal("-.5"), ".5", "-1/2", -.5, np.int64(-2)])
def test_signed_spin_validator_accepts_exact_real_labels(spin):
    with mp.workdps(5):
        expected = Fraction(str(spin))
        assert _signed_half_integer(spin) == expected


@pytest.mark.parametrize("value", [None, mp.inf, mp.nan, 1j, "bad", Fraction(1, 3), ".3", np.bool_(False)])
def test_signed_spin_validator_gives_a_domain_error(value):
    with pytest.raises(ValueError, match="finite real integer or half-integer"):
        _signed_half_integer(value)


def test_numpy_floating_spins_retain_binary_labels_without_rounding():
    with mp.workdps(40):
        expected_value = _call("value", Fraction(1, 2))
        expected_table = _call("symbolic", Fraction(-1, 2)).evaluate(3)
        expected_scalar = G_block("2.5", -1, ".3", ".45", HE, c=30, q_order=2)
        for dtype in (np.float16, np.float32, np.longdouble):
            assert _call("value", dtype(.5)) == expected_value
            assert _call("symbolic", dtype(-.5)).evaluate(3) == expected_table
            assert G_block("2.5", dtype(-1), ".3", ".45", HE, c=30, q_order=2) == expected_scalar
    with mp.workdps(5):
        for dtype in (np.float16, np.float32, np.longdouble):
            assert _signed_half_integer(dtype(.5)) == Fraction(1, 2)
            assert _signed_half_integer(dtype(-1)) == -1
            shifted = np.nextafter(dtype(.5), dtype(1))
            with pytest.raises(ValueError, match="integer or half-integer"):
                _signed_half_integer(shifted)
        large = np.float32(1e30)
        assert _signed_half_integer(large) == Fraction(*large.as_integer_ratio())
        assert mp.mp.dps == 5


def test_stored_high_precision_spin_offset_is_not_snapped_under_lower_precision():
    with mp.workdps(180):
        offset = mp.mpf(".5") + mp.mpf("1e-70")
    with mp.workdps(40), pytest.raises(ValueError, match="integer or half-integer"):
        _signed_half_integer(offset)


@pytest.mark.parametrize("spin", [Fraction(1, 2), Fraction(-3, 2)])
@pytest.mark.parametrize("kind", ["value", "prefactor", "numeric", "global", "symbolic", "folded", "global_interp", "sampler"])
def test_fraction_spins_match_exact_dyadic_controls(kind, spin):
    with mp.workdps(40):
        got = _call(kind, spin)
        expected = _call(kind, mp.mpf(str(float(spin))))
        if kind == "prefactor":
            assert got.poles == expected.poles
            assert got.evaluate(3) == expected.evaluate(3)
        elif kind in {"symbolic", "folded", "global_interp"}:
            if kind != "global_interp":
                assert got.spin == spin
            assert got.evaluate(3) == expected.evaluate(3)
        else:
            assert got == expected


@pytest.mark.parametrize("spin", [Fraction(1, 2), Fraction(-3, 2), Fraction(2*10**80+1, 2)])
def test_complex_exchange_weight_split_preserves_stored_components_exactly(spin):
    with mp.workdps(180):
        delta = mp.mpc(mp.mpf(".5")+mp.mpf("1e-70"), "1e-80")
    with mp.workdps(10):
        real, imaginary = literal_components(delta)
        assert literal_components(_exchange_weight(delta, spin)) == ((real+spin)/2, imaginary/2)
        assert mp.mp.dps == 10


@pytest.mark.parametrize("kind", ["value", "numeric", "interp", "sampler"])
@pytest.mark.parametrize("spin", [Fraction(1, 2), True, np.bool_(False)])
def test_scalar_G_paths_reject_half_integer_spins_and_booleans(kind, spin):
    with mp.workdps(40), pytest.raises(ValueError, match="integer"):
        if kind == "value":
            G_block("2.5", spin, ".3", ".45", HE, c=30, q_order=2)
        elif kind == "numeric":
            derivs.global_G_zzb_table("2.5", spin, HE, 0, 0)
        elif kind == "interp":
            global_scalar_G_interpolated_table("zzb", spin, HE, 0, 0, npoles=2, q_order=2)
        else:
            build_global_sampler("zzb", basis="global_scalar_G", ell=spin,
                                 h_externals=HE, hbar_externals=HE, m_max=0, n_max=0)


def test_scalar_G_preserves_integer_decimal_labels_and_signed_parity():
    with mp.workdps(40):
        positive = G_block("2.5", "1.0", ".3", ".45", HE, c=30, q_order=2)
        negative = G_block("2.5", Decimal("-1"), ".3", ".45", HE, c=30, q_order=2)
        # the scalar G is symmetric in h <-> hbar, so ell -> -ell is the same block (no (-1)^ell sign)
        assert positive == negative


def test_global_sampler_records_its_ordered_basis_and_exact_spin():
    sampler = build_global_sampler("zzb", basis="global_ordered_B", ell=Fraction(-1, 2),
                                   h_externals=HE, hbar_externals=HBE, m_max=0, n_max=0)
    assert sampler.basis == "global_ordered_B"
    assert sampler.ell == Fraction(-1, 2)


def test_real_complex_spin_is_accepted_but_any_resolved_imaginary_part_is_refused():
    with mp.workdps(180):
        real = mp.mpc(".5", 0)
        complex_spin = mp.mpc(".5", "1e-70")
    with mp.workdps(10):
        assert _signed_half_integer(real) == Fraction(1, 2)
        with pytest.raises(ValueError, match="real"):
            _signed_half_integer(complex_spin)


@pytest.mark.parametrize("label", [".5+0j", ".5000000000000000000000000000000000000000000000000000000000000000000001+0j"])
def test_spin_text_uses_real_literal_syntax_before_any_rounding(label):
    with mp.workdps(5), pytest.raises(ValueError, match="real"):
        _signed_half_integer(label)


