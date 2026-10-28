"""NS/R full-block spin inputs follow the integer-only CLI contract."""

from decimal import Decimal
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks.numeric_input import _integer_spin, literal_components, literal_number
from virasoro_blocks.superconformal_blocks import (
    SuperBlockSampler2d, _split_super_exchange, numeric_full_table,
    super_2d_interpolated_table, super_scalar_G_block,
)


HE = tuple(map(literal_number, (".1", ".2", ".3", ".4")))
HBE = tuple(map(literal_number, (".15", ".25", ".35", ".45")))


@pytest.mark.parametrize("spin", [True, np.bool_(False), Fraction(1, 2), ".3", mp.inf, 1j])
@pytest.mark.parametrize("dps", [5, 40])
def test_integer_spin_validator_rejects_unsupported_types(spin, dps):
    with mp.workdps(dps), pytest.raises(ValueError, match="integer"):
        _integer_spin(spin)


@pytest.mark.parametrize("family", ["ns", "ramond"])
@pytest.mark.parametrize("basis", ["scalar_G", "ordered_B", "parity_projected_B"])
@pytest.mark.parametrize("interpolated", [False, True])
@pytest.mark.parametrize("spin", [True, Fraction(1, 2)])
def test_full_super_builders_reject_unsupported_spins_before_sampling(family, basis, interpolated, spin):
    kwargs = dict(c=30, q_order=0, m_max=0, n_max=0, parity="+", hbar_ext=HBE)
    with mp.workdps(5):
        with pytest.raises(ValueError, match="integer"):
            if interpolated:
                super_2d_interpolated_table(family, basis, "zzb", spin, HE, npoles=1, **kwargs)
            else:
                numeric_full_table(family, basis, "2.5", spin, "zzb", HE, **kwargs)
        assert mp.mp.dps == 5


@pytest.mark.parametrize("family", ["ns", "ramond"])
@pytest.mark.parametrize("spin", [True, Fraction(1, 2), ".3", mp.nan])
def test_scalar_super_point_value_rejects_unsupported_spin(family, spin):
    with mp.workdps(40):
        with pytest.raises(ValueError, match="integer"):
            super_scalar_G_block(family, "2.5", spin, ".3", ".45", HE, c=30, q_order=0)
        assert mp.mp.dps == 40


@pytest.mark.parametrize("family", ["ns", "ramond"])
@pytest.mark.parametrize("spin", ["1.0", Decimal("-1"), np.int64(2), Fraction(-2)])
@pytest.mark.parametrize("basis", ["scalar_G", "ordered_B", "parity_projected_B"])
def test_integer_literal_spins_preserve_signed_parity_and_ordered_values(family, spin, basis):
    value = int(literal_number(spin))
    kwargs = dict(c=30, q_order=2, m_max=0, n_max=0, parity="+", hbar_ext=HBE,
                  center=".3", center_bar=".45")
    with mp.workdps(40):
        got = numeric_full_table(family, basis, "4.5", spin, "zzb", HE, **kwargs)
        control = numeric_full_table(family, basis, "4.5", value, "zzb", HE, **kwargs)
        assert got == control
        if basis == "scalar_G":
            args = (family, "4.5", spin, ".3", ".45", HE)
            got_point = super_scalar_G_block(*args, c=30, q_order=2)
            control_point = super_scalar_G_block(family, "4.5", value, ".3", ".45", HE, c=30, q_order=2)
            assert got_point == control_point
        assert mp.mp.dps == 40


@pytest.mark.parametrize("spin", [True, Fraction(1, 2), ".3"])
def test_sampler_constructor_never_truncates_unsupported_spin(spin):
    with pytest.raises(ValueError, match="integer"):
        SuperBlockSampler2d("ns", "zzb", spin, HE, HBE, c=30, q_order=0,
                            m_max=0, n_max=0, sigma=0, fold_parity=0,
                            chiral_norm=None, z_shift=None, zbar_shift=None,
                            sector=0, branch=0, center=None, center_bar=None)


def test_super_exchange_split_preserves_stored_complex_offset_under_lower_precision():
    with mp.workdps(180):
        delta = mp.mpc(-1+mp.mpf("1e-70"), "1e-80")
        real, imaginary = literal_components(delta)
    with mp.workdps(10):
        retained, left, right = _split_super_exchange(delta, 1)
        assert literal_components(retained) == (real, imaginary)
        assert literal_components(left) == ((real+1)/2, imaginary/2)
        assert literal_components(right) == ((real-1)/2, imaginary/2)
        assert mp.mp.dps == 10


def test_super_exchange_split_retains_real_decimal_and_stored_binary_labels():
    with mp.workdps(180):
        stored = 1+mp.mpf("1e-70")
    with mp.workdps(10):
        for delta in ("1.0000000000000000000000000000000000000000000000000000000000000000000001", stored):
            retained, left, right = _split_super_exchange(delta, -1)
            source = literal_number(delta)
            assert retained == source
            assert left == (source-1)/2 and right == (source+1)/2
