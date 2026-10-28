"""Scalar spin validation and conditioning through direct and cached builders."""

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks import polyderivs
from virasoro_blocks.derivs import G_zzb_table
from virasoro_blocks.composer import G_block
from virasoro_blocks.numeric_input import literal_number


HE = (".1", ".2", ".3", ".4")
COORDINATES = ("zzb", "yyb", "qqb", "xt", "ws", "qqb_ws",
               "xt_radial", "ws_radial", "qqb_ws_radial")


@pytest.mark.parametrize("coordinate", COORDINATES)
@pytest.mark.parametrize("spin", [True, Fraction(1, 2), ".3"])
def test_cached_scalar_builder_validates_integer_spin_before_parity(coordinate, spin):
    with mp.workdps(40):
        cache = polyderivs.build_chiral_cache([coordinate], HE, lambda_=0, c=30, q_order=2)
        left, right = cache[coordinate]
        with pytest.raises(ValueError, match="integer"):
            polyderivs.build_table_from_cache(
                coordinate, left, right, spin, m_max=0, n_max=0, basis="scalar_G",
            )
        assert mp.mp.dps == 40


@pytest.mark.parametrize("coordinate", COORDINATES)
def test_cached_scalar_builder_accepts_integer_decimal_labels(coordinate):
    with mp.workdps(40):
        cache = polyderivs.build_chiral_cache([coordinate], HE, lambda_=0, c=30, q_order=2)
        left, right = cache[coordinate]
        kwargs = dict(m_max=0, n_max=0, basis="scalar_G")
        got = polyderivs.build_table_from_cache(coordinate, left, right, "1.0", **kwargs)
        control = polyderivs.build_table_from_cache(coordinate, left, right, 1, **kwargs)
        assert got.spin == 1
        assert got.evaluate(3) == control.evaluate(3)
