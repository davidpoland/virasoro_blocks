"""Tests for the canonical serialized form of the block derivative tables."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.derivs import G_qqb_table, G_zzb_table
from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyderivs import (
    DeltaDerivTable,
    G_qqb_poly_table,
    G_ws_poly_table,
    G_xt_poly_table,
    G_zzb_poly_table,
    interpolate_delta_table,
    interpolate_delta_tables,
)
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.serialization import (
    SerializedDerivTable,
    serialize_deriv_tables,
)


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def _max_table_error(table_a, table_b):
    worst = mp.mpf(0)
    for row_a, row_b in zip(table_a, table_b):
        for entry_a, entry_b in zip(row_a, row_b):
            worst = max(worst, abs(mp.mpc(entry_a) - mp.mpc(entry_b)))
    return worst


C = mp.mpf("10")
SCALAR_EXT = [mp.mpf("0.15")] * 4


def test_serialize_exact_reproduces_table():
    """The serialized form of a DeltaDerivTable reproduces the block, checked
    against the independent numeric derivative table.

    `DeltaDerivTable.evaluate` is not a fair reference here: it evaluates a
    degree-~44 polynomial with O(1e20) coefficients in the *unshifted* Delta
    and loses many digits to cancellation. The serialized form evaluates in
    the shifted x, which is far better conditioned, so it is compared to the
    `mp.taylor`-based numeric table instead.
    """
    mp.mp.dps = 50
    b = b_from_c(C)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    serialized = serialize_deriv_tables({"zzb": exact})
    assert isinstance(serialized, SerializedDerivTable)
    for Delta in [mp.mpf("2.3"), mp.mpf("3.8"), mp.mpf("6.0")]:
        numeric = G_zzb_table(Delta, 2, SCALAR_EXT, 2, 2, b=b, q_order=8)
        err = _max_table_error(serialized.evaluate("zzb", Delta), numeric)
        assert err < mp.mpf("1e-25")


def test_serialize_exact_default_shift_is_unitarity_bound():
    """The default shift for a 2d-basis table is |spin| (the unitarity bound),
    which makes the polynomial variable `x = Delta - shift` start at x = 0 on
    the SDPB positivity interval `[Delta_unitarity, ...)`.
    """
    mp.mp.dps = 50
    b = b_from_c(C)
    for ell in (0, 1, 2):
        exact = G_zzb_poly_table(ell, SCALAR_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)
        serialized = serialize_deriv_tables({"zzb": exact})
        assert len(serialized.pole_list_x) <= len(exact.denominator_roots)
        assert serialized.shift == mp.mpf(abs(ell))


def test_serialize_interpolated_reproduces_table():
    """The serialized form of an InterpolatedDeltaTable reconstructs it."""
    mp.mp.dps = 60
    b = b_from_c(C)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=8)
    serialized = serialize_deriv_tables({"zzb": interp})
    node = interp.sample_points[2]
    err = _max_table_error(
        serialized.evaluate("zzb", node), interp.evaluate(node)
    )
    assert err < mp.mpf("1e-22")


def test_serialize_interpolated_stores_sample_points_in_shifted_x():
    mp.mp.dps = 60
    b = b_from_c(C)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=8, shift=mp.mpf("2.0"))
    serialized = serialize_deriv_tables({"zzb": interp})

    assert serialized.sample_points_x is not None
    expected = tuple(mp.mpf(point) - mp.mpf(interp.shift) for point in interp.sample_points)
    assert len(serialized.sample_points_x) == len(expected)
    for actual, want in zip(serialized.sample_points_x, expected):
        assert abs(mp.mpf(actual) - want) < mp.mpf("1e-25")


def test_serialize_multiple_coordinates_shares_prefactor():
    """zzb and qqb tables for the same (c, ell, externals) serialize into one
    file with a single shared shift / base / pole list."""
    mp.mp.dps = 50
    b = b_from_c(C)
    zzb = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)
    qqb = G_qqb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)
    serialized = serialize_deriv_tables({"zzb": zzb, "qqb": qqb})
    assert set(serialized.coordinates) == {"zzb", "qqb"}
    # Each coordinate still reconstructs its own block.
    Delta = mp.mpf("3.5")
    numeric = {
        "zzb": G_zzb_table(Delta, 2, SCALAR_EXT, 1, 1, b=b, q_order=8),
        "qqb": G_qqb_table(Delta, 2, SCALAR_EXT, 1, 1, b=b, q_order=8),
    }
    for coord in ("zzb", "qqb"):
        err = _max_table_error(serialized.evaluate(coord, Delta), numeric[coord])
        assert err < mp.mpf("1e-25")


def test_serialize_rejects_mixed_table_types():
    mp.mp.dps = 50
    b = b_from_c(C)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=6)
    with pytest.raises(ValueError):
        serialize_deriv_tables({"zzb": exact, "qqb": interp})


def test_serialize_rejects_empty_and_shift_on_interpolated():
    mp.mp.dps = 50
    b = b_from_c(C)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=6)
    with pytest.raises(ValueError):
        serialize_deriv_tables({})
    with pytest.raises(ValueError):
        serialize_deriv_tables({"zzb": interp}, shift=mp.mpf("1"))


def test_serialize_interpolated_coordinates_with_different_transforms():
    """Interpolated tables for coordinates with different chiral transforms
    (ws expands in z, xt in the radial y) serialize together. They share the
    universal Delta-prefactor (it depends only on c / ell / externals), so the
    multi-coordinate path uses `interpolate_delta_tables` to aggregate residue
    scores and produce a single shared prefactor + sample-point set, and the
    serializer's prefactor consistency check passes within roundoff tolerance.
    """
    mp.mp.dps = 50
    b = b_from_c(C)
    exacts = {
        "ws": G_ws_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8),
        "xt": G_xt_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8),
    }
    interpolated = interpolate_delta_tables(exacts, npoles=6)
    serialized = serialize_deriv_tables(interpolated)
    assert set(serialized.coordinates) == {"ws", "xt"}


def test_serialize_exact_strips_common_linear_factor():
    """Serializable exact tables should not emit denominator factors that are
    common to every numerator polynomial."""
    mp.mp.dps = 50
    removable = mp.mpf("1.25")
    surviving = mp.mpf("2.5")
    common = Polynomial.x_minus(removable, field=mp.mpf)
    remaining = Polynomial.x_minus(surviving, field=mp.mpf)
    exact = DeltaDerivTable(
        exp_base=mp.mpf("0.5"),
        denominator=common * remaining,
        denominator_roots=[removable, surviving],
        numerators=[
            [common * Polynomial([mp.mpf("2.0"), mp.mpf("3.0")], field=mp.mpf)]
        ],
        spin=0,
        # This table deliberately carries a removable common factor to exercise
        # serialization's strip primitive. Real tables only reach that primitive
        # when flagged by regularize_delta_table, so flag it here too.
        requires_pole_strip=True,
    )
    serialized = serialize_deriv_tables({"zzb": exact})
    assert serialized.pole_list_x == (surviving,)
    for Delta in (mp.mpf("0.3"), mp.mpf("1.7"), mp.mpf("4.2")):
        err = _max_table_error(
            serialized.evaluate("zzb", Delta),
            exact.evaluate(Delta),
        )
        assert err < mp.mpf("1e-35")
