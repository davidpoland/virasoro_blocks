"""Coverage for different choices of external operator dimensions.

The symbolic-in-Delta pipeline (`G_*_poly_table` -> interpolation ->
serialization -> writers) builds the *scalar-external* parity eigenstate
G = B(h,hbar) + (-1)^ell B(hbar,h). It accepts a 4-vector of external weights,
so the externals may be **non-identical** (each operator a different scalar
dimension); they are not required to be all equal. The tests below exercise
that non-identical-scalar case across the pipeline -- the existing symbolic
tests use the all-equal special case [0.15]*4.

Spinning externals (h_i != hbar_i) are a different code path: they use the
ordered block B(h,hbar) and the numeric `B_*_table` family (covered with
distinct h_externals / hbar_externals in test_derivs.py). The symbolic
`G_*_poly_table` family takes a single `h_externals` argument and so is
scalar-only by construction -- there is no way to express spinning externals
through it.
"""

from __future__ import annotations

import json

import mpmath as mp
import pytest

from virasoro_blocks.derivs import (
    B_qqb_table,
    B_yyb_table,
    B_zzb_table,
    G_qqb_table,
    G_ws_table,
    G_zzb_table,
)
from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyderivs import (
    B_qqb_poly_table,
    B_yyb_poly_table,
    B_zzb_poly_table,
    G_qqb_poly_table,
    G_yyb_poly_table,
    G_ws_poly_table,
    G_zzb_poly_table,
    interpolate_delta_table,
    parity_projected_B_qqb_poly_table,
    parity_projected_B_yyb_poly_table,
    parity_projected_B_zzb_poly_table,
)
from virasoro_blocks.serialization import serialize_deriv_tables
from virasoro_blocks.writers import write_json

C = mp.mpf("10")
# Four distinct scalar external weights (h_i = hbar_i, but not all equal).
NONID_EXT = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
# Spinning externals: holomorphic and antiholomorphic halves differ (h_i != hbar_i).
SPIN_H = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
SPIN_HBAR = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]

# coordinate -> (symbolic poly-table builder, numeric table builder)
_BUILDERS = {
    "zzb": (G_zzb_poly_table, G_zzb_table),
    "qqb": (G_qqb_poly_table, G_qqb_table),
    "ws": (G_ws_poly_table, G_ws_table),
}
# coordinate -> (symbolic ordered-B builder, numeric ordered-B builder)
_B_BUILDERS = {
    "zzb": (B_zzb_poly_table, B_zzb_table),
    "yyb": (B_yyb_poly_table, B_yyb_table),
    "qqb": (B_qqb_poly_table, B_qqb_table),
}

_PB_BUILDERS = {
    "zzb": parity_projected_B_zzb_poly_table,
    "yyb": parity_projected_B_yyb_poly_table,
    "qqb": parity_projected_B_qqb_poly_table,
}


def _max_table_error(table_a, table_b):
    worst = mp.mpf(0)
    for row_a, row_b in zip(table_a, table_b):
        for a, b in zip(row_a, row_b):
            worst = max(worst, abs(mp.mpc(a) - mp.mpc(b)))
    return worst


@pytest.mark.parametrize("coordinate", ["zzb", "qqb", "ws"])
def test_symbolic_table_non_identical_scalar_externals(coordinate):
    """The symbolic-in-Delta table reproduces the numeric block table when the
    four external dimensions are all distinct (a non-folded and a parity-folded
    coordinate are both checked)."""
    mp.mp.dps = 50
    b = b_from_c(C)
    poly_builder, numeric_builder = _BUILDERS[coordinate]
    # Full-grid comparison vs the full-square numeric table; non-folded symbolic
    # builders now default to the order-Lambda triangle, so opt into the square
    # (the folded ws builder is intrinsically m+2n<=Lambda and takes no lambda_cut).
    extra = {} if coordinate == "ws" else {"lambda_cut": None}
    sym = poly_builder(2, NONID_EXT, 2, 2, b=b, q_order=8, **extra)
    Delta = mp.mpf("3.1")
    numeric = numeric_builder(Delta, 2, NONID_EXT, 2, 2, b=b, q_order=8)
    assert _max_table_error(sym.evaluate(Delta), numeric) < mp.mpf("1e-25")


def test_interpolation_non_identical_scalar_externals():
    """interpolate_delta_table reconstructs the block at each (m, n) entry's
    own sample nodes for a non-identical-externals table. `sample_points`
    is the highest-degree entry's node set; check that the highest entry
    reconstructs exactly there."""
    mp.mp.dps = 55
    b = b_from_c(C)
    exact = G_zzb_poly_table(2, NONID_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=8)
    node = interp.sample_points[2]
    numeric = G_zzb_table(node, 2, NONID_EXT, 2, 2, b=b, q_order=8)
    hi_m, hi_n = interp.m_max, interp.n_max
    err_hi = abs(
        mp.mpc(interp.evaluate(node)[hi_m][hi_n])
        - mp.mpc(numeric[hi_m][hi_n])
    )
    assert err_hi < mp.mpf("1e-16")


def test_write_json_records_full_delta_external():
    """With non-identical externals the JSON `delta_external` array carries
    the four full external dimensions (not just their pairwise differences,
    since the Virasoro block depends on d_2 + d_3 and Sum d_i too)."""
    mp.mp.dps = 50
    b = b_from_c(C)
    exact = G_zzb_poly_table(2, NONID_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)
    serialized = serialize_deriv_tables({"zzb": exact})
    doc = json.loads(
        write_json(
            serialized,
            c=C,
            h_ext=NONID_EXT,
            hbar_ext=NONID_EXT,
            spin=2,
            q_order=8,
        )
    )
    # delta_external[i] = h_i + hbar_i = 2 * NONID_EXT[i]
    expected = [2 * d for d in NONID_EXT]
    for got, want in zip(doc["delta_external"], expected):
        assert mp.almosteq(mp.mpf(got), want)
    # Non-trivial differences are still derivable, just not stored as separate fields.
    assert mp.almosteq(
        mp.mpf(doc["delta_external"][0]) - mp.mpf(doc["delta_external"][1]),
        expected[0] - expected[1],
    )
    # Externals are scalar (h = hbar), so every external spin is 0.
    assert doc["spin_external"] == [0, 0, 0, 0]
    # delta_12 / delta_43 are intentionally not emitted.
    assert "delta_12" not in doc
    assert "delta_43" not in doc


# --- spinning externals (h_i != hbar_i): the ordered B table ---------------


@pytest.mark.parametrize("coordinate", ["zzb", "yyb", "qqb"])
@pytest.mark.parametrize("ell", [0, 2, -3])
def test_B_poly_table_matches_numeric_spinning(coordinate, ell):
    """The symbolic ordered-B table reproduces the numeric B table for spinning
    externals (h_i != hbar_i) and signed exchanged spin ell = h - hbar."""
    mp.mp.dps = 50
    b = b_from_c(C)
    poly_builder, numeric_builder = _B_BUILDERS[coordinate]
    # Full-grid comparison vs the full-square numeric table (all coords here are
    # non-folded), so opt out of the new default order-Lambda triangle.
    sym = poly_builder(ell, SPIN_H, SPIN_HBAR, 2, 2, b=b, q_order=8, lambda_cut=None)
    Delta = mp.mpf("3.4")
    h = (Delta + ell) / 2
    hbar = (Delta - ell) / 2
    numeric = numeric_builder(h, hbar, SPIN_H, SPIN_HBAR, 2, 2, b=b, q_order=8)
    assert _max_table_error(sym.evaluate(Delta), numeric) < mp.mpf("1e-25")


def test_B_table_scalar_externals_recover_G_parity_sum():
    """With h_externals == hbar_externals the ordered B tables combine into the
    scalar parity-even block: G(ell) = B(ell) + B(-ell) at every spin (odd ell
    included), where B(-ell) is the h <-> hbar swapped ordering."""
    mp.mp.dps = 50
    b = b_from_c(C)
    ell = 3
    Delta = mp.mpf("3.4")
    g = G_zzb_poly_table(ell, NONID_EXT, 2, 2, b=b, q_order=8, lambda_cut=None).evaluate(Delta)
    b_pos = B_zzb_poly_table(
        ell, NONID_EXT, NONID_EXT, 2, 2, b=b, q_order=8, lambda_cut=None
    ).evaluate(Delta)
    b_neg = B_zzb_poly_table(
        -ell, NONID_EXT, NONID_EXT, 2, 2, b=b, q_order=8, lambda_cut=None
    ).evaluate(Delta)
    combined = [
        [b_pos[m][n] + b_neg[m][n] for n in range(3)] for m in range(3)
    ]
    assert _max_table_error(g, combined) < mp.mpf("1e-22")


def test_B_poly_table_interpolates_spinning():
    """A spinning ordered-B table order-reduces through interpolate_delta_table.
    The highest-degree entry's node reconstruction is exact; lower-degree
    entries use per-entry node sets and are checked separately by
    `test_interpolation_is_exact_at_sample_nodes`."""
    mp.mp.dps = 55
    b = b_from_c(C)
    exact = B_zzb_poly_table(2, SPIN_H, SPIN_HBAR, 2, 2, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=8)
    node = interp.sample_points[2]
    numeric = B_zzb_table(
        (node + 2) / 2, (node - 2) / 2, SPIN_H, SPIN_HBAR, 2, 2, b=b, q_order=8
    )
    hi_m, hi_n = interp.m_max, interp.n_max
    err_hi = abs(
        mp.mpc(interp.evaluate(node)[hi_m][hi_n])
        - mp.mpc(numeric[hi_m][hi_n])
    )
    assert err_hi < mp.mpf("1e-16")


def test_B_poly_table_rejects_non_half_integer_ell():
    b = b_from_c(C)
    with pytest.raises(ValueError):
        B_zzb_poly_table(mp.mpf("1.25"), SPIN_H, SPIN_HBAR, 1, 1, b=b, q_order=6)


@pytest.mark.parametrize("coordinate", ["zzb", "yyb", "qqb"])
def test_parity_projected_B_poly_table_matches_ordered_definition(coordinate):
    mp.mp.dps = 50
    b = b_from_c(C)
    ell = mp.mpf("1.5")
    Delta = mp.mpf("3.4")
    ordered_builder, _ = _B_BUILDERS[coordinate]
    projected_builder = _PB_BUILDERS[coordinate]

    ordered = ordered_builder(ell, SPIN_H, SPIN_HBAR, 2, 2, b=b, q_order=8).evaluate(Delta)
    reversed_table = ordered_builder(-ell, SPIN_HBAR, SPIN_H, 2, 2, b=b, q_order=8).evaluate(Delta)
    plus = projected_builder(
        ell,
        SPIN_H,
        SPIN_HBAR,
        2,
        2,
        parity="+",
        b=b,
        q_order=8,
    ).evaluate(Delta)
    minus = projected_builder(
        ell,
        SPIN_H,
        SPIN_HBAR,
        2,
        2,
        parity="-",
        b=b,
        q_order=8,
    ).evaluate(Delta)

    plus_expected = [
        [ordered[m][n] + reversed_table[m][n] for n in range(3)]
        for m in range(3)
    ]
    minus_expected = [
        [ordered[m][n] - reversed_table[m][n] for n in range(3)]
        for m in range(3)
    ]
    assert _max_table_error(plus, plus_expected) < mp.mpf("1e-24")
    assert _max_table_error(minus, minus_expected) < mp.mpf("1e-24")


def test_parity_projected_B_poly_table_applies_phase_metadata():
    mp.mp.dps = 50
    b = b_from_c(C)
    ell = mp.mpf("1.5")
    Delta = mp.mpf("3.4")

    ordered = B_zzb_poly_table(ell, SPIN_H, SPIN_HBAR, 2, 2, b=b, q_order=8, lambda_cut=None).evaluate(Delta)
    reversed_table = B_zzb_poly_table(-ell, SPIN_HBAR, SPIN_H, 2, 2, b=b, q_order=8, lambda_cut=None).evaluate(Delta)
    projected = parity_projected_B_zzb_poly_table(
        ell,
        SPIN_H,
        SPIN_HBAR,
        2,
        2,
        parity="+",
        parity_external=["1", "1", "1", "1"],
        exchanged_parity="-1",
        b=b,
        q_order=8,
        lambda_cut=None,  # compared against the full-square ordered tables above
    ).evaluate(Delta)
    expected = [
        [ordered[m][n] - reversed_table[m][n] for n in range(3)]
        for m in range(3)
    ]
    assert _max_table_error(projected, expected) < mp.mpf("1e-24")


@pytest.mark.parametrize("coordinate,ell,parity", [("zzb", 0, "+"), ("yyb", 2, "+"), ("qqb", 3, "+")])
def test_parity_projected_B_scalar_limit_recovers_G(coordinate, ell, parity):
    mp.mp.dps = 50
    b = b_from_c(C)
    Delta = mp.mpf("3.4")
    projected_builder = _PB_BUILDERS[coordinate]
    g_builder = {
        "zzb": G_zzb_poly_table,
        "yyb": G_yyb_poly_table,
        "qqb": G_qqb_poly_table,
    }[coordinate]
    projected = projected_builder(
        ell, NONID_EXT, NONID_EXT, 2, 2, parity=parity, b=b, q_order=8
    ).evaluate(Delta)
    g = g_builder(ell, NONID_EXT, 2, 2, b=b, q_order=8).evaluate(Delta)
    if ell == 0:
        # B + P[B] = 2 B(h, h) at spin 0, where G is B(h, h) itself
        g = [[2 * entry for entry in row] for row in g]
    assert _max_table_error(projected, g) < mp.mpf("1e-22")


def test_parity_projected_B_poly_table_rejects_unknown_parity():
    b = b_from_c(C)
    with pytest.raises(ValueError):
        parity_projected_B_zzb_poly_table(
            1,
            SPIN_H,
            SPIN_HBAR,
            1,
            1,
            parity="even",
            b=b,
            q_order=6,
        )
