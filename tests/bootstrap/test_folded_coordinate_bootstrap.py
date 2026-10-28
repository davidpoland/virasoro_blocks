"""Tests for the parity-folded (xt/ws) identical-scalar bootstrap wiring.

The folded coordinates are a Delta-independent invertible linear recombination
of the non-folded (zzb/yyb) crossing derivatives, so an identical-scalar gap
bootstrap in xt must be GL-equivalent to (and hence give the same bound as) the
zzb bootstrap, and ws must match yyb. These tests pin the crossing index set,
prove the linear equivalence directly, exercise the end-to-end build, and
confirm the folded block tables still ride the flint fast path.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.derivs import G_zzb_table, G_xt_table, G_yyb_table, G_ws_table
from virasoro_bootstrap.crossing import F_minus_pairs
from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
)


@pytest.mark.parametrize("lam", [4, 5, 6, 7])
def test_folded_F_minus_pairs_count_matches_nonfolded(lam):
    """The folded odd-`a` index set has the same cardinality as the non-folded
    odd-(m+n) lower-triangle set (necessary for a basis change)."""
    n_zzb = len(F_minus_pairs(lam, coordinate="zzb"))
    for coord in ("xt", "ws"):
        pairs = F_minus_pairs(lam, coordinate=coord)
        assert len(pairs) == n_zzb
        # folded selection: odd a, a + 2 b <= lam
        for (a, b) in pairs:
            assert a % 2 == 1
            assert a + 2 * b <= lam


@pytest.mark.parametrize(
    "folded_builder,folded_coord,base_builder,base_coord",
    [
        (G_xt_table, "xt", G_zzb_table, "zzb"),
        (G_ws_table, "ws", G_yyb_table, "yyb"),
    ],
)
def test_folded_crossing_is_invertible_linear_transform(
    folded_builder, folded_coord, base_builder, base_coord
):
    """F^- vectors in the folded coordinate equal a fixed invertible linear
    transform of the non-folded F^- vectors, evaluated across (Delta, ell)
    samples and the vacuum limit -> identical bootstrap bound.  At lam = 4 there
    are 3 F^- components: the first 3 samples fix the transform and the other 2
    (a spin-2 block and the vacuum limit) check it."""
    with mp.workdps(50):
        h = mp.mpf("0.2")
        c = mp.mpf(32)
        lam = 4
        z_shift = -2 * h
        q_order = 6
        samples = [
            (mp.mpf("1.0"), 0), (mp.mpf("2.3"), 0), (mp.mpf("2.0"), 2),
            (mp.mpf("3.6"), 2), (mp.mpf("1e-30"), 0),
        ]

        def vec(builder, coord, Delta, ell):
            t = builder(Delta, ell, [h] * 4, lam, lam, c=c, q_order=q_order, z_shift=z_shift)
            return [t[m][n] for (m, n) in F_minus_pairs(lam, coordinate=coord)]

        base = mp.matrix([vec(base_builder, base_coord, D, l) for (D, l) in samples])
        folded = mp.matrix([vec(folded_builder, folded_coord, D, l) for (D, l) in samples])
        k = base.cols
        assert base.rows > k  # the extra rows are the actual check
        base_sq = mp.matrix([[base[i, j] for j in range(k)] for i in range(k)])
        folded_sq = mp.matrix([[folded[i, j] for j in range(k)] for i in range(k)])
        # M solves base_sq @ M = folded_sq; verify it maps every sample row.
        M = base_sq ** -1 * folded_sq
        residual = base * M - folded
        max_res = max(
            abs(residual[i, j]) for i in range(residual.rows) for j in range(residual.cols)
        )
        assert max_res < mp.mpf(10) ** (-35)
        assert abs(mp.det(M)) > mp.mpf(10) ** (-20)


@pytest.mark.parametrize("coordinate", ["xt", "ws"])
def test_build_gap_problem_in_folded_coordinate(coordinate):
    """The full identical-scalar gap problem assembles end-to-end in the folded
    coordinates with the same F^- dimension as the non-folded build."""
    common = dict(
        c=mp.mpf(32), h_ext=mp.mpf("0.2"), delta_gap=mp.mpf("1.0"), lam=4,
        spin_list=(0, 2), q_order=8, dps=40, npoles=4, num_threads=1,
    )
    folded = build_identical_scalar_gap_problem(
        IdenticalScalarGapSpec(coordinate=coordinate, **common)
    )
    base_coord = "zzb" if coordinate == "xt" else "yyb"
    base = build_identical_scalar_gap_problem(
        IdenticalScalarGapSpec(coordinate=base_coord, **common)
    )
    assert len(list(folded.F_pairs)) == len(list(base.F_pairs))
    assert len(folded.spin_channels) == len(base.spin_channels)
    assert folded.resolved_method == base.resolved_method


@pytest.mark.parametrize("builder", [G_xt_table, G_ws_table])
def test_folded_block_build_uses_flint(monkeypatch, builder):
    """Folded symbolic block tables ride the same flint acb_series fast path as
    zzb/yyb (xt on the z-family, ws on the y-family)."""
    pytest.importorskip("flint")
    import virasoro_blocks.polyderivs as P
    from virasoro_blocks.polyderivs import G_xt_poly_table, G_ws_poly_table

    poly_builder = G_xt_poly_table if builder is G_xt_table else G_ws_poly_table
    counts = {"flint": 0, "fallback": 0}
    orig = P._flint_taylor_series

    def wrapped(*args, **kwargs):
        result = orig(*args, **kwargs)
        counts["flint" if result is not None else "fallback"] += 1
        return result

    monkeypatch.setattr(P, "_flint_taylor_series", wrapped)
    with mp.workdps(60):
        poly_builder(
            0, [mp.mpf("0.2")] * 4, 5, 5, c=mp.mpf(32), q_order=10,
            z_shift=mp.mpf("-0.4"),
        )
    assert counts["flint"] >= 1
    assert counts["fallback"] == 0
