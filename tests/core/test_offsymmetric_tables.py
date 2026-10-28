"""Off-symmetric (diagonal z=zbar=center) symbolic tables -- Phase A.

The chiral expansion is recentered at u(center); the block evaluated off the
crossing-symmetric point z = zbar = 1/2. Verifies (a) center=None is the legacy
symmetric build, (b) the recentered exact table matches direct pointwise
evaluation, (c) it interpolates, and (d) the algebraic (m,n) selection rules
that survive on the diagonal still hold.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.derivs import evaluate_block
from virasoro_blocks.polyderivs import (
    B_qqb_poly_table,
    G_qqb_poly_table,
    G_zzb_poly_table,
    _NumericBlockSampler,
    _NumericOrderedBlockSampler,
    interpolate_delta_table,
    parity_projected_B_qqb_poly_table,
)

EXT = [mp.mpf("0.1")] * 4
C = mp.mpf("2.5")
# Every comparison here builds both sides (symbolic table vs evaluate_block /
# numeric sampler / another table) at the same q_order, so the q-truncation is
# common to both and the tolerances test the recentered construction, not
# q-convergence. 20 keeps the whole file green (as do 24 and 28); an
# under-converged q_order would fail loudly rather than pass silently.
QORDER = 20


def test_center_none_is_legacy_symmetric():
    mp.mp.dps = 40
    a = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER)
    b = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER, center=None)
    half = mp.mpf(1) / 2
    c2 = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER, center=half)
    D = mp.mpf("3.3")
    ga, gb, gc = a.evaluate(D), b.evaluate(D), c2.evaluate(D)
    for m in range(4):
        for n in range(4):
            assert gb[m][n] == ga[m][n]                     # None == default
            assert abs(gc[m][n] - ga[m][n]) < mp.mpf(10) ** (-35)  # center=1/2


@pytest.mark.parametrize("coord_builder,coord", [
    (G_qqb_poly_table, "qqb"),
    (G_zzb_poly_table, "zzb"),
])
def test_offsymmetric_value_matches_evaluate_block(coord_builder, coord):
    mp.mp.dps = 40
    z0 = mp.mpf("0.37")
    ell = 2
    tab = coord_builder(ell, EXT, 2, 2, c=C, q_order=QORDER, center=z0)
    D = mp.mpf("3.3")
    h, hbar = (D + ell) / 2, (D - ell) / 2
    # scalar G = B(h,hbar) + (-1)^ell B(hbar,h); on the diagonal both orderings
    # are genuine evaluate_block calls at the same point.
    direct = evaluate_block(z0, z0, h, hbar, EXT, c=C, q_order=QORDER, coord=coord) \
        + evaluate_block(z0, z0, hbar, h, EXT, c=C, q_order=QORDER, coord=coord)
    assert abs(tab.evaluate(D)[0][0] - direct) < mp.mpf(10) ** (-35)


def test_offsymmetric_interpolation_roundtrip():
    mp.mp.dps = 40
    z0 = mp.mpf("0.37")
    exact = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER, center=z0)
    interp = interpolate_delta_table(exact, npoles=10)
    for D in (mp.mpf("3.3"), mp.mpf("5.1")):
        ge, gi = exact.evaluate(D), interp.evaluate(D)
        scale = max(abs(ge[m][n]) for m in range(4) for n in range(4))
        err = max(abs(gi[m][n] - ge[m][n]) for m in range(4) for n in range(4))
        assert err / scale < mp.mpf(10) ** (-8)


def test_diagonal_selection_rules_survive_offsymmetric():
    mp.mp.dps = 40
    z0 = mp.mpf("0.37")
    D = mp.mpf("3.3")
    # the scalar G is m<->n symmetric at every spin; the parity '-' projection is
    # antisymmetric and its m==n diagonal vanishes.
    even = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER, center=z0).evaluate(D)
    odd_spin = G_qqb_poly_table(3, EXT, 3, 3, c=C, q_order=QORDER, center=z0).evaluate(D)
    odd = parity_projected_B_qqb_poly_table(
        3, EXT, EXT, 3, 3, parity="-", c=C, q_order=QORDER, center=z0).evaluate(D)
    for m in range(4):
        assert abs(odd[m][m]) < mp.mpf(10) ** (-35)        # odd diagonal vanish
        for n in range(4):
            assert abs(even[m][n] - even[n][m]) < mp.mpf(10) ** (-35)
            assert abs(odd_spin[m][n] - odd_spin[n][m]) < mp.mpf(10) ** (-35)
            assert abs(odd[m][n] + odd[n][m]) < mp.mpf(10) ** (-35)


def test_lambda_cut_imposes_triangle():
    """lambda_cut=Lambda zeros the m+n>Lambda corner (the order-Lambda triangle);
    default (None) keeps the full square. Works on- and off-symmetric alike."""
    mp.mp.dps = 40
    z0 = mp.mpf("0.37")
    D = mp.mpf("3.3")
    full = G_qqb_poly_table(
        2, EXT, 2, 2, c=C, q_order=QORDER, center=z0, lambda_cut=None
    ).evaluate(D)
    tri = G_qqb_poly_table(
        2, EXT, 2, 2, c=C, q_order=QORDER, center=z0  # default = order-Lambda triangle
    ).evaluate(D)
    for m in range(3):
        for n in range(3):
            if m + n > 2:
                assert abs(tri[m][n]) < mp.mpf(10) ** (-35)      # corner dropped
                assert abs(full[m][n]) > mp.mpf(10) ** (-35)     # ...but real in full
            else:
                assert abs(tri[m][n] - full[m][n]) < mp.mpf(10) ** (-35)


def test_folded_center_rejected():
    with pytest.raises(ValueError):
        from virasoro_blocks.polyderivs import _resolve_center_u
        _resolve_center_u("qqb_ws", mp.mpf("0.4"))


# --- Phase B: off-diagonal z != zbar (independent centers per axis) ---------

Z0, ZBAR0 = mp.mpf("0.37"), mp.mpf("0.55")
SPIN_DELTA_EXT = [mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.4"), mp.mpf("0.6")]
SPIN_EXT = [mp.mpf("0.2"), mp.mpf("-0.1"), mp.mpf("0.0"), mp.mpf("0.3")]
SPIN_H = [(d + s) / 2 for d, s in zip(SPIN_DELTA_EXT, SPIN_EXT)]
SPIN_HBAR = [(d - s) / 2 for d, s in zip(SPIN_DELTA_EXT, SPIN_EXT)]


@pytest.mark.parametrize("coord_builder,coord", [
    (G_qqb_poly_table, "qqb"),
    (G_zzb_poly_table, "zzb"),
])
@pytest.mark.parametrize("ell", [0, 2, 3])
def test_offdiagonal_grid_matches_evaluate_block(coord_builder, coord, ell):
    """G off the diagonal (m-axis at z0, n-axis at zbar0) reproduces the chiral
    factorization G = [B(h,hbar) + B(hbar,h)] / (1 + delta_{ell,0}) evaluated
    pointwise at (z0, zbar0) by evaluate_block, full (m, n) grid."""
    mp.mp.dps = 40
    D = mp.mpf("3.3")
    h, hbar = (D + ell) / 2, (D - ell) / 2
    tab = coord_builder(
        ell, EXT, 2, 2, c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0,
        lambda_cut=None,
    )
    got = tab.evaluate(D)
    ob1 = evaluate_block(Z0, ZBAR0, h, hbar, EXT, c=C, q_order=QORDER, coord=coord, m_max=2, n_max=2)
    ob2 = evaluate_block(Z0, ZBAR0, hbar, h, EXT, c=C, q_order=QORDER, coord=coord, m_max=2, n_max=2)
    for m in range(3):
        for n in range(3):
            ref = (mp.mpc(ob1[m][n]) + mp.mpc(ob2[m][n])) / (2 if ell == 0 else 1)
            assert abs(mp.mpc(got[m][n]) - ref) < mp.mpf(10) ** -35


def test_offdiagonal_diagonal_limit_is_unchanged():
    """center_bar=None and center_bar==center both reduce to the diagonal build
    bit-for-bit -- no off-diagonal code runs (and no second expansion is built)
    when the two centers coincide, so the diagonal path is untouched."""
    mp.mp.dps = 40
    D = mp.mpf("3.3")
    base = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER, center=Z0).evaluate(D)
    same = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER, center=Z0, center_bar=Z0).evaluate(D)
    for m in range(4):
        for n in range(4):
            assert same[m][n] == base[m][n]


def test_offdiagonal_interpolation_roundtrip():
    mp.mp.dps = 40
    exact = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0)
    interp = interpolate_delta_table(exact, npoles=10)
    for D in (mp.mpf("3.3"), mp.mpf("5.1")):
        ge, gi = exact.evaluate(D), interp.evaluate(D)
        cells = [(m, n) for m in range(4) for n in range(4) if ge[m][n] is not None]
        scale = max(abs(mp.mpc(ge[m][n])) for (m, n) in cells)
        err = max(abs(mp.mpc(gi[m][n]) - mp.mpc(ge[m][n])) for (m, n) in cells)
        assert err / scale < mp.mpf(10) ** -8


def test_offdiagonal_numeric_sampler_matches_exact_table_entries():
    mp.mp.dps = 50
    ell = 2
    exact = G_qqb_poly_table(
        ell, EXT, 3, 3, c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0,
        lambda_cut=None,
    )
    sampler = _NumericBlockSampler(
        "qqb", EXT, ell, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0, center_bar=ZBAR0,
    )
    entries = [(0, 0), (1, 2), (2, 1), (3, 0)]
    deltas = [mp.mpf("3.3"), mp.mpf("5.1")]
    tables, _is_arb = sampler(deltas, entries=entries)
    for D, table in zip(deltas, tables):
        ref = exact.evaluate(D)
        for (m, n) in entries:
            assert abs(mp.mpc(table[m][n]) - mp.mpc(ref[m][n])) < mp.mpf(10) ** -40


def test_offdiagonal_numeric_sampler_diagonal_limit_unchanged():
    mp.mp.dps = 50
    ell = 2
    entries = [(0, 0), (1, 1), (3, 0)]
    deltas = [mp.mpf("3.3"), mp.mpf("4.8")]
    base = _NumericBlockSampler(
        "qqb", EXT, ell, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0,
    )
    same = _NumericBlockSampler(
        "qqb", EXT, ell, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0, center_bar=Z0,
    )
    tb, _ = base(deltas, entries=entries)
    ts, _ = same(deltas, entries=entries)
    for btab, stab in zip(tb, ts):
        for (m, n) in entries:
            assert abs(mp.mpc(stab[m][n]) - mp.mpc(btab[m][n])) < mp.mpf(10) ** -45


def test_offdiagonal_interpolation_roundtrip_numeric_sampler():
    mp.mp.dps = 50
    exact = G_qqb_poly_table(2, EXT, 3, 3, c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0)
    exact.numeric_sampler = _NumericBlockSampler(
        "qqb", EXT, 2, exact.m_max, exact.n_max,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0, center_bar=ZBAR0,
    )
    interp = interpolate_delta_table(exact, npoles=10)
    for D in (mp.mpf("3.3"), mp.mpf("5.1")):
        ge, gi = exact.evaluate(D), interp.evaluate(D)
        cells = [(m, n) for m in range(4) for n in range(4) if ge[m][n] is not None]
        scale = max(abs(mp.mpc(ge[m][n])) for (m, n) in cells)
        err = max(abs(mp.mpc(gi[m][n]) - mp.mpc(ge[m][n])) for (m, n) in cells)
        assert err / scale < mp.mpf(10) ** -8


def test_offdiagonal_contribution_skeleton_matches_full():
    pytest.importorskip("flint")
    mp.mp.dps = 60
    full = G_qqb_poly_table(
        2, EXT, 3, 3, c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0
    )
    skel = G_qqb_poly_table(
        2, EXT, 3, 3, c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0,
        skeleton=True,
    )
    assert skel.numerators is None

    sampler = _NumericBlockSampler(
        "qqb", EXT, 2, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0, center_bar=ZBAR0,
    )
    full.numeric_sampler = sampler
    skel.numeric_sampler = _NumericBlockSampler(
        "qqb", EXT, 2, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0, center_bar=ZBAR0,
    )

    interp_full = interpolate_delta_table(
        full, npoles=10, pole_selection="contribution", pole_ranking="numeric"
    )
    interp_skel = interpolate_delta_table(
        skel, npoles=10, pole_selection="contribution", pole_ranking="numeric"
    )

    for D in (mp.mpf("3.3"), mp.mpf("5.1")):
        full_table = interp_full.evaluate(D)
        skel_table = interp_skel.evaluate(D)
        cells = [(m, n) for m in range(4) for n in range(4) if full_table[m][n] is not None]
        worst = max(
            abs(mp.mpc(skel_table[m][n]) - mp.mpc(full_table[m][n]))
            / max(abs(mp.mpc(full_table[m][n])), mp.mpf("1e-30"))
            for (m, n) in cells
        )
        assert worst < mp.mpf("1e-30"), f"Delta={D}: rel err {mp.nstr(worst, 5)}"


@pytest.mark.parametrize("ell", [0, 2, 3])
def test_offdiagonal_ordered_B_scalar_limit_matches_evaluate_block(ell):
    """With scalar externals, ordered B off the diagonal equals the direct
    ordered pointwise oracle evaluate_block at matching (z0, zbar0)."""
    mp.mp.dps = 50
    D = mp.mpf("3.3")
    h, hbar = (D + ell) / 2, (D - ell) / 2
    tab = B_qqb_poly_table(
        ell, EXT, EXT, 2, 2, c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0,
        lambda_cut=None,
    )
    got = tab.evaluate(D)
    ref = evaluate_block(
        Z0, ZBAR0, h, hbar, EXT, c=C, q_order=QORDER, coord="qqb", m_max=2, n_max=2,
    )
    for m in range(3):
        for n in range(3):
            assert abs(mp.mpc(got[m][n]) - mp.mpc(ref[m][n])) < mp.mpf(10) ** -40


@pytest.mark.parametrize("ell,parity", [(2, "+"), (3, "+")])
def test_offdiagonal_projected_B_scalar_limit_matches_G(ell, parity):
    """With scalar externals, the '+' projected B reduces to scalar G at every
    spin, odd included."""
    mp.mp.dps = 50
    D = mp.mpf("3.3")
    proj = parity_projected_B_qqb_poly_table(
        ell, EXT, EXT, 2, 2,
        parity=parity, parity_external=None, exchanged_parity="1",
        c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0, lambda_cut=None,
    ).evaluate(D)
    g = G_qqb_poly_table(
        ell, EXT, 2, 2, c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0,
        lambda_cut=None,
    ).evaluate(D)
    for m in range(3):
        for n in range(3):
            assert abs(mp.mpc(proj[m][n]) - mp.mpc(g[m][n])) < mp.mpf(10) ** -40


def test_offdiagonal_ordered_numeric_sampler_matches_symbolic_spinning():
    mp.mp.dps = 50
    ell = 1
    entries = [(0, 0), (1, 2), (3, 0)]
    deltas = [mp.mpf("2.7"), mp.mpf("3.6")]
    sampler = _NumericOrderedBlockSampler(
        "qqb", SPIN_H, SPIN_HBAR, ell, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None,
        z_shift=None, zbar_shift=None, projected=False,
        center=Z0, center_bar=ZBAR0,
    )
    tables, _ = sampler(deltas, entries=entries)
    exact = B_qqb_poly_table(
        ell, SPIN_H, SPIN_HBAR, 3, 3, c=C, q_order=QORDER,
        center=Z0, center_bar=ZBAR0, lambda_cut=None,
    )
    for D, got in zip(deltas, tables):
        ref = exact.evaluate(D)
        for (m, n) in entries:
            assert abs(mp.mpc(got[m][n]) - mp.mpc(ref[m][n])) < mp.mpf(10) ** -35


def test_offdiagonal_projected_numeric_sampler_matches_symbolic_spinning():
    mp.mp.dps = 50
    ell = 1
    entries = [(0, 0), (2, 1), (3, 0)]
    deltas = [mp.mpf("2.7"), mp.mpf("3.6")]
    sampler = _NumericOrderedBlockSampler(
        "qqb", SPIN_H, SPIN_HBAR, ell, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None,
        z_shift=None, zbar_shift=None, projected=True,
        parity="-", parity_external=None, exchanged_parity="1",
        center=Z0, center_bar=ZBAR0,
    )
    tables, _ = sampler(deltas, entries=entries)
    exact = parity_projected_B_qqb_poly_table(
        ell, SPIN_H, SPIN_HBAR, 3, 3,
        parity="-", parity_external=None, exchanged_parity="1",
        c=C, q_order=QORDER, center=Z0, center_bar=ZBAR0, lambda_cut=None,
    )
    for D, got in zip(deltas, tables):
        ref = exact.evaluate(D)
        for (m, n) in entries:
            assert abs(mp.mpc(got[m][n]) - mp.mpc(ref[m][n])) < mp.mpf(10) ** -35


def test_numeric_sampler_center_bar_equal_center_stays_diagonal_mode():
    s = _NumericBlockSampler(
        "qqb", EXT, 2, 2, 2,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0, center_bar=Z0,
    )
    assert s._offdiag is False


def test_numeric_ordered_sampler_center_bar_equal_center_stays_diagonal_mode():
    s = _NumericOrderedBlockSampler(
        "qqb", SPIN_H, SPIN_HBAR, 1, 2, 2,
        b=None, c=C, q_order=QORDER, chiral_norm=None,
        z_shift=None, zbar_shift=None, projected=False,
        center=Z0, center_bar=Z0,
    )
    assert s._offdiag is False


def test_numeric_sampler_center_bar_equal_center_takes_diagonal_path():
    """Guardrail: ``center_bar == center`` must take the single-center diagonal
    path (no second evaluator / 4-vector assembly), which is what guarantees no
    speed regression. Asserted structurally -- a wall-clock timing test flakes
    under parallel CI load -- by checking the sampler's off-diagonal flag and
    that the two coincident centers resolve to the same chiral expansion point.
    """
    mp.mp.dps = 40
    s_diag = _NumericBlockSampler(
        "qqb", EXT, 2, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0,
    )
    s_equal = _NumericBlockSampler(
        "qqb", EXT, 2, 3, 3,
        b=None, c=C, q_order=QORDER, chiral_norm=None, z_shift=None,
        center=Z0, center_bar=Z0,
    )
    # center_bar == center (or None) => diagonal mode: no off-diagonal flag, and
    # both axes share one expansion center, so only one evaluator is ever built.
    assert s_diag._offdiag is False
    assert s_equal._offdiag is False
    assert s_equal._u_center == s_equal._u_center_bar == s_diag._u_center


@pytest.mark.parametrize("parity", ["+", "-"])
@pytest.mark.parametrize("off_center", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_scalar_external_projection_dispatch_matches_requested_block(parity, off_center, cached):
    from virasoro_blocks.requests import BlockRequest, NumericalPolicy
    from virasoro_blocks.table_backends import (
        VirasoroBackendOptions, _backend_args, _process_spin_once, build_cli_chiral_cache,
    )
    from virasoro_blocks.polyderivs import parity_projected_B_zzb_poly_table
    from virasoro_blocks.params import b_from_c

    with mp.workdps(40):
        ext = [mp.mpf("0.1")] * 4
        center = mp.mpf("0.3") if off_center else None
        center_bar = mp.mpf("0.4") if off_center else None
        request = BlockRequest(
            operation="table", basis="parity_projected_B", parity=parity,
            central_charge=C, h_ext=tuple(ext), hbar_ext=tuple(ext),
            coordinates=("zzb",), center=center, center_bar=center_bar,
            policy=NumericalPolicy(precision=40, order=4, lambda_order=1),
        )
        args = _backend_args(request, VirasoroBackendOptions(method="full"))
        cache = (build_cli_chiral_cache(
            args, ext, ext, b_from_c(C), regularize=False, basis="parity_projected_B",
        ) if cached else None)
        _, exact, _ = _process_spin_once(
            2, args, ext, ext, b_from_c(C), C, False, True, False,
            "parity_projected_B", regularize=False, chiral_cache=cache,
        )
        delta = mp.mpf("3.3")
        expected = parity_projected_B_zzb_poly_table(
            2, ext, ext, 1, 1, c=C, q_order=4, parity=parity,
            center=center, center_bar=center_bar,
        ).evaluate(delta)
        symbolic = exact["zzb"].evaluate(delta)
        numeric, _ = exact["zzb"].numeric_sampler([delta], entries=[(0, 0), (1, 0)])
        for m, n in [(0, 0), (1, 0)]:
            assert abs(mp.mpc(symbolic[m][n]) - expected[m][n]) < mp.mpf("1e-30")
            assert abs(mp.mpc(numeric[0][m][n]) - expected[m][n]) < mp.mpf("1e-30")
