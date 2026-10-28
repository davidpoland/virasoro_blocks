"""Skeleton (numerator-free) regularization at singular ``c``.

At recursion-singular loci -- ``c = 1``, ``c = 25``, minimal models below 1,
and the ``c > 25`` rational-``b^2`` dual family (c = 50 from b^2 = 1/6,
c = 26 from 2/3, c = 33 from 1/3, ...) -- the regularized *pole structure* (denominator roots +
numerator degrees) can be recovered WITHOUT building the ~1e500-coefficient
symbolic numerators. The single strip that genuinely needs the numerators -- the
surviving-pair Kac *collision* strip -- is now performed analytically on the
skeleton by distinct ``(side, chiral_weight)`` counting inside each coincident
Delta cluster.

`regularize_delta_skeleton` exploits this: it epsilon-extrapolates only the
exp_base + poles, carries the (c-independent) numerator degrees straight
through, applies the numerator-free removable-pole strip, and skips the
symbolic-numerator path for collision stripping. The interpolation then reads
degrees + poles from the skeleton
and sample *values* from the regularized numeric sampler, so it is bit-identical
to the full symbolic path's interpolant -- only the conditioning differs (no
giant-polynomial evaluation).
"""

from __future__ import annotations

import mpmath as mp

import pytest

from virasoro_blocks.polyderivs import B_zzb_poly_table, parity_projected_B_zzb_poly_table, interpolate_delta_tables
import virasoro_blocks.regularization as R

SCALAR_EXT = [mp.mpf("0.1")] * 4
SCALAR_EXT_DISTINCT = [
    mp.mpf("0.05"), mp.mpf("0.11"), mp.mpf("0.07"), mp.mpf("0.13"),
]
SPIN_H = [mp.mpf("0.13")] * 4
SPIN_HBAR = [mp.mpf("0.07")] * 4
SPIN_H_DISTINCT = [
    mp.mpf("0.19"), mp.mpf("0.17"), mp.mpf("0.16"), mp.mpf("0.18"),
]
SPIN_HBAR_DISTINCT = [
    mp.mpf("0.07"), mp.mpf("0.05"), mp.mpf("0.09"), mp.mpf("0.03"),
]

# c = 1, c = 25, and three c > 25 rational-b^2 duals.
SINGULAR_CE_GE_ONE = ["1", "25", "50", "26", "33"]
# c = 1, its b = 1 dual c = 25, and a real-b c > 25 point: one of each regime
SINGULAR_CE_GE_ONE_REPRESENTATIVE = ["1", "25", "50"]


def _scalar_sampler_kwargs(ell, lam):
    return _scalar_sampler_kwargs_for(SCALAR_EXT, ell, lam)


def _scalar_sampler_kwargs_for(h_ext, ell, lam):
    return dict(
        coordinate="zzb", chiral=False, spinning=False, projected=False,
        h_externals=h_ext, hbar_externals=h_ext, ell=ell,
        m_max=lam, n_max=lam, b=None, q_order=8, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=None, center_bar=None,
        parity=None, parity_external=None, exchanged_parity=None,
    )


def _scalar_offdiag_sampler_kwargs(ell, lam):
    return dict(
        coordinate="zzb", chiral=False, spinning=False, projected=False,
        h_externals=SCALAR_EXT, hbar_externals=SCALAR_EXT, ell=ell,
        m_max=lam, n_max=lam, b=None, q_order=8, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=mp.mpf("0.37"), center_bar=mp.mpf("0.55"),
        parity=None, parity_external=None, exchanged_parity=None,
    )


def _spin_sampler_kwargs(ell, lam):
    return _spin_sampler_kwargs_for(SPIN_H, SPIN_HBAR, ell, lam)


def _spin_sampler_kwargs_for(h_ext, hbar_ext, ell, lam):
    return dict(
        coordinate="zzb", chiral=False, spinning=True, projected=False,
        h_externals=h_ext, hbar_externals=hbar_ext, ell=ell,
        m_max=lam, n_max=lam, b=None, q_order=8, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=None, center_bar=None,
        parity=None, parity_external=None, exchanged_parity=None,
    )


def _spin_offdiag_sampler_kwargs(ell, lam):
    return dict(
        coordinate="zzb", chiral=False, spinning=True, projected=False,
        h_externals=SPIN_H, hbar_externals=SPIN_HBAR, ell=ell,
        m_max=lam, n_max=lam, b=None, q_order=8, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=mp.mpf("0.37"), center_bar=mp.mpf("0.55"),
        parity=None, parity_external=None, exchanged_parity=None,
    )


def _projected_sampler_kwargs(ell, lam, parity, center=None, center_bar=None):
    return _projected_sampler_kwargs_for(
        SPIN_H, SPIN_HBAR, ell, lam, parity, center=center, center_bar=center_bar
    )


def _projected_sampler_kwargs_for(
    h_ext, hbar_ext, ell, lam, parity, center=None, center_bar=None
):
    return dict(
        coordinate="zzb", chiral=False, spinning=True, projected=True,
        h_externals=h_ext, hbar_externals=hbar_ext, ell=ell,
        m_max=lam, n_max=lam, b=None, q_order=8, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=center, center_bar=center_bar,
        parity=parity, parity_external=None, exchanged_parity="1",
    )


def _assert_regularized_interpolants_match_direct(
    *, c, coordinate, full, skel, sampler_kwargs, values, rel_tol, npoles=None
):
    full.numeric_sampler = R._RegularizedNumericSampler(c=c, **sampler_kwargs)
    skel.numeric_sampler = R._RegularizedNumericSampler(c=c, **sampler_kwargs)
    if npoles is None:
        npoles = len(full.denominator_roots)
    it_full = interpolate_delta_tables(
        {coordinate: full}, npoles=npoles,
        pole_selection="contribution",
    )[coordinate]
    it_skel = interpolate_delta_tables(
        {coordinate: skel}, npoles=npoles,
        pole_selection="contribution",
    )[coordinate]

    sampler = R._RegularizedNumericSampler(c=c, **sampler_kwargs)
    for value in values:
        direct = sampler([value])[0][0]
        ef = it_full.evaluate(value)
        es = it_skel.evaluate(value)
        for m in range(len(es)):
            for n in range(len(es[0])):
                efv = mp.mpc(ef[m][n])
                esv = mp.mpc(es[m][n])
                dv = mp.mpc(direct[m][n])
                scale = max(abs(efv), abs(esv), abs(dv), mp.mpf("1e-40"))
                assert abs(efv - esv) < rel_tol * scale
                assert abs(esv - dv) < rel_tol * scale
                assert abs(efv - dv) < rel_tol * scale


from virasoro_blocks.polyderivs import build_numeric_sampler  # noqa: E402


def _regularized_numeric_grid_with_workdps(
    *, c, delta, sampler_kwargs, work_dps, max_work_dps
):
    def build(c_val):
        return build_numeric_sampler(c=c_val, **sampler_kwargs)

    return R.regularized_numeric_grids(
        build, [delta], c=c, n_points=5,
        work_dps=work_dps, max_work_dps=max_work_dps,
    )[0]


# --- the structural premise: collision strip is a no-op at c >= 1 -------------


# --- skeleton pole structure == full symbolic post-strip structure -----------


# --- c < 1 minimal models are now covered by the skeleton ---------------------


# --- interpolation: skeleton == full, and both match the direct numeric block -


# --- parity_projected_B: skeleton interpolant == full interpolant -------------
#
# parity_projected_B also has a structural numerator degree, but with a twist:
# its swapped term combines two DIFFERENT chiral halves (h-externals vs hbar-
# externals). Their leading-in-h coefficients are proportional across orders, so
# the swap determinant vanishes identically and the leading terms add (swap sign
# != -1) or cancel exactly (swap sign == -1, the diagonal entry vanishing) --
# exactly scalar_G's even/odd-ell pattern. Unlike scalar_G's same-object swap,
# the cancellation is between distinct polynomials, so the FULL build's arb
# assembly leaves a roundoff residue at the canceled top order and its degree
# array can read one higher at the order-Lambda boundary. The skeleton uses the
# exact structural degree; the two interpolants still agree on the evaluated
# block (the full build's extra node redundantly fits the same lower-degree
# polynomial), which is the property asserted here.


def _build_projected_full_and_skeleton(c, ell, lam, parity, center, center_bar=None):
    full = R.regularize_delta_table(
        lambda cc: parity_projected_B_zzb_poly_table(
            ell, SPIN_H, SPIN_HBAR, lam, lam, parity=parity, parity_external=None,
            exchanged_parity="1", c=cc, q_order=8, center=center,
            center_bar=center_bar, lambda_cut=None,
        ),
        c=c, q_order=8, h_externals=SPIN_H, hbar_externals=SPIN_HBAR,
    )
    skel = R.regularize_delta_skeleton(
        lambda cc: parity_projected_B_zzb_poly_table(
            ell, SPIN_H, SPIN_HBAR, lam, lam, parity=parity, parity_external=None,
            exchanged_parity="1", c=cc, q_order=8, center=center,
            center_bar=center_bar, lambda_cut=None, skeleton=True,
        ),
        c=c, q_order=8, h_externals=SPIN_H, hbar_externals=SPIN_HBAR,
    )
    return full, skel


# a covering set: every (c, parity), (c, externals) and (parity, externals) pair once


@pytest.mark.parametrize("center_bar", [None, mp.mpf("0.55")])
def test_projected_skeleton_degree_rule(center_bar):
    """The skeleton's structural degree rule: parity '-' (swap sign -1) cancels
    on the diagonal (degree -1 off the table diagonal, m == n entry vanishing),
    while parity '+' and every off-diagonal (z != zbar) build keep the full
    cd_h[m] + cd_hbar[n] degree (no cancellation)."""
    mp.mp.dps = 60
    lam = 3
    ell = 2
    c = mp.mpf("10")  # generic c: a plain (non-regularized) skeleton build
    z0 = mp.mpf("0.37")
    off_diagonal = center_bar is not None

    def degrees(parity):
        t = parity_projected_B_zzb_poly_table(
            ell, SPIN_H, SPIN_HBAR, lam, lam, parity=parity, parity_external=None,
            exchanged_parity="1", c=c, q_order=8, center=z0, center_bar=center_bar,
            skeleton=True,
        )
        assert t.numerators is None
        return t.numerator_degrees

    # ordered_B supplies the un-canceled cd_h[m] + cd_hbar[n] reference.
    ref = B_zzb_poly_table(
        ell, SPIN_H, SPIN_HBAR, lam, lam, c=c, q_order=8, center=z0,
        center_bar=center_bar, skeleton=True,
    ).numerator_degrees

    dp = degrees("+")
    dm = degrees("-")
    for m in range(lam + 1):
        for n in range(lam + 1):
            if m + n > lam:  # corner-skipped entries report degree 0
                continue
            # parity '+' never cancels
            assert dp[m][n] == ref[m][n]
            if off_diagonal:
                # off-diagonal never cancels for either parity
                assert dm[m][n] == ref[m][n]
            elif m == n:
                # diagonal entry vanishes identically under the '-' swap
                assert dm[m][n] == 0
            else:
                # off-diagonal-of-the-table entry drops by one
                assert dm[m][n] == ref[m][n] - 1


# --- chiral_F: skeleton == full, both reproduce the direct chiral block -------
#
# The chiral F(h, z) table is a degenerate (m_max + 1) x 1 DeltaDerivTable
# ("Delta" = h), with no swap and no antiholomorphic half, so its numerator
# degree is purely structural (D + m) -- the cleanest skeleton of all. The same
# regularize machinery serves it: the removable / collision strips map the
# "chiral" provenance side tag to h_externals.


def _chiral_sampler_kwargs(lam):
    return dict(
        coordinate="z", chiral=True, spinning=False, projected=False,
        h_externals=SCALAR_EXT, hbar_externals=None, ell=0,
        m_max=lam, n_max=0, b=None, q_order=8, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=None, center_bar=None,
        parity=None, parity_external=None, exchanged_parity=None,
    )


# --- folded coordinates (ws / xt / qqb_ws): skeleton == full at the block level
#
# The fold is a Delta-independent linear recombination of the non-folded
# numerators over distinct (m+n) monomial blocks, so each folded entry's degree
# is the max of the structural source degrees -- no top-degree cancellation
# ACROSS the fold. (scalar_G and parity '+' projected fold from exact-canceling
# sources, so skeleton degrees match the full build; parity '-' folds from the
# roundoff-fragile canceling sources, so the structural degree can read one
# lower than the full build -- but the evaluated blocks still agree.)


def _folded_scalar_kwargs(ell, lam):
    return dict(
        coordinate="xt", chiral=False, spinning=False, projected=False,
        h_externals=SCALAR_EXT, hbar_externals=SCALAR_EXT, ell=ell,
        m_max=lam, n_max=lam, b=None, q_order=8, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=None, center_bar=None,
        parity=None, parity_external=None, exchanged_parity=None,
    )


def _folded_projected_kwargs(ell, lam, parity):
    return dict(
        coordinate="xt", chiral=False, spinning=True, projected=True,
        h_externals=SPIN_H, hbar_externals=SPIN_HBAR, ell=ell,
        m_max=lam, n_max=lam, b=None, q_order=8, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=None, center_bar=None,
        parity=parity, parity_external=["1", "1", "1", "1"], exchanged_parity="1",
    )


# --- radial slices (ws_radial / xt_radial / qqb_ws_radial): skeleton == full ---
#
# A radial slice is the n = 0 column of its folded parent. The skeleton builds
# the folded table at n_max = 0 (so the chiral Taylor order is the minimal
# m_max + (odd ? 1 : 0), NOT the full folded m_max + 2*n_max), and the numeric
# sampler maps the radial coordinate to the folded parent at n_max = 0.




@pytest.mark.parametrize('c', [10, 30])
@pytest.mark.parametrize('parity,phases', [('-', None), ('+', ('-1', '1', '1', '1'))])
def test_projected_full_degree_matches_skeleton_after_exact_cancellation(c, parity, phases):
    """Exact external labels must not expose roundoff as an extra leading power."""
    from virasoro_blocks.numeric_input import parse_number

    h = tuple(map(parse_number, ['.75', '-.2', '.25', '.3']))
    hb = tuple(map(parse_number, ['-.25', '.8', '.25', '.3']))
    with mp.workdps(40):
        kwargs = dict(c=c, q_order=8, parity=parity, parity_external=phases)
        full = parity_projected_B_zzb_poly_table(1, h, hb, 2, 2, **kwargs)
        skeleton = parity_projected_B_zzb_poly_table(1, h, hb, 2, 2, skeleton=True, **kwargs)
        assert [[p.degree for p in row] for row in full.numerators] == skeleton.numerator_degrees
