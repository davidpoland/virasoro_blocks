"""Numeric per-Delta parity-projected B tables and the global ordered /
parity-projected / folded-G completions added for full coordinate coverage.

The numeric builders are validated against the symbolic-in-Delta poly tables
of the same name (`polyderivs`) where they exist, and against the scalar /
ordered global blocks by the factorization identities otherwise.
"""
import mpmath as mp
import pytest

import virasoro_blocks.derivs as d
from virasoro_blocks.polyderivs import (
    parity_projected_B_qqb_poly_table,
    parity_projected_B_qqb_ws_poly_table,
    parity_projected_B_ws_poly_table,
    parity_projected_B_xt_poly_table,
    parity_projected_B_yyb_poly_table,
    parity_projected_B_zzb_poly_table,
)

HE = [mp.mpf("0.1"), mp.mpf("0.13"), mp.mpf("0.07"), mp.mpf("0.11")]
HBE = [mp.mpf("0.12"), mp.mpf("0.09"), mp.mpf("0.15"), mp.mpf("0.08")]
PE = ["1", "1", "1", "1"]
C = mp.mpf("18")
Q = 12  # numeric and symbolic tables are compared at the same q_order
DELTA = mp.mpf("1.6")
ELL = 2
M = 3  # symbolic builders index the swap combination on a square grid
CN = 4
ZS = mp.mpf("-0.2")
ZBS = mp.mpf("-0.15")


def _maxrel(got, ref):
    rows = len(ref)
    cols = len(ref[0])
    scale = max(abs(ref[m][n]) for m in range(rows) for n in range(cols))
    return max(
        abs(got[m][n] - ref[m][n]) for m in range(rows) for n in range(cols)
    ) / scale


@pytest.mark.parametrize(
    "numeric,symbolic,parity",
    [
        # one parity per coordinate, alternating, so each parity meets both
        # non-folded and folded coordinates
        (d.parity_projected_B_zzb_table, parity_projected_B_zzb_poly_table, "+"),
        (d.parity_projected_B_yyb_table, parity_projected_B_yyb_poly_table, "-"),
        (d.parity_projected_B_qqb_table, parity_projected_B_qqb_poly_table, "+"),
        (d.parity_projected_B_xt_table, parity_projected_B_xt_poly_table, "-"),
        (d.parity_projected_B_ws_table, parity_projected_B_ws_poly_table, "+"),
        (d.parity_projected_B_qqb_ws_table, parity_projected_B_qqb_ws_poly_table, "-"),
    ],
)
def test_numeric_parity_projected_B_matches_symbolic(numeric, symbolic, parity):
    """Each numeric parity-projected B table reproduces the symbolic-in-Delta
    poly table of the same name, evaluated at one Delta."""
    mp.mp.dps = 60
    # Full-grid comparison against the numeric table. Non-folded symbolic
    # builders now default to the order-Lambda triangle, so request the full
    # square; folded builders truncate via m+2n intrinsically (no lambda_cut).
    folded = "ws" in symbolic.__name__ or "xt" in symbolic.__name__
    extra = {} if folded else {"lambda_cut": None}
    sym = symbolic(
        ELL, HE, HBE, M, M, c=C, q_order=Q, parity=parity, parity_external=PE,
        **extra,
    )
    ref = sym.evaluate(DELTA)
    got = numeric(
        DELTA, ELL, HE, HBE, M, M, parity=parity, parity_external=PE,
        c=C, q_order=Q,
    )
    assert _maxrel(got, ref) < mp.mpf(10) ** -45


@pytest.mark.parametrize(
    "radial,full",
    [
        (d.parity_projected_B_xt_radial_table, d.parity_projected_B_xt_table),
        (d.parity_projected_B_ws_radial_table, d.parity_projected_B_ws_table),
        (
            d.parity_projected_B_qqb_ws_radial_table,
            d.parity_projected_B_qqb_ws_table,
        ),
    ],
)
def test_numeric_parity_projected_B_radial_is_column0(radial, full):
    """The radial slice is the s = 0 column of the folded table."""
    mp.mp.dps = 50
    r = radial(
        DELTA, ELL, HE, HBE, M, parity="-", parity_external=PE, c=C, q_order=Q
    )
    grid = full(
        DELTA, ELL, HE, HBE, M, 0, parity="-", parity_external=PE, c=C, q_order=Q
    )
    assert all(r[m] == grid[m][0] for m in range(M + 1))


def test_folded_parity_projected_B_rejects_complex_phase():
    """Folded coordinates need a real z<->zbar parity phase."""
    for dps in (5, 40):
        with mp.workdps(dps):
            with pytest.raises(ValueError, match="real"):
                d.parity_projected_B_xt_table(
                    DELTA, ELL, HE, HBE, M, parity="+", parity_external=["i", "1", "1", "1"],
                    c=C, q_order=Q,
                )


def test_folded_samplers_reject_complex_phases_without_restricting_unfolded_blocks():
    from virasoro_blocks.derivs import fold_parity_marker
    from virasoro_blocks.global_blocks import build_global_sampler
    from virasoro_blocks.large_spin import _LargeSpinTwistSampler
    from virasoro_blocks.polyderivs import build_numeric_sampler

    ext = ['.125']*4
    builders = (
        lambda coord, **phases: build_global_sampler(coord, basis='global_parity_projected_B',
            ell=0, h_externals=ext, hbar_externals=ext, m_max=0, n_max=0,
            parity='+', **phases),
        lambda coord, **phases: _LargeSpinTwistSampler('global', coord, ext, 4, None, 0, 0, '1',
            basis='parity_projected_B', parity_projection='+', **phases),
        lambda coord, **phases: build_numeric_sampler(coord, chiral=False, chiral_norm=4,
            spinning=True, projected=True, h_externals=ext, hbar_externals=ext,
            ell=0, m_max=0, n_max=0, c=32, q_order=2, parity='+', **phases),
    )
    for dps in (5, 40):
        with mp.workdps(dps):
            for i, builder in enumerate(builders):
                for phases in ({'exchanged_parity': 'i'}, {'parity_external': ['i', '1', '1', '1']}):
                    for coordinate in (('xt', 'ws') if i == 1 else ('xt', 'ws', 'xt_radial')):
                        with pytest.raises(ValueError, match='real.*parity phase'):
                            builder(coordinate, **phases)
                    unfolded = builder('zzb', **phases)
                    if i == 2:
                        values, _ = unfolded([mp.mpf(2)])
                        value = values[0]
                    else:
                        value = unfolded(mp.mpf(2))
                    assert value[0][0].imag != 0
                # Two imaginary unit phases give a valid real -1 total phase.
                builder('xt', exchanged_parity='i', parity_external=['i', '1', '1', '1'])
            for bad in (0, 2, mp.j, mp.inf, mp.nan):
                with pytest.raises(ValueError, match='real.*parity phase'):
                    fold_parity_marker(parity='+', phase=bad)


# --- global ordered / parity-projected / folded-G completions --------------


def test_global_B_factorizes_scalar_G():
    """Global ordered B(h, h) equals the global scalar block at ell = 0."""
    mp.mp.dps = 60
    h = DELTA / 2
    B = d.global_B_zzb_table(
        h, h, HE, HE, M, M, chiral_norm=CN, z_shift=ZS, zbar_shift=ZS
    )
    G0 = d.global_G_zzb_table(DELTA, 0, HE, M, M, chiral_norm=CN, z_shift=ZS)
    assert _maxrel([[B[m][n] for n in range(M + 1)] for m in range(M + 1)], G0) \
        < mp.mpf(10) ** -45


@pytest.mark.parametrize("ell,parity", [(2, "+"), (3, "+")])
def test_global_parity_projected_reduces_to_scalar_G(ell, parity):
    """Scalar externals + the '+' projection reduce the global parity-projected
    B block to the global scalar block (non-folded and folded) at every spin,
    odd included: the scalar G is z <-> zbar even."""
    mp.mp.dps = 60
    P = d.global_parity_projected_B_zzb_table(
        DELTA, ell, HE, HE, M, M, parity=parity, parity_external=PE,
        chiral_norm=CN, z_shift=ZS, zbar_shift=ZS,
    )
    G = d.global_G_zzb_table(DELTA, ell, HE, M, M, chiral_norm=CN, z_shift=ZS)
    assert _maxrel(P, G) < mp.mpf(10) ** -45

    Pf = d.global_parity_projected_B_xt_table(
        DELTA, ell, HE, HE, M, M, parity=parity, parity_external=PE,
        chiral_norm=CN, z_shift=ZS, zbar_shift=ZS,
    )
    Gf = d.global_G_xt_table(DELTA, ell, HE, M, M, chiral_norm=CN, z_shift=ZS)
    assert _maxrel(Pf, Gf) < mp.mpf(10) ** -45


def test_global_G_qqb_ws_and_radials_run():
    """The new folded global scalar qqb_ws table and the three global folded
    radial slices build and agree with their parent column 0."""
    mp.mp.dps = 50
    qw = d.global_G_qqb_ws_table(DELTA, ELL, HE, M, M, chiral_norm=CN, z_shift=ZS)
    assert len(qw) == M + 1 and len(qw[0]) == M + 1
    for radial, full in [
        (d.global_G_xt_radial_table, d.global_G_xt_table),
        (d.global_G_ws_radial_table, d.global_G_ws_table),
        (d.global_G_qqb_ws_radial_table, d.global_G_qqb_ws_table),
    ]:
        r = radial(DELTA, ELL, HE, M, chiral_norm=CN, z_shift=ZS)
        grid = full(DELTA, ELL, HE, M, 0, chiral_norm=CN, z_shift=ZS)
        assert all(r[m] == grid[m][0] for m in range(M + 1))
