"""Large-spin asymptotic blocks (fixed twist, variable spin L)."""
import mpmath as mp
import pytest
from collections import Counter
from fractions import Fraction

from virasoro_blocks.derivs import (
    G_qqb_table,
    G_qqb_ws_table,
    G_ws_table,
    G_xt_table,
    G_yyb_table,
    G_zzb_table,
    evaluate_block,
    global_G_qqb_table,
    global_G_qqb_ws_table,
    global_G_ws_table,
    global_G_xt_table,
    global_G_yyb_table,
    global_G_zzb_table,
)
from virasoro_blocks.global_blocks import global_evaluate_block
from virasoro_blocks.large_spin import (
    _LargeSpinTwistSampler, large_spin_base, large_spin_twist_table,
)

HE = mp.mpf("0.0625")
ZS = -2 * HE
CN = 4
LAM = 5
C = mp.mpf(32)
Q = 40


_EXACT_BUILDERS = {
    "virasoro": {"zzb": G_zzb_table, "yyb": G_yyb_table, "qqb": G_qqb_table},
    "global": {
        "zzb": global_G_zzb_table, "yyb": global_G_yyb_table,
        "qqb": global_G_qqb_table,
    },
}


def _exact(mode, coordinate, L, twist, m_max, n_max):
    delta = mp.mpf(L) + mp.mpf(twist)
    builder = _EXACT_BUILDERS[mode][coordinate]
    if mode == "virasoro":
        return builder(delta, int(L), [HE] * 4, m_max, n_max, c=C, q_order=Q,
                       chiral_norm=CN, z_shift=ZS)
    return builder(delta, int(L), [HE] * 4, m_max, n_max,
                   chiral_norm=CN, z_shift=ZS)


@pytest.mark.parametrize("mode", ["global", "virasoro"])
@pytest.mark.parametrize("coordinate", ["zzb", "yyb", "qqb"])
@pytest.mark.parametrize("twist", ["0", "0.25", "2.25"])
def test_factorization_matches_numeric_block(mode, coordinate, twist):
    """The k_h(z) k_hbar(zbar) + swap factorization (via the sampler) reproduces
    the exact numeric 2d scalar block at even integer L, twist 0 and twist > 0."""
    mp.mp.dps = 80
    twist = mp.mpf(twist)
    sampler = _LargeSpinTwistSampler(mode, coordinate, [HE] * 4, CN, ZS, LAM, LAM, twist,
                                     c=C, q_order=Q)
    for L in (4, 12, 20):
        mine = sampler(L)
        ref = _exact(mode, coordinate, L, twist, LAM, LAM)
        for m in range(LAM + 1):
            for n in range(LAM + 1):
                denom = abs(ref[m][n]) + mp.mpf(10) ** -60
                assert abs(mine[m][n] - ref[m][n]) / denom < mp.mpf(10) ** -50


@pytest.mark.parametrize("c_str", ["1/2", "1", "25"])
@pytest.mark.parametrize("L", [12])
def test_large_spin_regularizes_at_singular_c(c_str, L):
    """At a recursion-singular c the bare chiral evaluator divides by zero. The
    sampler uses the same merged Laurent recursion as the Delta tables.
    Covers both regimes: the
    minimal model c = 1/2 and c = 1 (imaginary / branch-point b), and c = 25
    (real b = 1, the dual of c = 1).

    Validate the regularized grid against the independently regularized
    Delta-table block at Delta = L + twist, ell = L (a separate code path).
    Even L and a generic twist, so neither h = L + twist/2 nor hbar = twist/2
    lands on a Kac pole (where the block genuinely diverges).
    """
    from virasoro_blocks.polyderivs import G_zzb_poly_table
    from virasoro_blocks.regularization import merged_delta_table

    mp.mp.dps = 60
    c = {"1/2": mp.mpf(1) / 2, "1": mp.mpf(1), "25": mp.mpf(25)}[c_str]
    twist = mp.mpf("0.7")
    h_ext = [mp.mpf("0.1")] * 4
    q = 12

    sampler = _LargeSpinTwistSampler(
        "virasoro", "zzb", h_ext, None, mp.mpf(0), 2, 2, twist, c=c, q_order=q
    )
    grid = sampler(L)

    treg = merged_delta_table(lambda cc: G_zzb_poly_table(L, h_ext, 2, 2, c=cc, q_order=q, lambda_cut=None), c=c)
    ref = treg.evaluate(mp.mpf(L) + twist)
    for m in range(3):
        for n in range(3):
            denom = abs(mp.mpc(ref[m][n])) + mp.mpf(10) ** -60
            rel = abs(mp.mpc(grid[m][n]) - mp.mpc(ref[m][n])) / denom
            assert rel < mp.mpf(10) ** -30, f"(m,n)=({m},{n}) rel={rel}"


@pytest.mark.parametrize("coordinate", ["zzb", "xt_radial"])
def test_large_spin_contribution_interpolates_at_c1(coordinate):
    """At c = 1 the full-Virasoro tower interpolates through the fixed-spin-
    style forced-positive-pole split: Bohr-Sommerfeld sees only the selectable
    left-of-Lmin poles while the returned prefactor still carries the physical
    positive real Kac poles."""
    mp.mp.dps = 80
    c = mp.mpf("1")
    twist = mp.mpf("0.7")
    h_ext = [mp.mpf("0.1")] * 4
    sampler = _LargeSpinTwistSampler(
        "virasoro", coordinate, h_ext, 4, mp.mpf("-0.2"), 2, 2, twist,
        c=c, q_order=20,
    )
    interp = large_spin_twist_table(
        twist,
        mode="virasoro",
        coordinate=coordinate,
        h_externals=h_ext,
        m_max=2,
        n_max=2,
        Lmax=mp.mpf("8"),
        npoles=8,
        c=c,
        q_order=20,
        chiral_norm=4,
        z_shift=mp.mpf("-0.2"),
        pole_selection="contribution",
    )
    for L in (8, 10, 12, 14):
        got = interp.evaluate(mp.mpf(L))
        ref = sampler(L)
        rel = abs(mp.mpc(got[0][0]) - mp.mpc(ref[0][0])) / max(
            abs(mp.mpc(ref[0][0])), mp.mpf("1e-40")
        )
        assert rel < mp.mpf("1e-6"), f"L={L} rel={rel}"


def test_large_spin_c1_keeps_genuine_inregion_poles():
    """Small principal coefficients must not hide genuine positive-axis poles.

    The c=1 block has a double pole at h=81/4, confirmed independently by
    the growth of its point values. The merged skeleton's surviving orders
    are kept even when finite-distance probes would look regular.
    """
    from virasoro_blocks.recursion import F_block

    with mp.workdps(140):
        scaled = [F_block(Fraction(81, 4)+eps, Fraction(1, 2), [Fraction(1, 10)]*4,
                          c=1, q_order=20, chiral_norm=4, z_shift=Fraction(-1, 5))*eps**2
                  for eps in (Fraction(1, 10**30), Fraction(1, 10**40))]
        assert scaled[1] != 0
        assert mp.almosteq(*scaled, rel_eps=mp.mpf('1e-10'), abs_eps=0)
    mp.mp.dps = 60
    c = mp.mpf("1")
    twist = Fraction(7, 10)
    h_ext = [Fraction(1, 10)] * 4
    sampler = _LargeSpinTwistSampler(
        "virasoro", "zzb", h_ext, 4, mp.mpf("-0.2"), 2, 2, twist,
        c=c, q_order=20,
    )
    interp = large_spin_twist_table(
        twist,
        mode="virasoro",
        coordinate="zzb",
        h_externals=h_ext,
        m_max=2,
        n_max=2,
        Lmax=mp.mpf("8"),
        npoles=19,
        c=c,
        q_order=20,
        chiral_norm=4,
        z_shift=mp.mpf("-0.2"),
        pole_selection="contribution",
    )
    # h=81/4 maps to L=19.9, or x=L-8=11.9, at order two.
    assert sum(mp.almosteq(p, mp.mpf('11.9')) for p in interp.prefactor.poles) == 2
    # The interpolant reproduces the block away from its genuine poles.
    for L in (8, 10, 12, 14):
        got = interp.evaluate(mp.mpf(L))
        ref = sampler(L)
        rel = abs(mp.mpc(got[0][0]) - mp.mpc(ref[0][0])) / max(
            abs(mp.mpc(ref[0][0])), mp.mpf("1e-40")
        )
        assert rel < mp.mpf("1e-5"), f"L={L} rel={rel}"


def test_large_spin_c1_pool_is_minimal_kac():
    """At c = 1 the Virasoro tower pole pool is the *minimal* Kac set: the chiral
    K_{r,s} = 0 removable poles (the even |r - s| family for identical externals,
    d_{r,s} = (r - s)^2 / 4) are stripped by the regularized-skeleton K-gate, and
    only the genuine odd-|r - s| poles survive. Locks in that the chiral-K strip
    is applied to the tower pool (so it is not carrying removable poles)."""
    from virasoro_blocks.large_spin import _virasoro_chiral_pole_pool
    mp.mp.dps = 70
    twist = mp.mpf("0.7")
    h_ext = [mp.mpf("0.1")] * 4
    pool = _virasoro_chiral_pole_pool(
        h_externals=h_ext, c=mp.mpf("1"), q_order=20, twist=twist,
    )
    # Each real pole sits at L = d_{r,s} - twist/2 with d_{r,s} = (r-s)^2/4 at
    # c = 1, so |r - s| = 2 * sqrt(L + twist/2). The genuine (non-K-removable)
    # set is exactly the odd |r - s|; no even |r - s| pole may remain.
    for p in pool:
        if abs(mp.im(p)) > mp.mpf("1e-20"):
            continue  # complex poles do not occur at c = 1, but skip if any
        h = mp.re(p) + twist / 2
        if h <= 0:
            continue
        rs = int(mp.nint(2 * mp.sqrt(h)))
        assert rs % 2 == 1, (
            f"even |r-s|={rs} pole at L={mp.nstr(mp.re(p),5)} survived "
            "(should be K-removable)"
        )


def test_large_spin_c1_prefactor_sign_definite_with_genuine_poles():
    """Genuine in-region poles have even order, retaining a positive prefactor."""
    mp.mp.dps = 70
    twist = mp.mpf("0.7")
    h_ext = [mp.mpf("0.1")] * 4
    interp = large_spin_twist_table(
        twist, mode="virasoro", coordinate="zzb", h_externals=h_ext,
        m_max=2, n_max=2, Lmax=mp.mpf("8"), npoles=12, c=mp.mpf("1"),
        q_order=20, chiral_norm=4, z_shift=mp.mpf("-0.2"),
        pole_selection="contribution",
    )
    forced = [p for p in interp.prefactor.poles
              if abs(mp.im(p)) < mp.mpf("1e-20") and mp.re(p) > 0]
    assert forced
    assert all(count % 2 == 0 for count in Counter(forced).values())
    # The prefactor is real and sign-definite (no sign change) across the domain.
    pref = interp.prefactor
    signs = {mp.sign(mp.re(pref.evaluate(mp.mpf(x)))) for x in [0, 1, 3, 5, 10, 20, 40]}
    assert signs == {mp.mpf(1)}, f"prefactor changes sign on the domain: {signs}"


@pytest.mark.parametrize("c", ["0.5", "0.7"])
def test_large_spin_c_below_1_tower_matches_block(c):
    """The Virasoro tower builds and interpolates at c < 1 minimal models. The
    recursion is singular there (the sampler uses the merged Laurent table), and the chiral
    pole pool carries surviving-pair Kac *collisions* -- distinct pairs sharing a
    chiral root, the (r, s) ~ (p - r, q - s) finite-Kac-table identifications that
    only exist below c = 1. Those removable surplus poles need the validated
    collision strip, so the c < 1 pool is taken from the full (numerator-carrying)
    regularized table rather than the skeleton (a c >= 1-only path). Locks in the
    whole composed pipeline against the direct numeric sampler."""
    mp.mp.dps = 40
    c = mp.mpf(c)
    twist = mp.mpf("0.4")
    h_ext = [mp.mpf("0.1")] * 4
    interp = large_spin_twist_table(
        twist, mode="virasoro", coordinate="zzb", h_externals=h_ext,
        m_max=2, n_max=2, Lmax=mp.mpf("20"), npoles=6, c=c, q_order=10,
        exchanged_parity="1", basis="scalar_G",
    )
    sampler = _LargeSpinTwistSampler(
        "virasoro", "zzb", h_ext, None, None, 2, 2, twist,
        c=c, q_order=10, exchanged_parity="1", basis="scalar_G",
    )
    for L in [mp.mpf("25"), mp.mpf("40")]:
        ref = sampler(L)[0][0]
        val = interp.evaluate(L)[0][0]
        rel = abs((val - ref) / ref)
        assert rel < mp.mpf("1e-8"), (
            f"c={float(c)} L={float(L)}: rel err {mp.nstr(rel, 4)} too large"
        )


def test_large_spin_c_below_1_pool_drops_collision_surplus():
    """At a c < 1 minimal model the chiral pole pool drops the surviving-pair
    collision surplus: the full regularized chiral table has fewer roots than the
    naive direct (unstripped) skeleton build, and the tower pool inherits the
    stripped set. (At c = 0.7 the direct skeleton over-counts by two.)"""
    from virasoro_blocks.polyderivs import F_z_poly_table
    from virasoro_blocks.regularization import merged_delta_table
    mp.mp.dps = 50
    c = mp.mpf("0.7")
    h_ext = [mp.mpf("0.1")] * 4
    direct = F_z_poly_table(h_ext, 1, c=c, q_order=10, skeleton=True)
    stripped = merged_delta_table(lambda cc: F_z_poly_table(h_ext, 1, c=cc, q_order=10, skeleton=False), c=c)
    assert len(stripped.denominator_roots) < len(direct.denominator_roots), (
        "regularized chiral table should drop collision-surplus poles at c < 1"
    )


@pytest.mark.parametrize("coordinate", ["zzb", "yyb"])
def test_large_spin_distinct_scalar_externals(coordinate):
    """The tower supports nonidentical *scalar* externals (h_i != h_j, each with
    h_i == hbar_i): the single external set feeds both chiral halves of the
    k_h k_hbar + swap factorization, which is exact for scalars. (Spinning
    externals, h_i != hbar_i, are a different block and rejected at the CLI.)
    """
    mp.mp.dps = 80
    h_ext = [mp.mpf("0.05"), mp.mpf("0.11"), mp.mpf("0.07"), mp.mpf("0.13")]
    sampler = _LargeSpinTwistSampler(
        "virasoro", coordinate, h_ext, CN, ZS, LAM, LAM, mp.mpf("0.25"),
        c=C, q_order=Q,
    )
    builder = G_zzb_table if coordinate == "zzb" else G_yyb_table
    for L in (4, 12):
        mine = sampler(L)
        ref = builder(mp.mpf(L) + mp.mpf("0.25"), int(L), h_ext, LAM, LAM,
                      c=C, q_order=Q, chiral_norm=CN, z_shift=ZS)
        for m in range(LAM + 1):
            for n in range(LAM + 1):
                denom = abs(ref[m][n]) + mp.mpf(10) ** -60
                assert abs(mine[m][n] - ref[m][n]) / denom < mp.mpf(10) ** -50


@pytest.mark.parametrize("L", [11, 13])
def test_large_spin_tower_at_odd_spin_matches_fixed_spin_block(L):
    """The scalar_G tower G = k_h k_hbar + k_hbar k_h is z <-> zbar even at every spin, so the one (default)
    tower reproduces the fixed-spin G_zzb at odd ell as well as even.  The z <-> zbar-odd k_h k_hbar - k_hbar k_h
    is the parity_projected_B tower with projection '-', which matches the ordered-B difference instead.
    """
    from virasoro_blocks.polyderivs import B_zzb_poly_table, G_zzb_poly_table

    mp.mp.dps = 50
    # distinct scalars so odd-spin operators are physical
    h_ext = [mp.mpf("0.05"), mp.mpf("0.11"), mp.mpf("0.07"), mp.mpf("0.13")]
    twist, c, q = mp.mpf("0.7"), mp.mpf(5), 12

    tower = _LargeSpinTwistSampler(
        "virasoro", "zzb", h_ext, None, mp.mpf(0), 3, 3, twist,
        c=c, q_order=q,  # default exchanged_parity="1"
    )
    odd = _LargeSpinTwistSampler(
        "virasoro", "zzb", h_ext, None, mp.mpf(0), 3, 3, twist,
        c=c, q_order=q, basis="parity_projected_B", parity_projection="-",
    )
    delta = mp.mpf(L) + twist
    ref = G_zzb_poly_table(L, h_ext, 3, 3, c=c, q_order=q, lambda_cut=None).evaluate(delta)
    forward = B_zzb_poly_table(L, h_ext, h_ext, 3, 3, c=c, q_order=q, lambda_cut=None).evaluate(delta)
    reverse = B_zzb_poly_table(-L, h_ext, h_ext, 3, 3, c=c, q_order=q, lambda_cut=None).evaluate(delta)

    def maxrel(g, target):
        return max(
            abs(mp.mpc(g[m][n]) - mp.mpc(target[m][n])) / (1 + abs(mp.mpc(target[m][n])))
            for m in range(4) for n in range(4)
        )

    grid = tower(L)
    assert maxrel(grid, ref) < mp.mpf(10) ** -40
    for m in range(4):
        for n in range(4):
            assert abs(mp.mpc(grid[m][n]) - mp.mpc(grid[n][m])) < mp.mpf(10) ** -40   # z <-> zbar even
    difference = [[forward[m][n] - reverse[m][n] for n in range(4)] for m in range(4)]
    assert maxrel(odd(L), difference) < mp.mpf(10) ** -40
    assert maxrel(odd(L), ref) > mp.mpf(10) ** -30


@pytest.mark.parametrize("phase", ["-1", "i"])
def test_large_spin_scalar_G_tower_refuses_exchanged_parity(phase):
    """There is one scalar_G tower (z <-> zbar even): an exchanged parity other than +1 is refused, pointing to
    the parity_projected_B tower for the z <-> zbar-odd structure."""
    with pytest.raises(ValueError, match="parity_projected_B"):
        _LargeSpinTwistSampler(
            "virasoro", "zzb", [mp.mpf("0.1")] * 4, None, mp.mpf(0), 2, 2,
            mp.mpf("0.7"), c=mp.mpf(5), q_order=12, exchanged_parity=phase,
        )
    with pytest.raises(ValueError, match="no odd scalar_G tower"):
        large_spin_twist_table(
            mp.mpf("0.7"), mode="global", coordinate="zzb", h_externals=[mp.mpf("0.1")] * 4,
            m_max=2, n_max=2, Lmax=3, npoles=4, exchanged_parity=phase,
        )


def test_large_spin_singular_sampler_is_picklable():
    """The singular-c sampler must survive pickling for the ProcessPool build:
    `__getstate__` drops the per-probe regularization evaluators (unpicklable
    closures), which each worker rebuilds on first call."""
    import pickle

    mp.mp.dps = 40
    sampler = _LargeSpinTwistSampler(
        "virasoro", "zzb", [mp.mpf("0.1")] * 4, None, mp.mpf(0), 2, 2,
        mp.mpf("0.7"), c=mp.mpf(1) / 2, q_order=12,
    )
    grid = sampler(8)  # prime the probe-evaluator cache
    revived = pickle.loads(pickle.dumps(sampler))
    grid2 = revived(8)  # worker rebuilds evaluators from dropped state
    for m in range(3):
        for n in range(3):
            assert abs(mp.mpc(grid[m][n]) - mp.mpc(grid2[m][n])) < mp.mpf(10) ** -50


_SPIN_H = [mp.mpf("1.0"), mp.mpf("0.6"), mp.mpf("0.5"), mp.mpf("1.1")]
_SPIN_HBAR = [mp.mpf("0.0"), mp.mpf("0.6"), mp.mpf("0.5"), mp.mpf("0.1")]  # spins 1,0,0,1


def _spin_maxrel(g, ref, N=3):
    return max(
        abs(mp.mpc(g[m][n]) - mp.mpc(ref[m][n])) / (1 + abs(mp.mpc(ref[m][n])))
        for m in range(N) for n in range(N)
    )


@pytest.mark.parametrize("L", [10, 11, 12])
def test_large_spin_ordered_B_spinning_matches_block(L):
    """ordered_B tower with non-identical (bosonic) spinning externals
    reproduces the fixed-spin B_zzb block at Delta = L + twist, ell = L. No
    parity restriction -- ordered_B is a single ordering, so any L works."""
    from virasoro_blocks.polyderivs import B_zzb_poly_table

    mp.mp.dps = 50
    twist, c, q = mp.mpf("0.7"), mp.mpf(5), 14
    s = _LargeSpinTwistSampler(
        "virasoro", "zzb", _SPIN_H, None, mp.mpf(0), 2, 2, twist,
        c=c, q_order=q, hbar_externals=_SPIN_HBAR, basis="ordered_B",
    )
    ref = B_zzb_poly_table(L, _SPIN_H, _SPIN_HBAR, 2, 2, c=c, q_order=q, lambda_cut=None).evaluate(
        mp.mpf(L) + twist)
    assert _spin_maxrel(s(L), ref) < mp.mpf(10) ** -30


@pytest.mark.parametrize("parity", ["+", "-"])
@pytest.mark.parametrize("L", [10, 11])
def test_large_spin_parity_projected_B_spinning_matches_block(parity, L):
    """parity_projected_B tower (non-identical spinning externals) reproduces
    the fixed-spin parity_projected_B_zzb block for both projections."""
    from virasoro_blocks.polyderivs import parity_projected_B_zzb_poly_table

    mp.mp.dps = 50
    twist, c, q = mp.mpf("0.7"), mp.mpf(5), 14
    s = _LargeSpinTwistSampler(
        "virasoro", "zzb", _SPIN_H, None, mp.mpf(0), 2, 2, twist,
        c=c, q_order=q, hbar_externals=_SPIN_HBAR, basis="parity_projected_B",
        parity_projection=parity, exchanged_parity="1",
    )
    ref = parity_projected_B_zzb_poly_table(
        L, _SPIN_H, _SPIN_HBAR, 2, 2, parity=parity, exchanged_parity="1",
        c=c, q_order=q, lambda_cut=None,  # tower emits the full square
    ).evaluate(mp.mpf(L) + twist)
    assert _spin_maxrel(s(L), ref) < mp.mpf(10) ** -30


def test_large_spin_spinning_regularizes_at_singular_c():
    """Two-external-set regularization (Pitfall E): the spinning ordered_B tower
    reproduces the regularized fixed-spin block at c = 1/2."""
    from virasoro_blocks.polyderivs import B_zzb_poly_table
    from virasoro_blocks.regularization import merged_delta_table

    mp.mp.dps = 50
    twist, q, c = mp.mpf("0.7"), 14, mp.mpf(1) / 2
    s = _LargeSpinTwistSampler(
        "virasoro", "zzb", _SPIN_H, None, mp.mpf(0), 2, 2, twist,
        c=c, q_order=q, hbar_externals=_SPIN_HBAR, basis="ordered_B",
    )
    L = 12
    treg = merged_delta_table(lambda cc: B_zzb_poly_table(L, _SPIN_H, _SPIN_HBAR, 2, 2, c=cc, q_order=q, lambda_cut=None), c=c)
    assert _spin_maxrel(s(L), treg.evaluate(mp.mpf(L) + twist)) < mp.mpf(10) ** -20


@pytest.mark.parametrize("parity", ["+", "-"])
def test_large_spin_projected_spinning_regularizes_at_singular_c(parity):
    """At c = 1/2 the parity_projected_B large-spin tower reproduces the
    regularized fixed-spin projected block for both parity projections."""
    from virasoro_blocks.polyderivs import parity_projected_B_zzb_poly_table
    from virasoro_blocks.regularization import merged_delta_table

    mp.mp.dps = 50
    twist, q, c = mp.mpf("0.7"), 14, mp.mpf(1) / 2
    s = _LargeSpinTwistSampler(
        "virasoro", "zzb", _SPIN_H, None, mp.mpf(0), 2, 2, twist,
        c=c, q_order=q, hbar_externals=_SPIN_HBAR, basis="parity_projected_B",
        parity_projection=parity, exchanged_parity="1",
    )
    L = 12
    treg = merged_delta_table(lambda cc: parity_projected_B_zzb_poly_table(
            L, _SPIN_H, _SPIN_HBAR, 2, 2, parity=parity,
            exchanged_parity="1", c=cc, q_order=q, lambda_cut=None,
        ), c=c)
    assert _spin_maxrel(s(L), treg.evaluate(mp.mpf(L) + twist)) < mp.mpf(10) ** -20


# Singular-c tower pole-selection scorer (folded / off-diagonal / diagonal):
# (kind, coordinate, kwargs) configurations exercising the generalized merged
# Laurent scorer `_singular_merged_residue_scores`.
_SINGULAR_SCORER_CASES = [
    ("diag", "zzb", {}),
    ("offdiag", "zzb", {"center": mp.mpf("0.4"), "center_bar": mp.mpf("0.55")}),
    ("offdiag", "yyb", {"center": mp.mpf("0.4"), "center_bar": mp.mpf("0.55")}),
    ("offdiag", "qqb", {"center": mp.mpf("0.4"), "center_bar": mp.mpf("0.55")}),
    ("folded", "xt", {"exchanged_parity": "1"}),
    ("folded", "ws", {"exchanged_parity": "1"}),
    ("ordered", "zzb", {"basis": "ordered_B"}),
    ("proj", "xt", {"basis": "parity_projected_B", "parity_projection": "+",
                    "parity_external": ["1", "1", "1", "1"]}),
    ("proj", "xt", {"basis": "parity_projected_B", "parity_projection": "-",
                    "parity_external": ["1", "1", "1", "1"]}),
]


@pytest.mark.parametrize("c_str", ["25", "1"])
@pytest.mark.parametrize("kind,coordinate,extra", _SINGULAR_SCORER_CASES)
def test_singular_tower_analytic_scorer_matches_contour(c_str, kind, coordinate, extra):
    """At a recognition-singular c the tower's closed-form merged-Laurent
    pole-selection scorer (`_singular_merged_residue_scores`) must agree with the
    finite-radius contour/DFT scorer to ~machine precision, for the diagonal,
    off-diagonal (center != center_bar), and folded (xt / ws) coordinates and
    the scalar_G / ordered_B / parity_projected_B bases."""
    from virasoro_blocks.interpolation import pole_selection_score_entries
    from virasoro_blocks.large_spin import _virasoro_chiral_pole_pool
    from virasoro_blocks.polyderivs import (
        _residue_score_indices_from_dims, _value_score_entry_from_sampler,
        principal_part_contribution_scores,
    )

    mp.mp.dps = 60
    c = mp.mpf(c_str)
    twist, q, lam, npoles = mp.mpf("0.7"), 8, 2, 6
    h_ext = [mp.mpf("0.1")] * 4
    base_kwargs = dict(exchanged_parity="1", basis="scalar_G")
    base_kwargs.update(extra)
    sampler = _LargeSpinTwistSampler(
        "virasoro", coordinate, h_ext, None, mp.mpf(0), lam, lam, twist,
        c=c, q_order=q, **base_kwargs,
    )
    assert sampler._singular  # this c routes through the merged scorer
    shift = mp.mpf(20)
    base = large_spin_base(
        "virasoro", None, center=extra.get("center"),
        center_bar=extra.get("center_bar"),
    )
    roots = [mp.mpc(r) for r in _virasoro_chiral_pole_pool(
        h_externals=h_ext, c=c, q_order=q, twist=twist,
    )]
    value_entry = _value_score_entry_from_sampler(sampler, n_max=lam)
    entries = _residue_score_indices_from_dims(
        lam, lam, pole_selection_score_entries("contribution"),
        value_entry=value_entry if value_entry is not None else (0, 0),
    )

    analytic = sampler._analytic_residue_scores(roots, shift, entries, base)
    assert analytic is not None  # the merged scorer covers this configuration

    sample_bucket = getattr(sampler, "sample_bucket", None)

    def _eval_block(pts, ent):
        if sample_bucket is not None:
            res = sample_bucket(list(pts), ent)
            if res is not None:
                return res
        return [sampler(p) for p in pts], False

    contour = principal_part_contribution_scores(
        roots, _eval_block, delta_eval=shift, entries=entries,
        exp_base=base, n_select=len(roots),
    )
    worst = max(
        abs(a - b) / max(abs(b), mp.mpf(1)) for a, b in zip(analytic, contour)
    )
    assert worst < mp.mpf(10) ** -15


# Fermionic externals: half-integer external spins s_i = h_i - hbar_i = 1/2.
_FERM_H = [mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("0.7"), mp.mpf("0.7")]
_FERM_HBAR = [mp.mpf("0.0"), mp.mpf("0.0"), mp.mpf("0.2"), mp.mpf("0.2")]

@pytest.mark.parametrize("L", [mp.mpf(10), mp.mpf("10.5"), mp.mpf("11.5")])
def test_large_spin_fermionic_ordered_B_matches_block(L):
    """Fermionic externals (half-integer external spins) and a half-integer
    exchanged spin L: the ordered_B tower reproduces B_zzb (which accepts signed
    half-integer ell). The continuous-L interpolation evaluates fine at
    half-integer L."""
    from virasoro_blocks.polyderivs import B_zzb_poly_table

    mp.mp.dps = 50
    twist, c, q = mp.mpf("0.7"), mp.mpf(5), 14
    s = _LargeSpinTwistSampler(
        "virasoro", "zzb", _FERM_H, None, mp.mpf(0), 2, 2, twist,
        c=c, q_order=q, hbar_externals=_FERM_HBAR, basis="ordered_B",
    )
    ref = B_zzb_poly_table(L, _FERM_H, _FERM_HBAR, 2, 2, c=c, q_order=q, lambda_cut=None).evaluate(
        L + twist)
    assert _spin_maxrel(s(L), ref) < mp.mpf(10) ** -30


@pytest.mark.parametrize("parity_external,exchanged", [
    (["i", "-i", "i", "-i"], "1"),
    (["i", "i", "i", "i"], "-1"),
])
def test_large_spin_fermionic_parity_projected_phases_match_block(parity_external, exchanged):
    """The Z4 external parity phases (i / -i) feed parity_reversal_phase, giving
    a (generally complex) reversed-block coefficient; the parity_projected_B
    tower matches the fixed-spin block with the same phases."""
    from virasoro_blocks.polyderivs import parity_projected_B_zzb_poly_table

    mp.mp.dps = 50
    twist, c, q, L = mp.mpf("0.7"), mp.mpf(5), 14, mp.mpf(10)
    s = _LargeSpinTwistSampler(
        "virasoro", "zzb", _FERM_H, None, mp.mpf(0), 2, 2, twist,
        c=c, q_order=q, hbar_externals=_FERM_HBAR, basis="parity_projected_B",
        parity_projection="+", parity_external=parity_external,
        exchanged_parity=exchanged,
    )
    ref = parity_projected_B_zzb_poly_table(
        L, _FERM_H, _FERM_HBAR, 2, 2, parity="+", parity_external=parity_external,
        exchanged_parity=exchanged, c=c, q_order=q, lambda_cut=None,  # tower=full square
    ).evaluate(L + twist)
    assert _spin_maxrel(s(L), ref) < mp.mpf(10) ** -30


def test_decay_base_matches_per_unit_L_ratio():
    """The block decays as base^L with base = large_spin_base."""
    mp.mp.dps = 80
    sampler = _LargeSpinTwistSampler("global", "zzb", [HE] * 4, CN, ZS, 1, 1,
                                     mp.mpf("0.25"))
    base = large_spin_base("global", CN)
    t1 = sampler(40)[0][0]
    t2 = sampler(60)[0][0]
    ratio = (t2 / t1) ** (mp.mpf(1) / 20)
    assert abs(ratio - base) / base < mp.mpf(10) ** -3


@pytest.mark.parametrize("mode,tol", [("global", mp.mpf(10) ** -16),
                                      ("virasoro", mp.mpf(10) ** -9)])
@pytest.mark.parametrize("coordinate", ["zzb", "yyb"])
def test_interpolated_table_reproduces_block(mode, tol, coordinate):
    """The L-interpolant reproduces the block inside the window, measured
    relative to the per-L table scale (per-entry relative error is meaningless
    on the structurally tiny / zero-crossing high-(m,n) entries)."""
    mp.mp.dps = 100
    twist = mp.mpf("0.25")
    tab = large_spin_twist_table(
        twist, mode=mode, coordinate=coordinate,
        h_externals=[HE] * 4, m_max=LAM, n_max=LAM,
        Lmax=14, npoles=8, c=C, q_order=Q, z_shift=ZS, chiral_norm=CN)
    worst = mp.mpf(0)
    for L in (16, 20, 24):
        approx = tab.evaluate(mp.mpf(L))
        ref = _exact(mode, coordinate, L, twist, LAM, LAM)
        scale = max(abs(ref[m][n]) for m in range(LAM + 1) for n in range(LAM + 1))
        for m in range(LAM + 1):
            for n in range(LAM + 1):
                worst = max(worst, abs(approx[m][n] - ref[m][n]) / scale)
    assert worst < tol, f"{mode}: worst table-scale rel {worst}"


_EXACT_FOLDED_BUILDERS = {
    "virasoro": {"xt": G_xt_table, "ws": G_ws_table, "qqb_ws": G_qqb_ws_table},
    "global": {
        "xt": global_G_xt_table, "ws": global_G_ws_table,
        "qqb_ws": global_G_qqb_ws_table,
    },
}


def _exact_folded(mode, coordinate, L, twist, m_max, n_max):
    delta = mp.mpf(L) + mp.mpf(twist)
    builder = _EXACT_FOLDED_BUILDERS[mode][coordinate]
    if mode == "virasoro":
        return builder(delta, int(L), [HE] * 4, m_max, n_max, c=C, q_order=Q,
                       chiral_norm=CN, z_shift=ZS)
    return builder(delta, int(L), [HE] * 4, m_max, n_max,
                   chiral_norm=CN, z_shift=ZS)


# both modes, every folded coordinate, and even and odd spins of the one tower -- not the full product
@pytest.mark.parametrize(
    "mode, tol, coordinate, L0",
    [
        ("virasoro", mp.mpf(10) ** -9, "xt", 16),
        ("virasoro", mp.mpf(10) ** -9, "qqb_ws", 17),
        ("global", mp.mpf(10) ** -16, "ws", 16),
        ("global", mp.mpf(10) ** -16, "xt", 17),
    ],
    ids=["virasoro-xt-even", "virasoro-qqb_ws-odd", "global-ws-even", "global-xt-odd"],
)
@pytest.mark.parametrize("radial", [False, True], ids=["full", "radial"])
def test_folded_interpolated_table_reproduces_block(
    mode, tol, coordinate, L0, radial
):
    """The folded `xt`/`ws` L-interpolant reproduces the fixed-spin folded
    scalar block (`scalar_G`) for both modes, at odd as well as even integer
    spins: the scalar G is z <-> zbar even at every spin, so one tower covers
    them all. The fold is the same L-independent `_fold_parity` recombination
    as the fixed-spin tables, so it must agree at the integer spins inside the
    window."""
    exchanged_parity = "1"
    mp.mp.dps = 60
    twist = mp.mpf("0.25")
    tab = large_spin_twist_table(
        twist, mode=mode, coordinate=coordinate + ("_radial" if radial else ""),
        h_externals=[HE] * 4, m_max=LAM, n_max=LAM,
        Lmax=L0 - 2, npoles=8, c=C, q_order=Q, z_shift=ZS, chiral_norm=CN,
        basis="scalar_G", exchanged_parity=exchanged_parity)
    worst = mp.mpf(0)
    for L in (L0, L0 + 8):  # same parity as L0
        approx = tab.evaluate(mp.mpf(L))
        ref = _exact_folded(mode, coordinate, L, twist, LAM, LAM)
        n_count = 1 if radial else LAM + 1
        assert len(approx) == LAM + 1
        assert all(len(row) == n_count for row in approx)
        scale = max(abs(ref[m][n]) for m in range(LAM + 1) for n in range(n_count))
        for m in range(LAM + 1):
            for n in range(n_count):
                worst = max(worst, abs(approx[m][n] - ref[m][n]) / scale)
    assert worst < tol, f"{mode} {coordinate} p={exchanged_parity}: rel {worst}"


@pytest.mark.parametrize("parity,L0", [("-", 17)])
@pytest.mark.parametrize("coordinate", ["xt", "xt_radial"])
def test_folded_parity_projected_tower_matches_numeric_block(parity, L0, coordinate):
    """The folded parity-projected B tower (global mode) reproduces the numeric
    fixed-spin folded parity-projected block (an independent code path) at the
    integer spins inside the window."""
    mp.mp.dps = 100
    twist = mp.mpf("0.25")
    hbar_ext = [mp.mpf("0.1")] * 4
    pext = ["1", "1", "1", "1"]
    tab = large_spin_twist_table(
        twist, mode="global", coordinate=coordinate,
        h_externals=[HE] * 4, hbar_externals=hbar_ext, m_max=LAM, n_max=LAM,
        Lmax=L0 - 2, npoles=8, z_shift=ZS, chiral_norm=CN,
        basis="parity_projected_B", parity_projection=parity,
        parity_external=pext)
    from virasoro_blocks.derivs import global_parity_projected_B_xt_table
    worst = mp.mpf(0)
    for L in (L0, L0 + 4):
        approx = tab.evaluate(mp.mpf(L))
        ref = global_parity_projected_B_xt_table(
            mp.mpf(L) + twist, int(L), [HE] * 4, hbar_ext, LAM, LAM,
            parity=parity, parity_external=pext, chiral_norm=CN, z_shift=ZS,
            zbar_shift=ZS)
        n_count = 1 if coordinate.endswith("_radial") else LAM + 1
        assert all(len(row) == n_count for row in approx)
        scale = max(abs(ref[m][n]) for m in range(LAM + 1) for n in range(n_count))
        for m in range(LAM + 1):
            for n in range(n_count):
                worst = max(worst, abs(approx[m][n] - ref[m][n]) / scale)
    assert worst < mp.mpf(10) ** -14, f"parity={parity}: rel {worst}"


@pytest.mark.parametrize("coordinate", ["xt", "xt_radial", "ws_radial", "qqb_ws_radial"])
def test_folded_ordered_B_rejected(coordinate):
    """`ordered_B` has no parity fold, so folded coordinates must reject it."""
    with pytest.raises(ValueError, match="no parity fold"):
        large_spin_twist_table(
            mp.mpf("0.25"), mode="global", coordinate=coordinate,
            h_externals=[HE] * 4, m_max=2, n_max=2, Lmax=6, npoles=6,
            basis="ordered_B")


@pytest.mark.parametrize("coordinate", ["zzb", "yyb"])
def test_bootstrap_spec_wiring_and_reuse(coordinate):
    """The spec large-spin fields append one tower channel per twist, built once
    in prepare and reused across gaps."""
    mp.mp.dps = 60
    from virasoro_bootstrap.identical_scalar import (
        IdenticalScalarGapSpec, prepare_identical_scalar_gap_build,
        build_identical_scalar_gap_problem)
    from virasoro_bootstrap.large_spin import double_twist_set

    spins = (0, 2)
    twists = tuple(double_twist_set(HE, 2))  # 2 double-twist slices

    def spec(gap):
        return IdenticalScalarGapSpec(
            c=C, h_ext=HE, delta_gap=mp.mpf(gap), delta_min_by_spin={}, lam=3,
            coordinate=coordinate,
            spin_list=spins, q_order=12, dps=60,            npoles=6, pole_selection="residue", num_threads=1,
            large_spin_lmin=mp.mpf(4), large_spin_twists=twists)

    prepared = prepare_identical_scalar_gap_build(spec("0.9"))
    assert len(prepared.large_spin_channels) == len(twists)
    # reuse the same prepared across two gaps; total = spins + tower
    for gap in ("1.0", "1.1"):
        problem = build_identical_scalar_gap_problem(spec(gap), prepared=prepared)
        assert len(problem.spin_channels) == len(spins) + len(twists)
        assert all(int(ch.ell) == 4 for ch in problem.spin_channels[len(spins):])
        assert problem.spec.large_spin_twists == twists


def test_double_twist_gap_filter():
    from virasoro_bootstrap.large_spin import double_twist_set
    # Delta_ext = 2*HE = 0.125 -> tau_n = 0.25 + 2n
    assert [float(t) for t in double_twist_set(HE, 3)] == [0.25, 2.25, 4.25]
    # a twist gap 1.0 is a problem's choice: twists_above_gap drops 0.25 (<= gap) and puts the gap first
    from virasoro_bootstrap.large_spin import twists_above_gap

    got = [float(t) for t in twists_above_gap(double_twist_set(HE, 3), 1.0)]
    assert got == [1.0, 2.25, 4.25]


def test_mode_validation():
    with pytest.raises(ValueError):
        large_spin_base("bogus", CN)
    with pytest.raises(ValueError):
        large_spin_twist_table(mp.mpf("0.25"), mode="virasoro", h_externals=[HE] * 4,
                               m_max=2, n_max=2, Lmax=14, npoles=6)  # missing c/q


def test_large_spin_offdiagonal_global_sampler_matches_pointwise_oracle():
    mp.mp.dps = 70
    twist = mp.mpf("0.7")
    z0, zbar0 = mp.mpf("0.37"), mp.mpf("0.55")
    sampler = _LargeSpinTwistSampler(
        "global", "zzb", [HE] * 4, CN, ZS, 2, 2, twist,
        basis="scalar_G", exchanged_parity="1", center=z0, center_bar=zbar0,
    )
    L = mp.mpf(12)
    got = sampler(L)
    h, hbar = L + twist / 2, twist / 2
    b1 = global_evaluate_block(
        z0, zbar0, h, hbar, [HE] * 4, coord="zzb", m_max=2, n_max=2,
        chiral_norm=CN, z_shift=ZS, zbar_shift=ZS,
    )
    b2 = global_evaluate_block(
        z0, zbar0, hbar, h, [HE] * 4, coord="zzb", m_max=2, n_max=2,
        chiral_norm=CN, z_shift=ZS, zbar_shift=ZS,
    )
    ref = [[b1[m][n] + b2[m][n] for n in range(3)] for m in range(3)]
    scale = max(abs(ref[m][n]) for m in range(3) for n in range(3))
    err = max(abs(got[m][n] - ref[m][n]) for m in range(3) for n in range(3))
    assert err / scale < mp.mpf(10) ** -30


def test_large_spin_offdiagonal_virasoro_sampler_matches_pointwise_oracle():
    mp.mp.dps = 70
    twist = mp.mpf("0.7")
    z0, zbar0 = mp.mpf("0.37"), mp.mpf("0.55")
    sampler = _LargeSpinTwistSampler(
        "virasoro", "zzb", [HE] * 4, CN, ZS, 2, 2, twist,
        c=C, q_order=Q, basis="scalar_G", exchanged_parity="1",
        center=z0, center_bar=zbar0,
    )
    L = mp.mpf(12)
    got = sampler(L)
    h, hbar = L + twist / 2, twist / 2
    b1 = evaluate_block(
        z0, zbar0, h, hbar, [HE] * 4, c=C, q_order=Q,
        coord="zzb", m_max=2, n_max=2, chiral_norm=CN, z_shift=ZS,
    )
    b2 = evaluate_block(
        z0, zbar0, hbar, h, [HE] * 4, c=C, q_order=Q,
        coord="zzb", m_max=2, n_max=2, chiral_norm=CN, z_shift=ZS,
    )
    ref = [[b1[m][n] + b2[m][n] for n in range(3)] for m in range(3)]
    scale = max(abs(ref[m][n]) for m in range(3) for n in range(3))
    err = max(abs(got[m][n] - ref[m][n]) for m in range(3) for n in range(3))
    assert err / scale < mp.mpf(10) ** -20


def test_large_spin_offdiagonal_interpolated_table_matches_oracle():
    mp.mp.dps = 80
    twist = mp.mpf("0.7")
    z0, zbar0 = mp.mpf("0.37"), mp.mpf("0.55")
    # The analytic contribution scorer selects the poles nearest the sample
    # region (the genuine, twist-shifted block poles), which is optimal for
    # interpolation; far extrapolation to L = 16 just wants a couple more poles
    # than the legacy deep-pole heuristic's lucky np=8 sweet spot. np=12 sits in
    # the production large_spin_npoles range and is well within the convergent
    # regime (L=16 error ~4e-8).
    tab = large_spin_twist_table(
        twist, mode="global", coordinate="zzb", h_externals=[HE] * 4,
        m_max=2, n_max=2, Lmax=10, npoles=12, z_shift=ZS, chiral_norm=CN,
        basis="scalar_G", exchanged_parity="1", center=z0, center_bar=zbar0,
    )
    for L in (mp.mpf(12), mp.mpf(16)):
        got = tab.evaluate(L)
        h, hbar = L + twist / 2, twist / 2
        b1 = global_evaluate_block(
            z0, zbar0, h, hbar, [HE] * 4, coord="zzb", m_max=2, n_max=2,
            chiral_norm=CN, z_shift=ZS, zbar_shift=ZS,
        )
        b2 = global_evaluate_block(
            z0, zbar0, hbar, h, [HE] * 4, coord="zzb", m_max=2, n_max=2,
            chiral_norm=CN, z_shift=ZS, zbar_shift=ZS,
        )
        ref = [[b1[m][n] + b2[m][n] for n in range(3)] for m in range(3)]
        scale = max(abs(ref[m][n]) for m in range(3) for n in range(3))
        err = max(abs(got[m][n] - ref[m][n]) for m in range(3) for n in range(3))
        assert err / scale < mp.mpf(10) ** -6


@pytest.mark.parametrize("coordinate", ["xt", "xt_radial", "ws_radial", "qqb_ws_radial"])
def test_large_spin_folded_rejects_offdiagonal_centers(coordinate):
    with pytest.raises(ValueError, match="off-diagonal centers"):
        large_spin_twist_table(
            mp.mpf("0.25"), mode="global", coordinate=coordinate, h_externals=[HE] * 4,
            m_max=2, n_max=2, Lmax=8, npoles=6, center=mp.mpf("0.37"),
        )


def test_sample_bucket_matches_full_grid():
    """`_LargeSpinTwistSampler.sample_bucket` (entry-restricted) returns the
    same values as the full per-node `__call__` grid for the requested
    entries -- only the (m, n) the interpolation bucket consumes are built.
    The full grid takes the shared assembler's flint matmul path and the
    entry-restricted build its elementwise path, so they agree to working
    precision (sub-ulp), not bit-for-bit."""
    mp.mp.dps = 80
    twist = mp.mpf("1.0")
    sampler = _LargeSpinTwistSampler(
        "virasoro", "zzb", [HE] * 4, CN, ZS, LAM, LAM, twist, c=C, q_order=Q,
    )
    values = [mp.mpf(12), mp.mpf(17), mp.mpf(23)]
    entries = [(0, 0), (2, 3), (LAM, 1), (1, LAM)]
    grids, _is_arb = sampler.sample_bucket(values, entries)
    assert grids is not None
    tol = mp.mpf(10) ** -70
    for v, grid in zip(values, grids):
        full = sampler(v)
        for (m, n) in entries:
            assert abs(grid[m][n] - full[m][n]) <= tol * (abs(full[m][n]) + 1), \
                (v, m, n)


def test_sample_bucket_matches_full_grid_at_singular_c():
    """Batch regularization agrees with per-node grids on the requested entries."""
    mp.mp.dps = 70
    twist = mp.mpf("0.7")
    sampler = _LargeSpinTwistSampler(
        "virasoro", "zzb", [HE] * 4, None, mp.mpf(0), 2, 2, twist,
        c=mp.mpf("1/2"), q_order=12,
    )
    values = [mp.mpf(4), mp.mpf(12), mp.mpf(20)]
    entries = [(0, 0), (1, 2), (2, 1)]
    grids, _is_arb = sampler.sample_bucket(values, entries)
    assert grids is not None
    tol = mp.mpf(10) ** -28
    for v, grid in zip(values, grids):
        full = sampler(v)
        for (m, n) in entries:
            assert abs(grid[m][n] - full[m][n]) <= tol * (abs(full[m][n]) + 1), \
                (v, m, n)


@pytest.mark.parametrize("parity", ["1", "-1"])
@pytest.mark.parametrize("coordinate", ["zzb", "yyb"])
def test_large_spin_entry_restricted_table_matches_full(parity, coordinate):
    # the z <-> zbar-odd tower ("-1") is the parity_projected_B '-' projection (the scalar_G tower is even)
    basis = dict(basis="scalar_G") if parity == "1" else dict(basis="parity_projected_B", parity_projection="-")
    """The entry-restricted interpolation (default) reproduces the full-grid
    path to working precision (it skips only unconsumed outer-product entries;
    the full path uses the shared assembler's flint matmul and the restricted
    path its elementwise build, so they agree sub-ulp, not bit-for-bit)."""
    mp.mp.dps = 100

    def build():
        return large_spin_twist_table(
            mp.mpf("1.0"), mode="virasoro", coordinate=coordinate,
            h_externals=[HE] * 4, m_max=6, n_max=6, Lmax=10, npoles=10,
            c=C, q_order=Q, pole_selection="contribution", **basis,
        )

    new = build()
    saved = _LargeSpinTwistSampler.sample_bucket
    _LargeSpinTwistSampler.sample_bucket = lambda *a, **k: None
    try:
        old = build()
    finally:
        _LargeSpinTwistSampler.sample_bucket = saved

    tol = mp.mpf(10) ** -75
    for rn, ro in zip(new.numerators, old.numerators):
        for pn, po in zip(rn, ro):
            assert len(pn.coeffs) == len(po.coeffs)
            for a, b in zip(pn.coeffs, po.coeffs):
                assert abs(a - b) <= tol * (abs(b) + 1)


# ---------------------------------------------------------------------------
# contribution_optimal on the large-spin tower
# ---------------------------------------------------------------------------


def _tower_kw():
    import mpmath as mp

    return dict(
        mode="virasoro", coordinate="zzb", h_externals=[mp.mpf("0.1")] * 4,
        m_max=3, n_max=3, Lmax=mp.mpf(8), c=mp.mpf(8) / 7, q_order=20,
        chiral_norm=mp.mpf(16), z_shift=-2 * mp.mpf("0.1"),
    )


def test_contribution_optimal_runs_on_the_virasoro_tower():
    """The Virasoro tower's pole pool is npoles-INDEPENDENT, so it qualifies.

    The complete Kac pool is bounded by ``q_order``. The global tower follows
    the same rule with its q-order-defined SL(2) pool; the shared core accepts
    both fixed candidate universes.
    """
    import mpmath as mp

    from virasoro_blocks.large_spin import large_spin_twist_table

    with mp.workdps(60):
        table = large_spin_twist_table(
            mp.mpf("1.2"), npoles=12,
            pole_selection="contribution_optimal", **_tower_kw()
        )
        assert table is not None


def test_contribution_optimal_runs_on_the_q_order_global_pool():
    """The global tower uses its q-order-bounded SL(2) pole universe."""
    import mpmath as mp

    from virasoro_blocks.large_spin import large_spin_twist_table

    kw = _tower_kw()
    kw.update(mode="global", c=None, q_order=20)
    with mp.workdps(60):
        short = large_spin_twist_table(
            mp.mpf("1.2"), npoles=6,
            pole_selection="contribution_optimal", **kw
        )
        long = large_spin_twist_table(
            mp.mpf("1.2"), npoles=8,
            pole_selection="contribution_optimal", **kw
        )
        assert list(short.prefactor.poles) == list(long.prefactor.poles[:6])


def test_contribution_optimal_is_the_most_accurate_tower_selector():
    """It beats both the configured default and bernstein, by many orders.

    Measured against the EXACT numeric tower block. This matters because the
    tower must keep ``large_spin_npoles`` LOW (8-12) for conditioning reasons,
    and the accuracy gap is largest exactly there -- ``contribution_optimal`` at
    npoles=8 beats ``contribution_potential`` at npoles=20 in the large-L tail.
    """
    import mpmath as mp

    from virasoro_blocks.large_spin import (
        _LargeSpinTwistSampler, large_spin_twist_table,
    )

    with mp.workdps(60):
        twist = mp.mpf("1.2")
        sampler = _LargeSpinTwistSampler(
            "virasoro", "zzb", [mp.mpf("0.1")] * 4, mp.mpf(16),
            -2 * mp.mpf("0.1"), 3, 3, twist, c=mp.mpf(8) / 7, q_order=20,
        )
        probes = [mp.mpf(x) for x in (8, 12, 40)]
        exact = {L: sampler(L)[0][0] for L in probes}

        def worst(selector, npoles):
            table = large_spin_twist_table(
                twist, npoles=npoles, pole_selection=selector, **_tower_kw()
            )
            return max(
                abs(table.evaluate(L)[0][0] - exact[L]) / abs(exact[L])
                for L in probes
            )

        optimal = worst("contribution_optimal", 12)
        bernstein = worst("contribution_bernstein", 12)
        potential = worst("contribution_potential", 12)

        assert optimal < bernstein
        assert bernstein < potential
        # The gaps are not marginal: orders of magnitude, not factors.
        assert optimal < bernstein * mp.mpf(10) ** -3
        assert optimal < potential * mp.mpf(10) ** -10


@pytest.mark.parametrize("center_bar", [None, ".4"])
def test_tower_sampler_keeps_tiny_exact_c_offset(center_bar):
    from fractions import Fraction
    c = Fraction(1) + Fraction(1, 10**60)
    with mp.workdps(30):
        sampler = _LargeSpinTwistSampler(
            "virasoro", "zzb", [HE] * 4, CN, ZS, 1, 1, ".3",
            c=c, q_order=4, center_bar=center_bar,
        )
        assert not sampler._singular
        got = sampler(4)
        bucket, _ = sampler.sample_bucket([mp.mpf(4)], [(0, 0), (1, 1)])
        for m, n in [(0, 0), (1, 1)]:
            assert abs(bucket[0][m][n] - got[m][n]) < mp.mpf("1e-28") * max(1, abs(got[m][n]))
    with mp.workdps(100):
        reference = _LargeSpinTwistSampler(
            "virasoro", "zzb", [HE] * 4, CN, ZS, 1, 1, ".3",
            c=c, q_order=4, center_bar=center_bar,
        )(4)
        for m in range(2):
            for n in range(2):
                assert abs(got[m][n] - reference[m][n]) < mp.mpf("1e-28") * max(1, abs(reference[m][n]))


@pytest.mark.parametrize("mode", ["global", "virasoro"])
@pytest.mark.parametrize("coordinate,center", [
    ("zzb", None), ("yyb", None), ("qqb", None),
    ("xt", None), ("ws", None), ("qqb_ws", None),
    ("zzb", ".3"), ("yyb", ".3"), ("qqb", ".3"),
])
def test_tower_sampler_refreshes_fixed_half_after_precision_change(mode, coordinate, center):
    def make():
        return _LargeSpinTwistSampler(
            mode, coordinate, [HE] * 4, CN, ZS, 1, 1, mp.mpf(".25"), c=32, q_order=4,
            center=center, center_bar=".4" if center is not None else None,
        )
    with mp.workdps(20):
        sampler = make()
        sampler(4)
    with mp.workdps(60):
        from virasoro_blocks.derivs import _arb_to_mpf

        entries = [(m, n) for m in range(2) for n in range(2)]
        bucket, is_arb = sampler.sample_bucket([mp.mpf(6)], entries)
        got, reference = sampler(6), make()(6)
        for m, n in entries:
            bound = mp.mpf("1e-55") * max(1, abs(reference[m][n]))
            sampled = _arb_to_mpf(bucket[0][m][n]) if is_arb else bucket[0][m][n]
            assert abs(sampled - reference[m][n]) < bound
            assert abs(got[m][n] - reference[m][n]) < bound


def test_tower_pole_pool_preserves_exact_shift_and_cache_identity():
    from fractions import Fraction
    from virasoro_blocks.large_spin import _virasoro_chiral_pole_pool

    def roots(c):
        return _virasoro_chiral_pole_pool(h_externals=[HE] * 4, c=c, q_order=4, twist=".25")

    first_c = Fraction(1) + Fraction(1, 10**60)
    second_c = Fraction(1) + Fraction(2, 10**60)
    with mp.workdps(100):
        reference = roots(first_c)
    with mp.workdps(30):
        first, second = roots(first_c), roots(second_c)
        assert first != second
        assert len(first) == len(reference)
        for got, expected in zip(first, reference):
            assert abs(got - expected) < mp.mpf("1e-28")
