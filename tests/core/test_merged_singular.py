"""The singular-c block from one Laurent-in-t table read as merged poles (recursion.build_merged_H_context).

It replaced epsilon-extrapolation in c for the sampled values and the t-series row assembly for the pole scores; both
are kept as references here, and the merged forms must reproduce them to the working precision.  On a Kac weight where
poles collide the value paths return the c -> c0 limit at fixed weight (the irreducible block; the free-boson anchor
below), elsewhere the merged form, which is continuous there."""

import mpmath as mp
import pytest

import virasoro_blocks.recursion as REC
import virasoro_blocks.regularization as R
from virasoro_blocks.polyderivs import G_zzb_poly_table, build_numeric_sampler
from virasoro_blocks.regularization import regularized_numeric_grids

pytestmark = pytest.mark.blocks

F = mp.mpf


@pytest.mark.parametrize('method', ['merged', 'laurent'])
@pytest.mark.parametrize('charge, order', [('44.2', 1), ('1', 4)])
def test_tiny_binary_exchange_is_distinct_from_the_identity_pole(method, charge, order):
    from fractions import Fraction
    from virasoro_blocks.requests import BlockPoleError

    ext = tuple(Fraction(k, 9) for k in range(1, 5))
    with mp.workdps(30):
        weight = mp.mpf(2) ** -400
        evaluate = R.H_q_series_regularized if method == 'merged' else REC.H_q_series_laurent
        kwargs = {'c': Fraction(charge)} if method == 'merged' else {'c0': Fraction(charge)}
        found = evaluate(weight, ext, q_order=order, **kwargs)
        # The level-one residue is exactly 8(h2-h1)(h3-h4)/h.
        assert abs(found[1] + mp.mpf(8) / (81 * weight)) < mp.mpf('1e-28') * abs(found[1])
        with pytest.raises((BlockPoleError, ValueError), match='pole|genuine'):
            evaluate(mp.mpf(0), ext, q_order=order, **kwargs)
        assert mp.mp.dps == 30
    with mp.workdps(240):
        reference = R.H_q_series_regularized(Fraction(1, 2**400), ext, c=Fraction(charge), q_order=order)
        assert all(abs(actual - expected) < mp.mpf('1e-28') * max(1, abs(expected))
                   for actual, expected in zip(found, reference))


@pytest.mark.parametrize('read', ['coefficients', 'verified'])
def test_genuine_kac_pole_is_a_block_pole_error_from_one_build(monkeypatch, read):
    # h = 3 = d_{4,1} at c = 4/5 with non-degenerate externals.  The c -> c0 limit at fixed h: a pole read from
    # valuations is confirmed by one build at doubled precision (a hidden structural zero reads the same way), then
    # raised as BlockPoleError.  The irreducible block (the default) decides it exactly from the fusion factors, with
    # no build at a raised precision.
    from fractions import Fraction
    from virasoro_blocks.requests import BlockPoleError

    precisions = []
    build = REC.H_q_series_laurent_precompute
    monkeypatch.setattr(REC, 'H_q_series_laurent_precompute',
                        lambda *a, **k: precisions.append(mp.mp.prec) or build(*a, **k))
    evaluate = R._merged_H_coefficients if read == 'coefficients' else R._verified_merged
    with mp.workdps(30), REC.kac_limit_context('c_limit'), pytest.raises(BlockPoleError):
        evaluate(3, [Fraction(1, 10)] * 4, c0=Fraction(4, 5), q_order=10)
    assert set(precisions) == {103, 203}    # (30 digits, then doubled once)
    precisions.clear()
    with mp.workdps(30), pytest.raises(BlockPoleError, match='do not decouple'):
        evaluate(3, [Fraction(1, 10)] * 4, c0=Fraction(4, 5), q_order=10)
    assert set(precisions) <= {103}


def _kwargs(ell=0, spinning=False, projected=False, coordinate="zzb", he=("0.21",) * 4, hbe=None, parity=None):
    he = [F(x) for x in he]
    return dict(coordinate=coordinate, chiral=False, spinning=spinning, projected=projected, h_externals=he,
                hbar_externals=he if hbe is None else [F(x) for x in hbe], ell=ell, m_max=3, n_max=3, b=None,
                q_order=12, chiral_norm=None, z_shift=None, zbar_shift=None, center=None, center_bar=None,
                parity=parity, parity_external=None if parity is None else ["1"] * 4, exchanged_parity="1")


def _max_rel(old, new):
    worst = mp.mpf(0)
    for go, gn in zip(old, new):
        scale = max(abs(mp.mpc(v)) for row in go for v in row)
        for ro, rn in zip(go, gn):
            for vo, vn in zip(ro, rn):
                worst = max(worst, abs(mp.mpc(vo) - mp.mpc(vn)) / scale)
    return worst


@pytest.mark.parametrize("c0", ["26", "25", "1", "0.5", "-2"])
@pytest.mark.parametrize("config", [
    dict(),
    dict(ell=2, coordinate="xt"),
    dict(spinning=True, projected=True, ell=1, he=("0.21", "0.33", "0.21", "0.33"),
         hbe=("0.27", "0.19", "0.27", "0.19"), parity="-"),
])
def test_merged_sampler_reproduces_epsilon_extrapolation(c0, config):
    """Every basis at every kind of singular c (c >= 25, c = 1, a minimal model, c = -2), odd-spin parity-projected B
    included: the sampler at c0 itself equals the extrapolation in c it replaced."""
    with mp.workdps(50):
        c = F(c0)
        kw = _kwargs(**config)
        deltas = [F("0.37") + kw["ell"], F("3.71") + kw["ell"], F("9.3") + kw["ell"]]
        old = regularized_numeric_grids(lambda cv: build_numeric_sampler(c=cv, **kw), deltas, c=c, n_points=5)
        new, _ = R._RegularizedNumericSampler(c=c, **kw)(deltas)
        assert _max_rel(old, new) < mp.mpf(10) ** -46


def test_merged_H_matches_the_laurent_H_series_with_residue_cancellation():
    """H_k(h) from the merged poles equals the t^0 of the Laurent recursion, including a degenerate external
    (h = 1/4 at c = 1) whose residues cancel."""
    from virasoro_blocks.polynomial import _acb_to_mpc, _arb_to_mpf

    with mp.workdps(50):
        for c0, ext in ((F(26), "0.45"), (F(1), "0.25")):
            h_ext = [F(ext)] * 4
            ctx = REC.build_merged_H_context(16, c0, h_ext)
            for h in (F("0.7"), F("3.3")):
                ref = REC.H_q_series_laurent(h, h_ext, c0=c0, q_order=16)
                got = REC.eval_merged_H(h, ctx, 16)
                for g, r in zip(got, ref):
                    g = _acb_to_mpc(g) if type(g).__name__ == "acb" else mp.mpc(_arb_to_mpf(g))
                    assert abs(g - r) <= mp.mpf(10) ** -45 * max(1, abs(r))


@pytest.mark.parametrize("c0, ell", [("26", 0), ("28", 3), ("1", 0)])
def test_merged_scorer_reproduces_the_laurent_scorer(c0, ell):
    with mp.workdps(60):
        c = F(c0)
        sc = [F("0.13")] * 4
        skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(ell, sc, 2, 2, c=cc, q_order=12, lambda_cut=None, skeleton=True), c=c)
        kw = dict(_kwargs(ell=ell, he=("0.13",) * 4), m_max=2, n_max=2)
        roots = [mp.mpc(r) for r in skel.denominator_roots]
        entries = [(0, 0), (1, 0), (2, 0), (1, 1)]
        parts = {}
        for method in ("laurent", "merged"):
            s = R._RegularizedNumericSampler(c=c, scorer_method=method, **kw)
            parts[method] = {mp.nstr(mp.re(loc), 12): rows
                             for loc, _idx, rows in s._analytic_principal_parts(roots, entries, skel.exp_base)}
        assert parts["laurent"].keys() == parts["merged"].keys()
        for key, rows_l in parts["laurent"].items():
            rows_m = parts["merged"][key]
            scale = max(abs(mp.mpc(g)) for row in rows_l for g in row)
            for rl, rm in zip(rows_l, rows_m):
                assert len(rl) == len(rm)
                for a, b in zip(rl, rm):
                    assert abs(mp.mpc(a) - mp.mpc(b)) <= mp.mpf(10) ** -50 * scale


@pytest.mark.parametrize("external", ["mpf", "rational", "shifted"])
def test_one_laurent_table_per_block(monkeypatch, external):
    """The scorer, the identity vector and the merged sampler of one singular-c block read ONE H-table build."""
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec, prepare_identical_scalar_gap_build
    import warnings

    calls = []
    orig = REC._laurent_levels
    monkeypatch.setattr(REC, "_laurent_levels", lambda *a, **k: calls.append(1) or orig(*a, **k))
    REC._H_LAURENT_PRECOMP_CACHE.clear()
    with mp.workdps(60), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from virasoro_blocks.numeric_input import literal_number
        h_ext = F("0.9") if external == "mpf" else literal_number(".9")
        if external == "shifted":
            h_ext += literal_number("1e-70")
        spec = IdenticalScalarGapSpec(c=F(26), h_ext=h_ext,
                                      delta_gap=F("1.8"), lam=5, spin_list=(0,), q_order=16, dps=60, npoles=10,
                                      num_threads=1)
        prepare_identical_scalar_gap_build(spec)
    assert len(calls) == 1


_SPIN_HE, _SPIN_HBE = ("1.11", "0.16") * 2, ("0.11", "0.16") * 2


def _skeleton_and_sampler(c, ell, kind, ext, m=2, q_order=12):
    from virasoro_blocks.polyderivs import B_zzb_poly_table

    if kind == "G":
        he = hbe = [F(ext)] * 4
        build = lambda cc: G_zzb_poly_table(ell, he, m, m, c=cc, q_order=q_order, lambda_cut=None, skeleton=True)
        kw = _kwargs(ell=ell, he=(ext,) * 4)
    else:
        he, hbe = [F(x) for x in _SPIN_HE], [F(x) for x in _SPIN_HBE]
        build = lambda cc: B_zzb_poly_table(ell, he, hbe, m, m, c=cc, q_order=q_order, skeleton=True)
        kw = _kwargs(ell=ell, spinning=True, projected=True, he=_SPIN_HE, hbe=_SPIN_HBE, parity=kind)
    kw.update(m_max=m, n_max=m, q_order=q_order)
    return R.merged_delta_skeleton(build, c=c), kw


@pytest.mark.parametrize("c0, ell, kind, ext", [
    ("26", 0, "G", "0.13"), ("0.5", 0, "G", "0.13"), ("0.5", 0, "G", "0.0625"), ("0.7", 0, "G", "0.4375"),
    ("-2", 0, "G", "0.13"), ("1", 0, "G", "0.25"),
    ("26", 1, "-", None), ("26", 1, "+", None), ("1", 3, "-", None), ("0.5", 1, "-", None), ("-2", 1, "-", None),
])
def test_merged_skeleton_orders_are_the_block_growth(c0, ell, kind, ext):
    """Every root of the skeleton built AT c0, at its multiplicity, is a pole of the block of exactly that order: the
    growth of the sampled block at 10^-30 and 10^-40 from it.  Covers the minimal-model chiral double poles (Ising
    sigma), the degenerate c = 1 external whose residues cancel, and odd-spin parity-projected B of both parities,
    where residues cancel between the two orderings (the epsilon-extrapolated skeleton counted some of these poles
    twice, e.g. c = 26 scalar at Delta = -28, order 4 for a double pole)."""
    with mp.workdps(90):
        skel, kw = _skeleton_and_sampler(F(c0), ell, kind, ext)
        assert not skel.requires_pole_strip
        sampler = R._RegularizedNumericSampler(c=F(c0), **kw)
        orders = {}
        for r in skel.denominator_roots:
            r = mp.mpc(r)
            orders.setdefault(mp.nstr(r, 30), [r, 0])[1] += 1
        for r, order in orders.values():
            grids, _ = sampler([mp.re(r) + mp.mpf(10) ** -30, mp.re(r) + mp.mpf(10) ** -40])
            size = [max(abs(mp.mpc(v)) for row in g for v in row) for g in grids]
            assert abs(mp.log10(size[1] / size[0]) / 10 - order) < mp.mpf("0.001"), (mp.nstr(r, 10), order)


@pytest.mark.parametrize("c0, ell", [("1", 3), ("26", 1)])
def test_merged_scorer_keeps_structural_zeros_of_parity_odd_entries(c0, ell):
    """Parity-odd projected B, odd spin: the merged principal parts agree with the t-series (laurent) ones per entry,
    and an entry whose principal part cancels is exactly zero -- not the location's rounding, which the high-order
    c <= 1 poles amplified into the table (c = 1, spin 3: 10 verified digits instead of 35)."""
    with mp.workdps(60):
        c = F(c0)
        skel, kw = _skeleton_and_sampler(c, ell, "-", None, m=3)
        roots = [mp.mpc(r) for r in skel.denominator_roots]
        entries = [(m, n) for m in range(4) for n in range(4) if m + n <= 3]
        parts = {}
        for method in ("laurent", "merged"):
            s = R._RegularizedNumericSampler(c=c, scorer_method=method, **kw)
            parts[method] = {mp.nstr(mp.re(loc), 12): rows
                             for loc, _idx, rows in s._analytic_principal_parts(roots, entries, skel.exp_base)}
        assert parts["laurent"].keys() == parts["merged"].keys()
        zeros = 0
        for key, rows_l in parts["laurent"].items():
            floor = max(abs(mp.mpc(g)) for row in rows_l for g in row) * mp.mpf(10) ** -50
            for rl, rm in zip(rows_l, parts["merged"][key]):
                n = max(len(rl), len(rm))
                for a, b in zip(list(rl) + [0] * (n - len(rl)), list(rm) + [0] * (n - len(rm))):
                    a, b = mp.mpc(a), mp.mpc(b)
                    if abs(a) <= floor:
                        assert b == 0, (key, a, b)
                        zeros += 1
                    else:
                        assert abs(a - b) <= mp.mpf(10) ** -45 * abs(a), (key, a, b)
        assert zeros






def test_free_boson_vertex_coefficient_fixes_the_physical_block():
    """The chiral free-boson correlator <V_a V_-a V_a V_-a> = z^{-a^2} (1 - z)^{-a^2} (all h_ext = a^2/2, c = 1)
    decomposes in the U(1) vacuum channel into the Virasoro primaries at h = n^2.  The level-4 one is
    P = a_{-1}^4 - 2 a_{-1} a_{-3} + 3/2 a_{-2}^2 (norm 54), so its exact coefficient is P(a)^2 / 54 =
    (a^4 - a^2/2)^2 / 54 (1/15000 at a^2 = 3/10).  Peeled with the singular-c blocks (h = 0 and 1 are c = 1 Kac
    weights) it comes out exactly; with the Fock-module blocks it was 25% off."""
    import virasoro_blocks.recursion as REC
    from virasoro_blocks.regularization import H_q_series_regularized

    with mp.workdps(40):
        a2 = F(3) / 10
        he = a2 / 2
        ext = [he] * 4
        exact = (a2 ** 2 - a2 / 2) ** 2 / 54

        def phi(h, N=40, r=F("0.05")):
            H = H_q_series_regularized(F(h), ext, c=1, q_order=24)
            out = [mp.mpc(0)] * 5
            for j in range(N):
                z = r * mp.expjpi(2 * F(j) / N)
                q = REC.q_from_z(z)
                Fv = REC.block_prefactor(F(h), z, ext, c=F(1), chiral_norm=1, z_shift=-2 * he) * sum(
                    Hk * q ** k for k, Hk in enumerate(H))
                u = Fv / z ** (F(h) - 2 * he)
                for k in range(5):
                    out[k] += u / z ** k / N
            return [mp.re(x) for x in out]

        g = mp.taylor(lambda z: (1 - z) ** (-a2), 0, 4)
        p0, p1 = phi(0), phi(1)
        a1 = g[1] - p0[1]
        a4 = g[4] - p0[4] - a1 * p1[3]
        assert abs(a1 - a2) < mp.mpf(10) ** -30
        assert abs(a4 / exact - 1) < mp.mpf(10) ** -25




@pytest.mark.parametrize("c0", ["26", "0.5", "1", "-2"])
def test_merged_full_table_matches_the_merged_sampler(c0):
    """The full symbolic table at a singular c (numerators from the merged rational H, merged_delta_table: the
    --sample-via symbolic and keep-all paths) equals the merged sampler at generic Delta, and has the skeleton's
    poles."""
    with mp.workdps(50):
        c, he = F(c0), [F("0.13")] * 4
        build = lambda cc, sk=False: G_zzb_poly_table(0, he, 2, 2, c=cc, q_order=12, lambda_cut=None, skeleton=sk)
        full = R.merged_delta_table(build, c=c)
        skel = R.merged_delta_skeleton(lambda cc: build(cc, True), c=c)
        assert _root_counts(full) == _root_counts(skel)
        sampler = R._RegularizedNumericSampler(c=c, **dict(_kwargs(he=("0.13",) * 4), m_max=2, n_max=2))
        deltas = [F("0.37"), F("3.3"), F("11.1")]
        grids, _ = sampler(deltas)
        for D, g in zip(deltas, grids):
            t = full.evaluate(D)
            scale = max(abs(mp.mpc(v)) for row in g for v in row)
            assert max(abs(mp.mpc(a) - mp.mpc(b)) for ra, rb in zip(t, g) for a, b in zip(ra, rb)) < mp.mpf(10) ** -40 * scale


def _root_counts(table):
    from collections import Counter

    return Counter((mp.nstr(mp.re(r), 20), mp.nstr(mp.im(r), 8)) for r in table.denominator_roots)


@pytest.mark.parametrize("c0, ext, h_kac", [("26", "0.13", None), ("0.7", "0.6", "0"), ("1", "0.0625", "0.25")])
def test_trimmed_laurent_table_is_exact(c0, ext, h_kac):
    """The H-table fill keeps only the positive t-powers a downstream t^0 reads: the merged columns and the fixed-h
    t^0 coefficients (a Kac weight included, where 1/(h - d_p) = -1/delta_p) equal those of the untrimmed table."""
    import virasoro_blocks.tseries as ts
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_blocks.singularity import exact_locus_c

    with mp.workdps(40):
        c, he = exact_locus_c(F(c0)), [literal_number(ext)] * 4
        saved = ts.window_half()
        ts.set_window(max(saved, REC.laurent_window_half(30, c)))
        try:
            full = REC.H_q_series_laurent_precompute(he, c0=c, q_order=30, trim=False)
            trimmed = REC.H_q_series_laurent_precompute(he, c0=c, q_order=30, trim=True)
            for h in [F("0.37"), F("2.9")] + ([F(h_kac)] if h_kac is not None else []):
                a, b = REC.H_q_series_laurent_eval(h, full), REC.H_q_series_laurent_eval(h, trimmed)
                assert max(abs(mp.mpc(x) - mp.mpc(y)) / max(1, abs(mp.mpc(x))) for x, y in zip(a, b)) < mp.mpf(10) ** -35
        finally:
            ts.set_window(saved)
