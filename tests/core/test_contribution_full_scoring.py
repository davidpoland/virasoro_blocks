"""Full-principal-part contribution scoring (the default ``contribution``).

The legacy ``contribution_leading`` selector ranks a pole of order ``m`` by
``|c_m| / |d|`` (leading Laurent coefficient over distance^1), which is exact for
simple poles but drops every sub-leading coefficient and uses the wrong distance
power for double+ poles. The default ``contribution`` ranks by the magnitude of
the pole's FULL principal part at the evaluation point,
``|Σ_{j<m} a_j d^j| / |d|^m``.
"""

from __future__ import annotations

import mpmath as mp

from virasoro_blocks.regularization import ranking_digits
import pytest

from virasoro_blocks.interpolation import (
    DampedRational,
    POLE_SELECTIONS,
    pole_selection_is_contribution,
    pole_selection_is_full_contribution,
    pole_selection_score_entries,
)
from virasoro_blocks.polyderivs import (
    G_zzb_poly_table,
    _aggregate_contribution_scores,
    _aggregate_residue_scores,
    _contribution_optimal_adaptive_order,
    _contribution_optimal_order,
    _contribution_scores,
    _group_roots_by_location,
    _take_contribution_optimal_prefix,
    build_numeric_sampler,
    interpolate_delta_tables,
)

SC = [mp.mpf("0.13")] * 4


def test_selector_registry():
    # The default `contribution` / `contribution_all` are the full-principal-part
    # selectors; `contribution_leading` / `contribution_leading_all` are the
    # legacy leading-residue/distance variants.
    for sel in ("contribution", "contribution_all",
                "contribution_leading", "contribution_leading_all",
                "contribution_optimal"):
        assert sel in POLE_SELECTIONS
    # full selectors
    assert pole_selection_is_full_contribution("contribution")
    assert pole_selection_is_full_contribution("contribution_all")
    assert pole_selection_is_full_contribution("contribution_optimal")
    assert not pole_selection_is_full_contribution("contribution_leading")
    # the full selectors must NOT also trigger the legacy distance division
    assert not pole_selection_is_contribution("contribution")
    assert not pole_selection_is_contribution("contribution_all")
    # legacy leading selectors use the distance division, not the full PP
    assert pole_selection_is_contribution("contribution_leading")
    assert pole_selection_is_contribution("contribution_leading_all")
    assert not pole_selection_is_full_contribution("contribution_leading_all")
    # value vs all-derivative scoring
    assert pole_selection_score_entries("contribution") == "value"
    assert pole_selection_score_entries("contribution_all") == "kept"
    assert pole_selection_score_entries("contribution_leading") == "value"
    assert pole_selection_score_entries("contribution_leading_all") == "kept"
    assert pole_selection_score_entries("contribution_optimal") == "kept"


def test_contribution_optimal_order_has_target_independent_prefix():
    """Stopping the greedy construction at N must give the same prefix as
    running it farther, which is the defining nested-selector property."""
    mp.mp.dps = 50
    roots = [mp.mpc(-1), mp.mpc(-2), mp.mpc(-3), mp.mpc(-4)]
    scores = [mp.mpf("4"), mp.mpf("3"), mp.mpf("2"), mp.mpf("1")]
    nodes = [mp.mpf("0.2"), mp.mpf("0.8"), mp.mpf("1.6")]
    short = _contribution_optimal_order(
        scores, roots, nodes, (), base=mp.mpf("0.2"), target_slots=2,
    )
    long = _contribution_optimal_order(
        scores, roots, nodes, (), base=mp.mpf("0.2"), target_slots=4,
    )
    assert short == long[:len(short)]


def test_contribution_optimal_prefix_does_not_skip_conjugate_pair():
    """Odd caps may leave a slot unused, but cannot replace an earlier pair
    with a later real pole, which would break nesting."""
    pair = [mp.mpc(-1, 1), mp.mpc(-1, -1)]
    real = mp.mpc(-2)
    full = DampedRational(
        constant=mp.mpc(1), poles=tuple(pair + [real]), base=mp.mpf("0.2"),
    )
    one = _take_contribution_optimal_prefix(full, pair + [real], 1)
    two = _take_contribution_optimal_prefix(full, pair + [real], 2)
    three = _take_contribution_optimal_prefix(full, pair + [real], 3)
    assert len(one.poles) == 0
    assert list(two.poles) == pair
    assert list(three.poles) == pair + [real]


def test_contribution_optimal_adaptive_order_has_nested_prefix():
    """Scale refreshes occur at fixed boundaries, so the order cannot depend
    on the final requested cap."""
    mp.mp.dps = 50
    roots = [mp.mpc(-i) for i in range(1, 13)]
    scores = [mp.mpf(13 - i) for i in range(1, 13)]
    shifted = DampedRational(
        constant=mp.mpc(1), poles=tuple(roots), base=mp.mpf("0.2"),
    )
    short = _contribution_optimal_adaptive_order(
        scores, roots, (), shifted, node_excess=2, base=mp.mpf("0.2"),
        target_slots=6, epoch_size=4,
    )
    long = _contribution_optimal_adaptive_order(
        scores, roots, (), shifted, node_excess=2, base=mp.mpf("0.2"),
        target_slots=12, epoch_size=4,
    )
    assert short == long[:len(short)]


@pytest.mark.parametrize("epoch_size", [2, 8])
def test_contribution_optimal_adaptive_mixed_prefix_is_cap_independent(epoch_size):
    """A partial epoch must give the same pole set as a cached longer order."""
    with mp.workdps(40):
        pair = [mp.mpc(-1, "0.2"), mp.mpc(-1, "-0.2")]
        roots = pair + [mp.mpc(-2)]
        scores = [mp.mpf("1e30"), mp.mpf("1e30"), mp.mpf(1)]
        shifted = DampedRational(
            constant=mp.mpc(1), poles=tuple(roots), base=mp.mpf("0.2"),
        )

        def ranking(cap):
            return _contribution_optimal_adaptive_order(
                scores, roots, (), shifted, node_excess=2,
                base=shifted.base, target_slots=cap, epoch_size=epoch_size,
            )

        full = ranking(3)
        assert full == roots
        for cap, expected in [(1, []), (2, pair), (3, roots)]:
            fresh = ranking(cap)
            assert fresh == full[:len(fresh)]
            fresh_poles = _take_contribution_optimal_prefix(shifted, fresh, cap).poles
            cached_poles = _take_contribution_optimal_prefix(shifted, full, cap).poles
            assert list(fresh_poles) == list(cached_poles) == expected


def test_contribution_optimal_epoch_fills_before_deferring_pair():
    """A fixed epoch boundary may defer a pair for a real pole without making
    the order depend on the caller's requested cap."""
    mp.mp.dps = 50
    roots = [
        mp.mpc(-1),
        mp.mpc(-2, 1), mp.mpc(-2, -1),
        mp.mpc(-5),
    ]
    scores = [mp.mpf("100"), mp.mpf("90"), mp.mpf("90"), mp.mpf("1")]
    nodes = [mp.mpf("0.2"), mp.mpf("0.8"), mp.mpf("1.6")]
    order = _contribution_optimal_order(
        scores, roots, nodes, (), base=mp.mpf("0.2"),
        target_slots=2, exact_slots=True,
    )
    assert len(order) == 2
    assert all(abs(mp.im(p)) == 0 for p in order)


def _degenerate_greedy_cases():
    """Inputs the float screen of the flint greedy must hand to flint: coincident and repeated poles, poles on a node,
    on a probe and on the evaluation point, kept poles among the candidates, zero / infinite / huge scores, ties."""
    nodes = [mp.mpf("0.25"), mp.mpf("0.75"), mp.mpf("1.5")]      # dyadic: the probes 1.5 (2j+1) / 128 are exact
    rng = __import__("random").Random(7)
    generic_roots, generic_scores = [], []
    for _ in range(18):
        re, im = -rng.uniform(0.05, 30), rng.uniform(0, 12)
        if rng.random() < 0.5:
            generic_roots.append(mp.mpc(re))
            generic_scores.append(mp.mpf(10) ** rng.uniform(-60, 60))
        else:
            generic_roots += [mp.mpc(re, im), mp.mpc(re, -im)]
            generic_scores += [mp.mpf(10) ** rng.uniform(-60, 60)] * 2
    yield "generic", generic_scores, generic_roots, nodes, ()
    degenerate = [mp.mpc(-1), mp.mpc(-1), mp.mpc(-1, 2), mp.mpc(-1, -2), mp.mpc("0.25"), mp.mpc("0.01171875"),
                  mp.mpc(-3), mp.mpc(-3) + mp.mpf(10) ** (-mp.mp.dps + 5), mp.mpc(0), mp.mpc(-7), mp.mpc(-9)]
    scores = [mp.mpf(5), mp.mpf(5), mp.mpf(3), mp.mpf(3), mp.mpf(2), mp.mpf(2), mp.mpf(1), mp.mpf(1),
              mp.mpf(4), mp.mpf(0), mp.inf]
    yield "degenerate", scores, degenerate, nodes, ()
    yield "degenerate+forced", scores, degenerate, nodes, (mp.mpc(-1), mp.mpc(-7), mp.mpc("0.75"))
    tied = [mp.mpc(-2), mp.mpc(-2), mp.mpc(-2, 1), mp.mpc(-2, -1), mp.mpc(-2, 1), mp.mpc(-2, -1)]
    yield "ties", [mp.mpf(1)] * len(tied), tied, nodes, ()


@pytest.mark.parametrize("case", [c[0] for c in _degenerate_greedy_cases()])
def test_contribution_optimal_order_flint_matches_mpmath(case):
    """The flint greedy (float-screened, flint-priced) ranks exactly as the independent pure-mpmath form, including
    the degenerate inputs the screen must leave to flint."""
    from virasoro_blocks.polyderivs import _contribution_optimal_order_mpmath

    mp.mp.dps = 60
    _name, scores, roots, nodes, forced = next(c for c in _degenerate_greedy_cases() if c[0] == case)
    for base in (mp.mpf("0.2"), mp.mpf("0.9")):
        for target, exact in ((None, False), (5, True), (8, False)):
            kwargs = dict(base=base, target_slots=target, exact_slots=exact)
            flint_order = _contribution_optimal_order(scores, roots, nodes, forced, **kwargs)
            mpmath_order = _contribution_optimal_order_mpmath(scores, roots, nodes, forced, **kwargs)
            assert flint_order == mpmath_order, (case, base, target)


def test_simple_poles_match_legacy_contribution():
    """ℓ ≠ 0 generic c has only simple poles, where the full principal part
    reduces to the leading residue: contribution must equal the legacy
    contribution score (residue / distance) bit-for-bit."""
    mp.mp.dps = 40
    t = G_zzb_poly_table(2, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None)
    roots = [mp.mpc(r) for r in t.denominator_roots]
    # all simple
    assert all(len(idx) == 1 for _, idx in _group_roots_by_location(roots, mp.mpf(10) ** -30))
    deval = mp.mpf(2)  # unitarity threshold = |ell|
    cfull = _aggregate_contribution_scores([t], delta_eval=deval, score_entries="value")
    res = _aggregate_residue_scores([t], score_entries="value", pole_ranking="symbolic")
    for i, p in enumerate(roots):
        legacy = res[i] / abs(p - deval)
        assert abs(cfull[i] - legacy) <= mp.mpf(10) ** -30 * max(abs(legacy), mp.mpf(1))


def test_double_poles_match_independent_principal_part():
    """At ℓ = 0 (double poles) the flint contribution score must equal an
    independent mpmath partial-fraction computation of the same quantity, and the
    principal parts must reconstruct the block: block = Σ_p PP_p + polynomial."""
    mp.mp.dps = 40
    t = G_zzb_poly_table(0, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None)
    num = t.numerators[0][0]
    den = t.denominator
    lead = den.coeffs[-1]
    roots = [mp.mpc(r) for r in t.denominator_roots]
    groups = _group_roots_by_location(roots, mp.mpf(10) ** -30)
    assert any(len(idx) == 2 for _, idx in groups)  # genuine double poles present

    def principal_part_at(loc, idxs, Dt):
        m = len(idxs)
        others = [roots[k] for k in range(len(roots)) if k not in set(idxs)]

        def cof(D):
            v = lead
            for q in others:
                v *= D - q
            return v

        a = mp.taylor(lambda D: num.evaluate(D) / cof(D), loc, m - 1)
        d = Dt - loc
        return sum(a[j] * d ** j for j in range(m)) / d ** m

    # (a) reconstruction: block(Dt) = Σ_p PP_p(Dt) + constant (deg num == deg den)
    c_const = num.coeffs[-1] / den.coeffs[-1]
    for Dt in [mp.mpc("5.3"), mp.mpc("-2.1"), mp.mpc("3.0", "1.5")]:
        recon = sum(principal_part_at(loc, idx, Dt) for loc, idx in groups) + c_const
        exact = num.evaluate(Dt) / den.evaluate(Dt)
        assert abs(recon - exact) <= mp.mpf(10) ** -30

    # (b) flint contribution == |PP_p(Δ_eval)|, the validated quantity
    deval = mp.mpf(0)
    flint_scores = _contribution_scores(t, delta_eval=deval, score_entries="value")
    for loc, idx in groups:
        ref = abs(principal_part_at(loc, idx, deval))
        assert abs(flint_scores[idx[0]] - ref) <= mp.mpf(10) ** -30 * max(ref, mp.mpf(1))


def test_pole_at_delta_eval_scores_infinite_and_is_kept():
    """Both scorers assign infinite score to a genuine pole at delta_eval.

    With non-identical externals and ell=0, the pole at the Delta=0 threshold
    must be retained ahead of every finite contribution.
    """
    from virasoro_blocks.polyderivs import _numeric_contribution_scores
    mp.mp.dps = 50
    ext = [mp.mpf("0.10"), mp.mpf("0.22"), mp.mpf("0.15"), mp.mpf("0.27")]
    c = mp.mpf("3.7")
    t = G_zzb_poly_table(0, ext, 2, 2, c=c, q_order=8, lambda_cut=None)
    roots = [mp.mpc(r) for r in t.denominator_roots]
    at0 = [i for i, r in enumerate(roots) if abs(r) < mp.mpf(10) ** -20]
    assert at0, "test setup: expected a genuine pole at the Δ=0 threshold"
    deval = mp.mpf(0)
    sym = _contribution_scores(t, delta_eval=deval, score_entries="value")
    assert all(sym[i] == mp.inf for i in at0)
    t.numeric_sampler = build_numeric_sampler(
        c=c, **{**_scalar_sampler(c, 0), "h_externals": ext, "hbar_externals": ext}
    )
    num = _numeric_contribution_scores(t, delta_eval=deval, score_entries="value")
    assert all(num[i] == mp.inf for i in at0)


def test_contribution_low_npoles_no_divergence_nonidentical():
    """End-to-end: with the on-pole pole kept, contribution no longer
    diverges at low npoles for non-identical externals (was rel err ~36 at
    np4 when the Δ=0 threshold pole was dropped)."""
    mp.mp.dps = 50
    ext = [mp.mpf("0.10"), mp.mpf("0.22"), mp.mpf("0.15"), mp.mpf("0.27")]
    c = mp.mpf("3.7")
    skw = {**_scalar_sampler(c, 0), "h_externals": ext, "hbar_externals": ext}
    exact = G_zzb_poly_table(0, ext, 2, 2, c=c, q_order=8, lambda_cut=None)
    e = G_zzb_poly_table(0, ext, 2, 2, c=c, q_order=8, lambda_cut=None)
    e.numeric_sampler = build_numeric_sampler(c=c, **skw)
    it = interpolate_delta_tables(
        {"zzb": e}, npoles=4, pole_selection="contribution"
    )["zzb"]
    for D in [mp.mpf("1.0"), mp.mpf("2.5"), mp.mpf("4.0")]:
        got, ref = it.evaluate(D), exact.evaluate(D)
        err = max(
            abs(mp.mpc(got[m][n]) - mp.mpc(ref[m][n])) / max(abs(mp.mpc(ref[m][n])), mp.mpf("1e-40"))
            for m in range(3) for n in range(3) if got[m][n] is not None
        )
        assert err < mp.mpf("1e-3"), f"Δ={D}: rel err {mp.nstr(err,3)}"


def test_vanishing_value_entry_falls_back_to_first_nonzero():
    """When the undifferentiated (0,0) entry vanishes identically -- parity='-'
    projected_B on an UNFOLDED coordinate, whose diagonal block is antisymmetric
    (never an odd-ell scalar_G, which is z <-> zbar even) -- 'value' scoring
    falls back to the first non-zero entry. The symbolic path (numerators) and
    the numeric/skeleton path (sampler probe) must resolve to the SAME entry,
    and the resulting scores are non-degenerate and numeric == symbolic."""
    from virasoro_blocks.polyderivs import (
        _value_score_entry, _numeric_contribution_scores,
        parity_projected_B_zzb_poly_table,
    )
    mp.mp.dps = 40
    he = [mp.mpf("0.1")] * 4
    c = mp.mpf("4")
    scalar = G_zzb_poly_table(3, he, 2, 2, c=c, q_order=8, lambda_cut=None)  # odd ell: z <-> zbar even
    assert any(mp.mpc(x) != 0 for x in scalar.numerators[0][0].coeffs)
    assert _value_score_entry(scalar) == (0, 0)
    full = parity_projected_B_zzb_poly_table(3, he, he, 2, 2, c=c, q_order=8, parity="-", lambda_cut=None)
    assert all(mp.mpc(x) == 0 for x in full.numerators[0][0].coeffs)  # (0,0) vanishes
    # symbolic (numerator-based) resolution
    assert _value_score_entry(full) == (0, 1)
    # numeric/skeleton (sampler-probe) resolution -> same entry
    skw = dict(coordinate="zzb", chiral=False, spinning=True, projected=True,
               h_externals=he, hbar_externals=he, ell=3, m_max=2, n_max=2, b=None,
               q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None,
               center=None, center_bar=None, parity="-", parity_external=None,
               exchanged_parity="1")
    skel = parity_projected_B_zzb_poly_table(3, he, he, 2, 2, c=c, q_order=8, parity="-", lambda_cut=None,
                                             skeleton=True)
    skel.numeric_sampler = build_numeric_sampler(c=c, **skw)
    assert skel.numerators is None
    assert _value_score_entry(skel) == (0, 1)
    # non-degenerate scores (not all equal) and numeric == symbolic
    full.numeric_sampler = build_numeric_sampler(c=c, **skw)
    sym = _contribution_scores(full, delta_eval=mp.mpf(3), score_entries="value")
    num = _numeric_contribution_scores(full, delta_eval=mp.mpf(3), score_entries="value")
    assert len(set(mp.nstr(s, 8) for s in sym)) > 1  # not degenerate
    worst = max(abs(num[i] - sym[i]) / max(abs(sym[i]), mp.mpf(1)) for i in range(len(sym)))
    assert worst < mp.mpf("1e-6")


def test_value_score_entry_structural_fast_paths_cover_folded_offdiagonal_and_projected():
    from virasoro_blocks.polyderivs import (
        G_qqb_poly_table,
        G_ws_poly_table,
        _value_score_entry,
        parity_projected_B_ws_poly_table,
        parity_projected_B_zzb_poly_table,
    )

    mp.mp.dps = 40
    he = [mp.mpf("0.1")] * 4
    c = mp.mpf("4")

    # Odd scalar on a folded coordinate is z <-> zbar even, so the value entry
    # is structurally (0,0).
    folded_odd = G_ws_poly_table(3, he, 2, 2, c=c, q_order=8, skeleton=True)
    assert _value_score_entry(folded_odd) == (0, 0)

    # Off-diagonal z != zbar: odd scalar G keeps a genuine value entry at (0,0).
    offdiag_odd = G_qqb_poly_table(
        3, he, 2, 2, c=c, q_order=8,
        center=mp.mpf("0.37"), center_bar=mp.mpf("0.55"),
        lambda_cut=None, skeleton=True,
    )
    assert _value_score_entry(offdiag_odd) == (0, 0)

    # Unfolded odd-parity projected B has a first-order structural zero on the
    # diagonal, so it falls back to the first kept derivative.
    proj_unfolded = parity_projected_B_zzb_poly_table(
        0, he, he, 2, 2, c=c, q_order=8, parity="-",
        lambda_cut=None, skeleton=True,
    )
    assert _value_score_entry(proj_unfolded) == (0, 1)

    # Folded projected B strips the same odd-parity factor, so (0,0) is live.
    proj_folded = parity_projected_B_ws_poly_table(
        0, he, he, 2, 2, c=c, q_order=8, parity="-", skeleton=True,
    )
    assert _value_score_entry(proj_folded) == (0, 0)


def test_value_score_entry_sampler_fallback_without_metadata():
    from virasoro_blocks.polyderivs import (
        G_zzb_poly_table,
        _value_score_entry,
        parity_projected_B_zzb_poly_table,
    )

    mp.mp.dps = 40
    he = [mp.mpf("0.1")] * 4
    c = mp.mpf("4")

    even = G_zzb_poly_table(0, he, 2, 2, c=c, q_order=8, lambda_cut=None, skeleton=True)
    even.numeric_sampler = build_numeric_sampler(
        coordinate="zzb", chiral=False, spinning=False, projected=False,
        h_externals=he, hbar_externals=he, ell=0, m_max=2, n_max=2, b=None,
        c=c, q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None,
        center=None, center_bar=None, parity=None, parity_external=None,
        exchanged_parity="1",
    )
    even.oracle_metadata = None
    assert _value_score_entry(even) == (0, 0)

    odd_proj = parity_projected_B_zzb_poly_table(
        0, he, he, 2, 2, c=c, q_order=8, parity="-",
        lambda_cut=None, skeleton=True,
    )
    odd_proj.numeric_sampler = build_numeric_sampler(
        coordinate="zzb", chiral=False, spinning=True, projected=True,
        h_externals=he, hbar_externals=he, ell=0, m_max=2, n_max=2, b=None,
        c=c, q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None,
        center=None, center_bar=None, parity="-", parity_external=None,
        exchanged_parity="1",
    )
    odd_proj.oracle_metadata = None
    assert _value_score_entry(odd_proj) == (0, 1)


def test_value_score_entry_generalizes_to_global_and_large_spin_samplers():
    from virasoro_blocks.global_blocks import _GlobalBlockSampler2d, _make_residue_scorer
    from virasoro_blocks.large_spin import _LargeSpinTwistSampler
    from virasoro_blocks.polyderivs import _value_score_entry_from_sampler

    mp.mp.dps = 40
    he = [mp.mpf("0.1")] * 4

    global_odd = _GlobalBlockSampler2d(
        "zzb", 3, he, he, mp.mpf(-1), 4, None, None, 2, 2,
        basis="global_parity_projected_B", parity="-",
    )
    assert _value_score_entry_from_sampler(global_odd, n_max=2) == (0, 1)

    # an odd-spin scalar G is z <-> zbar even: its (0, 0) value is live
    global_odd_spin = _GlobalBlockSampler2d(
        "zzb", 3, he, he, mp.mpf(1), 4, None, None, 2, 2,
        basis="global_scalar_G",
    )
    assert _value_score_entry_from_sampler(global_odd_spin, n_max=2) == (0, 0)

    global_even = _GlobalBlockSampler2d(
        "zzb", 2, he, he, mp.mpf(1), 4, None, None, 2, 2,
        basis="global_scalar_G",
    )
    assert _value_score_entry_from_sampler(global_even, n_max=2) == (0, 0)

    scorer = _make_residue_scorer(global_odd, mp.mpf("0.2"), 2, 2, score_entries="value")
    pool = [mp.mpf("2"), mp.mpf("4")]
    assert any(score > 0 for score in scorer(pool))

    large_spin_scalar = _LargeSpinTwistSampler(
        "global", "zzb", he, 4, mp.mpf(0), 2, 2, mp.mpf("0.5"),
        basis="scalar_G",
    )
    assert _value_score_entry_from_sampler(large_spin_scalar, n_max=2) == (0, 0)

    large_spin_proj = _LargeSpinTwistSampler(
        "global", "zzb", he, 4, mp.mpf(0), 2, 2, mp.mpf("0.5"),
        basis="parity_projected_B", parity_projection="-",
    )
    assert _value_score_entry_from_sampler(large_spin_proj, n_max=2) == (0, 1)


def test_build_table_from_cache_stamps_oracle_metadata_for_value_rule():
    from virasoro_blocks.polyderivs import (
        _value_score_entry,
        build_chiral_cache,
        build_table_from_cache,
    )

    mp.mp.dps = 40
    he = [mp.mpf("0.1")] * 4
    c = mp.mpf("4")
    cache = build_chiral_cache(
        ("zzb", "ws"), he, lambda_=5, c=c, q_order=8, skeleton=True, truncated=True
    )

    odd_unfolded = build_table_from_cache(
        "zzb", cache["zzb"][0], cache["zzb"][1], 3,
        m_max=2, n_max=2, basis="scalar_G",
    )
    assert odd_unfolded.oracle_metadata["basis"] == "scalar_G"
    assert odd_unfolded.oracle_metadata["folded"] is False
    assert _value_score_entry(odd_unfolded) == (0, 0)   # z <-> zbar even at odd spin

    odd_folded = build_table_from_cache(
        "ws", cache["ws"][0], cache["ws"][1], 3,
        m_max=2, n_max=2, basis="scalar_G",
    )
    assert odd_folded.oracle_metadata["basis"] == "scalar_G"
    assert odd_folded.oracle_metadata["folded"] is True
    assert _value_score_entry(odd_folded) == (0, 0)


def test_residue_value_entry_falls_back_to_first_nonzero():
    """`residue` value scoring uses the SAME value entry as `contribution` (they
    differ only by the distance weight), so when the undifferentiated (0,0) entry
    vanishes identically (parity '-' projected_B on an unfolded coordinate) the
    residue scorer must also fall back to the first non-zero entry instead of
    scoring on the all-zero (0,0). Symbolic and numeric residue scores agree and
    are non-degenerate."""
    from virasoro_blocks.polyderivs import _residue_scores, parity_projected_B_zzb_poly_table
    mp.mp.dps = 40
    he = [mp.mpf("0.1")] * 4
    c = mp.mpf("10")
    full = parity_projected_B_zzb_poly_table(3, he, he, 2, 2, c=c, q_order=8, parity="-", lambda_cut=None)
    assert all(mp.mpc(x) == 0 for x in full.numerators[0][0].coeffs)  # (0,0) vanishes

    skw = dict(coordinate="zzb", chiral=False, spinning=True, projected=True,
               h_externals=he, hbar_externals=he, ell=3, m_max=2, n_max=2, b=None,
               q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None,
               center=None, center_bar=None, parity="-", parity_external=None,
               exchanged_parity="1")
    full.numeric_sampler = build_numeric_sampler(c=c, **skw)

    sym = _residue_scores(full, score_entries="value", pole_ranking="symbolic")
    num = _residue_scores(full, score_entries="value", pole_ranking="numeric")
    # The antisymmetric value entry must give nonzero, distinct scores.
    assert any(abs(mp.mpc(s)) > mp.mpf("1e-30") for s in sym)
    assert len(set(mp.nstr(abs(mp.mpc(s)), 8) for s in sym)) > 1
    # numeric residue ranking matches the symbolic one
    worst = max(
        abs(abs(mp.mpc(num[i])) - abs(mp.mpc(sym[i]))) / max(abs(mp.mpc(sym[i])), mp.mpf(1))
        for i in range(len(sym))
    )
    assert worst < mp.mpf("1e-6")


def test_double_pole_score_differs_from_legacy():
    """The whole point: at a double pole the corrected score genuinely differs
    from the leading-only legacy score (so the selection can change)."""
    mp.mp.dps = 40
    t = G_zzb_poly_table(0, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None)
    roots = [mp.mpc(r) for r in t.denominator_roots]
    deval = mp.mpf(0)
    cfull = _aggregate_contribution_scores([t], delta_eval=deval, score_entries="value")
    res = _aggregate_residue_scores([t], score_entries="value", pole_ranking="symbolic")
    legacy = [res[i] / abs(roots[i] - deval) for i in range(len(roots))]
    worst = max(
        abs(cfull[i] - legacy[i]) / max(abs(legacy[i]), mp.mpf(1))
        for i in range(len(roots))
    )
    assert worst > mp.mpf("1e-4")


def test_end_to_end_interpolation_with_contribution():
    """A reduced interpolation using contribution reproduces the block."""
    mp.mp.dps = 40
    t = G_zzb_poly_table(0, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None)
    skw = dict(
        coordinate="zzb", chiral=False, spinning=False, projected=False,
        h_externals=SC, hbar_externals=SC, ell=0, m_max=2, n_max=2, b=None,
        q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None, center=None,
        center_bar=None, parity=None, parity_external=None, exchanged_parity="1",
    )
    t.numeric_sampler = build_numeric_sampler(c=mp.mpf("3.7"), **skw)
    it = interpolate_delta_tables(
        {"zzb": t}, npoles=8, pole_selection="contribution"
    )["zzb"]
    samp = build_numeric_sampler(c=mp.mpf("3.7"), **skw)
    for D in [mp.mpf("1.0"), mp.mpf("2.5"), mp.mpf("4.0")]:
        got = mp.mpc(it.evaluate(D)[0][0])
        ref = mp.mpc(samp([D])[0][0][0][0])
        assert abs(got - ref) <= mp.mpf("1e-10") * max(abs(ref), mp.mpf(1))


def test_contribution_scores_requires_full_numerators():
    """The symbolic Phase-1 scorer must reject a degree-only skeleton table."""
    mp.mp.dps = 40
    skel = G_zzb_poly_table(
        0, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None, skeleton=True
    )
    with pytest.raises(ValueError, match="full numerator"):
        _contribution_scores(skel, delta_eval=mp.mpf(0), score_entries="value")


# --- Numeric (sampler-based) principal-part extraction --------------


def _scalar_sampler(c, ell):
    return dict(
        coordinate="zzb", chiral=False, spinning=False, projected=False,
        h_externals=SC, hbar_externals=SC, ell=ell, m_max=2, n_max=2, b=None,
        q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None, center=None,
        center_bar=None, parity=None, parity_external=None, exchanged_parity="1",
    )


@pytest.mark.parametrize("ell", [0, 2])
@pytest.mark.parametrize("score_entries", ["value", "kept"])
def test_numeric_matches_symbolic_contribution(ell, score_entries):
    """The numeric contour/DFT principal-part scorer reproduces the symbolic
    full-principal-part scores. At the default (ranking-grade) ``n_extra`` the
    DFT aliasing floor is ~(radius_fraction)^(m+n_extra) ~ 1e-7..1e-9 -- far
    below any pole-ranking gap; the high-accuracy convergence is checked
    separately below."""
    from virasoro_blocks.polyderivs import _numeric_contribution_scores
    mp.mp.dps = 40
    t = G_zzb_poly_table(ell, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None)
    t.numeric_sampler = build_numeric_sampler(c=mp.mpf("3.7"), **_scalar_sampler(mp.mpf("3.7"), ell))
    deval = mp.mpf(ell)
    sym = _contribution_scores(t, delta_eval=deval, score_entries=score_entries)
    num = _numeric_contribution_scores(t, delta_eval=deval, score_entries=score_entries)
    worst = max(
        abs(num[i] - sym[i]) / max(abs(sym[i]), mp.mpf(1)) for i in range(len(sym))
    )
    assert worst < mp.mpf("1e-6")


@pytest.mark.parametrize("ell", [0, 2, 3])
def test_analytic_scores_match_symbolic_with_conjugate_mirrors(ell):
    """1 < c < 25: the analytic scorer (principal parts from chiral factors, scored as one rank-4 flint product, the
    conjugate location of each pair taken as its partner's mirror) reproduces the exact symbolic scores, and each
    mirror's rows are the exact conjugates of its partner's."""
    from virasoro_blocks.polyderivs import _OrderTwoPrincipalRows, _residue_score_indices_from_dims

    mp.mp.dps = 40
    c = mp.mpf("3.7")
    t = G_zzb_poly_table(ell, SC, 2, 2, c=c, q_order=8, lambda_cut=None)
    sampler = build_numeric_sampler(c=c, **_scalar_sampler(c, ell))
    roots = [mp.mpc(r) for r in t.denominator_roots]
    entries = _residue_score_indices_from_dims(2, 2, "kept")
    analytic = sampler._analytic_contribution_scores(roots, mp.mpf(ell), entries, t.exp_base)
    symbolic = _contribution_scores(t, delta_eval=mp.mpf(ell), score_entries="kept")
    worst = max(abs(a - s) / max(abs(s), mp.mpf(10) ** -30) for a, s in zip(analytic, symbolic))
    assert worst < mp.mpf("1e-25")

    parts = sampler._analytic_principal_parts(roots, entries, t.exp_base)
    mirrors = [(loc, rows) for loc, _idxs, rows in parts if getattr(rows, "mirror_of", None) is not None]
    assert mirrors, "no conjugate location was taken as a mirror"
    by_rows = {id(rows): loc for loc, _idxs, rows in parts}
    for loc, rows in mirrors:
        assert isinstance(rows, _OrderTwoPrincipalRows)
        assert mp.almosteq(mp.conj(by_rows[id(rows.mirror_of)]), loc, 1e-30)
        for (g1, g2), (p1, p2) in zip(rows, rows.mirror_of):
            for g, p in ((g1, p1), (g2, p2)):
                assert abs(g - p.conjugate()) <= 1e-30 * (abs(p) + 1)


def test_chiral_factor_reflection_across_roundoff():
    """A real-analytic chiral factor serves conj(x) -- exactly, or within roundoff -- as the conjugate of the data
    at x, and the reflection of a reflection as the original data (no nesting)."""
    from virasoro_blocks import derivs
    from virasoro_blocks.polyderivs import _ReflectedFactor, _analytic_chiral_factor

    mp.mp.dps = 40
    evaluator = derivs.chiral_z_series_evaluator(tuple(SC), 4, c=mp.mpf("3.7"), q_order=8)
    assert evaluator.real_analytic
    fac = _analytic_chiral_factor(evaluator, 4, deriv_factor=2)
    x1 = mp.mpc("0.3", "0.7")
    ulp = mp.mpf(10) ** (-mp.mp.dps + 2)
    first = fac(x1)
    near_twin = fac(mp.conj(x1) + ulp)            # the conjugate argument, off by roundoff
    assert isinstance(near_twin, _ReflectedFactor) and near_twin.source is first
    back = fac(mp.conj(mp.conj(x1) + ulp))        # its own conjugate: the original data, not a nested reflection
    assert back is first
    direct = evaluator(mp.conj(x1), return_acb=True)
    for got, want in zip(near_twin[1], direct):
        assert abs(got - want) <= mp.mpf(10) ** -30 * (abs(want) + 1)


@pytest.mark.parametrize("coordinate", ["xt", "ws"])
def test_folded_analytic_scores_match_their_rows(coordinate):
    """Folded coordinates: the score folded from the rank-4 product equals the mpmath score of the folded rows."""
    from virasoro_blocks.polyderivs import _scores_from_principal_parts

    mp.mp.dps = 40
    c, ell = mp.mpf("3.7"), 2
    kw = dict(_scalar_sampler(c, ell), coordinate=coordinate, m_max=4, n_max=2, q_order=10)
    sampler = build_numeric_sampler(c=c, **kw)
    t = G_zzb_poly_table(ell, SC, 2, 2, c=c, q_order=10, lambda_cut=None, skeleton=True)
    roots = [mp.mpc(r) for r in t.denominator_roots]
    entries = [(a, b) for a in range(5) for b in range(3)]
    parts = sampler._analytic_principal_parts(roots, entries, t.exp_base)
    assert any(getattr(rows, "mirror_of", None) is not None for _loc, _idxs, rows in parts)
    fast = _scores_from_principal_parts(parts, len(roots), mp.mpf(ell))
    for loc, idxs, rows in parts:
        D = mp.mpf(ell) - loc
        slow = max(abs(mp.fsum(mp.mpc(g) / D ** (a + 1) for a, g in enumerate(row))) for row in rows)
        assert abs(fast[idxs[0]] - slow) <= mp.mpf("1e-35") * slow


@pytest.mark.parametrize("ell", [0, 2])
def test_numeric_converges_to_symbolic_at_high_n_extra(ell):
    """Raising ``n_extra`` drives the contour/DFT extraction to the symbolic
    score: the method is exact, the default just trades aliasing floor for
    speed."""
    from virasoro_blocks.polyderivs import _numeric_contribution_scores
    mp.mp.dps = 40
    t = G_zzb_poly_table(ell, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None)
    t.numeric_sampler = build_numeric_sampler(c=mp.mpf("3.7"), **_scalar_sampler(mp.mpf("3.7"), ell))
    deval = mp.mpf(ell)
    sym = _contribution_scores(t, delta_eval=deval, score_entries="value")
    num = _numeric_contribution_scores(t, delta_eval=deval, score_entries="value", n_extra=14)
    worst = max(
        abs(num[i] - sym[i]) / max(abs(sym[i]), mp.mpf(1)) for i in range(len(sym))
    )
    assert worst < mp.mpf("1e-12")


@pytest.mark.parametrize("ell", [0, 2])
@pytest.mark.parametrize("c", ["10"])
def test_prefilter_selection_matches_full_scoring(ell, c):
    """The coarse->fine pre-filter (n_select set) must select exactly the same
    top-k poles as the full single-pass scorer (n_select=None) for every cap up
    to n_select. The pre-filter only skips full-fidelity scoring on poles that
    can't enter the kept set, so the ranking of the top poles is unchanged.

    c=10 is the regression guard for the far-complex-pole trap:
    poles far from delta_eval in the complex plane carry a large internal
    cancellation in their principal part, so any coarse proxy that is loose or
    poorly conditioned there (e.g. a small-M or upper-bound coarse pass)
    silently over-ranks them and corrupts the kept set."""
    from virasoro_blocks.polyderivs import (
        principal_part_contribution_scores, _residue_score_indices_from_dims,
    )
    mp.mp.dps = 50
    c = mp.mpf(c)
    npoles = 10
    lam = 6
    t = G_zzb_poly_table(ell, SC, lam, lam, c=c, q_order=16, lambda_cut=None)
    t.numeric_sampler = build_numeric_sampler(
        c=c, **{**_scalar_sampler(c, ell), "m_max": lam, "n_max": lam, "q_order": 16}
    )
    roots = [mp.mpc(r) for r in t.denominator_roots]
    entries = _residue_score_indices_from_dims(t.m_max, t.n_max, "kept")
    ev = lambda pts, e: t.numeric_sampler(pts, entries=e)
    kw = dict(delta_eval=mp.mpf(ell), entries=entries, exp_base=t.exp_base)
    full = principal_part_contribution_scores(roots, ev, **kw, n_select=None)
    pre = principal_part_contribution_scores(roots, ev, **kw, n_select=npoles)

    def order(sc):
        return sorted(range(len(sc)), key=lambda i: (sc[i] != mp.inf, sc[i]), reverse=True)

    of, op = order(full), order(pre)
    for cap in range(1, npoles + 1):
        assert set(of[:cap]) == set(op[:cap]), f"cap={cap}: pre-filter changed selection"


def test_numeric_contribution_on_skeleton_no_numerators():
    """contribution works on a degree-only skeleton (no symbolic
    numerators) via the numeric sampler -- the Phase-2 point."""
    from virasoro_blocks.polyderivs import _aggregate_contribution_scores
    mp.mp.dps = 40
    skel = G_zzb_poly_table(0, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None, skeleton=True)
    assert skel.numerators is None
    skel.numeric_sampler = build_numeric_sampler(c=mp.mpf("3.7"), **_scalar_sampler(mp.mpf("3.7"), 0))
    full = G_zzb_poly_table(0, SC, 2, 2, c=mp.mpf("3.7"), q_order=8, lambda_cut=None)
    num = _aggregate_contribution_scores(
        [skel], delta_eval=mp.mpf(0), score_entries="value", pole_ranking="numeric"
    )
    sym = _aggregate_contribution_scores(
        [full], delta_eval=mp.mpf(0), score_entries="value", pole_ranking="symbolic"
    )
    worst = max(abs(num[i] - sym[i]) / max(abs(sym[i]), mp.mpf(1)) for i in range(len(sym)))
    assert worst < mp.mpf("1e-6")


def test_global_contribution_double_poles():
    """The global ℓ=0 block has double poles; the full `contribution` runs there
    (the global 2d sampler now evaluates the contour's complex points), and it
    selects the same pole set as the legacy `contribution_leading` (the tower is
    ordered identically by both scorers), so the reduced interpolants coincide."""
    from virasoro_blocks.global_blocks import global_scalar_G_interpolated_table
    mp.mp.dps = 30
    he = [mp.mpf("0.2")] * 4
    # reduced selection: just needs to run and give a finite block
    red = global_scalar_G_interpolated_table(
        "zzb", 0, he, 3, 3, npoles=6, pole_selection="contribution"
    )
    v = mp.mpc(red.evaluate(mp.mpf("1.5"))[0][0])
    assert mp.isfinite(v.real) and mp.isfinite(v.imag)
    # full `contribution` and legacy `contribution_leading` keep the SAME pole
    # set here, so the reduced interpolants coincide. npoles sets the
    # interpolation degree / node count, so keep it modest -- a large value makes
    # the test minutes-slow for no gain.
    npx = 10
    a = global_scalar_G_interpolated_table(
        "zzb", 0, he, 3, 3, npoles=npx, pole_selection="contribution_leading"
    )
    b = global_scalar_G_interpolated_table(
        "zzb", 0, he, 3, 3, npoles=npx, pole_selection="contribution"
    )
    for D in [mp.mpf("1.2"), mp.mpf("2.7")]:
        assert abs(mp.mpc(a.evaluate(D)[0][0]) - mp.mpc(b.evaluate(D)[0][0])) < mp.mpf("1e-20")


def test_contribution_low_dps_high_npoles_stays_accurate():
    """At low dps + high npoles the legacy contour contribution scorer sampled
    the block on contours around far-out poles where the chiral evaluation
    starved (a precision collapse). The analytic contribution scorer builds each
    pole's principal part in closed form (exact residue + regularized-2F1 regular
    part), so it stays robust there: no cryptic crash, no silently-garbage
    interpolant -- it must reproduce the true block at low dps. (The
    precision-collapse guard still protects the contour fallback for the cases
    that genuinely need it: singular-c regularized towers.)"""
    from virasoro_blocks.global_blocks import (
        global_scalar_G_interpolated_table, global_evaluate_block,
    )
    he = [mp.mpf("0.2")] * 4
    mp.mp.dps = 30
    lo = global_scalar_G_interpolated_table(
        "zzb", 0, he, 3, 3, npoles=18, pole_selection="contribution"
    )
    mp.mp.dps = 80
    for D in (mp.mpf("1.5"), mp.mpf("3.0"), mp.mpf("5.0")):
        # scalar_G at ell = 0 is B(h = hbar = D/2).
        block = global_evaluate_block(
            mp.mpf("0.5"), mp.mpf("0.5"), D / 2, D / 2, he,
            coord="zzb", m_max=3, n_max=3,
            chiral_norm=None, z_shift=None, zbar_shift=None,
        )
        ref = block[0][0]
        got = mp.mpc(lo.evaluate(D)[0][0])
        assert abs(got - ref) / abs(ref) < mp.mpf("1e-20")


def test_large_spin_contribution_equals_contribution_leading():
    """Large-spin tower poles are simple (each twist a distinct L), so the full
    `contribution` reduces to the legacy `contribution_leading` -- it runs on the
    real-axis-only sampler and gives the identical interpolant."""
    from virasoro_blocks.large_spin import large_spin_twist_table
    mp.mp.dps = 30
    he = [mp.mpf("0.2")] * 4
    common = dict(
        mode="global", coordinate="zzb", h_externals=he, m_max=2, n_max=2,
        Lmax=10, npoles=4, c=mp.mpf("3.7"), q_order=10, chiral_norm=4,
        z_shift=-2 * mp.mpf("0.2"),
    )
    a = large_spin_twist_table(mp.mpf("0.5"), pole_selection="contribution_leading", **common)
    b = large_spin_twist_table(mp.mpf("0.5"), pole_selection="contribution", **common)
    pa = sorted(a.prefactor.poles, key=lambda z: (float(mp.re(z)), float(mp.im(z))))
    pb = sorted(b.prefactor.poles, key=lambda z: (float(mp.re(z)), float(mp.im(z))))
    assert len(pa) == len(pb)
    for x, y in zip(pa, pb):
        assert abs(mp.mpc(x) - mp.mpc(y)) < mp.mpf("1e-20")


def test_analytic_leading_residue_matches_numeric_pole_selection():
    """Analytic leading-residue scoring must preserve the selected-pole set
    of the legacy numeric residue scorer for both global and large-spin paths.

    Score magnitudes can differ in tiny tail regimes (numeric eps-probe noise
    vs exact zeros), but selected poles for practical caps must match.
    """
    from virasoro_blocks.global_blocks import (
        _GlobalBlockSampler2d,
        _externals_diffs,
        _genuine_2d_poles,
        _genuine_chiral_poles,
        _make_residue_scorer,
        global_block_exp_base,
    )
    from virasoro_blocks.interpolation import DampedRational, select_kept_poles
    from virasoro_blocks.large_spin import _LargeSpinTwistSampler, large_spin_base
    from virasoro_blocks.polyderivs import _residue_score_indices_from_dims

    mp.mp.dps = 60

    def _pole_key(z):
        zz = mp.mpc(z)
        return (round(float(mp.re(zz)), 14), round(float(mp.im(zz)), 14))

    def _same_selected_set(pref_a, pref_b):
        pa = sorted((_pole_key(p) for p in pref_a.poles))
        pb = sorted((_pole_key(p) for p in pref_b.poles))
        return pa == pb

    he = [mp.mpf("0.0625")] * 4
    d12, d34 = _externals_diffs(he)
    npoles = 8

    # Global 2d scalar block (includes repeated poles at ell=0).
    ell = 0
    shift = mp.mpf(0)
    base = global_block_exp_base(16)
    global_sampler = _GlobalBlockSampler2d(
        "zzb", ell, he, he, mp.mpf(1), 16, mp.mpf(0), mp.mpf(0), 3, 3,
        basis="scalar_G",
    )
    global_pool = _genuine_2d_poles(
        ell,
        2 * npoles + 4,
        dh12=d12,
        dh34=d34,
        dhb12=d12,
        dhb34=d34,
        shift=shift,
    )
    assert len(global_pool) == 2 * npoles + 4
    value_entries = _residue_score_indices_from_dims(3, 3, "value", value_entry=(0, 0))
    raw_global = _make_residue_scorer(
        global_sampler, base, 3, 3, score_entries="value"
    )(global_pool)
    ana_global = global_sampler._analytic_leading_residue_scores(
        global_pool, value_entries, base
    )
    full_global = DampedRational(
        constant=mp.mpc(1), poles=tuple(global_pool), base=base
    ).shift_x(shift)
    pref_raw = select_kept_poles(full_global, npoles, "residue", scores=raw_global)
    pref_ana = select_kept_poles(full_global, npoles, "residue", scores=ana_global)
    assert _same_selected_set(pref_raw, pref_ana)
    raw_cl = [
        s / float(abs(mp.mpc(q))) if abs(mp.mpc(q)) > 1e-12 else s
        for s, q in zip(raw_global, full_global.poles)
    ]
    ana_cl = [
        s / float(abs(mp.mpc(q))) if abs(mp.mpc(q)) > 1e-12 else s
        for s, q in zip(ana_global, full_global.poles)
    ]
    pref_raw_cl = select_kept_poles(
        full_global, npoles, "contribution_leading", scores=raw_cl
    )
    pref_ana_cl = select_kept_poles(
        full_global, npoles, "contribution_leading", scores=ana_cl
    )
    assert _same_selected_set(pref_raw_cl, pref_ana_cl)

    # Large-spin global tower (simple poles).
    twist = mp.mpf("0.5")
    Lmin = mp.mpf(10)
    tower_base = large_spin_base("global", 16)
    tower_sampler = _LargeSpinTwistSampler(
        "global", "zzb", he, 16, mp.mpf(0), 3, 3, twist,
        basis="scalar_G", exchanged_parity="1",
    )
    tower_pool = [
        p - twist / 2
        for p in _genuine_chiral_poles(
            2 * npoles + 4,
            delta12=d12,
            delta34=d34,
            shift=Lmin + twist / 2,
        )
    ]
    raw_tower = _make_residue_scorer(
        tower_sampler, tower_base, 3, 3, score_entries="value"
    )(tower_pool)
    ana_tower = tower_sampler._analytic_leading_residue_scores(
        tower_pool, value_entries, tower_base
    )
    full_tower = DampedRational(
        constant=mp.mpc(1), poles=tuple(tower_pool), base=tower_base
    ).shift_x(Lmin)
    pref_raw = select_kept_poles(full_tower, npoles, "residue", scores=raw_tower)
    pref_ana = select_kept_poles(full_tower, npoles, "residue", scores=ana_tower)
    assert _same_selected_set(pref_raw, pref_ana)
    raw_cl = [
        s / float(abs(mp.mpc(q))) if abs(mp.mpc(q)) > 1e-12 else s
        for s, q in zip(raw_tower, full_tower.poles)
    ]
    ana_cl = [
        s / float(abs(mp.mpc(q))) if abs(mp.mpc(q)) > 1e-12 else s
        for s, q in zip(ana_tower, full_tower.poles)
    ]
    pref_raw_cl = select_kept_poles(
        full_tower, npoles, "contribution_leading", scores=raw_cl
    )
    pref_ana_cl = select_kept_poles(
        full_tower, npoles, "contribution_leading", scores=ana_cl
    )
    assert _same_selected_set(pref_raw_cl, pref_ana_cl)


@pytest.mark.parametrize("c_value", ["1", "25"])
def test_numeric_contribution_at_singular_c(c_value):
    """contribution works at a singular c >= 1 through the regularized
    skeleton + sampler (collapsed multiple-pole principal part is finite),
    matching the full symbolic regularized table."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import _aggregate_contribution_scores
    mp.mp.dps = 60
    c = mp.mpf(c_value)
    skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(2, SC, 2, 2, c=cc, q_order=8, lambda_cut=None, skeleton=True), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(c=c, **_scalar_sampler(c, 2))
    full = R.merged_delta_table(lambda cc: G_zzb_poly_table(2, SC, 2, 2, c=cc, q_order=8, lambda_cut=None), c=c)
    num = _aggregate_contribution_scores(
        [skel], delta_eval=mp.mpf(2), score_entries="value", pole_ranking="numeric"
    )
    sym = _contribution_scores(full, delta_eval=mp.mpf(2), score_entries="value")
    worst = max(abs(num[i] - sym[i]) / max(abs(sym[i]), mp.mpf(1)) for i in range(len(sym)))
    assert worst < mp.mpf("1e-6")


@pytest.mark.parametrize("c_value", ["25", "28"])
def test_probe_pair_chiral_F_scorer_matches_tseries_scorer(c_value):
    """The certified probe-pair scorer reproduces the exact t-series scorer for a
    chiral ``F`` table (a single holomorphic half: no convolution, no swap)."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        F_z_poly_table,
        _aggregate_contribution_scores,
    )

    mp.mp.dps = 70
    c = mp.mpf(c_value)
    lam = 2
    q_order = 8
    chiral_kw = dict(
        coordinate="z", chiral=True, spinning=False, projected=False,
        h_externals=SC, hbar_externals=None, ell=0,
        m_max=lam, n_max=0, b=None, q_order=q_order, chiral_norm=None,
        z_shift=None, zbar_shift=None, center=None, center_bar=None,
        parity=None, parity_external=None, exchanged_parity=None,
    )

    def _skel(method):
        skel = R.merged_delta_skeleton(lambda cc: F_z_poly_table(SC, lam, c=cc, q_order=q_order, skeleton=True), c=c)
        skel.numeric_sampler = R._RegularizedNumericSampler(
            c=c, **chiral_kw, scorer_method=method
        )
        return skel

    skel_series = _skel("laurent")
    skel_extrap = _skel("pair")
    series = _aggregate_contribution_scores(
        [skel_series], delta_eval=mp.mpf(2), score_entries="value", pole_ranking="numeric"
    )
    extrap = _aggregate_contribution_scores(
        [skel_extrap], delta_eval=mp.mpf(2), score_entries="value", pole_ranking="numeric"
    )

    assert skel_series.numeric_sampler._last_analytic_diag["reason"] == "series"
    assert skel_extrap.numeric_sampler._last_analytic_diag["reason"] == "pair"
    assert len(extrap) > 0
    worst = max(
        abs(series[i] - extrap[i]) / max(abs(extrap[i]), mp.mpf(1))
        for i in range(len(extrap))
    )
    # the pair certifies the ranking digits (the scores only rank)
    assert worst < mp.mpf(10) ** (-ranking_digits())


@pytest.mark.parametrize("ell", [2, 3])
def test_probe_pair_scalar_G_scorer_matches_tseries_scorer(ell):
    """Compare probe-pair and exact t-series scores for even and odd spin.

    The value entry is (0,0) for the symmetric block and (0,1) for the
    antisymmetric block.
    """
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import _aggregate_contribution_scores

    mp.mp.dps = 70
    c = mp.mpf("25")

    def _skel(method):
        skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(
                ell, SC, 2, 2, c=cc, q_order=8, lambda_cut=None, skeleton=True), c=c)
        skel.numeric_sampler = R._RegularizedNumericSampler(
            c=c, **_scalar_sampler(c, ell), scorer_method=method
        )
        return skel

    skel_series, skel_extrap = _skel("laurent"), _skel("pair")
    series = _aggregate_contribution_scores(
        [skel_series], delta_eval=mp.mpf(ell), score_entries="value", pole_ranking="numeric"
    )
    extrap = _aggregate_contribution_scores(
        [skel_extrap], delta_eval=mp.mpf(ell), score_entries="value", pole_ranking="numeric"
    )
    assert skel_series.numeric_sampler._last_analytic_diag["reason"] == "series"
    assert skel_extrap.numeric_sampler._last_analytic_diag["reason"] == "pair"
    assert len(extrap) > 0
    worst = max(
        abs(series[i] - extrap[i]) / max(abs(extrap[i]), mp.mpf(1))
        for i in range(len(extrap))
    )
    # the pair certifies the ranking digits (the scores only rank)
    assert worst < mp.mpf(10) ** (-ranking_digits())


def test_probe_pair_ordered_B_scorer_matches_tseries_scorer():
    """The certified probe-pair scorer reproduces the exact t-series scorer for the
    ordered ``B`` block (spinning externals ``h_i != hbar_i``, ``sigma = 0``:
    no swap term, both orderings independent)."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        B_zzb_poly_table,
        _aggregate_contribution_scores,
    )

    mp.mp.dps = 70
    c = mp.mpf("25")
    h_ext = [mp.mpf("0.13")] * 4
    hbar_ext = [mp.mpf("0.17")] * 4
    ell = 2
    skw = dict(
        coordinate="zzb", chiral=False, spinning=True, projected=False,
        h_externals=h_ext, hbar_externals=hbar_ext, ell=ell, m_max=2, n_max=2,
        b=None, q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None,
        center=None, center_bar=None, parity=None, parity_external=None,
        exchanged_parity="1",
    )

    def _skel(method):
        skel = R.merged_delta_skeleton(lambda cc: B_zzb_poly_table(
                ell, h_ext, hbar_ext, 2, 2, c=cc, q_order=8,
                lambda_cut=None, skeleton=True), c=c)
        skel.numeric_sampler = R._RegularizedNumericSampler(
            c=c, **skw, scorer_method=method
        )
        return skel

    skel_series, skel_extrap = _skel("laurent"), _skel("pair")
    series = _aggregate_contribution_scores(
        [skel_series], delta_eval=mp.mpf(ell), score_entries="value", pole_ranking="numeric"
    )
    extrap = _aggregate_contribution_scores(
        [skel_extrap], delta_eval=mp.mpf(ell), score_entries="value", pole_ranking="numeric"
    )
    assert skel_series.numeric_sampler._last_analytic_diag["reason"] == "series"
    assert skel_extrap.numeric_sampler._last_analytic_diag["reason"] == "pair"
    assert len(extrap) > 0
    worst = max(
        abs(series[i] - extrap[i]) / max(abs(extrap[i]), mp.mpf(1))
        for i in range(len(extrap))
    )
    # the pair certifies the ranking digits (the scores only rank)
    assert worst < mp.mpf(10) ** (-ranking_digits())


@pytest.mark.parametrize("parity", ["+", "-"])
def test_tseries_parity_projected_B_scorer_matches_symbolic(parity):
    """Compare exact t-series scores with symbolic principal parts for both signs.

    The minus projection uses the antisymmetric value entry (0,1).
    """
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        parity_projected_B_zzb_poly_table,
        _aggregate_contribution_scores,
    )

    mp.mp.dps = 70
    c = mp.mpf("25")
    he = [mp.mpf("0.13")] * 4
    ell = 0
    skw = dict(
        coordinate="zzb", chiral=False, spinning=True, projected=True,
        h_externals=he, hbar_externals=he, ell=ell, m_max=2, n_max=2, b=None,
        q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None,
        center=None, center_bar=None, parity=parity, parity_external=None,
        exchanged_parity="1",
    )

    skel = R.merged_delta_skeleton(lambda cc: parity_projected_B_zzb_poly_table(
            ell, he, he, 2, 2, c=cc, q_order=8, parity=parity,
            lambda_cut=None, skeleton=True), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(
        c=c, **skw, scorer_method="laurent"
    )
    series = _aggregate_contribution_scores(
        [skel], delta_eval=mp.mpf(1), score_entries="value", pole_ranking="numeric"
    )
    assert skel.numeric_sampler._last_analytic_diag["reason"] == "series"

    full = R.merged_delta_table(lambda cc: parity_projected_B_zzb_poly_table(
            ell, he, he, 2, 2, c=cc, q_order=8, parity=parity, lambda_cut=None), c=c)
    sym = _contribution_scores(full, delta_eval=mp.mpf(1), score_entries="value")
    assert len(sym) > 0
    worst = max(
        abs(series[i] - sym[i]) / max(abs(sym[i]), mp.mpf(1))
        for i in range(len(sym))
    )
    assert worst < mp.mpf("1e-20")


@pytest.mark.parametrize("ell", [2, 3])
def test_tseries_folded_scalar_G_scorer_matches_symbolic(ell):
    """The exact t-series scorer reproduces the SYMBOLIC full-numerator scores
    for the parity-folded scalar ``G`` block on the ``ws`` (w, s) coordinate at
    both even ``ell`` (r-even, no smoothing) and odd ``ell`` (r-odd, one
    ``sqrt(s)`` factor stripped via the Delta-independent parity fold). The fold
    is applied on the reconstructed source-band constant parts; the symbolic
    folded table is the ground truth."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        G_ws_poly_table,
        _aggregate_contribution_scores,
    )

    mp.mp.dps = 70
    c = mp.mpf("25")
    skw = dict(
        coordinate="ws", chiral=False, spinning=False, projected=False,
        h_externals=SC, hbar_externals=SC, ell=ell, m_max=2, n_max=2, b=None,
        q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None, center=None,
        center_bar=None, parity=None, parity_external=None, exchanged_parity="1",
    )

    skel = R.merged_delta_skeleton(lambda cc: G_ws_poly_table(ell, SC, 2, 2, c=cc, q_order=8, skeleton=True), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(
        c=c, **skw, scorer_method="laurent"
    )
    series = _aggregate_contribution_scores(
        [skel], delta_eval=mp.mpf(ell), score_entries="value", pole_ranking="numeric"
    )
    assert skel.numeric_sampler._last_analytic_diag["reason"] == "series"

    full = R.merged_delta_table(lambda cc: G_ws_poly_table(ell, SC, 2, 2, c=cc, q_order=8), c=c)
    sym = _contribution_scores(full, delta_eval=mp.mpf(ell), score_entries="value")
    assert len(sym) > 0
    worst = max(
        abs(series[i] - sym[i]) / max(abs(sym[i]), mp.mpf(1))
        for i in range(len(sym))
    )
    assert worst < mp.mpf("1e-20")


@pytest.mark.parametrize("center,center_bar", [
    (mp.mpf("0.4"), None),                  # diagonal, off-symmetric recenter
    (mp.mpf("0.43"), mp.mpf("0.57")),       # off-diagonal (z != zbar)
])
def test_tseries_recentered_scalar_G_scorer_matches_symbolic(center, center_bar):
    """The exact t-series scorer reproduces the SYMBOLIC full-numerator scores
    for the recentered scalar ``G`` block: an off-symmetric diagonal recenter
    (``center`` only) and a genuine off-diagonal block (distinct ``center`` /
    ``center_bar``, so the n-axis carries its own recentered chiral half). The
    symbolic full table is the ground truth (the cancellation-bound
    extrapolation reference defers at this working precision for recentered
    blocks)."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        G_zzb_poly_table,
        _aggregate_contribution_scores,
    )

    mp.mp.dps = 70
    c = mp.mpf("25")
    ell = 2
    skw = dict(
        coordinate="zzb", chiral=False, spinning=False, projected=False,
        h_externals=SC, hbar_externals=SC, ell=ell, m_max=2, n_max=2, b=None,
        q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None,
        center=center, center_bar=center_bar, parity=None, parity_external=None,
        exchanged_parity="1",
    )

    skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(
            ell, SC, 2, 2, c=cc, q_order=8, center=center, center_bar=center_bar,
            lambda_cut=None, skeleton=True), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(
        c=c, **skw, scorer_method="laurent"
    )
    series = _aggregate_contribution_scores(
        [skel], delta_eval=mp.mpf(ell), score_entries="value", pole_ranking="numeric"
    )
    assert skel.numeric_sampler._last_analytic_diag["reason"] == "series"

    full = R.merged_delta_table(lambda cc: G_zzb_poly_table(
            ell, SC, 2, 2, c=cc, q_order=8, center=center, center_bar=center_bar,
            lambda_cut=None), c=c)
    sym = _contribution_scores(full, delta_eval=mp.mpf(ell), score_entries="value")
    assert len(sym) > 0
    worst = max(
        abs(series[i] - sym[i]) / max(abs(sym[i]), mp.mpf(1))
        for i in range(len(sym))
    )
    assert worst < mp.mpf("1e-20")


# ---------------------------------------------------------------------------
# Singular-c fixed-spin robust path (cluster-level, post-collision object).
#
# Strategy: the regularized FULL table's principal-part scorer operates on the
# post-collision object (collapsed multiple poles -> finite higher-order
# principal part), so it never sees the per-branch 1/eps blow-up.  The
# regularized-skeleton numeric *contour* sampler integrates the same object.
# These agree to ~machine precision across the whole fixed-spin matrix, which is
# the evidence that the robust path is well conditioned at singular c.
# ---------------------------------------------------------------------------


def _ws_sampler(c, ell):
    kw = dict(_scalar_sampler(c, ell))
    kw["coordinate"] = "ws"
    return kw


def _qqb_sampler(c, ell, center, center_bar):
    kw = dict(_scalar_sampler(c, ell))
    kw["coordinate"] = "qqb"
    kw["center"] = center
    kw["center_bar"] = center_bar
    return kw


def _spinning_sampler(c, ell, h_externals, hbar_externals):
    kw = dict(_scalar_sampler(c, ell))
    kw["h_externals"] = h_externals
    kw["hbar_externals"] = hbar_externals
    kw["spinning"] = True
    return kw


def _projected_sampler(c, ell, h_externals, hbar_externals, parity):
    kw = dict(_scalar_sampler(c, ell))
    kw["h_externals"] = h_externals
    kw["hbar_externals"] = hbar_externals
    kw["projected"] = True
    kw["parity"] = parity
    return kw


def _fixed_spin_robust_worst(
    builder, sampler_kw, *, c, ell, score_entries, h_reg=SC
):
    """Worst relative disagreement between the regularized full-table
    principal-part scorer (analytic) and the regularized-skeleton contour
    scorer at a singular c."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        _contribution_scores,
        _numeric_contribution_scores,
    )

    deval = mp.mpf(ell)
    skel = R.merged_delta_skeleton(lambda cc: builder(cc, skeleton=True), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(c=c, **sampler_kw)
    contour = _numeric_contribution_scores(
        skel, delta_eval=deval, score_entries=score_entries
    )

    full = R.merged_delta_table(lambda cc: builder(cc, skeleton=False), c=c)
    analytic = _contribution_scores(
        full, delta_eval=deval, score_entries=score_entries
    )
    return max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(contour))
    )


# both c, even and odd ell, and each entry set at both c -- not the full product
@pytest.mark.parametrize(
    "c_value, ell, score_entries",
    [("1", 0, "value"), ("1", 3, "kept"), ("25", 0, "kept"), ("25", 3, "value")],
)
def test_singular_fixed_spin_robust_matches_contour_diagonal(c_value, ell, score_entries):
    """Diagonal (zzb) singular-c fixed-spin: the cluster-level regularized
    principal-part scorer agrees with the contour scorer to ~machine precision
    across c in {1, 25}, ell in {0, 3}, and both value/kept entry sets.

    This is the robust path -- it scores the post-collision object, so the
    per-branch 1/eps cancellation that wrecks the tower probe never arises."""
    mp.mp.dps = 60
    c = mp.mpf(c_value)
    worst = _fixed_spin_robust_worst(
        lambda cc, skeleton: G_zzb_poly_table(
            ell, SC, 2, 2, c=cc, q_order=8, lambda_cut=None, skeleton=skeleton
        ),
        _scalar_sampler(c, ell),
        c=c, ell=ell, score_entries=score_entries,
    )
    assert worst < mp.mpf("1e-18")


@pytest.mark.parametrize("c_value", ["1", "25"])
@pytest.mark.parametrize("ell", [0, 3])
def test_singular_fixed_spin_robust_matches_contour_folded(c_value, ell):
    """Folded (ws) coordinate: same robust agreement at singular c.  Folding
    strips the odd-parity sqrt(s) factor but does not change the collision
    structure of the Kac poles, so the cluster-level scorer stays exact."""
    from virasoro_blocks.polyderivs import G_ws_poly_table

    mp.mp.dps = 60
    c = mp.mpf(c_value)
    worst = _fixed_spin_robust_worst(
        lambda cc, skeleton: G_ws_poly_table(
            ell, SC, 2, 2, c=cc, q_order=8, skeleton=skeleton
        ),
        _ws_sampler(c, ell),
        c=c, ell=ell, score_entries="value",
    )
    assert worst < mp.mpf("1e-18")


@pytest.mark.parametrize("c_value", ["1", "25"])
@pytest.mark.parametrize("ell", [0, 3])
def test_singular_fixed_spin_robust_matches_contour_offdiagonal(c_value, ell):
    """Off-diagonal (qqb, z != zbar) coordinate: robust agreement at singular
    c.  Detuning z from zbar reintroduces the odd-parity diagonal cancellation
    but, again, leaves the cluster structure intact for the robust scorer."""
    from virasoro_blocks.polyderivs import G_qqb_poly_table

    mp.mp.dps = 60
    c = mp.mpf(c_value)
    center = mp.mpf("0.37")
    center_bar = mp.mpf("0.55")
    worst = _fixed_spin_robust_worst(
        lambda cc, skeleton: G_qqb_poly_table(
            ell, SC, 2, 2, c=cc, q_order=8,
            center=center, center_bar=center_bar, skeleton=skeleton,
        ),
        _qqb_sampler(c, ell, center, center_bar),
        c=c, ell=ell, score_entries="value",
    )
    assert worst < mp.mpf("1e-18")


@pytest.mark.parametrize("ell", [1, 3])
@pytest.mark.parametrize("score_entries", ["kept"])
def test_singular_fixed_spin_c25_analytic_scorer_replaces_contour(ell, score_entries):
    """At c = 25 (b**2 = 1) the skeleton singular-c sampler scores the diagonal
    scalar-G block analytically (merged chiral Kac Laurent, no symbolic
    numerators, no contour) and matches the contour scorer to ~1e-21 (odd
    ell, the full kept entry set).  This is the production path: the skeleton has no numerators, so the
    scorer runs from chiral residues alone."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        _value_score_entry,
        _residue_score_indices_from_dims,
        principal_part_contribution_scores,
    )

    mp.mp.dps = 60
    c = mp.mpf("25")
    skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(
            ell, SC, 2, 2, c=cc, q_order=8, lambda_cut=None, skeleton=True
        ), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(c=c, **_scalar_sampler(c, ell))
    roots = [mp.mpc(r) for r in skel.denominator_roots]
    if score_entries == "value":
        entries = [_value_score_entry(skel)]
    else:
        entries = _residue_score_indices_from_dims(2, 2, "kept")

    analytic = skel.numeric_sampler._analytic_contribution_scores(
        roots, mp.mpf(ell), entries, skel.exp_base
    )
    assert analytic is not None, "c=25 analytic scorer unexpectedly deferred"

    contour = principal_part_contribution_scores(
        roots, lambda pts, e: skel.numeric_sampler(pts, entries=e),
        delta_eval=mp.mpf(ell), entries=entries, exp_base=skel.exp_base,
        n_select=len(roots),
    )
    worst = max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(roots))
    )
    assert worst < mp.mpf("1e-18")


# Every recognized recursion-singular regime, across spins: c = 25, the c > 25
# rational-b^2 dual c = 28 (order-3 chiral collisions), c = 1, and the c < 1
# minimal models. The analytic merged-Laurent scorer must FIRE at all of them.
_SINGULAR_FIRE_CASES = [
    ("25", 0),
    ("28", 0), ("28", 3),
    ("1", 1),
    ("0.5", 2),
    ("0.7", 0),
]


@pytest.mark.parametrize("c_value, ell", _SINGULAR_FIRE_CASES)
def test_singular_fixed_spin_analytic_scorer_fires_and_matches_contour(c_value, ell):
    """At every recognized singular c the analytic merged-Laurent scorer FIRES
    (status ``analytic-ok``, does not silently defer) and reproduces the contour
    to <1e-18. Asserting it fires -- not merely "agree-or-defer" -- locks in
    that the production path makes zero contour calls at these loci; a bare
    agree-or-defer check would pass even if the scorer declined everywhere."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        _value_score_entry,
        principal_part_contribution_scores,
    )

    mp.mp.dps = 60
    c = mp.mpf(c_value)
    skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(
            ell, SC, 2, 2, c=cc, q_order=8, lambda_cut=None, skeleton=True
        ), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(c=c, **_scalar_sampler(c, ell))
    roots = [mp.mpc(r) for r in skel.denominator_roots]
    entries = [_value_score_entry(skel)]

    analytic = skel.numeric_sampler._analytic_contribution_scores(
        roots, mp.mpf(ell), entries, skel.exp_base
    )
    diag = skel.numeric_sampler.last_analytic_diagnostic()
    assert analytic is not None, f"scorer declined at singular c={c_value} ell={ell}: {diag}"
    assert diag["status"] == "analytic-ok", diag

    contour = principal_part_contribution_scores(
        roots, lambda pts, e: skel.numeric_sampler(pts, entries=e),
        delta_eval=mp.mpf(ell), entries=entries, exp_base=skel.exp_base,
        n_select=len(roots),
    )
    worst = max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(roots))
    )
    assert worst < mp.mpf("1e-18")


# ---------------------------------------------------------------------------
# Singular-c analytic scorer across EVERY block basis and coordinate.
#
# The merged-Laurent scorer generalizes from diagonal scalar G to: ordered B,
# parity-projected B (diagonal + off-diagonal recenters), off-diagonal scalar G,
# folded G / projected B (ws / xt / qqb_ws), and chiral F (z / y / q). Each case
# must FIRE (status analytic-ok) and match the contour scorer to ~machine
# precision at a recognized singular c. Externals: spinning B uses distinct
# h / hbar so the two chiral halves genuinely differ.
# ---------------------------------------------------------------------------

_EB_H = [mp.mpf("0.13"), mp.mpf("0.13"), mp.mpf("0.17"), mp.mpf("0.17")]
_EB_HB = [mp.mpf("0.11"), mp.mpf("0.11"), mp.mpf("0.19"), mp.mpf("0.19")]
_EB_OFF = (mp.mpf("0.4"), mp.mpf("0.55"))


def _eb_sampler(**over):
    kw = dict(
        coordinate="zzb", chiral=False, spinning=False, projected=False,
        h_externals=_EB_H, hbar_externals=_EB_H, ell=0, m_max=2, n_max=2, b=None,
        q_order=8, chiral_norm=None, z_shift=None, zbar_shift=None, center=None,
        center_bar=None, parity=None, parity_external=None, exchanged_parity="1",
    )
    kw.update(over)
    return kw


def _extended_basis_cases():
    from virasoro_blocks.polyderivs import (
        B_zzb_poly_table, B_yyb_poly_table, B_qqb_poly_table,
        parity_projected_B_zzb_poly_table, parity_projected_B_qqb_poly_table,
        G_qqb_poly_table,
        G_ws_poly_table, G_xt_poly_table, G_qqb_ws_poly_table,
        parity_projected_B_ws_poly_table, parity_projected_B_xt_poly_table,
        parity_projected_B_qqb_ws_poly_table,
        F_z_poly_table, F_y_poly_table, F_q_poly_table,
    )
    cen, cenb = _EB_OFF
    spin = _EB_H != _EB_HB
    cases = []

    # ordered B, diagonal, every non-folded coordinate.
    # (one ell per coordinate, rotating through 0, 1, 2)
    for coord, ob, ell in (("zzb", B_zzb_poly_table, 0), ("yyb", B_yyb_poly_table, 1),
                           ("qqb", B_qqb_poly_table, 2)):
        cases.append((
            f"ordered_B[{coord}]-ell{ell}",
            (lambda cc, e=ell, b=ob: b(e, _EB_H, _EB_HB, 2, 2, c=cc, q_order=8, skeleton=True)),
            _eb_sampler(coordinate=coord, spinning=spin,
                        hbar_externals=_EB_HB, ell=ell),
            ell,
        ))

    # parity-projected B (+/-), diagonal (one ell per parity) and off-diagonal.
    for parity, ell in (("+", 0), ("-", 1)):
        cases.append((
            f"projected_B({parity})[zzb]-ell{ell}",
            (lambda cc, e=ell, p=parity: parity_projected_B_zzb_poly_table(
                e, _EB_H, _EB_HB, 2, 2, parity=p, c=cc, q_order=8, skeleton=True)),
            _eb_sampler(spinning=spin, hbar_externals=_EB_HB, ell=ell,
                        projected=True, parity=parity),
            ell,
        ))
        cases.append((
            f"projected_B({parity})[qqb-offdiag]",
            (lambda cc, p=parity: parity_projected_B_qqb_poly_table(
                1, _EB_H, _EB_HB, 2, 2, parity=p, c=cc, q_order=8,
                center=cen, center_bar=cenb, skeleton=True)),
            _eb_sampler(coordinate="qqb", spinning=spin, hbar_externals=_EB_HB,
                        ell=1, projected=True, parity=parity,
                        center=cen, center_bar=cenb),
            1,
        ))

    # off-diagonal scalar G (odd ell: the diagonal-canceling case).
    cases.append((
        "offdiag_G[qqb]-ell1",
        (lambda cc: G_qqb_poly_table(
            1, _EB_H, 2, 2, c=cc, q_order=8, center=cen, center_bar=cenb,
            lambda_cut=None, skeleton=True)),
        _eb_sampler(coordinate="qqb", ell=1, center=cen, center_bar=cenb),
        1,
    ))

    # folded scalar G + projected B on every folded coordinate.
    folded = (("ws", G_ws_poly_table, parity_projected_B_ws_poly_table),
              ("xt", G_xt_poly_table, parity_projected_B_xt_poly_table),
              ("qqb_ws", G_qqb_ws_poly_table, parity_projected_B_qqb_ws_poly_table))
    # (one G and one projected B per coordinate, rotating ell and parity)
    for (coord, gb, pb), ell, parity in zip(folded, (0, 1, 1), ("+", "-", "+")):
        cases.append((
            f"folded_G[{coord}]-ell{ell}",
            (lambda cc, e=ell, b=gb: b(e, _EB_H, 2, 2, c=cc, q_order=8, skeleton=True)),
            _eb_sampler(coordinate=coord, ell=ell),
            ell,
        ))
        cases.append((
            f"folded_projected_B({parity})[{coord}]",
            (lambda cc, b=pb, p=parity: b(
                0, _EB_H, _EB_HB, 2, 2, parity=p, c=cc, q_order=8, skeleton=True)),
            _eb_sampler(coordinate=coord, spinning=spin, hbar_externals=_EB_HB,
                        ell=0, projected=True, parity=parity),
            0,
        ))

    # chiral F on every chiral coordinate, symmetric + recentered.
    # (alternating symmetric / recentered across the coordinates)
    for coord, fb, center in (("z", F_z_poly_table, None), ("y", F_y_poly_table, cen),
                              ("q", F_q_poly_table, None)):
        cases.append((
            f"chiral_F[{coord}]-{'sym' if center is None else 'rc'}",
            (lambda cc, b=fb, ce=center: b(_EB_H, 3, c=cc, q_order=8, center=ce, skeleton=True)),
            _eb_sampler(coordinate=coord, chiral=True, ell=None, m_max=3,
                        n_max=None, center=center),
            mp.mpf("0.3"),
        ))
    return cases


_EXTENDED_BASIS_CASES = _extended_basis_cases()


@pytest.mark.parametrize("c_value", ["25", "1"])
@pytest.mark.parametrize(
    "label, builder, sampler_kw, deval",
    _EXTENDED_BASIS_CASES,
    ids=[c[0] for c in _EXTENDED_BASIS_CASES],
)
def test_singular_analytic_scorer_all_bases_coords(c_value, label, builder, sampler_kw, deval):
    """The singular-c merged-Laurent analytic scorer fires and matches the
    contour scorer for every block basis (scalar G, ordered B, parity-projected
    B, chiral F) and coordinate (non-folded zzb/yyb/qqb, folded ws/xt/qqb_ws,
    chiral z/y/q), diagonal and off-diagonal, at recognized singular c."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        _value_score_entry,
        principal_part_contribution_scores,
    )

    mp.mp.dps = 60
    c = mp.mpf(c_value)
    h_reg = sampler_kw["h_externals"]
    skel = R.merged_delta_skeleton(builder, c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(c=c, **sampler_kw)
    roots = [mp.mpc(r) for r in skel.denominator_roots]
    entries = [_value_score_entry(skel)]
    deval = mp.mpf(deval)

    analytic = skel.numeric_sampler._analytic_contribution_scores(
        roots, deval, entries, skel.exp_base
    )
    diag = skel.numeric_sampler.last_analytic_diagnostic()
    assert analytic is not None, f"{label}: scorer declined at c={c_value}: {diag}"
    assert diag["status"] == "analytic-ok", f"{label}: {diag}"

    contour = principal_part_contribution_scores(
        roots, lambda pts, e: skel.numeric_sampler(pts, entries=e),
        delta_eval=deval, entries=entries, exp_base=skel.exp_base,
        n_select=len(roots),
    )
    worst = max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(roots))
    )
    assert worst < mp.mpf("1e-15"), f"{label}: worst {mp.nstr(worst, 4)}"


@pytest.mark.parametrize("c_value", ["3.7", "0.2"])
def test_singular_fixed_spin_analytic_scorer_declines_at_generic_c(c_value):
    """At a non-singular c the merged scorer is not applicable (production uses
    the ordinary recursion there, never this sampler). It must decline cleanly
    -- never fail loudly, never silently score a non-Kac block pole. This is the
    only legitimate decline path (alongside an unsupported configuration)."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import _value_score_entry

    mp.mp.dps = 60
    ell = 0
    c = mp.mpf(c_value)
    skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(
            ell, SC, 2, 2, c=cc, q_order=8, lambda_cut=None, skeleton=True
        ), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(c=c, **_scalar_sampler(c, ell))
    roots = [mp.mpc(r) for r in skel.denominator_roots]
    entries = [_value_score_entry(skel)]

    analytic = skel.numeric_sampler._analytic_contribution_scores(
        roots, mp.mpf(ell), entries, skel.exp_base
    )
    assert analytic is None
    diag = skel.numeric_sampler.last_analytic_diagnostic()
    assert diag["reason"] == "non_singular_c", diag


@pytest.mark.parametrize("ell", [0])
def test_singular_fixed_spin_analytic_scorer_fires_at_degenerate_external(ell):
    """The Ising sigma external h = 1/16 is itself a degenerate Kac weight, so a
    cluster's genuine chiral pole order can differ from the skeleton
    multiplicity. The analytic scorer must still FIRE (no order/collapse gate)
    and reproduce the contour: an order mismatch does not change the score, so
    declining to contour there would be an unnecessary silent fallback at a
    singular c we can handle."""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        _value_score_entry,
        principal_part_contribution_scores,
    )

    mp.mp.dps = 70
    c = mp.mpf("1/2")
    lam = 2
    q_order = 16
    h_ext = [mp.mpf(1) / 16] * 4
    deval = mp.mpf(ell) + mp.mpf("1.13")
    skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(
            ell, h_ext, lam, lam, c=cc, q_order=q_order,
            lambda_cut=None, skeleton=True,
        ), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(
        c=c, **{**_scalar_sampler(c, ell), "h_externals": h_ext,
                "hbar_externals": h_ext, "m_max": lam, "n_max": lam,
                "q_order": q_order}
    )
    roots = [mp.mpc(r) for r in skel.denominator_roots]
    entries = [_value_score_entry(skel)]

    analytic = skel.numeric_sampler._analytic_contribution_scores(
        roots, deval, entries, skel.exp_base
    )
    diag = skel.numeric_sampler.last_analytic_diagnostic()
    assert analytic is not None, f"scorer declined at the degenerate external: {diag}"
    assert diag["status"] == "analytic-ok", diag

    contour = principal_part_contribution_scores(
        roots, lambda pts, e: skel.numeric_sampler(pts, entries=e),
        delta_eval=deval, entries=entries, exp_base=skel.exp_base,
        n_select=len(roots),
    )
    worst = max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(roots))
    )
    assert worst < mp.mpf("1e-14")


@pytest.mark.parametrize("c_value, ell", [("28", 3)])
def test_singular_fixed_spin_analytic_scorer_robust_to_high_q_order(c_value, ell):
    """Robustness to a larger q_order (more Kac poles, deeper collisions): the
    fixed-spin scorer keeps firing and matching the contour. Includes the
    hardest collision (c = 28, ell = 3, order-3 chiral) whose block-row
    extrapolation must stay convergent as the pole count grows.

    (If c = 1/2 is added back: a chiral half can land on the degenerate Ising
    weight h = 1/16, around which the pole crowding at high q_order leaves a
    small, precision-INDEPENDENT structural offset (~1e-14) in the score --
    still ~12 sig figs, far below any ranking gap. The tolerance below keeps
    that floor.)"""
    import virasoro_blocks.regularization as R
    from virasoro_blocks.polyderivs import (
        _value_score_entry,
        principal_part_contribution_scores,
    )

    mp.mp.dps = 70
    q_order = 16
    c = mp.mpf(c_value)
    skel = R.merged_delta_skeleton(lambda cc: G_zzb_poly_table(
            ell, SC, 2, 2, c=cc, q_order=q_order, lambda_cut=None, skeleton=True
        ), c=c)
    skel.numeric_sampler = R._RegularizedNumericSampler(
        c=c, **{**_scalar_sampler(c, ell), "q_order": q_order}
    )
    roots = [mp.mpc(r) for r in skel.denominator_roots]
    entries = [_value_score_entry(skel)]

    analytic = skel.numeric_sampler._analytic_contribution_scores(
        roots, mp.mpf(ell), entries, skel.exp_base
    )
    diag = skel.numeric_sampler.last_analytic_diagnostic()
    assert analytic is not None, f"scorer declined at c={c_value}, q={q_order}: {diag}"
    assert diag["status"] == "analytic-ok", diag

    contour = principal_part_contribution_scores(
        roots, lambda pts, e: skel.numeric_sampler(pts, entries=e),
        delta_eval=mp.mpf(ell), entries=entries, exp_base=skel.exp_base,
        n_select=len(roots),
    )
    worst = max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(roots))
    )
    assert worst < (mp.mpf("1e-13") if c_value == "0.5" else mp.mpf("1e-18"))


# Spinning externals (h != hbar) for the ordered/projected B blocks.
SPIN_H = [mp.mpf("0.13")] * 4
SPIN_HB = [mp.mpf("0.17")] * 4


@pytest.mark.parametrize("c_value", ["1", "25"])
@pytest.mark.parametrize("ell", [0, 3])
def test_singular_fixed_spin_robust_matches_contour_ordered_B(c_value, ell):
    """Ordered B block (spinning externals h != hbar): the robust cluster-level
    path is just as well conditioned as scalar G at singular c.  The ordered
    block carries the same Kac-pole collision structure, and the regularized
    skeleton has the right structural degree, so contour == analytic."""
    from virasoro_blocks.polyderivs import B_zzb_poly_table

    mp.mp.dps = 60
    c = mp.mpf(c_value)
    worst = _fixed_spin_robust_worst(
        lambda cc, skeleton: B_zzb_poly_table(
            ell, SPIN_H, SPIN_HB, 2, 2, c=cc, q_order=8, skeleton=skeleton
        ),
        _spinning_sampler(c, ell, SPIN_H, SPIN_HB),
        c=c, ell=ell, score_entries="value", h_reg=SPIN_H,
    )
    assert worst < mp.mpf("1e-18")


@pytest.mark.parametrize("c_value", ["3.7", "7.0"])
@pytest.mark.parametrize("parity", ["+", "-"])
def test_projected_B_analytic_scorer_matches_validated_symbolic_table(c_value, parity):
    """The parity_projected_B *analytic* contribution scorer is correct: it
    reproduces, bit-for-bit, the principal part computed independently from the
    symbolic poly table -- and that symbolic table is itself validated against
    the proper numeric block to 1e-45 in
    ``test_numeric_parity_projected.py``.  So the analytic scorer inherits that
    correctness.  This is the reference the singular-c robust path relies on,
    and it pins the blame for the contour disagreement (next test) on the
    numeric sampler, NOT on the analytic scorer."""
    from virasoro_blocks.polyderivs import (
        parity_projected_B_zzb_poly_table,
        _contribution_scores,
        _value_score_entry,
    )

    mp.mp.dps = 50
    c = mp.mpf(c_value)
    ell = 0
    deval = mp.mpf(ell)
    full = parity_projected_B_zzb_poly_table(
        ell, SPIN_H, SPIN_HB, 2, 2, parity=parity, c=c, q_order=8,
    )
    ent = _value_score_entry(full)
    num = full.numerators[ent[0]][ent[1]]
    den = full.denominator
    lead = den.coeffs[-1]
    roots = [mp.mpc(r) for r in full.denominator_roots]
    groups = _group_roots_by_location(roots, mp.mpf(10) ** -30)

    def pp_at(loc, idxs, Dt):
        m = len(idxs)
        others = [roots[k] for k in range(len(roots)) if k not in set(idxs)]

        def cof(D):
            v = lead
            for q in others:
                v *= D - q
            return v

        a = mp.taylor(lambda D: num.evaluate(D) / cof(D), loc, m - 1)
        d = Dt - loc
        return sum(a[j] * d ** j for j in range(m)) / d ** m

    analytic = _contribution_scores(full, delta_eval=deval, score_entries="value")
    for loc, idx in groups:
        ref = abs(pp_at(loc, idx, deval))
        assert abs(analytic[idx[0]] - ref) <= mp.mpf(10) ** -28 * max(ref, mp.mpf(1))


@pytest.mark.parametrize("c_value, ell", [("1", 0), ("25", 2)])
def test_singular_fixed_spin_projected_B_robust_matches_analytic(c_value, ell):
    """Merged-Laurent projected-B scores agree with symbolic principal parts.

    The numeric scoring interface uses the analytic path directly.
    """
    from virasoro_blocks.polyderivs import parity_projected_B_zzb_poly_table

    mp.mp.dps = 60
    c = mp.mpf(c_value)
    worst = _fixed_spin_robust_worst(
        lambda cc, skeleton: parity_projected_B_zzb_poly_table(
            ell, SPIN_H, SPIN_HB, 2, 2, parity="+", c=cc, q_order=8,
            skeleton=skeleton,
        ),
        _projected_sampler(c, ell, SPIN_H, SPIN_HB, "+"),
        c=c, ell=ell, score_entries="value", h_reg=SPIN_H,
    )
    # The analytic scorer now witnesses projected B: agreement at the robust floor.
    assert worst < mp.mpf("1e-18")


# ---------------------------------------------------------------------------
# Singular-c tower scores use merged Laurent coefficients at the exact charge.
# Colliding simple residues can diverge individually while their signed sums
# remain finite. Score the merged principal parts before taking magnitudes.
# The analytic path supports diagonal, off-diagonal, and folded coordinates;
# a genuine pole in the fixed antiholomorphic weight raises an error.
# ---------------------------------------------------------------------------


def _tower_setup(c):
    """Common large-spin virasoro tower configuration for the singular-c
    characterization tests."""
    from virasoro_blocks.large_spin import (
        _LargeSpinTwistSampler,
        _virasoro_chiral_pole_pool,
        large_spin_base,
    )

    shift = mp.mpf("8")
    twist = mp.mpf("0.5")
    entries = [(0, 0)]
    base = large_spin_base("virasoro", 4)
    h_ext = [mp.mpf("0.1")] * 4

    def pool(cc):
        return _virasoro_chiral_pole_pool(
            h_externals=h_ext, c=cc, q_order=12, twist=twist,
        )

    def sampler(cc):
        return _LargeSpinTwistSampler(
            "virasoro", "zzb", h_ext, 4, mp.mpf("-0.2"), 2, 2, twist,
            c=cc, q_order=12, exchanged_parity="1", basis="scalar_G",
            hbar_externals=h_ext,
        )

    return shift, twist, entries, base, h_ext, pool, sampler


@pytest.mark.parametrize("score_entries", ["value", "kept"])
def test_singular_tower_analytic_merged_scorer_matches_contour(score_entries):
    """The shipped fix: the diagonal Virasoro tower's analytic contribution
    scorer (`_LargeSpinTwistSampler._analytic_residue_scores`) now WORKS at a
    recognized-singular c via the merged higher-order chiral Laurent route, and
    agrees with the contour scorer to ~1e-24 on every cluster -- including the
    deeply collided ones where the rejected per-branch probe blew up by 1e80+.
    This is the analytic replacement for the contour at singular c."""
    from virasoro_blocks.polyderivs import principal_part_contribution_scores

    mp.mp.dps = 60
    c = mp.mpf("25")
    shift, _, _, base, _, pool, sampler = _tower_setup(c)
    entries = [(0, 0)] if score_entries == "value" else [
        (m, n) for m in range(3) for n in range(3)
    ]
    target_roots = pool(c)
    ss = sampler(c)
    assert ss._singular  # the path under test is the singular-c branch

    analytic = ss._analytic_residue_scores(list(target_roots), shift, entries, base)
    assert analytic is not None, "merged analytic scorer unexpectedly deferred"

    contour = principal_part_contribution_scores(
        target_roots,
        lambda pts, e: ([ss(p) for p in pts], False),
        delta_eval=shift, entries=entries, exp_base=base,
        n_select=len(target_roots),
    )
    worst = max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(target_roots))
    )
    assert worst < mp.mpf("1e-20")


@pytest.mark.parametrize("c_value", ["0.5", "0.7"])
def test_singular_tower_merged_scorer_extends_to_minimal_models(c_value):
    """Beyond c = 25, the tower's merged scorer now also FIRES at the c < 1
    minimal models: ``kac_poles`` carries the complex Kac weights of the
    regularization probe, and one matched-cluster ``pole_laurent`` call supplies
    the confluent holomorphic principal part. It reproduces the contour to
    ranking grade."""
    from virasoro_blocks.polyderivs import principal_part_contribution_scores

    mp.mp.dps = 60
    c = mp.mpf(c_value)
    shift, _, entries, base, _, pool, sampler = _tower_setup(c)
    ss = sampler(c)
    assert ss._singular
    target_roots = list(pool(c))

    analytic = ss._analytic_residue_scores(target_roots, shift, entries, base)
    assert analytic is not None, "tower merged scorer declined at a minimal model"

    contour = principal_part_contribution_scores(
        target_roots,
        lambda pts, e: ([ss(p) for p in pts], False),
        delta_eval=shift, entries=entries, exp_base=base,
        n_select=len(target_roots),
    )
    worst = max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(target_roots))
    )
    # c = 0.7's contour reference is itself only ~1e-9 accurate (tightly spaced
    # poles raise the DFT floor), so the bound tracks the reference, not the
    # analytic scorer; c = 0.5 matches to full ranking grade.
    assert worst < (mp.mpf("1e-20") if c_value == "0.5" else mp.mpf("1e-8"))


@pytest.mark.slow  # ~40 s: two q = 24 tower pole pools (slow tier only)
@pytest.mark.parametrize("c_value, q_order", [("25", 24), ("0.5", 24)])
def test_singular_tower_merged_scorer_robust_to_high_q_order(c_value, q_order):
    """High-order merged tower scores agree with contour extraction.

    The scale-free ratio-gap test must separate a colliding cluster from
    nearby distinct weights.
    """
    from virasoro_blocks.large_spin import (
        _LargeSpinTwistSampler,
        _virasoro_chiral_pole_pool,
        large_spin_base,
    )
    from virasoro_blocks.polyderivs import principal_part_contribution_scores

    mp.mp.dps = 70
    c = mp.mpf(c_value)
    shift = mp.mpf("8")
    twist = mp.mpf("0.5")
    entries = [(0, 0)]
    base = large_spin_base("virasoro", 4)
    h_ext = [mp.mpf("0.1")] * 4
    pool = list(_virasoro_chiral_pole_pool(
        h_externals=h_ext, c=c, q_order=q_order, twist=twist,
    ))
    ss = _LargeSpinTwistSampler(
        "virasoro", "zzb", h_ext, 4, mp.mpf("-0.2"), 2, 2, twist,
        c=c, q_order=q_order, exchanged_parity="1", basis="scalar_G",
        hbar_externals=h_ext,
    )

    analytic = ss._analytic_residue_scores(pool, shift, entries, base)
    assert analytic is not None, f"tower scorer declined at c={c_value}, q={q_order}"

    contour = principal_part_contribution_scores(
        pool, lambda pts, e: ([ss(p) for p in pts], False),
        delta_eval=shift, entries=entries, exp_base=base, n_select=len(pool),
    )
    worst = max(
        abs(analytic[i] - contour[i]) / max(abs(contour[i]), mp.mpf(1))
        for i in range(len(pool))
    )
    assert worst < mp.mpf("1e-18")


def test_singular_tower_c1_analytic_refuses_the_singular_block():
    """Reject a c=1 tower whose antiholomorphic half has a genuine Kac pole.

    Here hbar=twist/2=1/4; the error must identify the merged pole.
    """
    from virasoro_blocks.requests import BlockPoleError

    mp.mp.dps = 60
    c = mp.mpf("1")
    shift, _, _, base, _, pool, sampler = _tower_setup(c)
    ss = sampler(c)
    assert ss._singular
    with pytest.raises(BlockPoleError, match="genuinely singular"):
        ss._analytic_residue_scores(list(pool(c)), shift, [(0, 0)], base)


def test_singular_tower_c25_production_contour_path_is_well_conditioned():
    """The contour scorer is now only the FALLBACK for the singular-c tower (the
    diagonal Virasoro tower scores analytically via the merged chiral Laurent
    route -- test_singular_tower_analytic_merged_scorer_matches_contour -- and
    the contour still serves singular folded / off-diagonal coordinates and c=1).
    This test pins that the fallback remains well conditioned at c = 25: stable
    under changes of contour radius and oversampling, so it reads off the
    post-collision principal part, not an aliasing/cancellation artifact.

    Contrast with the rejected per-branch probe (extrapolating abs'd per-branch
    analytic scores), which blows up by 1e80+ on the same poles."""
    from virasoro_blocks.polyderivs import principal_part_contribution_scores

    mp.mp.dps = 60
    c = mp.mpf("25")
    shift, _, entries, base, _, pool, sampler = _tower_setup(c)
    target_roots = pool(c)
    ss = sampler(c)
    ev = lambda pts, e: ([ss(p) for p in pts], False)
    kw = dict(delta_eval=shift, entries=entries, exp_base=base,
              n_select=len(target_roots))

    # Two near-converged contours (heavy oversampling) at different radii: a
    # well-conditioned extraction is independent of the contour geometry.
    a = principal_part_contribution_scores(
        target_roots, ev, **kw, radius_fraction=mp.mpf("0.4"), n_extra=16
    )
    b = principal_part_contribution_scores(
        target_roots, ev, **kw, radius_fraction=mp.mpf("0.25"), n_extra=20
    )
    worst = max(
        abs(a[i] - b[i]) / max(abs(b[i]), mp.mpf("1e-300"))
        for i in range(len(target_roots))
    )
    # The decisive contrast with the rejected per-branch probe: the contour
    # scores stay finite, non-negative, and O(1) -- bounded by the largest
    # contour score, never the 1e32..1e83 blow-up the probe produces on the same
    # collided clusters -- and two independent contour geometries agree to
    # ~1e-3 (the deep multiplicity-6 collisions set this residual; the probe is
    # wrong by 80+ orders there).
    assert all(mp.isfinite(s) and s >= 0 for s in a)
    assert max(a) < mp.mpf("1e3")
    assert worst < mp.mpf("1e-2")


def test_singular_tower_c1_names_its_genuine_pole():
    """Name the genuine order-six pole at hbar=1/4 for this c=1 tower."""
    from virasoro_blocks.requests import BlockPoleError

    mp.mp.dps = 60
    _, _, _, _, _, _, sampler = _tower_setup(mp.mpf("1"))
    ss = sampler(mp.mpf("1"))
    with pytest.raises(BlockPoleError, match="genuinely singular"):
        ss(mp.mpf("8"))
