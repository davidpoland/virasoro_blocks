"""Tests for the read-only singularity diagnostics.

The exact-family classifiers (`c1_*`, `c25_*`, the minimal-model series, the
positive-rational `c > 25` branch) are checked against the tolerance-based
numeric analyzer at low recursion orders, and the model-recognition helpers
are checked on representative loci. The classifier is consumed by the
Laurent regularization in `recursion.H_q_series_laurent`;
cross-checks of that path live in `tests/core/test_laurent.py`.
"""

from fractions import Fraction
from collections import defaultdict

import mpmath as mp
import pytest

from virasoro_blocks.params import c_origin_of_b

from virasoro_blocks.singularity import (
    analyze_recursion_singularities,
    c1_A_rs_singular_factors,
    c1_collision_families,
    c25_A_rs_singular_factors,
    c25_collision_families,
    minimal_model_A_rs_singular_factors,
    minimal_model_collision_families,
    minimal_model_pq,
    negative_unit_b2_q,
    positive_rational_A_rs_singular_factors,
    positive_rational_b2_pq,
    nearby_singular_locus,
    positive_rational_collision_families,
    recognized_singular_locus,
    unitary_minimal_A_rs_singular_factors,
    unitary_minimal_collision_families,
    unitary_minimal_model_m,
)


def test_analyze_recursion_singularities_detects_c1_collisions_and_A_poles():
    mp.mp.dps = 50
    report = analyze_recursion_singularities(mp.mpf("1"), 4)

    assert report.first_a_pole_order == 2
    assert any(item.pair == (1, 2) and item.factor == (1, 1) for item in report.a_pole_factors)
    assert any(set(group.pairs) == {(1, 2), (2, 1)} for group in report.pole_collisions)


def test_analyze_recursion_singularities_detects_delayed_ising_A_pole_onset():
    mp.mp.dps = 50

    low_order = analyze_recursion_singularities(mp.mpf("0.5"), 14)
    onset = analyze_recursion_singularities(mp.mpf("0.5"), 15)

    assert low_order.a_pole_factors == ()
    assert low_order.first_collision_order == 1
    assert any(set(group.pairs) == {(1, 1), (3, 2)} for group in low_order.pole_collisions)
    assert onset.first_a_pole_order == 15
    assert any(item.pair == (5, 3) and item.factor == (4, 3) for item in onset.a_pole_factors)


def test_c1_collision_families_match_numeric_analyzer_through_q10():
    mp.mp.dps = 50
    numeric = analyze_recursion_singularities(mp.mpf("1"), 10)
    exact = c1_collision_families(10)

    numeric_groups = {tuple(group.pairs) for group in numeric.pole_collisions}
    exact_groups = {tuple(group.pairs) for group in exact}
    assert exact_groups == numeric_groups


def test_c1_A_rs_singular_factors_match_analyzer_through_q10():
    mp.mp.dps = 50
    report = analyze_recursion_singularities(mp.mpf("1"), 10)
    exact = c1_A_rs_singular_factors(10)

    report_factors = {(item.pair, item.factor) for item in report.a_pole_factors}
    exact_factors = {(item.pair, item.factor) for item in exact}
    assert exact_factors == report_factors


def test_unitary_minimal_model_m_recognizes_ising():
    assert unitary_minimal_model_m(mp.mpf("0.5")) == 3


def test_minimal_model_pq_recognizes_lee_yang():
    assert minimal_model_pq(mp.mpf("-4.4")) == (2, 5)


def test_positive_rational_b2_pq_recognizes_c28():
    assert positive_rational_b2_pq(mp.mpf("28")) == (1, 2)


def test_exact_dual_locus_keeps_the_denominator_cutoff():
    # c = 613.06 is b^2 = 1/100: singular only once q_order reaches the cutoff max(16, q_order) >= 100
    c = Fraction('613.06')
    assert recognized_singular_locus(c, q_order=40) is None
    assert recognized_singular_locus('613.06') is None
    assert recognized_singular_locus(c, q_order=100) == "b² = 1/100"


def test_standalone_recognizers_preserve_exact_labels_and_search_bounds():
    from decimal import Decimal

    with mp.workdps(15):
        for value in [Fraction(-22, 5), Decimal('-4.4'), '-4.4', '-22/5', -4.4]:
            assert minimal_model_pq(value) == (2, 5)
            assert minimal_model_pq(value, tolerance=0) == (2, 5)
            assert minimal_model_pq(value, max_n=2) is None
        for value in [Fraction(38, 5), Fraction(253, 10)]:
            # Only 253/10 is a positive-rational-b² charge (b² = 4/5).
            expected = (4, 5) if value == Fraction(253, 10) else None
            for label in [value, Decimal(str(float(value))), str(value), float(value)]:
                assert positive_rational_b2_pq(label) == expected
                assert positive_rational_b2_pq(label, max_q=4) is None
        m = 50
        charge = 1 - Fraction(6, m*(m+1))
        assert unitary_minimal_model_m(charge, max_m=m) == m
        assert unitary_minimal_model_m(charge, max_m=m-1) is None


def test_nearby_locus_preserves_literal_offsets_and_complex_distance():
    from decimal import Decimal

    caller_precision = mp.mp.prec
    with mp.workdps(15):
        for center, name in [(Fraction(7, 10), 'M(4,5)'), (Fraction(1), 'c = 1'), (Fraction(25), 'c = 25')]:
            for value in [center, str(center), Decimal(str(float(center))), float(center)]:
                assert nearby_singular_locus(value, max_n=10) is None
            for sign in (-1, 1):
                shift = Fraction(1, 10**70)
                label, distance = nearby_singular_locus(center + sign*shift, max_n=10)
                assert name in label
                assert mp.almosteq(distance, mp.mpf('1e-70'))
                assert nearby_singular_locus(center + sign*shift, abs_tolerance=shift/2, max_n=10) is None
        label, distance = nearby_singular_locus(mp.mpc('.500000000001', '1e-12'), max_n=10)
        assert 'Ising' in label
        assert mp.almosteq(distance, mp.sqrt(2)*mp.mpf('1e-12'), rel_eps=mp.mpf('1e-4'))
        assert nearby_singular_locus(Fraction(51, 100), abs_tolerance='.01', max_n=10) is not None
        assert nearby_singular_locus(Fraction(51, 100) + Fraction(1, 10**70),
                                     abs_tolerance='.01', max_n=10) is None
        for name, bad in [('max_n', True), ('max_n', -1), ('abs_tolerance', True),
                          ('abs_tolerance', -1), ('abs_tolerance', 'nan')]:
            with pytest.raises(ValueError, match=name):
                nearby_singular_locus('.5', **{name: bad})
        for bad in (True, 'nan', 'inf'):
            with pytest.raises(ValueError):
                nearby_singular_locus(bad)
    assert mp.mp.prec == caller_precision


def test_nearby_locus_finds_the_closest_bounded_coprime_model():
    from math import gcd

    with mp.workdps(35):
        for bound in (3, 6, 11):
            targets = [(1 - Fraction(6*(p-q)**2, p*q), f'M({p},{q})')
                       for q in range(3, bound+1) for p in range(2, q) if gcd(p, q) == 1]
            targets += [(Fraction(1), 'c = 1'), (Fraction(25), 'c = 25')]
            for charge in (Fraction(-4), Fraction(-101, 10), Fraction(-43, 100), Fraction(41, 100),
                           Fraction(97, 100), Fraction(10001, 10000)):
                expected, name = min(targets, key=lambda item: abs(charge - item[0]))
                label, distance = nearby_singular_locus(charge, max_n=bound, abs_tolerance=100)
                assert name in label
                assert mp.almosteq(distance, mp.mpf(str(abs(charge - expected).numerator)) /
                                   abs(charge - expected).denominator)


def test_standalone_recognizers_validate_controls_and_finite_charges():
    for recognize, limit in [(minimal_model_pq, 'max_n'), (positive_rational_b2_pq, 'max_q')]:
        for value in [True, -1, .5, mp.nan, mp.inf]:
            with pytest.raises(ValueError, match=limit):
                recognize('.5', **{limit: value})
        for value in [True, -.1, 'nan', 'inf', 1j]:
            with pytest.raises(ValueError, match='tolerance'):
                recognize('.5', tolerance=value)
        for value in [True, mp.nan, mp.inf, mp.mpc(1, mp.inf)]:
            with pytest.raises(ValueError):
                recognize(value)
        assert recognize('.5', **{limit: 0}) is None


@pytest.mark.parametrize('order', [True, Fraction(3, 2), -1, '3'])
def test_locus_queries_validate_cutoffs_before_classification(order):
    from virasoro_blocks.singularity import bare_singular_locus, exact_b_squared, exact_locus_c

    for charge in (1, mp.mpf(1), 10, '1+1e-70j'):
        for query in (recognized_singular_locus, exact_locus_c, exact_b_squared, bare_singular_locus):
            with pytest.raises(ValueError, match='q_order'):
                query(charge, q_order=order)


def test_exact_family_catalogues_validate_and_normalize_integer_controls():
    catalogues = [(c1_collision_families, ()), (c25_collision_families, ()),
                  (c1_A_rs_singular_factors, ()), (c25_A_rs_singular_factors, ()),
                  (positive_rational_collision_families, (1, 2)),
                  (positive_rational_A_rs_singular_factors, (1, 2)),
                  (minimal_model_collision_families, (3, 4)),
                  (minimal_model_A_rs_singular_factors, (3, 4)),
                  (unitary_minimal_collision_families, (3,)),
                  (unitary_minimal_A_rs_singular_factors, (3,))]
    for catalogue, args in catalogues:
        with pytest.raises(ValueError, match='q_order'):
            catalogue(*args, q_order=True)
        assert catalogue(*(float(arg) for arg in args), q_order=4.) == catalogue(*args, q_order=4)
    for catalogue in (positive_rational_collision_families, positive_rational_A_rs_singular_factors,
                      minimal_model_collision_families, minimal_model_A_rs_singular_factors):
        with pytest.raises(ValueError, match='p'):
            catalogue(Fraction(5, 2), 4, 0)
        with pytest.raises(ValueError, match='q'):
            catalogue(3, True, 0)
    for catalogue in (unitary_minimal_collision_families, unitary_minimal_A_rs_singular_factors):
        with pytest.raises(ValueError, match='m'):
            catalogue(Fraction(5, 2), 0)


@pytest.mark.parametrize('eps', [-1, 1])
def test_bare_singular_order_is_invariant_under_equivalent_ratios(eps):
    from virasoro_blocks.singularity import bare_recursion_singular_order

    order = 12
    pairs = [(r, s) for r in range(1, order+1) for s in range(1, order//r+1)]
    for p, q in ((1, 1), (1, 2), (2, 3), (2, 2), (3, 3), (4, 6), (6, 4)):
        # Enumerate the defining denominator products and shifted Kac
        # equalities directly, without assuming a reduced ratio.
        hits = [r*s for r, s in pairs
                if any(eps*p*m + q*n == 0
                       for m in range(1-r, r+1) for n in range(1-s, s+1)
                       if (m, n) not in ((0, 0), (r, s)))]
        hits += [r*s + a*b for r, s in pairs for a, b in pairs
                 if r*s + a*b <= order and abs(eps*p*r-q*s) == abs(eps*p*a+q*b)]
        expected = min(hits) if hits else None
        assert bare_recursion_singular_order(eps, p, q, order) == expected
        assert bare_recursion_singular_order(eps, p, q, expected-1 if expected else 0) is None


def test_bare_singular_order_rejects_invalid_family_labels():
    from virasoro_blocks.singularity import bare_recursion_singular_order

    for field, value in [('eps', 0), ('eps', True), ('p', 0), ('p', Fraction(3, 2)),
                         ('q', 0), ('q', True), ('q_order', Fraction(3, 2))]:
        args = dict(eps=1, p=1, q=1, q_order=4)
        args[field] = value
        with pytest.raises(ValueError, match=field):
            bare_recursion_singular_order(**args)


def test_ising_collision_families_match_analyzer_through_q10():
    mp.mp.dps = 50
    report = analyze_recursion_singularities(mp.mpf("0.5"), 10)
    exact = unitary_minimal_collision_families(3, 10)

    report_groups = {frozenset(group.pairs) for group in report.pole_collisions}
    exact_groups = {frozenset(group.pairs) for group in exact}
    assert exact_groups == report_groups


def test_ising_A_rs_singular_factors_match_analyzer_at_first_onset():
    mp.mp.dps = 50
    low_report = analyze_recursion_singularities(mp.mpf("0.5"), 14)
    onset_report = analyze_recursion_singularities(mp.mpf("0.5"), 15)
    low_exact = unitary_minimal_A_rs_singular_factors(3, 14)
    onset_exact = unitary_minimal_A_rs_singular_factors(3, 15)

    assert low_report.a_pole_factors == ()
    assert low_exact == ()
    assert {(item.pair, item.factor) for item in onset_report.a_pole_factors} == {
        (item.pair, item.factor) for item in onset_exact
    }
    assert any(item.pair == (5, 3) and item.factor == (4, 3) for item in onset_exact)


def test_lee_yang_collision_families_match_analyzer_through_q12():
    mp.mp.dps = 50
    report = analyze_recursion_singularities(mp.mpf("-4.4"), 12)
    exact = minimal_model_collision_families(2, 5, 12)

    report_groups = {frozenset(group.pairs) for group in report.pole_collisions}
    exact_groups = {frozenset(group.pairs) for group in exact}
    assert exact_groups == report_groups


def test_lee_yang_A_rs_singular_factors_match_analyzer_at_first_onset():
    mp.mp.dps = 50
    low_report = analyze_recursion_singularities(mp.mpf("-4.4"), 11)
    onset_report = analyze_recursion_singularities(mp.mpf("-4.4"), 12)
    low_exact = minimal_model_A_rs_singular_factors(2, 5, 11)
    onset_exact = minimal_model_A_rs_singular_factors(2, 5, 12)

    assert low_report.a_pole_factors == ()
    assert low_exact == ()
    assert {(item.pair, item.factor) for item in onset_report.a_pole_factors} == {
        (item.pair, item.factor) for item in onset_exact
    }
    assert onset_exact == (
        next(item for item in onset_exact if item.pair == (6, 2) and item.factor == (5, 2)),
    )


def test_c28_positive_rational_families_match_analyzer_through_q10():
    mp.mp.dps = 50
    report = analyze_recursion_singularities(mp.mpf("28"), 10)
    exact_collisions = positive_rational_collision_families(1, 2, 10)
    exact_factors = positive_rational_A_rs_singular_factors(1, 2, 10)

    report_groups = {frozenset(group.pairs) for group in report.pole_collisions}
    exact_groups = {frozenset(group.pairs) for group in exact_collisions}
    assert exact_groups == report_groups
    assert {(item.pair, item.factor) for item in report.a_pole_factors} == {
        (item.pair, item.factor) for item in exact_factors
    }


def test_exact_singularity_routing_accepts_equivalent_numeric_forms():
    mp.mp.dps = 50

    ising_reports = [
        analyze_recursion_singularities(mp.mpf(1) / 2, 15),
        analyze_recursion_singularities(mp.mpf("0.5"), 15),
        analyze_recursion_singularities(mp.mpf(1.0 / 2.0), 15),
    ]
    ising_collision_sets = [{frozenset(group.pairs) for group in report.pole_collisions} for report in ising_reports]
    ising_factor_sets = [{(item.pair, item.factor) for item in report.a_pole_factors} for report in ising_reports]
    assert ising_collision_sets[0] == ising_collision_sets[1] == ising_collision_sets[2]
    assert ising_factor_sets[0] == ising_factor_sets[1] == ising_factor_sets[2]

    c28_reports = [
        analyze_recursion_singularities(mp.mpf(28), 10),
        analyze_recursion_singularities(mp.mpf("28.0"), 10),
        analyze_recursion_singularities(mp.mpf(56) / 2, 10),
    ]
    c28_collision_sets = [{frozenset(group.pairs) for group in report.pole_collisions} for report in c28_reports]
    c28_factor_sets = [{(item.pair, item.factor) for item in report.a_pole_factors} for report in c28_reports]
    assert c28_collision_sets[0] == c28_collision_sets[1] == c28_collision_sets[2]
    assert c28_factor_sets[0] == c28_factor_sets[1] == c28_factor_sets[2]
    assert all(c_origin_of_b(report.b) == Fraction(1, 2) for report in ising_reports)
    assert all(c_origin_of_b(report.b) == 28 for report in c28_reports)


def test_singularity_report_accepts_exact_charge_labels():
    with mp.workdps(30):
        reports = [analyze_recursion_singularities(c, 4) for c in
                   (Fraction(7, 10), '.7', '7/10', 0.7)]
        assert all(c_origin_of_b(report.b) == Fraction(7, 10) for report in reports)
        assert all(report.a_pole_factors == reports[0].a_pole_factors and
                   report.pole_collisions == reports[0].pole_collisions for report in reports)
        generic = analyze_recursion_singularities(Fraction(32), 4.0, tolerance=Fraction(1, 100))
        assert c_origin_of_b(generic.b) == 32
        assert generic.tolerance == mp.mpf('.01')
        for invalid in (True, -1, Fraction(3, 2), mp.inf):
            with pytest.raises(ValueError, match='q_order'):
                analyze_recursion_singularities(32, invalid)
        for invalid in (True, -1, mp.inf, 1j):
            with pytest.raises(ValueError, match='tolerance'):
                analyze_recursion_singularities(32, 4, tolerance=invalid)


@pytest.mark.parametrize('boundary', [1, 25])
def test_singularity_report_preserves_exact_offsets_from_boundary_charges(boundary):
    charge = Fraction(boundary) + Fraction(1, 10**70)
    with mp.workdps(30):
        report = analyze_recursion_singularities(str(boundary) + '.' + '0'*69 + '1',
                                                4, tolerance=0)
        assert c_origin_of_b(report.b) == charge
        assert report.c.real != boundary
        assert report.a_pole_factors == ()
        assert report.pole_collisions == ()
        assert mp.mp.dps == 30


def test_singularity_report_exactly_detects_negative_rational_extensions():
    for q, charge in ((2, -2), (3, -7), (4, Fraction(-25, 2))):
        beta = Fraction(-1, q)
        groups = defaultdict(list)
        for r in range(1, 9):
            for s in range(1, 8//r + 1):
                pole = ((1-r*r)*beta + 2*(1-r*s) + (1-s*s)/beta)/4
                groups[pole].append((r, s))
        expected = {tuple(sorted(pairs)): pole for pole, pairs in groups.items() if len(pairs) > 1}
        with mp.workdps(30):
            for source in (charge, mp.convert(charge)):
                report = analyze_recursion_singularities(source, 8, tolerance=0)
                assert {item.pairs for item in report.pole_collisions} == set(expected)
                assert all(mp.almosteq(item.pole, mp.convert(expected[item.pairs]))
                           for item in report.pole_collisions)
                assert any(item.pair == (q+1, 1) and item.factor == (q, 1) and item.magnitude == 0
                           for item in report.a_pole_factors)
                assert c_origin_of_b(report.b) == charge


def test_c25_families_match_analyzer_through_q10():
    mp.mp.dps = 50
    report = analyze_recursion_singularities(mp.mpf("25"), 10)
    exact_collisions = c25_collision_families(10)
    exact_factors = c25_A_rs_singular_factors(10)

    report_groups = {frozenset(group.pairs) for group in report.pole_collisions}
    exact_groups = {frozenset(group.pairs) for group in exact_collisions}
    assert exact_groups == report_groups
    assert {(item.pair, item.factor) for item in report.a_pole_factors} == {
        (item.pair, item.factor) for item in exact_factors
    }


# ---------------------------------------------------------------------------
# recognized_singular_locus: the CLI's singular-c detector.
# ---------------------------------------------------------------------------


def test_recognized_singular_locus_c_one():
    assert recognized_singular_locus(mp.mpf(1)) == "c = 1"


def test_recognized_singular_locus_c_twenty_five():
    assert recognized_singular_locus(mp.mpf(25)) == "c = 25"


def test_recognized_singular_locus_ising():
    """c = 1/2 is the M(3, 4) Ising minimal model."""
    label = recognized_singular_locus(mp.mpf(1) / 2)
    assert label is not None
    assert "M(3,4)" in label
    assert "Ising" in label


def test_recognized_singular_locus_lee_yang():
    """c = -22/5 is the M(2, 5) Yang-Lee minimal model."""
    label = recognized_singular_locus(mp.mpf(-22) / 5)
    assert label is not None
    assert "Yang-Lee" in label


def test_recognized_singular_locus_unitary_minimal_model_default_label():
    """A unitary minimal model without a hand-curated nickname falls back to
    the M(p, q) label plus the unitary index m."""
    # c = 1 - 6 / (m (m+1)) for m = 10 -> p = 10, q = 11, c = 1 - 6 / 110 = 104/110.
    c = mp.mpf(1) - mp.mpf(6) / (10 * 11)
    label = recognized_singular_locus(c)
    assert label is not None
    assert "M(10,11)" in label
    assert "m = 10" in label


def test_recognized_singular_locus_generic_c_returns_none():
    assert recognized_singular_locus(mp.mpf(10)) is None
    assert recognized_singular_locus(mp.mpf("1.7")) is None


def test_recognized_singular_locus_rational_b2_above_25():
    """For c > 25 with rational b² = p/q, the recursion has an A_rs factor
    `m·b + n/b = 0` for (m, n) ∝ (q, -p), so the locus is structurally
    singular at q_order ≥ p·(q + 1) for p <= q. `recognized_singular_locus` flags
    these via `positive_rational_b2_pq` with max_q = 16 (the default
    q_order used in the bootstrap interpolation pipeline):

      c = 50  → (p, q) = (1, 6)   → label "b² = 1/6"
      c = 33  → (p, q) = (1, 3)   → label "b² = 1/3"
      c = 28  → (p, q) = (1, 2)   → label "b² = 1/2"
      c = 26  → (p, q) = (2, 3)   → label "b² = 2/3"
      c = 38.5 → (p, q) = (1, 4)  → label "b² = 1/4"
    """
    for c, expected in (
        (mp.mpf(50), "b² = 1/6"),
        (mp.mpf(33), "b² = 1/3"),
        (mp.mpf(28), "b² = 1/2"),
        (mp.mpf(26), "b² = 2/3"),
        (mp.mpf("38.5"), "b² = 1/4"),
    ):
        assert recognized_singular_locus(c) == expected


def test_recognized_singular_locus_rational_b2_after_b_roundtrip():
    """`c_from_b(b_from_c(50))` accumulates ~10^-59 round-trip error at
    dps=60; with the new precision-tolerant `positive_rational_b2_pq`, the
    roundtripped c is still recognized. (Without this tolerance, F_block's
    auto-route -- which derives c from b -- would miss the dual-family
    loci and silently return the bare-path garbage.)"""
    mp.mp.dps = 60
    # b_from_c is now round-trip exact at c = 50; the tolerance must still absorb a round-trip-sized residual
    c_rt = mp.mpf(50) + mp.mpf(10) ** -58
    assert recognized_singular_locus(c_rt) == "b² = 1/6"


def test_recognized_singular_locus_high_q_rational_b2_returns_none():
    """The dual-family check uses max_q = 16 by default, which catches all
    loci that break the recursion at the standard q_order = 16 truncation.
    A higher-q rational b² (here q = 17) is inert at standard truncations
    and is reported as non-singular by `recognized_singular_locus`."""
    # (p, q) = (1, 17) → c = 13 + 6 * (1/17 + 17) = 13 + 6/17 + 102 = 115 + 6/17
    c = mp.mpf(13) + mp.mpf(6) * (mp.mpf(1) / 17 + mp.mpf(17))
    assert recognized_singular_locus(c) is None


def test_recognized_singular_locus_high_m_unitary_minimal_model():
    """The inverse-solve recognizes high-m unitary minimal models that the
    previous brute-force enumeration (capped at q <= 200) silently missed.

    M(500, 501) is c = 1 - 6/(500*501) ~ 0.99998; M(999, 1000) is even closer
    to c = 1. Both are recursion-singular and should auto-regularize."""
    mp.mp.dps = 60
    for m in (250, 500, 999):
        c = mp.mpf(1) - mp.mpf(6) / (m * (m + 1))
        label = recognized_singular_locus(c)
        assert label is not None
        assert f"M({m},{m + 1})" in label
        assert f"m = {m}" in label


def test_recognized_singular_locus_accepts_non_bit_exact_rationals():
    """The brute-force enumeration's `abs(c - target) == 0` matching was
    fragile: `mp.mpf('-3') / 5` and `mp.mpf(1) - mp.mpf(6) * 4 / 15` are both
    -0.6 at the print level but differ by ~1 ulp internally, so only the
    formula form matched. The inverse-solve uses an mpmath-ulp tolerance, so
    both forms now recognize M(3, 5)."""
    mp.mp.dps = 60
    a = mp.mpf(-3) / 5
    b = mp.mpf("-0.6")
    formula = mp.mpf(1) - mp.mpf(6) * (3 - 5) ** 2 / (3 * 5)
    for c in (a, b, formula):
        label = recognized_singular_locus(c)
        assert label is not None
        assert "M(3,5)" in label


def test_recognized_singular_locus_near_but_not_exact_is_none():
    """Detection is exact at the current working precision; a perturbation
    that is meaningful (larger than mpmath ulp) reads as non-singular. The
    inverse-solve uses a tolerance of 10**(-(dps - 10)) on the integer-ness
    check, comfortably tighter than any sensible user perturbation."""
    mp.mp.dps = 50
    nearly_ising = mp.mpf(1) / 2 + mp.mpf("1e-15")
    assert recognized_singular_locus(nearly_ising) is None


# ---------------------------------------------------------------------------
# nearby_singular_locus: the CLI's near-miss detector.
# ---------------------------------------------------------------------------


def test_nearby_singular_locus_finds_close_ising():
    """A c value 1e-13 above 1/2 should resolve to (Ising, ~1e-13)."""
    c = mp.mpf(1) / 2 + mp.mpf("1e-13")
    result = nearby_singular_locus(c)
    assert result is not None
    label, distance = result
    assert "Ising" in label
    assert distance < mp.mpf("2e-13")


def test_nearby_singular_locus_finds_close_c1():
    c = mp.mpf(1) + mp.mpf("1e-15")
    result = nearby_singular_locus(c)
    assert result is not None
    label, distance = result
    assert "c = 1" in label
    assert distance < mp.mpf("2e-15")


def test_nearby_singular_locus_returns_none_at_exact_locus():
    """Exact matches are handled by recognized_singular_locus; this helper
    explicitly returns None for them so the caller doesn't double-fire."""
    assert nearby_singular_locus(mp.mpf(1) / 2) is None
    assert nearby_singular_locus(mp.mpf(1)) is None
    assert nearby_singular_locus(mp.mpf(25)) is None


def test_nearby_singular_locus_returns_none_outside_tolerance():
    """c that's 1e-5 from a locus is far enough not to be flagged."""
    c = mp.mpf(1) / 2 + mp.mpf("1e-5")
    assert nearby_singular_locus(c) is None


def test_nearby_singular_locus_returns_none_at_generic_c():
    assert nearby_singular_locus(mp.mpf(10)) is None
    assert nearby_singular_locus(mp.mpf("1.7")) is None


def test_nearby_singular_locus_picks_closest():
    """If c is between two singular loci within tolerance, the closer one wins."""
    # M(3,4) Ising is c = 1/2; M(2,3) trivial is c = 0. We sit a hair above 1/2.
    c = mp.mpf(1) / 2 + mp.mpf("1e-13")
    label, _ = nearby_singular_locus(c)
    assert "M(3,4)" in label


def test_nearby_singular_locus_custom_tolerance():
    """The tolerance is configurable."""
    c = mp.mpf(1) / 2 + mp.mpf("1e-8")
    assert nearby_singular_locus(c) is None  # outside default 1e-10
    assert nearby_singular_locus(c, abs_tolerance=mp.mpf("1e-6")) is not None
