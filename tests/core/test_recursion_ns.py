"""N=1 NS superconformal blocks: the validation ladder of the plan.

The decisive gate is the pair of CLOSED FORMS in [HJS-b] eq (6.2): at ``c = 3/2``
with all external weights ``1/8``,

    H^1_Delta      = theta_3(q^2)          (independent of Delta)
    H^(1/2)_Delta  = theta_2(q^2) / Delta

Those two series pin the entire transcription at once -- pole set, ``Delta_rs``,
``A_rs`` including its sign, both fusion polynomials with their mod-4 index
conditions, the starred/unstarred selection rule, the sector coupling, and the
``theta_3`` seed. ``c = 3/2`` is a singular point of the recursion (``beta = 1``
makes ``A_rs`` diverge against compensating zeros of ``P_rs``), so it is
approached as ``c = 3/2 + eps`` and the residual is expected to scale like
``eps`` -- which is itself asserted, since an ``eps``-independent residual would
mean a genuine error rather than the regularization.
"""

from fractions import Fraction

import mpmath as mp
import pytest

import virasoro_blocks.poles as vpoles
import virasoro_blocks.recursion_ns as ns

# c = 3/2 exactly is singular; approach it. dps is set well above the eps
# exponent so the closed-form residual is limited by eps, not by roundoff.
CLOSED_FORM_DPS = 90
CLOSED_FORM_EPS_EXP = 40

# A generic NS central charge: c < 3/2 keeps beta real, and beta^2 = 1.370... is
# irrational so no A_rs denominator can vanish. NOT 81/70 or 7/10 -- those are
# minimal models and hence recursion-singular (see test_degenerate_c_raises_loudly).
GENERIC_C = mp.mpf(6) / 5


def _near_free_c(eps_exp=CLOSED_FORM_EPS_EXP):
    return mp.mpf(3) / 2 + mp.mpf(10) ** (-eps_exp)


def _eighth_externals():
    return [mp.mpf(1) / 8] * 4


# --------------------------------------------------------------------------
# Gate 0: the closed forms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [30])  # the lower orders are prefixes of this series
@pytest.mark.parametrize("delta_str", ["0.37", "5.5", "-2.25"])
def test_even_sector_matches_theta3_closed_form(order, delta_str):
    """[HJS-b] (6.2): ``H^1 = theta_3(q^2)`` at ``c=3/2``, all ``Delta_i = 1/8``.

    Delta-INDEPENDENT, which is a strong statement: every pole contribution must
    cancel against the others at every level, for any exchanged weight.
    """
    with mp.workdps(CLOSED_FORM_DPS):
        got = ns.H_q_series_ns(
            mp.mpf(delta_str),
            _eighth_externals(),
            c=_near_free_c(),
            ns_order=order,
            sector=ns.SECTOR_EVEN,
        )
        want = ns.theta_half_series(3, order)
        err = max(abs(g - w) for g, w in zip(got, want))
        assert err < mp.mpf(10) ** (-CLOSED_FORM_EPS_EXP + 6), mp.nstr(err, 8)


@pytest.mark.parametrize("order", [30])  # the lower orders are prefixes of this series
def test_odd_sector_matches_theta2_over_delta_closed_form(order):
    """[HJS-b] (6.2): ``H^(1/2) = theta_2(q^2)/Delta``.

    The ``1/Delta`` is the ``(1, 1)`` pole at ``Delta_11 = 0``; this is the test
    that caught its omission.
    """
    with mp.workdps(CLOSED_FORM_DPS):
        delta = mp.mpf("0.37")
        got = ns.H_q_series_ns(
            delta,
            _eighth_externals(),
            c=_near_free_c(),
            ns_order=order,
            sector=ns.SECTOR_ODD,
        )
        want = [w / delta for w in ns.theta_half_series(2, order)]
        err = max(abs(g - w) for g, w in zip(got, want))
        assert err < mp.mpf(10) ** (-CLOSED_FORM_EPS_EXP + 6), mp.nstr(err, 8)


def test_closed_form_residual_scales_with_eps():
    """The residual is the ``c -> 3/2`` regularization, not a transcription error.

    Ten more digits of ``eps`` must buy ~ten more digits of agreement. A residual
    that plateaued would mean the closed form is only approximately reproduced,
    which is exactly the failure mode this ladder exists to exclude.
    """
    with mp.workdps(CLOSED_FORM_DPS):
        errs = []
        for eps_exp in (20, 30):
            got = ns.H_q_series_ns(
                mp.mpf("0.37"),
                _eighth_externals(),
                c=_near_free_c(eps_exp),
                ns_order=12,
                sector=ns.SECTOR_EVEN,
            )
            want = ns.theta_half_series(3, 12)
            errs.append(max(abs(g - w) for g, w in zip(got, want)))
        assert errs[1] < errs[0] * mp.mpf(10) ** -8


def test_ns_starred_rule_is_pinned_by_closed_forms():
    """Only ``starred = (rs mod 2) XOR sector`` reproduces BOTH closed forms.

    The papers' compressed ``H^{1,1/2}`` notation leaves this ambiguous, so it
    was settled numerically. This test freezes that result and, importantly,
    records that the three alternatives fail LOUDLY (relative errors of order 1
    to 1e7) rather than marginally -- so the rule is not a fitted choice among
    near-ties.
    """
    with mp.workdps(50):
        c = _near_free_c(25)
        h_ext = _eighth_externals()
        order = 12
        delta = mp.mpf("0.37")
        rules = {
            "rs": lambda rs, sec: rs % 2,
            "not_rs": lambda rs, sec: 1 - rs % 2,
            "rs_xor_sector": lambda rs, sec: (rs % 2) ^ sec,
            "rs_xor_not_sector": lambda rs, sec: (rs % 2) ^ (1 - sec),
        }
        want_even = ns.theta_half_series(3, order)
        want_odd = [w / delta for w in ns.theta_half_series(2, order)]

        errors = {}
        for name, rule in rules.items():
            fam = ns.NSFamily(c, h_ext)
            entries = []
            for sector in (0, 1):
                for (r, s, rs) in fam.enumerate_poles(order):
                    starred = bool(rule(rs, sector))
                    d = ns.ns_degenerate_weight(r, s, c)
                    a1, a2, a3, a4 = fam.alpha
                    K = (
                        ns._A_rs(r, s, fam.beta)
                        * ns._fusion_polynomial(
                            r, s, fam.beta, a4, a3, starred=starred
                        )
                        * ns._fusion_polynomial(
                            r, s, fam.beta, a1, a2, starred=starred
                        )
                        * mp.mpf(4) ** rs
                    )
                    entries.append(
                        dict(r=r, s=s, rs=rs, K=K, d=d,
                             v=d + mp.mpf(rs) / 2, sector=sector)
                    )
            entries.sort(key=lambda p: (p["rs"], p["sector"]))
            weights = vpoles.interaction_weights(
                entries, range(len(entries)), mp.mpc, couples=fam.couples
            )
            seeds = {s: fam.seed(s, order, mp.mpc) for s in (0, 1)}
            table = vpoles.fill_level_table(entries, weights, order, seeds)
            got = {
                sec: vpoles.eval_level_series(
                    mp.mpc(delta), entries, table, order,
                    fam.seed(sec, order, mp.mpc), mp.mpc,
                    couples=fam.couples, sector=sec,
                )
                for sec in (0, 1)
            }
            errors[name] = max(
                max(abs(g - w) for g, w in zip(got[0], want_even)),
                max(abs(g - w) for g, w in zip(got[1], want_odd)),
            )

        assert errors["rs_xor_sector"] < mp.mpf(10) ** -18
        for rejected in ("rs", "not_rs", "rs_xor_not_sector"):
            assert errors[rejected] > 1, f"{rejected} -> {mp.nstr(errors[rejected], 4)}"


def test_dropping_the_11_pole_kills_the_odd_leading_term():
    """``(1,1)`` is required: without it ``H^(1/2)`` loses its whole ``q^(1/2)`` term.

    Guards against "tidying up" the enumeration back to the ``1 < rs`` of
    [HJS-a] (2.3), which applies to a different expansion.
    """
    with mp.workdps(50):
        labels = list(ns.enumerate_ns_poles(12))
        assert (1, 1, 1) in labels
        # rs = 2 is impossible: r + s would be odd.
        assert not any(rs == 2 for (_, _, rs) in labels)


# --------------------------------------------------------------------------
# Parametrization and pole-set structure
# --------------------------------------------------------------------------


def test_beta_solves_its_defining_relation():
    """``chat = 1 - 2 (beta - 1/beta)^2`` for the beta of [HJS-a] (3.6)."""
    with mp.workdps(40):
        for c in (mp.mpf(3) / 2, mp.mpf(81) / 70, mp.mpf(7) / 10, mp.mpf(5) / 4):
            beta = ns.beta_from_c(c)
            assert mp.almosteq(
                1 - 2 * (beta - 1 / beta) ** 2, ns.chat_from_c(c), abs_eps=mp.mpf(10) ** -30
            )


def test_alpha_delta_round_trip():
    with mp.workdps(40):
        c = mp.mpf(81) / 70
        for d in (mp.mpf(0), mp.mpf(3) / 35, mp.mpf("1.7")):
            alpha = mp.sqrt(ns.alpha_sq_from_delta(d, c))
            assert mp.almosteq(ns.delta_from_alpha(alpha, c), d, abs_eps=mp.mpf(10) ** -30)


def test_degenerate_weight_matches_expanded_kac_formula():
    """Cross-check (3.5) computed via ``alpha_rs`` against its expanded form."""
    with mp.workdps(40):
        c = mp.mpf(81) / 70
        beta = ns.beta_from_c(c)
        for (r, s) in ((1, 1), (1, 3), (2, 2), (3, 3), (2, 4)):
            expanded = (
                -Fraction(r * s - 1, 4)
                + (r * r - 1) * beta**2 / 8
                + (s * s - 1) / (8 * beta**2)
            )
            assert mp.almosteq(
                ns.ns_degenerate_weight(r, s, c), expanded, abs_eps=mp.mpf(10) ** -28
            )


def test_delta_11_vanishes_identically_in_c():
    """``Delta_11 = 0`` for every ``c`` -- the vacuum pole of the odd sector."""
    with mp.workdps(40):
        for c in (mp.mpf(3) / 2, mp.mpf(81) / 70, mp.mpf(7) / 10, mp.mpf(11)):
            assert abs(ns.ns_degenerate_weight(1, 1, c)) < mp.mpf(10) ** -30


def test_pole_enumeration_is_ascending_and_parity_correct():
    labels = list(ns.enumerate_ns_poles(24))
    assert all((r + s) % 2 == 0 for (r, s, _) in labels)
    assert all(r * s == rs for (r, s, rs) in labels)
    assert [rs for (_, _, rs) in labels] == sorted(rs for (_, _, rs) in labels)
    # Levels present: odd >= 1, and multiples of 4 (both r, s even).
    for (_, _, rs) in labels:
        assert rs % 2 == 1 or rs % 4 == 0


def test_sector_parity_structure():
    """The even sector has only integer ``q`` powers, the odd only half-integer.

    A term in ``H^t`` comes from a chain of poles returning to sector ``t``, so
    it contains an EVEN number of odd-level poles; the total level is therefore
    even for ``t = even`` and odd for ``t = odd``. Together with the seeds
    (``theta_3(q^2)`` is even in ``t``, ``g^(1/2) = 0``) this forces a strict
    parity split -- which is exactly the papers' statement that ``F^1`` is a
    series in integer powers of ``x`` and ``F^(1/2)`` in half-integer ones.

    A cheap, sharp, generic-``c`` structural check that no closed form is
    involved in: it would break under almost any error in the sector coupling.
    """
    with mp.workdps(40):
        order = 14
        h_ext = [mp.mpf(1) / 10, mp.mpf(2) / 10, mp.mpf(3) / 10, mp.mpf(1) / 20]
        even = ns.H_q_series_ns(
            mp.mpf("0.6"), h_ext, c=GENERIC_C, ns_order=order,
            sector=ns.SECTOR_EVEN,
        )
        odd = ns.H_q_series_ns(
            mp.mpf("0.6"), h_ext, c=GENERIC_C, ns_order=order,
            sector=ns.SECTOR_ODD,
        )
        for k in range(order + 1):
            if k % 2:
                assert even[k] == 0, k
            else:
                assert odd[k] == 0, k
        # And the poles really do contribute: the even sector must differ from
        # its bare theta_3(q^2) seed.
        seed = ns.theta_half_series(3, order)
        assert max(abs(g - w) for g, w in zip(even, seed)) > mp.mpf(10) ** -20
        assert abs(odd[1]) > mp.mpf(10) ** -20


def test_degenerate_c_is_the_exact_c_limit():
    """At a minimal-model c the Kac poles collide (``c = 81/70``, SM(5): ``beta^2 = 7/5``, ``A_rs`` divides by zero
    once ``rs >= 35``; ``c = 7/10``: ``beta^2 = 5/4``).  The bare recursion raised there; it now runs exactly as a
    Laurent series in ``t`` (:mod:`virasoro_blocks.super_laurent`), the c -> c0 limit, which the blocks at
    ``c0 + 10^-40`` (carried at 200 digits) approach."""
    with mp.workdps(40):
        for c0, ext, order in ((mp.mpf(81) / 70, [mp.mpf(3) / 35] * 4, 40), (mp.mpf(7) / 10, [mp.mpf(1) / 10] * 4, 25)):
            got = ns.H_q_series_ns(mp.mpf("0.6"), ext, c=c0, ns_order=order)
            with mp.workdps(200):
                ref = ns.H_q_series_ns(mp.mpf("0.6"), ext, c=c0 + mp.mpf(10) ** -40, ns_order=order)
            scale = max(abs(x) for x in ref)
            assert max(abs(a - b) for a, b in zip(got, ref)) < mp.mpf(10) ** -35 * scale


def test_eps_offset_makes_degenerate_c_usable():
    """The documented workaround: evaluate at ``c* + eps`` and converge."""
    with mp.workdps(60):
        h_ext = [mp.mpf(3) / 35] * 4
        vals = []
        for eps_exp in (25, 30):
            c = mp.mpf(81) / 70 + mp.mpf(10) ** (-eps_exp)
            vals.append(
                ns.H_q_series_ns(mp.mpf("0.6"), h_ext, c=c, ns_order=40)[4]
            )
        assert abs(vals[1] - vals[0]) < mp.mpf(10) ** -15


def test_half_step_order_conversion():
    assert ns.ns_order_for_q_order(30) == 60
    with pytest.raises(ValueError):
        ns.ns_order_for_q_order(-1)


def test_level_unit_is_one_half():
    fam = ns.NSFamily(mp.mpf(81) / 70, [mp.mpf(0)] * 4)
    assert fam.level_unit == Fraction(1, 2)
    assert fam.n_sectors == 2


# --------------------------------------------------------------------------
# The assembled block
# --------------------------------------------------------------------------


def test_F_block_ns_leading_z_behaviour():
    """``F ~ z^(Delta + z_shift) / N^Delta`` as ``z -> 0``, as for Virasoro.

    Confirms the prefactor keeps the repo's ``chiral_norm`` / ``z_shift``
    conventions, which is what lets downstream crossing code stay NS-agnostic.
    """
    with mp.workdps(40):
        c = GENERIC_C
        delta = mp.mpf("0.43")
        h_ext = [mp.mpf(3) / 35] * 4
        N = mp.mpf(16)
        for z in (mp.mpf(10) ** -6, mp.mpf(10) ** -7):
            F = ns.F_block_ns(delta, z, h_ext, c=c, ns_order=8)
            lead = z**delta / N**delta
            assert mp.almosteq(F / lead, 1, abs_eps=mp.mpf(10) ** -5)


def test_F_block_ns_converges_in_ns_order():
    with mp.workdps(50):
        c = GENERIC_C
        h_ext = [mp.mpf(3) / 35] * 4
        z = mp.mpf(1) / 2
        vals = [
            ns.F_block_ns(mp.mpf("0.43"), z, h_ext, c=c, ns_order=n)
            for n in (16, 24, 32)
        ]
        assert abs(vals[2] - vals[1]) < abs(vals[1] - vals[0]) / 10


def test_ns_table_is_delta_independent_and_cached():
    """One Kac/table build must serve every exchanged weight."""
    with mp.workdps(40):
        c = GENERIC_C
        h_ext = [mp.mpf(3) / 35] * 4
        a = ns.ns_level_table(c, h_ext, 12)
        b = ns.ns_level_table(c, h_ext, 12)
        assert a is b


def test_sector_argument_is_validated():
    with mp.workdps(30):
        with pytest.raises(ValueError):
            ns.H_q_series_ns(
                mp.mpf("0.5"), [mp.mpf(0)] * 4, c=mp.mpf(81) / 70,
                ns_order=4, sector=7,
            )
        with pytest.raises(ValueError):
            ns.H_q_series_ns(
                mp.mpf("0.5"), [mp.mpf(0)] * 4, c=mp.mpf(81) / 70, ns_order=-1
            )
        with pytest.raises(ValueError):
            ns.NSFamily(mp.mpf(81) / 70, [mp.mpf(0)] * 3)


# --------------------------------------------------------------------------
# The shared kernel
# --------------------------------------------------------------------------


def test_kernel_reproduces_virasoro_recursion():
    """``poles.py`` with a constant seed IS the Virasoro q-recursion.

    Guards the refactor: ``recursion.py`` delegates its scalar fill here, so a
    divergence between the two would be a silent regression of the production
    Virasoro path.
    """
    from virasoro_blocks.recursion import H_q_series, _build_rs_data, \
        _pair_interaction_weights, _accumulator_field

    with mp.workdps(40):
        c = mp.mpf(30)  # real_b mode: no strip-twin canonicalization
        h_ext = tuple(mp.mpf(1) / 5 for _ in range(4))
        q_order = 10
        from virasoro_blocks.params import b_from_c
        b = b_from_c(c)
        pair_table = _build_rs_data(q_order, b, h_ext)
        weights = _pair_interaction_weights(pair_table)
        field = _accumulator_field(pair_table.pairs)
        seed = [field(0)] * (q_order + 1)
        seed[0] = field(1)
        table = vpoles.fill_level_table(
            pair_table.pairs, weights, q_order, {0: seed},
            outer_indices=pair_table.canonical_indices,
        )
        delta = mp.mpf("0.8")
        got = vpoles.eval_level_series(
            delta, pair_table.pairs, table, q_order, seed, field
        )
        want = H_q_series(delta, h_ext, c=c, q_order=q_order)
        for g, w in zip(got, want):
            assert mp.almosteq(g, w, rel_eps=mp.mpf(10) ** -25)


def test_kernel_couples_mask_suppresses_forbidden_pairs():
    """A ``couples`` returning False must zero the weight exactly."""
    with mp.workdps(30):
        entries = [
            dict(rs=1, K=mp.mpf(2), d=mp.mpf(0), v=mp.mpf(1) / 2, sector=0),
            dict(rs=1, K=mp.mpf(2), d=mp.mpf(0), v=mp.mpf(1) / 2, sector=1),
        ]
        weights = vpoles.interaction_weights(
            entries, range(2), mp.mpf,
            couples=lambda o, i: o["sector"] == i["sector"],
        )
        assert weights[0][1] == 0
        assert weights[1][0] == 0


def test_kernel_flint_cache_distinguishes_seed_and_routing_inputs():
    from types import SimpleNamespace
    from virasoro_blocks.polynomial import _mpf_to_acb

    with mp.workdps(40):
        entries = [dict(rs=1, K=mp.mpf(1), d=mp.mpf(0), v=mp.mpf(1), sector=0),
                   dict(rs=2, K=mp.mpf(2), d=mp.mpf(-1), v=mp.mpf(1), sector=0)]
        # The exact inner rows of this toy recursion, derived by counting its
        # level-1 and level-2 compositions. No numerical fill is the oracle.
        inner = [[mp.mpf(k) for k in (1, 1, 2, 3)],
                 [mp.mpf(k) for k in (1, 1, 2)]]
        table = vpoles.LevelTable(entries, inner, 4, SimpleNamespace(level_unit=1))
        seed = [mp.mpf(k) for k in (1, 0, 0, 0, 0)]
        changed = [mp.mpc(k) for k in (2, 3, 0, 0, 0)]
        switch = {"enabled": True}
        mutable_rule = lambda o, i: switch["enabled"]
        point_rule = lambda o, i: o["v"] > 2
        cases = [(2, seed, None), (2, changed, None),
                 (2, seed, lambda o, i: False), (2, seed, mutable_rule),
                 (2, seed, point_rule), (3, seed, point_rule)]
        for x, top_seed, rule in cases:
            result = vpoles.level_series_acb(table, mp.mpf(x), 4, top_seed, couples=rule)
            with mp.workdps(100):
                expected = [Fraction(int(mp.re(v))) for v in top_seed]
                for k in range(1, 5):
                    for j, pole in enumerate(entries):
                        if pole["rs"] <= k and (rule is None or rule({"sector": 0, "v": x}, pole)):
                            expected[k] += Fraction(int(pole["K"]), x-int(pole["d"])) * int(inner[j][k-pole["rs"]])
                assert all(got.contains(_mpf_to_acb(mp.convert(want))) for got, want in zip(result, expected))
        switch["enabled"] = False
        assert vpoles.level_series_acb(table, mp.mpf(2), 4, seed, couples=mutable_rule) == [
            _mpf_to_acb(value) for value in seed]


@pytest.mark.parametrize("sector", [0, 1])
def test_exact_c32_laurent_path_matches_closed_forms(sector):
    """The Laurent recursion exactly at ``c = 3/2`` (not ``3/2 + eps``) reproduces [HJS-b] (6.2)."""
    from virasoro_blocks.super_laurent import singular_H

    with mp.workdps(60):
        delta = mp.mpf("0.37")
        got = singular_H("ns", delta, [Fraction(1, 8)] * 4, c0=Fraction(3, 2), order=20, sector=sector)
        want = ns.theta_half_series(3 if sector == 0 else 2, 20)
        if sector == 1:
            want = [w / delta for w in want]
        assert max(abs(g - w) for g, w in zip(got, want)) < mp.mpf(10) ** -55
