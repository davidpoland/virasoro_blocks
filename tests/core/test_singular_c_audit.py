"""Independent edge checks for singular-c caches, recognition and verification."""

from fractions import Fraction
import subprocess
import sys

import mpmath as mp
import pytest


@pytest.mark.parametrize("coordinate", ["zzb", "yyb", "xt", "ws", "qqb", "qqb_ws"])
def test_batched_derivatives_preserve_tiny_exact_c_shift(coordinate):
    from virasoro_blocks import derivs
    from virasoro_blocks.numeric_input import parse_number

    c = parse_number("0.70000000000000000000000000000000000000000000000000000000000000000001")
    builder = getattr(derivs, f"G_{coordinate}_table_many")
    with mp.workdps(35):
        low = builder([mp.mpf("1.4")], 0, [mp.mpf(".125")] * 4, 1, c=c, q_order=16)
    with mp.workdps(100):
        high = builder([mp.mpf("1.4")], 0, [mp.mpf(".125")] * 4, 1, c=c, q_order=16)
        for row, reference in zip(low[0], high[0]):
            for value, expected in zip(row, reference):
                assert abs(value - expected) <= mp.mpf("1e-30") * max(1, abs(expected))


@pytest.mark.parametrize("family", ["ns", "ramond"])
@pytest.mark.parametrize("c0", [Fraction(7, 10), Fraction(81, 70), Fraction(3, 2), Fraction(27, 2)])
def test_singular_series_matches_independent_near_c_recursion(family, c0):
    from virasoro_blocks.recursion_ns import H_q_series_ns
    from virasoro_blocks.recursion_r import H_q_series_r

    fn = H_q_series_ns if family == "ns" else H_q_series_r
    with mp.workdps(30):
        ext = [mp.mpf(k) / 8 for k in (1, 2, 3, 4)]
        h = mp.mpf("0.37")
        for sector in (0, 1):
            for branch in ([0] if family == "ns" else range(4)):
                kw = {"ns_order" if family == "ns" else "r_order": 10, "sector": sector}
                if family == "ramond":
                    kw["branch"] = branch
                got = fn(h, ext, c=c0, **kw)
                with mp.workdps(180):
                    ref = fn(h, ext, c=mp.convert(c0) + mp.mpf("1e-50"), **kw)
                assert max(abs(g - r) / max(1, abs(r)) for g, r in zip(got, ref)) < mp.mpf("1e-28")


@pytest.mark.parametrize("family", ["ns", "ramond"])
def test_merged_poles_do_not_depend_on_previous_higher_order_build(family, monkeypatch):
    from virasoro_blocks import super_laurent as sl

    with mp.workdps(30):
        ext = list(map(mp.mpf, ["0.1", "0.2", "0.3", "0.4"]))
        monkeypatch.setattr(sl, "_MERGED", {})
        sl.merged_state(family, ext, c0=Fraction(3, 2), order=8)
        warm = sl.merged_state(family, ext, c0=Fraction(3, 2), order=2)
        monkeypatch.setattr(sl, "_MERGED", {})
        cold = sl.merged_state(family, ext, c0=Fraction(3, 2), order=2)
        assert sl.merged_pole_pool(warm) == sl.merged_pole_pool(cold)
        for h in (mp.mpf("0.37"), mp.mpf("0.5")):
            assert sl.merged_H(warm, h) == sl.merged_H(cold, h)


@pytest.mark.parametrize("family", ["ns", "ramond"])
def test_singular_series_keeps_precision_close_to_a_rational_pole(family):
    from virasoro_blocks import super_laurent as sl

    with mp.workdps(30):
        ext = [mp.mpf(k) / 8 for k in (1, 2, 3, 4)]
        h = mp.mpf(1) / 10 + mp.mpf("1e-23")
        state = sl.merged_state(family, ext, c0=Fraction(7, 10), order=10)
        got = sl.merged_H(state, h)
        direct = sl.singular_H(family, h, ext, c0=Fraction(7, 10), order=10)
        # Independent evaluation of the Laurent table at the supplied weight,
        # retaining it exactly while raising the arithmetic precision.
        with mp.workdps(100):
            ref = sl.singular_H(family, h, ext, c0=Fraction(7, 10), order=10)
        assert max(abs(g - r) / max(1, abs(r)) for g, r in zip(got, ref)) < mp.mpf("1e-28")
        assert max(abs(g - r) / max(1, abs(r)) for g, r in zip(direct, ref)) < mp.mpf("1e-28")


def test_small_genuine_higher_order_poles_are_not_pruned_as_noise():
    from virasoro_blocks import super_laurent as sl

    with mp.workdps(30):
        ext = [mp.mpf(1) / 8] * 4
        ext[0] += mp.mpf("1e-20")
        ext[2] += mp.mpf("2e-20")
        h = mp.mpf("1e-50")
        state = sl.merged_state("ns", ext, c0=Fraction(3, 2), order=8)
        got = sl.merged_H(state, h)
        with mp.workdps(120):
            ref = sl.singular_H("ns", h, ext, c0=Fraction(3, 2), order=8)
        assert max(abs(g - r) / max(1, abs(r)) for g, r in zip(got, ref)) < mp.mpf("1e-28")
        assert sum(p == 0 for p in sl.merged_pole_pool(state)) == 2


def test_tracked_series_retains_small_provably_nonzero_leading_terms():
    from virasoro_blocks.laurent_levels import Tracked

    with mp.workdps(30):
        tiny = mp.mpf("1e-24")
        series = Tracked.const(tiny) + Tracked.monomial(2)
        assert abs(series.reciprocal().s.coeff(0) * tiny - 1) < mp.mpf("1e-28")
        assert abs(series.sqrt().s.coeff(0) / mp.sqrt(tiny) - 1) < mp.mpf("1e-28")


@pytest.mark.parametrize("external", [Fraction(1, 10), Fraction(3, 80)])
def test_minimal_model_taylor_grid_certifies_its_laurent_coefficients(external):
    from virasoro_blocks.derivs import G_zzb_table

    kwargs = dict(c=Fraction(7, 10), q_order=20, z_shift=-2 * external)
    with mp.workdps(15):
        got = G_zzb_table(0, 0, [external] * 4, 7, 7, **kwargs)
    with mp.workdps(60):
        ref = G_zzb_table(0, 0, [external] * 4, 7, 7, **kwargs)
        assert max(abs(x - y) / max(1, abs(y)) for gr, rr in zip(got, ref)
                   for x, y in zip(gr, rr)) < mp.mpf("1e-12")


def test_merged_finite_part_and_all_pole_orders_match_a_weight_contour():
    from virasoro_blocks import super_laurent as sl

    with mp.workdps(60):
        ext = [mp.mpf(k) / 8 for k in (1, 2, 3, 4)]
        delta = mp.mpf(1) / 2
        state = sl.merged_state("ns", ext, c0=Fraction(3, 2), order=8)
        finite, principal = sl.merged_laurent_at(state, delta)
        assert set(principal) == {1, 2, 3, 4}
        contour = {a: [mp.mpc(0)] * 9 for a in (0, *principal)}
        for j in range(12):
            eps = mp.mpf("1e-4") * mp.expjpi(mp.mpf(j) / 6)
            values = sl.singular_H("ns", delta + eps, ext, c0=Fraction(3, 2), order=8)
            for a, row in contour.items():
                for k, value in enumerate(values):
                    row[k] += value * eps**a / 12
        for a, expected in [(0, finite), *principal.items()]:
            assert max(abs(v - e) / max(1, abs(e)) for v, e in zip(contour[a], expected)) < mp.mpf("1e-38")


def test_character_tail_check_terminates_when_subtraction_would_stall():
    # Bound execution time when additional terms no longer change the sum.
    subprocess.run([sys.executable, "-c", """
import mpmath as mp
from virasoro_blocks.torus_onept import irreducible_character_factor
mp.mp.dps = 15
assert irreducible_character_factor(mp.mpf(3), mp.mpf('0.37'), mp.mpf('.7')) == (1, False)
"""], check=True, timeout=5)


def test_character_near_a_kac_weight_remains_a_verma_character():
    from virasoro_blocks.torus_onept import irreducible_character_factor

    with mp.workdps(30):
        q = mp.mpf("0.001")
        h = mp.mpf(1) / 4 + mp.mpf("1e-18")
        assert irreducible_character_factor(mp.mpf(1), h, q) == (1, False)
        factor, degenerate = irreducible_character_factor(mp.mpf(1), mp.mpf(1) / 4, q)
        assert degenerate
        assert abs(factor - (1 - q**2)) < mp.mpf("1e-28")


def test_generic_character_near_vacuum_remains_a_verma_character():
    from virasoro_blocks.torus_onept import irreducible_character_factor

    with mp.workdps(30):
        assert irreducible_character_factor(mp.mpf(3), mp.mpf("1e-18"), mp.mpf("0.001")) == (1, False)


@pytest.mark.parametrize("c", ["0.50000001", "0.70000001"])
def test_raising_precision_does_not_snap_a_deliberately_nearby_c(c):
    from virasoro_blocks.singularity import exact_locus_c
    from virasoro_blocks.torus_onept import irreducible_character_factor

    with mp.workdps(15):
        c = mp.mpf(c)
    with mp.workdps(30):
        assert exact_locus_c(c) is None
        q = mp.mpf("0.001")
        value, degenerate = irreducible_character_factor(c, mp.mpf(0), q)
        assert degenerate and value == 1 - q


def test_nearby_external_cannot_bypass_torus_null_decoupling():
    from virasoro_blocks.torus_onept import TorusOnePtDecouplingError, irreducible_onept_F_series

    with mp.workdps(30):
        with pytest.raises(TorusOnePtDecouplingError):
            irreducible_onept_F_series(Fraction(1, 4), 1, 1 + mp.mpf("1e-18"), 4)
        assert irreducible_onept_F_series(Fraction(1, 4), 1, 1, 4, exact=True) == [1, 1, 1, 2, 3]


@pytest.mark.parametrize("q", ["1", "-1", "2", "nan", "inf"])
def test_character_refuses_nonconvergent_or_nonfinite_nome(q):
    from virasoro_blocks.torus_onept import irreducible_character_factor

    with pytest.raises(ValueError, match="nome"):
        irreducible_character_factor(1, 0, mp.mpf(q))


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_nonfinite_near_singular_results_cannot_pass_verification(bad, monkeypatch):
    from virasoro_blocks import singularity as s

    monkeypatch.setattr(s, "_NEAR_GUARD_CACHE", {})
    with mp.workdps(20), pytest.raises(ValueError):
        s.verified_guard_compute(("invalid", mp.mp.prec), lambda: [mp.mpf(bad)],
                                 what="audit", hint="invalid numerical result")


def test_cached_near_singular_guard_still_refuses_nonfinite_values(monkeypatch):
    from virasoro_blocks import singularity as s

    monkeypatch.setattr(s, "_NEAR_GUARD_CACHE", {})
    key = ("finite_then_nan", mp.mp.prec)
    s.verified_guard_compute(key, lambda: [mp.mpf(1)], what="audit", hint="retry")
    with pytest.raises(ValueError, match="non-finite"):
        s.verified_guard_compute(key, lambda: [mp.nan], what="audit", hint="retry")
