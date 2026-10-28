"""Exact NS/R inputs distinguish genuine poles, fusion zeros, and nearby values."""

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import literal_number, parse_number
from virasoro_blocks.recursion_ns import NSFamily, F_block_ns, alpha_sq_from_delta, beta_from_c
from virasoro_blocks.recursion_r import RFamily, F_block_r, ramond_parameter_from_delta
from virasoro_blocks.superconformal_blocks import (
    SuperconformalPoleError, _chiral_poles, _pole_family,
    ns_chiral_block, ramond_chiral_block, numeric_chiral_table,
    numeric_full_table, super_ordered_B_block,
)


HE = (".1", ".2", ".3", ".4")


def test_auto_policy_retains_nearby_native_c_instead_of_using_a_candidate_offset():
    from virasoro_blocks.superconformal_blocks import is_superconformal_singular_c, resolve_super_c

    with mp.workdps(90):
        near = mp.mpf('1.5') + mp.mpf('1e-60')
        complex_near = mp.mpc('1.5', '1e-60')
        rounded_locus = mp.mpf(81) / 70
    for dps in (30, 90):
        with mp.workdps(dps):
            for c in (near, complex_near, Fraction(3, 2) + Fraction(1, 10**60)):
                assert not is_superconformal_singular_c(c)
                c_input, c_eval, regularized = resolve_super_c(c, eps_exp=8)
                assert not regularized
                assert c_eval == c_input
            assert is_superconformal_singular_c(rounded_locus)
            assert is_superconformal_singular_c('1.5')
    with mp.workdps(30):
        for fn in (ns_chiral_block, ramond_chiral_block):
            kw = dict(c=near, q_order=2, eps_exp=8)
            direct = fn('.7', '.3', HE, regularize='off', **kw)
            auto = fn('.7', '.3', HE, regularize='auto', **kw)
            assert auto == direct


@pytest.mark.parametrize('family', ['ns', 'ramond'])
def test_laurent_identity_location_does_not_hide_a_tiny_nonzero_weight(family):
    from virasoro_blocks.super_laurent import singular_H, SuperLaurentPoleError

    ext = tuple(Fraction(k, 9) for k in range(1, 5))
    with mp.workdps(30):
        weight = mp.mpf(2) ** -400
        found = singular_H(family, weight, ext, c0=Fraction(3, 2), order=4)
        assert all(mp.isfinite(value) for value in found)
        assert abs(found[2] + mp.mpf(8) / (81 * weight)) < mp.mpf('1e-28') * abs(found[2])
        with pytest.raises(SuperLaurentPoleError, match='genuine pole'):
            singular_H(family, mp.mpf(0), ext, c0=Fraction(3, 2), order=4)
    with mp.workdps(240):
        reference = singular_H(family, Fraction(1, 2**400), ext, c0=Fraction(3, 2), order=4)
        assert all(abs(actual - expected) < mp.mpf('1e-28') * max(1, abs(expected))
                   for actual, expected in zip(found, reference))


@pytest.mark.parametrize("sign", [-1, 1])
def test_small_momentum_squares_survive_affine_input_cancellation(sign):
    eps = Fraction(sign, 10**70)
    with mp.workdps(40):
        alpha = alpha_sq_from_delta(Fraction(1, 48)+eps, 2)
        close(alpha, mp.convert(8*eps))
        gamma = ramond_parameter_from_delta(Fraction(1, 12)+eps, 2)
        close(gamma*gamma, mp.convert(-eps))
        assert mp.mp.dps == 40


def test_beta_square_roots_keep_a_deliberate_c_offset_below_the_caller_precision():
    c = Fraction(3, 2)+Fraction(1, 10**70)
    with mp.workdps(180):
        reference = beta_from_c(c)
    with mp.workdps(40):
        value = beta_from_c(c)
        assert value.imag != 0
        close(value.imag, reference.imag)
        assert mp.mp.dps == 40


def close(actual, expected, digits=37):
    with mp.workdps(200):
        assert abs(actual-expected) < mp.mpf(10)**-digits * abs(expected)


@pytest.mark.parametrize("family", ["ns", "ramond"])
@pytest.mark.parametrize("api", ["primitive", "chiral", "ordered", "chiral_table", "full_table"])
@pytest.mark.parametrize("rational_text", [False, True])
def test_decimal_and_rational_strings_agree_with_exact_fraction_inputs(family, api, rational_text):
    def run(convert):
        ext = [convert(h) for h in HE]
        kw = dict(c=convert("30"), q_order=2, chiral_norm=convert("3/7"), z_shift=convert("-1/5"))
        if api == "primitive":
            fn = F_block_ns if family == "ns" else F_block_r
            return [[fn(convert("1.75"), convert(".3"), ext, **kw)]]
        if api == "chiral":
            fn = ns_chiral_block if family == "ns" else ramond_chiral_block
            return [[fn(convert("1.75"), convert(".3"), ext, **kw)]]
        if api == "ordered":
            return [[super_ordered_B_block(family, convert("1.75"), convert(".75"),
                                           convert(".3"), convert(".45"), ext, **kw)]]
        if api == "chiral_table":
            return numeric_chiral_table(family, convert("1.75"), "z", ext, m_max=1, center=convert(".3"), **kw)
        return numeric_full_table(family, "ordered_B", convert("2.5"), 1, "zzb", ext,
                                  m_max=1, n_max=1, center=convert(".3"), center_bar=convert(".45"), **kw)

    text = lambda value: str(Fraction(value)) if rational_text else value
    with mp.workdps(40):
        actual, expected = run(text), run(parse_number)
        for row, ref_row in zip(actual, expected):
            for value, ref in zip(row, ref_row):
                close(value, ref)
        assert mp.mp.dps == 40


@pytest.mark.parametrize("family", ["ns", "ramond"])
@pytest.mark.parametrize("location", ["origin", "quarter"])
@pytest.mark.parametrize("table", [False, True])
def test_exact_genuine_kac_poles_are_still_refused(family, location, table):
    h, c = ("0", "30") if location == "origin" else ("1/4", "-1/2")
    with mp.workdps(40), pytest.raises(SuperconformalPoleError):
        if table:
            numeric_chiral_table(family, h, "z", HE, c=c, q_order=3, m_max=1)
        else:
            fn = ns_chiral_block if family == "ns" else ramond_chiral_block
            fn(h, ".3", HE, c=c, q_order=3)


@pytest.mark.parametrize("family", ["ns", "ramond"])
@pytest.mark.parametrize("shifted", [False, True])
def test_proven_fusion_zeros_are_removed_but_tiny_fusion_shifts_keep_the_pole(family, shifted):
    labels = ("1/4", "0", ".1", ".2") if family == "ns" else ("5/16", "1/16", ".1", ".2")
    ext = list(map(parse_number, labels))
    if shifted:
        ext[0] += Fraction(1, 10**70)
    with mp.workdps(120):
        cls = NSFamily if family == "ns" else RFamily
        row = cls(parse_number("7.5"), ext).build_pole((2, 2, 4), 0)
        assert (row["K"] != 0) == shifted
        rows = _chiral_poles(family, parse_number("7.5"), ext, q_order=4, sector=0, branch="++", with_labels=True)
        assert any(label == (2, 2, 4) for label, _pole in rows) == shifted
    fn = ns_chiral_block if family == "ns" else ramond_chiral_block
    if shifted:
        with mp.workdps(40), pytest.raises(SuperconformalPoleError):
            fn("-.75", ".3", ext, c="7.5", q_order=3)
    else:
        with mp.workdps(180):
            reference = fn("-.75", ".3", ext, c="7.5", q_order=3)
        with mp.workdps(40):
            actual = fn("-.75", ".3", ext, c="7.5", q_order=3)
            close(actual, reference)


@pytest.mark.parametrize("family", ["ns", "ramond"])
def test_cached_pole_families_distinguish_stored_parameters_below_the_caller_precision(family):
    with mp.workdps(180):
        original = mp.mpc("1.3")
        shifted = original+mp.mpf("1e-70")
    with mp.workdps(15):
        first, second = (_pole_family(family, c, HE) for c in (original, shifted))
        assert first is not second
        assert literal_number(first.c.real) != literal_number(second.c.real)


@pytest.mark.parametrize("cls", [NSFamily, RFamily])
def test_pole_family_rebuilds_derived_parameters_after_precision_changes(cls):
    with mp.workdps(20):
        family = cls(30, HE)
        family.build_pole((2, 2, 4), 0)
    with mp.workdps(180):
        reference = cls(30, HE).build_pole((2, 2, 4), 0)["K"]
    for dps in (80, 30, 80):
        with mp.workdps(dps):
            value = family.build_pole((2, 2, 4), 0)["K"]
            close(value, reference, digits=dps-5)
            assert mp.mp.dps == dps


def test_ramond_odd_block_keeps_relative_accuracy_when_external_fusion_is_tiny():
    ext = list(map(parse_number, (".1", ".1", ".3", ".4")))
    ext[1] += Fraction(1, 10**70)
    with mp.workdps(180):
        reference = ramond_chiral_block("1.2", ".3", ext, c=30, q_order=2, sector="odd")
    with mp.workdps(40):
        actual = ramond_chiral_block("1.2", ".3", ext, c=30, q_order=2, sector="odd")
        assert actual != 0
        close(actual, reference)
        assert mp.mp.dps == 40


