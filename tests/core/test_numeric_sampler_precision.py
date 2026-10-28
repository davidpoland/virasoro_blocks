"""A sampler reused at higher precision must rebuild its numerical caches."""
import mpmath as mp
import pytest

from virasoro_blocks.derivs import _arb_to_mpf
from virasoro_blocks.polyderivs import build_numeric_sampler


@pytest.mark.parametrize("basis,coordinate,center", [
    (basis, coordinate, center)
    for basis in ("scalar", "ordered", "projected", "chiral")
    for coordinate, center in (("zzb", None), ("qqb", None), ("yyb", ".3"), ("qqb", ".3"))
] + [
    (basis, coordinate, None)
    for basis in ("scalar", "projected")
    for coordinate in ("xt", "ws", "qqb_ws")
])
@pytest.mark.parametrize("restricted", [False, True])
def test_numeric_sampler_rebuilds_after_precision_change(basis, coordinate, center, restricted):
    chiral = basis == "chiral"
    coordinate = {"zzb": "z", "yyb": "y", "qqb": "q"}[coordinate] if chiral else coordinate
    n_max = 0 if chiral else 1
    entries = [(1, 0)] if restricted else None

    def make():
        return build_numeric_sampler(
            coordinate, chiral=chiral, spinning=basis in ("ordered", "projected"),
            projected=basis == "projected", h_externals=[mp.mpf(".125")] * 4,
            hbar_externals=[mp.mpf(".25")] * 4, ell=2, m_max=1, n_max=n_max,
            c=32, q_order=4, chiral_norm=4, z_shift=mp.mpf("-.25"),
            zbar_shift=mp.mpf("-.5"), parity="+", parity_external=("1",) * 4,
            center=center, center_bar=".4" if center is not None and not chiral else None,
        )

    with mp.workdps(20):
        sampler = make()
        sampler([mp.mpf(4)], entries=entries)
    with mp.workdps(60):
        values = [mp.mpf(4), mp.mpf(6)]  # cached point and new point
        got, got_arb = sampler(values, entries=entries)
        reference, ref_arb = make()(values, entries=entries)
        pairs = entries or [(m, n) for m in range(2) for n in range(n_max + 1)]
        for actual, expected in zip(got, reference):
            for m, n in pairs:
                a = _arb_to_mpf(actual[m][n]) if got_arb else actual[m][n]
                e = _arb_to_mpf(expected[m][n]) if ref_arb else expected[m][n]
                assert abs(a - e) < mp.mpf("1e-55") * max(1, abs(e))


@pytest.mark.parametrize("entry_form", ["all", "list", "iterator"])
def test_remainder_sampler_does_not_modify_inner_cache_or_repeated_points(entry_form):
    from virasoro_blocks.polyderivs import _RemainderSampler

    with mp.workdps(50):
        inner = build_numeric_sampler(
            "zzb", chiral=False, spinning=False, projected=False,
            h_externals=[mp.mpf(".125")] * 4, hbar_externals=None,
            ell=2, m_max=0, n_max=0, c=32, q_order=4,
            chiral_norm=4, z_shift=mp.mpf("-.25"),
        )
        original, is_arb = inner([mp.mpf(4)])
        original_value = _arb_to_mpf(original[0][0][0]) if is_arb else original[0][0][0]
        # A simple principal part 1/(Delta+1), independent of the block's value.
        wrapped = _RemainderSampler(inner, (mp.mpf(1), [
            (mp.mpf(-1), {(0, 0): (mp.mpf(1),)}),
        ]))
        expected = original_value - mp.mpf(1) / 5
        for _ in range(2):
            entries = None if entry_form == "all" else [(0, 0)]
            if entry_form == "iterator":
                entries = iter(entries)
            tables, is_arb = wrapped([mp.mpf(4), mp.mpf(4)], entries=entries)
            for table in tables:
                got = _arb_to_mpf(table[0][0]) if is_arb else table[0][0]
                assert abs(got - expected) < mp.mpf("1e-45")
        unchanged, is_arb = inner([mp.mpf(4)])
        got = _arb_to_mpf(unchanged[0][0][0]) if is_arb else unchanged[0][0][0]
        assert got == original_value


@pytest.mark.parametrize("basis", ["scalar", "ordered", "chiral"])
def test_numeric_pole_scoring_refreshes_precision_without_value_call(basis):
    from virasoro_blocks.polyblock import chiral_denominator_roots

    def make():
        return build_numeric_sampler(
            "z" if basis == "chiral" else "zzb",
            chiral=basis == "chiral", spinning=basis == "ordered", projected=False,
            h_externals=[mp.mpf(".125")] * 4, hbar_externals=[mp.mpf(".125")] * 4,
            ell=2, m_max=1, n_max=0 if basis == "chiral" else 1,
            c=32, q_order=4, chiral_norm=4, z_shift=mp.mpf("-.25"), zbar_shift=mp.mpf("-.25"),
        )

    with mp.workdps(80):
        roots = chiral_denominator_roots([mp.mpf(".125")] * 4, c=32, q_order=4)
        if basis != "chiral":
            roots = [2 * h + 2 for h in roots]
    with mp.workdps(20):
        sampler = make()
        sampler._analytic_contribution_scores(roots, mp.mpf(4), [(1, 0)], mp.mpf(".5"))
    with mp.workdps(60):
        args = (roots, mp.mpf(4), [(1, 0)], mp.mpf(".5"))
        actual = sampler._analytic_contribution_scores(*args)
        expected = make()._analytic_contribution_scores(*args)
        assert actual is not None and expected is not None
        for got, ref in zip(actual, expected):
            assert abs(got - ref) < mp.mpf("1e-50") * max(1, abs(ref))


@pytest.mark.parametrize("basis,coordinate,center", [
    (basis, coordinate, center)
    for basis in ("global_scalar_G", "global_ordered_B", "global_parity_projected_B")
    for coordinate, center in (("zzb", None), ("qqb", None), ("yyb", ".3"), ("qqb", ".3"))
] + [
    ("global_scalar_G", coordinate, None) for coordinate in ("xt", "ws", "qqb_ws")
] + [
    ("global_chiral_F", coordinate, center)
    for coordinate in ("z", "y", "q") for center in (None, ".3")
])
def test_global_sampler_refreshes_after_precision_change(basis, coordinate, center):
    from virasoro_blocks.global_blocks import build_global_sampler

    chiral = basis == "global_chiral_F"
    n_max = 0 if chiral else 1
    entries = [(m, n) for m in range(2) for n in range(n_max + 1)]

    def make():
        return build_global_sampler(
            coordinate, basis=basis, ell=2,
            h_externals=[mp.mpf(".125")] * 4, hbar_externals=[mp.mpf(".25")] * 4,
            m_max=1, n_max=n_max, chiral_norm=4,
            z_shift=mp.mpf("-.25"), zbar_shift=mp.mpf("-.5"),
            parity="+", parity_external=("1",) * 4,
            center=center, center_bar=".4" if center is not None and not chiral else None,
        )

    with mp.workdps(20):
        sampler = make()
        sampler(mp.mpf(4))
    with mp.workdps(60):
        reference = make()(mp.mpf(6))
        if not chiral:
            tables, is_arb = sampler.sample_bucket([mp.mpf(6)], entries)
            for m, n in entries:
                got = _arb_to_mpf(tables[0][m][n]) if is_arb else tables[0][m][n]
                assert abs(got - reference[m][n]) < mp.mpf("1e-55") * max(1, abs(reference[m][n]))
        actual = sampler(mp.mpf(6))
        for m, n in entries:
            assert abs(actual[m][n] - reference[m][n]) < mp.mpf("1e-55") * max(1, abs(reference[m][n]))


@pytest.mark.parametrize("method", ["_analytic_residue_scores", "_analytic_leading_residue_scores"])
@pytest.mark.parametrize("coordinate,center", [("qqb", None), ("qqb_ws", None), ("yyb", ".3")])
def test_global_pole_scoring_refreshes_precision(method, coordinate, center):
    from virasoro_blocks.global_blocks import build_global_sampler

    def make():
        return build_global_sampler(
            coordinate, basis="global_scalar_G", ell=0,
            h_externals=[mp.mpf(".125")] * 4, hbar_externals=None,
            m_max=1, n_max=1, chiral_norm=4, z_shift=mp.mpf("-.25"),
            center=center, center_bar=".4" if center is not None else None,
        )

    def score(sampler):
        roots, entries, base = [mp.mpf(-1)] * 2, [(1, 0)], mp.mpf(".25")
        if method == "_analytic_residue_scores":
            return getattr(sampler, method)(roots, mp.mpf(4), entries, base)
        return getattr(sampler, method)(roots, entries, base)

    with mp.workdps(20):
        sampler = make()
        score(sampler)
    with mp.workdps(60):
        actual, expected = score(sampler), score(make())
        assert actual is not None and expected is not None
        for got, ref in zip(actual, expected):
            assert abs(got - ref) < mp.mpf("1e-50") * max(1, abs(ref))


@pytest.mark.parametrize("basis,coordinate,center", [
    (basis, coordinate, center)
    for basis in ("scalar", "ordered", "projected")
    for coordinate, center in (("zzb", None), ("zzb", ".3"))
] + [("scalar", "ws", None), ("projected", "ws", None)])
def test_numeric_sampler_entry_iterators_match_lists(basis, coordinate, center):
    def make():
        return build_numeric_sampler(
            coordinate, chiral=False, spinning=basis != "scalar", projected=basis == "projected",
            h_externals=[mp.mpf(".125")] * 4, hbar_externals=[mp.mpf(".25")] * 4,
            ell=2, m_max=1, n_max=1, c=32, q_order=4, chiral_norm=4,
            parity="+", center=center, center_bar=".4" if center else None,
        )

    with mp.workdps(40):
        pairs = [(0, 0), (1, 0)]
        deltas = [mp.mpf(4), mp.mpf(6)]
        expected, expected_arb = make()(deltas, entries=pairs)
        sampler = make()
        for _ in range(2):  # both initial computation and cached reuse
            actual, actual_arb = sampler(deltas, entries=iter(pairs))
            for got, ref in zip(actual, expected):
                for m, n in pairs:
                    a = _arb_to_mpf(got[m][n]) if actual_arb else got[m][n]
                    e = _arb_to_mpf(ref[m][n]) if expected_arb else ref[m][n]
                    assert abs(a-e) < mp.mpf("1e-35") * max(1, abs(e))


@pytest.mark.parametrize("family,chiral", [("global", False), ("ns", False), ("ramond", False),
                                          ("ns", True), ("ramond", True)])
@pytest.mark.parametrize("iterator_field", ["values", "entries", "both"])
def test_other_sampler_buckets_preserve_iterable_inputs(family, chiral, iterator_field):
    from virasoro_blocks.global_blocks import build_global_sampler
    from virasoro_blocks.superconformal_blocks import SuperBlockSampler2d, SuperChiralSampler

    with mp.workdps(35):
        ext = [mp.mpf(".1")] * 4
        if family == "global":
            sampler = build_global_sampler("zzb", basis="global_scalar_G", ell=0,
                                          h_externals=ext, hbar_externals=ext, m_max=1, n_max=1)
        else:
            common = dict(c=mp.mpf("2.3"), q_order=4, m_max=1, chiral_norm=None,
                          z_shift=None, sector=0, branch="++")
            if chiral:
                sampler = SuperChiralSampler(family, "z", ext, center=None, **common)
            else:
                sampler = SuperBlockSampler2d(family, "zzb", 0, ext, ext, n_max=1,
                                              sigma=1, fold_parity=0, zbar_shift=None, center=None, center_bar=None, **common)
        values = [mp.mpf("1.7"), mp.mpf("2.3")]
        entries = [(0, 0), (1, 0)]
        expected, ref_arb = sampler.sample_bucket(values, entries)
        actual, got_arb = sampler.sample_bucket(
            iter(values) if iterator_field in {"values", "both"} else values,
            iter(entries) if iterator_field in {"entries", "both"} else entries,
        )
        assert len(actual) == len(expected)
        for got, ref in zip(actual, expected):
            for m, n in entries:
                a = _arb_to_mpf(got[m][n]) if got_arb else got[m][n]
                e = _arb_to_mpf(ref[m][n]) if ref_arb else ref[m][n]
                assert abs(a-e) < mp.mpf("1e-30") * max(1, abs(e))


def test_global_complex_bucket_preserves_both_iterators():
    from virasoro_blocks.global_blocks import build_global_sampler

    with mp.workdps(35):
        sampler = build_global_sampler("zzb", basis="global_scalar_G", ell=0,
                                      h_externals=[mp.mpf(".1")]*4, hbar_externals=None,
                                      m_max=1, n_max=1)
        values = [mp.mpc(2, ".1"), mp.mpc(3, ".2")]
        entries = [(0, 0), (1, 0)]
        expected, _ = sampler.sample_bucket(values, entries)
        actual, is_arb = sampler.sample_bucket(iter(values), iter(entries))
        assert not is_arb
        assert len(actual) == len(expected)
        for got, ref in zip(actual, expected):
            for m, n in entries:
                assert mp.almosteq(got[m][n], ref[m][n])


@pytest.mark.parametrize("basis", ["global_scalar_G", "global_ordered_B", "global_parity_projected_B"])
@pytest.mark.parametrize("value", [complex(2, .125), complex(2, 0), mp.mpc(2, 0)])
@pytest.mark.parametrize("center", [None, ".3"])
def test_global_batch_complex_number_types_match_single_values(basis, value, center):
    from virasoro_blocks.global_blocks import build_global_sampler

    with mp.workdps(40):
        sampler = build_global_sampler(
            "zzb", basis=basis, ell=0, h_externals=[mp.mpf(".1")]*4,
            hbar_externals=[mp.mpf(".2")]*4, m_max=1, n_max=1,
            parity="+", center=center, center_bar=".4" if center else None,
        )
        expected = sampler(value)
        actual, is_arb = sampler.sample_bucket([value], [(0, 0), (1, 0)])
        assert is_arb == (value.imag == 0)
        for m, n in [(0, 0), (1, 0)]:
            got = _arb_to_mpf(actual[0][m][n]) if is_arb else actual[0][m][n]
            assert abs(got-expected[m][n]) < mp.mpf("1e-35") * max(1, abs(expected[m][n]))


@pytest.mark.parametrize("basis,coordinate", [("scalar", "zzb"), ("scalar", "ws"),
    ("chiral", "z"), ("chiral", "q"), ("ordered", "zzb"), ("projected", "zzb")])
@pytest.mark.parametrize("value", [complex(4, 0), complex(4, .125)])
@pytest.mark.parametrize("restricted", [False, True])
def test_virasoro_samplers_accept_native_complex_numbers(basis, coordinate, value, restricted):
    def make():
        return build_numeric_sampler(
            coordinate, chiral=basis == "chiral", spinning=basis in {"ordered", "projected"},
            projected=basis == "projected", h_externals=[mp.mpf(".1")]*4,
            hbar_externals=[mp.mpf(".2")]*4, ell=2, m_max=1,
            n_max=0 if basis == "chiral" else 1, c=32, q_order=4, chiral_norm=16, parity="+",
        )

    with mp.workdps(40):
        entries = [(0, 0), (1, 0)] if restricted else None
        expected, ref_arb = make()([mp.mpc(value)], entries=entries)
        actual, got_arb = make()([value], entries=entries)
        for m, n in [(0, 0), (1, 0)]:
            a = _arb_to_mpf(actual[0][m][n]) if got_arb else actual[0][m][n]
            e = _arb_to_mpf(expected[0][m][n]) if ref_arb else expected[0][m][n]
            assert abs(a-e) < mp.mpf("1e-35") * max(1, abs(e))


@pytest.mark.parametrize("mixed", [False, True])
def test_chiral_sampler_arithmetic_flag_accounts_for_cached_values(mixed):
    with mp.workdps(40):
        sampler = build_numeric_sampler(
            "z", chiral=True, spinning=False, projected=False,
            h_externals=[mp.mpf(".1")]*4, hbar_externals=None,
            ell=0, m_max=1, n_max=0, c=32, q_order=4, chiral_norm=16,
        )
        cached = mp.mpc(4, ".125") if mixed else mp.mpf(4)
        sampler([cached])
        values = [cached, mp.mpf(6)] if mixed else [cached]
        tables, is_arb = sampler(values)
        assert is_arb is (not mixed)
        for value, table in zip(values, tables):
            reference, ref_arb = sampler([value])
            for m in range(2):
                actual = _arb_to_mpf(table[m][0]) if is_arb else mp.mpc(table[m][0])
                expected = _arb_to_mpf(reference[0][m][0]) if ref_arb else mp.mpc(reference[0][m][0])
                assert mp.almosteq(actual, expected)
