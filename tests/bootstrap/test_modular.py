"""Tests for the genus-1 (torus) modular tables and crossing assembly.

The tables are exact, so the central check is direct: the bivariate v-series
reconstructed from a table
must converge to the explicit reduced character

    sqrt(tau_2) * q_tau^{hhat} * qbar_tau^{hbarhat}

evaluated through the disk maps tau(v) = i (1+v)/(1-v),
taubar(vbar) = -i (1+vbar)/(1-vbar), to truncation accuracy.
"""

from __future__ import annotations

import mpmath as mp
import pytest
from fractions import Fraction

from virasoro_blocks.interpolation import DampedRational, get_sample_points
from virasoro_blocks.modular import (
    TORUS_COORDINATE,
    sqrt_tau2_table,
    torus_character_table,
    torus_current_table,
    torus_exp_base,
    torus_vacuum_table,
)
from virasoro_bootstrap.modular_crossing import (
    ModularGapSpec,
    S_odd_pairs,
    build_modular_gap_problem,
    prepare_modular_gap_build,
)

_I = mp.mpc(0, 1)


def _tau(v):
    return _I * (1 + v) / (1 - v)


def _taubar(vbar):
    return -_I * (1 + vbar) / (1 - vbar)


def _reduced_character(v, vbar, hhat, hbarhat):
    tau2 = (_tau(v) - _taubar(vbar)) / (2 * _I)
    return (
        mp.sqrt(tau2)
        * mp.exp(2 * _I * mp.pi * _tau(v) * hhat)
        * mp.exp(-2 * _I * mp.pi * _taubar(vbar) * hbarhat)
    )


def _series_sum(vals, v0, vbar0, lam):
    return sum(
        mp.mpc(vals[m][n]) * v0**m * vbar0**n
        for m in range(lam + 1)
        for n in range(lam + 1)
    )


def test_character_table_matches_explicit_function():
    with mp.workdps(60):
        c = mp.mpf("8.3")
        Delta = mp.mpf("1.7")
        j = 2
        lam = 12
        hhat = (Delta + j) / 2 - (c - 1) / 24
        hbarhat = (Delta - j) / 2 - (c - 1) / 24
        table = torus_character_table(c, j, lam, symmetrize=False)
        vals = table.evaluate(TORUS_COORDINATE, Delta)
        for v0, vbar0 in [
            (mp.mpf("0.05"), mp.mpf("0.03")),
            (mp.mpf("-0.08"), mp.mpf("0.06")),
        ]:
            ref = _reduced_character(v0, vbar0, hhat, hbarhat)
            got = _series_sum(vals, v0, vbar0, lam)
            # Truncation-limited: |v|^(lam+1) ~ 1e-14 at worst here.
            assert abs(got - ref) / abs(ref) < mp.mpf("1e-10")


def test_character_table_prefactor_form():
    """(0,0) entry is exactly e^{-2 pi (Delta - (c-1)/12)}; base/poles exact."""
    with mp.workdps(50):
        c = mp.mpf(12)
        Delta = mp.mpf("2.5")
        table = torus_character_table(c, 0, 4)
        assert table.pole_list_x == ()
        assert abs(mp.mpf(table.base) - torus_exp_base()) == 0
        vals = table.evaluate(TORUS_COORDINATE, Delta)
        exact = mp.exp(-2 * mp.pi * (Delta - (c - 1) / 12))
        assert abs(mp.mpc(vals[0][0]) - exact) / exact < mp.mpf("1e-45")
        # degree of entry (m, n) is exactly m + n
        nums = table.numerators[TORUS_COORDINATE]
        assert nums[3][1].degree == 4
        assert nums[0][0].degree == 0


def test_character_backend_preserves_exact_real_labels_and_roundtrips_them():
    import json
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_blocks.writers import write_torus_json
    from virasoro_blocks.serialization import serialized_table_from_json

    tiny = Fraction(1, 10**70)
    with mp.workdps(30):
        with pytest.raises(ValueError, match='unitarity'):
            torus_character_table(4, 1, 1, shift=1-tiny)
        label = Fraction(7, 10)+tiny
        table = torus_character_table(Fraction(7, 3), 0, 1, shift=label)
        assert literal_number(table.shift) == label
        document = json.loads(write_torus_json(table, c=Fraction(7, 3), spin=0))
        restored = serialized_table_from_json(document)
        assert literal_number(restored.shift) == label
        assert torus_vacuum_table(Fraction(7, 3), 1) == torus_vacuum_table('7/3', 1)
        assert torus_character_table(4.7, 0, 1).numerators == torus_character_table('4.7', 0, 1).numerators


def test_vacuum_table_matches_explicit_function():
    with mp.workdps(60):
        c = mp.mpf("8.3")
        lam = 12
        a0 = -(c - 1) / 24
        vac = torus_vacuum_table(c, lam)
        v0, vbar0 = mp.mpf("0.04"), mp.mpf("-0.05")
        q = mp.exp(2 * _I * mp.pi * _tau(v0))
        qbar = mp.exp(-2 * _I * mp.pi * _taubar(vbar0))
        ref = _reduced_character(v0, vbar0, a0, a0) * (1 - q) * (1 - qbar)
        got = _series_sum(vac, v0, vbar0, lam)
        # Truncation-limited at lam = 12 (observed ~3e-13).
        assert abs(got - ref) / abs(ref) < mp.mpf("1e-10")


def test_current_table_is_the_irreducible_current_pair_character():
    """(j, 0) + (0, j): the vacuum half carries 1 - qbar, unlike the generic table's Verma continuation at Delta = j."""
    with mp.workdps(60):
        c, lam, j = mp.mpf("8.3"), 12, 2
        a0 = -(c - 1) / 24
        table = torus_current_table(c, j, lam)
        v0, vbar0 = mp.mpf("0.04"), mp.mpf("-0.05")
        q = mp.exp(2 * _I * mp.pi * _tau(v0))
        qbar = mp.exp(-2 * _I * mp.pi * _taubar(vbar0))
        ref = (_reduced_character(v0, vbar0, j + a0, a0) * (1 - qbar)
               + _reduced_character(v0, vbar0, a0, j + a0) * (1 - q))
        assert abs(_series_sum(table, v0, vbar0, lam) - ref) / abs(ref) < mp.mpf("1e-10")
    with mp.workdps(40):
        # at tau = i (v = 0) the generic paired table exceeds the current pair by exactly 1/(1 - e^{-2 pi})
        current = torus_current_table(2, 1, 3)[0][0]
        generic = torus_character_table(2, 1, 3).evaluate(TORUS_COORDINATE, 1)[0][0]
        assert abs(generic / current - 1 / (1 - mp.exp(-2 * mp.pi))) < mp.mpf("1e-35")
        assert abs(current - mp.mpf("0.006293055597889081875250250")) < mp.mpf("1e-26")


def test_current_constraints_sit_where_the_spin_edge_is_the_unitarity_bound():
    with mp.workdps(60):
        spec = ModularGapSpec(c=12, delta_gap="1.5", lam=5, spin_list=range(4), dps=60)
        problem = build_modular_gap_problem(spec)
        # spins 2 and 3 start at their unitarity bound, so their currents are allowed; spins 0 and 1 are gapped
        current = [ch for ch in problem.spin_channels if ch.block.numerators[TORUS_COORDINATE][1][0].degree == 0]
        assert [(ch.spin, mp.mpf(ch.delta_min)) for ch in current] == [(2, 2), (3, 3)]
        values = torus_current_table(12, 3, 5)
        for m, n in S_odd_pairs(5):
            assert current[1].block.numerators[TORUS_COORDINATE][m][n].coeffs[0] == values[m][n]
        assert len(problem.pmp_doc["PositiveMatrixWithPrefactorArray"]) == 6
        # a spin whose edge is lifted above unitarity, or chiral_currents=False, has no current constraint
        lifted = build_modular_gap_problem(ModularGapSpec(c=12, delta_gap="1.5", lam=5, spin_list=range(4), dps=60,
                                                          delta_min_by_spin={3: "3.5"}))
        assert [ch.spin for ch in lifted.spin_channels].count(3) == 1
        none = build_modular_gap_problem(ModularGapSpec(c=12, delta_gap="1.5", lam=5, spin_list=range(4), dps=60,
                                                        chiral_currents=False))
        assert len(none.spin_channels) == 4
    with pytest.raises(TypeError, match="chiral_currents"):
        ModularGapSpec(c=12, delta_gap=1, lam=5, chiral_currents="yes")


def test_spin_symmetrization_is_pair_sum_and_scalar_symmetric():
    with mp.workdps(40):
        c = mp.mpf("5.5")
        Delta = mp.mpf("3.2")
        lam = 6
        plain = torus_character_table(c, 3, lam, symmetrize=False).evaluate(
            TORUS_COORDINATE, Delta
        )
        sym = torus_character_table(c, 3, lam).evaluate(TORUS_COORDINATE, Delta)
        for m in range(lam + 1):
            for n in range(lam + 1):
                assert abs(
                    mp.mpc(sym[m][n]) - mp.mpc(plain[m][n]) - mp.mpc(plain[n][m])
                ) < mp.mpf("1e-30")
        scalar = torus_character_table(c, 0, lam).evaluate(
            TORUS_COORDINATE, Delta
        )
        for m in range(lam + 1):
            for n in range(lam + 1):
                assert abs(mp.mpc(scalar[m][n]) - mp.mpc(scalar[n][m])) < mp.mpf(
                    "1e-30"
                )


def test_sqrt_tau2_table_squares_to_tau2():
    with mp.workdps(40):
        lam = 10
        S = sqrt_tau2_table(lam)
        for a in range(lam + 1):
            for b in range(lam + 1):
                acc = mp.mpf(0)
                for i in range(a + 1):
                    for jj in range(b + 1):
                        acc += S[i][jj] * S[a - i][b - jj]
                expected = mp.mpf(1) if (a == 0 or b == 0) else mp.mpf(0)
                assert abs(acc - expected) < mp.mpf("1e-35")


def test_zero_pole_prefactor_sample_points():
    """The sample-point machinery accepts an empty pole list."""
    with mp.workdps(40):
        pref = DampedRational(constant=mp.mpf(1), poles=(), base=torus_exp_base())
        pts = get_sample_points(pref, 8)
        assert len(pts) == 8
        assert all(pts[i] < pts[i + 1] for i in range(7))
        assert pts[0] > 0
        pts_j = get_sample_points(pref, 8, method="jacobi")
        assert len(pts_j) == 8


def test_S_odd_pairs_matches_sphere_truncation():
    assert S_odd_pairs(3) == [(1, 0), (2, 1), (3, 0)]
    for m, n in S_odd_pairs(11):
        assert (m + n) % 2 == 1
        assert n <= min(m, 11 - m)


def test_build_modular_gap_problem_structure():
    with mp.workdps(60):
        # the generic channels only; the current constraints have their own tests
        spec = ModularGapSpec(
            c=12, delta_gap="1.5", lam=5, spin_list=range(4), dps=60, chiral_currents=False
        )
        prepared = prepare_modular_gap_build(spec)
        problem = build_modular_gap_problem(spec, prepared=prepared)
        doc = problem.pmp_doc
        blocks = doc["PositiveMatrixWithPrefactorArray"]
        assert len(blocks) == 4
        n_pairs = len(S_odd_pairs(5))
        assert len(doc["normalization"]) == n_pairs
        assert all(mp.mpf(v) == 0 for v in doc["objective"])
        assert len(doc["objective"]) == n_pairs
        for entry in blocks:
            assert entry["DampedRational"]["poles"] == []
            # degree-matched nodes: max poly degree = lam
            assert len(entry["samplePoints"]) == 5 + 1
            for poly in entry["polynomials"][0][0]:
                assert len(poly) - 1 <= 5
        # spin channels anchored at max(gap, j)
        delta_mins = [mp.mpf(ch.delta_min) for ch in problem.spin_channels]
        assert delta_mins == [mp.mpf("1.5"), mp.mpf("1.5"), mp.mpf(2), mp.mpf(3)]

        # reanchored build from `prepared` agrees with a fresh direct build
        fresh = build_modular_gap_problem(spec)
        ch_a = problem.spin_channels[1].block
        ch_b = fresh.spin_channels[1].block
        Delta = mp.mpf("2.9")
        va = ch_a.evaluate(TORUS_COORDINATE, Delta)
        vb = ch_b.evaluate(TORUS_COORDINATE, Delta)
        for m in range(6):
            for n in range(min(m, 5 - m) + 1):
                denom = max(abs(mp.mpc(vb[m][n])), mp.mpf("1e-40"))
                assert abs(mp.mpc(va[m][n]) - mp.mpc(vb[m][n])) / denom < mp.mpf(
                    "1e-40"
                )


def test_modular_gap_problem_with_isolated_scalar_folds_into_normalization():
    with mp.workdps(60):
        base_spec = ModularGapSpec(
            c=12, delta_gap=2, lam=5, spin_list=range(4), dps=60
        )
        iso_spec = ModularGapSpec(
            c=12,
            delta_gap=2,
            lam=5,
            spin_list=range(4),
            dps=60,
            isolated_scalars=((mp.mpf("0.7"), 1),),
        )
        base = build_modular_gap_problem(base_spec)
        iso = build_modular_gap_problem(iso_spec)

        assert len(iso.pmp_doc["normalization"]) == len(base.pmp_doc["normalization"])
        assert iso.pmp_doc["objective"] == base.pmp_doc["objective"]
        assert iso.pmp_doc["normalization"] != base.pmp_doc["normalization"]
        # The continuum support is still anchored at the requested heavy gap.
        assert mp.mpf(iso.spin_channels[0].delta_min) == mp.mpf(2)


def test_modular_fixed_degeneracies_are_combined_before_rounding(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import virasoro_bootstrap.modular_crossing as modular

    tiny = Fraction(1, 10**70)
    spec = ModularGapSpec(c=12, delta_gap=2, lam=1, spin_list=(0,), dps=30)
    prepared = replace(prepare_modular_gap_build(spec), identity_vector=(-10**100,))
    # Exact integer entries isolate assembly from character evaluation error.
    monkeypatch.setattr(modular, 'torus_character_table', lambda *_a, **_kw:
                        SimpleNamespace(evaluate=lambda *_: [[0], [1]]))
    for degeneracies in ((10**100-1, 1+tiny), (1+tiny, 10**100-1)):
        isolated = tuple((1, d) for d in degeneracies)
        problem = build_modular_gap_problem(replace(spec, isolated_scalars=isolated), prepared=prepared)
        with mp.workdps(30):
            assert abs(problem.identity_vector[0]/mp.convert(tiny)-1) < mp.mpf('1e-28')
            assert abs(mp.mpf(problem.pmp_doc['normalization'][0])/mp.convert(tiny)-1) < mp.mpf('1e-28')
        assert problem.spec.isolated_scalars == isolated


def test_modular_gap_problem_with_allowed_isolated_scalar_adds_point_channel():
    with mp.workdps(60):
        base_spec = ModularGapSpec(
            c=12, delta_gap=2, lam=5, spin_list=range(4), dps=60
        )
        allowed_spec = ModularGapSpec(
            c=12,
            delta_gap=2,
            lam=5,
            spin_list=range(4),
            dps=60,
            allowed_isolated_scalars=(mp.mpf("0.7"),),
        )
        base = build_modular_gap_problem(base_spec)
        allowed = build_modular_gap_problem(allowed_spec)

        assert allowed.pmp_doc["normalization"] == base.pmp_doc["normalization"]
        assert len(allowed.spin_channels) == len(base.spin_channels) + 1
        assert mp.mpf(allowed.spin_channels[-1].delta_min) == mp.mpf("0.7")
        point_block = allowed.pmp_doc["PositiveMatrixWithPrefactorArray"][-1]
        assert len(point_block["samplePoints"]) == 1


def test_gap_problem_rejects_bad_specs():
    for field,value in [('c',mp.mpc(4,'1e-70')),('delta_gap',True)]:
        params = dict(c=4,delta_gap=1,lam=1,spin_list=(0,),dps=30)
        params[field] = value
        with pytest.raises(ValueError, match=field):
            build_modular_gap_problem(ModularGapSpec(**params))
    with pytest.raises(ValueError):
        build_modular_gap_problem(
            ModularGapSpec(c=0.9, delta_gap=1, lam=3, spin_list=(0,), dps=30)
        )
    with pytest.raises(ValueError):
        build_modular_gap_problem(
            ModularGapSpec(c=4, delta_gap=1, lam=3, spin_list=(1, 2), dps=30)
        )
    with pytest.raises(ValueError):
        build_modular_gap_problem(
            ModularGapSpec(
                c=4,
                delta_gap=1,
                lam=3,
                spin_list=(0, 2),
                dps=30,
                delta_min_by_spin={2: 1},
            )
        )


def test_write_torus_json_round_trip():
    import json

    from virasoro_blocks.writers import write_torus_json
    from virasoro_blocks.serialization import serialized_table_from_json

    with mp.workdps(40):
        table = torus_character_table(10, 2, 6)
        doc = json.loads(write_torus_json(table, c=10, spin=2))
        assert doc["surface"] == "torus"
        assert doc["nome_convention"] == "modular_2pi"
        assert doc["basis"] == "torus_G"
        assert doc["pole_list_x"] == []
        assert doc["order"] is None and doc["npoles"] is None
        back = serialized_table_from_json(doc)
        Delta = mp.mpf("3.3")
        a = table.evaluate(TORUS_COORDINATE, Delta)
        b = back.evaluate(TORUS_COORDINATE, Delta)
        for m in range(7):
            for n in range(min(m, 6 - m) + 1):
                denom = max(abs(mp.mpc(a[m][n])), mp.mpf("1e-40"))
                assert abs(mp.mpc(a[m][n]) - mp.mpc(b[m][n])) / denom < mp.mpf(
                    "1e-30"
                )


def test_torus_writer_retains_build_precision_outside_context():
    import json
    from virasoro_blocks.writers import write_torus_json
    from virasoro_blocks.serialization import serialized_table_from_json

    with mp.workdps(80):
        table = torus_character_table(10, 2, 4)
        expected = table.evaluate(TORUS_COORDINATE, "3.3")
    with mp.workdps(15):
        doc = json.loads(write_torus_json(table, c=10, spin=2))
        back = serialized_table_from_json(doc)
    assert back.working_dps == 80
    with mp.workdps(80):
        actual = back.evaluate(TORUS_COORDINATE, "3.3")
        for m in range(5):
            for n in range(min(m, 4 - m) + 1):
                assert abs(actual[m][n] / expected[m][n] - 1) < mp.mpf("1e-70")


def test_torus_cli_writes_loadable_files(tmp_path):
    import json

    from virasoro_blocks.__main__ import main
    from virasoro_blocks.serialization import serialized_table_from_json

    rc = main(
        [
            "--surface", "torus",
            "--central-charge", "10",
            "--spin-ranges", "0:1:2",
            "--lambda", "5",
            "--precision", "40",
            "-o", str(tmp_path),
        ]
    )
    assert rc == 0
    for spin in (0, 1, 2):
        doc = json.loads((tmp_path / f"derivs_{spin}.json").read_text())
        assert doc["surface"] == "torus"
        assert doc["spin"] == spin
        with mp.workdps(40):
            table = serialized_table_from_json(doc)
            assert table.pole_list_x == ()


def test_modular_vacuum_current_and_allowed_scalar_channels_are_the_cli_point_tables(tmp_path):
    """The modular bootstrap's identity vector, its current constraints and an allowed isolated scalar are the
    `point-table --surface torus` files: (0, 0), (j, 0) and (Delta, 0), read back at the problem's F-pairs."""
    import json

    from virasoro_blocks.__main__ import main
    from virasoro_blocks.serialization import serialized_table_from_json
    from virasoro_bootstrap.modular_crossing import _modular_allowed_scalar_block, _modular_current_block

    lam, dps = 5, 40
    for delta, spin, name in (("0", "0", "vac.json"), ("3", "3", "cur.json"), ("7/5", "0", "scalar.json")):
        assert main(["point-table", "--surface", "torus", "--family", "torus-character", "--central-charge", "12",
                     "--exchange-delta", delta, "--exchange-spin", spin, "--lambda", str(lam), "--precision",
                     str(dps), "-o", str(tmp_path), "--output-template", name]) == 0

    def load(name):
        doc = json.loads((tmp_path / name).read_text())
        assert doc["variable"] == "point"
        with mp.workdps(dps):
            return doc, serialized_table_from_json(doc)

    with mp.workdps(dps):
        prepared = prepare_modular_gap_build(ModularGapSpec(c=12, delta_gap="1.5", lam=lam, spin_list=range(4),
                                                            dps=dps))
        F_pairs = prepared.F_pairs
        doc, vacuum = load("vac.json")
        assert doc["regularization_policy"] == "irreducible"
        values = vacuum.evaluate(TORUS_COORDINATE, 0)
        for (m, n), expected in zip(F_pairs, prepared.identity_vector):
            assert abs(values[m][n] - expected) <= mp.mpf(10) ** -(dps - 3) * abs(expected)
        doc, current = load("cur.json")
        block = _modular_current_block(Fraction(12), lam, 3, F_pairs)
        assert mp.mpf(current.shift) == block.shift == 3
        for m, n in F_pairs:
            got = current.numerators[TORUS_COORDINATE][m][n].coeffs[0]
            expected = block.numerators[TORUS_COORDINATE][m][n].coeffs[0]
            assert abs(got - expected) <= mp.mpf(10) ** -(dps - 3) * abs(expected)
        doc, scalar = load("scalar.json")
        assert doc["regularization_policy"] == "direct"
        block = _modular_allowed_scalar_block(Fraction(12), lam, Fraction(7, 5), F_pairs)
        for m, n in F_pairs:
            got = scalar.numerators[TORUS_COORDINATE][m][n].coeffs[0]
            expected = block.numerators[TORUS_COORDINATE][m][n].coeffs[0]
            assert abs(got - expected) <= mp.mpf(10) ** -(dps - 12) * abs(expected)


def test_torus_cli_rejects_sphere_flags(tmp_path):
    from virasoro_blocks.__main__ import main

    for extra in (
        ["--order", "20"],
        ["--npoles", "8"],
        ["--delta-ext", "0.1", "0.1", "0.1", "0.1"],
        ["--format", "m"],
        ["--spin-ranges", "1/2"],
    ):
        argv = [
            "--surface", "torus",
            "--central-charge", "10",
            "--lambda", "4",
            "--precision", "30",
            "-o", str(tmp_path),
        ]
        with pytest.raises(SystemExit):
            main(argv + extra)
    # (a c <= 1 character table is the Verma character, written with a note; tests/core/test_point_table_families.py)


def test_modular_rational_c_matches_equivalent_string():
    from fractions import Fraction

    def prepare(c):
        return prepare_modular_gap_build(ModularGapSpec(c=c, delta_gap=2, lam=3, spin_list=(0,), dps=30))
    exact = prepare(Fraction(10, 3))
    text = prepare("10/3")
    assert exact.identity_vector == text.identity_vector
    assert exact.reuse_key == text.reuse_key


def test_modular_reuse_distinguishes_c_values_that_share_a_decimal_display():
    from dataclasses import replace

    with mp.workdps(30):
        c = mp.mpf(4)
        shifted = c + mp.eps * 4
        assert c != shifted
        assert mp.nstr(c, 30) == mp.nstr(shifted, 30)
    spec = ModularGapSpec(c=c, delta_gap=2, lam=1, spin_list=(0,), dps=30)
    prepared = prepare_modular_gap_build(spec)
    with pytest.raises(ValueError, match="incompatible"):
        build_modular_gap_problem(replace(spec, c=shifted), prepared=prepared)


@pytest.mark.parametrize("checkpoint_name", ["prior.ck", "out/initial.ck"])
def test_modular_hotstart_preserves_checkpoint_when_reusing_output(tmp_path, monkeypatch, checkpoint_name):
    from virasoro_bootstrap import modular_crossing as module
    from virasoro_bootstrap.hpc.runner import SdpbRunner

    problem = build_modular_gap_problem(ModularGapSpec(c=4, delta_gap=2, lam=1, spin_list=(0,), dps=30))
    seed = tmp_path / checkpoint_name
    seed.mkdir(parents=True)
    (seed / "checkpoint.0").write_text("prior-iterate")
    def command(*args, **kwargs):
        return ["sdpb"], tmp_path / "out"
    def run(argv, log, *, check=True, cwd=None):
        assert (cwd / "initial.ck" / "checkpoint.0").read_text() == "prior-iterate"
        assert (cwd / "sdp.ck").is_dir() and not list((cwd / "sdp.ck").iterdir())
        (cwd / "sdp.ck" / "checkpoint.0").write_text("new-iterate")
        log.write_text("-----found primal feasible solution-----\n")
        return 0
    monkeypatch.setattr(SdpbRunner, "command_argv", command)
    monkeypatch.setattr(module, "_run_command", run)
    monkeypatch.setattr(module, "write_sdp_directory", lambda *args, **kwargs: None)
    result = module.run_modular_gap_problem(problem, tmp_path / "out", initial_checkpoint_dir=seed)
    assert result.sdpb_status == "primal_feasible"
    assert (result.checkpoint_dir / "checkpoint.0").read_text() == "new-iterate"


def test_modular_solver_validates_before_writing_and_records_attempts(tmp_path, monkeypatch):
    import json
    from virasoro_bootstrap import modular_crossing as module
    from virasoro_bootstrap.hpc.runner import SdpbRunner
    from virasoro_bootstrap.hpc.storage import (
        SDP_SOURCE_NAME, recorded_sdp_source, cleanup_sdp_inputs, regenerate_sdp_inputs,
    )

    problem = build_modular_gap_problem(ModularGapSpec(c=4, delta_gap=2, lam=1, spin_list=(0,), dps=30))
    saved = {**problem.pmp_doc, "objective": ["1"] * len(problem.pmp_doc["normalization"])}
    pmp = tmp_path / "pmp.json"
    pmp.write_text(json.dumps(saved))
    before = pmp.read_bytes()
    runner = SdpbRunner(mode="native", bin_dir="/unused")
    for invalid in ({"sdpb_options": {"--precision": 0}}, {"eval_dps": True}):
        with pytest.raises(ValueError):
            module.run_modular_gap_problem(problem, tmp_path, runner=runner, **invalid)
        assert pmp.read_bytes() == before
        assert not (tmp_path / "sdp").exists()

    outcome = [0]
    def solve(command, log_path, *, check=True, cwd=None):
        output = cwd / "sdpb_out"
        with pytest.raises(ValueError, match="no completed attempt"):
            recorded_sdp_source(output)
        log_path.write_text("-----found dual feasible solution-----\n")
        if outcome[0] is None:
            raise RuntimeError("interrupted solver")
        return outcome[0]

    monkeypatch.setattr(module.shutil, "which", lambda name: str(name))
    monkeypatch.setattr(module, "_run_command", solve)
    for code in (0, 1, None):
        outcome[0] = code
        if code is None:
            with pytest.raises(RuntimeError, match="interrupted solver"):
                module.run_modular_gap_problem(problem, tmp_path, runner=runner, eval_dps=40)
        else:
            result = module.run_modular_gap_problem(problem, tmp_path, runner=runner, eval_dps=40)
            assert result.sdpb_status == ("dual_feasible" if code == 0 else "solver_error")
        output = tmp_path / "sdpb_out"
        record = json.loads((output / SDP_SOURCE_NAME).read_text())
        assert record["state"] == {0: "completed", 1: "failed", None: "running"}[code]
        if code == 0:
            assert recorded_sdp_source(output).resolve() == (tmp_path / "sdp").resolve()
            inputs = {p.name: json.loads(p.read_text()) for p in result.sdp_dir.glob("block_*.json")}
            cleanup_sdp_inputs(result.sdp_dir)
            regenerate_sdp_inputs(tmp_path)
            assert {p.name: json.loads(p.read_text()) for p in result.sdp_dir.glob("block_*.json")} == inputs
            assert recorded_sdp_source(output).resolve() == result.sdp_dir.resolve()
        else:
            with pytest.raises(ValueError, match="no completed attempt"):
                recorded_sdp_source(output)
