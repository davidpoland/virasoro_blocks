"""Regressions for the 2026-09-27 bootstrap-package audit.

Each test pins one confirmed defect: a build that crashed, a certificate that
passed when it proved nothing, a rebuild that solved a different problem, or a
value parsed at the wrong precision.  All are small builds (Lambda = 3-5, a few
seconds at most).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import mpmath as mp
import pytest

from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_fixed_ope_problem,
    build_identical_scalar_gap_problem,
    build_identical_scalar_ope_bound_problem,
    build_identical_scalar_ope_max_problem,
    prepare_identical_scalar_gap_build,
    write_identical_scalar_gap_problem,
)


def _spec(**overrides):
    kwargs = dict(
        c=mp.mpf("3.3"), h_ext=mp.mpf("0.1"), delta_gap=mp.mpf("1.1"), lam=3,
        spin_list=(0, 2), q_order=12, dps=40, npoles=6, num_threads=1,
    )
    kwargs.update(overrides)
    return IdenticalScalarGapSpec(**kwargs)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --- crossing: zero numerators in the simple-pole splitter --------------------

def test_generic_c_below_one_builds_with_a_spin_two_channel():
    """The non-kept n > m half of a spin >= 2 channel has zero numerators; the
    fast simple-pole splitter divided them by (x - root) and raised."""
    with mp.workdps(40):
        problem = _quiet(build_identical_scalar_gap_problem, _spec(
            c=mp.mpf("0.3"), delta_gap=mp.mpf("0.5"), q_order=8,
        ))
    assert len(problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"]) == 2


@pytest.mark.parametrize("power, direction", [(0.5, "read"), (True, "write")])
def test_moment_serialization_rejects_invalid_powers(power, direction):
    from virasoro_bootstrap.moments import MomentTerm

    with pytest.raises(ValueError, match="moment powers"):
        if direction == "read":
            MomentTerm.from_json({"coefficient": "1", "powers": {"Delta": power}})
        else:
            MomentTerm(1, {"Delta": power}).to_json()


def test_moment_json_does_not_truncate_spin_filter():
    from virasoro_bootstrap.moments import MomentWeight

    with pytest.raises(ValueError, match="spin_filter"):
        MomentWeight.from_json({"terms": [{"coefficient": "1", "powers": {}}], "spin_filter": 2.5})


def test_moment_json_retains_coefficient_digits_at_low_ambient_precision():
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.moments import MomentTerm

    coefficient = "0.123456789012345678901234567890123456789"
    with mp.workdps(15):
        term = MomentTerm.from_json({"coefficient": coefficient, "powers": {"Delta": 1}})
    assert literal_number(term.coefficient) == Fraction(coefficient)


@pytest.mark.parametrize("central_charge, singular", [(".7", True),
    ("0.70000000000000000000000000000000001", False)])
def test_bootstrap_preserves_exact_c_during_validation_and_manifest_roundtrip(central_charge, singular):
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.identical_scalar import _validate_gap_spec_common, _resolved_regularize
    from virasoro_bootstrap.manifest import _spec_to_json, _spec_from_json

    with mp.workdps(40):
        spec = _spec(c=central_charge)
        c = _validate_gap_spec_common(spec)[2]
        assert literal_number(c) == literal_number(central_charge)
        assert _resolved_regularize(c, "auto") is singular
        encoded = _spec_to_json(spec)
    with mp.workdps(15):
        restored = _spec_from_json(encoded)
    assert literal_number(restored.c) == literal_number(central_charge)


def test_bootstrap_mpf_c_retains_its_binary_value():
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.identical_scalar import _validate_gap_spec_common, _resolved_regularize
    from virasoro_bootstrap.manifest import _spec_to_json, _spec_from_json

    with mp.workdps(40):
        central_charge = mp.mpf(".7")
        spec = _spec(c=central_charge)
        validated = _validate_gap_spec_common(spec)[2]
        assert validated == literal_number(central_charge)
        assert _resolved_regularize(validated, "auto") is False
        restored = _spec_from_json(_spec_to_json(spec))
        assert restored.c == literal_number(central_charge)


def test_shifted_c_build_and_reload_preserve_the_input(tmp_path):
    from virasoro_bootstrap.manifest import write_identical_scalar_manifest, build_identical_scalar_problem_from_manifest

    central_charge = "0.70000000000000000000000000000000001"
    spec = _spec(c=central_charge, lam=1, q_order=4, spin_list=(0,), npoles=2)
    problem = _quiet(build_identical_scalar_gap_problem, spec)
    assert problem.used_regularize is False
    assert problem.spec.c == spec.c
    path = write_identical_scalar_manifest(problem, tmp_path)
    assert json.loads(path.read_text())["spec"]["c"] == central_charge
    with mp.workdps(15):
        restored = build_identical_scalar_problem_from_manifest(path)
    assert restored.used_regularize is False
    assert restored.spec.c == spec.c
    assert restored.identity_vector == problem.identity_vector


@pytest.mark.parametrize("value, field", [
    (mp.nan, "c"), ("inf", "point_constraints"), (mp.mpc(1, mp.inf), "large_spin_twists"),
])
def test_bootstrap_spec_rejects_nonfinite_inputs(value, field):
    from virasoro_bootstrap.identical_scalar import _validate_gap_spec_common, _resolved_delta_min_by_ell

    options = {field: value}
    if field == "point_constraints":
        options[field] = ((0, value),)
    elif field == "large_spin_twists":
        options = {field: (value,), "large_spin_lmin": 4}
    with pytest.raises(ValueError, match="finite|Invalid literal"):
        spec = _spec(**options)
        _validate_gap_spec_common(spec)
        _resolved_delta_min_by_ell(spec, spec.spin_list)


# --- certification: non-finite and unnormalized functionals -------------------

def test_block_positivity_rejects_a_non_finite_functional():
    from virasoro_bootstrap.positivity import block_positivity

    entry = {"DampedRational": {"base": "0.5", "constant": "1", "poles": ["-1"]},
             "polynomials": [[[["1", "-3", "1"], ["0", "1"]]]]}
    with mp.workdps(30):
        for bad in (mp.nan, mp.inf):
            detail = block_positivity(entry, [bad, mp.mpf(1)])
            assert detail["passed"] is None
            assert detail["reason"] == "non_finite_functional"


def test_certify_dual_functional_counts_nan_as_a_failure():
    from virasoro_bootstrap.certification import _certify_one_channel

    with mp.workdps(40):
        problem = _quiet(build_identical_scalar_gap_problem, _spec())
        y = [mp.nan] + [mp.mpf("0.1")] * (len(problem.F_pairs) - 1)
        detail = _certify_one_channel(
            y, problem.spin_channels[0], list(problem.F_pairs), "zzb", 10, 5.0,
        )
    assert detail["passed"] is False
    assert detail["n_negative"] == 10


@pytest.fixture(scope="module")
def gap_problem_with_sdp_dir(tmp_path_factory):
    from virasoro_bootstrap.sdp_writer import write_sdp_directory

    base = tmp_path_factory.mktemp("cert")
    with mp.workdps(40):
        problem = _quiet(build_identical_scalar_gap_problem, _spec())
        write_sdp_directory(
            problem.pmp_build.doc, base / "sdp", eval_dps=60,
            sample_values_by_block=problem.sample_values_by_block,
        )
    return problem, base


def _certify_y(problem, base: Path, values):
    from virasoro_bootstrap.certification import certify_dual_from_sdpb_output

    out = base / "sdpb_out"
    out.mkdir(exist_ok=True)
    (out / "y.txt").write_text(f"{len(values)} 1\n" + "\n".join(values) + "\n")
    with mp.workdps(40):
        return _quiet(certify_dual_from_sdpb_output, problem, out, q_gate="off",
                      n_test_points=10)


def test_load_y_vector_rejects_non_finite_components(gap_problem_with_sdp_dir):
    from virasoro_bootstrap.certification import NonFiniteDualVectorError, load_y_vector

    _problem, base = gap_problem_with_sdp_dir
    out = base / "sdpb_out"
    out.mkdir(exist_ok=True)
    (out / "y.txt").write_text("2 1\n0.5\nnan\n")
    with pytest.raises(NonFiniteDualVectorError):
        load_y_vector(out)


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_non_finite_y_does_not_certify(gap_problem_with_sdp_dir, bad):
    """A NaN compared ``< 0`` is False, so a NaN y used to pass every channel;
    an inf y crashed the precision estimate."""
    problem, base = gap_problem_with_sdp_dir
    n = len(problem.F_pairs)
    ok, details = _certify_y(problem, base, [bad] + ["0.1"] * (n - 2))
    assert ok is False
    assert details[0]["reason"] == "non_finite_dual_vector"


def test_unnormalized_full_length_y_does_not_certify(gap_problem_with_sdp_dir):
    """The all-zero functional is non-negative everywhere; only n . y = 1 makes
    a non-negative functional a certificate."""
    problem, base = gap_problem_with_sdp_dir
    n = len(problem.F_pairs)
    ok, details = _certify_y(problem, base, ["0"] * n)
    assert ok is False
    assert details[0]["reason"] == "normalization_violated"
    # the SDPB (N-1)-component vector is reconstructed through n . y = 1, so it
    # is normalized by construction and reaches the positivity check proper
    ok, details = _certify_y(problem, base, ["0"] * (n - 1))
    assert all(d.get("reason") != "normalization_violated" for d in details)
    assert {d["ell"] for d in details} == {0, 2}


# --- certification: exact channels at a recognized singular c -----------------

def test_exact_block_channels_build_at_singular_c():
    """At c = 1 the retained exact table is a skeleton (numerators=None);
    serializing it crashed the true-block certifier."""
    from virasoro_bootstrap.certification import _build_exact_block_channels

    with mp.workdps(40):
        problem = _quiet(build_identical_scalar_gap_problem, _spec(
            c=mp.mpf(1), delta_gap=mp.mpf("0.9"), q_order=10, dps=40,
            pole_selection="nearest",
        ))
        assert problem.exact_tables_by_ell[0].numerators is None
        channels = _build_exact_block_channels(problem)
        value = channels[1].prefactored.evaluate("zzb", mp.mpf("2.7"))[1][0]
    assert mp.isfinite(value) and value != 0


# --- OPE max: the distinguished operator must be a fixed-spin channel ---------

def test_ope_max_refuses_a_spin_star_only_the_tower_carries():
    spec = _spec(
        delta_gap=mp.mpf(1), large_spin_lmin=4, large_spin_twists=(mp.mpf("0.4"),),
        large_spin_npoles=6,
    )
    with mp.workdps(40):
        prepared = _quiet(prepare_identical_scalar_gap_build, spec)
        with pytest.raises(ValueError, match="fixed spins"):
            build_identical_scalar_ope_max_problem(
                spec, spin_star=4, delta_star=mp.mpf("4.4"), prepared=prepared,
            )


# --- precision: builders and readers work at the spec's dps -------------------

def test_normalized_moment_build_does_not_depend_on_ambient_precision():
    from virasoro_bootstrap.moments import (
        MomentWeight,
        build_identical_scalar_normalized_moment_lower_problem,
    )

    spec = _spec(delta_gap=mp.mpf(0), dps=60)
    docs = []
    for ambient in (15, 60):
        with mp.workdps(ambient):
            problem = _quiet(
                build_identical_scalar_normalized_moment_lower_problem,
                spec, MomentWeight.monomial(Delta=1), MomentWeight.one(),
            )
        docs.append(json.dumps(problem.pmp_build.doc))
    assert docs[0] == docs[1]


def test_zbar_half_problem_does_not_depend_on_ambient_precision(tmp_path):
    from fractions import Fraction
    from virasoro_blocks.__main__ import main as blocks_main
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    with mp.workdps(60):
        blocks_main([
            "table", "--family", "virasoro", "--basis", "chiral_F",
            "--central-charge", "1.2", "--h-ext", "0.3", "0.3", "0.3", "0.3",
            "--coordinates", "z", "--z-shift", "-0.6", "--lambda", "5",
            "--order", "14", "--npoles", "6", "--sample-via", "numeric",
            "--format", "json", "-o", str(tmp_path),
        ])
    problems = []
    for ambient in (15, 60):
        with mp.workdps(ambient):
            problems.append(_quiet(
                build_zbar_half_problem, tmp_path / "derivs_chiral.json", dps=60,
            ))
    assert json.dumps(problems[0].pmp_build.doc) == json.dumps(problems[1].pmp_build.doc)
    assert problems[0].spec.c == Fraction(6, 5)
    # the SDP data at the nodes are the file's numeric values (the block keeps its own nodes)
    assert set(problems[0].sample_values_by_block) == {0}


@pytest.fixture
def chiral_gap_block(tmp_path, request):
    from virasoro_blocks.__main__ import main as blocks_main

    blocks_main([
        "table", "--family", "virasoro", "--basis", "chiral_F", "--central-charge", "1.2",
        "--h-ext", ".3", ".3", ".3", ".3", "--coordinates", "z", "--z-shift", "-.6",
        "--lambda", "3", "--order", "8", "--npoles", "4", "--sample-via", "numeric",
        "--precision", "40", "--gap", ".35", "--format", "json", "-o", str(tmp_path),
        "--z-center", getattr(request, "param", ".5"),
    ])
    return tmp_path / "derivs_chiral.json"


@pytest.mark.parametrize("dps", [True, 30.5])
def test_chiral_file_builder_validates_precision_before_loading(tmp_path, dps):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    original = mp.mp.dps
    with pytest.raises(ValueError, match="dps"):
        build_zbar_half_problem(tmp_path / "missing.json", dps=dps)
    assert mp.mp.dps == original


def test_chiral_file_builder_normalizes_integral_precision(chiral_gap_block):
    from fractions import Fraction
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    expected = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
    for dps in (40.0, Fraction(40)):
        actual = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=dps)
        assert type(actual.spec.dps) is int
        assert actual.pmp_build.doc == expected.pmp_build.doc


def test_chiral_continuum_builder_rejects_actual_frozen_point_file(tmp_path):
    from virasoro_blocks.__main__ import main as blocks_main
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    _quiet(blocks_main, [
        "point-table", "--basis", "chiral_F", "--central-charge", "2",
        "--h-ext", ".1", ".1", ".1", ".1", "--z-shift", "-.2",
        "--coordinates", "z", "--lambda", "1", "--order", "4",
        "--precision", "30", "--exchange-h", ".35", "--q-gate", "off",
        "-o", str(tmp_path),
    ])
    path = tmp_path / "derivs_point_0.json"
    assert json.loads(path.read_text())["variable"] == "point"
    with pytest.raises(ValueError, match="variable.*Delta"):
        _quiet(build_zbar_half_problem, path, h_gap=".35", dps=30)


@pytest.mark.parametrize("variable", ["L"])
def test_chiral_continuum_builder_checks_variable_kind(chiral_gap_block, variable):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    doc = json.loads(chiral_gap_block.read_text())
    doc["variable"] = variable
    chiral_gap_block.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="variable.*Delta"):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)


@pytest.mark.parametrize("chiral_gap_block", [".3"], indirect=True)
def test_chiral_crossing_refuses_a_block_expanded_away_from_half(chiral_gap_block):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    assert json.loads(chiral_gap_block.read_text())["z_center"] == "0.3"
    with pytest.raises(ValueError, match="z_center"):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)


def test_chiral_crossing_checks_center_labels_without_rounding(chiral_gap_block):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    document = json.loads(chiral_gap_block.read_text())
    document["z_center"] = "0.5" + "0" * 59 + "1"
    chiral_gap_block.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="z_center"):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)


@pytest.mark.parametrize("weights", [
    [".3", ".3", ".3", "0.3" + "0" * 59 + "1"],
    [".3"] * 3,
])
def test_chiral_crossing_checks_exact_external_labels(chiral_gap_block, weights):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    document = json.loads(chiral_gap_block.read_text())
    document["h_external"] = weights
    chiral_gap_block.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="four identical externals"):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)


def test_chiral_crossing_accepts_equivalent_external_labels(chiral_gap_block):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    expected = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
    document = json.loads(chiral_gap_block.read_text())
    document["h_external"] = ["3/10", ".30", "0.3", "3e-1"]
    chiral_gap_block.write_text(json.dumps(document))
    problem = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
    assert problem.pmp_build.doc == expected.pmp_build.doc


# Offsets of 1e-11, above the roundoff tolerance 10^-(dps/2) at dps 40.
@pytest.mark.parametrize("key,value", [
    ("z_shift", "-0.6" + "0" * 9 + "1"),
])
def test_chiral_crossing_checks_exact_convention_labels(chiral_gap_block, key, value):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    document = json.loads(chiral_gap_block.read_text())
    document[key] = value
    chiral_gap_block.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=key):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)


def test_chiral_exact_certification_uses_chiral_weights_and_blocks(chiral_gap_block):
    from virasoro_blocks.derivs import chiral_z_taylor
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem
    from virasoro_bootstrap.certification import _build_exact_block_channels

    with mp.workdps(40):
        problem = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
        channels = _build_exact_block_channels(problem)
        h = mp.mpf(".8")
        expected = chiral_z_taylor(h, [problem.spec.h_ext] * 4, 4, c=problem.spec.c,
                                  q_order=8, z_shift=-2 * problem.spec.h_ext)
        actual = channels[0].prefactored.evaluate("zzb", h)
        assert len(channels) == 1
        assert all(len(row) == 1 for row in actual)
        for (m, n) in problem.F_pairs:
            assert abs(actual[m][n] - expected[m]) < mp.mpf("1e-30") * max(1, abs(expected[m]))


@pytest.mark.parametrize("exact", [True])
def test_chiral_certification_does_not_add_two_dimensional_spin_channels(chiral_gap_block, exact):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem
    from virasoro_bootstrap.certification import _build_extra_spin_channels

    with mp.workdps(40):
        problem = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
        assert _build_extra_spin_channels(problem, 2, exact_blocks=exact) == []


def test_chiral_global_control_certification_retains_global_family():
    from virasoro_blocks.derivs import global_chiral_z_taylor
    from virasoro_bootstrap.controls.chiral_control import build_chiral_control_problem
    from virasoro_bootstrap.certification import _build_exact_block_channels

    with mp.workdps(40):
        problem = build_chiral_control_problem(mode="global", c=2, h_ext=".3", h_gap=".35",
                                              lam=3, q_order=8, dps=40, n_sample_points=6)
        expected = global_chiral_z_taylor(mp.mpf(".8"), [problem.spec.h_ext] * 4, 4,
                                         z_shift=-2 * problem.spec.h_ext)
        actual = _build_exact_block_channels(problem)[0].prefactored.evaluate("zzb", mp.mpf(".8"))
        for (m, n) in problem.F_pairs:
            assert abs(actual[m][n] - expected[m]) < mp.mpf("1e-30") * max(1, abs(expected[m]))


def test_chiral_file_problem_records_its_pmp_vectors(chiral_gap_block):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    problem = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
    assert problem.normalization_vector == problem.identity_vector
    assert problem.objective_vector == (0,) * len(problem.F_pairs)


def test_chiral_file_problem_manifest_roundtrip(chiral_gap_block, tmp_path):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    with mp.workdps(40):
        problem = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
        path, _ = write_identical_scalar_gap_problem(problem, tmp_path / "saved")
        rebuilt = build_identical_scalar_problem_from_manifest(path)
        assert rebuilt.resolved_method == problem.resolved_method
        assert rebuilt.F_pairs == problem.F_pairs
        assert rebuilt.identity_vector == problem.identity_vector
        assert rebuilt.normalization_vector == problem.normalization_vector
        assert rebuilt.pmp_metadata == problem.pmp_metadata
        original = problem.spin_channels[0].prefactored.evaluate("zzb", mp.mpf(".8"))
        actual = rebuilt.spin_channels[0].prefactored.evaluate("zzb", mp.mpf(".8"))
        for m, n in problem.F_pairs:
            assert abs(actual[m][n] - original[m][n]) < mp.mpf("1e-30") * max(1, abs(original[m][n]))
        manifest = json.loads(path.read_text())
        assert manifest["crossing"]["sector_kind"] == "chiral_F_minus"
        assert manifest["blocks"][0]["kind"] == "chiral"
        block = json.loads((path.parent / manifest["blocks"][0]["path"]).read_text())
        assert block["basis"] == "chiral_F"
        assert block["coordinates"] == ["z"]
        assert block["spin"] is None


def test_chiral_rational_external_survives_export_and_reload(tmp_path, monkeypatch):
    from fractions import Fraction
    from virasoro_blocks.__main__ import main as blocks_main
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest
    from virasoro_bootstrap import chiral_from_blocks

    identity_inputs = []
    identity = chiral_from_blocks._chiral_identity_vector
    def record_identity(*args, **kwargs):
        identity_inputs.append(literal_number(kwargs['h_ext']))
        return identity(*args, **kwargs)
    monkeypatch.setattr(chiral_from_blocks, '_chiral_identity_vector', record_identity)

    blocks_main([
        "table", "--family", "virasoro", "--basis", "chiral_F", "--central-charge", "1.2",
        "--h-ext", "1/3", "1/3", "1/3", "1/3", "--coordinates", "z", "--z-shift=-2/3",
        "--lambda", "3", "--order", "8", "--npoles", "4", "--sample-via", "numeric",
        "--precision", "40", "--format", "json", "-o", str(tmp_path),
    ])
    problem = _quiet(build_zbar_half_problem, tmp_path / "derivs_chiral.json", dps=40)
    path, _ = write_identical_scalar_gap_problem(problem, tmp_path / "saved")
    manifest = json.loads(path.read_text())
    block_path = path.parent / manifest["blocks"][0]["path"]
    reloaded = _quiet(build_zbar_half_problem, block_path, dps=40)
    rebuilt = build_identical_scalar_problem_from_manifest(path)
    for instance in (problem, reloaded, rebuilt):
        assert literal_number(instance.spec.h_ext) == Fraction(1, 3)
    assert identity_inputs and all(value == Fraction(1, 3) for value in identity_inputs)
    block = json.loads(block_path.read_text())
    assert block["h_external"] == ["1/3"] * 4
    # z_shift is metadata, written as a decimal at the metadata digits
    assert abs(Fraction(block["z_shift"]) + Fraction(2, 3)) < Fraction(1, 10**35)
    assert manifest["spec"]["h_ext"] == "1/3"


def test_chiral_export_preserves_binary_external_input(tmp_path):
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem
    from virasoro_bootstrap.controls.chiral_control import build_chiral_control_problem

    with mp.workdps(40):
        h_ext = mp.mpf(1) / 3
        problem = build_chiral_control_problem(
            mode="virasoro_symbolic", c=2, h_ext=h_ext, h_gap=0,
            lam=3, q_order=8, dps=40, npoles=4,
        )
    path, _ = write_identical_scalar_gap_problem(problem, tmp_path)
    manifest = json.loads(path.read_text())
    block_path = path.parent / manifest["blocks"][0]["path"]
    reloaded = _quiet(build_zbar_half_problem, block_path, dps=40)
    assert literal_number(reloaded.spec.h_ext) == literal_number(h_ext)
    block = json.loads(block_path.read_text())
    assert abs(literal_number(block["z_shift"]) + 2 * literal_number(h_ext)) < Fraction(1, 10**35)


@pytest.mark.parametrize("mode", ["gap", "chiral_virasoro"])
def test_problem_builders_preserve_exact_external_labels(tmp_path, mode):
    from fractions import Fraction
    from virasoro_bootstrap.controls.chiral_control import build_chiral_control_problem
    from virasoro_bootstrap.controls.global_control import (
        build_global_identical_scalar_control_problem,
        build_numeric_virasoro_identical_scalar_control_problem,
    )

    if mode.startswith("chiral_"):
        problem = build_chiral_control_problem(
            mode=mode.removeprefix("chiral_"), c="2.5", h_ext="1/3", h_gap=0,
            lam=1, q_order=4, dps=40, npoles=2,
        )
    else:
        builder = {"gap": build_identical_scalar_gap_problem,
                   "global": build_global_identical_scalar_control_problem,
                   "virasoro": build_numeric_virasoro_identical_scalar_control_problem}[mode]
        problem = _quiet(builder, _spec(h_ext="1/3", c="2.5", spin_list=(0,), lam=1, q_order=4, npoles=2))
    assert problem.spec.h_ext == Fraction(1, 3)
    path, _ = write_identical_scalar_gap_problem(problem, tmp_path)
    manifest = json.loads(path.read_text())
    assert manifest["spec"]["h_ext"] == "1/3"
    block = json.loads((path.parent / manifest["blocks"][0]["path"]).read_text())
    assert block["h_external"] == ["1/3"] * 4
    # z_shift is metadata, written as a decimal at the metadata digits
    assert abs(Fraction(block["z_shift"]) + Fraction(2, 3)) < Fraction(1, 10**35)


@pytest.mark.parametrize("mode", ["global", "virasoro_symbolic"])
def test_chiral_control_manifest_rebuilds_a_moved_gap(tmp_path, mode):
    from virasoro_bootstrap.controls.chiral_control import build_chiral_control_problem
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    with mp.workdps(40):
        kwargs = dict(mode=mode, c=2, h_ext=".125", lam=3, q_order=8, dps=40, npoles=4,
                      regularize="on" if mode == "global" else "auto")
        problem = build_chiral_control_problem(h_gap=".5", **kwargs)
        path, _ = write_identical_scalar_gap_problem(problem, tmp_path)
        manifest = json.loads(path.read_text())
        manifest["spec"]["delta_gap"] = ".75"
        path.write_text(json.dumps(manifest))
        rebuilt = build_identical_scalar_problem_from_manifest(path)
        expected = build_chiral_control_problem(h_gap=".75", **kwargs)
        assert rebuilt.F_pairs == expected.F_pairs
        assert rebuilt.resolved_method == expected.resolved_method
        assert rebuilt.used_regularize == expected.used_regularize
        assert rebuilt.spec.regularize == expected.spec.regularize
        for h in (mp.mpf(".75"), mp.mpf("1.25")):
            actual = rebuilt.spin_channels[0].prefactored.evaluate("zzb", h)
            target = expected.spin_channels[0].prefactored.evaluate("zzb", h)
            for m, n in expected.F_pairs:
                assert abs(actual[m][n] - target[m][n]) < mp.mpf("1e-25") * max(1, abs(target[m][n]))


@pytest.mark.parametrize("edit", [None, "stored_shift", "tiny_shift"])
def test_chiral_point_manifest_labels_match_the_frozen_block(chiral_gap_block, tmp_path, edit):
    from virasoro_blocks.point import point_block_table
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem, _relabel_chiral_as_zzb
    from virasoro_bootstrap.identical_scalar import IdenticalScalarSpinChannel
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest
    from virasoro_bootstrap.point_constraint import append_positivity_channels

    with mp.workdps(40):
        problem = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
        h = mp.mpf(".2")
        table = _relabel_chiral_as_zzb(point_block_table(
            ["z"], basis="chiral_F", ell=0, point_value=h, h_externals=[problem.spec.h_ext] * 4,
            lam=3, c=problem.spec.c, q_order=8, z_shift=-2 * problem.spec.h_ext))
        point = IdenticalScalarSpinChannel(ell=0, delta_min=h, serialized=table, prefactored=table)
        problem = append_positivity_channels(problem, [point])
        path, _ = write_identical_scalar_gap_problem(problem, tmp_path / "saved_point")
        if edit == "stored_shift":
            manifest = json.loads(path.read_text())
            ref = next(ref for ref in manifest["blocks"] if ref["kind"] == "point")
            block = json.loads((path.parent / ref["path"]).read_text())
            ref["point_value"] = ref["delta_min"] = block["variable_minus_x"]
            path.write_text(json.dumps(manifest))
            edit = None
        if edit is not None:
            manifest = json.loads(path.read_text())
            ref = next(ref for ref in manifest["blocks"] if ref["kind"] == "point")
            ref["point_value" if edit == "tiny_shift" else edit] = (
                "0.2" + "0" * 59 + "1" if edit == "tiny_shift" else ".3")
            path.write_text(json.dumps(manifest))
            with mp.workdps(15), pytest.raises(ValueError, match="point"):
                build_identical_scalar_problem_from_manifest(path)
        else:
            rebuilt = build_identical_scalar_problem_from_manifest(path)
            assert len(rebuilt.point_channels) == 1
            actual = rebuilt.point_channels[0].prefactored.evaluate("zzb", h)
            target = point.prefactored.evaluate("zzb", h)
            for m, n in problem.F_pairs:
                assert abs(actual[m][n] - target[m][n]) < mp.mpf("1e-30") * max(1, abs(target[m][n]))


@pytest.mark.parametrize("family", ["global", "chiral_virasoro_symbolic"])
def test_control_problems_record_their_pmp_vectors(family):
    from virasoro_bootstrap.controls import global_control, chiral_control

    if family.startswith("chiral_"):
        problem = _quiet(chiral_control.build_chiral_control_problem,
                         mode=family.removeprefix("chiral_"), c=2, h_ext=".125", h_gap=".5",
                         lam=3, q_order=8, dps=40, npoles=4)
    else:
        builder = getattr(global_control, f"build_{family}_identical_scalar_control_problem")
        problem = _quiet(builder, _spec())
    assert problem.normalization_vector == problem.identity_vector
    assert problem.objective_vector == (0,) * len(problem.F_pairs)


@pytest.mark.parametrize("family", ["global", "numeric_virasoro"])
def test_control_manifest_retains_family_and_metadata(tmp_path, family):
    from virasoro_bootstrap.controls import global_control
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    with mp.workdps(40):
        builder = getattr(global_control, f"build_{family}_identical_scalar_control_problem")
        identity_options = {"n_sample_points": 3}
        if family == "global":
            identity_options["global_identity"] = True
        problem = _quiet(builder, _spec(), **identity_options)
        path, _ = write_identical_scalar_gap_problem(problem, tmp_path / family)
        rebuilt = build_identical_scalar_problem_from_manifest(path)
        assert rebuilt.resolved_method == problem.resolved_method
        assert rebuilt.pmp_metadata == problem.pmp_metadata
        assert rebuilt.objective_vector == problem.objective_vector
        assert rebuilt.normalization_vector == problem.normalization_vector
        document = json.loads(path.read_text())
        block = json.loads((path.parent / document["blocks"][0]["path"]).read_text())
        assert block["basis"] == ("global_scalar_G" if family == "global" else "scalar_G")
        document["spec"]["delta_gap"] = "1.3"
        path.write_text(json.dumps(document))
        moved = build_identical_scalar_problem_from_manifest(path)
        expected = _quiet(builder, _spec(delta_gap=mp.mpf("1.3")), **identity_options)
        assert moved.resolved_method == expected.resolved_method
        for delta in (mp.mpf("1.3"), mp.mpf("1.8")):
            actual_grid = moved.spin_channels[0].prefactored.evaluate("zzb", delta)
            expected_grid = expected.spin_channels[0].prefactored.evaluate("zzb", delta)
            for m, n in moved.F_pairs:
                assert abs(actual_grid[m][n] - expected_grid[m][n]) < mp.mpf("1e-25") * max(1, abs(expected_grid[m][n]))


@pytest.mark.parametrize("field, value", [("c", "3.4"), ("regularize", "off")])
def test_global_control_manifest_refuses_stale_identity_inputs(tmp_path, field, value):
    from virasoro_bootstrap.controls.global_control import build_global_identical_scalar_control_problem
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    with mp.workdps(40):
        problem = _quiet(build_global_identical_scalar_control_problem, _spec())
        path, _ = write_identical_scalar_gap_problem(problem, tmp_path)
        document = json.loads(path.read_text())
        document["spec"][field] = value
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="control manifest identity settings"):
            build_identical_scalar_problem_from_manifest(path)


@pytest.mark.parametrize("family, normalized", [
    ("global", False), ("numeric_virasoro", True), ("chiral_global", True), ("chiral_virasoro_symbolic", False),
])
def test_control_moment_manifest_keeps_its_base_problem(tmp_path, family, normalized):
    from virasoro_bootstrap.controls import global_control, chiral_control
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest
    from virasoro_bootstrap.moments import (
        MomentWeight, build_identical_scalar_moment_upper_problem,
        build_identical_scalar_normalized_moment_upper_problem,
    )

    with mp.workdps(40):
        if family.startswith("chiral_"):
            base = chiral_control.build_chiral_control_problem(
                mode=family.removeprefix("chiral_"), c=2, h_ext=".125", h_gap=".5",
                lam=3, q_order=8, dps=40, npoles=4)
            if family == "chiral_global":
                assert abs(mp.mpf(base.pmp_metadata["self_dual_identity_measure"]) - 2 ** mp.mpf(".25")) < mp.mpf("1e-35")
        else:
            builder = getattr(global_control, f"build_{family}_identical_scalar_control_problem")
            options = {"global_identity": True} if family == "global" else {}
            base = _quiet(builder, _spec(), **options)
        weight = MomentWeight.monomial(Delta=1)
        if normalized:
            problem = build_identical_scalar_normalized_moment_upper_problem(
                base.spec, weight, MomentWeight.one(), base_problem=base)
        else:
            problem = build_identical_scalar_moment_upper_problem(base.spec, weight, base_problem=base)
        if base.pmp_metadata and "self_dual_identity_measure" in base.pmp_metadata:
            assert problem.pmp_metadata["self_dual_identity_measure"] == base.pmp_metadata["self_dual_identity_measure"]
        path, _ = write_identical_scalar_gap_problem(problem, tmp_path / family)
        rebuilt = build_identical_scalar_problem_from_manifest(path)
        assert rebuilt.resolved_method == base.resolved_method
        assert rebuilt.problem_kind == problem.problem_kind
        actual_meta = dict(rebuilt.pmp_metadata["moment"])
        expected_meta = dict(problem.pmp_metadata["moment"])
        if normalized:
            for key in ("alpha", "beta"):
                assert mp.mpf(actual_meta.pop(key)) == mp.mpf(expected_meta.pop(key))
        assert actual_meta == expected_meta
        if base.pmp_metadata and "self_dual_identity_measure" in base.pmp_metadata:
            assert rebuilt.pmp_metadata["self_dual_identity_measure"] == base.pmp_metadata["self_dual_identity_measure"]
        for name in ("objective_vector", "normalization_vector"):
            actual, expected = getattr(rebuilt, name), getattr(problem, name)
            assert len(actual) == len(expected)
            assert all(abs(a - b) < mp.mpf("1e-30") * max(1, abs(b)) for a, b in zip(actual, expected))
        def compare_pmp(actual, expected):
            if isinstance(expected, dict):
                assert actual.keys() == expected.keys()
                for key in expected:
                    compare_pmp(actual[key], expected[key])
            elif isinstance(expected, list):
                assert len(actual) == len(expected)
                for a, b in zip(actual, expected):
                    compare_pmp(a, b)
            else:
                a, b = mp.mpf(actual), mp.mpf(expected)
                assert abs(a - b) < mp.mpf("1e-30") * max(1, abs(b))
        compare_pmp(rebuilt.pmp_build.doc, problem.pmp_build.doc)


@pytest.mark.parametrize("shifted", [False, True])
def test_chiral_exact_c_numerics_agree_at_higher_precision(shifted):
    from fractions import Fraction
    from virasoro_bootstrap.controls.chiral_control import _chiral_symbolic_exact_table

    c = Fraction(7, 10) + (Fraction(1, 10**60) if shifted else 0)
    rows = []
    for dps in (40, 100):
        with mp.workdps(dps):
            table = _chiral_symbolic_exact_table(h_ext=".125", lam=2, c=c, q_order=6, chiral_norm=None)
            rows.append(table.evaluate(mp.mpf(".375")))
    with mp.workdps(100):
        for actual, expected in zip(rows[0], rows[1]):
            assert abs(actual[0] - expected[0]) < mp.mpf("1e-30") * max(1, abs(expected[0]))


@pytest.mark.parametrize("gap", ["1", "nan"])
def test_chiral_gap_check_rejects_mismatches_and_nonfinite(chiral_gap_block, gap):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    with pytest.raises(ValueError, match="h_gap"):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=mp.mpf(gap), dps=40)


@pytest.mark.parametrize("kind", ["float"])
def test_chiral_gap_check_accepts_matching_decimal_inputs(chiral_gap_block, kind):
    from fractions import Fraction
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    with mp.workdps(15):
        gap = {"text": ".35", "float": .35, "fraction": Fraction(7, 20), "mpf": mp.mpf(".35")}[kind]
    problem = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=gap, dps=40)
    assert abs(problem.spin_channels[0].delta_min - mp.mpf(".35")) < mp.mpf("1e-15")


@pytest.mark.parametrize("dps, kind", [(20, "text"), (40, "fraction")])
def test_chiral_gap_check_preserves_resolvable_exact_offsets(chiral_gap_block, dps, kind):
    from fractions import Fraction
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    gap = "0.3500000000000000000000000000001"
    if kind == "fraction":
        gap = Fraction(gap)
    with pytest.raises(ValueError, match="h_gap"):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=gap, dps=dps)


def test_chiral_gap_check_retains_supplied_mpf_bits(chiral_gap_block):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem
    with mp.workdps(15):
        gap = mp.mpf('.35')
    with pytest.raises(ValueError, match='h_gap'):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=gap, dps=40)


@pytest.mark.parametrize('overrides,message', [
    ({2: '1.' + '9'*70}, 'unitarity'),
    ({0: '1.' + '0'*69 + '1'}, 'conflicts'),
])
def test_lower_edge_validation_uses_the_exact_side_of_the_boundary(overrides, message):
    from virasoro_bootstrap.identical_scalar import _resolved_delta_min_by_ell
    spec = _spec(delta_gap=1, delta_min_by_spin=overrides, dps=30)
    with mp.workdps(15), pytest.raises(ValueError, match=message):
        _resolved_delta_min_by_ell(spec, spec.spin_list)


@pytest.mark.parametrize('mode', ['virasoro_symbolic', 'global_2d'])
def test_control_builders_keep_exact_gap_labels(mode):
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.controls.chiral_control import build_chiral_control_problem
    gap = Fraction(1, 3) + Fraction(1, 10**70)
    if mode.endswith('_2d'):
        from virasoro_bootstrap.controls.global_control import (
            build_global_identical_scalar_control_problem,
            build_numeric_virasoro_identical_scalar_control_problem)
        spec = _spec(delta_gap=gap, c=2, h_ext='.125', q_order=2, lam=1,
                     dps=30, npoles=0, spin_list=(0,), num_threads=1)
        builder = (build_global_identical_scalar_control_problem if mode == 'global_2d' else
                   build_numeric_virasoro_identical_scalar_control_problem)
        problem = _quiet(builder, spec, n_sample_points=2)
    else:
        problem = _quiet(build_chiral_control_problem, mode=mode, h_gap=gap,
            h_ext='.125', c=2, q_order=2, lam=1, dps=30, npoles=0, n_sample_points=2)
    assert literal_number(problem.spec.delta_gap) == gap
    channel = problem.spin_channels[0]
    assert literal_number(channel.delta_min) == literal_number(channel.serialized.shift) == gap


@pytest.mark.parametrize("gap", ["1/3", "0"])
def test_chiral_gap_check_respects_serialized_significant_digits(tmp_path, gap):
    from fractions import Fraction
    from virasoro_blocks.__main__ import main as blocks_main
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    _quiet(blocks_main, [
        "table", "--basis", "chiral_F", "--central-charge", "2",
        "--h-ext", ".1", ".1", ".1", ".1", "--z-shift", "-.2",
        "--coordinates", "z", "--lambda", "1", "--order", "4",
        "--npoles", "2", "--precision", "20", "--gap", gap,
        "--q-gate", "off", "-o", str(tmp_path),
    ])
    problem = _quiet(build_zbar_half_problem, tmp_path / "derivs_chiral.json",
                     h_gap=Fraction(gap), dps=40)
    assert abs(problem.spin_channels[0].delta_min - mp.mpf(gap)) < mp.mpf("1e-15")
    offset = Fraction("1e-10" if gap == "1/3" else "1e-60")
    with pytest.raises(ValueError, match="h_gap"):
        _quiet(build_zbar_half_problem, tmp_path / "derivs_chiral.json",
               h_gap=Fraction(gap) + offset, dps=40)


@pytest.mark.parametrize("field, value", [("order", 8.5), ("npoles", True)])
def test_chiral_loader_refuses_invalid_integer_metadata(chiral_gap_block, field, value):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    document = json.loads(chiral_gap_block.read_text())
    document[field] = value
    chiral_gap_block.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=field):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)


@pytest.mark.parametrize("lam", [True, 5])
def test_chiral_loader_refuses_unavailable_or_invalid_derivative_order(chiral_gap_block, lam):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    with pytest.raises(ValueError, match="lam"):
        _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40, lam=lam)


@pytest.mark.parametrize("lam", [2])
def test_chiral_loader_can_select_a_lower_derivative_order(chiral_gap_block, lam):
    from virasoro_bootstrap.chiral_from_blocks import build_zbar_half_problem

    full = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40)
    reduced = _quiet(build_zbar_half_problem, chiral_gap_block, h_gap=".35", dps=40, lam=lam)
    assert full.F_pairs == ((1, 0), (3, 0))
    assert reduced.F_pairs == ((1, 0),)
    assert reduced.spec.lam == lam
    assert reduced.identity_vector == full.identity_vector[:1]
    assert reduced.pmp_build.doc["normalization"] == full.pmp_build.doc["normalization"][:1]


def test_manifest_reload_does_not_depend_on_ambient_precision(tmp_path):
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    with mp.workdps(40):
        problem = _quiet(build_identical_scalar_gap_problem, _spec(
            point_constraints=((0, mp.mpf("0.7")),),
        ))
        manifest_path, _ = write_identical_scalar_gap_problem(problem, tmp_path / "m")
    rebuilt = []
    for ambient in (15, 40):
        with mp.workdps(ambient):
            rebuilt.append(build_identical_scalar_problem_from_manifest(manifest_path))
    with mp.workdps(40):
        assert rebuilt[0].spec.c == mp.mpf("3.3")
        assert rebuilt[0].spec.h_ext == mp.mpf("0.1")
        assert rebuilt[0].spec.point_constraints[0][1] == mp.mpf("0.7")
    assert json.dumps(rebuilt[0].pmp_build.doc) == json.dumps(rebuilt[1].pmp_build.doc)


# --- physical-block moment certification honors spin_filter ------------------

def test_physical_moment_row_honours_spin_filter():
    from virasoro_bootstrap.certification import (
        _build_exact_block_channels,
        _normalized_moment_cert_weights,
        _normalized_moment_physical_row_from_table,
    )
    from virasoro_bootstrap.moments import (
        MomentTerm,
        MomentWeight,
        build_identical_scalar_normalized_moment_lower_problem,
    )

    with mp.workdps(40):
        spec = _spec(delta_gap=mp.mpf(0))
        numerator = MomentWeight((MomentTerm(1, {"Delta": 1}),), spin_filter=0)
        denominator = MomentWeight((MomentTerm(1, {}),), spin_filter=0)
        problem = _quiet(build_identical_scalar_normalized_moment_lower_problem,
                         spec, numerator, denominator)
        channel = _build_exact_block_channels(problem)[1]
        n = len(problem.F_pairs)
        y = [mp.mpf(0)] * n + [mp.mpf(0), mp.mpf(1)]  # the denominator component only
        delta = mp.mpf(3)
        table = channel.prefactored.evaluate("zzb", delta)
        row = _normalized_moment_physical_row_from_table(
            y, problem, channel, delta, table,
            weights=_normalized_moment_cert_weights(problem),
        )
    assert row == 0


# --- hpc: the rebuilt problem is the solved problem ---------------------------

class _Captured(Exception):
    pass


def _capture_prepare(monkeypatch):
    import virasoro_bootstrap.identical_scalar as identical_scalar_mod

    specs = []

    def fake_prepare(spec):
        specs.append(spec)
        raise _Captured

    monkeypatch.setattr(identical_scalar_mod, "prepare_identical_scalar_gap_build", fake_prepare)
    return specs


def _ctx(tmp_path, params):
    from virasoro_bootstrap.hpc.journal import Journal
    from virasoro_bootstrap.hpc.tasks import TaskContext

    return TaskContext(params=params, output_dir=tmp_path,
                       journal=Journal(tmp_path / "progress.json"))


def test_scalar_gap_bisection_spec_matches_the_recertify_rebuild(tmp_path, monkeypatch):
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_number
    import virasoro_bootstrap.identical_scalar as identical_scalar_mod
    from virasoro_bootstrap.hpc.recertify import build_scalar_gap_problem
    from virasoro_bootstrap.hpc.tasks import scalar_gap_bisection

    params = {"c": "3.3", "lam": 7, "dps": 30, "spins": [0, 2], "pole_selection": "nearest",
              "gap_lo": ".35", "large_spin_pole_selection": "residue"}
    with mp.workdps(150):
        params["twist_gap"] = mp.mpf('.35') + mp.mpf('1e-70')
    specs = _capture_prepare(monkeypatch)
    result = scalar_gap_bisection(_ctx(tmp_path, params))
    assert result["reason"] == "prepare_error:_Captured"
    solved = specs[0]
    assert literal_number(solved.delta_gap) == Fraction(7, 20)
    assert literal_number(solved.large_spin_twists[0]) == literal_number(params['twist_gap'])
    assert solved.pole_selection == "nearest"
    assert solved.large_spin_pole_selection == "residue"

    monkeypatch.setattr(identical_scalar_mod, "build_identical_scalar_gap_problem", lambda spec: spec)
    with mp.workdps(solved.dps):
        rebuilt = build_scalar_gap_problem(params, solved.delta_gap)
    assert rebuilt == solved


def test_moment_spec_helper_matches_the_solved_spec(tmp_path, monkeypatch):
    from virasoro_bootstrap.hpc.tasks import moment_nu1_over_nu0, _moment_nu1_over_nu0_spec, _load_default_params_profile

    params = {
        "c": "8/7", "delta_ext": "0.2", "lam": 19, "delta_gap": "0.05",
        "all_spins_twist_gap": True, "pole_selection": "nearest",
        "large_spin_pole_selection": "residue", "shared_sample_points": True,
        "source_output_dir": str(tmp_path / "source"), "moment_cert_q_order": 40,
    }
    specs = _capture_prepare(monkeypatch)
    moment_nu1_over_nu0(_ctx(tmp_path / "solve", params))
    assert len(specs) == 1
    solved = specs[0]
    with mp.workdps(solved.dps):
        rechecked = _moment_nu1_over_nu0_spec(params, _load_default_params_profile(19),
                                               lam=19, dps=solved.dps, num_threads=solved.num_threads)
    assert float(solved.delta_gap) == 0.05
    assert solved.delta_min_by_spin
    assert solved.pole_selection == "nearest"
    assert solved.shared_sample_points is True
    assert rechecked == solved


def test_worker_journal_is_reset_when_the_spec_changes(tmp_path):
    from virasoro_bootstrap.hpc.worker import _open_journal

    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"spec_hash": "old", "probes": {"gap_1p0": {"allowed": True}}}))
    journal = _open_journal(path, "new")
    assert "probes" not in journal.data
    assert journal.data["spec_hash"] == "new"
    assert (tmp_path / "progress.old.json").exists()
    # a journal of the same spec is resumed; one without a spec hash (but of this convention) is adopted
    journal.update(probes={"gap_2p0": {"allowed": False}})
    assert "gap_2p0" in _open_journal(path, "new").data["probes"]
    path.write_text(json.dumps({"probes": {"gap_3p0": {"allowed": True}}, "scalar_G_spin0": "single"}))
    adopted = _open_journal(path, "new")
    assert "gap_3p0" in adopted.data["probes"] and adopted.data["spec_hash"] == "new"




def test_collect_reports_a_result_of_another_spec_as_stale(tmp_path):
    from virasoro_bootstrap.hpc import collect
    from virasoro_bootstrap.hpc.worker import spec_hash

    run_dir = tmp_path / "run"
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "tasks" / "0000").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "r", "task": "t", "n_tasks": 1}))
    (run_dir / "specs" / "0000.json").write_text(json.dumps(
        {"task": "t", "params": {"c": "2"}, "output_dir": "tasks/0000"}
    ))
    result_path = run_dir / "tasks" / "0000" / "result.json"
    result_path.write_text(json.dumps(
        {"spec_hash": spec_hash("t", {"c": "1"}), "result": {"delta_gap": "1.0"}}
    ))
    _manifest, views = collect.load_run(run_dir)
    assert views[0].state == "stale" and views[0].result is None
    result_path.write_text(json.dumps(
        {"spec_hash": spec_hash("t", {"c": "2"}), "result": {"delta_gap": "1.0"}}
    ))
    _manifest, views = collect.load_run(run_dir)
    assert views[0].state == "done"




def test_spec_canonicalizes_list_fields_to_tuples():
    spec = _spec(spin_list=[0, 2], large_spin_lmin=4,
                 large_spin_twists=[mp.mpf("0.4")], point_constraints=[[0, mp.mpf("0.7")]])
    assert spec.spin_list == (0, 2)
    assert spec.large_spin_twists == (mp.mpf("0.4"),)
    assert spec.point_constraints == ((0, mp.mpf("0.7")),)
    assert hash(spec) == hash(_spec(spin_list=(0, 2), large_spin_lmin=4,
                                    large_spin_twists=(mp.mpf("0.4"),),
                                    point_constraints=((0, mp.mpf("0.7")),)))


# --- point constraints: coordinate, partition, normalization ------------------

@pytest.mark.parametrize("ell, delta", [(2.5, 3), (True, 3), (2, "1.9")])
def test_direct_point_channel_refuses_invalid_operator_before_sampling(monkeypatch, ell, delta):
    import virasoro_bootstrap.point_constraint as points

    calls = []
    monkeypatch.setattr(points, "point_block_table", lambda *args, **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="point constraint"):
        points.build_point_channel(ell, delta, h_ext=".1", c=3, q_order=8, lam=1)
    assert not calls


@pytest.mark.parametrize("ell", [False])
def test_spec_point_constraints_refuse_invalid_spin_types(ell):
    from virasoro_bootstrap.identical_scalar import _resolved_point_constraints

    with pytest.raises(ValueError, match="point constraint spin"):
        _resolved_point_constraints(_spec(point_constraints=((ell, 3),)))


@pytest.mark.parametrize("ell", [True])
def test_fixed_spin_validation_refuses_invalid_types(ell):
    from virasoro_bootstrap.identical_scalar import _validate_even_spins

    with pytest.raises(ValueError, match="spin_list entries must be integers"):
        _validate_even_spins((0, ell))


def test_point_constraint_helpers_on_a_non_zzb_coordinate_and_through_the_manifest(tmp_path):
    from virasoro_bootstrap.certification import _build_exact_block_channels
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest
    from virasoro_bootstrap.point_constraint import (
        append_positivity_channels,
        input_fixed_operators,
        point_constraint_channel,
    )

    with mp.workdps(40):
        xt = _quiet(build_identical_scalar_gap_problem, _spec(delta_gap=mp.mpf(1), coordinate="xt"))
        spec = xt.spec
        point = point_constraint_channel(
            xt.spin_channels[0], mp.mpf("0.5"), coordinate="xt", h_ext=spec.h_ext, c=spec.c,
            q_order=12, lam=3, z_shift=-2 * spec.h_ext, zbar_shift=-2 * spec.h_ext,
        )
        appended = append_positivity_channels(xt, [point])
        assert appended.point_channels[-1] is point
        input_fixed_operators(xt, [(xt.spin_channels[0], mp.mpf("1.5"), mp.mpf("0.1"))])

        problem = _quiet(build_identical_scalar_gap_problem, _spec(delta_gap=mp.mpf(1)))
        spec = problem.spec
        point = point_constraint_channel(
            problem.spin_channels[0], mp.mpf("0.5"), h_ext=spec.h_ext, c=spec.c,
            q_order=12, lam=3, z_shift=-2 * spec.h_ext, zbar_shift=-2 * spec.h_ext,
        )
        appended = append_positivity_channels(problem, [point])
        # the exact certifier keeps the appended point channel
        assert any(ch is point for ch in _build_exact_block_channels(appended))
        manifest_path, _ = write_identical_scalar_gap_problem(appended, tmp_path / "append")
        reloaded = build_identical_scalar_problem_from_manifest(manifest_path)
        assert len(reloaded.pmp_build.doc["PositiveMatrixWithPrefactorArray"]) == 3

        fixed = input_fixed_operators(
            problem, [(problem.spin_channels[0], mp.mpf("1.5"), mp.mpf("0.1"))],
        )
        doc_normalization = [mp.mpf(v) for v in fixed.pmp_build.doc["normalization"]]
        assert all(abs(a - b) < mp.mpf(10) ** -30
                   for a, b in zip(doc_normalization, fixed.normalization_vector))
        manifest_path, _ = write_identical_scalar_gap_problem(fixed, tmp_path / "fixed")
        reloaded = build_identical_scalar_problem_from_manifest(manifest_path)
        assert reloaded.pmp_build.doc["normalization"] == fixed.pmp_build.doc["normalization"]

        with pytest.raises(ValueError, match="point-constraint channels only"):
            append_positivity_channels(problem, [problem.spin_channels[1]])


@pytest.mark.parametrize("operation", ["append", "fixed"])
def test_point_problem_modifications_preserve_requested_precision(operation):
    from virasoro_bootstrap.point_constraint import (
        append_positivity_channels, input_fixed_operators, point_constraint_channel,
    )

    with mp.workdps(40):
        problem = _quiet(build_identical_scalar_gap_problem, _spec())
        point = point_constraint_channel(
            problem.spin_channels[0], ".5", h_ext=problem.spec.h_ext, c=problem.spec.c,
            q_order=12, lam=3, z_shift=-2 * problem.spec.h_ext, zbar_shift=-2 * problem.spec.h_ext,
        )
        def modify():
            if operation == "append":
                return append_positivity_channels(problem, [point])
            return input_fixed_operators(problem, [(problem.spin_channels[0], "1.5", "0.1234567890123456789")])
        expected = modify()
    with mp.workdps(15):
        actual = modify()
        assert mp.mp.dps == 15
    assert actual.pmp_build.doc == expected.pmp_build.doc
    assert actual.identity_vector == expected.identity_vector


@pytest.mark.parametrize("kind", ["upper", "max"])
def test_fixed_operators_preserve_ope_objective_roles(tmp_path, kind):
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest
    from virasoro_bootstrap.point_constraint import input_fixed_operators, normalization_with_fixed_operators

    with mp.workdps(40):
        if kind == "max":
            problem = _quiet(build_identical_scalar_ope_max_problem, _spec(), delta_star=".5")
        else:
            problem = _quiet(build_identical_scalar_ope_bound_problem, _spec(), delta_star=".5", direction=kind)
        known = [(problem.spin_channels[0], "1.5", ".2")]
        expected = normalization_with_fixed_operators(problem.identity_vector, problem.F_pairs, known)
        modified = input_fixed_operators(problem, known)
        assert modified.objective_vector == expected
        assert modified.identity_vector == expected
        assert modified.normalization_vector == problem.normalization_vector
        assert modified.normalization_operator == problem.normalization_operator
        assert modified.problem_kind == problem.problem_kind
        assert tuple(mp.mpf(value) for value in modified.pmp_build.doc["objective"]) == expected
        path, _ = write_identical_scalar_gap_problem(modified, tmp_path / kind)
        rebuilt = build_identical_scalar_problem_from_manifest(path)
        assert rebuilt.objective_vector == expected
        assert rebuilt.normalization_vector == problem.normalization_vector


# --- manifest: tower block files record the tower's pole count ----------------

def test_fixed_operators_refuse_moment_problem_reinterpretation():
    from virasoro_bootstrap.moments import MomentWeight, build_identical_scalar_moment_upper_problem
    from virasoro_bootstrap.point_constraint import input_fixed_operators

    problem = _quiet(build_identical_scalar_moment_upper_problem, _spec(), MomentWeight.one())
    with pytest.raises(ValueError, match="does not support problem kind 'moment_upper'"):
        input_fixed_operators(problem, [(problem.spin_channels[0], "1.5", ".2")])


@pytest.mark.parametrize("kind", ["lower", "fixed"])
def test_ope_manifest_preserves_normalization_operator_metadata(tmp_path, kind):
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    with mp.workdps(40):
        if kind == "fixed":
            problem = _quiet(build_identical_scalar_fixed_ope_problem, _spec(),
                             delta_star=".5", ope_squared=".1234567890123456789")
        else:
            problem = _quiet(build_identical_scalar_ope_bound_problem, _spec(),
                             delta_star=".5", direction=kind)
        path, _ = write_identical_scalar_gap_problem(problem, tmp_path / kind)
        document = json.loads(path.read_text())
        saved = document["pmp"]["normalization_operator"]
        rebuilt = build_identical_scalar_problem_from_manifest(path)
        expected = problem.normalization_operator
        assert saved.keys() == expected.keys()
        assert rebuilt.normalization_operator.keys() == expected.keys()
        if kind == "fixed":
            from virasoro_blocks.numeric_input import literal_number
            assert literal_number(saved["ope_squared"]) == expected["ope_squared"]
            assert rebuilt.normalization_operator["ope_squared"] == expected["ope_squared"]
        else:
            assert saved["direction"] == rebuilt.normalization_operator["direction"] == kind
        assert rebuilt.problem_kind == problem.problem_kind
        assert rebuilt.objective_vector == problem.objective_vector
        assert rebuilt.normalization_vector == problem.normalization_vector


def test_tower_block_files_record_large_spin_npoles(tmp_path):
    with mp.workdps(40):
        problem = _quiet(build_identical_scalar_gap_problem, _spec(
            large_spin_lmin=4, large_spin_twists=(mp.mpf("0.4"),), large_spin_npoles=4,
        ))
        manifest_path, _ = write_identical_scalar_gap_problem(problem, tmp_path / "m")
    manifest = json.loads(Path(manifest_path).read_text())
    tower = [ref for ref in manifest["blocks"] if ref["kind"] == "large_spin"]
    assert tower
    doc = json.loads((Path(manifest_path).parent / tower[0]["path"]).read_text())
    assert doc["npoles"] == 4


# --- validation of isolated operators and tower twists ------------------------

@pytest.mark.parametrize("point, message", [
    ((2.5, mp.mpf(3)), "integer"),
    ((1, mp.mpf(3)), "even"),
    ((2, mp.mpf("1.5")), "unitarity"),
])
def test_invalid_point_constraints_are_rejected(point, message):
    with mp.workdps(40), pytest.raises(ValueError, match=message):
        prepare_identical_scalar_gap_build(_spec(point_constraints=(point,)))


def test_negative_tower_twists_are_rejected():
    with mp.workdps(40), pytest.raises(ValueError, match="large_spin_twists"):
        prepare_identical_scalar_gap_build(_spec(
            large_spin_lmin=4, large_spin_twists=(mp.mpf("-0.1"),),
        ))


@pytest.mark.parametrize("builder, kwargs", [
    (build_identical_scalar_ope_bound_problem, {"spin_star": 2.5, "delta_star": 3}),
    (build_identical_scalar_ope_bound_problem, {"spin_star": 0, "delta_star": -1}),
    (build_identical_scalar_fixed_ope_problem, {"spin_star": 1, "delta_star": 3, "ope_squared": 1}),
    (build_identical_scalar_ope_max_problem, {"spin_star": 2.5}),
])
def test_invalid_distinguished_operators_are_rejected(builder, kwargs):
    with mp.workdps(40), pytest.raises(ValueError):
        builder(_spec(), **kwargs)


@pytest.mark.parametrize("option", [
    {"large_spin_lmin": 4, "large_spin_twists": (mp.mpf(1),)},
    {"shared_sample_points": True},
])
def test_control_problems_reject_options_they_cannot_honour(option):
    """The global / numeric-Virasoro control builders used to drop large-spin tails, point constraints and shared
    sample points silently; they now fail closed (and read spec.n_sample_points)."""
    from virasoro_bootstrap import (
        IdenticalScalarGapSpec,
        build_global_identical_scalar_control_problem,
        build_numeric_virasoro_identical_scalar_control_problem,
    )

    spec = IdenticalScalarGapSpec(
        c=mp.mpf(8), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf("1.5"), delta_min_by_spin={}, lam=3,
        spin_list=(0, 2), q_order=20, dps=40, npoles=6, pole_selection="residue", num_threads=1, **option)
    for builder in (build_global_identical_scalar_control_problem,
                    build_numeric_virasoro_identical_scalar_control_problem):
        with pytest.raises(ValueError, match="does not support"):
            builder(spec)


def test_control_sampling_inputs_fail_before_numerical_work(monkeypatch):
    from virasoro_bootstrap.controls import global_control as G, chiral_control as C

    def unexpected(*args, **kwargs):
        pytest.fail('invalid control reached block evaluation')
    monkeypatch.setattr(C, '_chiral_block_coeffs', unexpected)
    monkeypatch.setattr(G, '_numeric_identity_vector', unexpected)
    monkeypatch.setattr(G, 'global_scalar_G_interpolated_table', unexpected)
    for coordinate in G._NUMERIC_CONTROL_BUILDERS:
        monkeypatch.setitem(G._NUMERIC_CONTROL_BUILDERS, coordinate, unexpected)
    chiral = dict(mode='global', h_ext='.1', h_gap=1, lam=1)
    numeric = dict(c=32, q_order=4, ell=0, h_ext='.1', delta_min=1, lam=1)
    with pytest.raises(ValueError, match='global_identity'):
        G.build_global_identical_scalar_control_problem(_spec(), global_identity='false')
    with pytest.raises(ValueError, match='n_sample_points'):
        G.build_global_identical_scalar_control_problem(_spec(lam=1), n_sample_points=2.5)
    with pytest.raises(ValueError, match='n_sample_points'):
        G.build_global_scalar_spin_channel(ell=0, h_ext='.1', delta_min=1, lam=1, n_sample_points=-1)
    for options in [{'n_sample_points': True}, {'node_search_dps': 0}, {'sample_x': (.7, '.7')},
                    {'sample_x': (mp.nan,)}, {'sample_x': (0, 1), 'n_sample_points': 3}]:
        with pytest.raises(ValueError):
            C.build_chiral_channel(**chiral, **options)
        with pytest.raises(ValueError):
            G.build_numeric_virasoro_scalar_spin_channel(**numeric, **options)


def test_chiral_control_validates_its_own_spec_before_building(monkeypatch):
    from virasoro_bootstrap.controls import chiral_control as C
    def unexpected(*args, **kwargs):
        pytest.fail('invalid control reached block evaluation')
    monkeypatch.setattr(C, '_chiral_block_coeffs', unexpected)
    common = dict(mode='global', c=32, h_ext='.1', h_gap=1, lam=1, dps=20)
    for field, value in [('lam', True), ('dps', 9), ('q_order', 0),
                         ('explicit_sample_points', 'false'), ('regularize', 'typo')]:
        with pytest.raises(ValueError, match=field):
            C.build_chiral_control_problem(**(common | {field: value}))
    with pytest.raises(ValueError, match='does not apply'):
        C.build_chiral_control_problem(**(common | {'mode': 'virasoro_symbolic', 'node_search_dps': 20}))
    with pytest.raises(ValueError, match='mode'):
        C.build_chiral_channel(mode='typo', h_ext='.1', h_gap=1, lam=1)


def test_control_float_samples_interpolate_at_their_declared_decimal_labels():
    from fractions import Fraction
    from virasoro_blocks.derivs import global_chiral_z_taylor, G_zzb_table
    from virasoro_bootstrap.controls import global_control as G, chiral_control as C
    with mp.workdps(60):
        chiral = C.build_chiral_channel(mode='global', h_ext='.1', h_gap=1, lam=1,
                                       sample_x=iter((.7, 1.2)))
        numeric = G.build_numeric_virasoro_scalar_spin_channel(c=32, q_order=4, ell=0,
                     h_ext='.1', delta_min=1, lam=1, sample_x=(.7, 1.2))
        assert chiral.serialized.sample_points_x == numeric.serialized.sample_points_x == (Fraction(7, 10), Fraction(6, 5))
        for label in ['1.7', '2.2']:
            h = mp.mpf(label)
            actual = chiral.serialized.evaluate('zzb', h)
            expected = global_chiral_z_taylor(label, ['.1']*4, 2, z_shift='-.2')
            assert all(abs(actual[m][0]-expected[m])/max(1, abs(expected[m])) < mp.mpf('1e-56') for m in range(2))
            actual = numeric.serialized.evaluate('zzb', h)
            expected = G_zzb_table(label, 0, ['.1']*4, 1, 1, c=32, q_order=4, z_shift='-.2')
            assert all(abs(actual[m][n]-expected[m][n])/max(1, abs(expected[m][n])) < mp.mpf('1e-56')
                       for m in range(2) for n in range(2))
    with mp.workdps(15):
        native = mp.mpf('.7')
    with mp.workdps(60):
        channel = C.build_chiral_channel(mode='global', h_ext='.1', h_gap=1, lam=0, sample_x=(native,))
        from virasoro_blocks.numeric_input import literal_number
        assert channel.serialized.sample_points_x[0] == literal_number(native) != Fraction(7, 10)


def test_control_specs_retain_resolved_sampling_overrides():
    from virasoro_bootstrap.controls import global_control as G
    spec = _spec(c=32, h_ext='.1', lam=1, q_order=4, dps=30, spin_list=(0,))
    global_problem = G.build_global_identical_scalar_control_problem(spec, n_sample_points=3, global_identity=True)
    assert global_problem.spec.npoles == 3
    fresh = G.build_global_identical_scalar_control_problem(global_problem.spec, global_identity=True)
    assert fresh.pmp_build.doc == global_problem.pmp_build.doc
    numeric = G.build_numeric_virasoro_identical_scalar_control_problem(spec, n_sample_points=3)
    assert numeric.spec.n_sample_points == 3
    fresh = G.build_numeric_virasoro_identical_scalar_control_problem(numeric.spec)
    assert fresh.pmp_build.doc == numeric.pmp_build.doc


# --- tower verification and the per-entry shortfall report ---------------------

def test_tower_slices_are_verified_like_the_fixed_spin_blocks():
    with mp.workdps(40):
        prepared = _quiet(prepare_identical_scalar_gap_build, _spec(
            large_spin_lmin=4, large_spin_twists=(mp.mpf("0.4"),), large_spin_npoles=4,
        ))
    (tower,) = prepared.large_spin_channels
    assert tower.serialized.verified_digits is not None and tower.serialized.verification is not None


def test_a_block_entry_without_a_correct_digit_is_reported():
    """Lambda 19, c = 1.00001, npoles 84: spin 0 at -1.9 digits per entry and 5.6 against its largest entry was
    silent (only the largest-entry measure decided) and gave bounds the exact c = 1 theories violate."""
    from types import SimpleNamespace

    from virasoro_bootstrap.identical_scalar import _gap_precision_request, _warn_block_shortfalls
    from virasoro_blocks.verified_build import Verification

    request = _gap_precision_request(_spec(dps=240))

    def channel(ell, per_entry, largest, twist=None):
        v = Verification(digits=per_entry, largest_digits=largest, cause="npoles", far_x=100)
        return SimpleNamespace(ell=ell, twist=twist, serialized=SimpleNamespace(verification=v))

    with pytest.warns(RuntimeWarning, match=r"the spin-0 block \(npoles 84\).*per entry.*raise npoles"):
        _warn_block_shortfalls([channel(0, -1.9, 5.6), channel(2, 12.0, 14.0)], request, npoles=84, work=240)
    with pytest.warns(RuntimeWarning, match=r"tower slice at twist 8\.8 \(L >= 26, large_spin_npoles 12\).*"
                                            r"raise large_spin_npoles"):
        _warn_block_shortfalls([], request, npoles=84, work=240, tower_npoles=12,
                               tower_channels=[channel(26, -2.4, 19.7, twist=mp.mpf("8.8"))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)      # every entry at 3 digits or more: the npoles asked for
        _warn_block_shortfalls([channel(0, 5.0, 12.0)], request, npoles=84, work=240)


def test_moment_tower_constrains_twist_zero_without_a_twist_gap():
    """double_twist_set is the double twists alone; the moment problem adds the unitarity threshold twist 0 when no
    twist gap is assumed, and takes an explicit twist list as given."""
    from virasoro_bootstrap.hpc.tasks import _moment_nu1_over_nu0_spec
    from virasoro_bootstrap.large_spin import double_twist_set

    profile = {"q_order": 10, "npoles": 8, "spin_list": [0, 2], "large_spin_npoles": 4}
    p = {"c": "1.00001", "delta_ext": "1.0", "large_spin_double_twists": 3}
    with mp.workdps(40):
        assert [float(t) for t in double_twist_set(mp.mpf("0.5"), 3)] == [2.0, 4.0, 6.0]
        spec = _moment_nu1_over_nu0_spec(p, profile, lam=3, dps=40, num_threads=1)
        assert [float(t) for t in spec.large_spin_twists] == [0.0, 2.0, 4.0, 6.0]
        gapped = _moment_nu1_over_nu0_spec(dict(p, delta_gap="0.5", all_spins_twist_gap=True), profile, lam=3, dps=40,
                                           num_threads=1)
        assert [float(t) for t in gapped.large_spin_twists] == [0.5, 2.0, 4.0, 6.0]
        explicit = _moment_nu1_over_nu0_spec(dict(p, large_spin_twists=["2", "4"]), profile, lam=3, dps=40,
                                             num_threads=1)
        assert [float(t) for t in explicit.large_spin_twists] == [2.0, 4.0]


def test_low_twist_tower_slices_cover_the_twists_below_the_double_twists():
    from virasoro_bootstrap.hpc.tasks import _moment_nu1_over_nu0_spec

    profile = {"q_order": 10, "npoles": 8, "spin_list": [0, 2], "large_spin_npoles": 4}
    p = {"c": "1.00001", "delta_ext": "1.0", "large_spin_double_twists": 3, "large_spin_low_twist_step": "0.5"}
    with mp.workdps(40):
        spec = _moment_nu1_over_nu0_spec(p, profile, lam=3, dps=40, num_threads=1)
        assert [float(t) for t in spec.large_spin_twists] == [0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0]
        with pytest.raises(ValueError, match="large_spin_low_twist_step"):
            _moment_nu1_over_nu0_spec(dict(p, large_spin_low_twist_step="0"), profile, lam=3, dps=40, num_threads=1)


@pytest.mark.parametrize('operation', ['prefactor', 'normalization', 'complex_poles'])
def test_crossing_transforms_preserve_stored_table_precision(operation):
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_bootstrap.crossing import apply_prefactor_to_table, apply_norm_base_scaling, drop_complex_pole_pairs

    with mp.workdps(100):
        source = SerializedDerivTable(0, 1, (), ('zzb',),
                                      {'zzb': [[Polynomial([mp.mpf('1e50')+mp.pi, -mp.mpf('1e50')])]]},
                                      working_dps=100)
    with mp.workdps(15):
        transformed = (apply_prefactor_to_table(source, [[1]]) if operation == 'prefactor' else
                       apply_norm_base_scaling(source, 1) if operation == 'normalization' else
                       drop_complex_pole_pairs(source))
        assert transformed.working_dps == 100
        assert mp.mp.dps == 15
    with mp.workdps(70):
        assert abs(transformed.evaluate('zzb', 1)[0][0]-mp.pi) < mp.mpf('1e-48')


@pytest.mark.parametrize("c_input", [".7", "1." + "0" * 59 + "1"])
def test_hpc_scalar_tasks_preserve_exact_c_labels(tmp_path, monkeypatch, c_input):
    import virasoro_bootstrap.identical_scalar as scalar
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.hpc.recertify import build_scalar_gap_problem
    from virasoro_bootstrap.hpc.tasks import scalar_gap_bisection

    params = {"c": c_input, "lam": 7, "dps": 30, "spins": [0, 2], "tower": False}
    specs = _capture_prepare(monkeypatch)
    scalar_gap_bisection(_ctx(tmp_path, params))
    assert specs[0].c == literal_number(c_input)
    monkeypatch.setattr(scalar, "build_identical_scalar_gap_problem", lambda spec: spec)
    with mp.workdps(30):
        rebuilt = build_scalar_gap_problem(params, specs[0].delta_gap)
    assert rebuilt.c == literal_number(c_input)


@pytest.mark.parametrize("c_input", ["1." + "0" * 59 + "1"])
def test_hpc_moment_task_preserves_exact_c_labels(tmp_path, monkeypatch, c_input):
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.hpc.tasks import moment_nu1_over_nu0

    params = {"c": c_input, "delta_ext": ".2", "lam": 19, "dps": 30,
              "source_output_dir": str(tmp_path / "source"), "moment_cert_q_order": 40}
    specs = _capture_prepare(monkeypatch)
    moment_nu1_over_nu0(_ctx(tmp_path / "solve", params))
    assert len(specs) == 1
    assert specs[0].c == literal_number(c_input)


@pytest.mark.parametrize("tamper", [False, True])
def test_recertification_uses_saved_problem_not_rounded_name_or_task_params(tmp_path, monkeypatch, tamper):
    import virasoro_bootstrap.hpc.recertify as recertify
    import virasoro_bootstrap.certification as certification

    name = "gap_0p12345678901234567"
    task_dir = tmp_path / "tasks/custom/shared"
    (tmp_path / "specs").mkdir()
    # A stale task spec and a rounded directory name must not redefine the SDP.
    (tmp_path / "specs" / "0000.json").write_text(json.dumps({
        "task": "scalar_gap_bisection", "params": {"c": "4", "lam": 7, "dps": 30},
        "output_dir": str(task_dir) if tamper else "tasks/custom/shared",
    }))
    probe_dir = task_dir / name
    with mp.workdps(70):
        saved = _quiet(build_identical_scalar_gap_problem, _spec(
            c="3.3", h_ext=".125", delta_gap=".12345678901234567890123456789",
            lam=1, spin_list=(0,), q_order=4, dps=70, npoles=2, chiral_norm=16,
        ))
        manifest_path, _ = write_identical_scalar_gap_problem(saved, probe_dir)
    (probe_dir / "sdpb_out").mkdir()
    (probe_dir / "sdpb_out/y.txt").write_text("0 1\n")   # a finished solve
    if tamper:
        manifest = json.loads(manifest_path.read_text())
        block_path = probe_dir / manifest["blocks"][0]["path"]
        block_path.write_text(block_path.read_text().replace('"3.3"', '"4.0"'))
    checked = []
    monkeypatch.setattr(certification, "certify_dual_from_sdpb_output", lambda problem, *args, **kw: (checked.append(problem) or True, []))
    with mp.workdps(15):
        if tamper:
            with pytest.raises((ValueError, SystemExit), match="hash|manifest"):
                recertify.certify_probe(tmp_path, 0, name)
            assert not checked
        else:
            ok, _, info = recertify.certify_probe(tmp_path, 0, name)
            assert ok and checked[0].spec == saved.spec
            assert info["probe_value"] == saved.spec.delta_gap
            assert info["dps"] == 70


def test_recertification_refuses_a_missing_probe_manifest(tmp_path, monkeypatch):
    import virasoro_bootstrap.hpc.recertify as recertify

    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "0000.json").write_text(json.dumps({
        "task": "scalar_gap_bisection", "params": {"c": "3.3", "lam": 7},
    }))
    (tmp_path / "tasks" / "0000" / "gap_1p0" / "sdpb_out").mkdir(parents=True)
    (tmp_path / "tasks" / "0000" / "gap_1p0" / "sdpb_out" / "y.txt").write_text("0 1\n")
    def unexpected(*args):
        pytest.fail("must not reconstruct a different problem from task parameters")
    monkeypatch.setitem(recertify.PROBLEM_BUILDERS, "scalar_gap_bisection", unexpected)
    with pytest.raises(SystemExit, match="manifest"):
        recertify.certify_probe(tmp_path, 0, "gap_1p0")


@pytest.mark.parametrize("module_name,function_name,sign", [
    ("identical_scalar", "ope_squared_bound_from_objective", -1),
    ("moments", "normalized_moment_lower_bound_from_objective", 1),
])
def test_objective_to_bound_preserves_loaded_precision(tmp_path, module_name, function_name, sign):
    from importlib import import_module
    from virasoro_bootstrap.identical_scalar import read_sdpb_objective

    token = "1.00000000000000000000000000000000000000000000000000000000000000000001"
    (tmp_path / "out.txt").write_text(f"dualObjective = {token};\n")
    with mp.workdps(15):
        objective = read_sdpb_objective(tmp_path)["dual"]
        bound = getattr(import_module(f"virasoro_bootstrap.{module_name}"), function_name)(objective)
    with mp.workdps(100):
        assert bound == sign * objective
        assert bound != sign


@pytest.mark.parametrize("field,key", [("dualObjective", "dual")])
def test_solver_objective_reader_matches_complete_field_names(tmp_path, field, key):
    from virasoro_bootstrap.identical_scalar import read_sdpb_objective

    (tmp_path / "out.txt").write_text(f"  {field} = -2;\n{field}Error = 0.01;\n")
    assert read_sdpb_objective(tmp_path) == {key: mp.mpf(-2)}
    (tmp_path / "out.txt").write_text(f"{field}Error = 0.01;\n")
    assert read_sdpb_objective(tmp_path) is None


def test_moment_cache_handles_mutable_spec_fields_without_stale_reuse(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from virasoro_bootstrap import moments

    monkeypatch.setattr(moments, "_BASE_PROBLEM_CACHE", {})
    built = []
    def build(spec):
        result = SimpleNamespace(spec=spec)
        built.append(result)
        return result
    monkeypatch.setattr(moments, "build_identical_scalar_gap_problem", build)
    overrides = {2: "2.5"}
    spec = _spec(delta_min_by_spin=overrides)
    first = moments._cached_base_problem(spec)
    assert moments._cached_base_problem(replace(spec, delta_min_by_spin={2: "2.5"})) is first
    overrides[2] = "3.0"
    second = moments._cached_base_problem(spec)
    assert second is not first
    assert first.spec.delta_min_by_spin == {2: "2.5"}
    assert moments._cached_base_problem(replace(spec, delta_min_by_spin={2: "2.5"})) is first
    assert len(built) == 2


@pytest.mark.parametrize("direction", ["upper"])
def test_moment_bisection_uses_spec_precision_for_close_brackets(monkeypatch, direction):
    from virasoro_bootstrap import moments as M

    monkeypatch.setattr(M, "_resolve_base_problem", lambda *args: None)
    monkeypatch.setattr(M, "_effective_ratio_constants", lambda *args, **kwargs: (0, 0, 0, 0))
    builder = "build_identical_scalar_normalized_ratio_" + ("lower_" if direction == "lower" else "") + "probe_problem"
    monkeypatch.setattr(M, builder, lambda *args, **kwargs: None)
    with mp.workdps(80):
        unit = mp.mpf("1e-50")
        lo, hi, threshold = (mp.nstr(1 + k * unit, 85) for k in (1, 9, 5))
    precisions = []
    def solve(problem, r):
        precisions.append(mp.mp.dps)
        return mp.mpf(1) if r < mp.mpf(threshold) else mp.mpf(-1)
    fn = M.bound_identical_scalar_normalized_moment_by_bisection if direction == "upper" else M.bound_identical_scalar_normalized_moment_lower_by_bisection
    with mp.workdps(15):
        result = fn(_spec(dps=80), M.MomentWeight.one(), M.MomentWeight.one(),
                    lo=lo, hi=hi, tol="1e-51", **{f"solve_probe_{direction}_bound": solve})
    with mp.workdps(80):
        left = mp.mpf(result["lower_uncertified" if direction == "upper" else "lower_certified"])
        right = mp.mpf(result["upper_certified" if direction == "upper" else "upper_uncertified"])
        assert left < right and right - left <= mp.mpf("1e-51")
        assert left <= mp.mpf(threshold) <= right
    assert set(precisions) == {80}


@pytest.mark.parametrize("direction", ["lower"])
def test_moment_bisection_builds_base_blocks_only_once(monkeypatch, direction):
    from virasoro_bootstrap import moments as M

    original = M.build_identical_scalar_gap_problem
    builds = []
    def build(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(M, "build_identical_scalar_gap_problem", build)
    fn = M.bound_identical_scalar_normalized_moment_by_bisection if direction == "upper" else M.bound_identical_scalar_normalized_moment_lower_by_bisection
    def solve(problem, r):
        return mp.mpf(1) if r < 2 else mp.mpf(-1)
    result = _quiet(fn, _spec(lam=1, q_order=4, npoles=2, spin_list=(0,)),
                    M.MomentWeight.monomial(Delta=1), M.MomentWeight.one(),
                    lo=1, hi=3, tol=".25", identity_mode="exclude",
                    **{f"solve_probe_{direction}_bound": solve})
    assert result["iterations"] == 3
    assert len(builds) == 1


@pytest.mark.parametrize("direction", ["upper"])
def test_moment_bisection_stops_before_repeating_an_unresolvable_probe(monkeypatch, direction):
    from virasoro_bootstrap import moments as M

    monkeypatch.setattr(M, "_resolve_base_problem", lambda *args: None)
    monkeypatch.setattr(M, "_effective_ratio_constants", lambda *args, **kwargs: (0, 0, 0, 0))
    builder = "build_identical_scalar_normalized_ratio_" + ("lower_" if direction == "lower" else "") + "probe_problem"
    monkeypatch.setattr(M, builder, lambda *args, **kwargs: None)
    with mp.workdps(15):
        hi = mp.nstr(1 + mp.eps, 30)
    probes = []
    def solve(problem, r):
        probes.append(r)
        return mp.mpf(1) if r == 1 else mp.mpf(-1)
    fn = M.bound_identical_scalar_normalized_moment_by_bisection if direction == "upper" else M.bound_identical_scalar_normalized_moment_lower_by_bisection
    with pytest.raises(RuntimeError, match="increase.*dps"):
        fn(_spec(dps=15), M.MomentWeight.one(), M.MomentWeight.one(),
           lo=1, hi=hi, tol="1e-100", max_iterations=3,
           **{f"solve_probe_{direction}_bound": solve})
    assert len(probes) == 2


@pytest.mark.parametrize("changed", [{"c": "4"}, {"lam": True}])
def test_moment_builder_rejects_a_conflicting_base_spec(changed):
    from dataclasses import replace
    from virasoro_bootstrap.moments import MomentWeight, build_identical_scalar_moment_upper_problem

    spec = _spec(lam=1, q_order=4, npoles=2, spin_list=(0,))
    base = _quiet(build_identical_scalar_gap_problem, spec)
    with pytest.raises(ValueError, match="base_problem|lam must be an integer"):
        _quiet(build_identical_scalar_moment_upper_problem,
               replace(spec, **changed), MomentWeight.one(), base_problem=base)


@pytest.mark.parametrize("direction, invalid_probe", [("upper", 0), ("lower", 2)])
def test_moment_bisection_rejects_nan_solver_results(monkeypatch, direction, invalid_probe):
    from virasoro_bootstrap import moments as M

    monkeypatch.setattr(M, "_resolve_base_problem", lambda *args: None)
    monkeypatch.setattr(M, "_effective_ratio_constants", lambda *args, **kwargs: (0, 0, 0, 0))
    builder = "build_identical_scalar_normalized_ratio_" + ("lower_" if direction == "lower" else "") + "probe_problem"
    monkeypatch.setattr(M, builder, lambda *args, **kwargs: None)
    probes = []
    def solve(problem, r):
        probes.append(r)
        if len(probes) - 1 == invalid_probe:
            return mp.nan
        return mp.mpf(1) if r < 2 else mp.mpf(-1)
    fn = M.bound_identical_scalar_normalized_moment_by_bisection if direction == "upper" else M.bound_identical_scalar_normalized_moment_lower_by_bisection
    with pytest.raises(ValueError, match="solver.*NaN"):
        fn(_spec(), M.MomentWeight.one(), M.MomentWeight.one(),
           lo=1, hi=3, tol=".25", **{f"solve_probe_{direction}_bound": solve})
    assert len(probes) == invalid_probe + 1


@pytest.mark.parametrize("damage", ["missing", "tampered", "wrong_weight", "none"])
def test_moment_recertification_checks_the_saved_problem(tmp_path, monkeypatch, damage):
    from virasoro_bootstrap.hpc import tasks
    from virasoro_bootstrap import moments as M, certification

    profile = {"dps": 40, "q_order": 4, "npoles": 2, "spin_list": [0],
               "large_spin_double_twists": 0, "pole_selection": "nearest"}
    monkeypatch.setattr(tasks, "_load_default_params_profile", lambda lam: profile)
    params = {"c": "3.3", "delta_ext": ".2", "lam": 1, "num_threads": 1,
              "source_output_dir": str(tmp_path / "source"), "moment_cert_q_order": 6,
              "identity_modes": ["include"], "directions": ["lower"]}
    with mp.workdps(40):
        spec = tasks._moment_nu1_over_nu0_spec(params, profile, lam=1, dps=40, num_threads=1)
        builder = M.build_identical_scalar_normalized_moment_upper_problem if damage == "wrong_kind" else M.build_identical_scalar_normalized_moment_lower_problem
        saved = _quiet(builder, spec, M.MomentWeight.monomial(Delta=2 if damage == "wrong_weight" else 1),
                       M.MomentWeight.one(), identity_mode="exclude" if damage == "wrong_identity" else "include",
                       alpha=1 if damage == "offset" else 0)
        directory = tmp_path / "source" / "include_lower"
        manifest, _ = write_identical_scalar_gap_problem(saved, directory)
    (directory / "sdpb_out").mkdir()
    if damage == "missing":
        manifest.unlink()
    elif damage == "tampered":
        ref = json.loads(manifest.read_text())["blocks"][0]["path"]
        block = directory / ref
        block.write_text(block.read_text().replace('"3.3"', '"4.0"'))
    elif damage == "wrong_spec":
        params["c"] = "4"
    checked = []
    def certify(problem, *args, **kwargs):
        checked.append(problem)
        return True, []
    monkeypatch.setattr(certification, "certify_dual_from_sdpb_output", certify)
    result = _quiet(tasks.moment_recertify, _ctx(tmp_path / "recert", params))
    entry = result["include"]["lower"]
    if damage == "none":
        assert len(checked) == 1, entry
        assert checked[0].spec.c == saved.spec.c
        assert checked[0].spec.dps == saved.spec.dps
        assert checked[0].normalization_vector == saved.normalization_vector
    else:
        assert "error" in entry
        assert checked == []


@pytest.mark.parametrize("field,original,equivalent,reload", [
    ("h_ext", ".125", "1/8", False),
    ("delta_min_by_spin", {2: "2.5"}, {2: "5/2"}, True),
    ("point_constraints", ((0, "1.5"),), ((0, "3/2"),), False),
])
def test_moment_base_accepts_equivalent_numeric_inputs(tmp_path, field, original, equivalent, reload):
    from dataclasses import replace
    from virasoro_bootstrap import moments as M

    extra = {"large_spin_lmin": 4, "large_spin_npoles": 2} if field == "large_spin_twists" else {}
    spec = _spec(lam=1, q_order=4, npoles=2, spin_list=(0, 2), **{field: original}, **extra)
    base = _quiet(build_identical_scalar_gap_problem, spec)
    if reload:
        from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest
        manifest, _ = write_identical_scalar_gap_problem(base, tmp_path)
        base = build_identical_scalar_problem_from_manifest(manifest)
    if field == "point_constraints":
        # Points remain unsupported for moments, but equivalent spelling must
        # reach that explicit limitation rather than a false spec mismatch.
        with pytest.raises(NotImplementedError, match="point-constraint"):
            _quiet(M.build_identical_scalar_moment_upper_problem,
                   replace(spec, **{field: equivalent}), M.MomentWeight.one(), base_problem=base)
    else:
        reused = _quiet(M.build_identical_scalar_moment_upper_problem,
                        replace(spec, **{field: equivalent}), M.MomentWeight.one(), base_problem=base)
        assert reused.F_pairs == base.F_pairs


def test_moment_base_does_not_round_away_exact_c_differences():
    from dataclasses import replace
    from virasoro_bootstrap import moments as M

    spec = _spec(c="3.3", lam=1, q_order=4, npoles=2, spin_list=(0,))
    base = _quiet(build_identical_scalar_gap_problem, spec)
    shifted = replace(spec, c="3.3" + "0" * 60 + "1")
    with mp.workdps(spec.dps):
        from virasoro_bootstrap.identical_scalar import _as_real_mpf
        assert _as_real_mpf(spec.c, name="c") == _as_real_mpf(shifted.c, name="c")
    with pytest.raises(ValueError, match="specification differs in: c"):
        _quiet(M.build_identical_scalar_moment_upper_problem,
               shifted, M.MomentWeight.one(), base_problem=base)


def test_twist_gap_helper_accepts_exact_numeric_labels():
    from fractions import Fraction
    from virasoro_bootstrap import twists_above_gap

    with mp.workdps(60):
        result = twists_above_gap([".25", Fraction(1, 2), "3/4", 1], "1/2")
        assert result == [mp.mpf(".5"), "3/4", 1]
        exact = Fraction(1, 3)
        assert twists_above_gap([exact], 0)[1] is exact


@pytest.mark.parametrize("value", [mp.nan])
def test_twist_gap_helper_rejects_nonfinite_slices(value):
    from virasoro_bootstrap import twists_above_gap

    with pytest.raises(ValueError, match="finite"):
        twists_above_gap([value, 2], 1)


@pytest.mark.parametrize("count", [True])
def test_double_twist_helper_rejects_invalid_counts(count):
    from virasoro_bootstrap import double_twist_set

    with pytest.raises(ValueError, match="n_count"):
        double_twist_set(".1", count)


def test_double_twist_helper_valid_counts():
    from virasoro_bootstrap import double_twist_set

    assert double_twist_set(".25", 0) == []
    assert double_twist_set(".25", 3) == [1, 3, 5]
