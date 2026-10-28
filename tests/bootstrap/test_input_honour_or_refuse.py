"""Regressions for the 2026-09-29 bootstrap input-handling audit ("honor or refuse").

Each test pins one input that used to be silently ignored, truncated, or rejected with an obscure error: Fraction
numbers, moment spin filters, the large-spin tower floor, manifest spec edits, the modular spec, and
``n_sample_points``.  All builds are small (Lambda = 3, a second or two).
"""
from __future__ import annotations

from fractions import Fraction
import json
import warnings

import mpmath as mp
import pytest

from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_fixed_ope_problem,
    build_identical_scalar_gap_problem,
    build_identical_scalar_ope_bound_problem,
    build_identical_scalar_ope_max_problem,
    prepare_identical_scalar_gap_build,
)
from virasoro_bootstrap.moments import (
    MomentTerm,
    MomentWeight,
    build_identical_scalar_moment_upper_problem,
    build_identical_scalar_normalized_moment_lower_problem,
)

DPS = 40


@pytest.mark.parametrize('value', [True, mp.mpc('.7','1e-70')])
def test_real_input_labels_reject_booleans_and_any_nonzero_imaginary_part(value):
    from virasoro_bootstrap.identical_scalar import _as_real_label

    with mp.workdps(30), pytest.raises(ValueError, match='value.*real'):
        _as_real_label(value, name='value')


def test_numeric_real_input_conversion_uses_the_same_contract():
    from virasoro_bootstrap.identical_scalar import _as_real_mpf

    with mp.workdps(30), pytest.raises(ValueError, match='value.*real'):
        _as_real_mpf(mp.mpc('.7','1e-70'), name='value')


def test_zero_imaginary_inputs_keep_their_stored_real_components():
    from virasoro_blocks.numeric_input import literal_components
    from virasoro_bootstrap.identical_scalar import _as_real_label, _as_real_mpf

    with mp.workdps(150):
        stored = mp.mpc(mp.mpf('.7')+mp.mpf('1e-70'),0)
    with mp.workdps(30):
        assert _as_real_label(stored,name='value') == literal_components(stored)[0]
        assert _as_real_label(complex(.7,0),name='value') == literal_components(complex(.7,0))[0]
        assert _as_real_mpf(0.7,name='value') == _as_real_mpf('.7',name='value') == mp.mpf('.7')


def test_numpy_integer_real_labels_use_unbounded_integer_arithmetic():
    import numpy as np
    from virasoro_bootstrap.identical_scalar import _as_real_label

    value = _as_real_label(np.int64(2**62), name='value')
    assert value*4 == 2**64


def _spec(**overrides):
    kwargs = dict(
        c=mp.mpf(8), h_ext=Fraction(1, 16), delta_gap=1, lam=3, spin_list=(0, 2), q_order=12, dps=DPS,
        npoles=4, num_threads=1,
    )
    kwargs.update(overrides)
    return IdenticalScalarGapSpec(**kwargs)


def test_chiral_normalization_is_integral_before_precision_rounding():
    from decimal import Decimal
    from virasoro_bootstrap.identical_scalar import _validate_gap_spec_common

    label = '16.' + '0' * 69 + '1'
    with mp.workdps(150):
        stored = mp.mpf(label)
    with mp.workdps(15):
        for norm in (Fraction(label), stored):
            with pytest.raises(ValueError, match='chiral_norm.*positive integer'):
                _validate_gap_spec_common(_spec(chiral_norm=norm))
        for norm in (16, Decimal(16)):
            _validate_gap_spec_common(_spec(chiral_norm=norm))


@pytest.mark.parametrize('field,value', [('h_ext',mp.mpc('.7','1e-70'))])
def test_problem_spec_real_fields_use_the_shared_input_contract(field,value):
    from virasoro_bootstrap.identical_scalar import _validate_gap_spec_common, _resolved_delta_min_by_ell

    spec = _spec(**{field:value})
    with pytest.raises(ValueError, match=field):
        _validate_gap_spec_common(spec)
        _resolved_delta_min_by_ell(spec,spec.spin_list)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _doc(problem):
    return json.dumps(problem.pmp_build.doc, sort_keys=True)


@pytest.fixture(scope="module")
def prepared():
    return _quiet(prepare_identical_scalar_gap_build, _spec())


# --- 1. Fraction inputs (docs/API.md: int, str, Fraction or mpf) ----------------------------------------------------

def test_fraction_tower_and_point_inputs_build_the_same_problem_as_strings():
    as_fraction = _spec(large_spin_lmin=4, large_spin_twists=(Fraction(1, 4), Fraction(9, 4)),
                        point_constraints=((0, Fraction(2, 5)),))
    as_string = _spec(large_spin_lmin="4", large_spin_twists=("0.25", "2.25"), point_constraints=((0, "0.4"),))
    a = _quiet(build_identical_scalar_gap_problem, as_fraction)
    b = _quiet(build_identical_scalar_gap_problem, as_string)
    assert _doc(a) == _doc(b)
    # The returned spec retains exact tower labels for later precision changes.
    assert a.spec.large_spin_lmin == 4
    assert a.spec.large_spin_twists == (Fraction(1, 4), Fraction(9, 4))
    assert all(isinstance(t, Fraction) for t in a.spec.large_spin_twists)
    assert a.spec.point_constraints == ((0, Fraction(2, 5)),)


def test_fraction_lmin_is_accepted():
    problem = _quiet(build_identical_scalar_gap_problem,
                     _spec(large_spin_lmin=Fraction(4), large_spin_twists=(Fraction(1, 4),)))
    assert [channel.ell for channel in problem.large_spin_channels] == [4]


def test_fraction_operator_inputs_match_strings(prepared):
    spec = _spec()
    pairs = [
        (build_identical_scalar_ope_max_problem, dict(delta_star=Fraction(3, 2)), dict(delta_star="1.5")),
        (build_identical_scalar_ope_bound_problem, dict(delta_star=Fraction(1, 3)), dict(delta_star="1/3")),
        (build_identical_scalar_fixed_ope_problem, dict(delta_star=Fraction(1, 3), ope_squared=Fraction(2, 7)),
         dict(delta_star="1/3", ope_squared="2/7")),
    ]
    for builder, fraction_kwargs, string_kwargs in pairs:
        a = _quiet(builder, spec, prepared=prepared, **fraction_kwargs)
        b = _quiet(builder, spec, prepared=prepared, **string_kwargs)
        assert a.normalization_vector == b.normalization_vector
        assert _doc(a) == _doc(b)


def test_fraction_moment_coefficients_scale_and_ratio_constants(prepared):
    spec = _spec()
    third = MomentWeight((MomentTerm(Fraction(1, 3), {"Delta": 1}),))
    assert third.to_json()["terms"][0]["coefficient"] == "1/3"
    a = _quiet(build_identical_scalar_moment_upper_problem, spec, third, prepared=prepared)
    b = _quiet(build_identical_scalar_moment_upper_problem, spec,
               MomentWeight((MomentTerm("1/3", {"Delta": 1}),)), prepared=prepared)
    assert _doc(a) == _doc(b)
    scaled = MomentWeight.monomial(Delta=1).scale(Fraction(1, 3))
    assert scaled.terms[0].coefficient == Fraction(1, 3)
    c = _quiet(build_identical_scalar_moment_upper_problem, spec, scaled, prepared=prepared)
    assert _doc(c) == _doc(b)
    kwargs = dict(prepared=prepared)
    d = _quiet(build_identical_scalar_normalized_moment_lower_problem, spec, MomentWeight.monomial(Delta=1),
               MomentWeight.one(), alpha=Fraction(1, 3), beta=Fraction(1, 7), **kwargs)
    e = _quiet(build_identical_scalar_normalized_moment_lower_problem, spec, MomentWeight.monomial(Delta=1),
               MomentWeight.one(), alpha="1/3", beta="1/7", **kwargs)
    assert _doc(d) == _doc(e)


def test_fraction_modular_delta_gap():
    from virasoro_bootstrap import ModularGapSpec, build_modular_gap_problem

    base = dict(c=8, lam=3, spin_list=(0, 1, 2), dps=30)
    a = build_modular_gap_problem(ModularGapSpec(delta_gap=Fraction(6, 5), delta_min_by_spin={1: Fraction(3, 2)},
                                                 **base))
    b = build_modular_gap_problem(ModularGapSpec(delta_gap="1.2", delta_min_by_spin={1: "1.5"}, **base))
    assert json.dumps(a.pmp_doc, sort_keys=True) == json.dumps(b.pmp_doc, sort_keys=True)


# --- 2. MomentWeight.spin_filter ------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [2.5, "2", True])
def test_spin_filter_must_be_an_integer(value):
    with pytest.raises(ValueError, match="spin_filter must be an integer"):
        MomentWeight((MomentTerm(1, {}),), spin_filter=value)


@pytest.mark.parametrize("value", [6, 3, -2])
def test_spin_filter_selecting_no_channel_is_refused(prepared, value):
    weight = MomentWeight((MomentTerm(1, {}),), spin_filter=value)
    with pytest.raises(ValueError, match=r"selects no channel of the problem, whose fixed spins are \[0, 2\]"):
        _quiet(build_identical_scalar_moment_upper_problem, _spec(), weight, prepared=prepared)
    with pytest.raises(ValueError, match="selects no channel"):
        _quiet(build_identical_scalar_normalized_moment_lower_problem, _spec(), MomentWeight.one(), weight,
               prepared=prepared)


def test_spin_filter_on_a_listed_spin_builds(prepared):
    weight = MomentWeight((MomentTerm(1, {}),), spin_filter=mp.mpf(2))
    assert weight.spin_filter == 2 and isinstance(weight.spin_filter, int)
    problem = _quiet(build_identical_scalar_moment_upper_problem, _spec(), weight, prepared=prepared)
    assert problem.problem_kind == "moment_upper"


# --- 3. large-spin tower floor --------------------------------------------------------------------------------------

@pytest.mark.parametrize("extra, match", [
    (dict(large_spin_lmin=30, large_spin_twists=("0.25",)), r"leaves the spins \[4, 6, 8, 10, 12, 14\] \.\.\. "
                                                             "unconstrained"),
    (dict(large_spin_lmin=6, large_spin_twists=("0.25",)), r"leaves the spins \[4\] unconstrained"),
    (dict(large_spin_lmin="1e6", large_spin_twists=("0.25",)), "unconstrained"),
    pytest.param(dict(large_spin_lmin=10**400, large_spin_twists=("0.25",)), "unconstrained",
                 id="huge-integer-tower-floor"),
    (dict(large_spin_lmin=-2, large_spin_twists=("0.25",)), "non-negative integer spin"),
    (dict(large_spin_lmin=3.5, large_spin_twists=("0.25",)), "non-negative integer spin"),
    (dict(large_spin_lmin=True, large_spin_twists=("0.25",)), "non-negative integer spin"),
    (dict(large_spin_lmin=4), "needs large_spin_twists"),
    # the tower's L = 0 member is the doubled 2 B(h, h), not the spin-0 G = B(h, h): refused however spelled
    *[pytest.param(dict(large_spin_lmin=zero, large_spin_twists=("0.25",)), "starts the tower at spin 0",
                   id=f"zero-floor-{zero!r}") for zero in (0, "0", "0.0", "0/1", Fraction(0), mp.mpf(0))],
    (dict(large_spin_twists=("0.25",)), "needs large_spin_lmin"),
])
def test_tower_floor_is_refused_before_any_build(extra, match):
    with pytest.raises(ValueError, match=match):
        prepare_identical_scalar_gap_build(_spec(**extra))


@pytest.mark.parametrize("zero", ["0", "0.0", "0/1"])
def test_moment_channels_refuse_a_zero_tower_floor_by_value(zero):
    """The moment guard compares the floor's value, not its spelling (``"0" == 0`` is false)."""
    from dataclasses import replace
    from types import SimpleNamespace

    from virasoro_bootstrap.moments import _moment_channels

    spec = replace(_spec(), large_spin_lmin=zero, large_spin_twists=("0.25",))
    problem = SimpleNamespace(spec=spec, point_channels=(), spin_channels=(object(), object()),
                              large_spin_channels=(object(),))
    with pytest.raises(ValueError, match="large_spin_lmin >= 1"):
        _moment_channels(problem, include_tower=True)


def test_tower_floor_up_to_the_first_uncovered_spin_builds():
    problem = _quiet(build_identical_scalar_gap_problem, _spec(large_spin_lmin=4, large_spin_twists=("0.25",)))
    assert len(problem.large_spin_channels) == 1


# --- 4. manifest spec edits -----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest_dir(tmp_path_factory):
    from virasoro_bootstrap.identical_scalar import write_identical_scalar_gap_problem

    problem = _quiet(build_identical_scalar_gap_problem, _spec(point_constraints=((0, "0.4"),)))
    out = tmp_path_factory.mktemp("manifest") / "base"
    write_identical_scalar_gap_problem(problem, out)
    return out, problem


def _edited(manifest_dir, tmp_path, **edit):
    import shutil

    source, _problem = manifest_dir
    target = tmp_path / "edited"
    shutil.copytree(source, target)
    path = target / "manifest.json"
    doc = json.loads(path.read_text())
    doc["spec"].update(edit)
    path.write_text(json.dumps(doc))
    return path


def test_unedited_manifest_reloads(manifest_dir, tmp_path):
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    reloaded = _quiet(build_identical_scalar_problem_from_manifest, _edited(manifest_dir, tmp_path))
    assert len(reloaded.point_channels) == 1
    blocks = "PositiveMatrixWithPrefactorArray"
    assert len(reloaded.pmp_build.doc[blocks]) == len(manifest_dir[1].pmp_build.doc[blocks]) == 3


@pytest.mark.parametrize("edit, match", [
    ({"point_constraints": []}, "spec.point_constraints was edited"),
    ({"point_constraints": [[0, "0.5"]]}, "spec.point_constraints was edited"),
    ({"shared_sample_points": True}, "spec.shared_sample_points was edited"),
    ({"sample_via": "symbolic"}, "spec.sample_via was edited"),
    # the same block-file refusal as an edited c, not a prefactor_exponent error
    ({"h_ext": "0.1"}, r"does not match the spec: .*delta_external"),
])
def test_manifest_edits_the_blocks_cannot_honour_are_refused(manifest_dir, tmp_path, edit, match):
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    with pytest.raises(ValueError, match=match):
        _quiet(build_identical_scalar_problem_from_manifest, _edited(manifest_dir, tmp_path, **edit))


def test_manifest_prefactor_edit_is_still_refused(manifest_dir, tmp_path):
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    with pytest.raises(ValueError, match="prefactor_exponent"):
        _quiet(build_identical_scalar_problem_from_manifest,
               _edited(manifest_dir, tmp_path, prefactor_exponent="0.1"))


def test_manifest_keeps_appended_point_channels(tmp_path):
    from virasoro_bootstrap.identical_scalar import write_identical_scalar_gap_problem
    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest
    from virasoro_bootstrap.point_constraint import append_positivity_channels, build_point_channel

    problem = _quiet(build_identical_scalar_gap_problem, _spec())
    with mp.workdps(DPS):
        h = mp.mpf(1) / 16
        point = build_point_channel(0, Fraction(1, 2), h_ext=h, c=mp.mpf(8), q_order=12, lam=3,
                                    z_shift=-2 * h, zbar_shift=-2 * h)
    extended = append_positivity_channels(problem, [point])
    write_identical_scalar_gap_problem(extended, tmp_path / "x")
    reloaded = _quiet(build_identical_scalar_problem_from_manifest, tmp_path / "x" / "manifest.json")
    assert len(reloaded.point_channels) == 1


# --- 5. ModularGapSpec ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("extra, match", [
    (dict(delta_min_by_spin={5: 6}), r"spin 5, which is not in spin_list \[0, 1, 2\]"),
    (dict(delta_min_by_spin={"1": 2}), "keys must be integer spins"),
    (dict(delta_gap="-0.1"), "delta_gap must be >= 0"),
])
def test_modular_spec_refuses_inputs_it_would_ignore(extra, match):
    from virasoro_bootstrap import ModularGapSpec, build_modular_gap_problem

    kwargs = dict(c=8, delta_gap=1, lam=3, spin_list=(0, 1, 2), dps=30)
    kwargs.update(extra)
    with pytest.raises(ValueError, match=match):
        build_modular_gap_problem(ModularGapSpec(**kwargs))


# --- 6. n_sample_points ---------------------------------------------------------------------------------------------

def _nodes(problem, index=0):
    return len(problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][index]["samplePoints"])


def test_n_sample_points_below_the_per_entry_minimum_is_refused(prepared):
    natural = _nodes(_quiet(build_identical_scalar_gap_problem, _spec(), prepared=prepared))
    with pytest.raises(ValueError, match=f"n_sample_points = {natural - 1} is below the {natural} nodes"):
        _quiet(build_identical_scalar_gap_problem, _spec(n_sample_points=natural - 1))
    # a larger count takes effect
    assert _nodes(_quiet(build_identical_scalar_gap_problem, _spec(n_sample_points=natural + 2))) == natural + 2


def test_n_sample_points_below_the_minimum_takes_effect_with_shared_nodes():
    problem = _quiet(build_identical_scalar_gap_problem, _spec(n_sample_points=4, shared_sample_points=True))
    assert _nodes(problem) == 4
    assert all(len(poly) <= 4 for poly in problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][0]
               ["polynomials"][0][0])
