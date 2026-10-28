from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import mpmath as mp
import pytest

from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.serialization import SerializedDerivTable
from virasoro_bootstrap.moments import (
    MomentTerm,
    MomentWeight,
    _constant_scalar_block,
    bound_identical_scalar_normalized_moment_by_bisection,
    bound_identical_scalar_normalized_moment_lower_by_bisection,
    build_identical_scalar_moment_lower_problem,
    build_identical_scalar_moment_upper_problem,
    build_identical_scalar_normalized_moment_lower_problem,
    build_identical_scalar_normalized_ratio_probe_problem,
    build_identical_scalar_normalized_ratio_lower_probe_problem,
    _moment_channels,
    moment_lower_bound_from_objective,
    moment_upper_bound_from_objective,
    moment_weight_polynomial,
    moment_weight_polynomial_in_delta,
    normalized_moment_lower_bound_from_objective,
    normalized_moment_upper_bound_from_objective,
    normalized_ratio_probe_threshold,
)
from virasoro_blocks.polyderivs import DeltaDerivTable, _realify_conjugate_denominator
from virasoro_bootstrap.certification import (
    _certify_normalized_moment_adaptive_pmp_blocks,
    _certify_normalized_moment_extra_pmp_blocks,
    _pmp_block_min,
    certify_dual_from_sdpb_output,
)
from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    prepare_identical_scalar_gap_build,
    read_sdpb_objective,
    run_identical_scalar_gap_problem_in_docker,
    self_dual_identity_measure_value,
    write_identical_scalar_gap_problem,
)
from virasoro_bootstrap.controls.global_control import (
    build_global_identical_scalar_control_problem,
)
from virasoro_bootstrap.pmp_writer import build_pmp_doc
from virasoro_bootstrap.sdp_writer import write_sdp_directory


def _poly(value):
    return Polynomial([mp.mpf(value)], field=mp.mpf)


def _block():
    return SerializedDerivTable(
        shift=mp.mpf(0),
        base=mp.mpf("0.5"),
        pole_list_x=(),
        coordinates=("zzb",),
        numerators={"zzb": [[_poly(1)]]},
    )


def test_moment_weight_polynomial_fixed_spin_variables():
    channel = SimpleNamespace(
        ell=2,
        prefactored=SimpleNamespace(shift=mp.mpf(3)),
    )
    weight = MomentWeight(
        (
            MomentTerm(1, {"Delta": 1, "L": 2}),
            MomentTerm(2, {"h": 1}),
        )
    )

    poly = moment_weight_polynomial(weight, channel)

    # At x = 4: Delta = 7, L = 2, h = 9/2.
    assert poly.evaluate(mp.mpf(4)) == mp.mpf(37)


def test_moment_weight_polynomial_in_delta_matches_x_shifted():
    # W in x = Delta - shift evaluated at x, vs W in Delta evaluated at Delta,
    # must agree at every Delta for a fixed-spin channel.
    channel = SimpleNamespace(
        ell=2,
        twist=None,
        prefactored=SimpleNamespace(shift=mp.mpf(3)),
    )
    weight = MomentWeight(
        (MomentTerm(1, {"Delta": 2}), MomentTerm(3, {"h": 1}))
    )
    w_x = moment_weight_polynomial(weight, channel)
    w_delta = moment_weight_polynomial_in_delta(weight, channel)
    shift = mp.mpf(3)
    for x in (mp.mpf("0.5"), mp.mpf(2), mp.mpf("7.25")):
        assert w_x.evaluate(x) == w_delta.evaluate(x + shift)


def test_moment_weight_linear_combinations():
    channel = SimpleNamespace(
        ell=4,
        prefactored=SimpleNamespace(shift=mp.mpf(1)),
    )
    weight = MomentWeight.monomial(Delta=1).minus(
        MomentWeight.one().scale(mp.mpf("2.5"))
    )

    poly = moment_weight_polynomial(weight, channel)

    # At x = 3: Delta = 4, so Delta - 2.5 = 1.5.
    assert poly.evaluate(mp.mpf(3)) == mp.mpf("1.5")


def test_build_pmp_doc_appends_extra_polynomial_components():
    extra = Polynomial([mp.mpf(2), mp.mpf(3)], field=mp.mpf)

    doc = build_pmp_doc(
        [_block()],
        F_pairs=[(0, 0)],
        objective=[mp.mpf(0), mp.mpf(1)],
        normalization=[mp.mpf(1), mp.mpf(0)],
        extra_polynomials_per_block=[(extra,)],
    )

    poly_vector = doc["PositiveMatrixWithPrefactorArray"][0]["polynomials"][0][0]
    assert len(poly_vector) == 2
    assert poly_vector[0] == ["1.0"]
    assert poly_vector[1] == ["2.0", "3.0"]


def test_build_pmp_doc_extra_polynomial_length_mismatch_raises():
    with pytest.raises(ValueError, match="objective length"):
        build_pmp_doc(
            [_block()],
            F_pairs=[(0, 0)],
            objective=[mp.mpf(0)],
            normalization=[mp.mpf(1)],
            extra_polynomials_per_block=[(_poly(2),)],
        )


def test_constant_scalar_block_shape_for_normalized_dual():
    block = _constant_scalar_block(
        [mp.mpf(-2), mp.mpf("0.25"), mp.mpf("-0.5")],
        sig_figs=20,
    )

    assert block["samplePoints"] == ["0"]
    assert block["sampleScalings"] == ["1"]
    assert block["DampedRational"]["poles"] == []
    assert block["polynomials"][0][0] == [["-2.0"], ["0.25"], ["-0.5"]]


def test_pmp_block_min_uses_sample_midpoints():
    block = {
        "samplePoints": ["0", "2"],
        "polynomials": [[[
            ["0.75", "-2", "1"],  # (x - 0.5) * (x - 1.5)
        ]]],
    }

    min_val, min_x, n_negative = _pmp_block_min([mp.mpf(1)], block)

    assert min_val == mp.mpf("-0.25")
    assert min_x == mp.mpf(1)
    assert n_negative == 1


def test_normalized_objective_conversion_helpers():
    # SDPB maximizes objective.dual. Lower duals report the bound directly;
    # upper duals report its negation.
    assert moment_lower_bound_from_objective(mp.mpf("1.25")) == mp.mpf("1.25")
    assert moment_upper_bound_from_objective(mp.mpf("1.25")) == mp.mpf("-1.25")
    assert normalized_moment_lower_bound_from_objective(mp.mpf("-1.25")) == mp.mpf("-1.25")
    assert normalized_moment_upper_bound_from_objective(mp.mpf("1.25")) == mp.mpf("-1.25")
    assert normalized_ratio_probe_threshold(
        mp.mpf("3"), alpha=mp.mpf("0.5"), beta=mp.mpf("0.25")
    ) == mp.mpf("1.25")


def test_real_denominator_realification_snaps_near_real_roots():
    with mp.workdps(120):
        table = DeltaDerivTable(
            exp_base=mp.mpf("0.5"),
            denominator=Polynomial(
                [mp.mpf("0.25"), mp.mpf("-1.0"), mp.mpf("1.0")]
            ),
            denominator_roots=[
                mp.mpc("0.5", "1e-80"),
                mp.mpc("0.5", "1e-80"),
            ],
            numerators=[[Polynomial([mp.mpf("1")])]],
        )

        fixed = _realify_conjugate_denominator(table)

    assert all(mp.im(root) == 0 for root in fixed.denominator_roots)


def test_moment_node_values_are_the_numeric_measure_times_the_weight():
    # each moment component's SDP data: the self-dual measure's numeric value at the block's own nodes times W there
    spec = _small_virasoro_spec()
    prepared = prepare_identical_scalar_gap_build(spec)
    W = MomentWeight.monomial(Delta=2)
    prob = build_identical_scalar_normalized_moment_lower_problem(
        spec, W, MomentWeight.one(), identity_mode="exclude", prepared=prepared
    )
    channel = prob.spin_channels[0]
    block = prob.pmp_build.doc["PositiveMatrixWithPrefactorArray"][0]
    scalings = [mp.mpf(v) for v in block["sampleScalings"]]
    measure = channel.prefactored.sample_block_values[spec.coordinate]
    n_f = len(prob.F_pairs)
    for k, x in enumerate(channel.prefactored.sample_points_x):
        delta = mp.mpf(channel.delta_min) + mp.mpf(x)
        expected = mp.re(measure[k][0][0]) * delta ** 2 / scalings[k]
        assert abs(prob.sample_values_by_block[0][k][n_f] - expected) <= mp.mpf(10) ** (10 - mp.mp.dps) * abs(expected)


def _small_virasoro_spec():
    return IdenticalScalarGapSpec(
        c=mp.mpf("2"),
        h_ext=mp.mpf("0.2"),
        delta_gap=mp.mpf("0.1"),
        lam=3,
        spin_list=(0,),
        q_order=8,
        dps=60,
        npoles=4,
        num_threads=1,
    )


def _small_virasoro_tower_spec():
    from virasoro_bootstrap.large_spin import double_twist_set

    h_ext = mp.mpf("0.2")
    spins = (0, 2)
    return IdenticalScalarGapSpec(
        c=mp.mpf("2"),
        h_ext=h_ext,
        delta_gap=mp.mpf("0.1"),
        lam=3,
        spin_list=spins,
        q_order=8,
        dps=60,
        npoles=4,
        pole_selection="contribution",
        large_spin_lmin=max(spins) + 2,
        large_spin_twists=tuple(double_twist_set(h_ext, 2)),
        large_spin_pole_selection="contribution",
        num_threads=1,
    )


def test_moment_tower_channels_use_delta_equals_twist_plus_spin():
    spec = _small_virasoro_tower_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
        include_tower=True,
    )

    n_fixed = len(spec.spin_list)
    channels = _moment_channels(problem, include_tower=True)
    # Fixed spins plus one slice per double-twist twist.
    assert len(channels) == n_fixed + len(spec.large_spin_twists)

    tower = channels[n_fixed]
    assert tower.twist is not None
    shift = mp.mpf(tower.prefactored.shift)
    tau = mp.mpf(tower.twist)
    # In the tower the positivity variable is the spin L (x = L - Lmax), so the
    # moment weight Delta must evaluate to tau + L = tau + (x + Lmax).
    wDelta = moment_weight_polynomial(MomentWeight.monomial(Delta=1), tower)
    for x in (mp.mpf(0), mp.mpf("2.5")):
        assert mp.almosteq(wDelta.evaluate(x), tau + shift + x)

    # The tower contributes positivity blocks beyond the fixed spins (plus the
    # scalar normalization constraint block).
    n_blocks = len(problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"])
    assert n_blocks >= n_fixed + len(spec.large_spin_twists)


def test_normalized_moment_extra_spin_certification_rebuilds_new_blocks():
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
    )
    y = [mp.mpf(0)] * len(problem.objective_vector)

    all_positive, details = _certify_normalized_moment_extra_pmp_blocks(
        y,
        problem,
        extra_spins=1,
        extra_twists=0,
        n_test_points=5,
        delta_range=4,
    )

    assert all_positive
    assert len(details) == 1
    assert details[0]["extra"] is True
    assert details[0]["basis"] == "pmp_polynomial_extra"
    assert details[0]["ell"] == max(spec.spin_list) + 2
    assert details[0]["twist"] is None


def test_normalized_moment_output_certification_checks_extra_spins(tmp_path):
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
    )
    # a normalized functional (n . y = 1): an all-zero y is rejected by the normalization check
    with mp.workdps(spec.dps):
        n = [mp.mpf(v) for v in problem.normalization_vector]
        y = [v / mp.fsum(u * u for u in n) for v in n]
    y_path = tmp_path / "y.txt"
    y_path.write_text(f"{len(y)} 1\n" + "\n".join(mp.nstr(v, 40) for v in y) + "\n")

    _all_positive, details = certify_dual_from_sdpb_output(
        problem,
        tmp_path,
        extra_spins=1,
        warn=False,
    )

    extra = [d for d in details if d.get("extra")]
    assert len(extra) == 1
    assert extra[0]["ell"] == max(spec.spin_list) + 2


def test_normalized_moment_direct_sdp_certification_uses_physical_extra_blocks(tmp_path):
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
    )
    write_sdp_directory(
        problem.pmp_build.doc,
        tmp_path / "sdp",
        eval_dps=80,
        output_digits=70,
        sample_values_by_block=problem.sample_values_by_block,
    )
    sdpb_out = tmp_path / "sdpb_out"
    sdpb_out.mkdir()
    y_len = len(problem.objective_vector) - 1
    (sdpb_out / "y.txt").write_text(
        f"{y_len} 1\n" + "\n".join(["0"] * y_len) + "\n"
    )

    all_positive, details = certify_dual_from_sdpb_output(
        problem,
        sdpb_out,
        extra_spins=1,
        physical_moment_blocks=True,
        n_test_points=5,
        delta_range=2,
        warn=False,
    )

    assert all_positive
    extra = [d for d in details if d.get("extra")]
    assert len(extra) == 1
    assert extra[0]["basis"] == "physical_numeric_blocks"
    assert extra[0]["ell"] == max(spec.spin_list) + 2


def test_normalized_moment_direct_sdp_exact_positivity_actually_decides(tmp_path):
    """The exact root-isolation check must REACH A VERDICT on included blocks.

    Regression for an off-by-one that made it silently never run: SDPB's y.txt
    has N-1 components (one is eliminated by the normalization constraint) while
    the PMP ``polynomials`` array has N, so passing the raw y_sdpb to
    ``block_positivity`` tripped its length guard on every block. It returned
    ``passed=None, reason="dimension_mismatch"``, which the caller recorded as
    ``exact_positivity_undecided`` without raising -- leaving positivity resting
    entirely on the per-sample scan, whose nodes are the very points the block
    polynomial was fitted through. Every direct-SDP moment run between the
    introduction of this path and 2026-08-03 reports ``passed: null`` here.

    Asserting ``passed is not None`` (rather than ``is True``) is deliberate:
    the point is that a verdict is reached, and a genuine False must stay
    reportable.
    """
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
    )
    write_sdp_directory(
        problem.pmp_build.doc,
        tmp_path / "sdp",
        eval_dps=80,
        output_digits=70,
        sample_values_by_block=problem.sample_values_by_block,
    )
    sdpb_out = tmp_path / "sdpb_out"
    sdpb_out.mkdir()
    y_len = len(problem.objective_vector) - 1
    (sdpb_out / "y.txt").write_text(
        f"{y_len} 1\n" + "\n".join(["0"] * y_len) + "\n"
    )

    _, details = certify_dual_from_sdpb_output(
        problem,
        sdpb_out,
        n_test_points=5,
        delta_range=2,
        warn=False,
    )

    included = [
        d for d in details
        if d.get("basis") == "direct_sdp_samples" and "exact_positivity" in d
    ]
    assert included, "no included-block details carried an exact_positivity record"
    for detail in included:
        exact = detail["exact_positivity"]
        assert exact.get("reason") != "dimension_mismatch", (
            f"exact check bailed on the length guard: {exact}"
        )
        assert exact.get("passed") is not None, (
            f"exact check reached no verdict: {exact}"
        )

    # block_positivity reports in mpmath precision. The HPC worker json.dumps
    # task results straight to disk, so an mpf reaching this dict kills the task
    # with "Object of type mpf is not JSON serializable" -- which is exactly what
    # happened to cluster tasks once the exact check first produced values.
    # Serializability is part of the contract, not an afterthought.
    json.dumps(details)


def test_force_exact_included_checks_true_numeric_blocks(tmp_path):
    """The force-exact switch must not fall back to PMP interpolants.

    This is the regression for the c->1 audit: before this test was added,
    ``force_exact_included=True`` only selected the non-direct branch, which
    still evaluated the serialized interpolation polynomials.  The returned
    included records must now come from the true-block finite-grid path.
    """
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
    )
    write_sdp_directory(
        problem.pmp_build.doc,
        tmp_path / "sdp",
        eval_dps=80,
        output_digits=70,
        sample_values_by_block=problem.sample_values_by_block,
    )
    sdpb_out = tmp_path / "sdpb_out"
    sdpb_out.mkdir()
    y_len = len(problem.objective_vector) - 1
    (sdpb_out / "y.txt").write_text(
        f"{y_len} 1\n" + "\n".join(["0"] * y_len) + "\n"
    )

    _, details = certify_dual_from_sdpb_output(
        problem,
        sdpb_out,
        force_exact_included=True,
        n_test_points=5,
        delta_range=2,
        warn=False,
    )

    included = [
        d for d in details
        if not d.get("extra") and d.get("basis") != "exact_scalar_block"
    ]
    assert included
    assert all(d.get("basis") == "physical_numeric_blocks" for d in included)
    assert all(d.get("certificate_scope") == "finite_true_block_grid" for d in included)
    scalar = [d for d in details if d.get("basis") == "exact_scalar_block"]
    assert len(scalar) == 1
    assert scalar[0]["passed"] is True


def test_normalized_moment_adaptive_certification_checks_wide_stage_once():
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
    )
    y = [mp.mpf(0)] * len(problem.objective_vector)

    all_positive, details = _certify_normalized_moment_adaptive_pmp_blocks(
        y,
        problem,
        extra_spins=1,
        extra_twists=0,
        max_extra_spins=2,
        max_extra_twists=0,
        n_test_points=5,
        delta_range=4,
    )

    assert all_positive
    assert [(d["ell"], d["adaptive_stage"]) for d in details] == [
        (max(spec.spin_list) + 2, "initial"),
        (max(spec.spin_list) + 4, "spin_wide"),
    ]


def test_moment_include_tower_false_drops_tower_channels():
    spec = _small_virasoro_tower_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
        include_tower=False,
    )
    channels = _moment_channels(problem, include_tower=False)
    assert len(channels) == len(spec.spin_list)
    assert all(ch.twist is None for ch in channels)


def test_unnormalized_identity_mode_include_adds_vacuum_objective_shift():
    spec = _small_virasoro_spec()
    include = build_identical_scalar_moment_upper_problem(
        spec, MomentWeight.one(), identity_mode="include"
    )
    exclude = build_identical_scalar_moment_upper_problem(
        spec, MomentWeight.one(), identity_mode="exclude"
    )
    vacuum = self_dual_identity_measure_value(
        spec.c,
        spec.h_ext,
        spec.lam,
        spec.q_order,
        coordinate=spec.coordinate,
        chiral_norm=spec.chiral_norm,
    )

    # Upper dual (SDPB maximizes): objective = [+b, -I[W]]; bound = -dualObjective.
    assert mp.almosteq(include.objective_vector[-1], -vacuum)
    assert exclude.objective_vector[-1] == mp.mpf(0)
    assert all(
        mp.almosteq(obj, identity)
        for obj, identity in zip(include.objective_vector[:-1], include.identity_vector)
    )
    assert mp.almosteq(
        mp.mpf(include.pmp_metadata["moment"]["identity_shift"]), vacuum
    )


def test_unnormalized_lower_direct_dual_objective_signs():
    spec = _small_virasoro_spec()
    lower_nu0 = build_identical_scalar_moment_lower_problem(
        spec, MomentWeight.one(), identity_mode="include"
    )
    lower_nu1 = build_identical_scalar_moment_lower_problem(
        spec, MomentWeight.monomial(Delta=1), identity_mode="include"
    )

    # Lower dual (SDPB maximizes): objective = [+b, +I[W]]; bound = +dualObjective.
    assert all(
        mp.almosteq(obj, identity)
        for obj, identity in zip(
            lower_nu0.objective_vector[:-1], lower_nu0.identity_vector
        )
    )
    assert all(
        mp.almosteq(obj, identity)
        for obj, identity in zip(
            lower_nu1.objective_vector[:-1], lower_nu1.identity_vector
        )
    )
    # The lower dual carries +I[W] in the s-component.
    assert mp.almosteq(
        lower_nu0.objective_vector[-1],
        self_dual_identity_measure_value(
            spec.c,
            spec.h_ext,
            spec.lam,
            spec.q_order,
            coordinate=spec.coordinate,
            chiral_norm=spec.chiral_norm,
        ),
    )
    assert lower_nu1.objective_vector[-1] == mp.mpf(0)


def test_normalized_identity_mode_include_shifts_effective_denominator():
    spec = _small_virasoro_spec()
    include = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="include",
    )
    exclude = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
    )
    vacuum = self_dual_identity_measure_value(
        spec.c,
        spec.h_ext,
        spec.lam,
        spec.q_order,
        coordinate=spec.coordinate,
        chiral_norm=spec.chiral_norm,
    )

    include_meta = include.pmp_metadata["moment"]
    exclude_meta = exclude.pmp_metadata["moment"]
    assert mp.almosteq(mp.mpf(include_meta["identity_denominator"]), vacuum)
    assert mp.almosteq(mp.mpf(include_meta["alpha_effective"]), vacuum)
    assert mp.mpf(include_meta["identity_numerator"]) == mp.mpf(0)
    assert mp.mpf(include_meta["beta_effective"]) == mp.mpf(0)
    assert mp.mpf(exclude_meta["alpha_effective"]) == mp.mpf(0)
    assert mp.mpf(exclude_meta["identity_denominator"]) == mp.mpf(0)


def test_normalized_scalar_block_uses_plus_identity():
    """The scalar inequality is  beta + alpha*y + b.z >= 0  (eq. in the paper's
    notation has -b.z, but the paper's vacuum b = -identity_vector here). The
    scalar block's z-components must therefore be +identity_vector, not -.
    """
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
    )
    scalar_block = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][-1]
    # polynomials[0][0] is the per-dual-variable vector; the first n entries are
    # the z-components (one per F-pair), followed by the s and y slots.
    z_components = scalar_block["polynomials"][0][0][: len(problem.F_pairs)]
    for entry, identity in zip(z_components, problem.identity_vector):
        assert mp.almosteq(mp.mpf(entry[0]), mp.mpf(identity)), (
            f"scalar z-component {entry[0]} != +identity {mp.nstr(identity, 8)}"
        )


def test_normalized_scalar_block_survives_direct_sdp_writer(tmp_path):
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        alpha=mp.mpf(1),
        beta=mp.mpf(0),
        identity_mode="exclude",
    )

    write_sdp_directory(
        problem.pmp_build.doc, tmp_path / "sdp", eval_dps=100, output_digits=60
    )
    block_info = json.loads((tmp_path / "sdp" / "block_info_1.json").read_text())
    block_data = json.loads((tmp_path / "sdp" / "block_data_1.json").read_text())

    assert block_info == {"dim": 1, "num_points": 1}
    assert mp.almosteq(mp.mpf(block_data["bilinear_bases_even"][0][0]), mp.mpf(1))
    assert block_data["bilinear_bases_odd"] == []
    assert mp.almosteq(mp.mpf(block_data["c"][0]), mp.mpf(0))
    # The global normalization pivot is the fixed s-slot.  After eliminating it,
    # the scalar inequality row still contains z-components and the y coefficient.
    assert len(block_data["B"]) == 1
    assert len(block_data["B"][0]) == len(problem.objective_vector) - 1
    assert mp.almosteq(mp.mpf(block_data["B"][0][-1]), mp.mpf("-1"))
    assert len(block_data["B"][0][-1].partition(".")[2]) >= 59


def test_global_normalized_direct_can_use_nonratio_blocks():
    spec = _small_virasoro_spec()
    base = build_global_identical_scalar_control_problem(
        spec, global_identity=True
    )
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
        base_problem=base,
        block_basis="nonratio",
    )

    moment_meta = problem.pmp_metadata["moment"]
    assert moment_meta["block_basis"] == "F_plus_self_dual_measure_moments"
    assert moment_meta["sample_value_basis"] == "direct_nonratio_at_damped_nodes"
    assert problem.sample_values_by_block is not None

    first_block = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][0]
    denominator_component = [
        mp.mpf(c) for c in first_block["polynomials"][0][0][-1]
    ]
    first_x = mp.mpf(first_block["samplePoints"][0])
    first_sample_row = problem.sample_values_by_block[0][0]

    # In the old damped-rational basis the denominator insertion is represented
    # as the reduced polynomial for G_self * V0, not the direct ratio-basis
    # constant V0=1.  The clean direct values fed to the sdp writer must match
    # this reduced polynomial at the explicit interpolation nodes.
    value = mp.mpf(0)
    for coeff in reversed(denominator_component):
        value = value * first_x + coeff
    assert not mp.almosteq(first_sample_row[-1], mp.mpf(1))
    assert abs(value - first_sample_row[-1]) < mp.mpf("1e-12")


def _ratio_base_problem(family, spec):
    """``base_problem`` for the ratio-basis tests: the global control problem, or
    ``None`` (the ordinary Virasoro build)."""
    if family == "global":
        return build_global_identical_scalar_control_problem(spec, global_identity=True)
    return None


@pytest.mark.parametrize("family", ["global", "virasoro"])
def test_normalized_direct_uses_ratio_blocks(family):
    # (formerly test_global_/test_virasoro_normalized_direct_uses_ratio_blocks)
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        identity_mode="exclude",
        base_problem=_ratio_base_problem(family, spec),
        block_basis="ratio",
    )

    moment_meta = problem.pmp_metadata["moment"]
    assert moment_meta["block_basis"] == "F_over_self_dual_measure"
    assert moment_meta["sample_value_basis"] == "direct_ratio_at_sdp_nodes"
    assert problem.sample_values_by_block is not None

    first_block = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][0]
    components = first_block["polynomials"][0][0]
    denominator_component = components[-1]
    first_sample_row = problem.sample_values_by_block[0][0]

    # In the paper-normalized basis the denominator insertion is V0=1, not an
    # interpolated self-dual block polynomial whose tail can change sign.
    # Ordinary Virasoro moment bounds use the same sampled F/G_self basis as
    # global controls.
    assert len(denominator_component) == len(first_block["samplePoints"])
    assert all(mp.mpf(value) == 0 for value in denominator_component[1:])
    assert mp.almosteq(mp.mpf(denominator_component[0]), mp.mpf(1))
    assert mp.almosteq(mp.mpf(first_sample_row[-1]), mp.mpf(1))


@pytest.mark.parametrize("family", ["global", "virasoro"])
def test_unnormalized_direct_uses_ratio_blocks(family):
    # (formerly test_global_/test_virasoro_unnormalized_direct_uses_ratio_blocks)
    mp.mp.dps = 30  # explicit: these compare with a bare mp.almosteq
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        identity_mode="exclude",
        base_problem=_ratio_base_problem(family, spec),
        block_basis="ratio",
    )

    moment_meta = problem.pmp_metadata["moment"]
    assert moment_meta["block_basis"] == "F_over_self_dual_measure"
    assert moment_meta["sample_value_basis"] == "direct_ratio_at_sdp_nodes"
    assert problem.sample_values_by_block is not None

    first_block = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][0]
    moment_component = first_block["polynomials"][0][0][-1]
    first_sample_row = problem.sample_values_by_block[0][0]
    first_x = mp.mpf(first_block["samplePoints"][0])

    # The unnormalized moment component is W=Delta=x+shift in the
    # paper-normalized F/G basis, not G_self*W.
    assert len(moment_component) == len(first_block["samplePoints"])
    assert all(mp.mpf(value) == 0 for value in moment_component[2:])
    assert mp.almosteq(mp.mpf(moment_component[0]), mp.mpf("0.1"))
    assert mp.almosteq(mp.mpf(moment_component[1]), mp.mpf(1))
    assert mp.almosteq(mp.mpf(first_sample_row[-1]), mp.mpf("0.1") + first_x)


def test_global_upper_ratio_probe_uses_ratio_blocks():
    mp.mp.dps = 30  # explicit: these compare with a bare mp.almosteq
    spec = _small_virasoro_spec()
    base = build_global_identical_scalar_control_problem(
        spec, global_identity=True
    )
    problem = build_identical_scalar_normalized_ratio_probe_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        mp.mpf("2"),
        identity_mode="exclude",
        base_problem=base,
        block_basis="ratio",
    )

    moment_meta = problem.pmp_metadata["moment"]
    assert moment_meta["block_basis"] == "F_over_self_dual_measure"
    assert moment_meta["sample_value_basis"] == "direct_ratio_at_sdp_nodes"
    assert problem.sample_values_by_block is not None

    first_block = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][0]
    probe_component = first_block["polynomials"][0][0][-1]
    first_sample_row = problem.sample_values_by_block[0][0]
    first_x = mp.mpf(first_block["samplePoints"][0])

    # Upper probes append -(Delta-r). With the small spec shift Delta=x+0.1
    # and r=2, this is 1.9 - x. A G_self multiplier here would reintroduce the
    # sign-changing interpolant failure fixed for direct global moments.
    assert len(probe_component) == len(first_block["samplePoints"])
    assert all(mp.mpf(value) == 0 for value in probe_component[2:])
    assert mp.almosteq(mp.mpf(probe_component[0]), mp.mpf("1.9"))
    assert mp.almosteq(mp.mpf(probe_component[1]), mp.mpf("-1"))
    assert mp.almosteq(mp.mpf(first_sample_row[-1]), mp.mpf("1.9") - first_x)


def test_moment_metadata_serializes_weights_into_manifest(tmp_path):
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        MomentWeight.monomial(Delta=2),
        MomentWeight.monomial(L=1),
        identity_mode="exclude",
    )

    manifest_path, _ = write_identical_scalar_gap_problem(problem, tmp_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    moment_meta = manifest["provenance"]["pmp_metadata"]["moment"]

    assert moment_meta["numerator"]["terms"][0]["powers"] == {"Delta": 2}
    assert moment_meta["denominator"]["terms"][0]["powers"] == {"L": 1}
    assert moment_meta["numerator"]["spin_filter"] is None


def test_normalized_bisection_helper_with_fake_solver():
    spec = _small_virasoro_spec()

    def fake_solve(_problem, r):
        return mp.mpf("-1") if r >= mp.mpf("2") else mp.mpf("1")

    result = bound_identical_scalar_normalized_moment_by_bisection(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        lo=mp.mpf("1"),
        hi=mp.mpf("3"),
        tol=mp.mpf("0.25"),
        solve_probe_upper_bound=fake_solve,
        identity_mode="exclude",
        max_iterations=10,
    )

    assert mp.mpf(result["lower_uncertified"]) < mp.mpf("2")
    assert mp.mpf(result["upper_certified"]) >= mp.mpf("2")


def test_normalized_lower_probe_uses_signed_weight():
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_ratio_lower_probe_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        mp.mpf("2"),
        identity_mode="exclude",
    )
    moment = problem.pmp_metadata["moment"]

    assert moment["kind"] == "normalized_ratio_probe"
    assert moment["direction"] == "lower"
    assert moment["operator_weight"]["terms"][0]["powers"] == {"Delta": 1}
    assert moment["operator_weight"]["terms"][1]["coefficient"].startswith("-2")
    assert moment["operator_weight"]["terms"][1]["powers"] == {}
    assert problem.objective_vector[-1] == mp.mpf(0)


def test_normalized_lower_probe_keeps_identity_only_in_threshold():
    spec = _small_virasoro_spec()
    problem = build_identical_scalar_normalized_ratio_lower_probe_problem(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        mp.mpf("2"),
        identity_mode="include",
    )
    moment = problem.pmp_metadata["moment"]
    with mp.workdps(spec.dps):       # the builders work at the spec's precision
        vacuum = self_dual_identity_measure_value(
            spec.c,
            spec.h_ext,
            spec.lam,
            spec.q_order,
            coordinate=spec.coordinate,
            chiral_norm=spec.chiral_norm,
        )

    assert problem.objective_vector[-1] == mp.mpf(0)
    with mp.workdps(spec.dps):
        assert abs(mp.mpf(moment["identity_denominator"]) - vacuum) < mp.mpf(10) ** (-(spec.dps - 5))
        assert abs(mp.mpf(moment["alpha_effective"]) - vacuum) < mp.mpf(10) ** (-(spec.dps - 5))


def test_normalized_lower_bisection_helper_with_fake_solver():
    spec = _small_virasoro_spec()

    def fake_solve(_problem, r):
        return mp.mpf("1") if r <= mp.mpf("2") else mp.mpf("-1")

    result = bound_identical_scalar_normalized_moment_lower_by_bisection(
        spec,
        MomentWeight.monomial(Delta=1),
        MomentWeight.one(),
        lo=mp.mpf("1"),
        hi=mp.mpf("3"),
        tol=mp.mpf("0.25"),
        solve_probe_lower_bound=fake_solve,
        identity_mode="exclude",
        max_iterations=10,
    )

    assert mp.mpf(result["lower_certified"]) <= mp.mpf("2")
    assert mp.mpf(result["upper_uncertified"]) > mp.mpf("2")


def _docker_sdpb_available() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        # an unresponsive Docker daemon must skip these tests, not hang collection
        result = subprocess.run(
            [docker, "image", "inspect", "bootstrapcollaboration/sdpb:master"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


_SIGN_TEST_SDPB_OPTS = {
    "--precision": 768,
    "--writeSolution": "x,y",
    "--maxIterations": 1000,
    # Above the conditioning floor of this tiny instance; the sign invariants
    # do not need a tight gap, only convergence to a finite optimum.
    "--dualityGapThreshold": "1e-10",
    "--noFinalCheckpoint": True,
}


@pytest.mark.skipif(
    not _docker_sdpb_available(),
    reason="requires the bootstrapcollaboration/sdpb:master docker image",
)
def test_unnormalized_moment_bound_signs_numeric(tmp_path):
    """Solve a small one-shot moment bound and check the sign invariant.

    Unnormalized moments over the free positive cone are one-sided unbounded, so
    ``upper(Delta)`` is +inf (SDPB stalls at max_complementarity) and only
    ``lower(Delta)`` and ``upper(-Delta)`` converge. Those two are the same PMP
    with opposite conversions, so ``lower(W) == -upper(-W)`` must hold to solver
    tolerance and both must converge -- a wrong objective/conversion sign breaks
    it. (This is a consistency check on the builder pair, not an independent
    value check; for that see the global-block validation in the diagnostics.)
    """
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("4"),
        h_ext=mp.mpf("0.0625"),
        delta_gap=mp.mpf("0.1"),
        lam=3,
        spin_list=(0, 2, 4),
        q_order=20,
        dps=120,
        npoles=8,
        pole_selection="contribution",
        num_threads=2,
    )
    W = MomentWeight.monomial(Delta=1)

    def solve(problem, tag, *, require_optimal):
        run = run_identical_scalar_gap_problem_in_docker(
            problem,
            tmp_path / tag,
            backend="sdp",
            sdpb_options=_SIGN_TEST_SDPB_OPTS,
            pmp2sdp_log_name="pmp2sdp.log",
            sdpb_log_name="sdpb.log",
        )
        obj = read_sdpb_objective(run.sdpb_out_dir) if run.sdpb_out_dir else None
        if require_optimal:
            assert run.sdpb_status == "optimal", f"{tag}: status {run.sdpb_status}"
        assert obj and "dual" in obj, f"{tag}: no dual objective"
        return obj["dual"]

    # lower(Delta) and upper(-Delta) are the bounded side and converge to optimal.
    lower = moment_lower_bound_from_objective(
        solve(build_identical_scalar_moment_lower_problem(spec, W), "lower",
              require_optimal=True)
    )
    upper_neg = moment_upper_bound_from_objective(
        solve(build_identical_scalar_moment_upper_problem(spec, -W), "upper_neg",
              require_optimal=True)
    )
    lower_via_neg = -upper_neg

    # lower(W) == -upper(-W) to solver tolerance (the builder-pair sign check).
    scale = max(abs(lower), mp.mpf("1e-20"))
    assert abs(lower - lower_via_neg) / scale < mp.mpf("1e-6"), (
        f"lower {mp.nstr(lower, 8)} != -upper(-W) {mp.nstr(lower_via_neg, 8)}"
    )


def _stub_base_problem_builder(monkeypatch, M):
    """Replace the (expensive) block build behind ``_cached_base_problem`` with a
    stub returning a fresh object per call, on a private cache; returns the list
    of specs the builder was called with."""
    calls = []

    def fake_build(spec):
        calls.append(spec)
        return object()

    monkeypatch.setattr(M, "build_identical_scalar_gap_problem", fake_build)
    monkeypatch.setattr(M, "_BASE_PROBLEM_CACHE", {})
    return calls


def test_rebuild_normalized_moment_problem_reuses_prepared_blocks(monkeypatch):
    """Certification rebuilds must share one block build, not repeat it.

    The positivity blocks depend only on the spec -- not on the direction, the
    identity mode, or the moment weights, which enter as the per-channel
    insertion assembled on top. Certification rebuilds the extended problem once
    per solved direction, i.e. FOUR times per moment task, so without reuse
    three of the four block builds are pure waste. That was the dominant cost of
    a moment task: ~12 minutes of a ~15-minute direction, against ~2.5 minutes
    of SDPB.  (The builder is stubbed: the property is the memoization.)
    """
    import mpmath as mp

    from virasoro_bootstrap import moments as M
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec
    from virasoro_bootstrap.large_spin import double_twist_set

    with mp.workdps(50):
        h = mp.mpf("0.25")
        spec = IdenticalScalarGapSpec(
            c=mp.mpf(8) / 7, h_ext=h, delta_gap=mp.mpf("0.1"),
            lam=3, q_order=16, dps=50, npoles=6, spin_list=(0, 2),
            large_spin_lmin=4, large_spin_twists=tuple(double_twist_set(h, 2)),
            num_threads=1,
        )
        calls = _stub_base_problem_builder(monkeypatch, M)
        first = M._cached_base_problem(spec)
        second = M._cached_base_problem(spec)
        # Identity, not merely equality: the point is that no rebuild happened.
        assert first is second
        assert calls == [spec]

        # A different spec must NOT collide.
        import dataclasses

        other = dataclasses.replace(spec, spin_list=(0, 2, 4))
        assert M._cached_base_problem(other) is not first
        assert calls == [spec, other]


def test_base_problem_cache_is_bounded(monkeypatch):
    """The cached problems are large; the cache must not grow without limit.
    (The builder is stubbed: the property is the cache bound.)"""
    import dataclasses

    import mpmath as mp

    from virasoro_bootstrap import moments as M
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec

    with mp.workdps(50):
        base = IdenticalScalarGapSpec(
            c=mp.mpf(8) / 7, h_ext=mp.mpf("0.25"), delta_gap=mp.mpf("0.1"),
            lam=3, q_order=16, dps=50, npoles=6, spin_list=(0, 2), num_threads=1,
        )
        calls = _stub_base_problem_builder(monkeypatch, M)
        for n in range(M._BASE_PROBLEM_CACHE_MAXSIZE + 2):
            M._cached_base_problem(
                dataclasses.replace(base, delta_gap=mp.mpf(1) / (n + 3))
            )
        assert len(calls) == M._BASE_PROBLEM_CACHE_MAXSIZE + 2
        assert len(M._BASE_PROBLEM_CACHE) <= M._BASE_PROBLEM_CACHE_MAXSIZE


def test_interstitial_twist_set_places_lines_between_slices():
    """The certifier must be able to look BETWEEN double-twist slices.

    The SDP imposes tower positivity only ON the discrete lines tau = tau_n,
    while unitarity permits operators at any tau >= 0 at large spin. The
    normalized-moment path's only tower knob was a double-twist COUNT, which
    extends the set upward and can therefore never express a line between two
    slices -- so that failure mode was structurally unmonitored.
    """
    import mpmath as mp

    from virasoro_bootstrap.certification import _interstitial_twist_set
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec
    from virasoro_bootstrap.large_spin import double_twist_set

    with mp.workdps(50):
        h = mp.mpf("0.8")
        base = tuple(double_twist_set(h, 4))
        spec = IdenticalScalarGapSpec(
            c=mp.mpf(8) / 7, h_ext=h, delta_gap=mp.mpf(0), lam=3,
            spin_list=(0, 2), q_order=16, dps=50, npoles=6,
            large_spin_lmin=4, large_spin_twists=base,
        )
        out = _interstitial_twist_set(spec, 3)
        assert out is not None
        # every base slice survives
        for t in base:
            assert any(abs(mp.mpf(t) - o) < mp.mpf(10) ** -30 for o in out)
        # and there are genuinely interior points in the first gap
        lo, hi = mp.mpf(base[0]), mp.mpf(base[1])
        interior = [o for o in out if lo < o < hi]
        assert len(interior) == 3, interior
        # plus lines above the top slice
        assert max(out) > max(mp.mpf(t) for t in base)


def test_zero_functional_does_not_certify_a_moment_upper_problem(tmp_path):
    """A normalized-moment PMP has one component more than the F-pairs (the moment weight s); SDPB's y has
    the pivot s eliminated, so it is len(F_pairs) long and must still be reconstructed -- otherwise only z.F
    is checked and z = 0 certifies although the PMP constraint -G W is negative."""
    from virasoro_bootstrap.certification import certify_dual_from_sdpb_output
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec
    from virasoro_bootstrap.moments import MomentWeight, build_identical_scalar_moment_upper_problem
    from virasoro_bootstrap.sdp_writer import write_sdp_directory

    spec = IdenticalScalarGapSpec(c=mp.mpf(2), h_ext=mp.mpf("0.1"), delta_gap=mp.mpf("0.5"), lam=3,
                                  spin_list=(0, 2), q_order=20, dps=40, npoles=4, num_threads=1)
    problem = build_identical_scalar_moment_upper_problem(spec, MomentWeight.monomial(Delta=1),
                                                          identity_mode="exclude")
    write_sdp_directory(problem.pmp_build.doc, tmp_path / "sdp", eval_dps=60,
                        sample_values_by_block=problem.sample_values_by_block)
    (tmp_path / "sdpb_out").mkdir()
    n = len(problem.F_pairs)
    (tmp_path / "sdpb_out" / "y.txt").write_text(f"{n} 1\n" + "0\n" * n)
    ok, _details = certify_dual_from_sdpb_output(problem, tmp_path / "sdpb_out", n_test_points=20,
                                                 delta_range=5, q_gate="off")
    assert not ok


def test_normalized_upper_and_certification_rebuild_keep_basis_and_tower_choice():
    """The upper builder's metadata and the certification rebuild carry ``block_basis`` and ``include_tower``
    (both were dropped: a no-tower ratio problem was rebuilt with the tower, in the non-ratio basis)."""
    from virasoro_bootstrap import moments as M

    with mp.workdps(60):
        spec = _small_virasoro_tower_spec()
        one = MomentWeight((MomentTerm(1, {}),))
        delta = MomentWeight((MomentTerm(1, {"Delta": 1}),))
        problem = M.build_identical_scalar_normalized_moment_upper_problem(
            spec, delta, one, block_basis="ratio", include_tower=False)
        meta = problem.pmp_metadata["moment"]
        assert meta["include_tower"] is False
        assert meta["block_basis"] == "F_over_self_dual_measure"
        rebuilt = M.rebuild_normalized_moment_problem(problem)
        assert rebuilt.pmp_metadata["moment"]["include_tower"] is False
        assert rebuilt.pmp_metadata["moment"]["block_basis"] == "F_over_self_dual_measure"
        assert len(rebuilt.pmp_build.doc["PositiveMatrixWithPrefactorArray"]) == len(
            problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"])


def test_spin_filter_is_refused_inside_the_large_spin_tower():
    """A spin filter matched tower channels by their label ell = int(Lmax): spin_filter = Lmax weighted every tower
    slice (all spins >= Lmax), spin_filter = Lmax + 2 weighted none."""
    from virasoro_bootstrap import moments as M

    with mp.workdps(60):
        spec = _small_virasoro_tower_spec()
        prepared = M.prepare_identical_scalar_gap_build(spec) if hasattr(M, "prepare_identical_scalar_gap_build") else None
        low = MomentWeight((MomentTerm(1, {}),), spin_filter=2)
        problem = M.build_identical_scalar_normalized_moment_upper_problem(
            spec, low, MomentWeight.one(), identity_mode="exclude", prepared=prepared)
        assert problem is not None
        with pytest.raises(ValueError, match="inside the large-spin tower"):
            M.build_identical_scalar_normalized_moment_upper_problem(
                spec, MomentWeight((MomentTerm(1, {}),), spin_filter=int(spec.large_spin_lmin)), MomentWeight.one(),
                identity_mode="exclude", prepared=prepared)
