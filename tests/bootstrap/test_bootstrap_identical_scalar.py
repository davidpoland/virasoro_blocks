"""Tests for the minimal identical-scalar bootstrap gap driver."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import ANY

import mpmath as mp
import pytest

from virasoro_blocks.derivs import (
    G_ws_table,
    G_xt_table,
    G_yyb_table,
    global_G_yyb_table,
)
from virasoro_bootstrap import certification as certification_module
from virasoro_bootstrap.controls.global_control import (
    build_global_identical_scalar_control_problem,
    build_global_scalar_spin_channel,
    build_numeric_virasoro_identical_scalar_control_problem,
    build_numeric_virasoro_scalar_spin_channel,
)
from virasoro_blocks.global_blocks import global_block_exp_base
from virasoro_bootstrap.defaults import (
    DEFAULT_LARGE_SPIN_POLE_SELECTION,
    DEFAULT_POLE_SELECTION,
)
from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
    classify_sdpb_log_text,
    prepare_identical_scalar_gap_build,
    read_sdpb_objective,
    run_identical_scalar_gap_problem_in_docker,
    write_identical_scalar_gap_problem,
)
from virasoro_bootstrap.crossing import F_minus_pairs


def _has_positive_real_pole(serialized) -> bool:
    tol = mp.mpf("1e-25")
    return any(
        abs(mp.im(mp.mpc(pole))) <= tol and mp.re(mp.mpc(pole)) > tol
        for pole in serialized.pole_list_x
    )


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


def test_singular_c_numeric_sampling_matches_symbolic_pmp():
    """At Ising c=1/2 with sigma externals the regularized interpolation draws
    its node values from the value-space regularized numeric sampler
    (sample_via='numeric', the default) instead of the ill-conditioned symbolic
    P(Delta)/D(Delta). Both must produce the same *block* to bootstrap tolerance.

    The comparison is at the block level (not leaf-by-leaf on the PMP doc): with
    the default `contribution_optimal` selector the two sampling
    paths can keep equivalent-but-not-identical pole sets (the numeric and
    symbolic scores agree only to the DFT aliasing floor), so the serialized
    prefactor/sample-points/polynomial differ in representation even though the
    reconstructed block is the same. We also pin both against the independent
    npoles="all" reference (all poles kept) to confirm each is correct."""
    common = dict(
        c=mp.mpf(1) / 2, h_ext=mp.mpf(1) / 16, delta_gap=mp.mpf("0.9"),
        lam=2, spin_list=(0, 2), q_order=20, dps=50, regularize="auto", num_threads=1,
    )
    prob_num = build_identical_scalar_gap_problem(IdenticalScalarGapSpec(
        sample_via="numeric", npoles=6,
        **common))
    prob_sym = build_identical_scalar_gap_problem(IdenticalScalarGapSpec(
        sample_via="symbolic", npoles=6,
        **common))
    prob_full = build_identical_scalar_gap_problem(IdenticalScalarGapSpec(
        npoles="all", **common))
    assert prob_num.used_regularize and prob_sym.used_regularize

    mp.mp.dps = 50
    channels = zip(prob_num.spin_channels, prob_sym.spin_channels,
                   prob_full.spin_channels)
    worst_ns = mp.mpf(0)   # numeric vs symbolic: same interpolant
    worst_ref = mp.mpf(0)  # each vs the all-poles reference: interp accuracy
    for ch_n, ch_s, ch_f in channels:
        # Δ values in the physical window, stepped off pole locations.
        for off in (mp.mpf("0.37"), mp.mpf("1.13"), mp.mpf("2.71")):
            D = ch_f.delta_min + off
            vn = ch_n.serialized.evaluate("zzb", D)
            vs = ch_s.serialized.evaluate("zzb", D)
            vf = ch_f.serialized.evaluate("zzb", D)
            for m in range(min(len(vn), 2)):
                for n in range(min(len(vn[m]), 2)):
                    a, b, f = (mp.mpc(vn[m][n]), mp.mpc(vs[m][n]),
                               mp.mpc(vf[m][n]))
                    scale = max(abs(f), mp.mpf("1e-30"))
                    worst_ns = max(worst_ns, abs(a - b) / scale)
                    worst_ref = max(worst_ref, abs(a - f) / scale,
                                    abs(b - f) / scale)
    # The numeric and symbolic node-value paths build the SAME interpolant
    # (they keep the same pole set; only the node-value conditioning differs),
    # so they agree to working precision -- this is the real claim.
    assert worst_ns < mp.mpf(10) ** -12, f"numeric vs symbolic {mp.nstr(worst_ns, 5)}"
    # And both reproduce the all-poles block to npoles=6 interpolation accuracy
    # (a looser bound: derivative entries at the window edge carry the most
    # truncation error), confirming the numeric c < 1 path is correct.
    assert worst_ref < mp.mpf(10) ** -6, f"vs full reference {mp.nstr(worst_ref, 5)}"


def test_skeleton_regularize_c_ge_one_matches_symbolic_pmp():
    """At the singular c = 1 the numeric path builds its regularized tables from
    a degree-only *skeleton* -- no ~1e500-coeff symbolic numerators -- since the
    collision strip is a no-op at c >= 1. With a geometric (non-residue) pole
    selection the skeleton serves the whole pipeline, and the PMP must match the
    symbolic-sampling path (which builds the full numerators) to bootstrap
    tolerance: same interpolant, only the numerator coefficients are skipped on
    the numeric path. (h_ext = 0.1 is a non-degenerate external at c = 1, so the
    block keeps its Delta-poles; h = 1/16 happens to be fully degenerate there.)
    """
    common = dict(
        c=mp.mpf(1), h_ext=mp.mpf("0.1"), delta_gap=mp.mpf("0.9"),
        lam=2, spin_list=(0, 2), q_order=12, dps=50,        npoles=6, num_threads=1, regularize="auto", pole_selection="nearest",
    )
    prob_num = build_identical_scalar_gap_problem(
        IdenticalScalarGapSpec(sample_via="numeric", **common))
    prob_sym = build_identical_scalar_gap_problem(
        IdenticalScalarGapSpec(sample_via="symbolic", **common))
    assert prob_num.used_regularize and prob_sym.used_regularize

    def _leaves(doc):
        out = []
        def rec(x):
            if isinstance(x, dict):
                for k in sorted(x):
                    rec(x[k])
            elif isinstance(x, list):
                for v in x:
                    rec(v)
            else:
                out.append(x)
        rec(doc)
        return out

    a, b = _leaves(prob_num.pmp_build.doc), _leaves(prob_sym.pmp_build.doc)
    assert len(a) == len(b)
    worst = mp.mpf(0)
    for x, y in zip(a, b):
        try:
            za, zb = mp.mpf(x), mp.mpf(y)
        except (TypeError, ValueError):
            assert x == y
            continue
        worst = max(worst, abs(za - zb) / max(abs(za), abs(zb), mp.mpf(1)))
    assert worst < mp.mpf(10) ** -12, f"PMP diff {mp.nstr(worst, 5)}"


def test_skeleton_regularize_builds_no_symbolic_numerators():
    """The c >= 1 numeric regularize path with a geometric pole selection must
    never construct the symbolic numerators: every exact scalar table it builds
    carries numerators=None (degree-only skeleton)."""
    from virasoro_bootstrap import identical_scalar as _isc
    tables = _isc._build_exact_scalar_tables(
        mp.mpf(1), mp.mpf("0.1"), "zzb", (0, 2), lam=2, q_order=20,
        used_regularize=True, resolved_method="interpolation",
        chiral_norm=4, sample_via="numeric", pole_selection="nearest",
    )
    for ell, table in tables.items():
        assert table.numerators is None, f"spin {ell} built symbolic numerators"
        assert table.numerator_degrees is not None
        assert table.numeric_sampler is not None


def test_skeleton_regularize_builds_no_symbolic_numerators_at_c_lt_1():
    """The minimal-model numeric regularize path now uses the same skeleton
    fast path for non-legacy selectors: exact scalar tables are degree-only and
    rely on the regularized numeric sampler for node values."""
    from virasoro_bootstrap import identical_scalar as _isc
    tables = _isc._build_exact_scalar_tables(
        mp.mpf(1) / 2, mp.mpf(1) / 16, "zzb", (0, 2), lam=2, q_order=20,
        used_regularize=True, resolved_method="interpolation",
        chiral_norm=4, sample_via="numeric", pole_selection="contribution",
    )
    for ell, table in tables.items():
        assert table.numerators is None, f"spin {ell} built symbolic numerators"
        assert table.numerator_degrees is not None
        assert table.numeric_sampler is not None


def test_skeleton_regularize_residue_selection_falls_back_to_full():
    """Legacy residue / contribution_leading pole selection at a singular c >= 1
    needs per-pole LEADING residue magnitudes the regularized numeric sampler
    cannot supply, so the build must fall back to the full symbolic numerators
    (numerators is not None). (The corrected `contribution` ranks by the full
    principal part, which IS skeleton-extractable -- so it does not fall back;
    hence this test uses the legacy `contribution_leading`.)"""
    from virasoro_bootstrap import identical_scalar as _isc
    tables = _isc._build_exact_scalar_tables(
        mp.mpf(1), mp.mpf("0.1"), "zzb", (2,), lam=2, q_order=20,
        used_regularize=True, resolved_method="interpolation",
        chiral_norm=4, sample_via="numeric", pole_selection="contribution_leading",
    )
    for ell, table in tables.items():
        assert table.numerators is not None, (
            f"spin {ell}: residue selection must keep symbolic numerators"
        )


def test_method_interpolation_now_permitted_at_c_below_one():
    """method='interpolation' is no longer disabled at c < 1: the
    `interpolate_delta_table(force_keep_real_positive=True)` default routes
    the real-positive shifted Kac poles into the InterpolatedDeltaTable's
    prefactor, where the downstream `lower_positive_real_poles` post-
    processor picks them up the same way it does from npoles="all". The
    per-block max polynomial degree drops by exactly `dropped_pole_count`,
    so `method='auto'` now selects interpolation at c < 1 too whenever
    `npoles` is set (same heuristic as c >= 1). `method='auto'` only
    falls back to 'full' when `npoles is None`."""
    # Common spec at Ising c=1/2
    common = dict(
        c=mp.mpf(1) / 2,
        h_ext=mp.mpf(1) / 16,
        delta_gap=mp.mpf("1.0"),
        lam=4,
        spin_list=(0, 2),
        q_order=8,
        dps=60,
        regularize='auto',
        num_threads=1,
    )

    # auto + npoles=4: now routes to interpolation (the post-fix behavior;
    # an explicit interpolation request at c < 1 used to raise).
    spec_auto = IdenticalScalarGapSpec(npoles=4, **common)
    prob_auto = build_identical_scalar_gap_problem(spec_auto)
    assert prob_auto.resolved_method == "interpolation"
    assert prob_auto.used_regularize  # c=1/2 is the Ising minimal model
    # The c < 1 build lowers its real-positive poles and carries a nonzero
    # identity vector (formerly test_build_identical_scalar_gap_problem_c_lt_1).
    assert len(prob_auto.identity_vector) == len(prob_auto.F_pairs)
    assert any(abs(mp.mpc(value)) > mp.mpf("1e-20") for value in prob_auto.identity_vector)
    assert any(
        lowering is not None and len(lowering.selected_groups) > 0
        for lowering in prob_auto.pmp_build.positive_pole_lowerings
    )
    assert all(
        not _has_positive_real_pole(block)
        for block in prob_auto.pmp_build.positivity_blocks
    )
    # the lowered blocks get new nodes: their SDP data are the numeric block values there (times the clearing
    # factor), never the cleared polynomial evaluated
    assert set(prob_auto.sample_values_by_block) == set(range(len(prob_auto.spin_channels)))

    # auto + npoles=None: still resolves to 'full' (interpolation requires
    # a pole-count budget).
    spec_full = IdenticalScalarGapSpec(npoles="all", **common)
    prob_full = build_identical_scalar_gap_problem(spec_full)
    assert prob_full.resolved_method == "full"

    # Numerical agreement between full and interp at safe Δ values above
    # each channel's delta_min. Limit to Δ within a few units of delta_min
    # (i.e. in the bootstrap's physical range): at large Δ the block decays
    # as exp_base^Δ ≈ 0.173^Δ, so any tiny absolute interpolation error
    # blows up in relative terms outside the interpolation interval.
    for ch_f, ch_i in zip(prob_full.spin_channels, prob_auto.spin_channels):
        # Avoid pole locations by stepping with non-integer offsets.
        for offset in (mp.mpf("0.3"), mp.mpf("0.7"), mp.mpf("1.5")):
            Delta = ch_f.delta_min + offset
            v_f = ch_f.serialized.evaluate('zzb', Delta)
            v_i = ch_i.serialized.evaluate('zzb', Delta)
            for m in range(min(len(v_f), 2)):
                for n in range(min(len(v_f[m]), 2)):
                    val_f = mp.mpc(v_f[m][n])
                    val_i = mp.mpc(v_i[m][n])
                    scale = max(abs(val_f), mp.mpf("1e-30"))
                    rel_err = abs(val_f - val_i) / scale
                    assert rel_err < mp.mpf("1e-5"), (
                        f"interp vs full mismatch at ell={ch_f.ell}, m={m}, "
                        f"n={n}, Δ={float(Delta):.3f}: rel_err={float(rel_err):.3e}"
                    )


def test_regularize_on_at_complex_pole_c_yields_real_pmp():
    """Forcing regularize='on' off a recognized singular locus (here c=1.1, where
    1 < c < 25 makes the Kac Delta-poles complex) used to crash the PMP realness
    guard: the denominator, built from complex (conjugate-paired) (Delta - root)
    factors, only cancels its imaginary part approximately when evaluated, leaking
    a residue into the interpolant even though the block is real. The interpolation
    now rebuilds the denominator from its conjugate-symmetric roots as real
    quadratics (real arithmetic, exact), so the block value -- and every emitted
    numerator -- is exactly real with no chopping."""
    mp.mp.dps = 60
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("1.1"), h_ext=mp.mpf(1) / 16, delta_gap=mp.mpf("1.0"),
        lam=2, spin_list=(0, 2), q_order=12, dps=60,        npoles=6, pole_selection="contribution", regularize="on", num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)
    assert problem.used_regularize
    # Every emitted block numerator is real-valued (the complex-pole residue was
    # projected away upstream); the field may stay mpc after the prefactor
    # widens it, but the imaginary part is gone, so the strict PMP guard passes.
    max_im = mp.mpf(0)
    for ch in problem.spin_channels:
        for rows in ch.prefactored.numerators.values():
            for row in rows:
                for poly in row:
                    for coeff in poly.coeffs:
                        max_im = max(max_im, abs(mp.im(mp.mpc(coeff))))
    assert max_im == 0, f"block numerator left complex: max|Im|={mp.nstr(max_im, 4)}"
    # The PMP doc assembled without tripping the realness guard.
    assert problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"]


def test_identity_vector_regularizes_at_singular_minimal_model():
    """The vacuum identity vector is the only block path that used to skip
    regularization. At c = 1/2 with q_order high enough to reach the Kac
    collision the bare numeric recursion divides by zero; the regularized
    identity vector instead succeeds and reproduces the nearby bare block."""
    from virasoro_blocks.derivs import G_zzb_table
    from virasoro_bootstrap.identical_scalar import _numeric_identity_vector

    mp.mp.dps = 60
    lam, he, q = 2, mp.mpf(1) / 16, 20
    F = [(m, n) for m in range(lam + 1) for n in range(lam + 1)]
    eps = mp.mpf(10) ** -100

    # The bare recursion divides by zero at c = 1/2 once q_order reaches the collision -- this is what
    # the regularization fixes (G_zzb_table itself regularizes at a recognized singular c since 2026-09-27).

    reg = _numeric_identity_vector(
        mp.mpf(1) / 2, he, lam, q, F, coordinate="zzb", chiral_norm=4)
    near = G_zzb_table(eps, 0, [he] * 4, lam, lam,
                       c=mp.mpf(1) / 2 + mp.mpf(10) ** -8, q_order=q, z_shift=-2 * he)
    near_v = [near[m][n] for (m, n) in F]
    rel = max(abs(mp.mpc(x) - mp.mpc(y)) / (1 + abs(mp.mpc(y)))
              for x, y in zip(reg, near_v))
    assert rel < mp.mpf(10) ** -6


def test_identity_vector_regularizes_at_c_ge_1_singular():
    """The singular c >= 1 identity vector is built from the Laurent-regularized
    chiral vacuum Taylor coefficients and stays continuous with nearby generic c.
    """
    from virasoro_blocks.derivs import G_zzb_table
    from virasoro_bootstrap.identical_scalar import _numeric_identity_vector

    mp.mp.dps = 60
    lam, he, q = 3, mp.mpf(1) / 16, 20
    F = [(m, n) for m in range(lam + 1) for n in range(lam + 1)]
    eps = mp.mpf(10) ** -30
    delta = mp.mpf(10) ** -8

    for c in (mp.mpf(1), mp.mpf(25)):
        reg = _numeric_identity_vector(
            c, he, lam, q, F, coordinate="zzb", chiral_norm=4,
        )
        near = G_zzb_table(
            eps, 0, [he] * 4, lam, lam,
            c=c + delta, q_order=q, z_shift=-2 * he,
        )
        near_v = [near[m][n] for (m, n) in F]
        rel = max(
            abs(mp.mpc(x) - mp.mpc(y)) / (1 + abs(mp.mpc(y)))
            for x, y in zip(reg, near_v)
        )
        assert rel < mp.mpf(10) ** -6


def test_identity_vector_genuinely_singular_for_nondegenerate_external():
    """At a non-dyadic minimal model (M(4,5), c = 7/10) the level-12 vacuum null
    state resonates with the exchanged weight (d_{4,3} = 0). For a *generic*
    external (h_ext = 1/16, not a tricritical-Ising degenerate field) it does not
    decouple, so the vacuum block genuinely diverges -- this must raise, not
    silently return garbage. (Regression: at high work_dps the leaked structural
    zero used to slip below tolerance and return a ~1e60 value.)
    """
    from virasoro_bootstrap.identical_scalar import _numeric_identity_vector

    mp.mp.dps = 60
    lam, q = 3, 20
    F = [(m, n) for m in range(lam + 1) for n in range(lam + 1)]
    with pytest.raises(ValueError, match="minimal model|genuinely singular"):
        _numeric_identity_vector(
            "7/10", mp.mpf(1) / 16, lam, q, F,
            coordinate="zzb", chiral_norm=4,
        )


def test_identity_vector_at_nondyadic_minimal_model_degenerate_external():
    """At c = 7/10 (b^2 = -4/5, non-dyadic q) with a *degenerate* external
    (h_ext = 3/80, a tricritical-Ising field) the vacuum null decouples and the
    block is finite. This needs the exact-integer structural-zero snaps in the
    Laurent path -- without them m*b^2 + n / the collision constants leak to
    ~1e-dps roundoff for non-dyadic q. Check finiteness and continuity with the
    bare block just off the locus.
    """
    from virasoro_blocks.derivs import G_zzb_table
    from virasoro_bootstrap.identical_scalar import _numeric_identity_vector

    mp.mp.dps = 80
    lam, he, q = 2, Fraction(3, 80), 24
    F = [(m, n) for m in range(lam + 1) for n in range(lam + 1)]
    reg = _numeric_identity_vector(
        "7/10", he, lam, q, F, coordinate="zzb", chiral_norm=1,
    )
    near = G_zzb_table(
        mp.mpf(10) ** -50, 0, [he] * 4, lam, lam,
        c=mp.mpf(7) / 10 + mp.mpf(10) ** -12, q_order=q, z_shift=-2 * he,
    )
    near_v = [near[m][n] for (m, n) in F]
    rel = max(
        abs(mp.mpc(x) - mp.mpc(y)) / (1 + abs(mp.mpc(y)))
        for x, y in zip(reg, near_v)
    )
    assert rel < mp.mpf(10) ** -9


@pytest.mark.parametrize(
    ("coordinate", "builder"),
    [
        ("yyb", G_yyb_table),
        ("xt", G_xt_table),
        ("ws", G_ws_table),
    ],
)
def test_identity_vector_regularizes_at_singular_c_for_other_coordinates(
    coordinate,
    builder,
):
    from virasoro_bootstrap.identical_scalar import _numeric_identity_vector

    mp.mp.dps = 50
    lam, he, q = 2, mp.mpf(1) / 16, 12
    F = F_minus_pairs(lam, coordinate=coordinate)
    eps = mp.mpf(10) ** -30
    delta = mp.mpf(10) ** -8

    reg = _numeric_identity_vector(
        mp.mpf(1), he, lam, q, F, coordinate=coordinate, chiral_norm=4,
    )
    near = builder(
        eps,
        0,
        [he] * 4,
        lam,
        lam,
        c=mp.mpf(1) + delta,
        q_order=q,
        z_shift=-2 * he,
    )
    near_v = [near[m][n] for (m, n) in F]
    rel = max(
        abs(mp.mpc(x) - mp.mpc(y)) / (1 + abs(mp.mpc(y)))
        for x, y in zip(reg, near_v)
    )
    assert rel < mp.mpf(10) ** -6


def test_build_identical_scalar_gap_problem_c_gt_1():
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=3,
        spin_list=(0, 2),
        q_order=8,
        dps=60,
        npoles=4, num_threads=1,
    )

    problem = build_identical_scalar_gap_problem(spec)

    assert problem.resolved_method == "interpolation"
    assert problem.spec.pole_selection == DEFAULT_POLE_SELECTION
    assert not problem.used_regularize
    assert len(problem.F_pairs) > 0
    assert len(problem.identity_vector) == len(problem.F_pairs)
    assert len(problem.spin_channels) == 2
    assert [channel.ell for channel in problem.spin_channels] == [0, 2]
    assert problem.spin_channels[0].delta_min == mp.mpf("1.0")
    assert problem.spin_channels[1].delta_min == mp.mpf("2.0")
    assert len(problem.pmp_build.positivity_blocks) == 2
    assert all(
        lowering is not None and lowering.selected_groups == ()
        for lowering in problem.pmp_build.positive_pole_lowerings
    )
    assert all(
        not _has_positive_real_pole(block)
        for block in problem.pmp_build.positivity_blocks
    )
    assert len(problem.pmp_build.doc["normalization"]) == len(problem.F_pairs)


def test_build_identical_scalar_gap_problem_threads_explicit_sample_points():
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=3,
        spin_list=(0, 2),
        q_order=8,
        dps=60,
        npoles=4,
        n_sample_points=9, num_threads=1,
    )

    problem = build_identical_scalar_gap_problem(spec)

    assert problem.spec.n_sample_points == 9
    for channel in problem.spin_channels:
        assert channel.serialized.sample_points_x is not None
        assert len(channel.serialized.sample_points_x) == 9


def test_full_method_channels_carry_sample_points():
    """`npoles="all"` is unified to keep-all interpolation, so every channel
    now carries sample points (it can use the sdp writer like interpolation)."""
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"), h_ext=mp.mpf("0.15"), delta_gap=mp.mpf("1.0"),
        lam=3, spin_list=(0, 2), q_order=8, dps=60, npoles="all", num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)
    for channel in problem.spin_channels:
        assert channel.serialized.sample_points_x is not None
        assert len(channel.serialized.sample_points_x) > 0
    # The PMP carries explicit samplePoints for every block (sdp-writer ready).
    blocks = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"]
    assert all("samplePoints" in b for b in blocks)


def test_sample_via_symbolic_matches_numeric():
    """`sample_via` is orthogonal to the interpolant: numeric and symbolic
    node evaluation produce the same reduced block to working precision."""
    mp.mp.dps = 120

    def build(sample_via):
        spec = IdenticalScalarGapSpec(
            c=mp.mpf("4"), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf("0.5"),
            lam=3, spin_list=(0, 2), q_order=14, dps=120,            npoles=4, sample_via=sample_via, num_threads=1,
        )
        return build_identical_scalar_gap_problem(spec).spin_channels[0].prefactored

    a = build("numeric")
    b = build("symbolic")
    coord = next(iter(a.numerators))
    max_diff = mp.mpf(0)
    for row_a, row_b in zip(a.numerators[coord], b.numerators[coord]):
        for poly_a, poly_b in zip(row_a, row_b):
            for ca, cb in zip(poly_a.coeffs, poly_b.coeffs):
                max_diff = max(max_diff, abs(mp.mpc(ca) - mp.mpc(cb)))
    # c = 4 is recursion-singular, so the numeric path reconstructs node block
    # values through the regularized sampler's principal-part contour (radius =
    # radius_fraction * true-nearest-pole gap). That contour caps the agreement
    # at ~1e-70 at dps=120 -- still ~50 digits, far below any bootstrap need.
    # (A tighter 1e-90 only held by accident when the nearest-gap helper
    # under-estimated the radius; the correct 2d nearest gap is required for
    # spinning pole selection, see test_sample_via_numeric_matches_symbolic_spinning.)
    assert max_diff < mp.mpf("1e-60")


def test_sdp_v2_clean_samples_match_pmp_polynomials_with_per_entry_nodes():
    """Clean v2 samples must represent the reduced block encoded in the PMP.

    With `shared_sample_points=False` (the default), lower-degree entries fit on
    their own node sets, so re-sampling the *exact* block at the shared writer
    nodes drifts from the emitted reduced polynomial. The stored v2 values must
    instead match the PMP polynomial at those shared nodes.
    """
    mp.mp.dps = 120

    def evalpoly(coeffs, x):
        value = mp.mpf(0)
        for coeff in reversed(coeffs):
            value = value * x + coeff
        return value

    spec = IdenticalScalarGapSpec(
        c=mp.mpf("4"), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf("6.0"),
        lam=4, spin_list=(0, 2, 4, 6), q_order=18, dps=120,
        npoles=6, pole_selection="contribution",
        num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    block_index = 3  # ell = 6: this block showed the Stage-B mismatch.
    block_doc = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][block_index]
    clean_values = problem.sample_values_by_block[block_index]
    sample_points = [mp.mpf(s) for s in block_doc["samplePoints"]]
    polynomials = block_doc["polynomials"][0][0]

    for k, x in enumerate(sample_points):
        for a, coeffs in enumerate(polynomials):
            truth = evalpoly([mp.mpf(c) for c in coeffs], x)
            rel = abs(clean_values[k][a] - truth) / max(abs(truth), mp.mpf(1))
            assert rel < mp.mpf("1e-80")


def test_build_identical_scalar_gap_problem_supports_yyb_coordinate():
    h_ext = mp.mpf("0.15")
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"),
        h_ext=h_ext,
        delta_gap=mp.mpf("1.0"),
        lam=3,
        spin_list=(0, 2),
        q_order=8,
        dps=60,
        npoles="all",
        coordinate="yyb", num_threads=1,
    )

    problem = build_identical_scalar_gap_problem(spec)

    assert problem.spec.coordinate == "yyb"
    assert len(problem.pmp_build.positivity_blocks) == 2

    channel = problem.spin_channels[1]
    Delta = mp.mpf("2.5")
    got = channel.serialized.evaluate("yyb", Delta)
    want = G_yyb_table(
        Delta,
        channel.ell,
        [h_ext] * 4,
        spec.lam,
        spec.lam,
        c=spec.c,
        q_order=spec.q_order,
        z_shift=-2 * h_ext,
    )
    # scalar G folds to the n <= m lower triangle (z<->zbar symmetry); the
    # n > m half is determined by symmetry, never read by the crossing
    # functional, and so is not interpolated -- it is emitted as exact zero.
    for m in range(2):
        for n in range(2):
            if n <= m:
                scale = max(abs(mp.mpc(want[m][n])), mp.mpf("1e-30"))
                rel = abs(mp.mpc(got[m][n]) - mp.mpc(want[m][n])) / scale
                assert rel < mp.mpf("5e-8")
            else:
                assert mp.mpc(got[m][n]) == 0


def test_build_identical_scalar_gap_problem_rejects_unsupported_coordinate():
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=3,
        coordinate="qqb", num_threads=1,
    )

    with pytest.raises(ValueError, match="coordinate must be one of"):
        build_identical_scalar_gap_problem(spec)


def test_build_global_scalar_spin_channel_matches_direct_global_table():
    mp.mp.dps = 50
    h_ext = mp.mpf("0.15")
    channel = build_global_scalar_spin_channel(
        ell=2,
        delta_min=mp.mpf("2.0"),
        h_ext=h_ext,
        lam=3,
        chiral_norm=mp.mpf(4),
        n_sample_points=17,
    )

    from virasoro_blocks.derivs import global_G_zzb_table

    for Delta in (mp.mpf("2.25"), mp.mpf("3.0"), mp.mpf("4.5")):
        got = channel.serialized.evaluate("zzb", Delta)
        want = global_G_zzb_table(
            Delta,
            2,
            [h_ext] * 4,
            3,
            3,
            z_shift=-2 * h_ext,
            chiral_norm=mp.mpf(4),
        )
        for m in range(2):
            for n in range(2):
                scale = max(abs(mp.mpc(want[m][n])), mp.mpf("1e-30"))
                rel = abs(mp.mpc(got[m][n]) - mp.mpc(want[m][n])) / scale
                assert rel < mp.mpf("3e-7")


def test_build_global_scalar_spin_channel_supports_yyb_coordinate():
    mp.mp.dps = 50
    h_ext = mp.mpf("0.15")
    channel = build_global_scalar_spin_channel(
        coordinate="yyb",
        ell=2,
        delta_min=mp.mpf("2.0"),
        h_ext=h_ext,
        lam=3,
        chiral_norm=mp.mpf(4),
        n_sample_points=17,
    )

    for Delta in (mp.mpf("2.25"), mp.mpf("3.0"), mp.mpf("4.5")):
        got = channel.serialized.evaluate("yyb", Delta)
        want = global_G_yyb_table(
            Delta,
            2,
            [h_ext] * 4,
            3,
            3,
            z_shift=-2 * h_ext,
            chiral_norm=mp.mpf(4),
        )
        for m in range(2):
            for n in range(2):
                scale = max(abs(mp.mpc(want[m][n])), mp.mpf("1e-30"))
                rel = abs(mp.mpc(got[m][n]) - mp.mpc(want[m][n])) / scale
                assert rel < mp.mpf("3e-7")


def test_build_global_scalar_spin_channel_uses_verified_global_base():
    mp.mp.dps = 50
    h_ext = mp.mpf("0.15")
    channel = build_global_scalar_spin_channel(
        ell=2,
        delta_min=mp.mpf("2.0"),
        h_ext=h_ext,
        lam=3,
        chiral_norm=mp.mpf(4),
        n_sample_points=17,
    )

    expected = global_block_exp_base(mp.mpf(4))
    assert abs(mp.mpf(channel.serialized.base) - expected) < mp.mpf("1e-40")
    # N^Delta is no longer reapplied: blocks stay in the (4 rho / N) bootstrap-
    # natural normalization, so prefactored shares the serialized base (the N factor
    # is absorbed into the OPE coefficients).
    assert abs(mp.mpf(channel.prefactored.base) - expected) < mp.mpf("1e-40")


def test_build_global_identical_scalar_control_problem_reuses_identity_vector():
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("200"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=3,
        spin_list=(0, 2),
        q_order=10,
        dps=60,
        npoles=4, num_threads=1,
    )
    virasoro_problem = build_identical_scalar_gap_problem(spec)
    global_problem = build_global_identical_scalar_control_problem(
        spec,
        n_sample_points=9,
    )

    assert global_problem.resolved_method == "global_control"
    assert global_problem.identity_vector == virasoro_problem.identity_vector
    assert [channel.ell for channel in global_problem.spin_channels] == [0, 2]
    assert len(global_problem.pmp_build.positivity_blocks) == 2
    assert len(global_problem.pmp_build.doc["normalization"]) == len(global_problem.F_pairs)


def test_build_global_identical_scalar_control_problem_supports_yyb_coordinate():
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("200"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=3,
        coordinate="yyb",
        spin_list=(0, 2),
        q_order=10,
        dps=60,
        npoles=4, num_threads=1,
    )

    problem = build_global_identical_scalar_control_problem(spec, n_sample_points=9)

    assert problem.spec.coordinate == "yyb"
    assert len(problem.pmp_build.positivity_blocks) == 2
    assert problem.spin_channels[1].serialized.evaluate("yyb", mp.mpf("2.5"))


def test_build_numeric_virasoro_scalar_spin_channel_matches_direct_numeric_table():
    mp.mp.dps = 50
    h_ext = mp.mpf("0.15")
    channel = build_numeric_virasoro_scalar_spin_channel(
        c=mp.mpf("50"),
        q_order=10,
        ell=2,
        delta_min=mp.mpf("2.0"),
        h_ext=h_ext,
        lam=3,
        n_sample_points=17,
    )

    from virasoro_blocks.derivs import G_zzb_table

    for Delta in (mp.mpf("2.25"), mp.mpf("3.0"), mp.mpf("4.5")):
        got = channel.serialized.evaluate("zzb", Delta)
        want = G_zzb_table(
            Delta,
            2,
            [h_ext] * 4,
            3,
            3,
            c=mp.mpf("50"),
            q_order=10,
            z_shift=-2 * h_ext,
        )
        for m in range(2):
            for n in range(2):
                scale = max(abs(mp.mpc(want[m][n])), mp.mpf("1e-30"))
                rel = abs(mp.mpc(got[m][n]) - mp.mpc(want[m][n])) / scale
                assert rel < mp.mpf("5e-6")


def test_build_numeric_virasoro_scalar_spin_channel_supports_yyb_coordinate():
    mp.mp.dps = 50
    h_ext = mp.mpf("0.15")
    channel = build_numeric_virasoro_scalar_spin_channel(
        coordinate="yyb",
        c=mp.mpf("50"),
        q_order=10,
        ell=2,
        delta_min=mp.mpf("2.0"),
        h_ext=h_ext,
        lam=3,
        n_sample_points=17,
    )

    for Delta in (mp.mpf("2.25"), mp.mpf("3.0"), mp.mpf("4.5")):
        got = channel.serialized.evaluate("yyb", Delta)
        want = G_yyb_table(
            Delta,
            2,
            [h_ext] * 4,
            3,
            3,
            c=mp.mpf("50"),
            q_order=10,
            z_shift=-2 * h_ext,
        )
        for m in range(2):
            for n in range(2):
                scale = max(abs(mp.mpc(want[m][n])), mp.mpf("1e-30"))
                rel = abs(mp.mpc(got[m][n]) - mp.mpc(want[m][n])) / scale
                assert rel < mp.mpf("5e-6")


def test_build_numeric_virasoro_identical_scalar_control_problem_reuses_identity_vector():
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("200"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=3,
        spin_list=(0, 2),
        q_order=10,
        dps=60,
        npoles=4, num_threads=1,
    )
    virasoro_problem = build_identical_scalar_gap_problem(spec)
    numeric_problem = build_numeric_virasoro_identical_scalar_control_problem(
        spec,
        n_sample_points=17,
    )

    assert numeric_problem.resolved_method == "numeric_virasoro_control"
    assert numeric_problem.identity_vector == virasoro_problem.identity_vector
    assert [channel.ell for channel in numeric_problem.spin_channels] == [0, 2]
    assert len(numeric_problem.pmp_build.positivity_blocks) == 2
    assert len(numeric_problem.pmp_build.doc["normalization"]) == len(numeric_problem.F_pairs)


def test_build_numeric_virasoro_identical_scalar_control_problem_supports_yyb_coordinate():
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("200"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=3,
        coordinate="yyb",
        spin_list=(0, 2),
        q_order=10,
        dps=60,
        npoles=4, num_threads=1,
    )

    problem = build_numeric_virasoro_identical_scalar_control_problem(
        spec,
        n_sample_points=17,
    )

    assert problem.spec.coordinate == "yyb"
    assert len(problem.pmp_build.positivity_blocks) == 2
    assert problem.spin_channels[1].serialized.evaluate("yyb", mp.mpf("2.5"))


def test_build_identical_scalar_gap_problem_parallel_matches_serial():
    serial_spec = IdenticalScalarGapSpec(
        c=mp.mpf("2"),
        h_ext=mp.mpf("0.25"),
        delta_gap=mp.mpf("2.0"),
        lam=3,
        spin_list=(0, 2),
        q_order=6,
        dps=50,
        npoles=4,
        num_threads=1,
    )
    parallel_spec = IdenticalScalarGapSpec(
        c=serial_spec.c,
        h_ext=serial_spec.h_ext,
        delta_gap=serial_spec.delta_gap,
        lam=serial_spec.lam,
        spin_list=serial_spec.spin_list,
        q_order=serial_spec.q_order,
        dps=serial_spec.dps,
        npoles=serial_spec.npoles,
        num_threads=2,
    )

    serial_problem = build_identical_scalar_gap_problem(serial_spec)
    parallel_problem = build_identical_scalar_gap_problem(parallel_spec)

    assert serial_problem.resolved_method == parallel_problem.resolved_method
    assert serial_problem.identity_vector == parallel_problem.identity_vector
    assert [channel.ell for channel in serial_problem.spin_channels] == [channel.ell for channel in parallel_problem.spin_channels]
    assert [channel.delta_min for channel in serial_problem.spin_channels] == [channel.delta_min for channel in parallel_problem.spin_channels]
    assert serial_problem.pmp_build.doc == parallel_problem.pmp_build.doc


def test_build_identical_scalar_gap_problem_reuses_gap_independent_work():
    base_spec = IdenticalScalarGapSpec(
        c=mp.mpf("2"),
        h_ext=mp.mpf("0.25"),
        delta_gap=mp.mpf("2.0"),
        lam=2,
        spin_list=(0, 2, 4),
        q_order=4,
        dps=40,
        npoles=3,
        num_threads=1,
    )
    wider_gap_spec = IdenticalScalarGapSpec(
        c=base_spec.c,
        h_ext=base_spec.h_ext,
        delta_gap=mp.mpf("3.0"),
        lam=base_spec.lam,
        spin_list=base_spec.spin_list,
        q_order=base_spec.q_order,
        dps=base_spec.dps,
        npoles=base_spec.npoles,
        num_threads=1,
    )
    spin_gap_spec = IdenticalScalarGapSpec(
        c=base_spec.c,
        h_ext=base_spec.h_ext,
        delta_gap=base_spec.delta_gap,
        delta_min_by_spin={4: mp.mpf("6.0")},
        lam=base_spec.lam,
        spin_list=base_spec.spin_list,
        q_order=base_spec.q_order,
        dps=base_spec.dps,
        npoles=base_spec.npoles,
        num_threads=1,
    )

    prepared = prepare_identical_scalar_gap_build(base_spec)
    direct_problem = build_identical_scalar_gap_problem(base_spec)
    reused_problem = build_identical_scalar_gap_problem(base_spec, prepared=prepared)
    direct_wider = build_identical_scalar_gap_problem(wider_gap_spec)
    reused_wider = build_identical_scalar_gap_problem(wider_gap_spec, prepared=prepared)
    direct_spin_gap = build_identical_scalar_gap_problem(spin_gap_spec)
    reused_spin_gap = build_identical_scalar_gap_problem(spin_gap_spec, prepared=prepared)

    assert reused_problem.pmp_build.doc == direct_problem.pmp_build.doc
    assert reused_wider.pmp_build.doc == direct_wider.pmp_build.doc
    assert reused_spin_gap.pmp_build.doc == direct_spin_gap.pmp_build.doc
    assert reused_wider.spin_channels[0].delta_min == mp.mpf("3.0")
    assert reused_problem.identity_vector == reused_wider.identity_vector
    assert reused_wider.spin_channels[1].serialized is prepared.prepared_spin_channels[1].serialized
    assert reused_spin_gap.spin_channels[1].prefactored is prepared.prepared_spin_channels[1].prefactored
    assert reused_spin_gap.spin_channels[2].delta_min == mp.mpf("6.0")


def test_write_identical_scalar_gap_problem(tmp_path):
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=2,
        spin_list=(0, 2),
        q_order=6,
        dps=50,
        npoles=3, num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    manifest_path, pmp_path = write_identical_scalar_gap_problem(problem, tmp_path)

    assert manifest_path == Path(tmp_path) / "manifest.json"
    assert pmp_path == Path(tmp_path) / "pmp.json"
    assert manifest_path.exists()
    assert pmp_path.exists()
    # The unified format no longer emits the legacy problem.json summary; the
    # manifest plus its blocks/ directory are the single canonical artifact.
    assert not (Path(tmp_path) / "problem.json").exists()
    assert (Path(tmp_path) / "blocks").is_dir()


def test_classify_sdpb_log_text():
    assert classify_sdpb_log_text("foo\n-----found dual feasible solution-----\n") == "dual_feasible"
    assert classify_sdpb_log_text("foo\n-----found primal feasible solution-----\n") == "primal_feasible"
    assert classify_sdpb_log_text("foo\n-----dual feasible jump detected-----\n") == "dual_feasible"
    assert classify_sdpb_log_text("foo\n-----primal feasible jump detected-----\n") == "primal_feasible"
    assert classify_sdpb_log_text("foo\n-----maxComplementarity exceeded-----\n") == "max_complementarity"
    assert classify_sdpb_log_text("foo\n-----found primal-dual optimal solution-----\n") == "optimal"
    assert classify_sdpb_log_text("foo\n-----maxIterations exceeded-----\n") == "max_iterations"
    assert classify_sdpb_log_text("foo\n-----maxRuntime exceeded-----\n") == "max_runtime"
    assert classify_sdpb_log_text("-----primal step too small-----\n") == "primal_step_too_small"
    assert classify_sdpb_log_text("-----dual step too small-----\n") == "dual_step_too_small"
    assert classify_sdpb_log_text("foo\nbar\n") is None


def test_certify_dual_from_sdpb_output_uses_problem_dps(monkeypatch, tmp_path):
    sdpb_out = tmp_path / "sdpb_out"
    sdpb_out.mkdir()
    (sdpb_out / "y.txt").write_text("1 1\n0\n")

    observed = {}

    def fake_certify_dual_functional(y, spin_channels, F_pairs, **kwargs):
        observed["dps"] = mp.mp.dps
        observed["y"] = list(y)
        observed["F_pairs"] = list(F_pairs)
        observed["coordinate"] = kwargs.get("coordinate")
        return True, []

    monkeypatch.setattr(
        certification_module,
        "certify_dual_functional",
        fake_certify_dual_functional,
    )

    problem = SimpleNamespace(
        spec=SimpleNamespace(dps=80, coordinate="yyb"),
        F_pairs=((0, 0),),
        spin_channels=(),
    )

    mp.mp.dps = 15
    certified, details = certification_module.certify_dual_from_sdpb_output(
        problem,
        sdpb_out,
        warn=False,
    )

    assert certified is True
    assert details == []
    assert observed["y"] == [mp.mpf(0)]
    assert observed["F_pairs"] == [(0, 0)]
    assert observed["coordinate"] == "yyb"
    assert observed["dps"] >= 80
    assert mp.mp.dps == 15


def test_certify_dual_from_sdpb_output_checks_full_pmp_vector(tmp_path):
    sdp_dir = tmp_path / "sdp"
    sdpb_out = tmp_path / "out"
    sdp_dir.mkdir()
    sdpb_out.mkdir()
    (sdp_dir / "normalization.json").write_text(
        json.dumps({"normalization": ["0", "1", "0"]})
    )
    # SDPB omits the normalized component, so reconstruction gives [1, 1, -3].
    (sdpb_out / "y.txt").write_text("2 1\n1\n-3\n")

    pmp_doc = {
        "PositiveMatrixWithPrefactorArray": [
            {
                "DampedRational": {"base": "0.5", "constant": "1", "poles": []},
                "polynomials": [[[
                    ["0"],
                    ["0"],
                    [
                        "0",
                        "0.333333333333333333333333333333",
                        "-0.333333333333333333333333333333",
                    ],
                ]]],
                "samplePoints": ["0", "1"],
                "sampleScalings": ["1", "1"],
            }
        ]
    }
    problem = SimpleNamespace(
        spec=SimpleNamespace(dps=50, coordinate="zzb"),
        F_pairs=((0, 0),),
        spin_channels=(SimpleNamespace(ell=0),),
        pmp_build=SimpleNamespace(doc=pmp_doc),
    )

    certified, details = certification_module.certify_dual_from_sdpb_output(
        problem,
        sdpb_out,
        warn=False,
    )

    assert certified is False
    assert details[0]["basis"] == "pmp_polynomial"
    assert details[0]["block_index"] == 0
    assert details[0]["n_negative"] == 1
    assert details[0]["min_x"] == pytest.approx(0.5)


def test_normalized_moment_direct_sdp_certification_uses_sdp_data(tmp_path):
    sdp_dir = tmp_path / "sdp"
    sdpb_out = tmp_path / "out"
    sdp_dir.mkdir()
    sdpb_out.mkdir()
    (sdp_dir / "control.json").write_text(json.dumps({"num_blocks": 1}))
    (sdp_dir / "normalization.json").write_text(
        json.dumps({"normalization": ["1", "0"]})
    )
    (sdp_dir / "block_data_0.json").write_text(
        json.dumps({"c": ["4"], "B": [["3"]]})
    )
    (sdpb_out / "y.txt").write_text("1 1\n1\n")

    pmp_doc = {
        "PositiveMatrixWithPrefactorArray": [
            {
                "DampedRational": {"base": "0.5", "constant": "1", "poles": []},
                # Reconstructed full functional is [1, 1], so this PMP
                # polynomial is NEGATIVE (q(x) = -x). The per-sample basis must
                # still come from block_data_0's c - B.y, which is +1 -- that is
                # what this test exists to pin down.
                #
                # The fixture is deliberately INCONSISTENT: no real solve has a
                # PMP polynomial that disagrees in sign with the sample values
                # written from it (sdp_writer derives the sdp dir from this same
                # doc, and sample_values_by_block is the same constraint
                # evaluated cleanly). Since 2026-08-03 the exact root-isolation
                # check actually runs -- it was silently dying on a length guard
                # before -- so it now sees this contrived polynomial and
                # correctly refuses to certify. Hence `certified is False` with
                # the sample-basis assertions unchanged: precedence for the
                # SAMPLE scan still belongs to block_data, and the whole-ray
                # check independently gets a veto.
                "polynomials": [[[["0"], ["-1"]]]],
                "samplePoints": ["0"],
                "sampleScalings": ["1"],
            }
        ]
    }
    problem = SimpleNamespace(
        spec=SimpleNamespace(dps=50, coordinate="zzb", spin_list=(0,)),
        F_pairs=(),
        spin_channels=(SimpleNamespace(ell=0),),
        pmp_build=SimpleNamespace(doc=pmp_doc),
        pmp_metadata={"moment": {"kind": "normalized"}},
    )

    certified, details = certification_module.certify_dual_from_sdpb_output(
        problem,
        sdpb_out,
        warn=False,
    )

    assert details[0]["basis"] == "direct_sdp_samples"
    assert details[0]["min_value"] == pytest.approx(1.0)
    # The sample scan passed on block_data; the exact check vetoed on the
    # (contrived) PMP polynomial. Both facts are asserted so that a regression
    # in either one is distinguishable from a regression in the other.
    assert details[0]["exact_positivity"]["passed"] is False
    assert certified is False


def _make_linear_cert_channel(ell, delta_min, a, b):
    """Build a real (picklable) certification channel whose dual-functional
    value along the test grid is the linear function ``a + b * Delta``.

    Uses a genuine :class:`SerializedDerivTable` (``base=1``, no poles, a single
    ``(0, 0)`` linear numerator) so the parallel certification path exercises
    actual cross-process pickling of the channel/block objects.
    """
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_bootstrap.identical_scalar import IdenticalScalarSpinChannel

    table = SerializedDerivTable(
        shift=mp.mpf(0),
        base=mp.mpf(1),
        pole_list_x=(),
        coordinates=("zzb",),
        numerators={"zzb": [[Polynomial([mp.mpf(a), mp.mpf(b)])]]},
        sample_points_x=None,
    )
    return IdenticalScalarSpinChannel(
        ell=ell, delta_min=mp.mpf(delta_min), serialized=None, prefactored=table,
    )


def test_certify_dual_functional_parallel_matches_serial():
    # A mix of channels: some stay positive over the grid, some dip negative,
    # so the verdict, per-channel ordering, and min values all matter.
    channels = [
        _make_linear_cert_channel(ell=0, delta_min=0, a="1.0", b="0.5"),
        _make_linear_cert_channel(ell=2, delta_min=2, a="3.0", b="-0.2"),
        _make_linear_cert_channel(ell=4, delta_min=4, a="0.5", b="0.1"),
        _make_linear_cert_channel(ell=6, delta_min=6, a="2.0", b="-0.05"),
        _make_linear_cert_channel(ell=8, delta_min=8, a="10.0", b="1.0"),
    ]
    y = [mp.mpf(1)]
    F_pairs = [(0, 0)]

    serial_ok, serial_details = certification_module.certify_dual_functional(
        y, channels, F_pairs, coordinate="zzb",
        n_test_points=40, delta_range=20.0, num_threads=1,
    )
    parallel_ok, parallel_details = certification_module.certify_dual_functional(
        y, channels, F_pairs, coordinate="zzb",
        n_test_points=40, delta_range=20.0, num_threads=4,
    )

    assert parallel_ok == serial_ok
    assert [d["ell"] for d in parallel_details] == [d["ell"] for d in serial_details]
    for sd, pd in zip(serial_details, parallel_details):
        assert pd["ell"] == sd["ell"]
        assert pd["passed"] == sd["passed"]
        assert pd["n_negative"] == sd["n_negative"]
        assert pd["min_value"] == pytest.approx(sd["min_value"], rel=1e-30, abs=1e-30)
        assert pd["min_delta"] == pytest.approx(sd["min_delta"], rel=1e-30, abs=1e-30)


def _max_rel_table(a, b):
    worst = mp.mpf(0)
    for i in range(len(a)):
        for j in range(len(a[i])):
            num = abs(a[i][j] - b[i][j])
            den = abs(a[i][j]) + abs(b[i][j])
            if den > 0:
                worst = max(worst, num / den)
    return worst


def test_large_spin_pole_selection_defaults_to_the_configured_tower_selector():
    from virasoro_bootstrap.large_spin import double_twist_set

    mp.mp.dps = 60
    h_ext = mp.mpf("0.0625")
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("32"), h_ext=h_ext, delta_gap=mp.mpf("1.0"),
        lam=3, q_order=16, dps=60, npoles=6,
        spin_list=(0, 2),
        large_spin_lmin=4, large_spin_twists=double_twist_set(h_ext, 1), num_threads=1,
    )
    # The tower is its OWN knob with its own default, asserted against the
    # config rather than a literal so the two cannot drift apart. Since
    # 2026-07-30 it happens to carry the same VALUE as the fixed-spin selector
    # (they were measured bit-identical on the tower, so one name is simpler to
    # explain), but it stays independently settable -- the tower's inverted
    # geometry is its own conditioning regime. The next assertion checks the
    # knob is real, not that the values differ.
    assert spec.large_spin_pole_selection == DEFAULT_LARGE_SPIN_POLE_SELECTION

    # It is an independent knob: overriding the Delta pole_selection does not
    # change the tower selection (and vice versa).
    spec_residue_delta = dataclasses.replace(spec, pole_selection="residue")
    assert (
        spec_residue_delta.large_spin_pole_selection
        == DEFAULT_LARGE_SPIN_POLE_SELECTION
    )
    spec_residue_tower = dataclasses.replace(
        spec, large_spin_pole_selection="residue"
    )
    assert spec_residue_tower.pole_selection == spec.pole_selection
    assert spec_residue_tower.large_spin_pole_selection == "residue"


def test_exact_block_certification_channels_catch_tower_interp_error():
    from virasoro_bootstrap.certification import _build_exact_block_channels
    from virasoro_bootstrap.large_spin import double_twist_set

    mp.mp.dps = 70
    h_ext = mp.mpf("0.0625")
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("32"), h_ext=h_ext, delta_gap=mp.mpf("1.0"),
        lam=3, q_order=16, dps=70, npoles=6,
        spin_list=(0, 2),
        large_spin_lmin=4, large_spin_twists=double_twist_set(h_ext, 1), num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)
    prod = list(problem.spin_channels)
    exact = _build_exact_block_channels(problem)
    assert len(exact) == len(prod)

    coord = spec.coordinate or "zzb"
    n_tower = len(spec.large_spin_twists)
    n_fixed = len(prod) - n_tower

    # Fixed-spin: the numeric-sampler exact block matches the SDP interpolant
    # closely at delta_min (the npoles interpolation is good in the Delta channels).
    fc_p, fc_e = prod[0], exact[0]
    D = mp.mpf(fc_p.delta_min)
    assert _max_rel_table(
        fc_p.prefactored.evaluate(coord, D),
        fc_e.prefactored.evaluate(coord, D),
    ) < mp.mpf("1e-10")

    # Tower: at the interpolation node L = Lmax the exact channel reproduces the
    # production library-normalized block (self-dual base 16 q(1/2)/N) to working
    # precision...
    tc_p, tc_e = prod[-1], exact[-1]
    Lmax = mp.mpf(tc_p.delta_min)
    rel_node = _max_rel_table(
        tc_p.prefactored.evaluate(coord, Lmax),
        tc_e.prefactored.evaluate(coord, Lmax),
    )
    assert rel_node < mp.mpf("1e-12")

    # ...but far from the node the interpolant deviates from the true block, which
    # is exactly the interpolation error exact-block certification now catches.
    Lfar = Lmax + 80
    rel_far = _max_rel_table(
        tc_p.prefactored.evaluate(coord, Lfar),
        tc_e.prefactored.evaluate(coord, Lfar),
    )
    assert rel_far > rel_node


def test_large_spin_blocks_use_clean_sdp_v2_samples():
    from virasoro_bootstrap.large_spin import double_twist_set

    def evalpoly(coeffs, x):
        value = mp.mpf(0)
        for coeff in reversed(coeffs):
            value = value * x + coeff
        return value

    mp.mp.dps = 80
    h_ext = mp.mpf("0.0625")
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("32"), h_ext=h_ext, delta_gap=mp.mpf("1.0"),
        lam=3, q_order=16, dps=80, npoles=6,
        spin_list=(0, 2), large_spin_lmin=4,
        large_spin_twists=tuple(double_twist_set(h_ext, 1)), num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    assert len(problem.sample_values_by_block) == len(problem.spin_channels)

    tower_index = len(problem.spin_channels) - 1
    tower_block = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"][tower_index]
    clean = problem.sample_values_by_block[tower_index]
    xs = [mp.mpf(s) for s in tower_block["samplePoints"]]
    polys = tower_block["polynomials"][0][0]

    worst = mp.mpf(0)
    for k, x in enumerate(xs):
        for a, coeffs in enumerate(polys):
            truth = evalpoly([mp.mpf(c) for c in coeffs], x)
            worst = max(worst, abs(clean[k][a] - truth) / max(abs(truth), mp.mpf(1)))
    assert worst < mp.mpf("1e-60")


def test_large_spin_symbolic_falls_back_to_v1_writer_path():
    from virasoro_bootstrap.large_spin import double_twist_set

    mp.mp.dps = 80
    h_ext = mp.mpf("0.0625")
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("32"), h_ext=h_ext, delta_gap=mp.mpf("1.0"),
        lam=3, q_order=16, dps=80, npoles=6,
        spin_list=(0, 2), sample_via="symbolic", large_spin_lmin=4,
        large_spin_twists=tuple(double_twist_set(h_ext, 1)), num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    tower_index = len(problem.spin_channels) - 1
    tower_channel = problem.spin_channels[tower_index]

    assert tower_channel.serialized.sample_via == "symbolic"
    assert tower_channel.serialized.sample_block_values is None
    assert tower_index not in problem.sample_values_by_block


def test_extra_twist_channels_full_gap_sweep():
    from virasoro_bootstrap.certification import _build_extra_twist_channels
    from virasoro_bootstrap.large_spin import double_twist_set

    mp.mp.dps = 60
    h_ext = mp.mpf("0.0625")
    base_twists = double_twist_set(h_ext, 3)  # tau_n spaced by 2.0
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("32"), h_ext=h_ext, delta_gap=mp.mpf("1.0"),
        lam=3, q_order=16, dps=60, npoles=6,
        spin_list=(0, 2),
        large_spin_lmin=4, large_spin_twists=base_twists, num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    # extra_twists=0 -> no extra channels.
    assert _build_extra_twist_channels(problem, 0) == []

    extra = _build_extra_twist_channels(problem, 3)
    # 2 interior gaps + 1 above-top gap, 3 samples each = 9 channels.
    assert len(extra) == 9

    base_sorted = sorted(mp.mpf(t) for t in base_twists)
    # All channels share the tower lower edge L = Lmax and the exact evaluator.
    for ch in extra:
        assert mp.mpf(ch.delta_min) == mp.mpf(spec.large_spin_lmin)
        assert ch.ell == int(spec.large_spin_lmin)
        # Full-gap: samples land strictly between base twists (or just above top),
        # never on a base twist.
        L = mp.mpf("6")
        table = ch.prefactored.evaluate(spec.coordinate or "zzb", L)
        assert len(table) == spec.lam + 1

    # Interior samples for the first gap [tau0, tau1] at the 1/4, 1/2, 3/4 marks.
    lo, hi = base_sorted[0], base_sorted[1]
    step = (hi - lo) / 4
    expected_first_gap = [lo + step, lo + 2 * step, lo + 3 * step]
    # Reconstruct the twist each channel samples by comparing its block to a
    # direct sampler at the expected twist. The certification channel feeds the
    # library-normalized block (self-dual base 16 q(1/2)/N) straight through, so
    # it matches the sampler with no extra N^L rescaling.
    from virasoro_blocks.large_spin import _LargeSpinTwistSampler
    z_shift = -2 * h_ext
    for k, tw in enumerate(expected_first_gap):
        sampler = _LargeSpinTwistSampler(
            "virasoro", spec.coordinate or "zzb", [h_ext] * 4,
            spec.chiral_norm, z_shift, spec.lam, spec.lam, tw,
            c=mp.mpf(spec.c), q_order=spec.q_order,
        )
        L = mp.mpf("9")
        expected = [[sampler(L)[m][n] for n in range(spec.lam + 1)]
                    for m in range(spec.lam + 1)]
        got = extra[k].prefactored.evaluate(spec.coordinate or "zzb", L)
        assert _max_rel_table(got, expected) < mp.mpf("1e-40")


def test_extra_spin_channels_use_numeric_sampling_path():
    from virasoro_bootstrap.certification import _build_extra_spin_channels

    mp.mp.dps = 60
    h_ext = mp.mpf("0.0625")
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("32"), h_ext=h_ext, delta_gap=mp.mpf("1.0"),
        lam=3, q_order=16, dps=60, npoles=6,
        spin_list=(0, 4), sample_via="numeric", num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    extra = _build_extra_spin_channels(problem, 1)
    assert [ch.ell for ch in extra] == [2, 6]
    for ch in extra:
        assert ch.serialized.sample_via == "numeric"
        assert ch.serialized.sample_block_values is not None


def test_chiral_control_defaults_to_contribution_optimal():
    from virasoro_bootstrap.controls.chiral_control import (
        build_chiral_control_problem,
    )

    mp.mp.dps = 60
    common = dict(
        mode="virasoro_symbolic",
        c=mp.mpf("32"),
        h_ext=mp.mpf("0.0625"),
        h_gap=mp.mpf("1.0"),
        lam=3,
        q_order=16,
        dps=60,
        npoles=6,
    )
    default_problem = build_chiral_control_problem(**common)
    explicit_problem = build_chiral_control_problem(
        **common, pole_selection="contribution_optimal"
    )

    assert default_problem.spec.pole_selection == "contribution_optimal"
    assert (
        default_problem.spin_channels[0].serialized.pole_list_x
        == explicit_problem.spin_channels[0].serialized.pole_list_x
    )


def test_run_identical_scalar_gap_problem_in_docker_forwards_precision(monkeypatch, tmp_path):
    # these tests pin the docker argv; a cluster profile's SDPB environment would change it
    for name in ("VB_SDPB_MODE", "VB_SDPB_IMAGE", "VB_SDPB_BIN_DIR", "VB_MPI_LAUNCHER"):
        monkeypatch.delenv(name, raising=False)
    from virasoro_bootstrap import identical_scalar as identical_scalar_module

    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=2,
        spin_list=(0, 2),
        q_order=6,
        dps=50,
        npoles=3, num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    commands = []

    def fake_run_command(command, log_path, *, check=True, cwd=None):
        commands.append(tuple(command))
        if log_path is None:
            return 0
        text = "Processed 2 SDP blocks in 0.001 seconds, output: sdp\n"
        if "sdpb" in command:
            text = "-----found dual feasible solution-----\n"
        log_path.write_text(text)
        return 0

    monkeypatch.setattr(identical_scalar_module.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(identical_scalar_module, "_run_command", fake_run_command)

    run = run_identical_scalar_gap_problem_in_docker(
        problem,
        tmp_path,
        backend="pmp2sdp",
        sdpb_options={"--precision": 512},
        run_sdpb=True,
        pmp2sdp_log_name="pmp2sdp.log",
        sdpb_log_name="sdpb.log",
    )
    assert run.precision_bits == 512
    assert run.pmp2sdp_command == (
        "/usr/bin/docker",
        "run",
        "--rm",
        "-v",
        ANY,
        "-w",
        "/work",
        "bootstrapcollaboration/sdpb:master",
        "pmp2sdp",
        "-i",
        "pmp.json",
        "-o",
        "sdp",
        "-p",
        "512",
    )
    assert run.sdpb_command is not None
    assert "--precision" in run.sdpb_command
    assert run.sdpb_command[run.sdpb_command.index("--precision") + 1] == "512"
    assert run.sdpb_status == "dual_feasible"


def test_run_identical_scalar_gap_problem_in_docker_supports_jump_detection(monkeypatch, tmp_path):
    from virasoro_bootstrap import identical_scalar as identical_scalar_module

    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=2,
        spin_list=(0, 2),
        q_order=6,
        dps=50,
        npoles=3, num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    def fake_run_command(command, log_path, *, check=True, cwd=None):
        if log_path is None:
            return 0
        text = "Processed 2 SDP blocks in 0.001 seconds, output: sdp\n"
        if "sdpb" in command:
            text = "-----dual feasible jump detected-----\n"
        log_path.write_text(text)
        return 0

    monkeypatch.setattr(identical_scalar_module.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(identical_scalar_module, "_run_command", fake_run_command)

    run = run_identical_scalar_gap_problem_in_docker(
        problem,
        tmp_path,
        run_sdpb=True,
        sdpb_options={
            "--precision": 512,
            "--detectPrimalFeasibleJump": True,
            "--detectDualFeasibleJump": True,
        },
        pmp2sdp_log_name="pmp2sdp.log",
        sdpb_log_name="sdpb.log",
    )

    assert run.sdpb_command is not None
    assert "--detectPrimalFeasibleJump" in run.sdpb_command
    assert "--detectDualFeasibleJump" in run.sdpb_command
    assert "--findPrimalFeasible" not in run.sdpb_command
    assert "--findDualFeasible" not in run.sdpb_command
    assert run.sdpb_status == "dual_feasible"


@pytest.mark.parametrize("checkpoint_name", ["prior.ck", "out/initial.ck"])
def test_run_identical_scalar_gap_problem_in_docker_hotstart_wiring(monkeypatch, tmp_path, checkpoint_name):
    """initial_checkpoint_dir stages the prior checkpoint under the mount, passes
    --initialCheckpointDir, and the written final checkpoint is returned for
    chaining (the hotstart warm-start plumbing)."""
    # these tests pin the docker argv; a cluster profile's SDPB environment would change it
    for name in ("VB_SDPB_MODE", "VB_SDPB_IMAGE", "VB_SDPB_BIN_DIR", "VB_MPI_LAUNCHER"):
        monkeypatch.delenv(name, raising=False)
    from virasoro_bootstrap import identical_scalar as identical_scalar_module

    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"), h_ext=mp.mpf("0.15"), delta_gap=mp.mpf("1.0"),
        lam=2, spin_list=(0, 2), q_order=6, dps=50,
        npoles=3, num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    prior_ck = tmp_path / checkpoint_name
    prior_ck.mkdir(parents=True)
    (prior_ck / "checkpoint.0").write_text("prior-iterate")

    def fake_run_command(command, log_path, *, check=True, cwd=None):
        if "sdpb" in command:
            # Simulate SDPB writing a final checkpoint into the mounted dir.
            host = command[command.index("-v") + 1].split(":")[0]
            ck = Path(host) / "sdp.ck"
            ck.mkdir(parents=True, exist_ok=True)
            (ck / "checkpoint.0").write_text("new-iterate")
            if log_path is not None:
                log_path.write_text("-----found primal feasible solution-----\n")
        elif log_path is not None:
            log_path.write_text("Processed 2 SDP blocks\n")
        return 0

    monkeypatch.setattr(identical_scalar_module.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(identical_scalar_module, "_run_command", fake_run_command)

    run = run_identical_scalar_gap_problem_in_docker(
        problem, tmp_path / "out", run_sdpb=True,
        sdpb_options={"--precision": 512, "--findPrimalFeasible": True},
        sdpb_log_name="sdpb.log",
        initial_checkpoint_dir=prior_ck,
    )

    # The warm-start flag points at the staged copy (reachable inside the container).
    assert "--initialCheckpointDir" in run.sdpb_command
    idx = run.sdpb_command.index("--initialCheckpointDir")
    assert run.sdpb_command[idx + 1] == "initial.ck"
    assert (run.output_dir / "initial.ck" / "checkpoint.0").read_text() == "prior-iterate"
    # The final checkpoint is exposed so a binary search can chain solves.
    assert run.checkpoint_dir == run.output_dir / "sdp.ck"
    assert (run.checkpoint_dir / "checkpoint.0").read_text() == "new-iterate"


def test_run_identical_scalar_gap_problem_in_docker_no_hotstart_no_flag(monkeypatch, tmp_path):
    """Without initial_checkpoint_dir there is no --initialCheckpointDir flag, and
    --noFinalCheckpoint leaves checkpoint_dir None."""
    from virasoro_bootstrap import identical_scalar as identical_scalar_module

    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"), h_ext=mp.mpf("0.15"), delta_gap=mp.mpf("1.0"),
        lam=2, spin_list=(0, 2), q_order=6, dps=50,
        npoles=3, num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    def fake_run_command(command, log_path, *, check=True, cwd=None):
        if log_path is not None:
            log_path.write_text(
                "-----found primal feasible solution-----\n"
                if "sdpb" in command else "Processed 2 SDP blocks\n"
            )
        return 0

    monkeypatch.setattr(identical_scalar_module.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(identical_scalar_module, "_run_command", fake_run_command)

    run = run_identical_scalar_gap_problem_in_docker(
        problem, tmp_path / "out", run_sdpb=True,
        sdpb_options={"--precision": 512, "--noFinalCheckpoint": True},
        sdpb_log_name="sdpb.log",
    )
    assert "--initialCheckpointDir" not in run.sdpb_command
    assert run.checkpoint_dir is None


def test_run_identical_scalar_gap_problem_sdp_backend_writes_sdp_directory(monkeypatch, tmp_path):
    """The default ``backend="sdp"`` writes the SDP dir in-process (no pmp2sdp)."""
    from virasoro_bootstrap import identical_scalar as identical_scalar_module

    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"), h_ext=mp.mpf("0.15"), delta_gap=mp.mpf("1.0"),
        lam=2, spin_list=(0, 2), q_order=6, dps=50,
        npoles=3, num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    def fake_run_command(command, log_path, *, check=True, cwd=None):
        # Only sdpb should reach Docker; pmp2sdp must NOT be invoked.
        assert "pmp2sdp" not in command
        if log_path is not None:
            log_path.write_text("-----found dual feasible solution-----\n")
        return 0

    monkeypatch.setattr(identical_scalar_module.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(identical_scalar_module, "_run_command", fake_run_command)

    run = run_identical_scalar_gap_problem_in_docker(
        problem, tmp_path, sdpb_options={"--precision": 1536},
        pmp2sdp_log_name="conv.log", sdpb_log_name="sdpb.log",
    )

    # sdp backend taken: no pmp2sdp Docker command, sdp dir populated by the writer.
    assert run.pmp2sdp_command == ()
    assert run.pmp2sdp_summary is not None and "sdp_writer" in run.pmp2sdp_summary
    # the SDP data are written at the working precision spec.dps, whatever SDPB's --precision
    assert "eval_dps=50" in run.pmp2sdp_summary
    assert "output_digits=50" in run.pmp2sdp_summary
    assert (run.sdp_dir / "block_data_0.json").exists()
    assert (run.sdp_dir / "objectives.json").exists()
    assert run.sdpb_status == "dual_feasible"
    assert run.sdpb_summary == "-----found dual feasible solution-----"


def test_read_sdpb_objective_does_not_re_add_direct_sdp_constant(tmp_path):
    """SDPB already includes ``objectives.json``'s constant -- we must not add it.

    This test previously asserted the opposite (that the reader adds the
    offset), which is what the code did from 06422771 until 2026-08-14. It was
    wrong, and settled by experiment rather than argument: one ``sdp/``
    directory was solved twice, identical but for ``constant`` zeroed in one
    copy, and ``dualObjective`` shifted by EXACTLY the constant
    (-1.80300314873 vs +0.46220176721 against a constant of -2.26520491593).

    It went unnoticed because the normalized-moment problem puts the pivot on
    the s-slot with ``obj[piv] = 0``, so every validated moment run has
    ``constant = 0.0`` and the double-count was a no-op there. It only distorted
    OPE-max, where ``normalization = V_*`` makes the constant nonzero and
    jumpy -- see :func:`read_sdpb_objective`'s docstring.

    Independently confirmed by physics: with the double-count removed, the
    OPE-max bound at c=6/5, Delta_ext=1/8, delta_star=1 reproduces the exact
    Ising x TIM coefficient lambda^2 = 4.0 to seven significant figures.
    """
    sdp_dir = tmp_path / "sdp"
    out_dir = tmp_path / "sdpb_out"
    sdp_dir.mkdir()
    out_dir.mkdir()
    (sdp_dir / "objectives.json").write_text(
        json.dumps({"constant": "2.5", "b": ["0"]})
    )
    (out_dir / "out.txt").write_text(
        "primalObjective = 1.25;\n"
        "dualObjective = -0.75;\n"
    )

    objective = read_sdpb_objective(out_dir)

    # Reported verbatim: the constant is NOT re-added.
    assert objective == {"primal": mp.mpf("1.25"), "dual": mp.mpf("-0.75")}


def test_run_identical_scalar_gap_problem_sdp_backend_falls_back_without_sample_points(
    monkeypatch, tmp_path
):
    """``backend="sdp"`` falls back to pmp2sdp when the PMP has no sample points.

    Both methods now carry interpolation nodes (``full`` == keep-all
    interpolation), so the only no-sample-points PMP is
    ``explicit_sample_points=False`` (pmp2sdp derives its own nodes)."""
    from virasoro_bootstrap import identical_scalar as identical_scalar_module

    spec = IdenticalScalarGapSpec(
        c=mp.mpf("10"), h_ext=mp.mpf("0.15"), delta_gap=mp.mpf("1.0"),
        lam=2, spin_list=(0, 2), q_order=6, dps=50,        npoles=3, explicit_sample_points=False, num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)

    def fake_run_command(command, log_path, *, check=True, cwd=None):
        if log_path is None:
            return 0
        text = "Processed 2 SDP blocks in 0.001 seconds, output: sdp\n"
        if "sdpb" in command:
            text = "-----found dual feasible solution-----\n"
        log_path.write_text(text)
        return 0

    monkeypatch.setattr(identical_scalar_module.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(identical_scalar_module, "_run_command", fake_run_command)

    run = run_identical_scalar_gap_problem_in_docker(
        problem, tmp_path, backend="sdp", sdpb_options={"--precision": 256},
        pmp2sdp_log_name="pmp2sdp.log", sdpb_log_name="sdpb.log",
    )

    # Fallback engaged: the pmp2sdp Docker command was built and run.
    assert run.pmp2sdp_command and "pmp2sdp" in run.pmp2sdp_command
    assert run.pmp2sdp_summary is not None and "Processed" in run.pmp2sdp_summary


_DOCKER_PMP2SDP_SPECS = {
    # (c, h_ext, delta_gap) as strings, parsed at the spec's dps
    # c > 1: complex Kac poles, dropped before emission
    "c=10": (("10", "0.15", "1.0"), dict(q_order=6, dps=50, npoles=3)),
    # c < 1: real positive poles, lowered before emission
    "c=0.5": (("0.5", "0.0625", "0.2"), dict(q_order=8, dps=60, npoles=4, positive_pole_selection="residue")),
}


@pytest.mark.skipif(not _docker_sdpb_available(), reason="SDPB Docker image unavailable")
@pytest.mark.parametrize("which", sorted(_DOCKER_PMP2SDP_SPECS))
def test_run_identical_scalar_gap_problem_in_docker_c_lt_1(tmp_path, which):
    """pmp2sdp in Docker converts the emitted PMP into an SDP directory, at
    c < 1 and (the former ``..._c_gt_1`` twin) at c > 1."""
    (c, h_ext, gap), options = _DOCKER_PMP2SDP_SPECS[which]
    mp.mp.dps = options["dps"]
    spec = IdenticalScalarGapSpec(
        c=mp.mpf(c), h_ext=mp.mpf(h_ext), delta_gap=mp.mpf(gap), lam=2, spin_list=(0, 2),
        num_threads=1, **options,
    )
    problem = build_identical_scalar_gap_problem(spec)

    run = run_identical_scalar_gap_problem_in_docker(
        problem,
        tmp_path,
        backend="pmp2sdp",
        sdpb_options={"--precision": 256},
        run_sdpb=False,
        pmp2sdp_log_name="pmp2sdp.log",
    )

    assert run.manifest_path.exists()
    assert run.pmp_path.exists()
    assert run.sdp_dir.exists()
    assert any(run.sdp_dir.iterdir())
    assert run.sdpb_command is None
    assert run.pmp2sdp_log_path == Path(tmp_path) / "pmp2sdp.log"
    assert run.pmp2sdp_log_path.exists()
    assert run.pmp2sdp_summary is not None and "Processed" in run.pmp2sdp_summary
    assert run.sdpb_log_path is None
    assert run.sdpb_status is None


def test_shared_sample_points_threads_through_spec():
    """`shared_sample_points` on the spec reaches interpolation: both modes emit
    one shared sample set with uniformly-padded polynomials (what pmp2sdp sees),
    but the interpolated coefficients differ (per-entry zero-pad vs shared fit)."""
    mp.mp.dps = 80
    common = dict(
        c=mp.mpf("2"), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf("1.0"),
        lam=5, spin_list=(0, 2), q_order=20, dps=80,        npoles=8, pole_selection="nearest", num_threads=1,
    )
    per = build_identical_scalar_gap_problem(
        IdenticalScalarGapSpec(shared_sample_points=False, **common))
    shared = build_identical_scalar_gap_problem(
        IdenticalScalarGapSpec(shared_sample_points=True, **common))

    def pmp(prob):
        return prob.pmp_build.doc["PositiveMatrixWithPrefactorArray"][0]

    for prob in (per, shared):
        entry = pmp(prob)
        # one shared sample set, every polynomial padded to the same degree
        assert "samplePoints" in entry
        lengths = {len(v) for mat in entry["polynomials"] for row in mat for v in row}
        assert len(lengths) == 1
        assert len(entry["samplePoints"]) == lengths.pop()

    # the flag actually changes the interpolation (coefficients differ)
    assert pmp(per)["polynomials"] != pmp(shared)["polynomials"]
