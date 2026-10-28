"""Point constraints (degree-0 positivity blocks)."""
import mpmath as mp

from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec
from virasoro_bootstrap.controls.global_control import (
    build_global_identical_scalar_control_problem,
)
from virasoro_bootstrap.point_constraint import (
    append_positivity_channels, input_fixed_operators,
    normalization_with_fixed_operators, point_constraint_channel,
)
from virasoro_bootstrap.pmp_writer import _compute_bilinear_bases
from virasoro_blocks.global_blocks import global_evaluate_block
from virasoro_blocks.point import point_block_table
from virasoro_blocks.polyderivs import G_zzb_poly_table
from virasoro_blocks.regularization import merged_delta_table
import pytest


def test_point_block_at_singular_c_matches_regularized_table():
    """At a recursion-singular c (Ising c = 1/2) the point block is computed via
    exact Laurent regularization with merged Kac poles, as the fixed-spin
    tables do, and reproduces that block at a non-degenerate dimension."""
    mp.mp.dps = 60
    ell, q_order, lam = 0, 16, 2
    c, Dp = mp.mpf("0.5"), mp.mpf("1.3")
    hext = [mp.mpf("0.0625")] * 4
    st = point_block_table(["zzb"], basis="scalar_G", ell=ell, point_value=Dp,
                           h_externals=hext, hbar_externals=hext, lam=lam,
                           c=c, q_order=q_order)
    reg = merged_delta_table(lambda cv: G_zzb_poly_table(ell, hext, lam, lam, c=cv, q_order=q_order), c=c).evaluate(Dp)
    # Same convention as the bootstrap fixed-spin channels (which use this table).
    assert abs(st.numerators["zzb"][0][0].coeffs[0] - reg[0][0]) < mp.mpf(10) ** -40


def test_point_block_genuine_pole_raises_but_physical_externals_compute():
    """Exchanged Delta = 1 at Ising c = 1/2 is the degenerate energy dimension
    (h = 1/2). For unphysical external dimensions the block has a genuine pole
    there (ill-defined point constraint -> error); for the physical sigma x sigma
    externals (Delta_ext = 1/8) the pole is removable and the block is finite."""
    mp.mp.dps = 60
    ell, q_order, lam = 0, 16, 2
    c, Dp = mp.mpf("0.5"), mp.mpf("1.0")

    # Delta_ext = 0: genuine pole at the degenerate dimension -> clear error
    # (not silent ~1e20 garbage).
    with pytest.raises(ValueError, match="did not converge|genuine pole"):
        point_block_table(["zzb"], basis="scalar_G", ell=ell, point_value=Dp,
                          h_externals=[mp.mpf(0)] * 4, hbar_externals=[mp.mpf(0)] * 4,
                          lam=lam, c=c, q_order=q_order)

    # Delta_ext = 1/8 (Ising sigma): the exchange is physical, the pole is
    # removable, and the frozen block is finite AND correct -- it matches the
    # independent regularized composer.G_block (this is the value the buggy
    # large-root deflation used to corrupt by an O(1) factor).
    from virasoro_blocks.composer import G_block
    hsig = [mp.mpf("1/16")] * 4
    st = point_block_table(["zzb"], basis="scalar_G", ell=ell, point_value=Dp,
                           h_externals=hsig, hbar_externals=hsig, lam=lam,
                           c=c, q_order=q_order)
    v = st.numerators["zzb"][0][0].coeffs[0]
    half = mp.mpf(1) / 2
    g = G_block(Dp, ell, half, half, hsig, c=c, q_order=q_order, regularize="auto")
    assert abs(v - g) < mp.mpf(10) ** -40


def test_bilinear_bases_degree_zero():
    """A single sample point (degree-0 block) gives one row in bilinearBasis_0
    and an empty bilinearBasis_1 (no x*measure SOS term) -- the fix that lets a
    point constraint through the PMP writer."""
    mp.mp.dps = 40
    b0, b1 = _compute_bilinear_bases((mp.mpf("3.0"),), [mp.mpf("1.0")], 0, 20)
    assert len(b0) == 1
    assert b1 == []
    # degree 1 (two points) still has both bases non-empty
    b0b, b1b = _compute_bilinear_bases(
        (mp.mpf("3.0"), mp.mpf("4.0")), [mp.mpf("1.0"), mp.mpf("1.0")], 1, 20)
    assert len(b0b) == 1 and len(b1b) == 1


def _problem():
    mp.mp.dps = 60
    spec = IdenticalScalarGapSpec(
        c=mp.mpf(8), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf("1.5"),
        delta_min_by_spin={}, lam=3, spin_list=(0, 2), q_order=20, dps=60,
        npoles=6, pole_selection="residue", num_threads=1)
    return build_global_identical_scalar_control_problem(spec)


def test_point_channel_reproduces_block_and_builds_pmp():
    from dataclasses import replace
    from types import SimpleNamespace

    problem = _problem()
    ch0 = problem.spin_channels[0]
    # Delta = 1, below the gap. The global control problem carries global_scalar_G
    # blocks; the point block is recomputed numerically (exact 2F1) at the point,
    # Make the reference's evaluator unusable to check that the point is
    # recomputed, rather than relying on an interpolation-error threshold.
    reference = replace(ch0, prefactored=SimpleNamespace(
        evaluate=lambda *_: pytest.fail('point construction evaluated the reference interpolant')))
    point = point_constraint_channel(
        reference, "1.0", basis="global_scalar_G", h_ext=mp.mpf("0.0625"), lam=3)
    f_pt = point.prefactored.evaluate("zzb", mp.mpf("1.0"))
    # The point block matches the independent exact 2F1 evaluator: for ell = 0,
    # G(Delta) = k_{Delta/2}(1/2)^2 with the crossing shifts on both axes.
    half = mp.mpf(1) / 2
    hext = [mp.mpf("0.0625")] * 4
    b = global_evaluate_block(half, half, half, half, hext, coord="zzb",
                              m_max=0, n_max=0, z_shift=-2*hext[0], zbar_shift=-2*hext[0])
    g_true = b[0][0] if isinstance(b, list) else b
    assert abs(f_pt[0][0] - g_true) < mp.mpf(10) ** -40
    # The point block has constant numerators and a positive decay prefactor,
    # so evaluating it away from the frozen point stays finite.
    f_pt5 = point.prefactored.evaluate("zzb", mp.mpf("5.0"))
    assert all(mp.isfinite(f_pt5[m][n]) for m in range(len(f_pt5))
               for n in range(len(f_pt5[0])))

    # Appending the point channel rebuilds the PMP WITHOUT crashing on the
    # degree-0 block (the bug this fix addresses).
    extended = append_positivity_channels(problem, iter((point,)))
    assert len(extended.spin_channels) == len(problem.spin_channels) + 1
    assert extended.pmp_build is not None
    doc = extended.pmp_build.doc["PositiveMatrixWithPrefactorArray"]
    assert len(doc) == len(problem.spin_channels) + 1
    import numpy as np

    for empty in (None, (), iter(()), np.empty(0, dtype=object)):
        assert append_positivity_channels(problem, empty) is problem
    extended_array = append_positivity_channels(problem, np.array([point, point], dtype=object))
    assert extended_array.point_channels == problem.point_channels + (point, point)
    assert len(extended_array.pmp_build.doc["PositiveMatrixWithPrefactorArray"]) == len(problem.spin_channels) + 2


def test_input_fixed_operator_folds_into_normalization():
    """Inputting an operator with a specified OPE coefficient folds it into the
    normalization vector (grouped with the identity), not a new constraint row."""
    problem = _problem()
    ch0 = problem.spin_channels[0]
    coeff = mp.mpf("0.7")
    fvec = ch0.prefactored.evaluate("zzb", mp.mpf("1.0"))
    Fp = list(problem.F_pairs)

    N = normalization_with_fixed_operators(problem.identity_vector, Fp,
                                           [(ch0, "1.0", coeff)])
    # N = identity + coeff * F(1) componentwise over the F-pairs
    for i, (m, n) in enumerate(Fp):
        assert abs(N[i] - (problem.identity_vector[i] + coeff * fvec[m][n])) < mp.mpf(10) ** -40

    inputted = input_fixed_operators(problem, [(ch0, "1.0", coeff)])
    # same number of channels (no extra row); normalization changed; PMP rebuilt
    assert len(inputted.spin_channels) == len(problem.spin_channels)
    assert tuple(inputted.identity_vector) == N
    assert inputted.pmp_build is not None


def test_fixed_ope_known_vector_vanishes_for_exact_ising():
    """<sigma^4> of 2d Ising: sigma x sigma = 1 + eps (Delta = 1, library OPE^2 = 4) solves crossing, so the
    fixed-OPE known vector V_id + 4 V_eps is zero, and the OPE-bound normalization is V_eps itself.  The point
    channel's table is shifted to Delta*: evaluating it at Delta = 0 scaled V* by base^(-Delta*)."""
    from virasoro_bootstrap.identical_scalar import (
        build_identical_scalar_fixed_ope_problem, build_identical_scalar_ope_bound_problem,
        prepare_identical_scalar_gap_build,
    )

    spec = IdenticalScalarGapSpec(c=mp.mpf(1) / 2, h_ext=mp.mpf(1) / 16, delta_gap=mp.mpf(3), lam=3,
                                  spin_list=(0, 2), q_order=20, dps=40, npoles=6, num_threads=1)
    prepared = prepare_identical_scalar_gap_build(spec)
    fixed = build_identical_scalar_fixed_ope_problem(spec, delta_star=1, ope_squared=4, prepared=prepared)
    bound = build_identical_scalar_ope_bound_problem(spec, delta_star=1, direction="upper", prepared=prepared)
    with mp.workdps(40):
        # q_order = 20 truncation-limited
        assert max(abs(v) for v in fixed.normalization_vector) < 1e-22
        for v_id, v_star in zip(prepared.identity_vector, bound.normalization_vector):
            assert abs(v_id + 4 * v_star) < 1e-22


def test_exact_certification_keeps_point_channels_and_the_twist_floor():
    """The exact-block certification set dropped point-constraint channels (a functional negative at the
    point still certified), and a large-spin tower (twist 0 in ell / delta_min) collapsed the extra-spin
    twist floor to unitarity."""
    from virasoro_bootstrap.certification import _build_exact_block_channels, _build_extra_spin_channels
    from virasoro_bootstrap.identical_scalar import build_identical_scalar_gap_problem

    base = dict(c=mp.mpf(2), h_ext=mp.mpf("0.1"), lam=3, q_order=12, dps=40, npoles=4, num_threads=1)
    with mp.workdps(40):
        spec = IdenticalScalarGapSpec(delta_gap=mp.mpf(2), spin_list=(0, 2),
                                      point_constraints=((0, mp.mpf("0.3")),), **base)
        problem = build_identical_scalar_gap_problem(spec)
        assert len(_build_exact_block_channels(problem)) == len(problem.spin_channels) == 3
        tau = mp.mpf("0.5")
        spec = IdenticalScalarGapSpec(delta_gap=tau, spin_list=(0, 2, 4), delta_min_by_spin={2: 2 + tau, 4: 4 + tau},
                                      large_spin_lmin=6, large_spin_twists=(tau, mp.mpf("0.9")),
                                      large_spin_npoles=3, **base)
        extra = _build_extra_spin_channels(build_identical_scalar_gap_problem(spec), 2, exact_blocks=True)
        # extra spins sit above the largest FIXED spin (4); the tower's int(Lmax) is not a spin sector
        assert [(ch.ell, ch.delta_min) for ch in extra] == [(6, 6 + tau), (8, 8 + tau)]


def test_extra_spins_ignore_an_odd_tower_floor():
    """The extra-spin set is built from the fixed-spin channels: a tower channel carries ell = int(Lmax), and an
    odd Lmax made the spins above it odd."""
    from virasoro_bootstrap.certification import _build_extra_spin_channels
    from virasoro_bootstrap.identical_scalar import build_identical_scalar_gap_problem

    tau = mp.mpf("0.5")
    with mp.workdps(40):
        spec = IdenticalScalarGapSpec(c=mp.mpf(2), h_ext=mp.mpf("0.1"), lam=3, q_order=12, dps=40, npoles=4,
                                      num_threads=1, delta_gap=tau, spin_list=(0, 2, 4),
                                      delta_min_by_spin={2: 2 + tau, 4: 4 + tau}, large_spin_lmin=5,
                                      large_spin_twists=(tau,), large_spin_npoles=3)
        extra = _build_extra_spin_channels(build_identical_scalar_gap_problem(spec), 2, exact_blocks=True)
    assert [ch.ell for ch in extra] == [6, 8]
