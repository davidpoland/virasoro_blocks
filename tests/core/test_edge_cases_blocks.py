"""Block-engine edge cases: singular charges, tiny z, exact inputs, and recentered samplers."""

from __future__ import annotations

import random
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.chiral import F_block_chiral_h
from virasoro_blocks.derivs import (
    G_zzb_table,
    _fold_parity,
    chiral_z_series_evaluator,
    chiral_z_taylor,
    evaluate_block,
)
from virasoro_blocks.global_blocks import global_evaluate_block
from virasoro_blocks.polyderivs import (
    B_zzb_poly_table,
    F_z_poly_table,
    G_zzb_poly_table,
    build_numeric_sampler,
    parity_projected_B_zzb_poly_table,
)
from virasoro_blocks.polynomial import _arb_to_mpf
from virasoro_blocks.qgate import q_tail_report
from virasoro_blocks.recursion import F_block, H_q_series, _fixed_h_block_evaluator, q_from_z
from virasoro_blocks.regularization import F_block_regularized
from virasoro_blocks.singularity import (
    SingularCentralChargeError,
    bare_recursion_singular_order,
    negative_unit_b2_q,
    recognized_singular_locus,
)
from virasoro_blocks.torus_onept import build_torus_onept_kernel, torus_onept_f_coeffs

EXT = [mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.15"), mp.mpf("0.3")]
H = mp.mpf("0.77")


def _rel(a, b):
    return abs(mp.mpc(a) - mp.mpc(b)) / abs(mp.mpc(b))


# --- item 1: bare builders fail closed at recognized singular c ------------------------------------


def _bare_calls(c, q_order=12):
    ext = [mp.mpf(x) for x in EXT]
    return {
        "evaluate_block": lambda: evaluate_block(
            mp.mpf("0.3"), mp.mpf("0.4"), H, H, ext, c=c, q_order=q_order
        ),
        "series_evaluator": lambda: chiral_z_series_evaluator(ext, 3, c=c, q_order=q_order),
        "fixed_h_evaluator": lambda: _fixed_h_block_evaluator(H, ext, c=c, q_order=q_order),
        "F_block_chiral_h": lambda: F_block_chiral_h(mp.mpf("0.5"), ext, c=c, q_order=q_order),
    }


@pytest.mark.parametrize("c_str", ["26", "50", "0", "-22/5", "1/2", "1", "25", "-2"])
def test_bare_builders_raise_at_singular_c(c_str):
    mp.mp.dps = 30
    c = mp.convert(Fraction(c_str))
    for name, call in _bare_calls(c).items():
        with pytest.raises(SingularCentralChargeError, match="regularize"):
            call()


@pytest.mark.parametrize("c_str", ["26", "1", "25", "-2"])
def test_symbolic_tables_raise_at_singular_c(c_str):
    mp.mp.dps = 30
    c = mp.convert(Fraction(c_str))
    with pytest.raises(ValueError, match="merged_delta_table"):
        G_zzb_poly_table(2, [mp.mpf("0.1")] * 4, 1, 1, c=c, q_order=10)
    with pytest.raises(SingularCentralChargeError):
        F_z_poly_table(EXT, 1, c=c, q_order=10)


@pytest.mark.parametrize("c_str", ["26", "-22/5", "1", "-2"])
def test_per_h_taylor_tables_regularize_at_singular_c(c_str):
    """chiral_*_taylor (and the G_* / B_* tables built on it) take the exact-c regularizer where
    the bare recursion is singular, like F_block / G_block do."""
    mp.mp.dps = 30
    c = mp.convert(Fraction(c_str))
    got = chiral_z_taylor(H, EXT, 3, c=c, q_order=12)
    want = mp.taylor(lambda z: F_block(H, z, EXT, c=c, q_order=12), mp.mpf(1) / 2, 2)
    assert max(_rel(a, b) for a, b in zip(got, want)) < mp.mpf("1e-20")
    ext = [mp.mpf("0.1")] * 4
    D, half = mp.mpf("2.7"), mp.mpf(1) / 2
    table = G_zzb_table(D, 2, ext, 1, 1, c=c, q_order=12)
    from virasoro_blocks.api import G_block

    assert _rel(table[0][0], G_block(D, 2, half, half, ext, c=c, q_order=12)) < mp.mpf("1e-25")


def test_live_singular_terms_depend_on_the_externals():
    """At c = 50 (b^2 = 1/6) every singular term up to q_order 7 has a residue that pairwise
    identical externals kill, so the bare evaluator runs there and is exact; generic externals
    are singular from q_order 7."""
    mp.mp.dps = 30
    c, z = mp.mpf(50), mp.mpf("0.4")
    ident = [mp.mpf("0.15")] * 4
    bare = _fixed_h_block_evaluator(H, ident, c=c, q_order=7)(z)
    assert _rel(bare, F_block_regularized(H, z, ident, c=c, q_order=7)) < mp.mpf("1e-25")
    with pytest.raises(SingularCentralChargeError):
        _fixed_h_block_evaluator(H, ident, c=c, q_order=8)
    with pytest.raises(SingularCentralChargeError):
        _fixed_h_block_evaluator(H, EXT, c=c, q_order=7)


def test_singular_c_error_is_also_a_zero_division_error():
    """Callers that caught the bare recursion's ZeroDivisionError keep working."""
    assert issubclass(SingularCentralChargeError, ValueError)
    assert issubclass(SingularCentralChargeError, ZeroDivisionError)


def test_bare_builders_pass_below_the_first_singular_order():
    """A recognized locus the truncated recursion never reaches is exact: Ising first bites at
    q_order 7, so q_order 6 runs bare and agrees with the regularized block."""
    mp.mp.dps = 30
    c = mp.mpf(1) / 2
    assert bare_recursion_singular_order(-1, 3, 4, 40) == 7
    bare = _fixed_h_block_evaluator(H, EXT, c=c, q_order=6)(mp.mpf("0.4"))
    reg = F_block_regularized(H, mp.mpf("0.4"), EXT, c=c, q_order=6)
    assert _rel(bare, reg) < mp.mpf("1e-25")
    with pytest.raises(SingularCentralChargeError):
        _fixed_h_block_evaluator(H, EXT, c=c, q_order=8)


def test_minimal_model_symbolic_builds_are_let_through():
    """The direct symbolic build at a c < 1 minimal model is the regularizer's pole-multiplicity
    oracle, so it is let through (full and skeleton); c = 1 is refused either way."""
    mp.mp.dps = 30
    F_z_poly_table(EXT, 1, c=mp.mpf(1) / 2, q_order=10, skeleton=True)
    G_zzb_poly_table(2, [mp.mpf("0.15")] * 4, 1, 1, c=mp.mpf(1) / 2, q_order=8)
    with pytest.raises(SingularCentralChargeError):
        F_z_poly_table(EXT, 1, c=mp.mpf(1), q_order=10, skeleton=True)


# --- item 2: q(z) at small |z| ------------------------------------------------------------------


@pytest.mark.parametrize("dps", [15, 30, 60])
def test_q_from_z_keeps_relative_precision_at_small_z(dps):
    mp.mp.dps = dps
    for e in (-3, -10, -20, -40):
        z = mp.mpf(10) ** e
        with mp.workdps(dps + 80):
            ref = mp.qfrom(m=mp.mpf(10) ** e)
        q = q_from_z(z)
        assert isinstance(q, mp.mpf)
        assert _rel(q, ref) < mp.mpf(10) ** (-(dps - 2))
    assert q_from_z(mp.mpf(0)) == 0


@pytest.mark.parametrize("dps", [30, 60])
def test_q_from_z_matches_nome_definition_off_small_z(dps):
    mp.mp.dps = dps
    for z in (mp.mpf("0.3"), mp.mpf("-0.4"), mp.mpf("0.97"), mp.mpc("0.2", "0.3"), mp.mpf("1.5")):
        with mp.workdps(dps + 40):
            zz = +z
            ref = mp.exp(-mp.pi * mp.ellipk(1 - zz) / mp.ellipk(zz))
        q = q_from_z(z)
        assert _rel(q, ref) < mp.mpf(10) ** (-(dps - 3))
        if isinstance(z, mp.mpf) and z < 1:
            assert isinstance(q, mp.mpf)


def test_F_block_at_tiny_z_is_precise():
    mp.mp.dps = 30
    z = mp.mpf("1e-8")
    v = F_block(mp.mpf("2.5"), z, [mp.mpf("0.1")] * 4, c=mp.mpf("3.3"), q_order=10)
    with mp.workdps(80):
        ref = F_block(mp.mpf("2.5"), mp.mpf("1e-8"), [mp.mpf("0.1")] * 4, c=mp.mpf("3.3"), q_order=10)
    assert _rel(v, ref) < mp.mpf("1e-27")


# --- item 3: b^2 = -1/q loci and the q_order-aware dual cutoff ------------------------------------


@pytest.mark.parametrize("c_str, q", [("-2", 2), ("-7", 3), ("-25/2", 4), ("-91/5", 5)])
def test_negative_unit_b2_loci_are_recognized(c_str, q):
    mp.mp.dps = 30
    c = mp.convert(Fraction(c_str))
    assert negative_unit_b2_q(c) == q
    assert recognized_singular_locus(c) == f"b² = -1/{q}"
    assert recognized_singular_locus(c + mp.mpf("1e-12")) is None


def test_F_block_auto_regularizes_at_c_minus_two():
    mp.mp.dps = 30
    v = F_block(H, mp.mpf("0.5"), EXT, c=mp.mpf(-2), q_order=12)
    with mp.workdps(90):
        near = F_block(H, mp.mpf("0.5"), EXT, c=mp.mpf(-2) + mp.mpf("1e-20"), q_order=12,
                       regularize="off")
    assert _rel(v, near) < mp.mpf("1e-15")


def test_dual_locus_cutoff_follows_q_order():
    mp.mp.dps = 30
    c = 13 + 6 * (mp.mpf(1) / 17 + 17)          # b^2 = 1/17, first bites at q_order 34+
    assert recognized_singular_locus(c) is None  # default max_q = 16
    assert recognized_singular_locus(c, q_order=40) == "b² = 1/17"
    v = F_block(H, mp.mpf("0.5"), EXT, c=c, q_order=40)
    with mp.workdps(90):
        near = F_block(H, mp.mpf("0.5"), EXT, c=c + mp.mpf("1e-25"), q_order=40, regularize="off")
    assert _rel(v, near) < mp.mpf("1e-15")


def test_recognition_does_not_flag_generic_c():
    mp.mp.dps = 30
    rng = random.Random(7)
    for _ in range(20):
        c = mp.mpf(rng.uniform(-60, 300))
        assert recognized_singular_locus(c, q_order=40) is None


# --- item 4: q-gate at singular loci --------------------------------------------------------------


@pytest.mark.parametrize("c_str", ["1", "25"])
def test_q_gate_is_converged_at_singular_loci(c_str):
    mp.mp.dps = 30
    ext = [mp.mpf("0.1")] * 4
    rep = q_tail_report(c=mp.mpf(c_str), h_externals=ext, q_order=20, n_terms=6)
    near = q_tail_report(c=mp.mpf(c_str) + mp.mpf("1e-4"), h_externals=ext, q_order=20, n_terms=6)
    assert rep["status"] == "ok"
    assert abs(rep["digits"] - near["digits"]) < 0.5


# --- item 5: torus kernel with Im c < 0 ---------------------------------------------------------


def test_torus_kernel_accepts_negative_imaginary_c():
    mp.mp.dps = 30
    d, h = mp.mpf("0.3"), mp.mpf("0.9")
    for c in (mp.mpc("3.3", "0.5"), mp.mpc("3.3", "-0.5")):
        coeffs = torus_onept_f_coeffs(build_torus_onept_kernel(4, d_ext=d, c=c), h)
        assert _rel(coeffs[1], d * (d - 1) / (2 * h)) < mp.mpf("1e-25")


# --- item 6: global_evaluate_block on the diagonal with n_max > m_max ----------------------------


def test_global_evaluate_block_diagonal_with_n_max_above_m_max():
    mp.mp.dps = 30
    z, h = mp.mpf("0.3"), mp.mpf("1.3")
    diag = global_evaluate_block(z, z, h, h, EXT, coord="zzb", m_max=0, n_max=2)
    off = global_evaluate_block(z, z + mp.mpf("1e-40"), h, h, EXT, coord="zzb", m_max=0, n_max=2)
    assert len(diag) == 1 and len(diag[0]) == 3
    for a, b in zip(diag[0], off[0]):
        assert _rel(a, b) < mp.mpf("1e-25")


# --- item 7: Fraction inputs ----------------------------------------------------------------------


def test_fraction_inputs_match_mpf_inputs():
    mp.mp.dps = 30
    ext = [mp.mpf(1) / 10] * 4
    ref = F_block(mp.mpf(7) / 10, mp.mpf(3) / 10, ext, c=mp.mpf(33) / 10, q_order=8)
    fr = F_block(Fraction(7, 10), Fraction(3, 10), [Fraction(1, 10)] * 4, c=Fraction(33, 10), q_order=8)
    assert _rel(fr, ref) < mp.mpf("1e-28")
    norm = F_block(mp.mpf(7) / 10, mp.mpf(3) / 10, ext, c=mp.mpf(33) / 10, q_order=8, chiral_norm=4)
    assert _rel(F_block(mp.mpf(7) / 10, mp.mpf(3) / 10, ext, c=mp.mpf(33) / 10, q_order=8,
                        chiral_norm=Fraction(4)), norm) < mp.mpf("1e-28")
    # a Fraction minimal-model c is recognized and auto-regularized
    assert recognized_singular_locus(Fraction(7, 10)) is not None
    reg = F_block(mp.mpf(7) / 10, mp.mpf(3) / 10, ext, c=mp.mpf(7) / 10, q_order=8)
    assert _rel(F_block(Fraction(7, 10), Fraction(3, 10), [Fraction(1, 10)] * 4, c=Fraction(7, 10),
                        q_order=8), reg) < mp.mpf("1e-25")
    Hs = H_q_series(Fraction(7, 10), [Fraction(1, 10)] * 4, c=mp.mpf(33) / 10, q_order=4)
    assert _rel(Hs[4], H_q_series(mp.mpf(7) / 10, ext, c=mp.mpf(33) / 10, q_order=4)[4]) < mp.mpf("1e-28")
    assert _rel(q_from_z(Fraction(3, 10)), q_from_z(mp.mpf(3) / 10)) < mp.mpf("1e-29")
    chi = F_block_chiral_h(Fraction(3, 10), [Fraction(1, 10)] * 4, c=Fraction(33, 10), q_order=8)
    assert _rel(chi.evaluate(mp.mpf(7) / 10), ref) < mp.mpf("1e-25")


# --- item 8: numeric samplers with a diagonal off-symmetric center -------------------------------


def _as_mp(x):
    return x if isinstance(x, (mp.mpf, mp.mpc)) else mp.mpf(_arb_to_mpf(x))


@pytest.mark.parametrize("center_bar", [None, "0.3"])
def test_numeric_samplers_honour_a_diagonal_center(center_bar):
    mp.mp.dps = 30
    ext = [mp.mpf("0.1")] * 4
    c, D, center = mp.mpf("3.3"), mp.mpf("2.7"), mp.mpf("0.3")
    cbar = None if center_bar is None else mp.mpf(center_bar)
    kw = dict(h_externals=ext, hbar_externals=ext, ell=2, m_max=2, n_max=2, c=c, q_order=10,
              chiral_norm=None, center=center, center_bar=cbar)
    cases = [
        (G_zzb_poly_table(2, ext, 2, 2, c=c, q_order=10, center=center, center_bar=cbar, lambda_cut=None),
         build_numeric_sampler("zzb", chiral=False, spinning=False, projected=False, **kw)),
        (B_zzb_poly_table(2, ext, ext, 2, 2, c=c, q_order=10, center=center, center_bar=cbar,
                          lambda_cut=None),
         build_numeric_sampler("zzb", chiral=False, spinning=True, projected=False, **kw)),
        (parity_projected_B_zzb_poly_table(2, ext, ext, 2, 2, c=c, q_order=10, parity="+",
                                           center=center, center_bar=cbar,
                                           lambda_cut=None),
         build_numeric_sampler("zzb", chiral=False, spinning=True, projected=True, parity="+",
                               **kw)),
    ]
    for symbolic, sampler in cases:
        want = symbolic.evaluate(D)
        got = sampler([D])[0][0]
        for m in range(3):
            for n in range(3):
                assert _rel(_as_mp(got[m][n]), want[m][n]) < mp.mpf("1e-25")
    # and the ordered B is the product of the chiral halves at z = zbar = 0.3
    h, hbar = (D + 2) / 2, (D - 2) / 2
    direct = evaluate_block(center, center, h, hbar, ext, c=c, q_order=10, coord="zzb", m_max=2, n_max=2)
    got = cases[1][1]([D])[0][0]
    for m in range(3):
        for n in range(3):
            assert _rel(_as_mp(got[m][n]), direct[m][n]) < mp.mpf("1e-25")


# --- item 9: folding a grid that mixes mpf and mpc entries ---------------------------------------


@pytest.mark.parametrize("ell", [0, 1])
def test_fold_parity_mixed_real_and_complex_entries(ell):
    mp.mp.dps = 30
    size = 8
    grid = [[mp.mpf(1) / (1 + m + 2 * n) for n in range(size)] for m in range(size)]
    grid[1][2] = mp.mpc(grid[1][2], "0.25")
    grid[3][0] = mp.mpc(grid[3][0], 0)
    as_mpc = [[mp.mpc(v) for v in row] for row in grid]
    got = _fold_parity(grid, ell, 2, 2)
    want = _fold_parity(as_mpc, ell, 2, 2)
    for a, b in zip(got, want):
        for x, y in zip(a, b):
            assert abs(mp.mpc(x) - mp.mpc(y)) < mp.mpf("1e-28")


def test_symbolic_tables_fail_closed_at_a_minimal_model_but_skeletons_do_not():
    """Direct symbolic tables at a c < 1 minimal model are wrong beyond q_order ~ 12 (0.2 at c = 1/2, q_order 24);
    the full build fails closed with the regularizer as the route, a skeleton (the collision-strip oracle) does not."""
    from virasoro_blocks.singularity import SingularCentralChargeError, bare_recursion_probes

    externals = [mp.mpf("0.13")] * 4
    with mp.workdps(30):
        with pytest.raises(SingularCentralChargeError, match="merged_delta_table"):
            G_zzb_poly_table(2, externals, 2, 2, c=mp.mpf("0.5"), q_order=16, lambda_cut=None)
        skeleton = G_zzb_poly_table(2, externals, 2, 2, c=mp.mpf("0.5"), q_order=16, lambda_cut=None, skeleton=True)
        assert skeleton.numerators is None
        with bare_recursion_probes():
            G_zzb_poly_table(2, externals, 2, 2, c=mp.mpf("0.5"), q_order=8, lambda_cut=None)


def test_superconformal_blocks_accept_complex_and_fraction_c():
    """F_block_r raised TypeError at complex c (mp.mpf(c) in the Ramond parameter and the super-c resolver), and
    both sectors rejected Fraction inputs.  Complex c is continuous with real c; Fraction matches mpf."""
    from virasoro_blocks.superconformal_blocks import F_block_ns, F_block_r

    with mp.workdps(30):
        args = (mp.mpf("0.4"), mp.mpf("0.3"), [mp.mpf("0.1")] * 4)
        for block in (F_block_ns, F_block_r):
            real = block(*args, c=mp.mpf("1.7"), q_order=8)
            assert abs(block(*args, c=mp.mpc("1.7", "1e-20"), q_order=8) - real) < mp.mpf("1e-18")
            assert abs(block(*args, c=mp.mpc("1.7", "0.3"), q_order=8) - real) > mp.mpf("1e-6")
            exact = block(Fraction(2, 5), Fraction(3, 10), [Fraction(1, 10)] * 4, c=Fraction(17, 10), q_order=8)
            assert abs(exact - real) < mp.mpf("1e-25")


def test_F_block_at_kac_weights_and_ope_points():
    """F_block at an exact genuine Kac pole returned ~1e25 rounding garbage (or ZeroDivisionError), and z = 0 gave
    nan; a genuine pole now raises BlockPoleError, a removable Kac weight returns the limit, z = 0, 1 raise."""
    from virasoro_blocks.recursion import F_block
    from virasoro_blocks.requests import BlockPoleError

    with mp.workdps(30):
        # At c=13, b^2=i and these weights have exactly representable
        # components. A numerically rounded irrational Kac weight instead
        # denotes its supplied binary value, not the exact pole.
        c = 13
        h_12 = mp.mpc("-.5", ".75")
        generic = [mp.mpf("0.2")] * 4
        with pytest.raises(BlockPoleError):
            F_block(h_12, mp.mpf("0.3"), generic, c=c, q_order=10)
        unequal = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.1")]
        with pytest.raises(BlockPoleError):      # h = 0 is a genuine pole unless h1 = h2 or h3 = h4
            F_block(mp.mpf(0), mp.mpf("0.3"), unequal, c=c, q_order=8)
        degenerate = [h_12] * 4                  # (1, 3) is removable for four (1, 2) externals
        h_13 = mp.mpc(-1, 2)
        exact = F_block(h_13, mp.mpf("0.3"), degenerate, c=c, q_order=8)
        near = F_block(h_13 + mp.mpf("1e-14"), mp.mpf("0.3"), degenerate, c=c, q_order=8)
        assert abs(exact - near) < mp.mpf("1e-11")
        for z in (0, 1):
            with pytest.raises(ValueError, match="OPE point"):
                F_block(mp.mpf("0.9"), z, generic, c=c, q_order=8)


def test_float_c_snaps_to_the_singular_locus_it_means():
    """A Python float within float rounding of a singular locus is that locus (6/7 typed as a float is M(6,7), with a
    notice where its decimal spelling is not the locus); any other float is its decimal spelling. An mpf, a string
    or a Fraction is never snapped."""
    import warnings as _warnings

    from virasoro_blocks.derivs import G_zzb_table
    from virasoro_blocks.recursion import F_block
    from virasoro_blocks.singularity import SnappedCentralChargeWarning, _SNAP_WARNED, snap_float_c

    _SNAP_WARNED.clear()
    with mp.workdps(30):
        assert snap_float_c(0.7) == Fraction(7, 10)
        for typed, locus in ((6 / 7, Fraction(6, 7)), (25 / 28, Fraction(25, 28)), (-68 / 7, Fraction(-68, 7)),
                             (0.7000000000000001, Fraction(7, 10))):
            with pytest.warns(SnappedCentralChargeWarning):
                assert snap_float_c(typed) == locus
        assert snap_float_c(mp.mpf(6) / 7) == mp.mpf(6) / 7
        assert snap_float_c("0.8571428571428571") == Fraction("0.8571428571428571")
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            assert snap_float_c(0.5) == Fraction(1, 2)          # exact: no notice
            assert snap_float_c(0.73) == Fraction(73, 100)
            deliberate = mp.mpf(7) / 10 + mp.mpf("1e-20")
            assert snap_float_c(deliberate) is deliberate
        ext = [mp.mpf("0.1")] * 4
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", SnappedCentralChargeWarning)
            assert F_block(mp.mpf("0.9"), mp.mpf("0.3"), ext, c=0.7, q_order=16) == \
                F_block(mp.mpf("0.9"), mp.mpf("0.3"), ext, c=Fraction(7, 10), q_order=16)
            float_table = G_zzb_table(mp.mpf("0.9"), 0, ext, 2, 2, c=0.7, q_order=16)
        exact_table = G_zzb_table(mp.mpf("0.9"), 0, ext, 2, 2, c=Fraction(7, 10), q_order=16)
        assert float_table == exact_table


def test_bootstrap_spec_snaps_a_float_c():
    import warnings as _warnings

    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        spec = IdenticalScalarGapSpec(c=0.7, h_ext=mp.mpf("0.1"), delta_gap=mp.mpf(1), lam=3, spin_list=(0,),
                                      q_order=8, dps=30)
        minimal = IdenticalScalarGapSpec(c=6 / 7, h_ext=mp.mpf("0.1"), delta_gap=mp.mpf(1), lam=3, spin_list=(0,),
                                         q_order=8, dps=30)
    assert spec.c == Fraction(7, 10)
    assert minimal.c == Fraction(6, 7)


def test_near_singular_guard_is_verified():
    """A c deliberately near a minimal model loses digits in the objects built from the bare recursion
    (F_block_chiral_h at 7/10 + 1e-16, dps 30: 14 of 30).  The certified recursion's measured loss triggers the
    verification and the guard is measured per call site on the site's own result (no fitted first guess)."""
    import virasoro_blocks.singularity as S
    from virasoro_blocks.chiral import F_block_chiral_h
    from virasoro_blocks.derivs import G_zzb_table

    S._NEAR_GUARD_CACHE.clear()
    ext = [mp.mpf("0.1")] * 4
    with mp.workdps(30):
        c = mp.mpf(7) / 10 + mp.mpf("1e-16")
        assert S._measured_recursion_loss(c, ext, 16) > 0
        assert S._measured_recursion_loss(mp.mpf("3.7"), ext, 16) == 0
        table = G_zzb_table(mp.mpf("0.9"), 0, ext, 2, 2, c=c, q_order=16)
        chiral = F_block_chiral_h(mp.mpf("0.3"), ext, c=c, q_order=16).evaluate(mp.mpf("0.9"))
        with mp.workdps(300):
            reference = G_zzb_table(mp.mpf("0.9"), 0, ext, 2, 2, c=c, q_order=16)
            chiral_ref = F_block_chiral_h(mp.mpf("0.3"), ext, c=c, q_order=16).evaluate(mp.mpf("0.9"))
        for m in range(3):
            for n in range(3):
                assert abs(table[m][n] - reference[m][n]) <= mp.mpf("1e-27") * max(abs(reference[m][n]), 1)
        assert abs(chiral - chiral_ref) <= mp.mpf("1e-27") * max(abs(chiral_ref), 1)
        guards = [value for key, value in S._NEAR_GUARD_CACHE.items() if key[0] == "site"]
        assert guards and all(value >= 3 for value in guards)
    S._NEAR_GUARD_CACHE.clear()


def test_roundoff_imaginary_c_is_recognized_as_the_real_locus():
    """c = 1/2 + 1e-40 i at dps 30 divided by zero: every real-axis recognition check required im(c) == 0, so the
    bare recursion ran at the (effectively exact) Ising collision.  An imaginary part below the recognition
    tolerance is now roundoff; a resolvable one (1e-15) is still a complex c."""
    from virasoro_blocks.derivs import G_zzb_table
    from virasoro_blocks.recursion import F_block
    from virasoro_blocks.singularity import recognized_singular_locus

    with mp.workdps(30):
        ext = [mp.mpf("0.1")] * 4
        real = F_block(mp.mpf("0.9"), mp.mpf("0.3"), ext, c=mp.mpf("0.5"), q_order=12)
        tiny = mp.mpc("0.5", "1e-40")
        assert recognized_singular_locus(tiny) is not None
        assert abs(F_block(mp.mpf("0.9"), mp.mpf("0.3"), ext, c=tiny, q_order=12) - real) < mp.mpf("1e-28")
        G_zzb_table(mp.mpf("0.9"), 0, ext, 1, 1, c=tiny, q_order=12)
        assert recognized_singular_locus(mp.mpc("0.5", "1e-15")) is None
