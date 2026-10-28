"""The mpf-storage fast path (real b, real externals, lambda^2 > 0) must agree
with the mpc default path to working precision, and use the right storage type.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyblock import H_q_series_rational
from virasoro_blocks.recursion import F_block, H_q_series


def _mpc_force(value):
    """Force a value to be an mp.mpc with zero imaginary part."""
    return mp.mpc(mp.mpf(value), mp.mpf(0))


@pytest.mark.parametrize("q_order", [4, 8, 12])
def test_mpf_path_active_for_c_above_25_small_externals(q_order):
    """c = 30, small h_ext: pairs and polynomials should all be mpf."""
    mp.mp.dps = 40
    c = mp.mpf(30)
    b = b_from_c(c)
    d = [mp.mpf("0.1"), mp.mpf("0.15"), mp.mpf("0.12"), mp.mpf("0.13")]

    H = H_q_series_rational(d, b=b, q_order=q_order)

    assert all(isinstance(p["K"], mp.mpf) for p in H.pairs)
    assert all(isinstance(p["d"], mp.mpf) for p in H.pairs)
    assert H.denominator.field is mp.mpf
    assert all(num.field is mp.mpf for num in H.numerators)


def test_mpf_path_inactive_for_strip_c():
    """c = 10 (in (1, 25)): b is on the unit circle, K becomes mpc.

    Conjugate-pair assembly produces mpf-field numerators and denominator,
    so the output polynomials are mpf even though
    the intermediate K values are mpc.
    """
    mp.mp.dps = 30
    c = mp.mpf(10)
    b = b_from_c(c)
    d = [mp.mpf("0.1")] * 4

    H = H_q_series_rational(d, b=b, q_order=6)

    assert any(isinstance(p["K"], mp.mpc) for p in H.pairs)
    # Strip-symmetric assembly produces real polynomials despite complex K.
    assert H.denominator.field is mp.mpf


def test_strip_public_numeric_path_is_exactly_real_and_matches_rational():
    """The public H/F numerical path should inherit the strip optimization,
    not just the symbolic rational builder."""
    mp.mp.dps = 100
    c = mp.mpf(10)
    b = b_from_c(c)
    d = [mp.mpf("0.15")] * 4
    h = mp.mpf("8.0")
    q = mp.mpf("0.04")

    coeffs = H_q_series(h, d, b=b, q_order=40)
    assert all(not isinstance(val, mp.mpc) or val.imag == 0 for val in coeffs)

    H_val = mp.fsum(coeffs[k] * q**k for k in range(len(coeffs)))
    H_rat = H_q_series_rational(d, b=b, q_order=40)
    H_rat_val = H_rat.evaluate(h, q)
    rel = abs(mp.mpc(H_val) - mp.mpc(H_rat_val)) / (
        abs(mp.mpc(H_val)) + abs(mp.mpc(H_rat_val)) + 1
    )
    assert rel < mp.mpf("1e-70")

    F_val = F_block(h, mp.mpf("0.5"), d, b=b, q_order=40)
    assert mp.im(mp.mpc(F_val)) == 0


def test_mpf_path_active_when_lambda_sq_negative():
    """c = 30 with heavy externals (lambda^2 < 0): the manifestly-real fusion
    polynomial keeps the recursion in mp.mpf, even though individual
    lambda = sqrt(lambda^2) would be imaginary.
    """
    mp.mp.dps = 30
    c = mp.mpf(30)
    b = b_from_c(c)
    # (c-1)/24 = 29/24 ~ 1.21, so h_ext = 2 pushes lambda^2 < 0.
    d = [mp.mpf(2)] * 4

    H = H_q_series_rational(d, b=b, q_order=6)

    assert all(isinstance(p["K"], mp.mpf) for p in H.pairs)
    assert all(isinstance(p["d"], mp.mpf) for p in H.pairs)
    assert H.denominator.field is mp.mpf


@pytest.mark.parametrize("c_value", ["30", "100"])
@pytest.mark.parametrize("q_order", [4, 10])
def test_mpf_and_mpc_paths_agree(c_value, q_order):
    """The real-arithmetic fast path (real b/externals -> mpf storage, arb
    flint backend) must reproduce the mpc path to working precision on the
    physical block value F(h, z). The mpc path is forced by a tiny imaginary
    perturbation eps; with eps far below the tolerance, eps * (any derivative)
    is negligible.

    The comparison is on F_block, not on the intermediate H_q_series_rational
    numerator polynomials: those numerators carry tens of digits of internal
    cancellation (their coefficients reach ~1e130), so the real (arb_poly) and
    complex (acb_poly) assemblies round the canceled digits independently and
    agree only to ~1e-26 in the worst case here -- even though the block they
    build, and every downstream physics quantity, agrees to ~1e-55. The fast
    path's faithfulness is a property of the block, so that is what we assert.
    """
    mp.mp.dps = 60
    c = mp.mpf(c_value)
    b_mpf = b_from_c(c)
    d_real = [mp.mpf("0.1"), mp.mpf("0.05"), mp.mpf("0.08"), mp.mpf("0.06")]

    # The fast path must still produce a real (mpf-stored) table.
    H_real = H_q_series_rational(d_real, b=b_mpf, q_order=q_order)
    assert H_real.denominator.field is mp.mpf

    # Perturbation small enough to be irrelevant at the comparison tolerance.
    eps = mp.mpf(10) ** (-50)
    d_eps = [mp.mpc(di, eps) for di in d_real]
    b_eps = mp.mpc(b_mpf) + mp.mpc(0, eps)

    # Observed worst-case F_block agreement across this grid is ~8e-55; 1e-48
    # is tight enough to catch a genuine fast-path bug, loose enough to absorb
    # the eps perturbation and its amplification.
    tol = mp.mpf("1e-48")
    for h, z in [(mp.mpf("1.3"), mp.mpf("0.3")), (mp.mpf("4.0"), mp.mpf("0.5"))]:
        f_real = F_block(h, z, d_real, b=b_mpf, q_order=q_order)
        f_eps = F_block(mp.mpc(h), z, d_eps, b=b_eps, q_order=q_order)
        diff = abs(mp.mpc(f_real) - mp.mpc(f_eps))
        scale = abs(mp.mpc(f_real)) + 1
        assert diff / scale < tol, (
            f"c={c_value} q_order={q_order} h={h} z={z}: rel diff {diff / scale}"
        )


@pytest.mark.parametrize("c_value,expected", [
    (0.5, False),   # below strip: b on imaginary axis, |b| != 1
    (5, True),      # in strip
    (10, True),
    (15, True),
    (20, True),
    (30, False),    # above strip: b real
    (100, False),
])
def test_strip_symmetric_detection(c_value, expected):
    """_build_rs_data should flag strip symmetry iff c in (1, 25)."""
    mp.mp.dps = 30
    from virasoro_blocks.recursion import _build_rs_data
    c = mp.mpf(c_value)
    b = b_from_c(c)
    d = [mp.mpf("0.1")] * 4

    pt = _build_rs_data(8, b, d)
    assert pt.is_strip_symmetric is expected

    if expected:
        # In strip mode, the canonical set should be (r <= s) pairs only.
        for idx, canonical in enumerate(pt.is_canonical):
            r, s = pt.pairs[idx]["r"], pt.pairs[idx]["s"]
            assert canonical == (r <= s)
        # Twins should be (s, r) for each (r, s).
        by_rs = {(p["r"], p["s"]): i for i, p in enumerate(pt.pairs)}
        for idx, p in enumerate(pt.pairs):
            r, s = p["r"], p["s"]
            assert pt.twin_of[idx] == by_rs[(s, r)]


def test_strip_symmetric_off_when_externals_complex():
    """Strip symmetry requires all externals to be real."""
    mp.mp.dps = 30
    from virasoro_blocks.recursion import _build_rs_data
    b = b_from_c(mp.mpf(10))
    d = [mp.mpc("0.1", "0.01"), mp.mpf("0.1"), mp.mpf("0.1"), mp.mpf("0.1")]

    pt = _build_rs_data(6, b, d)
    assert pt.is_strip_symmetric is False


def test_evaluate_returns_mpf_when_inputs_real():
    """HRational.evaluate(h, q) returns mp.mpf when h and q are mpf and the
    underlying polynomials are mpf."""
    mp.mp.dps = 30
    c = mp.mpf(30)
    b = b_from_c(c)
    d = [mp.mpf("0.1")] * 4
    H = H_q_series_rational(d, b=b, q_order=6)

    value = H.evaluate(mp.mpf(5), mp.mpf("0.1"))
    assert isinstance(value, mp.mpf)


# --- imag_b mode (c < 1 with real externals) ------------------------------


@pytest.mark.parametrize("c_value", ["0.6", "-0.3", "-2.0"])
def test_imag_b_mode_active_for_c_below_1(c_value):
    """At generic c < 1 (b purely imaginary), `_build_rs_data` should produce
    pairs whose K and d are mp.mpf, and the symbolic table's polynomials
    should also be mp.mpf. Minimal-model c values are intentionally avoided
    so the bare recursion is well-defined."""
    from virasoro_blocks.recursion import _build_rs_data
    mp.mp.dps = 50
    c = mp.mpf(c_value)
    b = b_from_c(c)
    d = [mp.mpf("0.05"), mp.mpf("0.07"), mp.mpf("0.09"), mp.mpf("0.11")]

    pt = _build_rs_data(8, b, d)
    assert pt.mode == "imag_b"
    assert all(isinstance(p["K"], mp.mpf) for p in pt.pairs)
    assert all(isinstance(p["d"], mp.mpf) for p in pt.pairs)

    H = H_q_series_rational(d, b=b, q_order=8)
    assert H.denominator.field is mp.mpf
    assert all(num.field is mp.mpf for num in H.numerators)


@pytest.mark.parametrize("c_value", ["0.6", "-0.3"])
@pytest.mark.parametrize("q_order", [4, 10, 16])
def test_imag_b_matches_generic_complex_path(c_value, q_order):
    """The new exact-real imag_b path matches the generic complex path to
    working precision: forcing the generic path via a tiny imaginary
    perturbation in one external must give the same H coefficients."""
    mp.mp.dps = 80
    c = mp.mpf(c_value)
    b = b_from_c(c)
    d_real = [mp.mpf("0.05"), mp.mpf("0.07"), mp.mpf("0.09"), mp.mpf("0.11")]
    h = mp.mpf("2.0")

    coeffs_imag_b = H_q_series(h, d_real, b=b, q_order=q_order)
    # Force generic_complex by adding a tiny imaginary part to one external.
    eps = mp.mpf(10) ** (-70)
    d_eps = list(d_real)
    d_eps[0] = mp.mpc(d_real[0], eps)
    coeffs_generic = H_q_series(mp.mpc(h), d_eps, b=b, q_order=q_order)

    tol = mp.mpf(10) ** (-30)
    for k, (a, b_c) in enumerate(zip(coeffs_imag_b, coeffs_generic)):
        diff = abs(mp.mpc(a) - mp.mpc(b_c))
        scale = abs(mp.mpc(a)) + abs(mp.mpc(b_c)) + 1
        assert diff / scale < tol, f"k={k}, c={c_value}: rel diff {diff/scale}"


def test_imag_b_F_block_is_exactly_real():
    """F_block at c < 1 with real externals lands as a pure mp.mpc with
    exactly-zero imaginary part once the imag_b path is used."""
    mp.mp.dps = 60
    c = mp.mpf("0.6")
    b = b_from_c(c)
    d = [mp.mpf("0.1")] * 4
    h = mp.mpf("2.5")
    z = mp.mpf("0.5")

    F = F_block(h, z, d, b=b, q_order=12)
    # F_block always returns mpc (the block_prefactor goes through theta_3,
    # 16q, etc.), but for c < 1 with real externals the imag part should be
    # vanishingly small (no roundoff cascade from complex K/d recursion).
    assert mp.im(F) == 0 or abs(mp.im(F)) / (abs(F) + 1) < mp.mpf(10) ** (-50)


# --- real_b mode with heavy externals (lambda^2 < 0) ----------------------


def test_real_b_stays_mpf_with_heavy_externals():
    """At c > 25 with heavy externals (h_ext large enough to push lambda^2 < 0),
    the new manifestly-real fusion polynomial keeps the recursion in mp.mpf
    instead of detouring through mp.mpc through a sqrt of a negative number."""
    mp.mp.dps = 40
    c = mp.mpf(30)
    b = b_from_c(c)
    # (c-1)/24 ~ 1.21; d=2 forces lambda^2 < 0 for each external.
    d = [mp.mpf(2), mp.mpf("1.8"), mp.mpf("2.2"), mp.mpf("1.7")]

    H = H_q_series_rational(d, b=b, q_order=8)
    assert all(isinstance(p["K"], mp.mpf) for p in H.pairs)
    assert all(isinstance(p["d"], mp.mpf) for p in H.pairs)
    assert H.denominator.field is mp.mpf
    assert all(num.field is mp.mpf for num in H.numerators)

    # Cross-check against the generic complex path (forced via tiny imag eps).
    eps = mp.mpf(10) ** (-30)
    d_eps = [mp.mpc(di, eps) for di in d]
    H_eps = H_q_series_rational(d_eps, b=b, q_order=8)
    h = mp.mpf("5.5")
    q = mp.mpf("0.04")
    v_real = H.evaluate(h, q)
    v_eps = H_eps.evaluate(mp.mpc(h), mp.mpc(q))
    assert abs(mp.mpc(v_real) - mp.mpc(v_eps)) / (abs(mp.mpc(v_eps)) + 1) < mp.mpf(10) ** (-15)


# --- arithmetic_mode classification ----------------------------------------


@pytest.mark.parametrize("c_value,expected_mode", [
    ("30", "real_b"),
    ("100", "real_b"),
    ("25.001", "real_b"),  # just above the singular endpoint
    ("10", "strip_twin"),
    ("5", "strip_twin"),
    ("24.999", "strip_twin"),  # just below the singular endpoint
    ("1.001", "strip_twin"),
    ("0.6", "imag_b"),
    ("-0.3", "imag_b"),
    ("0.999", "imag_b"),  # just below the singular endpoint
    ("1", "generic_complex"),  # singular endpoint: falls back to generic
    ("25", "generic_complex"),  # singular endpoint: falls back to generic
])
def test_arithmetic_mode_classification(c_value, expected_mode):
    """`arithmetic_mode` dispatches by the real-c interval, with c in {1, 25}
    routed to generic_complex (regularization wrappers handle those)."""
    from virasoro_blocks.params import arithmetic_mode
    mp.mp.dps = 30
    c = mp.mpf(c_value)
    d = [mp.mpf("0.1")] * 4
    assert arithmetic_mode(c, d) == expected_mode


def test_arithmetic_mode_falls_back_on_complex_external():
    """Any complex external (imag != 0) forces generic_complex."""
    from virasoro_blocks.params import arithmetic_mode
    mp.mp.dps = 30
    d = [mp.mpc("0.1", "0.01"), mp.mpf("0.1"), mp.mpf("0.1"), mp.mpf("0.1")]
    assert arithmetic_mode(mp.mpf(30), d) == "generic_complex"
    assert arithmetic_mode(mp.mpf(10), d) == "generic_complex"
    assert arithmetic_mode(mp.mpf("0.5"), d) == "generic_complex"


def test_arithmetic_mode_falls_back_on_complex_c():
    """A complex central charge (imag != 0 above tolerance) is generic_complex."""
    from virasoro_blocks.params import arithmetic_mode
    mp.mp.dps = 30
    d = [mp.mpf("0.1")] * 4
    assert arithmetic_mode(mp.mpc("30", "1e-5"), d) == "generic_complex"


def test_numerical_H_q_series_returns_mpf_for_real_inputs():
    """`recursion.H_q_series` should return mp.mpf coefficients when c is in
    the real_b or imag_b interval, externals are real, and h is real. This
    extends the symbolic mpf chain to the numerical path."""
    mp.mp.dps = 50
    for c_val in ["30", "0.6"]:
        c = mp.mpf(c_val)
        b = b_from_c(c)
        d = [mp.mpf("0.1")] * 4
        h = mp.mpf("3.5")
        coeffs = H_q_series(h, d, b=b, q_order=8)
        assert all(isinstance(c_k, mp.mpf) for c_k in coeffs), (
            f"c={c_val}: some H coefficient came back as mp.mpc"
        )


def test_F_block_returns_mpf_for_real_inputs():
    """`F_block` should land in mp.mpf when c is real (outside [1, 25] or in
    the strip), h is real, z is in (0, 1), and externals are real."""
    mp.mp.dps = 50
    z = mp.mpf("0.5")
    h = mp.mpf("3.5")
    d = [mp.mpf("0.1")] * 4
    for c_val in ["30", "10", "0.6"]:
        c = mp.mpf(c_val)
        b = b_from_c(c)
        F = F_block(h, z, d, b=b, q_order=10)
        assert isinstance(F, mp.mpf), (
            f"c={c_val}: F_block returned {type(F).__name__}, expected mpf"
        )


def test_numerical_real_path_matches_complex_path():
    """The exact-real numerical path must agree with the mpc-forced generic
    path to working precision across each mode."""
    mp.mp.dps = 60
    z = mp.mpf("0.5")
    h = mp.mpf("3.5")
    eps = mp.mpf(10) ** (-40)
    for c_val in ["30", "10", "0.6"]:
        c = mp.mpf(c_val)
        b = b_from_c(c)
        d = [mp.mpf("0.1"), mp.mpf("0.12"), mp.mpf("0.14"), mp.mpf("0.16")]
        d_eps = [mp.mpc(di, eps) for di in d]
        F_real = F_block(h, z, d, b=b, q_order=12)
        F_eps = F_block(mp.mpc(h), z, d_eps, b=b, q_order=12)
        rel = abs(mp.mpc(F_real) - mp.mpc(F_eps)) / (abs(mp.mpc(F_eps)) + 1)
        assert rel < mp.mpf(10) ** (-25), f"c={c_val}: rel diff {rel}"


# --- symbolic Delta-table path stays in mp.mpf for real-c / real-externals ---


def test_G_zzb_poly_table_real_inputs_produce_mpf_table():
    """The user-facing symbolic Delta-table builder should produce mp.mpf
    exp_base, numerators, denominator, and evaluate() output when c is real
    (outside the singular endpoints), externals are real, and Delta is real."""
    from virasoro_blocks.polyderivs import G_zzb_poly_table
    mp.mp.dps = 50
    SCALAR = (mp.mpf("0.1"),) * 4
    for c_val in ["30", "10", "0.6"]:
        tbl = G_zzb_poly_table(2, SCALAR, 2, 2, c=mp.mpf(c_val), q_order=6)
        assert isinstance(tbl.exp_base, mp.mpf), f"c={c_val}: exp_base"
        assert tbl.denominator.field is mp.mpf, f"c={c_val}: denominator field"
        assert tbl.numerators[0][0].field is mp.mpf, f"c={c_val}: numerator field"

        val = tbl.evaluate(mp.mpf("3.5"))
        for row in val:
            for entry in row:
                assert isinstance(entry, mp.mpf), (
                    f"c={c_val}: evaluate returned {type(entry).__name__}"
                )


def test_interpolated_delta_table_real_inputs_produce_mpf():
    """InterpolatedDeltaTable.evaluate stays in mp.mpf for the supported
    real-c interpolation regimes.

    Strip mode (1 < c < 25) has conjugate-pair Kac poles, but the DampedRational
    partitions them into real quadratic factors (x - Re p)^2 + (Im p)^2 at
    construction so the evaluation runs entirely in mp.mpf.
    """
    from virasoro_blocks.polyderivs import G_zzb_poly_table, interpolate_delta_table
    mp.mp.dps = 50
    SCALAR = (mp.mpf("0.1"),) * 4

    for c_val in ["30", "10"]:
        tbl = G_zzb_poly_table(2, SCALAR, 2, 2, c=mp.mpf(c_val), q_order=6)
        it = interpolate_delta_table(tbl, 3)
        val = it.evaluate(mp.mpf("3.5"))
        assert isinstance(val[0][0], mp.mpf), f"c={c_val}: evaluate"
        assert it.numerators[0][0].field is mp.mpf, f"c={c_val}: numerator field"


def test_strip_mode_damped_rational_partitions_conjugate_pairs():
    """DampedRational at strip c records the conjugate pairs in its partition
    data and keeps `_constant_real` set so the mp.mpf path stays active."""
    from virasoro_blocks.polyderivs import G_zzb_poly_table, interpolate_delta_table
    mp.mp.dps = 50
    SCALAR = (mp.mpf("0.1"),) * 4
    tbl = G_zzb_poly_table(2, SCALAR, 2, 2, c=mp.mpf(10), q_order=6)
    it = interpolate_delta_table(tbl, 3)
    # Strip c should have at least one conjugate pair in the kept prefactor.
    assert len(it.prefactor._pair_data) >= 1
    # The constant is mathematically real, so the mpf path is active.
    assert it.prefactor._constant_real is not None


def test_symbolic_table_matches_forced_mpc_path():
    """The exact-real symbolic table matches the mpc-forced path to working
    precision in every real-c mode."""
    from virasoro_blocks.polyderivs import G_zzb_poly_table
    mp.mp.dps = 60
    SCALAR = (mp.mpf("0.1"),) * 4
    SCALAR_eps = (mp.mpc("0.1", mp.mpf(10) ** (-40)),) + (mp.mpf("0.1"),) * 3
    for c_val in ["30", "10", "0.6"]:
        tbl_real = G_zzb_poly_table(2, SCALAR, 2, 2, c=mp.mpf(c_val), q_order=8)
        tbl_mpc = G_zzb_poly_table(2, SCALAR_eps, 2, 2, c=mp.mpf(c_val), q_order=8)
        for Delta in [mp.mpf("3.5"), mp.mpf("8.2")]:
            v_real = tbl_real.evaluate(Delta)
            v_mpc = tbl_mpc.evaluate(mp.mpc(Delta))
            for m in range(len(v_real)):
                for n in range(len(v_real[m])):
                    a, b = mp.mpc(v_real[m][n]), mp.mpc(v_mpc[m][n])
                    rel = abs(a - b) / (abs(b) + 1)
                    assert rel < mp.mpf(10) ** (-25), (
                        f"c={c_val} Delta={Delta} (m,n)=({m},{n}): rel diff {rel}"
                    )


def test_fusion_polynomial_via_squares_matches_legacy_form():
    """`fusion_polynomial_via_squares` (manifestly-real, no sqrt) matches the
    legacy `fusion_polynomial` for representative (r, s) and c in each
    interval."""
    from virasoro_blocks.params import (
        fusion_polynomial, fusion_polynomial_via_squares,
    )
    mp.mp.dps = 60
    for c_val in [mp.mpf(30), mp.mpf(10), mp.mpf("0.6")]:
        b = b_from_c(c_val)
        b_sq = b * b
        Q_sq = b_sq + 2 + 1 / b_sq
        d_L = mp.mpf("0.3")
        d_R = mp.mpf("0.7")
        lam_L_sq = Q_sq - 4 * d_L
        lam_R_sq = Q_sq - 4 * d_R
        for r, s in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 3), (4, 3)]:
            legacy = fusion_polynomial(r, s, b, d_L, d_R)
            via_sq = fusion_polynomial_via_squares(r, s, b_sq, lam_L_sq, lam_R_sq)
            diff = abs(mp.mpc(legacy) - mp.mpc(via_sq))
            scale = abs(mp.mpc(legacy)) + 1
            assert diff / scale < mp.mpf(10) ** (-50), (
                f"c={c_val}, (r,s)=({r},{s}): rel diff {diff/scale}"
            )
