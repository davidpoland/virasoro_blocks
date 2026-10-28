"""Validate the q-expansion Zamolodchikov recursion."""

from __future__ import annotations

import contextlib

import mpmath as mp

from virasoro_blocks.params import b_from_c
from virasoro_blocks.recursion import DEFAULT_CHIRAL_NORM, F_block, H_q_series, q_from_z


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def test_q_small_z():
    # q(z) ~ z/16 for small z.
    mp.mp.dps = 50
    z = mp.mpf("1e-6")
    q = q_from_z(z)
    assert _close(q, z / 16, tol=mp.mpf("1e-11"))


def test_q_half():
    # q(1/2) = exp(-pi).
    mp.mp.dps = 50
    q = q_from_z(mp.mpf(1) / 2)
    assert _close(q, mp.exp(-mp.pi), tol=mp.mpf("1e-40"))


def test_F_approaches_one_at_small_z():
    # In the library convention (z_shift=0) F(h, z, d_i, c) -> N^{-h} * z^h as z -> 0.
    # Use small but nonzero z to avoid the (16q)^{h-sigma} singularity at z = 0.
    mp.mp.dps = 50
    c = mp.mpf("1.7")
    b = b_from_c(c)
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    z = mp.mpf("1e-6")
    F = F_block(h, z, d, b=b, q_order=6)
    expected = mp.power(DEFAULT_CHIRAL_NORM, -h) * z ** h
    assert _close(F, expected, tol=mp.mpf("1e-4"))


def test_F_matches_2F1_in_large_c_limit():
    # As c -> infty, the library-normalized block (z_shift=0) approaches
    # N^{-h} * z^h * 2F1(h+d2-d1, h+d3-d4, 2h, z).
    # The q-expansion form has a (16q)^{h-sigma} prefactor whose conditioning gets bad for
    # extremely large c, so we use a moderate c value (c~200) where the asymptotic agreement
    # is already at the 1e-3 level.
    mp.mp.dps = 50
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    h = mp.mpf("1.4")
    z = mp.mpf("0.3")
    c = mp.mpf("200")
    b = b_from_c(c)
    F = F_block(h, z, d, b=b, q_order=20)
    F2F1 = (
        mp.power(DEFAULT_CHIRAL_NORM, -h)
        * z ** h
        * mp.hyp2f1(h + d[1] - d[0], h + d[2] - d[3], 2 * h, z)
    )
    assert _close(F, F2F1, tol=mp.mpf("3e-3"))


def test_H_zero_order_is_one():
    # H_0(h) = 1 always.
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.3"))
    coeffs = H_q_series(mp.mpf("2.7"), [0, 0, 0, 0], b=b, q_order=6)
    assert coeffs[0] == mp.mpc(1)


def test_recursion_converges_with_q_order():
    # F at z=1/2 should converge as q_order is increased. Use generic c (not on the
    # minimal-model lattice c = 1 - 6/(m(m+1)) where A_rs has spurious zeros in the
    # denominator).
    mp.mp.dps = 60
    c = mp.mpf("0.75")
    b = b_from_c(c)
    d = [mp.mpf("0.1")] * 4
    h = mp.mpf("0.6")
    z = mp.mpf(1) / 2

    F8 = F_block(h, z, d, b=b, q_order=8)
    F16 = F_block(h, z, d, b=b, q_order=16)
    F24 = F_block(h, z, d, b=b, q_order=24)
    # Successive differences should shrink; at z=1/2, q=e^-pi~0.043, so q^8 ~ 1e-11.
    diff_8_16 = abs(F16 - F8)
    diff_16_24 = abs(F24 - F16)
    assert diff_16_24 < diff_8_16
    assert diff_16_24 < mp.mpf("1e-15")


def test_F_leading_z_expansion_matches_global():
    # The library-normalized F (z_shift=0) has leading z behavior:
    #   F = N^{-h} * z^h * (1 + c_1*z + ...)
    # where c_1 = (h+d2-d1)*(h+d3-d4)/(2h).
    # Test by extracting the first correction coefficient via finite difference.
    mp.mp.dps = 60
    c = mp.mpf("0.75")
    b = b_from_c(c)
    d = [mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.15"), mp.mpf("0.25")]
    h = mp.mpf("1.3")

    expected_c1 = (h + d[1] - d[0]) * (h + d[2] - d[3]) / (2 * h)

    z = mp.mpf("1e-3")
    z2 = mp.mpf("5e-4")
    F = F_block(h, z, d, b=b, q_order=20)
    F2 = F_block(h, z2, d, b=b, q_order=20)
    # Extract the ratio F / (N^{-h} * z^h) = 1 + c_1*z + ...
    norm = mp.power(DEFAULT_CHIRAL_NORM, -h)
    ratio = F / (norm * z ** h)
    ratio2 = F2 / (norm * z2 ** h)
    # Linear extrapolation: ratio = 1 + c_1*z => c_1 = (ratio - ratio2) / (z - z2)
    c1_num = (ratio - ratio2) / (z - z2)
    assert _close(c1_num, expected_c1, tol=mp.mpf("1e-2"))


@contextlib.contextmanager
def _force_mpmath_H_recursion():
    """Disable every flint H-context builder so H_q_series runs the pure-mpmath
    recursion -- the reference the flint fast paths must reproduce."""
    import virasoro_blocks.recursion as R

    saved = (
        R._build_H_arb_context,
        R._build_H_acb_strip_context,
        R._build_H_acb_nonstrip_context,
    )
    R._build_H_arb_context = lambda *a, **k: None
    R._build_H_acb_strip_context = lambda *a, **k: None
    R._build_H_acb_nonstrip_context = lambda *a, **k: None
    R._H_CONTEXT_CACHE.clear()
    try:
        yield R
    finally:
        (
            R._build_H_arb_context,
            R._build_H_acb_strip_context,
            R._build_H_acb_nonstrip_context,
        ) = saved
        R._H_CONTEXT_CACHE.clear()


def test_H_q_series_flint_paths_match_mpmath():
    """The flint acb/arb H-recursion fast paths (strip mode for 1 < c < 25,
    real-b mode for c > 25) reproduce the pure-mpmath recursion to working
    precision."""
    import virasoro_blocks.recursion as R

    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    worst = mp.mpf(0)
    # c < 1, strip (1<c<25), real-b; 0.55 rather than 0.7, which as a binary number sits 1e-81 from the tricritical
    # Ising locus 7/10, where the 80-digit mpmath reference (midpoint table, canceling top-level sum) is garbage
    for c in ("0.55", "7", "100"):
        for h in ("0.05", "4.7"):
            flint_res = R.H_q_series(mp.mpf(h), h_ext, c=mp.mpf(c), q_order=30)
            with _force_mpmath_H_recursion():
                mp_res = R.H_q_series(mp.mpf(h), h_ext, c=mp.mpf(c), q_order=30)
            for a, b in zip(mp_res, flint_res):
                rel = abs(mp.mpc(a) - mp.mpc(b)) / (abs(mp.mpc(a)) + mp.mpf("1e-200"))
                worst = max(worst, rel)
    assert worst < mp.mpf("1e-55"), worst


def test_H_q_series_complex_h_flint_paths_match_mpmath():
    """The flint acb complex-h H-recursion fast paths (strip and non-strip,
    real Kac data) reproduce the pure-mpmath recursion at complex h to working
    precision -- the eps-shifted "complex pole" sampling regime."""
    import virasoro_blocks.recursion as R

    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    worst = mp.mpf(0)
    for c in ("1.5", "2", "7", "20", "100"):  # strip (1<c<25) + real-b non-strip
        for h in (mp.mpc("0.5", "0.03"), mp.mpc("1.3", "-0.2"), mp.mpc("4.7", "0.7")):
            flint_res = R.H_q_series(h, h_ext, c=mp.mpf(c), q_order=30)
            with _force_mpmath_H_recursion():
                mp_res = R.H_q_series(h, h_ext, c=mp.mpf(c), q_order=30)
            for a, b in zip(mp_res, flint_res):
                rel = abs(mp.mpc(a) - mp.mpc(b)) / (abs(mp.mpc(a)) + mp.mpf("1e-200"))
                worst = max(worst, rel)
    assert worst < mp.mpf("1e-55"), worst


def test_H_q_series_complex_c_flint_path_matches_mpmath():
    """The acb non-strip evaluator reproduces the pure-mpmath recursion in the
    generic-complex mode (complex b => complex Kac data), at both real and
    complex exchanged h."""
    import virasoro_blocks.recursion as R

    mp.mp.dps = 80
    h_ext = [mp.mpf("0.0625")] * 4
    worst = mp.mpf(0)
    for b in (mp.mpc("1.1", "0.3"), mp.mpc("0.8", "-0.45")):
        # Confirm we are exercising the new acb path, not arb/strip.
        _, _, ctx = R._build_H_context(30, b, tuple(h_ext))
        assert ctx is not None and ctx.get("mode") == "complex_nonstrip"
        for h in (mp.mpf("0.5"), mp.mpc("1.3", "0.2"), mp.mpc("4.7", "-0.5")):
            flint_res = R.H_q_series(h, h_ext, b=b, q_order=30)
            with _force_mpmath_H_recursion():
                mp_res = R.H_q_series(h, h_ext, b=b, q_order=30)
            for a, c in zip(flint_res, mp_res):
                rel = abs(mp.mpc(a) - mp.mpc(c)) / (abs(mp.mpc(a)) + mp.mpf("1e-200"))
                worst = max(worst, rel)
    assert worst < mp.mpf("1e-55"), worst


def test_q_from_z_is_complex_for_real_z_above_one_at_high_precision():
    """The flint nome path retains the imaginary part for real z above one."""
    from virasoro_blocks.recursion import _q_from_z_cache

    with mp.workdps(50):
        _q_from_z_cache.clear()
        for z in (mp.mpf(2), mp.mpf("0.3"), mp.mpf(-1)):
            ref = mp.exp(-mp.pi * mp.ellipk(1 - z) / mp.ellipk(z))
            assert abs(q_from_z(z) - ref) < mp.mpf(10) ** -45
        _q_from_z_cache.clear()


def test_numeric_taylor_tables_keep_small_coefficients():
    """mp.taylor's default chop zeroed every coefficient of a large-weight block (F ~ 0.04^h, below 100 eps at
    h ~ 0.7 dps); the direct table must agree with the batched one."""
    from virasoro_blocks.derivs import G_zzb_table, G_zzb_table_many

    hx = [mp.mpf("0.1")] * 4
    with mp.workdps(30):
        for delta in (mp.mpf(44), mp.mpf(60)):
            direct = G_zzb_table(delta, 0, hx, 1, 1, c=mp.mpf("1.7"), q_order=8)
            batched = G_zzb_table_many([delta], 0, hx, 1, 1, c=mp.mpf("1.7"), q_order=8)[0]
            assert direct[0][0] != 0
            assert abs(direct[0][0] / batched[0][0] - 1) < 1e-25


def test_evaluate_block_on_the_diagonal_with_more_zbar_derivatives():
    """z = zbar, h = hbar reuses one chiral half; it must reach n_max > m_max (it raised IndexError)."""
    from virasoro_blocks.derivs import evaluate_block

    hx = [mp.mpf("0.1")] * 4
    kw = dict(c=mp.mpf("0.63"), q_order=12, m_max=1, n_max=3, coord="zzb")
    with mp.workdps(30):
        z = mp.mpf("0.3")
        diag = evaluate_block(z, z, mp.mpf("0.4"), mp.mpf("0.4"), hx, **kw)
        near = evaluate_block(z, z + mp.mpf("1e-40"), mp.mpf("0.4"), mp.mpf("0.4"), hx, **kw)
        assert max(abs(diag[m][n] - near[m][n]) for m in range(2) for n in range(4)) < 1e-25


def test_nome_at_the_endpoints_z_zero_and_one():
    """q(0) = 0 and q(1) = 1 on both the mpmath (dps < 40) and the flint (dps >= 40) paths; flint's K(1) gave nan."""
    from virasoro_blocks.recursion import q_from_z

    for dps in (30, 60):
        with mp.workdps(dps):
            assert q_from_z(mp.mpf(0)) == 0 and q_from_z(mp.mpf(1)) == 1
