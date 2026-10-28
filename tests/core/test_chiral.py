"""Validate the symbolic-h representation of the chiral block F(h, z)."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.chiral import F_block_chiral_h
from virasoro_blocks.params import b_from_c, d_rs
from virasoro_blocks.recursion import F_block


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


@pytest.mark.parametrize("q_order", [12])
def test_chiral_h_matches_F_block(q_order):
    """FChiralH.evaluate(h) matches the direct numerical F at the same (h, z, d, c)."""
    mp.mp.dps = 50
    c = mp.mpf("1.7")
    b = b_from_c(c)
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    z = mp.mpf("0.4")

    chiral = F_block_chiral_h(z, d, b=b, q_order=q_order)

    for h in [mp.mpf("0.5"), mp.mpf("1.4"), mp.mpf("3.2"), mp.mpf("10")]:
        F_direct = F_block(h, z, d, b=b, q_order=q_order)
        F_sym = chiral.evaluate(h)
        # The two routes evaluate the same expression but in different orders:
        # the symbolic route builds a degree-O(q_order log q_order) polynomial
        # in h and evaluates by Horner, losing a few digits of the working
        # precision. Tolerance scaled to match.
        assert _close(F_sym, F_direct, tol=mp.mpf("1e-30"))


def test_chiral_h_handles_multiple_z():
    """A separate FChiralH is built per z; each matches F_block at its z."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.1")] * 4
    h = mp.mpf("0.6")

    for z in [mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.7")]:
        chiral = F_block_chiral_h(z, d, b=b, q_order=10)
        assert _close(
            chiral.evaluate(h),
            F_block(h, z, d, b=b, q_order=10),
            tol=mp.mpf("1e-35"),
        )


def test_chiral_h_denominator_roots_are_d_rs():
    """With generic externals, the denominator factors as prod(h - d_{r,s})."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.3"))
    chiral = F_block_chiral_h(
        mp.mpf("0.5"),
        [mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.4")],
        b=b,
        q_order=5,
    )

    expected_rs = []
    for n in range(1, 6):
        for r in range(1, n + 1):
            if n % r == 0:
                expected_rs.append((r, n // r))
    expected_d = [d_rs(r, s, b) for r, s in expected_rs]

    # The set of denominator_roots should equal expected_d up to ordering.
    assert len(chiral.denominator_roots) == len(expected_d)
    # Match them up by closest.
    for d_expected in expected_d:
        matched = min(chiral.denominator_roots, key=lambda x: abs(x - d_expected))
        assert _close(matched, d_expected, tol=mp.mpf("1e-40"))


def test_chiral_h_evaluation_at_pole_diverges():
    """At h = d_{r,s} the denominator vanishes; with generic externals (so the
    residue does not accidentally vanish), F diverges as h -> d_{r,s}."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.3"))
    # Generic unequal externals: with d_i all equal the (1,1) fusion polynomial
    # is proportional to lambda_L^2 - lambda_R^2 and vanishes, killing the
    # residue at d_{1,1}. Distinct d_i keeps the residue.
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    chiral = F_block_chiral_h(mp.mpf("0.5"), d, b=b, q_order=5)

    pole = chiral.denominator_roots[0]
    eps = mp.mpf("1e-10")
    val_close = chiral.evaluate(pole + eps)
    val_far = chiral.evaluate(pole + mp.mpf("0.5"))
    assert abs(val_close) > 1e6 * abs(val_far)


def test_chiral_h_factorizes_exp_part():
    """At small z, q(z) = z/16 + z^2/32 + O(z^3) so 16 q / N = z/N + z^2/(2 N) + O(z^3)
    with N = chiral_norm (default 16)."""
    from virasoro_blocks.recursion import DEFAULT_CHIRAL_NORM
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.7"))
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    z = mp.mpf("1e-6")

    chiral = F_block_chiral_h(z, d, b=b, q_order=6)
    N = DEFAULT_CHIRAL_NORM
    # 16 q / N ~ z/N + z^2/(2 N) + O(z^3). At z = 1e-6 the correction is ~ z^2/(2N).
    assert _close(chiral.exp_base, z / N + z ** 2 / (2 * N), tol=mp.mpf("1e-11"))


def test_chiral_h_validates_inputs():
    with pytest.raises(ValueError):
        F_block_chiral_h(mp.mpf("0.4"), [mp.mpf("0.1")] * 3, c=mp.mpf("1.5"), q_order=4)
    with pytest.raises(ValueError):
        F_block_chiral_h(mp.mpf("0.4"), [mp.mpf("0.1")] * 4, q_order=4)
    with pytest.raises(ValueError):
        F_block_chiral_h(
            mp.mpf("0.4"),
            [mp.mpf("0.1")] * 4,
            b=mp.mpf("1"),
            c=mp.mpf("1.5"),
            q_order=4,
        )
    with pytest.raises(ValueError):
        F_block_chiral_h(mp.mpf("0.4"), [mp.mpf("0.1")] * 4, c=mp.mpf("1.5"), q_order=-1)


def test_chiral_h_exchange_literals_follow_the_exact_input_policy():
    with mp.workdps(30):
        external = [Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5)]
        chiral = F_block_chiral_h(Fraction(3, 10), external, c=32, q_order=4)
        reference = F_block(Fraction(7, 10), Fraction(3, 10), external, c=32, q_order=4)
        for value in (Fraction(7, 10), Decimal('.7'), '.7', 0.7):
            assert mp.almosteq(chiral.evaluate(value), reference,
                               rel_eps=mp.mpf('1e-28'), abs_eps=0)


def test_chiral_h_distinct_external_inputs_keep_the_identity_pole():
    external = [Fraction(1, 8) + Fraction(1, 10**70), Fraction(1, 8),
                Fraction(1, 5), Fraction(3, 10)]
    with mp.workdps(30):
        chiral = F_block_chiral_h(Fraction(3, 10), external, c=32, q_order=2)
        assert 0 in chiral.denominator_roots
        with pytest.raises(ZeroDivisionError):
            chiral.evaluate(0)


def test_exponent_guards_preserve_native_charge_rejection_before_promotion():
    from virasoro_blocks.derivs import chiral_z_series_evaluator
    from virasoro_blocks.polyderivs import F_z_poly_table
    from virasoro_blocks.singularity import SingularCentralChargeError

    with mp.workdps(30):
        charge = -mp.mpf(22)/5
        external = [k*10**10 for k in (1, 2, 3, 4)]
        for build in (
            lambda: F_block_chiral_h('.5', external, c=charge, q_order=10),
            lambda: chiral_z_series_evaluator(external, 1, c=charge, q_order=10),
            lambda: F_z_poly_table(external, 0, c=charge, q_order=10),
        ):
            with pytest.raises(SingularCentralChargeError):
                build()
            assert mp.mp.dps == 30


def test_near_charge_guard_is_independent_of_the_first_cross_ratio(monkeypatch):
    from virasoro_blocks import singularity
    from virasoro_blocks.recursion import F_block

    charge = Fraction(7, 10)+Fraction(1, 10**16)
    external = [Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5)]
    weight = Fraction(7, 10)
    points = (Fraction(1, 100), Fraction(99, 100))
    with mp.workdps(140):
        references = {z: F_block(weight, z, external, c=charge, q_order=12) for z in points}
    for ordering in (points, points[::-1]):
        monkeypatch.setattr(singularity, '_NEAR_GUARD_CACHE', {})
        with mp.workdps(30):
            for z in ordering:
                value = F_block_chiral_h(z, external, c=charge, q_order=12).evaluate(weight)
                assert mp.almosteq(value, references[z], rel_eps=mp.mpf('1e-28'), abs_eps=0)
            assert mp.mp.dps == 30
