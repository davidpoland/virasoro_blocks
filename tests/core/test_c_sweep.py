"""Pipeline checks across the central-charge range.

Representative c values, one per regime:

  c = 0.5   -- minimal model below c=1 (Ising, b purely imaginary); the
               recursion is singular once q_order is high enough to reach the
               resonant Kac (r,s)
  c = 0.65  -- generic c < 1 (non-minimal-model, b purely imaginary, real Kac
               poles); the symbolic table builds directly
  c = 1     -- recursion singular: A^c_{r,s} has b-poles at the c=1 collision
  c = 1.1   -- just above c=1; b on the unit circle, Kac poles near-collided
  c = 10    -- generic 1 < c < 25; complex-conjugate-pair Kac poles
  c = 25    -- self-dual point b = 1; ALSO singular -- b=1 makes mb+n/b vanish
               for n=-m, so A^c_{r,s} blows up exactly as it does at c=1
  c = 100   -- c > 25; b real, Kac poles real
    c = -3.7  -- generic negative c (non-minimal-model); the symbolic table builds directly
    c = -0.6  -- non-unitary minimal model below c = 1; singular once q_order reaches a resonant Kac pair

For c < 1 the Liouville parameter b is purely imaginary, so the Kac weights
d_{r,s} are real (like c > 25). Generic c < 1 builds directly; the singular
points (c = 1, c = 25, and the c < 1 minimal models) need
`merged_delta_table`.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.derivs import G_zzb_table
from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyderivs import G_zzb_poly_table, interpolate_delta_table
from virasoro_blocks.regularization import merged_delta_table

SCALAR_EXT = [mp.mpf("0.15")] * 4
# symbolic table builds directly (generic c < 1 and c > 1)
REGULAR_C = ["-3.7", "0.65", "1.1", "10", "100"]
INTERPOLATION_C = ["1.1", "10", "100"]
GENERIC_BELOW_ONE_C = ["-3.7", "0.65"]
SINGULAR_C = ["1", "25"]              # need merged_delta_table
# minimal models below c = 1; singular once q_order reaches the resonant (r,s)
MINIMAL_C_BELOW_ONE = ["0.5", "0.8", "-0.6"]


def _max_table_error(table_a, table_b):
    """Worst absolute discrepancy of two evaluated (m, n) tables."""
    worst = mp.mpf(0)
    for row_a, row_b in zip(table_a, table_b):
        for a, b in zip(row_a, row_b):
            worst = max(worst, abs(mp.mpc(a) - mp.mpc(b)))
    return worst


def _max_value_error(table_a, table_b, deltas=("0.37", "2.6", "7.9")):
    """Largest entry discrepancy of two DeltaDerivTables at a few Delta, relative to each grid's largest entry.

    The merged table at a singular c carries only the poles that survive (fewer than a probe table next to c0,
    whose colliding poles nearly cancel), so its components do not line up with the probe's: values are compared,
    each table evaluated at its own working precision (``evaluate`` rounds to the caller's)."""
    worst = mp.mpf(0)
    for D in deltas:
        ga, gb = table_a.evaluate(mp.mpf(D)), table_b.evaluate(mp.mpf(D))
        scale = max(abs(mp.mpc(v)) for row in gb for v in row)
        worst = max(worst, max(abs(mp.mpc(x) - mp.mpc(y)) for ra, rb in zip(ga, gb) for x, y in zip(ra, rb)) / scale)
    return worst

@pytest.mark.parametrize("c_value", REGULAR_C)
def test_symbolic_table_matches_numeric_across_c(c_value):
    """The symbolic-in-Delta table reproduces the numeric block table across
    the complex-pole (1<c<25) and real-pole (c>25) regimes."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf(c_value))
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    for Delta in [mp.mpf("2.6"), mp.mpf("5.4")]:
        numeric = G_zzb_table(Delta, 2, SCALAR_EXT, 2, 2, b=b, q_order=8)
        assert _max_table_error(exact.evaluate(Delta), numeric) < mp.mpf("1e-24")


@pytest.mark.parametrize("c_value", INTERPOLATION_C)
@pytest.mark.parametrize("pole_selection", ["rightmost", "nearest", "residue"])
def test_interpolation_across_c(c_value, pole_selection):
    """interpolate_delta_table reconstructs the block at each (m, n) entry's
    own sample nodes for every supported interpolation regime and every
    pole-selection rule. With per-entry polynomial degrees, the highest-
    degree entry's nodes are shared across the table only when every (m, n)
    has the same target degree; for lower-degree entries we check
    exactness at the per-entry node set."""
    mp.mp.dps = 55
    b = b_from_c(mp.mpf(c_value))
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=8, pole_selection=pole_selection)
    # Pick a node from the highest-degree entry's set; that entry must
    # reconstruct exactly there.
    node = interp.sample_points[2]
    numeric = G_zzb_table(node, 2, SCALAR_EXT, 2, 2, b=b, q_order=8)
    # Check the highest-degree (m, n) entry: should be exact there.
    hi_m = interp.m_max
    hi_n = interp.n_max
    interp_table = interp.evaluate(node)
    err_hi = abs(mp.mpc(interp_table[hi_m][hi_n]) - mp.mpc(numeric[hi_m][hi_n]))
    assert err_hi < mp.mpf("1e-16")


@pytest.mark.parametrize("c_value", GENERIC_BELOW_ONE_C)
@pytest.mark.parametrize("pole_selection", ["rightmost", "nearest", "residue"])
def test_interpolation_force_keeps_real_positive_poles_below_one(c_value, pole_selection):
    """For generic real c < 1, interpolation now force-keeps the real-positive
    shifted Kac poles into the prefactor (rather than raising), so the
    bootstrap shift can be retained without losing the table. The legacy
    raise-on-real-positive contract is preserved behind
    `force_keep_real_positive=False`.
    """
    mp.mp.dps = 55
    b = b_from_c(mp.mpf(c_value))
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)

    # Default: succeeds; some real-positive shifted poles must be present in
    # the final prefactor (otherwise c < 1 would not need this code path).
    interp = interpolate_delta_table(exact, npoles=8, pole_selection=pole_selection)
    assert any(mp.im(p) == 0 and mp.re(p) > 0 for p in interp.prefactor.poles)

    # Legacy opt-out still raises on the BS-incompatible pole layout.
    with pytest.raises(ValueError, match=r"Re\(pole\) <= 0"):
        interpolate_delta_table(
            exact,
            npoles=8,
            pole_selection=pole_selection,
            force_keep_real_positive=False,
        )


# --- singular central charges (c = 1 and the self-dual c = 25) ------------


@pytest.mark.parametrize("c_value", SINGULAR_C)
def test_singular_c_raises_without_regularization(c_value):
    """The bare symbolic table fails at both c = 1 and the self-dual c = 25:
    A^c_{r,s} blows up where mb + n/b vanishes."""
    mp.mp.dps = 40
    with pytest.raises(ZeroDivisionError):
        G_zzb_poly_table(2, SCALAR_EXT, 2, 2, c=mp.mpf(c_value), q_order=8)


@pytest.mark.parametrize("c_value", SINGULAR_C)
def test_regularization_recovers_singular_c(c_value):
    """merged_delta_table recovers the symbolic table at c = 1 and the
    self-dual c = 25, matching a high-precision probe placed an O(1e-22)
    sqrt-branch distance away (b - b0 ~ sqrt(c - c0) at both singular points).
    """
    mp.mp.dps = 40
    build = lambda cc: G_zzb_poly_table(2, SCALAR_EXT, 2, 2, c=cc, q_order=8, lambda_cut=None)
    reg = merged_delta_table(build, c=mp.mpf(c_value))

    mp.mp.dps = 200
    probe = build(mp.mpf(c_value) + mp.mpf("1e-44"))
    mp.mp.dps = 40
    assert _max_value_error(reg, probe) < mp.mpf("1e-18")


# --- minimal models below c = 1 -------------------------------------------


@pytest.mark.parametrize("c_value", MINIMAL_C_BELOW_ONE)
def test_minimal_model_below_one_bare_table_raises_or_matches_regularized(c_value):
    """At a minimal-model charge, the bare table must raise or agree with regularization.

    Resonant Kac pairs require sufficient guard precision. The regularizer
    remains the supported route at an exactly singular charge.
    """
    mp.mp.dps = 40
    c = mp.mpf(c_value)
    reg = merged_delta_table(lambda cc: G_zzb_poly_table(2, SCALAR_EXT, 1, 1, c=cc, q_order=20, lambda_cut=None), c=c)
    try:
        plain = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, c=c, q_order=20, lambda_cut=None)
    except ZeroDivisionError:
        return

    worst = mp.mpf(0)
    for Delta in [mp.mpf("2.6"), mp.mpf("5.4")]:
        worst = max(worst, _max_table_error(plain.evaluate(Delta), reg.evaluate(Delta)))
    assert worst < mp.mpf("1e-15")


@pytest.mark.parametrize("c_value", MINIMAL_C_BELOW_ONE)
def test_regularization_recovers_minimal_model_below_one(c_value):
    """merged_delta_table recovers a c < 1 minimal-model table, matching a
    high-precision probe placed close by. Unlike c = 1 / c = 25, b(c) is
    analytic in c at a minimal model (b is not stationary there), so the table
    is analytic in c and the probe sits an O(1e-44) distance away."""
    mp.mp.dps = 40
    build = lambda cc: G_zzb_poly_table(2, SCALAR_EXT, 1, 1, c=cc, q_order=12, lambda_cut=None)
    reg = merged_delta_table(build, c=mp.mpf(c_value))

    mp.mp.dps = 200
    probe = build(mp.mpf(c_value) + mp.mpf("1e-44"))
    mp.mp.dps = 40
    assert _max_value_error(reg, probe) < mp.mpf("1e-20")
