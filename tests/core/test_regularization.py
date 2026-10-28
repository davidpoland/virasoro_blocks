"""Tests for the epsilon-extrapolation regularization at singular c.

The bare Zamolodchikov recursion blows up at c = 1 and at minimal-model
c = 1 - 6/(m(m+1)); `F_block_regularized` / `H_q_series_regularized` recover
the physical (finite) block by analytic continuation in c.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.composer import B_block, G_block
from virasoro_blocks.params import b_from_c, d_rs
from virasoro_blocks.polyderivs import G_zzb_poly_table
from virasoro_blocks.recursion import F_block, H_q_series
from virasoro_blocks.regularization import (
    B_block_regularized,
    F_block_regularized,
    G_block_regularized,
    H_q_series_regularized,
    merged_delta_table,
    regularized_numeric_grids,
)
from virasoro_blocks.polyderivs import build_numeric_sampler
def _extrapolate_to_zero(xs, ys):
    """Lagrange extrapolation of the samples ``(xs, ys)`` to ``x = 0``."""
    total = mp.mpc(0)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        weight = mp.mpf(1)
        for j, xj in enumerate(xs):
            if j != i:
                weight *= xj / (xj - xi)
        total += weight * yi
    return total


def _independent_regularized_H_q_series(h, d, c, q_order: int, *, n_points: int = 5):
    eps0 = mp.mpf("1e-8")
    epsilons = [eps0 / (2**k) for k in range(n_points)]
    return [
        _extrapolate_to_zero(
            epsilons,
            [H_q_series(h, d, c=mp.mpc(c) + eps, q_order=q_order)[k] for eps in epsilons],
        )
        for k in range(q_order + 1)
    ]


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def _root_multiset_counts(roots):
    out = {}
    for root in roots:
        key = round(float(mp.re(mp.mpc(root))), 12)
        out[key] = out.get(key, 0) + 1
    return out


def test_regularized_matches_plain_at_generic_c():
    """At a generic (non-singular) c the regularized block equals F_block:
    extrapolating an analytic function through nearby points is exact."""
    mp.mp.dps = 40
    args = (mp.mpf("1.0"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4)
    plain = F_block(*args, c=mp.mpf("5"), q_order=8)
    reg = F_block_regularized(*args, c=mp.mpf("5"), q_order=8)
    assert _close(reg, plain, tol=mp.mpf("1e-35"))


def test_plain_F_block_raises_at_c_equals_one():
    """The legacy bare recursion still fails at c = 1 when auto-regularization is disabled."""
    mp.mp.dps = 40
    from virasoro_blocks.singularity import SingularCentralChargeError

    with pytest.raises(SingularCentralChargeError):     # (a ZeroDivisionError subclass)
        F_block(mp.mpf("1.0"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4,
                c=mp.mpf("1"), q_order=8, regularize="off")


def test_F_block_auto_regularizes_at_c_equals_one():
    """The public F_block path defaults to auto-regularization at recognized singular c."""
    mp.mp.dps = 40
    args = (mp.mpf("1.0"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4)
    auto = F_block(*args, c=mp.mpf("1"), q_order=8)
    reg = F_block_regularized(*args, c=mp.mpf("1"), q_order=8)
    assert _close(auto, reg, tol=mp.mpf("1e-30"))


def test_regularized_at_c_equals_one_q_converges():
    """The regularized c=1 block converges as the q-truncation grows."""
    mp.mp.dps = 40
    args = (mp.mpf("1.0"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4)
    r16 = F_block_regularized(*args, c=mp.mpf("1"), q_order=16)
    r24 = F_block_regularized(*args, c=mp.mpf("1"), q_order=24)
    # q(0.4) ~ 0.025, so q^16 ~ 1e-26; the two truncations agree well.
    assert _close(r16, r24, tol=mp.mpf("1e-20"))


def test_regularized_at_c_equals_one_matches_epsilon_limit():
    """The regularized c=1 value matches an independent direct evaluation at
    c = 1 + tiny epsilon (computed at high precision). The regularizer
    extrapolates the O(epsilon) term away, so it agrees with the direct probe
    to better than the probe's own O(epsilon) offset.  (h = 0.9, off the c = 1
    Kac weights (r - s)^2/4: on one the regularized value is the limit in h,
    not this fixed-h one; see test_c1_kac_weight_is_the_limit_in_h.)"""
    mp.mp.dps = 40
    args = (mp.mpf("0.9"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4)
    reg = F_block_regularized(*args, c=mp.mpf("1"), q_order=16)

    mp.mp.dps = 140
    probe = F_block(
        mp.mpf("0.9"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4,
        c=mp.mpf(1) + mp.mpf("1e-25"), q_order=16,
    )
    mp.mp.dps = 40
    # Direct probe carries an O(1e-25) offset from sitting at c = 1 + 1e-25;
    # the regularized value is the epsilon -> 0 limit, so they agree to
    # within that offset.
    assert _close(reg, probe, tol=mp.mpf("1e-23"))


def test_regularized_minimal_model_is_not_garbage():
    """At the Ising c = 1/2 with q_order = 12 the legacy bare F_block is unusable,
    while the public auto-regularized value is finite and O(1)."""
    mp.mp.dps = 40
    c = mp.mpf(1) / 2
    args = (mp.mpf("0.3"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4)
    try:
        bare = F_block(*args, c=c, q_order=12, regularize="off")
        bare_failed = abs(bare) > mp.mpf("1e10")
    except ZeroDivisionError:
        bare_failed = True
    assert bare_failed, "expected bare F_block to be unusable at c=1/2, q_order=12"

    auto = F_block(*args, c=c, q_order=12)
    reg = F_block_regularized(*args, c=c, q_order=12)
    assert _close(auto, reg, tol=mp.mpf("1e-30"))
    assert abs(reg) < mp.mpf("10")


@pytest.fixture(scope="module")
def ising_sigma_scalar_zzb():
    """The regularized q_order = 30 scalar G zzb table for Ising sigma externals
    at c = 1/2 (m = n = 3), plus the direct nearby-c table at c + 1e-25
    (m = n = 2), each built once: neither depends on the Delta probed."""
    q_order = 30
    hext = [mp.mpf("1/16")] * 4   # Ising sigma: Delta_ext = 1/8
    c = mp.mpf(1) / 2
    with mp.workdps(70):
        reg = merged_delta_table(lambda cv: G_zzb_poly_table(0, hext, 3, 3, c=cv, q_order=q_order), c=c)
        near = G_zzb_poly_table(0, hext, 2, 2, c=c + mp.mpf("1e-25"), q_order=q_order)
    return {"q_order": q_order, "hext": hext, "c": c, "reg": reg, "near": near}


def test_regularized_numeric_grids_matches_symbolic_and_limit(ising_sigma_scalar_zzb):
    """The value-space regularizer (regularized_numeric_grids: extrapolate the
    numeric-recursion grid in c, no symbolic table, no pole strip) agrees with
    both the fixed symbolic table path and the nearby non-singular block, for
    scalar sigma externals at the Ising c = 1/2."""
    mp.mp.dps = 60
    fx = ising_sigma_scalar_zzb
    hext, c, q_order = fx["hext"], fx["c"], fx["q_order"]
    ell = 0
    kept = [(0, 0), (1, 0), (1, 1), (2, 0)]

    def build_sampler(c_val):
        return build_numeric_sampler(
            "zzb", chiral=False, spinning=False, projected=False,
            h_externals=hext, hbar_externals=hext, ell=ell, m_max=2, n_max=2,
            b=None, c=c_val, q_order=q_order, chiral_norm=None, z_shift=None,
        )

    deltas = (mp.mpf("1.1"), mp.mpf("13.7"))
    grids = regularized_numeric_grids(build_sampler, list(deltas), c=c, entries=kept)
    for D, grid in zip(deltas, grids):
        sym = fx["reg"].evaluate(D)
        ref = fx["near"].evaluate(D)
        for (m, n) in kept:
            assert abs(grid[m][n] - sym[m][n]) < mp.mpf(10) ** -30, (D, m, n)
            assert abs(grid[m][n] - ref[m][n]) < mp.mpf(10) ** -12, (D, m, n)


def test_regularized_numeric_grids_raises_on_genuine_pole():
    """A genuine (non-removable) pole -- exchanged Delta = 1 at c = 1/2 with
    Delta_ext = 0 -- makes the extrapolation diverge, so the value-space
    regularizer raises rather than emitting garbage."""
    mp.mp.dps = 50
    h0 = [mp.mpf(0)] * 4

    def build_sampler(c_val):
        return build_numeric_sampler(
            "zzb", chiral=False, spinning=False, projected=False,
            h_externals=h0, hbar_externals=h0, ell=0, m_max=0, n_max=0,
            b=None, c=c_val, q_order=20, chiral_norm=None, z_shift=None,
        )

    with pytest.raises(ValueError, match="did not converge"):
        regularized_numeric_grids(
            build_sampler, [mp.mpf("1.0")], c=mp.mpf(1) / 2, entries=[(0, 0)],
            work_dps=200, max_work_dps=200,
        )


def test_regularized_minimal_model_kac_externals_q_converges():
    """Ising <sigma sigma sigma sigma>: regularized block converges in q."""
    mp.mp.dps = 40
    c = mp.mpf(1) / 2
    b = b_from_c(c)
    sigma = d_rs(1, 2, b)  # chiral weight of the Ising spin field
    args = (mp.mpf("0.3"), mp.mpf("0.4"), [sigma] * 4)
    r12 = F_block_regularized(*args, c=c, q_order=12)
    r20 = F_block_regularized(*args, c=c, q_order=20)
    assert _close(r12, r20, tol=mp.mpf("1e-15"))


def test_regularized_nonunitary_minimal_identity_channel_converges():
    """Yang-Lee <Phi Phi Phi Phi> still has a finite identity block.

    At c = -22/5 the bare recursion becomes singular once q_order reaches the
    resonant terms, but in the physical Phi x Phi -> 1 channel the c-singularity
    is removable and the regularized value is stable.
    """
    mp.mp.dps = 50
    c = Fraction(-22, 5)
    # The null state decouples at exactly -1/5; an mpf division instead
    # supplies a nearby binary external label with a nonzero pole residue.
    phi = Fraction(-1, 5)
    args = (mp.mpf(0), mp.mpf("0.2"), [phi] * 4)
    r12 = F_block_regularized(*args, c=c, q_order=12)
    r20 = F_block_regularized(*args, c=c, q_order=20)
    assert _close(r12, r20, tol=mp.mpf("1e-15"))


def test_regularized_reports_likely_pole_for_singular_identity_channel():
    """Not every h = 0 block at a minimal-model c has a removable c-singularity.

    For generic external weights at the Ising c = 1/2, the identity exchange has
    a genuine pole (h = 0 = h_{1,1} is a Kac weight whose residue survives), so
    the regularizer must fail cleanly instead of returning a bogus finite
    answer, and the failure message names the pole and its order.
    """
    mp.mp.dps = 50
    with pytest.raises(ValueError, match="genuine pole") as excinfo:
        F_block_regularized(
            mp.mpf(0),
            mp.mpf("0.2"),
            [mp.mpf("0.2")] * 4,
            c=mp.mpf(1) / 2,
            q_order=12,
        )
    assert "of order 1" in str(excinfo.value)


def test_H_q_series_regularized_matches_plain_at_generic_c():
    mp.mp.dps = 40
    plain = H_q_series(mp.mpf("1.0"), [mp.mpf("0.1")] * 4, c=mp.mpf("5"), q_order=6)
    reg = H_q_series_regularized(
        mp.mpf("1.0"), [mp.mpf("0.1")] * 4, c=mp.mpf("5"), q_order=6
    )
    assert len(reg) == len(plain) == 7
    for a, b_ in zip(reg, plain):
        assert _close(a, b_, tol=mp.mpf("1e-35"))


def test_H_q_series_regularized_at_c_equals_one():
    """H_0 = 1 always; the rest are finite at c = 1."""
    mp.mp.dps = 40
    H = H_q_series_regularized(
        mp.mpf("1.3"), [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")],
        c=mp.mpf("1"), q_order=5,
    )
    assert _close(H[0], mp.mpc(1), tol=mp.mpf("1e-30"))
    assert all(abs(coeff) < mp.mpf("1e6") for coeff in H)


@pytest.mark.parametrize(
    "c_text,tol",
    [
        ("25", mp.mpf("1e-18")),
        ("-4.4", mp.mpf("1e-16")),
        ("28", mp.mpf("1e-18")),
    ],
)
def test_H_q_series_regularized_matches_independent_extrapolation_on_broader_singular_loci(c_text, tol):
    mp.mp.dps = 60
    c_value = mp.mpf(c_text)
    h = mp.mpf("1.3")
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]

    regularized = H_q_series_regularized(
        h,
        d,
        c=c_value,
        q_order=4,
    )
    extrapolated = _independent_regularized_H_q_series(h, d, c_value, 4)

    assert len(regularized) == len(extrapolated) == 5
    for got, expected in zip(regularized, extrapolated):
        assert abs(got - expected) < tol


def test_regularized_validates_inputs():
    args = (mp.mpf("1.0"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4)
    with pytest.raises(ValueError):
        F_block_regularized(*args, c=mp.mpf("5"), q_order=-1)
    threaded = F_block_regularized(*args, b=mp.mpf("2"), c=mp.mpf("5"), q_order=8)
    plain_c = F_block_regularized(*args, c=mp.mpf("5"), q_order=8)
    assert _close(threaded, plain_c, tol=mp.mpf("1e-35"))
    with pytest.raises(ValueError):
        F_block_regularized(*args, q_order=8)  # neither b nor c


# --- regularized 2d blocks (B and G) ---


def test_B_block_regularized_matches_plain_at_generic_c():
    """At a generic c, B_block_regularized equals B_block."""
    mp.mp.dps = 40
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    args = (mp.mpf("1.4"), mp.mpf("1.1"), mp.mpf("0.4"), mp.mpf("0.35"),
            h_ext, hbar_ext)
    plain = B_block(*args, c=mp.mpf("5"), q_order=8)
    reg = B_block_regularized(*args, c=mp.mpf("5"), q_order=8)
    assert _close(reg, plain, tol=mp.mpf("1e-35"))


def test_G_block_regularized_matches_plain_at_generic_c():
    """At a generic c, G_block_regularized equals G_block."""
    mp.mp.dps = 40
    h_ext = [mp.mpf("0.15")] * 4
    args = (mp.mpf("3.1"), 2, mp.mpf("0.4"), mp.mpf("0.35"), h_ext)
    plain = G_block(*args, c=mp.mpf("5"), q_order=8)
    reg = G_block_regularized(*args, c=mp.mpf("5"), q_order=8)
    assert _close(reg, plain, tol=mp.mpf("1e-35"))


def test_B_block_regularized_at_c1_factorizes():
    """At c = 1, B_block_regularized equals the product of the two
    independently regularized chiral factors F(h, z) and F(hbar, zbar)."""
    mp.mp.dps = 40
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h, hbar = mp.mpf("1.4"), mp.mpf("1.1")
    z, zbar = mp.mpf("0.4"), mp.mpf("0.35")

    B_reg = B_block_regularized(
        h, hbar, z, zbar, h_ext, hbar_ext, c=mp.mpf("1"), q_order=10
    )
    F_h = F_block_regularized(h, z, h_ext, c=mp.mpf("1"), q_order=10)
    F_hbar = F_block_regularized(hbar, zbar, hbar_ext, c=mp.mpf("1"), q_order=10)
    assert _close(B_reg, F_h * F_hbar, tol=mp.mpf("1e-25"))


def test_B_block_regularized_factorizes_at_c25():
    mp.mp.dps = 50
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h, hbar = mp.mpf("1.4"), mp.mpf("1.1")
    z, zbar = mp.mpf("0.4"), mp.mpf("0.35")

    B_reg = B_block_regularized(
        h,
        hbar,
        z,
        zbar,
        h_ext,
        hbar_ext,
        c=mp.mpf("25"),
        q_order=4,
    )
    F_h = F_block_regularized(
        h,
        z,
        h_ext,
        c=mp.mpf("25"),
        q_order=4,
    )
    F_hbar = F_block_regularized(
        hbar,
        zbar,
        hbar_ext,
        c=mp.mpf("25"),
        q_order=4,
    )
    assert _close(B_reg, F_h * F_hbar, tol=mp.mpf("1e-20"))


def test_B_block_regularized_reports_likely_pole_for_singular_identity_channel():
    mp.mp.dps = 50
    with pytest.raises(ValueError, match="genuine pole"):
        B_block_regularized(
            mp.mpf(0),
            mp.mpf(0),
            mp.mpf("0.2"),
            mp.mpf("0.25"),
            [mp.mpf("0.2")] * 4,
            [mp.mpf("0.2")] * 4,
            c=mp.mpf(1) / 2,
            q_order=12,
        )


def test_G_block_regularized_at_c1_q_converges():
    """The regularized c=1 scalar block G converges as q-truncation grows."""
    mp.mp.dps = 40
    h_ext = [mp.mpf("0.15")] * 4
    args = (mp.mpf("3.1"), 2, mp.mpf("0.4"), mp.mpf("0.35"), h_ext)
    g16 = G_block_regularized(*args, c=mp.mpf("1"), q_order=16)
    g24 = G_block_regularized(*args, c=mp.mpf("1"), q_order=24)
    assert _close(g16, g24, tol=mp.mpf("1e-20"))


def test_G_block_regularized_at_c1_parity():
    """At c = 1, G_{Delta, ell}(zbar, z) = G_{Delta, ell}(z, zbar) at every spin (z <-> zbar even)."""
    mp.mp.dps = 40
    h_ext = [mp.mpf("0.15")] * 4
    Delta = mp.mpf("3.1")
    z, zbar = mp.mpf("0.45"), mp.mpf("0.4")
    for ell in (0, 1, 2, 3):
        g_zzb = G_block_regularized(
            Delta, ell, z, zbar, h_ext, c=mp.mpf("1"), q_order=10
        )
        g_zbz = G_block_regularized(
            Delta, ell, zbar, z, h_ext, c=mp.mpf("1"), q_order=10
        )
        assert _close(g_zbz, g_zzb, tol=mp.mpf("1e-25"))


def test_G_block_regularized_at_c1_matches_epsilon_limit():
    """The regularized c=1 scalar block matches a direct c = 1 + tiny probe."""
    mp.mp.dps = 40
    h_ext = [mp.mpf("0.15")] * 4
    args = (mp.mpf("3.1"), 2, mp.mpf("0.4"), mp.mpf("0.35"), h_ext)
    reg = G_block_regularized(*args, c=mp.mpf("1"), q_order=12)

    mp.mp.dps = 140
    probe = G_block(*args, c=mp.mpf(1) + mp.mpf("1e-25"), q_order=12)
    mp.mp.dps = 40
    assert _close(reg, probe, tol=mp.mpf("1e-23"))


def test_G_block_regularized_reports_likely_pole_for_singular_identity_channel():
    mp.mp.dps = 50
    with pytest.raises(ValueError, match="genuine pole"):
        G_block_regularized(
            mp.mpf(0),
            0,
            mp.mpf("0.2"),
            mp.mpf("0.25"),
            [mp.mpf("0.2")] * 4,
            c=mp.mpf(1) / 2,
            q_order=12,
        )


# --- regularized symbolic-in-Delta derivative tables ---


def _max_table_value_error(table_a, table_b):
    """Worst absolute discrepancy of two evaluated (numeric) tables."""
    worst = mp.mpf(0)
    for row_a, row_b in zip(table_a, table_b):
        for entry_a, entry_b in zip(row_a, row_b):
            worst = max(worst, abs(mp.mpc(entry_a) - mp.mpc(entry_b)))
    return worst


def _max_poly_rel_error(poly_a, poly_b):
    """Worst coefficient discrepancy of two polynomials, normalized by the
    dominant coefficient of the first."""
    scale = max(abs(coeff) for coeff in poly_a.coeffs)
    if scale == 0:
        return mp.mpf(0)
    worst = mp.mpf(0)
    for coeff_a, coeff_b in zip(poly_a.coeffs, poly_b.coeffs):
        worst = max(worst, abs(coeff_a - coeff_b) / scale)
    return worst


def _max_table_component_error(table_a, table_b):
    """Worst relative discrepancy of two symbolic DeltaDerivTables, compared
    component-wise: exp_base, the Delta-poles, the denominator polynomial, and
    the numerator polynomials (each polynomial normalized by its own dominant
    coefficient).

    Comparing the symbolic components is the numerically clean check: the
    reconstructed `evaluate` cancels a high-degree polynomial with O(1e20)
    coefficients down to an O(1) block value, losing many digits.
    """
    worst = mp.mpf(0)
    exp_a, exp_b = mp.mpc(table_a.exp_base), mp.mpc(table_b.exp_base)
    if abs(exp_a) > 0:
        worst = max(worst, abs(exp_a - exp_b) / abs(exp_a))
    # Poles are normalized by the largest pole magnitude (a pole near Delta = 0
    # carries only that overall absolute accuracy).
    root_scale = max(abs(mp.mpc(r)) for r in table_a.denominator_roots)
    for root_a, root_b in zip(
        table_a.denominator_roots, table_b.denominator_roots
    ):
        if root_scale > 0:
            worst = max(
                worst, abs(mp.mpc(root_a) - mp.mpc(root_b)) / root_scale
            )
    worst = max(worst, _max_poly_rel_error(table_a.denominator, table_b.denominator))
    for row_a, row_b in zip(table_a.numerators, table_b.numerators):
        for poly_a, poly_b in zip(row_a, row_b):
            worst = max(worst, _max_poly_rel_error(poly_a, poly_b))
    return worst


def test_plain_poly_table_raises_at_c_equals_one():
    """Sanity: the bare symbolic table really does fail at c = 1."""
    mp.mp.dps = 40
    with pytest.raises(ZeroDivisionError):
        G_zzb_poly_table(2, [mp.mpf("0.15")] * 4, 1, c=mp.mpf("1"), q_order=8)


# --- Laurent value-oracle cross-check for merged_delta_table ---

