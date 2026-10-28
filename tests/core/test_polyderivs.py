"""Cross-checks for the symbolic-in-Delta derivative tables.

The symbolic table, evaluated at a numeric Delta, must reproduce the numeric
`derivs.py` tables (which `mp.taylor` the recursion directly).
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.coordinates import z_symmetric
from virasoro_blocks.derivs import (
    G_qqb_table,
    G_qqb_ws_radial_table,
    G_qqb_ws_table,
    G_ws_radial_table,
    G_ws_table,
    G_xt_radial_table,
    G_xt_table,
    G_yyb_table,
    G_zzb_table,
    _fold_parity,
    chiral_z_taylor,
)
from virasoro_blocks.hseries import HSeries, exp_h_times
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyderivs import (
    G_qqb_poly_table,
    G_qqb_ws_poly_table,
    G_qqb_ws_radial_poly_table,
    G_ws_poly_table,
    G_ws_radial_poly_table,
    G_xt_poly_table,
    G_xt_radial_poly_table,
    G_yyb_poly_table,
    G_zzb_poly_table,
    InterpolatedDeltaTable,
    _NumericBlockSampler,
    _NumericOrderedBlockSampler,
    _numeric_residue_scores,
    _residue_scores,
    numeric_residue_scoring_available,
    B_qqb_poly_table,
    B_yyb_poly_table,
    B_zzb_poly_table,
    parity_projected_B_qqb_poly_table,
    parity_projected_B_zzb_poly_table,
    _build_B_poly_table,
    _build_F_chiral_poly_table,
    _build_G_poly_table,
    _default_interpolation_shift,
    build_chiral_cache,
    build_table_from_cache,
    chiral_symbolic_useries,
    interpolate_delta_table,
    interpolate_delta_tables,
    parity_projected_B_qqb_ws_poly_table,
    parity_projected_B_yyb_poly_table,
    parity_projected_B_ws_poly_table,
    parity_projected_B_ws_radial_poly_table,
    parity_projected_B_xt_poly_table,
)


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def _assert_tables_close(symbolic_table, numeric_table, tol):
    assert len(symbolic_table) == len(numeric_table)
    for row_s, row_n in zip(symbolic_table, numeric_table):
        assert len(row_s) == len(row_n)
        for entry_s, entry_n in zip(row_s, row_n):
            assert _close(entry_s, entry_n, tol)


C = mp.mpf("1.7")
D_EXT = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
SCALAR_EXT = [mp.mpf("0.15")] * 4


_FOLDED_PROJECTED_BUILDERS = {
    "ws": parity_projected_B_ws_poly_table,
    "xt": parity_projected_B_xt_poly_table,
    "qqb_ws": parity_projected_B_qqb_ws_poly_table,
}

_FOLDED_G_BUILDERS = {
    "ws": G_ws_poly_table,
    "xt": G_xt_poly_table,
    "qqb_ws": G_qqb_ws_poly_table,
}


def test_chiral_cache_accepts_coordinates_from_an_iterator():
    coordinates = ['zzb', 'yyb', 'xt']
    options = dict(h_externals=['.1']*4, lambda_=1, c=32, q_order=2, skeleton=True)
    expected = build_chiral_cache(coordinates, **options)
    actual = build_chiral_cache(iter(coordinates), **options)
    assert set(actual) == set(coordinates)
    for coordinate in coordinates:
        chiral, antichiral = actual[coordinate]
        reference, _ = expected[coordinate]
        assert antichiral is None
        assert chiral.numerator_degrees == reference.numerator_degrees
        assert chiral.denominator_roots == reference.denominator_roots
        assert chiral.q_order == reference.q_order
    assert actual['zzb'][0] is actual['xt'][0]


# --- HSeries / exp_h_times unit checks ---


def test_hseries_orders_follow_the_shared_integer_control_contract():
    source = HSeries.from_numeric([1, 2, 3])
    for invalid in (True, False, 0, -1, 2.5, Fraction(3) + Fraction(1, 10**70)):
        with pytest.raises(ValueError, match='order'):
            source.truncate(invalid)
        with pytest.raises(ValueError, match='order'):
            exp_h_times([0, 1], invalid)
    expected = exp_h_times([0, 1], 3)
    for integral in (3.0, Fraction(3), mp.mpf(3)):
        assert source.truncate(integral).coeffs == source.coeffs
        assert [poly.coeffs for poly in exp_h_times([0, 1], integral).coeffs] == [
            poly.coeffs for poly in expected.coeffs]
    padded = source.truncate(5.0)
    assert padded.order == 5 and all(poly.is_zero for poly in padded.coeffs[3:])


def test_exp_h_times_low_order_coefficients():
    """exp(h s(u)) with s = s1 u + s2 u^2 + ...: G_0 = 1, G_1 = h s1,
    G_2 = h s2 + (h s1)^2 / 2."""
    mp.mp.dps = 40
    s1, s2 = mp.mpf("0.7"), mp.mpf("-0.4")
    series = exp_h_times([mp.mpf(0), s1, s2], 3)
    h = mp.mpf("1.3")
    assert _close(series.coeffs[0].evaluate(h), mp.mpf(1), tol=mp.mpf("1e-35"))
    assert _close(series.coeffs[1].evaluate(h), h * s1, tol=mp.mpf("1e-35"))
    assert _close(
        series.coeffs[2].evaluate(h),
        h * s2 + (h * s1) ** 2 / 2,
        tol=mp.mpf("1e-35"),
    )


def test_hseries_multiplication_matches_explicit_cauchy_product():
    mp.mp.dps = 40
    a = HSeries([
        Polynomial([1, 2], field=mp.mpf),
        Polynomial([3], field=mp.mpf),
    ])
    b = HSeries([
        Polynomial([5], field=mp.mpf),
        Polynomial([7, 11], field=mp.mpf),
    ])

    prod = a * b

    expected = [
        Polynomial([5, 10], field=mp.mpf),
        Polynomial([22, 25, 22], field=mp.mpf),
        Polynomial([21, 33], field=mp.mpf),
    ]
    assert len(prod.coeffs) == len(expected)
    for got, want in zip(prod.coeffs, expected):
        assert got == want


# --- chiral symbolic series ---


@pytest.mark.parametrize("q_order", [4, 8])
def test_chiral_symbolic_matches_numeric_taylor(q_order):
    """ChiralSymbolic.chiral_coeff(m, h) matches the numeric chiral_z_taylor."""
    mp.mp.dps = 50
    b = b_from_c(C)
    n_terms = 6
    cs = chiral_symbolic_useries(
        D_EXT,
        n_terms,
        b=b,
        c=C,
        z_of_u=lambda u: u,
        u_symmetric=z_symmetric(),
        q_order=q_order,
    )
    for h in [mp.mpf("0.6"), mp.mpf("1.4"), mp.mpf("3.2")]:
        numeric = chiral_z_taylor(h, D_EXT, n_terms, b=b, q_order=q_order)
        for m in range(n_terms):
            assert _close(cs.chiral_coeff(m, h), numeric[m], tol=mp.mpf("1e-28"))


# --- non-folded G tables ---


@pytest.mark.parametrize("ell", [0, 1, 2, 3])
def test_G_zzb_poly_table_matches_numeric(ell):
    mp.mp.dps = 50
    b = b_from_c(C)
    Delta = mp.mpf("3.1")
    m_max, n_max = 3, 3
    sym = G_zzb_poly_table(ell, SCALAR_EXT, m_max, n_max, b=b, q_order=8, lambda_cut=None)
    numeric = G_zzb_table(Delta, ell, SCALAR_EXT, m_max, n_max, b=b, q_order=8)
    _assert_tables_close(sym.evaluate(Delta), numeric, tol=mp.mpf("1e-25"))


@pytest.mark.parametrize("m_max", [12])
def test_G_zzb_poly_table_matches_numeric_high_order(m_max):
    """Regression: flint's acb_series truncates at the global ``ctx.cap``
    (default 10), not at any per-object kwarg. The symbolic table must raise the
    cap to n_terms; otherwise every coefficient of order >= 10 is silently
    dropped and the symbolic blocks diverge from the numeric ones by many orders
    of magnitude for m_max >= 11 (Lambda >= 11)."""
    mp.mp.dps = 60
    b = b_from_c(C)
    Delta = mp.mpf("3.1")
    sym = G_zzb_poly_table(0, SCALAR_EXT, m_max, m_max, b=b, q_order=m_max + 6, lambda_cut=None)
    numeric = G_zzb_table(Delta, 0, SCALAR_EXT, m_max, m_max, b=b, q_order=m_max + 6)
    _assert_tables_close(sym.evaluate(Delta), numeric, tol=mp.mpf("1e-30"))


@pytest.mark.parametrize("ell", [0, 2])
def test_G_yyb_poly_table_matches_numeric(ell):
    mp.mp.dps = 50
    b = b_from_c(C)
    Delta = mp.mpf("2.7")
    sym = G_yyb_poly_table(ell, SCALAR_EXT, 3, 3, b=b, q_order=8, lambda_cut=None)
    numeric = G_yyb_table(Delta, ell, SCALAR_EXT, 3, 3, b=b, q_order=8)
    _assert_tables_close(sym.evaluate(Delta), numeric, tol=mp.mpf("1e-22"))


@pytest.mark.parametrize("ell", [0, 1, 2])
def test_G_qqb_poly_table_matches_numeric(ell):
    mp.mp.dps = 50
    b = b_from_c(C)
    Delta = mp.mpf("4.2")
    sym = G_qqb_poly_table(ell, SCALAR_EXT, 3, 3, b=b, q_order=8, lambda_cut=None)
    numeric = G_qqb_table(Delta, ell, SCALAR_EXT, 3, 3, b=b, q_order=8)
    _assert_tables_close(sym.evaluate(Delta), numeric, tol=mp.mpf("1e-22"))


def test_G_qqb_poly_table_uses_flint(monkeypatch):
    """The q (nome) coordinate rides the flint acb_series fast path: z(q) =
    (theta2/theta3)^4 is a clean integer power series, so the q-coordinate no
    longer falls back to mp.taylor. Guards against silent regression."""
    pytest.importorskip("flint")
    import virasoro_blocks.polyderivs as P

    mp.mp.dps = 50
    b = b_from_c(C)
    counts = {"flint": 0, "fallback": 0}
    orig = P._flint_taylor_series

    def wrapped(*args, **kwargs):
        result = orig(*args, **kwargs)
        counts["flint" if result is not None else "fallback"] += 1
        return result

    monkeypatch.setattr(P, "_flint_taylor_series", wrapped)
    P.G_qqb_poly_table(0, SCALAR_EXT, 3, 3, b=b, q_order=8, lambda_cut=None)

    assert counts["flint"] >= 1
    assert counts["fallback"] == 0


def test_G_poly_table_is_delta_independent_object():
    """One symbolic table reproduces the numeric table at several Delta."""
    mp.mp.dps = 50
    b = b_from_c(C)
    sym = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    for Delta in [mp.mpf("2.3"), mp.mpf("3.8"), mp.mpf("6.0")]:
        numeric = G_zzb_table(Delta, 2, SCALAR_EXT, 2, 2, b=b, q_order=8)
        _assert_tables_close(sym.evaluate(Delta), numeric, tol=mp.mpf("1e-24"))


def test_G_zzb_poly_table_large_q_real_b_uses_guard_digits():
    """The large-c exact symbolic table stays aligned with the direct numeric
    table once the high-q guard digits are applied internally.
    """
    mp.mp.dps = 150
    c = mp.mpf("10000")
    h_ext = [mp.mpf(1) / 16] * 4
    Delta = mp.mpf("10")
    sym = G_zzb_poly_table(
        4,
        h_ext,
        3,
        3,
        c=c,
        q_order=50,
        z_shift=-2 * h_ext[0],
        lambda_cut=None,  # full-grid comparison against the numeric table
    )
    numeric = G_zzb_table(
        Delta,
        4,
        h_ext,
        3,
        3,
        c=c,
        q_order=50,
        z_shift=-2 * h_ext[0],
    )

    worst = mp.mpf(0)
    for row_s, row_n in zip(sym.evaluate(Delta), numeric):
        for entry_s, entry_n in zip(row_s, row_n):
            denom = max(abs(mp.mpc(entry_n)), mp.mpf("1e-80"))
            worst = max(worst, abs(mp.mpc(entry_s) - mp.mpc(entry_n)) / denom)

    assert worst < mp.mpf("1e-22")


# --- parity-folded G tables ---


@pytest.mark.parametrize("ell", [0, 1, 2, 3])
def test_G_ws_poly_table_matches_numeric(ell):
    mp.mp.dps = 60
    b = b_from_c(C)
    Delta = mp.mpf("3.1")
    sym = G_ws_poly_table(ell, SCALAR_EXT, 2, 2, b=b, q_order=8)
    numeric = G_ws_table(Delta, ell, SCALAR_EXT, 2, 2, b=b, q_order=8)
    _assert_tables_close(sym.evaluate(Delta), numeric, tol=mp.mpf("1e-20"))


@pytest.mark.parametrize("ell", [0, 2])
def test_G_xt_poly_table_matches_numeric(ell):
    mp.mp.dps = 60
    b = b_from_c(C)
    Delta = mp.mpf("2.7")
    sym = G_xt_poly_table(ell, SCALAR_EXT, 2, 2, b=b, q_order=8)
    numeric = G_xt_table(Delta, ell, SCALAR_EXT, 2, 2, b=b, q_order=8)
    _assert_tables_close(sym.evaluate(Delta), numeric, tol=mp.mpf("1e-18"))


@pytest.mark.parametrize("ell", [0, 1, 3])
def test_G_qqb_ws_poly_table_matches_numeric(ell):
    mp.mp.dps = 60
    b = b_from_c(C)
    Delta = mp.mpf("4.2")
    sym = G_qqb_ws_poly_table(ell, SCALAR_EXT, 2, 2, b=b, q_order=8)
    numeric = G_qqb_ws_table(Delta, ell, SCALAR_EXT, 2, 2, b=b, q_order=8)
    _assert_tables_close(sym.evaluate(Delta), numeric, tol=mp.mpf("1e-18"))


@pytest.mark.parametrize(
    "coordinate,ell,parity",
    [("ws", 0, "+"), ("xt", 2, "+"), ("qqb_ws", 3, "+")],
)
def test_parity_projected_folded_scalar_limit_recovers_G(coordinate, ell, parity):
    mp.mp.dps = 60
    b = b_from_c(C)
    Delta = mp.mpf("3.3")
    projected_builder = _FOLDED_PROJECTED_BUILDERS[coordinate]
    g_builder = _FOLDED_G_BUILDERS[coordinate]
    projected = projected_builder(
        ell,
        SCALAR_EXT,
        SCALAR_EXT,
        2,
        2,
        parity=parity,
        b=b,
        q_order=8,
    )
    g_table = g_builder(ell, SCALAR_EXT, 2, 2, b=b, q_order=8)
    g_values = g_table.evaluate(Delta)
    if ell == 0:
        # B + P[B] = 2 B(h, h) at spin 0, where G is B(h, h) itself
        g_values = [[2 * entry for entry in row] for row in g_values]
    _assert_tables_close(projected.evaluate(Delta), g_values, tol=mp.mpf("1e-20"))


def test_parity_projected_folded_phase_metadata_changes_sign():
    mp.mp.dps = 60
    b = b_from_c(C)
    Delta = mp.mpf("3.3")
    plus = parity_projected_B_ws_poly_table(
        mp.mpf("1.5"),
        D_EXT,
        [d + mp.mpf("0.01") for d in D_EXT],
        2,
        2,
        parity="+",
        b=b,
        q_order=8,
    ).evaluate(Delta)
    flipped = parity_projected_B_ws_poly_table(
        mp.mpf("1.5"),
        D_EXT,
        [d + mp.mpf("0.01") for d in D_EXT],
        2,
        2,
        parity="+",
        parity_external=["1", "1", "1", "1"],
        exchanged_parity="-1",
        b=b,
        q_order=8,
    ).evaluate(Delta)
    assert any(
        abs(mp.mpc(a) - mp.mpc(bv)) > mp.mpf("1e-24")
        for row_a, row_b in zip(plus, flipped)
        for a, bv in zip(row_a, row_b)
    )


@pytest.mark.parametrize("coordinate", ["ws", "xt", "qqb_ws"])
def test_parity_projected_folded_rejects_non_real_phase(coordinate):
    """A non-real total parity phase (odd number of i / -i parities) has no
    definite z<->zbar parity, so the parity-folded builders must refuse it."""
    mp.mp.dps = 50
    b = b_from_c(C)
    with pytest.raises(ValueError):
        _FOLDED_PROJECTED_BUILDERS[coordinate](
            1,
            D_EXT,
            D_EXT,
            1,
            1,
            parity="+",
            parity_external=["i", "1", "1", "1"],
            b=b,
            q_order=6,
        )


def test_parity_projected_folded_accepts_real_phase_from_paired_i():
    """An even number of i / -i parities multiplies to a real phase, which a
    parity-folded builder accepts."""
    mp.mp.dps = 50
    b = b_from_c(C)
    table = parity_projected_B_ws_poly_table(
        1,
        D_EXT,
        D_EXT,
        1,
        1,
        parity="+",
        parity_external=["i", "i", "1", "1"],  # i * i = -1, real
        b=b,
        q_order=6,
    )
    assert len(table.numerators) == 2


def test_parity_projected_folded_matches_manual_numeric_fold_nonidentical():
    mp.mp.dps = 60
    b = b_from_c(C)
    Delta = mp.mpf("3.3")
    hbar_shifted = [d + mp.mpf("0.01") for d in D_EXT]
    ell = 1
    m_max = 2
    n_max = 2

    full = parity_projected_B_yyb_poly_table(
        ell,
        D_EXT,
        hbar_shifted,
        m_max + 2 * n_max + 1,
        m_max + 2 * n_max + 1,
        parity="-",
        b=b,
        q_order=8,
    ).evaluate(Delta)
    folded_manual = _fold_parity(full, 1, m_max, n_max)

    folded = parity_projected_B_ws_poly_table(
        ell,
        D_EXT,
        hbar_shifted,
        m_max,
        n_max,
        parity="-",
        b=b,
        q_order=8,
    ).evaluate(Delta)

    _assert_tables_close(folded, folded_manual, tol=mp.mpf("1e-20"))


# --- radial-only slices ---


@pytest.mark.parametrize("coordinate", ["ws", "xt", "qqb_ws"])
def test_radial_poly_table_matches_numeric(coordinate):
    """The symbolic radial slice reproduces the numeric radial table and has a
    single n-column (the squared variable fixed at 0)."""
    mp.mp.dps = 50
    b = b_from_c(C)
    poly_builder = {
        "ws": G_ws_radial_poly_table,
        "xt": G_xt_radial_poly_table,
        "qqb_ws": G_qqb_ws_radial_poly_table,
    }[coordinate]
    num_builder = {
        "ws": G_ws_radial_table,
        "xt": G_xt_radial_table,
        "qqb_ws": G_qqb_ws_radial_table,
    }[coordinate]
    sym = poly_builder(2, D_EXT, 3, b=b, q_order=8)
    Delta = mp.mpf("3.1")
    numeric = num_builder(Delta, 2, D_EXT, 3, b=b, q_order=8)
    evaluated = sym.evaluate(Delta)
    assert all(len(row) == 1 for row in evaluated)  # radial: single column
    for m in range(4):
        assert _close(evaluated[m][0], numeric[m], tol=mp.mpf("1e-25"))


def test_parity_projected_radial_scalar_limit_recovers_G_radial():
    """With scalar externals the projected radial table reduces to the scalar
    G radial table (parity '+' matches even ell)."""
    mp.mp.dps = 50
    b = b_from_c(C)
    Delta = mp.mpf("3.1")
    projected = parity_projected_B_ws_radial_poly_table(
        2, D_EXT, D_EXT, 3, parity="+", b=b, q_order=8
    ).evaluate(Delta)
    g = G_ws_radial_poly_table(2, D_EXT, 3, b=b, q_order=8).evaluate(Delta)
    for m in range(4):
        assert _close(projected[m][0], g[m][0], tol=mp.mpf("1e-22"))


# --- denominator structure ---


def test_denominator_roots_are_kac_delta_poles():
    """The Delta-poles sit at Delta = 2 d_{r,s} +/- ell."""
    mp.mp.dps = 50
    b = b_from_c(C)
    ell = 2
    sym = G_zzb_poly_table(ell, SCALAR_EXT, 2, 2, b=b, q_order=6, lambda_cut=None)
    # Each root should make the denominator polynomial vanish: |D(root)| is
    # tiny relative to the polynomial's scale a unit step away.
    for root in sym.denominator_roots:
        at_root = abs(sym.denominator.evaluate(root))
        nearby = abs(sym.denominator.evaluate(root + 1))
        assert at_root < mp.mpf("1e-25") * nearby


def test_poly_table_validates_inputs():
    with pytest.raises(ValueError):
        G_zzb_poly_table(2, SCALAR_EXT, -1, b=b_from_c(C), q_order=4)
    with pytest.raises(ValueError):
        G_zzb_poly_table(2, SCALAR_EXT, 2, b=b_from_c(C), c=C, q_order=4)
    with pytest.raises(ValueError):
        G_zzb_poly_table(2, SCALAR_EXT, 2, q_order=4, lambda_cut=None)  # neither b nor c
    with pytest.raises(ValueError):
        G_zzb_poly_table(2, [mp.mpf("0.1")] * 3, 2, b=b_from_c(C), q_order=4)


# --- The interpolation method (order reduction) ---

# Unitary c in the complex-pole regime, matching the interpolation tests.
C_INTERP = mp.mpf("10")


def _max_table_error(table_a, table_b):
    worst = mp.mpf(0)
    for row_a, row_b in zip(table_a, table_b):
        for entry_a, entry_b in zip(row_a, row_b):
            worst = max(worst, abs(mp.mpc(entry_a) - mp.mpc(entry_b)))
    return worst


def test_interpolation_is_exact_at_sample_nodes():
    """Each (m, n) entry's interpolation polynomial reconstructs the exact
    table at the entry's own sample nodes. With per-entry polynomial
    degrees (each entry uses degree `exact.numerators[m][n].degree -
    dropped_pole_count`), the sample-node set for (m, n) is a per-entry
    BS run rather than the shared set used by the highest-degree entry."""
    mp.mp.dps = 60
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=10)
    assert isinstance(interp, InterpolatedDeltaTable)
    assert interp.numerator_sample_points is not None
    for m in range(interp.m_max + 1):
        for n in range(interp.n_max + 1):
            for node in interp.numerator_sample_points[m][n]:
                interp_table = interp.evaluate(node)
                exact_table = exact.evaluate(node)
                err = abs(mp.mpc(interp_table[m][n]) - mp.mpc(exact_table[m][n]))
                assert err < mp.mpf("1e-25")


def test_default_shift_for_2d_basis_is_unitarity_bound():
    """For the 2d bases (scalar G, ordered B, parity-projected B), the default
    interpolation shift is `|spin|` -- the unitarity bound, putting the
    polynomial-variable origin on the SDPB positivity interval boundary."""
    mp.mp.dps = 50
    b = b_from_c(C_INTERP)
    for ell in (0, 1, 2, 3):
        exact = G_zzb_poly_table(ell, SCALAR_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)
        assert exact.spin == ell
        interp = interpolate_delta_table(exact, npoles=6)
        assert interp.shift == mp.mpf(abs(ell))


def test_c_less_than_one_interpolation_keeps_bootstrap_shift():
    """At c < 1, interpolation keeps the bootstrap shift and force-keeps the
    real-positive shifted Kac poles into the final prefactor, so the table
    interpolates cleanly without changing the shift. The legacy
    `force_keep_real_positive=False` opt-out restores the previous raise-on-
    real-positive-pole contract.
    """
    mp.mp.dps = 50

    b = b_from_c(mp.mpf("0.65"))
    exact = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)
    shift = mp.mpf(2)
    assert exact.spin == shift
    assert _default_interpolation_shift(exact) == shift

    # Default behavior: force-keep real-positive poles; interpolation succeeds.
    interp = interpolate_delta_table(exact, npoles=6)
    assert interp.shift == shift
    # At least one pole in the realized prefactor must be a real positive
    # one (i.e. above the unitarity shift), proving it survived the partition.
    has_forced = any(
        mp.im(p) == 0 and mp.re(p) > 0 for p in interp.prefactor.poles
    )
    assert has_forced

    # Reconstruction matches the exact table at a Delta safely away from any
    # forced pole. Choose a sample point that lies between the lowest two
    # sample points (matrix-model nodes), which BS keeps away from the
    # forced-pole locations.
    nodes = sorted(mp.mpf(p) for p in interp.sample_points)
    test_delta = (nodes[0] + nodes[1]) / 2
    err = _max_table_error(interp.evaluate(test_delta), exact.evaluate(test_delta))
    assert err < mp.mpf("1e-10")

    # Legacy opt-out preserves the original raise-on-real-positive contract.
    with pytest.raises(ValueError, match=r"Re\(pole\) <= 0"):
        interpolate_delta_table(exact, npoles=6, force_keep_real_positive=False)


def test_interpolation_error_decreases_with_npoles():
    """Keeping more rightmost poles (and more nodes) tightens the order-reduced
    approximation of the exact table at a fixed physical Delta."""
    mp.mp.dps = 60
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=8, lambda_cut=None)

    small = interpolate_delta_table(exact, npoles=2)
    nodes = sorted(mp.mpf(p) for p in small.sample_points)
    test_delta = (nodes[0] + nodes[1]) / 2  # inside the smallest support

    errors = {}
    for npoles in (2, 6, 12):
        interp = interpolate_delta_table(exact, npoles=npoles)
        errors[npoles] = _max_table_error(
            interp.evaluate(test_delta), exact.evaluate(test_delta)
        )
    assert errors[6] < errors[2]
    assert errors[12] < errors[6]
    # The high-order interpolant is at bootstrap-grade accuracy.
    assert errors[12] < mp.mpf("1e-12")


def test_interpolant_accessor_matches_table():
    """`interpolant(m, n)` yields a DeltaInterpolant consistent with evaluate."""
    mp.mp.dps = 60
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=8)
    test_delta = (mp.mpf(interp.sample_points[1]) + mp.mpf(interp.sample_points[2])) / 2
    table = interp.evaluate(test_delta)
    for m in range(3):
        for n in range(3):
            single = interp.interpolant(m, n).evaluate(test_delta)
            assert _close(single, table[m][n], tol=mp.mpf("1e-28"))


def test_interpolate_delta_table_validates_inputs():
    mp.mp.dps = 50
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=6, lambda_cut=None)
    with pytest.raises(ValueError):
        interpolate_delta_table(exact, npoles=-1)
    with pytest.raises(ValueError):
        interpolate_delta_table(exact, npoles=4, n_sample_points=0)


def test_interpolate_delta_table_accepts_explicit_sample_points():
    mp.mp.dps = 60
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=6, n_sample_points=7)

    assert len(interp.sample_points) == 7
    test_delta = mp.mpf("4.5")
    table = interp.evaluate(test_delta)
    assert len(table) == 3
    assert len(table[0]) == 3


def test_interpolate_delta_table_pole_selection_modes():
    """All pole-selection rules produce a valid InterpolatedDeltaTable."""
    mp.mp.dps = 60
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)
    for selection in ("rightmost", "nearest", "residue", "real_first", "real_deep"):
        interp = interpolate_delta_table(
            exact, npoles=8, pole_selection=selection
        )
        assert isinstance(interp, InterpolatedDeltaTable)
        # Each rule interpolates exactly at its own sample nodes.
        node = interp.sample_points[1]
        table = interp.evaluate(node)
        assert len(table) == 3 and len(table[0]) == 3


def test_interpolate_delta_table_defaults_to_contribution_optimal():
    mp.mp.dps = 60
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 2, 2, b=b, q_order=8, lambda_cut=None)

    default_interp = interpolate_delta_table(exact, npoles=8)
    optimal_interp = interpolate_delta_table(
        exact, npoles=8, pole_selection="contribution_optimal"
    )

    assert default_interp.prefactor.poles == optimal_interp.prefactor.poles
    assert default_interp.sample_points == optimal_interp.sample_points

    test_delta = mp.mpf(default_interp.sample_points[2])
    default_table = default_interp.evaluate(test_delta)
    optimal_table = optimal_interp.evaluate(test_delta)
    for default_row, optimal_row in zip(default_table, optimal_table):
        for default_value, optimal_value in zip(default_row, optimal_row):
            assert _close(default_value, optimal_value, tol=mp.mpf("1e-28"))


def test_c2_identical_scalar_source_strip_avoids_zero_shift_pole():
    mp.mp.dps = 60
    c = mp.mpf("2")
    b = b_from_c(c)
    h_ext = [mp.mpf("0.25")] * 4
    tol = mp.mpf("1e-30")

    cs = chiral_symbolic_useries(
        h_ext,
        4,
        b=b,
        c=c,
        z_of_u=lambda u: u,
        u_symmetric=z_symmetric(),
        q_order=8,
    )
    assert all(abs(mp.mpc(root)) > tol for root in cs.denominator_roots)

    exact = G_zzb_poly_table(0, h_ext, 1, 1, c=c, q_order=8, lambda_cut=None)
    interp = interpolate_delta_table(exact, npoles=4, pole_selection="residue")
    assert all(abs(mp.mpc(pole)) > tol for pole in interp.prefactor.poles)


def test_interpolate_delta_table_rejects_unknown_pole_selection():
    mp.mp.dps = 50
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=6, lambda_cut=None)
    with pytest.raises(ValueError):
        interpolate_delta_table(exact, npoles=4, pole_selection="bogus")


def test_residue_scores_one_per_pole():
    """_residue_scores returns one positive score per denominator root."""
    from virasoro_blocks.polyderivs import _residue_scores

    mp.mp.dps = 50
    b = b_from_c(C_INTERP)
    exact = G_zzb_poly_table(2, SCALAR_EXT, 1, 1, b=b, q_order=6, lambda_cut=None)
    scores = _residue_scores(exact)
    assert len(scores) == len(exact.denominator_roots)
    assert all(s > 0 for s in scores)


@pytest.mark.parametrize("c_value", [mp.mpf("2"), mp.mpf("100")])
def test_interpolation_numeric_pole_ranking_matches_symbolic(c_value):
    """Interpolation pole ranking agrees between numeric and symbolic scorers.

    This checks one strip regime point (c=2) and one real_b point (c=100)
    across a representative spin set.
    """
    from virasoro_blocks.polyderivs import _NumericBlockSampler

    mp.mp.dps = 70
    h_ext = [mp.mpf("0.1")] * 4
    npoles = 6

    for ell in (0, 2, 4, 6):
        exact = G_zzb_poly_table(ell, h_ext, 2, 2, c=c_value, q_order=8, lambda_cut=None)
        exact.numeric_sampler = _NumericBlockSampler(
            "zzb",
            h_ext,
            ell,
            2,
            2,
            b=None,
            c=c_value,
            q_order=8,
            chiral_norm=None,
            z_shift=None,
        )
        sym = interpolate_delta_table(
            exact,
            npoles=npoles,
            pole_selection="contribution",
            pole_ranking="symbolic",
        )
        num = interpolate_delta_table(
            exact,
            npoles=npoles,
            pole_selection="contribution",
            pole_ranking="numeric",
        )

        # Compare the kept pole *sets* (order-independent): the full-principal-
        # part numeric and symbolic scores agree only to the DFT aliasing floor
        # (~1e-11 at the default n_extra), so two near-equal-score poles can be
        # returned in a different order even when the same set is kept.
        assert len(sym.prefactor.poles) == len(num.prefactor.poles)
        key = lambda z: (float(mp.re(z)), float(mp.im(z)))
        for p_sym, p_num in zip(
            sorted(sym.prefactor.poles, key=key),
            sorted(num.prefactor.poles, key=key),
        ):
            assert abs(mp.mpc(p_sym) - mp.mpc(p_num)) < mp.mpf("1e-28")


def test_symbolic_workdps_can_disable_guard_for_interpolation():
    from virasoro_blocks.polyderivs import _symbolic_guard_digits, _symbolic_workdps_for_q_order

    mp.mp.dps = 80
    q_order = 60
    assert _symbolic_workdps_for_q_order(q_order, use_guard=False) == mp.mp.dps
    assert _symbolic_workdps_for_q_order(q_order, use_guard=True) == (
        mp.mp.dps + _symbolic_guard_digits(q_order)
    )


@pytest.mark.parametrize("c_value", [mp.mpf("30"), mp.mpf("400")])
def test_numeric_ranking_matches_direct_residue_real_b_high_q(c_value):
    """At c > 25 and high q the numeric ranking must match the *direct* residue.

    The symbolic score evaluates a degree-~q_order numerator (coefficients ~1e70+
    at large c) at the pole, which is catastrophically ill-conditioned even with
    the guard, so it cannot be used as the reference here. Instead validate the
    numeric ``K*H(v)`` ranking against the residue read straight off the block,
    Res_{Delta=p} G_00 = lim_{eps->0} eps * G_00(p + eps), with the prefactor
    stripped. For real_b the poles are real, so the numeric sampler evaluates at
    real Delta near the pole. We compare the top poles whose residue is well
    above the eps-extraction floor.
    """
    from virasoro_blocks.polyderivs import _NumericBlockSampler, _numeric_residue_scores

    mp.mp.dps = 80
    h_ext = [mp.mpf("0.125")] * 4
    ell = 2
    q_order = 44
    exact = G_zzb_poly_table(ell, h_ext, 2, 2, c=c_value, q_order=q_order, lambda_cut=None)
    exact.numeric_sampler = _NumericBlockSampler(
        "zzb", h_ext, ell, 2, 2,
        b=None, c=c_value, q_order=q_order, chiral_norm=None, z_shift=None,
    )

    num = _numeric_residue_scores(exact, score_entries="value")
    assert num is not None and len(num) == len(exact.denominator_roots)

    roots = [mp.mpc(r) for r in exact.denominator_roots]
    assert all(abs(mp.im(p)) < mp.mpf("1e-40") for p in roots)  # real_b: real poles
    exp_base = exact.exp_base
    sampler = exact.numeric_sampler
    eps = mp.mpf(10) ** (-35)

    def direct_residue(p):
        pr = mp.mpf(mp.re(p))
        g00 = sampler([pr + eps])[0][0][0][0]
        return abs(eps * mp.mpc(g00)) / abs(mp.mpc(exp_base) ** pr)

    true_scores = [direct_residue(p) for p in roots]

    def top(scores, n):
        return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:n])

    # The top poles (residues O(1), far above the ~eps extraction floor) must
    # agree between the numeric ranking and the direct block residue.
    assert top(num, 4) == top(true_scores, 4)


def test_numeric_sampler_reuses_cached_deltas(monkeypatch):
    from virasoro_blocks.polyderivs import _NumericBlockSampler

    mp.mp.dps = 50
    calls = []
    # The non-folded (yyb) sampler computes block values through
    # `_call_restricted` (the batched flint assembler), not the legacy
    # `G_yyb_table_many` helper, so monkeypatch the actual compute delegate.
    # `__call__` hands it only the deduplicated, uncached sample Deltas, so this
    # still exercises the dedup + cache-reuse path the test targets.
    original = _NumericBlockSampler._call_restricted

    def wrapped(self, _derivs, batch, entries):
        calls.append([mp.mpf(delta) for delta in batch])
        return original(self, _derivs, batch, entries)

    monkeypatch.setattr(_NumericBlockSampler, "_call_restricted", wrapped)
    sampler = _NumericBlockSampler(
        "yyb", [mp.mpf("0.15")] * 4, 2, 2, 2,
        b=None, c=mp.mpf("10"), q_order=8,
        chiral_norm=None, z_shift=-mp.mpf("0.3"),
    )
    deltas = [mp.mpf("2.25"), mp.mpf("2.75"), mp.mpf("2.25")]

    first, _ = sampler(deltas)
    second, _ = sampler(deltas)

    assert calls == [[mp.mpf("2.25"), mp.mpf("2.75")]]
    for lhs, rhs in zip(first, second):
        assert lhs == rhs


@pytest.mark.parametrize("kind", ["scalarG", "projected"])
@pytest.mark.parametrize("ell,parity", [(2, "+"), (1, "-")])
def test_folded_triangle_truncation_is_exact_and_shortens_chiral(
    kind, ell, parity, monkeypatch
):
    """Restricting a folded sampler to the m + 2n <= Lambda triangle must give
    the same values (to working precision) as the full-square call on those
    entries, while the chiral series shrinks to the max requested source degree
    (a + 2b). The truncation is exact; the residual ~1e-62 difference is pure
    roundoff -- the full series even accumulates slightly *more* of it (more
    terms summed), so the shorter triangle series is marginally more accurate.
    Force a single fold path so the comparison isolates the truncation, not the
    sub-floor matmul/sparse rounding the contribution-count threshold would
    otherwise flip between the two entry sets."""
    import virasoro_blocks.derivs as derivs_mod
    from virasoro_blocks.truncation import kept_pairs

    monkeypatch.setattr(derivs_mod, "_FOLD_MATMUL_MAX_CONTRIBS", 0)  # always sparse
    mp.mp.dps = 60
    c = mp.mpf("30")
    lam = 8
    q_order = 16
    deltas = [mp.mpf("2.3") + mp.mpf("0.1") * i for i in range(4)]
    full_square = [(m, n) for m in range(lam + 1) for n in range(lam + 1)]

    if kind == "scalarG":
        triangle = kept_pairs("ws", lam, basis="scalar_G", ell=ell)

        def build():
            return _NumericBlockSampler(
                "ws", SCALAR_EXT, ell, lam, lam, b=None, c=c, q_order=q_order,
                chiral_norm=None, z_shift=None)
        nterms_attr = "_folded_evaluator_nterms"
    else:
        triangle = kept_pairs(
            "ws", lam, basis="parity_projected_B", ell=ell, parity_projection=parity)

        def build():
            return _NumericOrderedBlockSampler(
                "ws", _SPINNING_H, _SPINNING_HBAR, ell, lam, lam, b=None, c=c,
                q_order=q_order, chiral_norm=None, z_shift=None, zbar_shift=None,
                projected=True, parity=parity, parity_external=None,
                exchanged_parity="1")
        nterms_attr = "_eval_nterms"

    s_tri = build()
    tri_tables, _ = s_tri(deltas, entries=triangle)
    s_full = build()
    full_tables, _ = s_full(deltas, entries=full_square)

    # chiral series shrank to the triangle's max source degree (<< full square)
    assert getattr(s_tri, nterms_attr) < getattr(s_full, nterms_attr)
    for gt, gf in zip(tri_tables, full_tables):
        for (m, n) in triangle:
            got, want = mp.mpf(gt[m][n]), mp.mpf(gf[m][n])
            assert abs(got - want) <= mp.mpf(10) ** -55 * (abs(want) + 1)


@pytest.mark.parametrize("kind", ["scalarG", "projected"])
def test_fold_matmul_and_sparse_branches_agree(kind, monkeypatch):
    """The folded fold has two interchangeable contraction strategies -- the
    dense flint matmul (used below the contribution-count threshold) and the
    per-Delta sparse loop. They must produce identical values; force each branch
    by toggling the threshold and compare."""
    import virasoro_blocks.derivs as derivs_mod

    mp.mp.dps = 50
    c = mp.mpf("30")
    m_max = n_max = 4
    q_order = 12
    deltas = [mp.mpf("2.2") + mp.mpf("0.1") * i for i in range(4)]
    entries = [(0, 0), (2, 1), (4, 0), (3, 3), (1, 4)]

    def build():
        if kind == "scalarG":
            return _NumericBlockSampler(
                "ws", SCALAR_EXT, 2, m_max, n_max, b=None, c=c, q_order=q_order,
                chiral_norm=None, z_shift=None)
        return _NumericOrderedBlockSampler(
            "ws", _SPINNING_H, _SPINNING_HBAR, 1, m_max, n_max, b=None, c=c,
            q_order=q_order, chiral_norm=None, z_shift=None, zbar_shift=None,
            projected=True, parity="-", parity_external=None, exchanged_parity="1")

    monkeypatch.setattr(derivs_mod, "_FOLD_MATMUL_MAX_CONTRIBS", 10 ** 9)
    matmul_tables, _ = build()(deltas, entries=entries)
    monkeypatch.setattr(derivs_mod, "_FOLD_MATMUL_MAX_CONTRIBS", 0)
    sparse_tables, _ = build()(deltas, entries=entries)

    for gm, gs in zip(matmul_tables, sparse_tables):
        for (m, n) in entries:
            assert abs(mp.mpf(gm[m][n]) - mp.mpf(gs[m][n])) < mp.mpf(10) ** -45


# every coordinate, even and odd ell, both externals -- not the full cartesian product
@pytest.mark.parametrize(
    "coordinate, ell, h_ext",
    [
        ("ws", 1, D_EXT), ("xt", 2, SCALAR_EXT), ("qqb_ws", 3, D_EXT), ("ws", 0, SCALAR_EXT),
    ],
    ids=["ws-1-nonidentical", "xt-2-identical", "qqb_ws-3-nonidentical", "ws-0-identical"],
)
def test_restricted_folded_sampler_matches_full(coordinate, ell, h_ext):
    """The entry-restricted folded sampler (ws/xt/qqb_ws) returns arb-native
    values equal to the full folded builder, on a sparse irregular entry set,
    for even *and* odd ell and for identical *and* nonidentical scalar
    externals. Non-requested entries must stay ``None`` (truly restricted)."""
    mp.mp.dps = 50
    c = mp.mpf("30")
    b = b_from_c(c)
    m_max = n_max = 5
    q_order = 14
    full_fn = {"ws": G_ws_table, "xt": G_xt_table, "qqb_ws": G_qqb_ws_table}[coordinate]
    entries = [(0, 0), (2, 1), (5, 0), (3, 3), (1, 4), (4, 2)]
    deltas = [mp.mpf("1.5") + mp.mpf("0.13") * i for i in range(2)]

    sampler = _NumericBlockSampler(
        coordinate, h_ext, ell, m_max, n_max,
        b=None, c=c, q_order=q_order, chiral_norm=None, z_shift=None,
    )
    tables, is_arb = sampler(deltas, entries=entries)
    assert is_arb  # arb-native restricted path was taken (not the mpmath fallback)

    for delta, grid in zip(deltas, tables):
        full = full_fn(delta, ell, h_ext, m_max, n_max, b=b, q_order=q_order)
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                if (m, n) in entries:
                    assert abs(mp.mpf(grid[m][n]) - mp.mpf(full[m][n])) < mp.mpf(10) ** -35
                else:
                    assert grid[m][n] is None


_SPIN_DELTA_EXT = [mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.4"), mp.mpf("0.6")]
_SPIN_EXT = [mp.mpf("0.2"), mp.mpf("-0.1"), mp.mpf("0.0"), mp.mpf("0.3")]
_SPINNING_H = [(d + s) / 2 for d, s in zip(_SPIN_DELTA_EXT, _SPIN_EXT)]
_SPINNING_HBAR = [(d - s) / 2 for d, s in zip(_SPIN_DELTA_EXT, _SPIN_EXT)]

_ORDERED_POLY = {
    "zzb": B_zzb_poly_table,
    "yyb": B_yyb_poly_table,
    "qqb": B_qqb_poly_table,
}
_PROJECTED_POLY = {
    "zzb": parity_projected_B_zzb_poly_table,
    "qqb": parity_projected_B_qqb_poly_table,
    "ws": parity_projected_B_ws_poly_table,
    "xt": parity_projected_B_xt_poly_table,
    "qqb_ws": parity_projected_B_qqb_ws_poly_table,
}


def _ordered_sampler_diff(sampler, ref_fn, deltas, entries):
    """Max |sampler - symbolic| over `entries` for an ordered/projected sampler,
    asserting non-requested entries stay None."""
    tables, _is_arb = sampler(deltas, entries=entries)
    worst = mp.mpf(0)
    for delta, grid in zip(deltas, tables):
        ref = ref_fn(delta)
        for m in range(sampler.m_max + 1):
            for n in range(sampler.n_max + 1):
                if (m, n) in entries:
                    worst = max(worst, abs(mp.mpc(grid[m][n]) - mp.mpc(ref[m][n])))
                else:
                    assert grid[m][n] is None
    return worst


@pytest.mark.parametrize("coordinate, ell", [("zzb", 0), ("yyb", 1), ("qqb", 2)])
def test_numeric_ordered_B_sampler_matches_symbolic(coordinate, ell):
    """The ordered_B numeric sampler equals the symbolic ordered B table on
    spinning externals (independent h-side / hbar-side weights, two evaluators)."""
    mp.mp.dps = 70
    c = mp.mpf("30")
    b = b_from_c(c)
    m_max = n_max = 4
    q_order = 12
    deltas = [mp.mpf("2.3") + mp.mpf("0.11") * i for i in range(4)]
    entries = [(0, 0), (3, 1), (4, 0), (2, 3)]
    sampler = _NumericOrderedBlockSampler(
        coordinate, _SPINNING_H, _SPINNING_HBAR, ell, m_max, n_max,
        b=None, c=c, q_order=q_order, chiral_norm=None,
        z_shift=None, zbar_shift=None, projected=False,
    )

    def ref_fn(delta):
        # The sampler probes corner entries (m+n>Lambda); the symbolic builders
        # now default to the order-Lambda triangle, so request the full square.
        return _ORDERED_POLY[coordinate](
            ell, _SPINNING_H, _SPINNING_HBAR, m_max, n_max, b=b, q_order=q_order,
            lambda_cut=None,
        ).evaluate(delta)

    assert _ordered_sampler_diff(sampler, ref_fn, deltas, entries) < mp.mpf(10) ** -30


# Folded coordinates only admit a real (+/-1) parity phase, so the complex phase
# is exercised on the non-folded coordinates only -- enumerate the valid
# (coordinate, parity_external) pairs directly rather than skipping the
# meaningless folded+complex cells of a full cartesian product.
@pytest.mark.parametrize(
    "coordinate, parity_external",
    [
        ("zzb", None), ("qqb", None), ("ws", None), ("xt", None), ("qqb_ws", None),
        ("zzb", ["i", "1", "1", "1"]), ("qqb", ["i", "1", "1", "1"]),
    ],
    ids=[
        "zzb-realphase", "qqb-realphase", "ws-realphase", "xt-realphase",
        "qqb_ws-realphase", "zzb-complexphase", "qqb-complexphase",
    ],
)
@pytest.mark.parametrize("ell, parity", [(1, "-"), (2, "+")])
def test_numeric_projected_B_sampler_matches_symbolic(
    coordinate, ell, parity, parity_external
):
    """The parity_projected_B numeric sampler equals the symbolic projected B
    table for non-folded and folded coordinates, real and imaginary parity
    phases (the swap scale becomes complex -> mpc output), spinning externals."""
    folded = coordinate in ("ws", "xt", "qqb_ws")
    mp.mp.dps = 70
    c = mp.mpf("30")
    b = b_from_c(c)
    m_max = n_max = 4
    q_order = 12
    deltas = [mp.mpf("2.3") + mp.mpf("0.11") * i for i in range(4)]
    entries = [(0, 0), (3, 1), (4, 0), (2, 3)]
    sampler = _NumericOrderedBlockSampler(
        coordinate, _SPINNING_H, _SPINNING_HBAR, ell, m_max, n_max,
        b=None, c=c, q_order=q_order, chiral_norm=None,
        z_shift=None, zbar_shift=None, projected=True, parity=parity,
        parity_external=parity_external, exchanged_parity="1",
    )
    _tables, is_arb = sampler(deltas[:1], entries=entries)
    # imaginary phase -> complex block -> mpmath output (not arb-native)
    assert is_arb == (parity_external is None)

    # Non-folded symbolic builders default to the order-Lambda triangle; the
    # sampler probes corner entries, so request the full square (the folded
    # builders truncate via m+2n intrinsically and take no lambda_cut).
    extra = {} if folded else {"lambda_cut": None}

    def ref_fn(delta):
        return _PROJECTED_POLY[coordinate](
            ell, _SPINNING_H, _SPINNING_HBAR, m_max, n_max, parity=parity,
            parity_external=parity_external, exchanged_parity="1",
            b=b, q_order=q_order, **extra,
        ).evaluate(delta)

    assert _ordered_sampler_diff(sampler, ref_fn, deltas, entries) < mp.mpf(10) ** -30


@pytest.mark.parametrize("ell,parity", [(2, "+"), (1, "-")])
def test_folded_projected_B_reduced_n_max_matches_full_square(ell, parity):
    """Capping n_max at the kept maximum (Lambda // 2) for the folded
    parity-projected B build -- how parity_projected_B avoids its full-square
    over-build on the numeric path, since it cannot use the degree-only skeleton
    -- leaves the kept-triangle entries (values *and* degrees) exactly unchanged
    from the (lam, lam) square; the folded builder is symmetric in its two
    halves so the asymmetric (m_max, n_max) is safe."""
    from virasoro_blocks.truncation import kept_pairs

    mp.mp.dps = 50
    c = mp.mpf("30")
    b = b_from_c(c)
    lam = 8
    q_order = 14
    kept = kept_pairs("ws", lam, basis="parity_projected_B", ell=ell,
                      parity_projection=parity)
    n_cap = max(n for _m, n in kept)
    full = parity_projected_B_ws_poly_table(
        ell, _SPINNING_H, _SPINNING_HBAR, lam, lam, parity=parity,
        parity_external=None, exchanged_parity="1", b=b, q_order=q_order)
    reduced = parity_projected_B_ws_poly_table(
        ell, _SPINNING_H, _SPINNING_HBAR, lam, n_cap, parity=parity,
        parity_external=None, exchanged_parity="1", b=b, q_order=q_order)
    delta = mp.mpf("2.7")
    ef, er = full.evaluate(delta), reduced.evaluate(delta)
    for (m, n) in kept:
        assert full.numerators[m][n].degree == reduced.numerators[m][n].degree
        assert mp.mpc(ef[m][n]) == mp.mpc(er[m][n])  # bit-identical


@pytest.mark.parametrize("basis", ["scalar_G", "ordered_B", "parity_projected_B"])
@pytest.mark.parametrize("ell", [3])
def test_nonfolded_corner_skip_matches_full_square(basis, ell):
    """The non-folded lambda_cut corner-skip leaves the kept (m + n <= Lambda)
    triangle bit-identical to the full-square build, zeroes the discarded corner,
    and -- crucially -- the skeleton degree branch reports the same zero-degree
    corner so the shared sample-point sizing matches the coefficient build."""
    from virasoro_blocks.truncation import kept_pairs

    mp.mp.dps = 50
    c = mp.mpf("30")
    lam = 8
    q_order = 4 * lam + 8
    hbar = None if basis == "scalar_G" else _SPINNING_H
    h = _SPINNING_H if basis != "scalar_G" else [mp.mpf("0.1")] * 4
    kw = {}
    if basis == "parity_projected_B":
        kw = dict(parity_projection="+", parity_external=["1", "1", "1", "1"])

    cache = build_chiral_cache(
        ["zzb"], h, hbar_externals=hbar, lambda_=lam, c=c,
        q_order=q_order, chiral_norm=16)
    cs_h, cs_hbar = cache["zzb"]

    def build(cut):
        return build_table_from_cache(
            "zzb", cs_h, cs_hbar, ell, m_max=lam, n_max=lam,
            basis=basis, lambda_cut=cut, **kw)

    full, cut = build(None), build(lam)
    kept = set(kept_pairs(
        "zzb", lam, basis=basis, ell=ell,
        parity_projection=kw.get("parity_projection")))
    for m in range(lam + 1):
        for n in range(lam + 1):
            fa = [mp.mpc(x) for x in full.numerators[m][n].coeffs]
            ca = [mp.mpc(x) for x in cut.numerators[m][n].coeffs]
            if m + n <= lam:
                assert fa == ca  # kept triangle bit-identical
            else:
                assert ca == [mp.mpc(0)]  # corner zeroed

    # Skeleton (degree-only) path: corner reports degree 0 in both branches that
    # support it (scalar_G, ordered_B); parity_projected_B has no skeleton.
    if basis != "parity_projected_B":
        skel = build_chiral_cache(
            ["zzb"], h, hbar_externals=hbar, lambda_=lam, c=c,
            q_order=q_order, chiral_norm=16, skeleton=True)
        sk_h, sk_hbar = skel["zzb"]
        dt = build_table_from_cache(
            "zzb", sk_h, sk_hbar, ell, m_max=lam, n_max=lam,
            basis=basis, lambda_cut=lam, **kw)
        for m in range(lam + 1):
            for n in range(lam + 1):
                if m + n > lam:
                    assert dt.numerator_degrees[m][n] == 0
    assert kept  # sanity: the kept set is non-empty


# one ell per basis, alternating the scored entry set
@pytest.mark.parametrize(
    "ell, score_entries, basis",
    [(1, "value", "ordered"), (3, "kept", "proj+"), (1, "kept", "proj-"), (3, "value", "proj-ws")],
)
def test_two_sided_numeric_residue_scores_match_symbolic(ell, score_entries, basis):
    """The two-sided numeric residue scorer reproduces the symbolic residue
    scores for spinning ordered_B / parity_projected_B (the symbolic scorer uses
    the full numerators and is the correctness oracle)."""
    mp.mp.dps = 40
    c = mp.mpf("7")
    b = b_from_c(c)
    m_max = n_max = 2
    q_order = 10
    h, hbar = _SPINNING_H, _SPINNING_HBAR
    if basis == "ordered":
        exact = B_zzb_poly_table(ell, h, hbar, m_max, n_max, b=b, q_order=q_order, lambda_cut=None)
        sampler = _NumericOrderedBlockSampler(
            "zzb", h, hbar, ell, m_max, n_max, b=None, c=c, q_order=q_order,
            chiral_norm=None, z_shift=None, zbar_shift=None, projected=False)
    else:
        parity = "+" if basis == "proj+" else "-"
        coord = "ws" if basis == "proj-ws" else "zzb"
        builder = (parity_projected_B_ws_poly_table if coord == "ws"
                   else parity_projected_B_zzb_poly_table)
        exact = builder(ell, h, hbar, m_max, n_max, parity=parity,
                        parity_external=None, exchanged_parity="1", b=b, q_order=q_order)
        sampler = _NumericOrderedBlockSampler(
            coord, h, hbar, ell, m_max, n_max, b=None, c=c, q_order=q_order,
            chiral_norm=None, z_shift=None, zbar_shift=None, projected=True,
            parity=parity, parity_external=None, exchanged_parity="1")
    exact.numeric_sampler = sampler
    if not numeric_residue_scoring_available(exact):
        pytest.skip("degenerate poles -> numeric scoring declines (symbolic fallback)")

    numeric = _numeric_residue_scores(exact, score_entries=score_entries)
    symbolic = _residue_scores(exact, score_entries=score_entries, pole_ranking="symbolic")
    assert numeric is not None and len(numeric) == len(symbolic)
    scale = max(symbolic) if max(symbolic) > 0 else mp.mpf(1)
    # The symbolic scorer evaluates the lazy flint-backed numerators, whose
    # cached arb/acb_poly carry interval radii from the build accumulation; that
    # perturbs the scores at the working-precision floor (~10^-(dps-15), here
    # dps=40). The tolerance sits safely above that floor while staying far
    # tighter than the O(1) structural separation between distinct poles. The
    # agreement sharpens by ~20 digits per +20 dps (verified to 3.5e-145 at
    # dps=160).
    for got, want in zip(numeric, symbolic):
        assert abs(got - want) < scale * mp.mpf(10) ** -23


@pytest.mark.parametrize("ell", [0, 1])
@pytest.mark.parametrize(
    "c_str, h, hbar",
    [
        # generic c, generic spinning externals
        ("7", _SPINNING_H, _SPINNING_HBAR),
        # minimal model (Ising) -- surviving-pair structure, still a valid (non
        # recursion-singular) c; the skeleton degree must track the full build.
        ("1/2", _SPINNING_H, _SPINNING_HBAR),
        # special externals: integer / half-integer differences and an equal
        # pair (h1 == h2 -> h12 == 0), which is where a naive degree count might
        # fear a vanishing leading coefficient. It cannot: the chiral leading
        # coefficient is N_lead * s1^m/m!, independent of c and externals.
        ("7", [mp.mpf("1.0"), mp.mpf("0.0"), mp.mpf("0.5"), mp.mpf("0.0")],
         [mp.mpf("1.5"), mp.mpf("0.0"), mp.mpf("0.2"), mp.mpf("0.2")]),
        ("1/2", [mp.mpf("0.2"), mp.mpf("0.2"), mp.mpf("0.4"), mp.mpf("0.1")],
         [mp.mpf("0.3"), mp.mpf("0.3"), mp.mpf("0.4"), mp.mpf("0.1")]),
    ],
    ids=["generic", "ising", "special-ext", "ising-special-ext"],
)
def test_skeleton_ordered_B_degrees_match_full(ell, c_str, h, hbar):
    """The ordered_B degree-only skeleton records exactly the numerator degrees
    of the full build (deg B[m][n] = cd_h[m] + cd_hbar[n], no cancellation),
    across even/odd ell, a minimal-model c, and special external differences --
    none of which makes a chiral leading coefficient vanish."""
    mp.mp.dps = 60
    c = mp.mpf(c_str)
    b = b_from_c(c)
    m_max = n_max = 4
    q_order = 12
    n_terms = max(m_max, n_max) + 1
    common = dict(b=b, c=c, q_order=q_order, z_of_u=lambda u: u,
                  u_symmetric=z_symmetric())
    from virasoro_blocks.singularity import bare_recursion_probes

    # at the Ising c the full symbolic build fails closed (its values are unreliable there); the degrees are what
    # is compared, so the bare build is taken deliberately
    with bare_recursion_probes():
        full = B_zzb_poly_table(ell, h, hbar, m_max, n_max, b=b, q_order=q_order, lambda_cut=None)
    cs_h = chiral_symbolic_useries(h, n_terms, skeleton=True, **common)
    cs_hbar = chiral_symbolic_useries(hbar, n_terms, skeleton=True, **common)
    skel = _build_B_poly_table(cs_h, cs_hbar, ell, m_max, n_max)
    assert skel.numerators is None
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            assert skel.numerator_degree(m, n) == full.numerators[m][n].degree


def test_skeleton_folded_projected_B_degrees_pair_the_two_halves():
    """Folded skeletons pair degrees from both chiral halves.

    Pairwise-equal hbar externals can cancel poles retained by the h half,
    so the denominators and degrees need not agree.
    """
    from virasoro_blocks.polyderivs import build_chiral_cache, build_table_from_cache

    h = [Fraction(43, 40), Fraction(3, 40)] * 2
    hbar = [Fraction(3, 40)] * 4
    with mp.workdps(40):
        tables = {}
        for skeleton in (False, True):
            cache = build_chiral_cache(("xt",), h, hbar, lambda_=3, c=8, q_order=8,
                                       skeleton=skeleton, truncated=True)
            tables[skeleton] = build_table_from_cache(
                "xt", *cache["xt"], 1, m_max=3, n_max=1, basis="parity_projected_B",
                parity_projection="+", parity_external=("1",) * 4)
    full, skel = tables[False], tables[True]
    assert len(skel.denominator_roots) == len(full.denominator_roots)
    assert skel.numerators is None
    assert [[skel.numerator_degree(m, n) for n in range(2)] for m in range(4)] == _full_degree_grid(full)


def test_numeric_projected_B_identical_externals_matches_scalar_G():
    """With identical scalar externals the parity '+' projected sampler reduces
    exactly to the scalar-G sampler (single shared evaluator)."""
    mp.mp.dps = 50
    c = mp.mpf("30")
    scalar = [mp.mpf("0.2")] * 4
    m_max = n_max = 4
    q_order = 12
    ell = 2
    deltas = [mp.mpf("2.3") + mp.mpf("0.13") * i for i in range(3)]
    entries = [(m, n) for m in range(m_max + 1) for n in range(n_max + 1)]
    ordered = _NumericOrderedBlockSampler(
        "zzb", scalar, scalar, ell, m_max, n_max, b=None, c=c, q_order=q_order,
        chiral_norm=None, z_shift=None, zbar_shift=None, projected=True,
        parity="+", parity_external=None, exchanged_parity="1",
    )
    g = _NumericBlockSampler(
        "zzb", scalar, ell, m_max, n_max, b=None, c=c, q_order=q_order,
        chiral_norm=None, z_shift=None,
    )
    to, _ = ordered(deltas, entries=entries)
    tg, _ = g(deltas, entries=entries)
    for go, gg in zip(to, tg):
        for (m, n) in entries:
            assert abs(mp.mpf(go[m][n]) - mp.mpf(gg[m][n])) < mp.mpf(10) ** -40


# --- degree-only (skeleton) build for the numeric-sampler path -------------


def _full_degree_grid(table):
    return [
        [table.numerators[m][n].degree for n in range(table.n_max + 1)]
        for m in range(table.m_max + 1)
    ]


@pytest.mark.parametrize("c", ["1.2", "7", "30"])
@pytest.mark.parametrize("ell", [0, 2, 3, 5])
def test_skeleton_G_table_degrees_match_full(c, ell):
    """The non-folded scalar_G degree-only skeleton table records exactly the
    numerator degrees of the full build, for even and odd ell (the odd-ell
    diagonal vanishes and the off-diagonal drops one degree)."""
    mp.mp.dps = 80
    h_ext = SCALAR_EXT
    b = b_from_c(mp.mpf(c))
    m_max = n_max = 6
    n_terms = max(m_max, n_max) + 1
    common = dict(
        b=b, c=mp.mpf(c), z_of_u=lambda u: u, u_symmetric=z_symmetric(),
        q_order=14,
    )
    cs_full = chiral_symbolic_useries(h_ext, n_terms, **common)
    cs_skel = chiral_symbolic_useries(h_ext, n_terms, skeleton=True, **common)
    assert cs_skel.numerators is None
    assert cs_skel.numerator_degrees == [p.degree for p in cs_full.numerators]

    full = _build_G_poly_table(cs_full, ell, m_max, n_max)
    skel = _build_G_poly_table(cs_skel, ell, m_max, n_max)
    assert skel.numerators is None
    assert skel.numerator_degrees == _full_degree_grid(full)
    assert skel.max_numerator_degree() == full.max_numerator_degree()


def test_skeleton_chiral_F_degrees_match_full():
    mp.mp.dps = 80
    h_ext = D_EXT
    b = b_from_c(C)
    common = dict(
        b=b, c=C, z_of_u=lambda u: u, u_symmetric=z_symmetric(), q_order=12,
    )
    cs_full = chiral_symbolic_useries(h_ext, 8, **common)
    cs_skel = chiral_symbolic_useries(h_ext, 8, skeleton=True, **common)
    full = _build_F_chiral_poly_table(cs_full, 7)
    skel = _build_F_chiral_poly_table(cs_skel, 7)
    assert skel.numerators is None
    assert skel.numerator_degrees == _full_degree_grid(full)


@pytest.mark.parametrize("coord, ell", [("zzb", 3), ("yyb", 0), ("qqb", 2)])
def test_skeleton_interpolation_identical_to_full(coord, ell):
    """For the non-folded scalar_G coordinates the degree-only skeleton build
    yields a bit-identical interpolant to the full build: the sample values
    (numeric sampler) and numerator degrees match, so the node placement and
    fitted polynomials are the same."""
    mp.mp.dps = 60
    h_ext = SCALAR_EXT
    c = mp.mpf("1.2")
    b = b_from_c(c)
    lam = 6
    q_order = 16

    def build(skeleton):
        cs_h = build_chiral_cache(
            [coord], h_ext, lambda_=lam, b=b, q_order=q_order, skeleton=skeleton
        )[coord][0]
        table = build_table_from_cache(
            coord, cs_h, None, ell, m_max=lam, n_max=lam, basis="scalar_G"
        )
        table.numeric_sampler = _NumericBlockSampler(
            coord, h_ext, ell, table.m_max, table.n_max,
            b=None, c=c, q_order=q_order, chiral_norm=None, z_shift=None,
        )
        return interpolate_delta_tables(
            {coord: table}, npoles=len(table.denominator_roots),
            pole_selection="contribution",
        )[coord]

    full = build(skeleton=False)
    skel = build(skeleton=True)
    assert [len(r) for r in skel.numerators] == [len(r) for r in full.numerators]
    for rs, rf in zip(skel.numerators, full.numerators):
        for ps, pf in zip(rs, rf):
            assert len(ps.coeffs) == len(pf.coeffs)
            for a, b_ in zip(ps.coeffs, pf.coeffs):
                assert mp.mpf(abs(mp.mpc(a) - mp.mpc(b_))) <= mp.mpf("1e-50")


@pytest.mark.parametrize("c, ell", [("4", 3), ("30", 0)])
def test_block_sampler_entry_restricted_matches_full(c, ell):
    """The entry-restricted / batched / arb-native sampler path used by the
    interpolation (passing `entries`) yields the same interpolant as the
    full-table path, to working precision. Guards the batched H-recursion,
    the restricted outer product, and the arb-native (no round-trip) values."""
    pytest.importorskip("flint")
    mp.mp.dps = 100
    h_ext = [mp.mpf("0.0625")] * 4
    b = b_from_c(mp.mpf(c))
    lam, q = 6, 16

    def build():
        cache = build_chiral_cache(
            ["zzb"], h_ext, lambda_=lam, b=b, q_order=q, skeleton=False
        )
        t = build_table_from_cache(
            "zzb", cache["zzb"][0], None, ell, m_max=lam, n_max=lam, basis="scalar_G"
        )
        t.numeric_sampler = _NumericBlockSampler(
            "zzb", h_ext, ell, t.m_max, t.n_max,
            b=None, c=mp.mpf(c), q_order=q, chiral_norm=None, z_shift=None,
        )
        return t

    new = interpolate_delta_tables(
        {"zzb": build()}, npoles=12, pole_selection="contribution"
    )["zzb"]

    # Force the full-table path (no entry restriction / no arb-native values).
    saved = _NumericBlockSampler._call_restricted
    _NumericBlockSampler._call_restricted = lambda *a, **k: None
    try:
        old = interpolate_delta_tables(
            {"zzb": build()}, npoles=12, pole_selection="contribution"
        )["zzb"]
    finally:
        _NumericBlockSampler._call_restricted = saved

    # Compare the reconstructed interpolant (the meaningful invariant), not the
    # raw reduced-polynomial coefficients: for an odd-ell block the (0,0) entry
    # vanishes so contribution scores on (0,1), which can select a more
    # ill-conditioned pole set whose interpolation amplifies ~1e-95 node noise
    # into the high-degree coefficients (they cancel on evaluation). The block
    # itself still matches the full-sampler path to working precision.
    worst = mp.mpf(0)
    for D in (mp.mpf("4.3"), mp.mpf("6.7"), mp.mpf("9.1")):
        vn, vo = new.evaluate(D), old.evaluate(D)
        for rn, ro in zip(vn, vo):
            for a, b_ in zip(rn, ro):
                if a is None or b_ is None:
                    continue
                a, b_ = mp.mpc(a), mp.mpc(b_)
                worst = max(worst, abs(a - b_) / max(abs(b_), mp.mpf("1e-60")))
    assert worst < mp.mpf("1e-60"), (c, ell, worst)


def test_root_provenance_key_normalization():
    """Provenance-key normalizer: (side, labels) -> hashable key; malformed -> None."""
    from virasoro_blocks.polyderivs import _root_provenance_key
    assert _root_provenance_key(("left", ((2, 1),))) == ("left", ((2, 1),))
    assert _root_provenance_key(("right", [(3, 1), (1, 5)])) == ("right", ((3, 1), (1, 5)))
    # malformed / empty -> None (caller declines, falls back to symbolic)
    assert _root_provenance_key(((2, 1),)) is None          # no side string
    assert _root_provenance_key(("left", ())) is None        # empty labels
    assert _root_provenance_key(("left", None)) is None      # non-container labels
    assert _root_provenance_key(None) is None


def test_ordered_scores_from_provenance_restores_and_guards():
    """Order restoration from a provenance-keyed score map, with the legacy
    coverage guard and distinct-collision handling."""
    from types import SimpleNamespace
    from virasoro_blocks.polyderivs import (
        _ordered_scores_from_provenance, _root_provenance_key,
    )
    # Two distinct Kac origins that collide to the SAME root value must stay
    # distinct keys -> distinct scores, restored in root order.
    prov = [("left", ((2, 1),)), ("right", ((2, 1),)), ("left", ((1, 3),))]
    exact = SimpleNamespace(root_provenance=prov, denominator_roots=[1.0, 1.0, 4.0])
    score_map = {
        ("left", ((2, 1),)): mp.mpf("10"),
        ("right", ((2, 1),)): mp.mpf("20"),
        ("left", ((1, 3),)): mp.mpf("30"),
    }
    assert _ordered_scores_from_provenance(exact, score_map) == [
        mp.mpf("10"), mp.mpf("20"), mp.mpf("30")]
    # A duplicate provenance key resolves to the same score for both roots.
    prov_dup = [("left", ((2, 1),)), ("left", ((2, 1),))]
    exact_dup = SimpleNamespace(root_provenance=prov_dup, denominator_roots=[1.0, 1.0])
    assert _ordered_scores_from_provenance(
        exact_dup, {("left", ((2, 1),)): mp.mpf("7")}) == [mp.mpf("7"), mp.mpf("7")]
    # Missing coverage / None map -> None (defer to symbolic).
    assert _ordered_scores_from_provenance(exact, {("left", ((2, 1),)): mp.mpf("1")}) is None
    assert _ordered_scores_from_provenance(exact, None) is None


@pytest.mark.parametrize('skeleton', [False, True])
def test_pole_free_chiral_interpolation_retains_constant_numerators(skeleton):
    from virasoro_blocks.derivs import chiral_z_taylor
    from virasoro_blocks.polyderivs import F_z_poly_table, build_numeric_sampler, interpolate_delta_table

    with mp.workdps(40):
        external = ['.125'] * 4
        exact = F_z_poly_table(external, 1, c=32, q_order=1, skeleton=skeleton)
        if skeleton:
            exact.numeric_sampler = build_numeric_sampler('z', chiral=True, spinning=False,
                projected=False, h_externals=external, hbar_externals=None, ell=0,
                m_max=1, n_max=0, c=32, q_order=1, chiral_norm=None)
        interpolated = interpolate_delta_table(exact, None)
        for weight in (1, 2):
            expected = chiral_z_taylor(weight, external, 2, c=32, q_order=1)
            actual = interpolated.evaluate(weight)
            assert expected[0] != 0
            for m in range(2):
                assert abs(actual[m][0] - expected[m]) < mp.mpf('1e-35') * max(1, abs(expected[m]))


def test_small_c_symbolic_tables_carry_guard_digits():
    """At c < 2 the common-denominator h-rational build cancels between nearby Kac poles, about 1.9 digits
    per unit of q_order at c = 0.8: a dps-60 build at q_order 30 had coefficients good to 2e-7 only.  (c = 0.83:
    0.8 denotes the Potts minimal-model charge 4/5, where generic symbolic tables fail closed.)"""
    from virasoro_blocks import derivs as D
    from virasoro_blocks.polyderivs import G_zzb_poly_table

    hx = [mp.mpf("0.11"), mp.mpf("0.13"), mp.mpf("0.17"), mp.mpf("0.19")]
    zs, c, delta = mp.mpf(-0.3), mp.mpf("0.83"), mp.mpf("4.61")
    with mp.workdps(60):
        table = G_zzb_poly_table(1, hx, 3, 3, c=c, q_order=30, z_shift=zs)
    with mp.workdps(200):
        ref = D.G_zzb_table(delta, 1, hx, 3, 3, c=c, q_order=30, z_shift=zs)
        got = table.evaluate(delta)
        err = max(abs(got[m][n] / ref[m][n] - 1) for m in range(4) for n in range(4) if m + n <= 3 and ref[m][n] != 0)
    assert err < mp.mpf(10) ** -50


@pytest.mark.parametrize("coefficient", [1j, mp.mpc(0, 1), complex(1, 0), mp.mpc(1, 0), Fraction(1, 3)])
@pytest.mark.parametrize("iterator", [False, True])
def test_exp_h_times_numeric_types_and_iterators(coefficient, iterator):
    with mp.workdps(50):
        coefficients = [0, coefficient]
        series = exp_h_times(iter(coefficients) if iterator else coefficients, 5)
        expected_field = mp.mpc if mp.im(coefficient) else mp.mpf
        for n, polynomial in enumerate(series.coeffs):
            expected = (2 * mp.mpc(mp.convert(coefficient))) ** n / mp.factorial(n)
            assert abs(polynomial.evaluate(2) - expected) < mp.mpf('1e-45')
            assert polynomial.field is expected_field
