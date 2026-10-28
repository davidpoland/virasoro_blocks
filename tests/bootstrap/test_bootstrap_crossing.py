"""Tests for `virasoro_bootstrap.crossing` helpers.

The bootstrap-side serialized transforms have two jobs:

- fold complex conjugate pole pairs or genuine positive real poles into
    positivity-equivalent representations that the PMP layer can consume, and
- fold the crossing prefactor into the serialized numerator polynomials.

These tests pin both the algebraic semantics and the PMP-facing invariants.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyderivs import G_zzb_poly_table
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.regularization import merged_delta_table
from virasoro_blocks.serialization import (
    SerializedDerivTable,
    serialize_deriv_tables,
)
from virasoro_bootstrap.crossing import (
    F_minus_pairs,
    _remove_selected_root_groups,
    apply_prefactor_to_table,
    apply_norm_base_scaling,
    build_positive_cleared_block,
    drop_complex_pole_pairs,
    interpolate_serialized_block,
    lower_positive_real_poles,
    select_positive_real_root_groups,
    split_serialized_block_by_pole_groups,
    v_prefactor_taylor,
)
from virasoro_bootstrap.pmp_writer import build_pmp_doc, build_prefactored_pmp_doc


C = mp.mpf("10")  # in the 1<c<25 "strip"; produces complex conjugate pole pairs
SCALAR_EXT = [mp.mpf("0.15")] * 4
ISING_C = mp.mpf("0.5")
ISING_SIGMA = mp.mpf("0.0625")


def _max_table_error(table_a, table_b):
    worst = mp.mpf(0)
    for row_a, row_b in zip(table_a, table_b):
        for entry_a, entry_b in zip(row_a, row_b):
            worst = max(worst, abs(mp.mpc(entry_a) - mp.mpc(entry_b)))
    return worst


def _sum_tables(table_a, table_b):
    return [
        [a + b for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(table_a, table_b)
    ]


def _scale_table(table, scalar):
    return [[scalar * value for value in row] for row in table]


def _table_scale(table):
    return max(abs(mp.mpc(value)) for row in table for value in row)


def _is_complex(p):
    return abs(mp.im(mp.mpc(p))) > mp.mpf("1e-30")


def _count_complex_poles(serialized):
    return sum(1 for p in serialized.pole_list_x if _is_complex(p))


def _dropped_pair_product(input_serialized, output_serialized, x):
    """At a given x = Delta - shift, evaluate prod over dropped poles of
    (x - p). Includes both members of each conjugate pair, so the product
    is real and positive."""
    remaining = list(output_serialized.pole_list_x)
    product = mp.mpc(1)
    for p in input_serialized.pole_list_x:
        if remaining and abs(mp.mpc(p) - mp.mpc(remaining[0])) < mp.mpf("1e-30"):
            remaining.pop(0)
            continue
        product *= mp.mpc(x) - mp.mpc(p)
    return product


def _strip_c_serialized(ell=2, m_max=2, n_max=2, q_order=8):
    """Build a SerializedDerivTable at c=10, which has complex conjugate poles."""
    mp.mp.dps = 50
    b = b_from_c(C)
    table = G_zzb_poly_table(ell, SCALAR_EXT, m_max, n_max, b=b, q_order=q_order)
    return serialize_deriv_tables({"zzb": table})


@pytest.fixture(scope="module")
def ising_prefactored_serialized():
    mp.mp.dps = 80
    table = merged_delta_table(lambda c_value: G_zzb_poly_table(
            0,
            [ISING_SIGMA] * 4,
            4,
            4,
            c=c_value,
            q_order=12,
        ), c=ISING_C)
    serialized = serialize_deriv_tables({"zzb": table})
    return apply_prefactor_to_table(
        serialized,
        v_prefactor_taylor(ISING_SIGMA, 4, 4),
    )


def _has_positive_real_pole(serialized):
    return any(
        (not _is_complex(p)) and mp.re(mp.mpc(p)) > mp.mpf("1e-25")
        for p in serialized.pole_list_x
    )


def _sample_nonpole_xs(blocks):
    candidates = [
        mp.mpf("0.137"),
        mp.mpf("0.411"),
        mp.mpf("0.943"),
        mp.mpf("1.707"),
        mp.mpf("2.931"),
        mp.mpf("4.217"),
    ]
    safe = []
    for x in candidates:
        tol = mp.mpf("1e-25") * max(abs(x), mp.mpf(1))
        if all(
            abs(mp.mpc(x) - mp.mpc(pole)) > tol
            for block in blocks
            for pole in block.pole_list_x
        ):
            safe.append(x)
    return safe


def _positive_clearing_value(selected_groups, x):
    value = mp.mpf(1)
    for root, multiplicity in selected_groups:
        exponent = multiplicity if multiplicity % 2 == 0 else multiplicity + 1
        value *= (x - root) ** exponent
    return value


# ---------------------------------------------------------------------------
# Semantics: output = input * (dropped-pair product). Positivity-equivalent on
# the real axis since each dropped pair contributes a strictly-positive Q(x).
# ---------------------------------------------------------------------------


def test_dropped_table_equals_input_times_dropped_pair_product():
    """For the c=10 case, output(Delta) = input(Delta) * prod_dropped (x - p)
    at every real Delta. The dropped-pair product is real-positive."""
    serialized = _strip_c_serialized()
    # Sanity: the c=10 case really does have complex poles for us to drop.
    assert _count_complex_poles(serialized) > 0

    dropped = drop_complex_pole_pairs(serialized)

    for Delta in (mp.mpf("2.3"), mp.mpf("3.8"), mp.mpf("6.0"), mp.mpf("12.5")):
        x = Delta - serialized.shift
        prod = _dropped_pair_product(serialized, dropped, x)
        # The dropped-pair product is a product of (x - p)(x - p*) over pairs,
        # which is strictly positive on real x.
        assert mp.re(prod) > 0
        assert abs(mp.im(prod)) < abs(mp.re(prod)) * mp.mpf("1e-30")

        before = serialized.evaluate("zzb", Delta)
        after = dropped.evaluate("zzb", Delta)
        scaled = [[mp.mpc(prod) * v for v in row] for row in before]
        err = _max_table_error(scaled, after)
        # Block values at the strip-coordinate origin are O(1)-ish times the
        # dropped-pair product; demand a generous relative-style tolerance.
        scale = max(
            mp.mpf(1),
            max(abs(mp.mpc(v)) for row in after for v in row),
        )
        assert err < mp.mpf("1e-25") * scale, (Delta, err, scale)


def test_dropped_table_keeps_only_real_poles():
    """`pole_list_x` of the output contains no complex entries, and keeps
    exactly the real entries of the input's `pole_list_x`."""
    serialized = _strip_c_serialized()
    n_complex = _count_complex_poles(serialized)
    n_real = len(serialized.pole_list_x) - n_complex
    assert n_complex % 2 == 0  # well-formed conjugate pairing

    dropped = drop_complex_pole_pairs(serialized)
    for p in dropped.pole_list_x:
        assert not _is_complex(p), p
    assert len(dropped.pole_list_x) == n_real


def test_numerator_polynomials_are_left_unchanged():
    """Dropping pole pairs touches only the denominator; numerators are the
    same Polynomial objects (or at least equal coefficient by coefficient)."""
    serialized = _strip_c_serialized()
    dropped = drop_complex_pole_pairs(serialized)
    for row_in, row_out in zip(
        serialized.numerators["zzb"], dropped.numerators["zzb"]
    ):
        for poly_in, poly_out in zip(row_in, row_out):
            assert poly_out.degree == poly_in.degree
            assert poly_out.coeffs == poly_in.coeffs


def test_remove_selected_root_groups_symmetrizes_remaining_conjugate_pair():
    """Root cleanup should canonicalize matched conjugate remnants exactly."""
    mp.mp.dps = 80
    all_roots = [
        mp.mpf(0),
        mp.mpc(mp.mpf("2.0"), mp.mpf("1.0")),
        mp.mpc(mp.mpf("2.0") + mp.mpf("1e-80"), mp.mpf("-1.0")),
    ]
    remaining = _remove_selected_root_groups(all_roots, [(mp.mpf(0), 1)])
    assert len(remaining) == 2
    z0 = mp.mpc(remaining[0])
    z1 = mp.mpc(remaining[1])
    assert z0 == mp.conj(z1)


# ---------------------------------------------------------------------------
# Edge cases: degenerate inputs.
# ---------------------------------------------------------------------------


def test_real_poles_only_passes_through_unchanged():
    """If there are no complex poles to drop, the table is returned with its
    pole list intact."""
    mp.mp.dps = 40
    poles = (mp.mpf("-1.5"), mp.mpf("-0.3"))
    poly = Polynomial([mp.mpf("2"), mp.mpf("-1")], field=mp.mpf)
    serialized = SerializedDerivTable(
        shift=mp.mpf(0),
        base=mp.mpf("0.5"),
        pole_list_x=poles,
        coordinates=("zzb",),
        numerators={"zzb": [[poly]]},
    )

    dropped = drop_complex_pole_pairs(serialized)
    assert dropped.pole_list_x == poles
    assert dropped.numerators["zzb"][0][0].coeffs == poly.coeffs


def test_orphan_complex_pole_raises():
    """A complex pole whose conjugate is absent from `pole_list_x` indicates
    a block that is not real on the real axis -- refuse rather than silently
    drop it."""
    mp.mp.dps = 40
    orphan = mp.mpc(mp.mpf("-1"), mp.mpf("2"))
    poly = Polynomial.one(field=mp.mpc)
    serialized = SerializedDerivTable(
        shift=mp.mpf(0),
        base=mp.mpf(2),
        pole_list_x=(orphan,),
        coordinates=("zzb",),
        numerators={"zzb": [[poly]]},
    )
    with pytest.raises(ValueError, match="conjugate partner"):
        drop_complex_pole_pairs(serialized)


def test_synthetic_conjugate_pair_rescales_correctly():
    """Hand-built table with one conjugate pair and a known numerator: the
    output is the input scaled by (x - p)(x - p*) at every real x."""
    mp.mp.dps = 40
    p = mp.mpc(mp.mpf("-2"), mp.mpf("3"))
    p_conj = mp.mpc(mp.mpf("-2"), mp.mpf("-3"))
    poly = Polynomial([mp.mpf(1), mp.mpf(1)], field=mp.mpf)  # x + 1
    serialized = SerializedDerivTable(
        shift=mp.mpf(0),
        base=mp.mpf("0.5"),
        pole_list_x=(p, p_conj),
        coordinates=("zzb",),
        numerators={"zzb": [[poly]]},
    )

    dropped = drop_complex_pole_pairs(serialized)
    assert dropped.pole_list_x == ()
    # Numerator unchanged.
    assert dropped.numerators["zzb"][0][0].coeffs == poly.coeffs

    for x_val in (mp.mpf("1.5"), mp.mpf("5.0"), mp.mpf("-3.0")):
        # |x - p|^2 = (x - Re p)^2 + (Im p)^2 = (x + 2)^2 + 9
        q = (x_val + mp.mpf(2)) ** 2 + mp.mpf(9)
        before = serialized.evaluate("zzb", x_val)[0][0]
        after = dropped.evaluate("zzb", x_val)[0][0]
        expected = mp.mpc(before) * q
        assert abs(mp.mpc(after) - expected) < mp.mpf("1e-30")


def test_lower_positive_real_poles_is_noop_without_positive_groups():
    serialized = _strip_c_serialized()

    lowered = lower_positive_real_poles(serialized)

    assert lowered.selected_groups == ()
    assert lowered.selected_block is None
    assert lowered.remainder_block is serialized
    assert lowered.cleared_block is serialized
    assert lowered.interpolated_remainder_block is None
    assert lowered.cleared_interpolated_block is None


def test_lower_positive_real_poles_ising_split_reconstructs_original(
    ising_prefactored_serialized,
):
    mp.mp.dps = 80
    rel_tol = mp.mpf("1e-24")
    lowered = lower_positive_real_poles(ising_prefactored_serialized)

    assert lowered.selected_block is not None
    assert len(lowered.selected_groups) > 0
    assert not _has_positive_real_pole(lowered.remainder_block)

    safe_xs = _sample_nonpole_xs(
        [
            ising_prefactored_serialized,
            lowered.selected_block,
            lowered.remainder_block,
            lowered.cleared_block,
        ]
    )
    assert safe_xs

    for x in safe_xs:
        Delta = ising_prefactored_serialized.shift + x

        original = ising_prefactored_serialized.evaluate("zzb", Delta)
        split = _sum_tables(
            lowered.selected_block.evaluate("zzb", Delta),
            lowered.remainder_block.evaluate("zzb", Delta),
        )
        split_err = _max_table_error(original, split)
        split_scale = max(_table_scale(original), mp.mpf(1))
        assert split_err < rel_tol * split_scale

        cleared = lowered.cleared_block.evaluate("zzb", Delta)
        factor = _positive_clearing_value(lowered.selected_groups, x)
        expected_cleared = _scale_table(original, factor)
        cleared_err = _max_table_error(expected_cleared, cleared)
        cleared_scale = max(_table_scale(expected_cleared), mp.mpf(1))
        assert cleared_err < rel_tol * cleared_scale


@pytest.mark.parametrize("pole_selection", ["rightmost", "residue"])
def test_lower_positive_real_poles_interpolated_remainder_is_pmp_admissible(
    ising_prefactored_serialized,
    pole_selection,
):
    mp.mp.dps = 80
    lowered = lower_positive_real_poles(
        ising_prefactored_serialized,
        npoles=6,
        pole_selection=pole_selection,
    )

    assert lowered.interpolated_remainder_block is not None
    assert lowered.cleared_interpolated_block is not None
    assert lowered.requested_npoles == 6
    assert lowered.pole_selection == pole_selection
    assert lowered.kept_npoles is not None
    assert 0 < lowered.kept_npoles < lowered.requested_npoles
    assert len(lowered.sample_points_x) > 0
    assert len(lowered.interpolated_remainder_block.pole_list_x) == lowered.kept_npoles
    assert not _has_positive_real_pole(lowered.interpolated_remainder_block)

    F = F_minus_pairs(4)
    objective = [mp.mpf(0)] * len(F)
    normalization = [mp.mpf(1)] + [mp.mpf(0)] * (len(F) - 1)
    build_pmp_doc(
        [lowered.remainder_block],
        F_pairs=F,
        objective=objective,
        normalization=normalization,
    )
    build_pmp_doc(
        [lowered.cleared_block],
        F_pairs=F,
        objective=objective,
        normalization=normalization,
    )
    build_pmp_doc(
        [lowered.interpolated_remainder_block],
        F_pairs=F,
        objective=objective,
        normalization=normalization,
    )
    build_pmp_doc(
        [lowered.cleared_interpolated_block],
        F_pairs=F,
        objective=objective,
        normalization=normalization,
    )


# ---------------------------------------------------------------------------
# v^{Delta_phi} prefactor: closed-form Taylor coefficients at z = zbar = 1/2.
# ---------------------------------------------------------------------------


def test_v_prefactor_delta_zero_is_identity():
    """v^0 = 1, so the Taylor table is 1 at (0, 0) and zero everywhere else."""
    pre = v_prefactor_taylor(mp.mpf(0), 3, 3)
    for m in range(4):
        for n in range(4):
            expected = mp.mpf(1) if (m, n) == (0, 0) else mp.mpf(0)
            assert pre[m][n] == expected, (m, n, pre[m][n])


def test_v_prefactor_delta_one_matches_closed_form():
    """v^1 = (1/2 - a)(1/2 - b) = 1/4 - a/2 - b/2 + a b at (a, b) = 0."""
    pre = v_prefactor_taylor(mp.mpf(1), 2, 2)
    expected = {
        (0, 0): mp.mpf("0.25"),
        (1, 0): mp.mpf("-0.5"),
        (0, 1): mp.mpf("-0.5"),
        (1, 1): mp.mpf(1),
    }
    for m in range(3):
        for n in range(3):
            want = expected.get((m, n), mp.mpf(0))
            assert pre[m][n] == want, (m, n, pre[m][n], want)


def test_v_prefactor_delta_two_matches_closed_form():
    """v^2 = (1/2 - a)^2 (1/2 - b)^2 = (1/4 - a + a^2)(1/4 - b + b^2)."""
    # 1D factor f(t) = 1/4 - t + t^2 -> f = [1/4, -1, 1, 0, ...].
    f = [mp.mpf("0.25"), mp.mpf(-1), mp.mpf(1)]
    pre = v_prefactor_taylor(mp.mpf(2), 3, 3)
    for m in range(4):
        for n in range(4):
            want = (f[m] if m < len(f) else mp.mpf(0)) * (
                f[n] if n < len(f) else mp.mpf(0)
            )
            assert pre[m][n] == want, (m, n, pre[m][n], want)


def test_v_prefactor_noninteger_delta_evaluates_at_central_point():
    """At (a, b) = (0, 0) the Taylor evaluation is just (0,0); v^{Delta_phi}(1/2)
    = (1/4)^{Delta_phi}. The Taylor convolution at a generic point is harder to
    spot-check, but the central-point value must match (1/4)^Delta exactly."""
    mp.mp.dps = 50
    Delta = mp.mpf("0.6")
    pre = v_prefactor_taylor(Delta, 5, 5)
    expected = mp.mpf("0.25") ** Delta
    assert abs(pre[0][0] - expected) < mp.mpf("1e-45")


def test_crossing_helpers_validate_numeric_inputs_and_integral_cutoffs():
    from fractions import Fraction
    import numpy as np

    table = _scalar_table([[1]])
    for invalid in (True, mp.nan):
        with pytest.raises(ValueError, match='Delta_phi'):
            v_prefactor_taylor(invalid, 1, 1)
        with pytest.raises(ValueError, match='chiral_norm'):
            apply_norm_base_scaling(table, invalid)
        with pytest.raises(ValueError, match='prefactor_taylor coefficient'):
            apply_prefactor_to_table(table, [[invalid]])
    for invalid in (0, mp.mpc(1, '1e-100')):
        with pytest.raises(ValueError, match='chiral_norm'):
            apply_norm_base_scaling(table, invalid)
    for invalid in (1.5,):
        for counts in ((invalid, 1), (1, invalid)):
            with pytest.raises(ValueError, match='[mn]_max'):
                v_prefactor_taylor('.7', *counts)
    result = v_prefactor_taylor(np.int64(1), Fraction(2), np.int64(1))
    assert (len(result), len(result[0])) == (3, 2)
    assert apply_norm_base_scaling(table, np.int64(1)).base == table.base


# ---------------------------------------------------------------------------
# apply_prefactor_to_table: 2D convolution of the Taylor prefactor into every
# numerator polynomial, preserving the damped-rational denominator.
# ---------------------------------------------------------------------------


def _scalar_table(values, shift=mp.mpf(0), base=mp.mpf("0.5"), poles=()):
    """Build a single-coordinate SerializedDerivTable from a (m, n) array of
    scalars, each entry stored as a constant polynomial in x."""
    rows = [
        [Polynomial([mp.mpf(v)], field=mp.mpf) for v in row] for row in values
    ]
    return SerializedDerivTable(
        shift=shift,
        base=base,
        pole_list_x=tuple(poles),
        coordinates=("zzb",),
        numerators={"zzb": rows},
    )


def _assert_table_coeffs_close(table, expected, tol):
    """Assert numerators[coord][m][n] matches `expected[m][n]` as a polynomial
    coefficient list (within `tol`)."""
    for coord, rows in table.numerators.items():
        for m, row in enumerate(rows):
            for n, poly in enumerate(row):
                want = expected[m][n]
                got = poly.coeffs
                assert len(got) == len(want), (m, n, got, want)
                for g, w in zip(got, want):
                    assert abs(mp.mpc(g) - mp.mpc(w)) < tol, (m, n, g, w)


@pytest.mark.parametrize("multiplicity", [0])
def test_pole_split_rejects_invalid_multiplicities(multiplicity):
    table = _scalar_table([[1]], poles=(1, -1))
    with pytest.raises(ValueError, match="multiplicity.*positive integer"):
        split_serialized_block_by_pole_groups(table, [(1, multiplicity)])


@pytest.mark.parametrize("multiplicity", [3])
def test_pole_split_reconstructs_repeated_poles(multiplicity):
    with mp.workdps(60):
        table = _scalar_table([[1, 0], [2, 3]], poles=(1,) * multiplicity + (-1,))
        selected, remainder = split_serialized_block_by_pole_groups(
            table, [(1, mp.mpf(multiplicity))],
        )
        assert selected.pole_list_x == (1,) * multiplicity
        assert remainder.pole_list_x == (-1,)
        for delta in [mp.mpf(".3"), mp.mpf("1.7"), mp.mpf(4)]:
            original = table.evaluate("zzb", delta)
            reconstructed = _sum_tables(
                selected.evaluate("zzb", delta), remainder.evaluate("zzb", delta),
            )
            assert _max_table_error(original, reconstructed) < mp.mpf("1e-55")


def test_positive_pole_controls_use_integer_validation():
    from fractions import Fraction
    import numpy as np

    table = _scalar_table([[1]], poles=(1, -1))
    regular = _scalar_table([[1]], poles=(-1,))
    for invalid in (True, 1.5):
        with pytest.raises(ValueError, match='strip_count'):
            select_positive_real_root_groups(table, invalid)
        with pytest.raises(ValueError, match='multiplicity.*positive integer'):
            split_serialized_block_by_pole_groups(table, [(1, invalid)])
        for function in (interpolate_serialized_block, lower_positive_real_poles):
            with pytest.raises(ValueError, match='npoles'):
                function(regular, npoles=invalid)
            with pytest.raises(ValueError, match='n_sample_points'):
                function(regular, npoles=1, n_sample_points=invalid)
    for valid in (Fraction(1), np.int64(1)):
        assert select_positive_real_root_groups(table, valid) == [(1, 1)]
        selected, remainder = split_serialized_block_by_pole_groups(table, [(Fraction(1), valid)])
        assert selected.pole_list_x == (1,)
        assert remainder.pole_list_x == (-1,)
        interpolated, nodes = interpolate_serialized_block(regular, valid, n_sample_points=valid)
        assert interpolated.pole_list_x == (-1,)
        assert len(nodes) == 1
    for roots in ((Fraction(1), Fraction(-1)), ('1', '-1')):
        literal_table = _scalar_table([[1]], poles=roots)
        result = build_prefactored_pmp_doc([literal_table], [(0, 0)], objective=[0])
        assert result.positive_pole_lowerings[0].selected_groups == ((1, 1),)
        assert result.positivity_blocks[0].numerators['zzb'][0][0].coeffs == [-1, 1]


def test_pole_clearing_rejects_mismatched_derivative_shapes():
    selected = _scalar_table([[1, 2]], poles=(1,))
    for values in ([[3]], [[3, 4], [5, 6]]):
        remainder = _scalar_table(values, poles=(-1,))
        with pytest.raises(ValueError, match='numerator shape'):
            build_positive_cleared_block(selected, remainder, [(1, 1)])


def test_thin_conjugate_pair_matching_checks_width_and_multiplicity():
    with mp.workdps(40):
        width = mp.mpf('1e-70')
        upper, lower = mp.mpc(1, width), mp.mpc(1, -width)
        invalid_lists = ((upper, upper, lower),
                         (upper, mp.mpc(1, -2*width)),
                         (upper, mp.mpc(1+mp.mpf('1e-38'), -width)),
                         (mp.mpc('1e100', width), mp.mpc('1e100', -2*width)))
        for poles in invalid_lists:
            with pytest.raises(ValueError, match='conjugate partner'):
                drop_complex_pole_pairs(_scalar_table([[1]], poles=poles))
        # Cleanup cannot manufacture an exact pair from unequal widths.
        unequal = invalid_lists[1]
        assert _remove_selected_root_groups(unequal, []) == list(unequal)
        assert _remove_selected_root_groups((upper, lower), [(upper, 1)]) == [lower]
    with mp.workdps(3):
        # Exact dyadic roots have no stored mantissa floor to raise the caller
        # precision. A low-precision tolerance must still reject a wrong width.
        width = mp.ldexp(1, -1000)
        poles = (mp.mpc(1, width), mp.mpc(1, -2*width))
        with pytest.raises(ValueError, match='conjugate partner'):
            drop_complex_pole_pairs(_scalar_table([[1]], poles=poles))
        table = _scalar_table([[1]], poles=(1, -1))
        table.numerators['zzb'][0][0] = Polynomial([mp.mpc(1, 1)], field=mp.mpc)
        selected, remainder = split_serialized_block_by_pole_groups(table, [(1, 1)])
        assert selected.numerators['zzb'][0][0].coeffs == [mp.mpc('.5', '.5')]
        assert remainder.numerators['zzb'][0][0].coeffs == [mp.mpc('-.5', '-.5')]


def test_apply_prefactor_unity_leaves_table_unchanged():
    """The Delta_phi = 0 prefactor is the identity 2D table, so the convolution
    returns the input unchanged."""
    serialized = _strip_c_serialized()
    pre = v_prefactor_taylor(mp.mpf(0), 3, 3)
    result = apply_prefactor_to_table(serialized, pre)
    for coord, rows in serialized.numerators.items():
        for row_in, row_out in zip(rows, result.numerators[coord]):
            for p_in, p_out in zip(row_in, row_out):
                assert p_out.coeffs == p_in.coeffs


@pytest.mark.parametrize("prefactor", [[[1], [2, 3]]])
def test_apply_prefactor_rejects_ragged_prefactors(prefactor):
    table = _scalar_table([[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="prefactor_taylor.*rectangular"):
        apply_prefactor_to_table(table, prefactor)


@pytest.mark.parametrize("values", [[[1, 2], [3]]])
def test_apply_prefactor_rejects_ragged_numerator_tables(values):
    with pytest.raises(ValueError, match="numerators.*rectangular"):
        apply_prefactor_to_table(_scalar_table(values), [[1]])


def test_apply_prefactor_smaller_rectangle_is_zero_padded():
    table = _scalar_table([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = apply_prefactor_to_table(table, [[2, 3], [5, 7]])
    assert [[poly.evaluate(0) for poly in row] for row in result.numerators["zzb"]] == [
        [2, 3, 0], [5, 7, 0], [0, 0, 0],
    ]
    ignored = apply_prefactor_to_table(_scalar_table([[1]]), [[2, 'unused'], [object(), mp.nan]])
    assert ignored.numerators['zzb'][0][0].coeffs == [2]
    mixed = _scalar_table([[1, 2, 3]])
    mixed.coordinates = ('zzb', 'yyb')
    mixed.numerators['yyb'] = _scalar_table([[4], [5], [6]]).numerators['zzb']
    ignored = apply_prefactor_to_table(mixed, [[1, 0, 0], [0, object(), 0], [0, 0, mp.nan]])
    for coord in mixed.coordinates:
        assert [[p.coeffs for p in row] for row in ignored.numerators[coord]] == [
            [p.coeffs for p in row] for row in mixed.numerators[coord]
        ]


def test_apply_prefactor_preserves_damped_rational_metadata():
    """The cross-ratio prefactor only acts on the numerator polynomials; shift,
    base, and pole_list_x are carried through unchanged."""
    serialized = _strip_c_serialized()
    pre = v_prefactor_taylor(mp.mpf("0.3"), 3, 3)
    result = apply_prefactor_to_table(serialized, pre)
    assert result.shift == serialized.shift
    assert result.base == serialized.base
    assert result.pole_list_x == serialized.pole_list_x
    assert result.coordinates == serialized.coordinates


def test_apply_prefactor_on_unit_delta_function_recovers_prefactor():
    """If the block table is the unit delta function -- 1 at (0, 0), 0
    elsewhere, all as constant polynomials -- then the convolution at (m, n)
    is just `prefactor[m][n]` itself."""
    delta_table = _scalar_table(
        [[mp.mpf(1 if (m, n) == (0, 0) else 0) for n in range(3)] for m in range(3)]
    )
    pre = v_prefactor_taylor(mp.mpf(1), 2, 2)
    result = apply_prefactor_to_table(delta_table, pre)
    for m in range(3):
        for n in range(3):
            poly = result.numerators["zzb"][m][n]
            want = pre[m][n] if (m < len(pre) and n < len(pre[0])) else mp.mpf(0)
            # Constant polynomial (degree 0) with value want -- or the zero
            # polynomial if want is 0.
            if want == 0:
                assert poly.coeffs == [mp.mpf(0)]
            else:
                assert len(poly.coeffs) == 1
                assert abs(poly.coeffs[0] - want) < mp.mpf("1e-45")


def test_apply_prefactor_composition_equals_combined_power():
    """v^a * v^b = v^(a+b) on the real axis, so convolving with v^a and then
    with v^b should match convolving once with v^(a+b)."""
    serialized = _strip_c_serialized(m_max=3, n_max=3, q_order=6)
    a = mp.mpf("1")
    b = mp.mpf("2")
    pre_a = v_prefactor_taylor(a, 3, 3)
    pre_b = v_prefactor_taylor(b, 3, 3)
    pre_ab = v_prefactor_taylor(a + b, 3, 3)

    composed = apply_prefactor_to_table(
        apply_prefactor_to_table(serialized, pre_a), pre_b
    )
    direct = apply_prefactor_to_table(serialized, pre_ab)

    # Strip-c coefficients can be O(1e20); use relative tolerance.
    rel_tol = mp.mpf("1e-30")
    for row_c, row_d in zip(composed.numerators["zzb"], direct.numerators["zzb"]):
        for poly_c, poly_d in zip(row_c, row_d):
            assert len(poly_c.coeffs) == len(poly_d.coeffs)
            for cc, dd in zip(poly_c.coeffs, poly_d.coeffs):
                diff = abs(mp.mpc(cc) - mp.mpc(dd))
                mag = max(abs(mp.mpc(cc)), abs(mp.mpc(dd)), mp.mpf(1))
                assert diff < rel_tol * mag, (cc, dd, diff, mag)


def test_apply_prefactor_hand_computed_2x2():
    """End-to-end spot check on a hand-built 2x2 table.

    Take a numerator table with all entries = 1 (the constant polynomial),
    convolve with v^1 = 1/4 - a/2 - b/2 + ab. The Leibniz convolution gives

        (P * G)[m][n] = sum_{i<=m, j<=n} P[i][j] G[m-i][n-j]
                      = sum_{i<=m, j<=n} P[i][j]                (since G == 1)
                      = (sum_i P[i][.]) row-cumulative

    Concretely with P from v^1 (only (0,0), (1,0), (0,1), (1,1) nonzero):
        (P*G)[0][0] = 1/4
        (P*G)[1][0] = 1/4 + (-1/2) = -1/4
        (P*G)[0][1] = 1/4 + (-1/2) = -1/4
        (P*G)[1][1] = 1/4 + (-1/2) + (-1/2) + 1 = 1/4
    """
    G = _scalar_table([[mp.mpf(1), mp.mpf(1)], [mp.mpf(1), mp.mpf(1)]])
    pre = v_prefactor_taylor(mp.mpf(1), 1, 1)
    result = apply_prefactor_to_table(G, pre)
    expected = [
        [[mp.mpf("0.25")], [mp.mpf("-0.25")]],
        [[mp.mpf("-0.25")], [mp.mpf("0.25")]],
    ]
    _assert_table_coeffs_close(result, expected, mp.mpf("1e-45"))


# ---------------------------------------------------------------------------
# F_minus index set.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lam, expected",
    [
        (0, []),
        (1, [(1, 0)]),
        (2, [(1, 0)]),
        (3, [(1, 0), (2, 1), (3, 0)]),
        (4, [(1, 0), (2, 1), (3, 0)]),  # m+n must be <= 4 and odd
        (5, [(1, 0), (2, 1), (3, 0), (3, 2), (4, 1), (5, 0)]),
        (6, [(1, 0), (2, 1), (3, 0), (3, 2), (4, 1), (5, 0)]),
    ],
)
def test_F_minus_pairs_explicit(lam, expected):
    assert F_minus_pairs(lam) == expected


@pytest.mark.parametrize("lam", [0, 1, 2, 3, 4, 5, 7, 11])
def test_F_minus_pairs_satisfy_invariants(lam):
    """Every (m, n) in F_minus_pairs has m + n odd, n <= m, m + n <= lam."""
    pairs = F_minus_pairs(lam)
    for m, n in pairs:
        assert (m + n) % 2 == 1
        assert n <= m
        assert m + n <= lam
    # Sorted lexicographically.
    assert pairs == sorted(pairs)


def test_F_minus_pairs_yyb_matches_zzb():
    assert F_minus_pairs(7, coordinate="yyb") == F_minus_pairs(7, coordinate="zzb")


def test_F_minus_pairs_rejects_unsupported_coordinate():
    with pytest.raises(ValueError, match="supports identical-scalar coordinates"):
        F_minus_pairs(4, coordinate="qqb")
