"""Tests for JSON and Mathematica .m writers."""

from __future__ import annotations

import json

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyderivs import (
    B_qqb_poly_table,
    B_yyb_poly_table,
    B_zzb_poly_table,
    F_q_poly_table,
    F_y_poly_table,
    F_z_poly_table,
    G_qqb_poly_table,
    G_qqb_ws_poly_table,
    G_qqb_ws_radial_poly_table,
    G_ws_poly_table,
    G_ws_radial_poly_table,
    G_xt_poly_table,
    G_xt_radial_poly_table,
    G_yyb_poly_table,
    G_zzb_poly_table,
    interpolate_delta_table,
    parity_projected_B_qqb_poly_table,
    parity_projected_B_qqb_ws_poly_table,
    parity_projected_B_qqb_ws_radial_poly_table,
    parity_projected_B_ws_poly_table,
    parity_projected_B_ws_radial_poly_table,
    parity_projected_B_xt_poly_table,
    parity_projected_B_xt_radial_poly_table,
    parity_projected_B_yyb_poly_table,
    parity_projected_B_zzb_poly_table,
)
from virasoro_blocks.serialization import serialize_deriv_tables
from virasoro_blocks.truncation import kept_pairs
from virasoro_blocks.writers import (
    _num_str,
    mathematica_filename,
    write_json,
    write_mathematica,
)


C = mp.mpf("10")
SCALAR_EXT = [mp.mpf("0.15")] * 4
SPIN = 2


def _serialized():
    mp.mp.dps = 40
    b = b_from_c(C)
    exact = G_zzb_poly_table(SPIN, SCALAR_EXT, 2, 2, b=b, q_order=8)
    return serialize_deriv_tables({"zzb": exact})


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


def test_json_required_fields():
    s = _serialized()
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8)
    )
    for field in (
        "central_charge",
        "h_external",
        "hbar_external",
        "delta_external",
        "spin_external",
        "basis",
        "parity_projection",
        "parity_basis",
        "parity_external",
        "parity_exchanged",
        "label",
        "spin",
        "variable_minus_x",
        "exp_base",
        "order",
        "lambda",
        "coordinates",
        "pole_list_x",
        "index_names",
        "derivs",
        "npoles",
    ):
        assert field in doc, f"missing field: {field}"


def test_json_spin_and_order():
    s = _serialized()
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8)
    )
    assert doc["spin"] == SPIN
    assert doc["order"] == 8
    assert doc["spin_external"] == [0, 0, 0, 0]
    assert doc["basis"] == "scalar_G"
    assert doc["parity_projection"] is None
    assert doc["parity_basis"] is None
    assert doc["parity_external"] is None
    assert doc["parity_exchanged"] is None
    assert doc["lambda"] == 2


def test_json_half_integer_spin_external_roundtrip():
    s = _serialized()
    half_h = [mp.mpf("0.5")] * 4
    half_hbar = [mp.mpf("0.0")] * 4
    doc = json.loads(
        write_json(s, c=C, h_ext=half_h, hbar_ext=half_hbar, spin=mp.mpf("1.5"), q_order=8)
    )
    assert doc["spin"] == 1.5
    assert doc["spin_external"] == [0.5, 0.5, 0.5, 0.5]
    assert doc["basis"] == "ordered_B"
    assert doc["label"] == "h_hbar"


def test_json_parity_metadata_roundtrip():
    s = _serialized()
    doc = json.loads(
        write_json(
            s,
            c=C,
            h_ext=SCALAR_EXT,
            hbar_ext=SCALAR_EXT,
            spin=SPIN,
            q_order=8,
            parity_external=["1", "-1", "i", "-i"],
        )
    )
    assert doc["parity_basis"] == "diagonal"
    assert doc["parity_external"] == ["1", "-1", "i", "-i"]


def test_json_projected_basis_metadata_roundtrip():
    s = _serialized()
    doc = json.loads(
        write_json(
            s,
            c=C,
            h_ext=[mp.mpf("0.5")] * 4,
            hbar_ext=[mp.mpf("0.0")] * 4,
            spin=mp.mpf("1.5"),
            q_order=8,
            basis="parity_projected_B",
            parity_projection="-",
            parity_external=["1", "1", "1", "1"],
            parity_exchanged="-1",
        )
    )
    assert doc["basis"] == "parity_projected_B"
    assert doc["parity_projection"] == "-"
    assert doc["parity_basis"] == "diagonal"
    assert doc["parity_external"] == ["1", "1", "1", "1"]
    assert doc["parity_exchanged"] == "-1"
    assert doc["label"] == "parity_pm"


def test_json_rejects_inconsistent_parity_projection_metadata():
    s = _serialized()
    with pytest.raises(ValueError):
        write_json(
            s,
            c=C,
            h_ext=SCALAR_EXT,
            hbar_ext=SCALAR_EXT,
            spin=SPIN,
            q_order=8,
            basis="scalar_G",
            parity_projection="+",
        )

    with pytest.raises(ValueError):
        write_json(
            s,
            c=C,
            h_ext=[mp.mpf("0.5")] * 4,
            hbar_ext=[mp.mpf("0.0")] * 4,
            spin=mp.mpf("1.5"),
            q_order=8,
            basis="parity_projected_B",
        )


def test_json_derivs_shape():
    """Scalar G with even ell on zzb uses blocks_3d-style jagged truncation:
    m + n <= Lambda and n <= m, so for --lambda=2 we get pairs (0,0), (1,0),
    (1,1), (2,0)."""
    s = _serialized()
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8)
    )
    derivs = doc["derivs"]
    assert len(derivs) == 1  # one coordinate: zzb
    zzb = derivs[0]
    assert [len(row) for row in zzb] == [1, 2, 1]
    assert all(len(coeffs) >= 1 for row in zzb for coeffs in row)


def test_json_poles_count_matches_serialized():
    s = _serialized()
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8)
    )
    assert len(doc["pole_list_x"]) == len(s.pole_list_x)


def test_num_str_formats_negative_imaginary_parts_without_plusminus():
    assert _num_str(mp.mpc(1, -2), 20) == "(1.0-2.0j)"
    # a precision mark on every Mathematica decimal (a bare short literal is MachinePrecision there)
    assert _num_str(mp.mpc(1, -2), 20, mma=True) == "(1.0`20 - 2.0`20*I)"


def test_num_str_hides_guard_digit_noise_in_real_values():
    # Pin the precision: at dps = 50 the literal retains its ~1e-52 tail (so the
    # guard-trim path is genuinely exercised) yet the tail is within the writer's
    # noise tolerance 10**-(dps-5), so it is hidden. Without this, a prior
    # parallel test leaving a higher dps would expose the tail and fail.
    mp.mp.dps = 50
    noisy = mp.mpf("1.5000000000000000000000000000000000000000000000005346")
    assert noisy != mp.mpf("1.5")  # the guard-digit tail is present...
    assert _num_str(noisy, 50) == "1.5"  # ...and the writer hides it


def test_strip_json_output_avoids_guard_digit_noise_and_plusminus():
    mp.mp.dps = 50
    b = b_from_c(C)
    exact = G_zzb_poly_table(0, SCALAR_EXT, 2, 2, b=b, q_order=6)
    s = serialize_deriv_tables({"zzb": exact})
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=0, q_order=6)
    )
    assert all("+-" not in pole for pole in doc["pole_list_x"])


def test_json_npoles_defaults_to_pole_count():
    s = _serialized()
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8)
    )
    assert doc["npoles"] == len(s.pole_list_x)


def test_json_npoles_explicit():
    s = _serialized()
    doc = json.loads(
        write_json(
            s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8,
            npoles=6
        )
    )
    assert doc["npoles"] == 6


def test_json_coeff_roundtrip():
    """Constant term of the (0,0) numerator parses back to the original value."""
    mp.mp.dps = 40
    s = _serialized()
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8)
    )
    coeff_str = doc["derivs"][0][0][0][0]  # coord=zzb, m=0, n=0, degree-0 term
    with mp.workdps(doc["precision_actual"]):
        coeff_parsed = mp.mpf(coeff_str)
        coeff_orig = mp.re(s.numerators["zzb"][0][0].coeffs[0])
        assert abs(coeff_parsed - coeff_orig) < mp.mpf("1e-30")


def test_json_optional_fields_absent_by_default():
    s = _serialized()
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8)
    )
    assert doc["precision_actual"] == s.working_dps
    for field in ("precision_input", "spin_input", "output_template", "args"):
        assert field not in doc


def test_json_optional_fields_present_when_supplied():
    s = _serialized()
    doc = json.loads(
        write_json(
            s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8,
            precision_input=120, precision_actual=133,
            spin_input=[0, 2, 4],
            output_template="out/deriv_{spin}.json",
            args="--spin 2",
        )
    )
    assert doc["precision_input"] == 120
    assert doc["precision_actual"] == 133
    assert doc["spin_input"] == [0, 2, 4]
    assert doc["output_template"] == "out/deriv_{spin}.json"
    assert doc["args"] == "--spin 2"


def test_json_precision_actual_is_the_table_working_precision():
    """A lower caller precision must not discard a table's working precision."""
    s = _serialized()
    mp.mp.dps = 50
    doc = json.loads(
        write_json(
            s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8,
            precision_input=40,
        )
    )
    assert doc["precision_input"] == 40
    assert doc["precision_actual"] == max(50, s.working_dps)


def test_json_multiple_coordinates():
    """Both zzb and qqb appear in derivs list."""
    mp.mp.dps = 40
    b = b_from_c(C)
    zzb = G_zzb_poly_table(SPIN, SCALAR_EXT, 1, 1, b=b, q_order=8)
    qqb = G_qqb_poly_table(SPIN, SCALAR_EXT, 1, 1, b=b, q_order=8)
    s = serialize_deriv_tables({"zzb": zzb, "qqb": qqb})
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8)
    )
    assert set(doc["coordinates"]) == {"zzb", "qqb"}
    assert len(doc["derivs"]) == 2


def test_json_interpolated_path():
    """JSON writer accepts InterpolatedDeltaTable-based SerializedDerivTable."""
    mp.mp.dps = 40
    b = b_from_c(C)
    exact = G_zzb_poly_table(SPIN, SCALAR_EXT, 1, 1, b=b, q_order=8)
    interp = interpolate_delta_table(exact, npoles=4)
    s = serialize_deriv_tables({"zzb": interp})
    doc = json.loads(
        write_json(
            s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=SPIN, q_order=8,
            npoles=4
        )
    )
    assert doc["npoles"] == 4
    assert len(doc["pole_list_x"]) == len(s.pole_list_x)


# ---------------------------------------------------------------------------
# Mathematica .m writer
# ---------------------------------------------------------------------------


def test_mma_has_deriv_entries():
    s = _serialized()
    mma = write_mathematica(s, coordinate="zzb", ell=SPIN)
    assert "zzbDeriv[0,0]" in mma
    # (2,0) is on the n <= m boundary; (2,2) is pruned by m+n <= Lambda=2.
    assert "zzbDeriv[2,0]" in mma
    assert "zzbDeriv[2,2]" not in mma


def test_json_parity_smoothing_metadata():
    """Scalar G is z <-> zbar even at every spin: p_plus at odd ell too, and it records scalar_G_odd."""
    s = _serialized()
    doc_even = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=2, q_order=8)
    )
    assert doc_even["parity_smoothing"] == "p_plus"
    doc_odd = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=1, q_order=8)
    )
    assert doc_odd["parity_smoothing"] == "p_plus"
    assert doc_even["scalar_G_odd"] == doc_odd["scalar_G_odd"] == "even"


def test_mma_records_parity_smoothing():
    s = _serialized()
    mma_even = write_mathematica(s, coordinate="zzb", ell=2)
    assert 'paritySmoothing -> "p_plus"' in mma_even
    mma_odd = write_mathematica(s, coordinate="zzb", ell=1)
    assert 'paritySmoothing -> "p_plus"' in mma_odd
    assert 'scalarGOdd -> "even"' in mma_odd


def test_mma_has_kept_entries_only():
    """Only the blocks_3d-style kept (m, n) pairs appear; n > m and
    m + n > Lambda are pruned."""
    s = _serialized()
    mma = write_mathematica(s, coordinate="zzb", ell=SPIN)
    kept = {(0, 0), (1, 0), (1, 1), (2, 0)}
    for m in range(3):
        for n in range(3):
            present = f"zzbDeriv[{m},{n}]" in mma
            assert present == ((m, n) in kept), f"unexpected ({m},{n}) presence={present}"


def test_mma_has_shifted_poles():
    s = _serialized()
    mma = write_mathematica(s, coordinate="zzb")
    assert "shiftedPoles" in mma


def test_mma_self_contained_reconstruction_and_kinematics():
    """The .m footer carries everything needed to reconstruct/interpret the
    table without parsing the filename: the prefactor base and shift (required
    for evaluation), the variable, and the kinematic header fields."""
    s = _serialized()
    mma = write_mathematica(
        s, coordinate="zzb", ell=2, basis="scalar_G",
        c=mp.mpf("2.5"),
        h_ext=[mp.mpf("0.2")] * 4, hbar_ext=[mp.mpf("0.2")] * 4,
        q_order=12, npoles=6, variable="Delta",
    )
    # Reconstruction-critical (were missing before):
    assert "expBase ->" in mma
    assert "shift ->" in mma
    assert 'variable -> "Delta"' in mma
    # Filename-encoded fields now in-body:
    assert "centralCharge -> 2.5" in mma
    assert "hExternal -> {" in mma and "hbarExternal -> {" in mma
    assert "deltaExternal -> {" in mma and "spinExternal -> {" in mma
    assert "spin -> 2" in mma
    assert "lambda ->" in mma and "order -> 12" in mma and "npoles -> 6" in mma
    assert 'coordinate -> "zzb"' in mma


def test_mma_interpolation_metadata_parity_fields():
    mp.mp.dps = 40
    b = b_from_c(C)
    exact = G_zzb_poly_table(SPIN, SCALAR_EXT, 1, 1, b=b, q_order=8)
    interp = interpolate_delta_table(exact, npoles=4)
    s = serialize_deriv_tables({"zzb": interp})

    mma = write_mathematica(
        s,
        coordinate="zzb",
        precision_input=30,
        output_template="derivs_{spin}.json",
        args="--spin 2",
    )

    assert "samplePointsX -> {" in mma
    assert "precisionInput -> 30" in mma
    expected_precision = max(mp.mp.dps, s.working_dps)
    assert f"precisionActual -> {expected_precision}" in mma
    assert 'outputTemplate -> "derivs_{spin}.json"' in mma
    assert 'args -> "--spin 2"' in mma


def test_mma_has_basis_metadata():
    s = _serialized()
    mma = write_mathematica(
        s,
        coordinate="zzb",
        basis="parity_projected_B",
        parity_projection="+",
        parity_external=["1", "1", "1", "1"],
        parity_exchanged="-1",
    )
    assert 'basis -> "parity_projected_B"' in mma
    assert 'parityProjection -> "+"' in mma
    assert 'parityBasis -> "diagonal"' in mma
    assert 'parityExternal -> {"1", "1", "1", "1"}' in mma
    assert 'parityExchanged -> "-1"' in mma


def test_mma_rejects_inconsistent_parity_projection_metadata():
    s = _serialized()
    with pytest.raises(ValueError):
        write_mathematica(
            s,
            coordinate="zzb",
            basis="ordered_B",
            parity_projection="-",
        )

    with pytest.raises(ValueError):
        write_mathematica(
            s,
            coordinate="zzb",
            basis="parity_projected_B",
        )


def test_mma_arrow_syntax():
    """Each entry uses the Mathematica -> rule syntax."""
    s = _serialized()
    mma = write_mathematica(s, coordinate="zzb")
    assert "zzbDeriv[0,0] ->" in mma
    assert "shiftedPoles ->" in mma


def test_mma_coefficients_are_braced_list():
    """Coefficients for each (m, n) are wrapped in { }."""
    s = _serialized()
    mma = write_mathematica(s, coordinate="zzb")
    # Find the zzbDeriv[0,0] line and check it ends with {…}
    for line in mma.splitlines():
        if "zzbDeriv[0,0]" in line:
            assert "{" in line and "}" in line
            break


def test_mma_outer_braces():
    """The full output is a Mathematica list {...}."""
    s = _serialized()
    mma = write_mathematica(s, coordinate="zzb").strip()
    assert mma.startswith("{")
    assert mma.endswith("}")


# ---------------------------------------------------------------------------
# Filename helper
# ---------------------------------------------------------------------------


def test_mma_filename_structure():
    fname = mathematica_filename(
        "zzb",
        c=C,
        h_ext=SCALAR_EXT,
        hbar_ext=SCALAR_EXT,
        spin=SPIN,
        q_order=8,
        m_max=2,
        n_max=2,
        npoles=6,
    )
    assert fname.startswith("zzb")
    assert f"L{SPIN}" in fname
    assert "npoles6" in fname
    assert "order8" in fname
    assert fname.endswith(".m")


def test_mma_filename_ordered_basis_suffix():
    fname = mathematica_filename(
        "zzb",
        c=C,
        h_ext=[mp.mpf("0.5")] * 4,
        hbar_ext=[mp.mpf("0.0")] * 4,
        spin=mp.mpf("1.5"),
        q_order=8,
        m_max=2,
        n_max=2,
        basis="ordered_B",
    )
    assert "basisOrderedB" in fname


def test_mma_filename_projected_basis_suffix():
    fname = mathematica_filename(
        "ws",
        c=C,
        h_ext=[mp.mpf("0.5")] * 4,
        hbar_ext=[mp.mpf("0.0")] * 4,
        spin=mp.mpf("1.5"),
        q_order=8,
        m_max=2,
        n_max=2,
        basis="parity_projected_B",
        parity_projection="-",
    )
    assert "basisParityMinus" in fname


def test_mma_filename_no_npoles_omits_field():
    fname = mathematica_filename(
        "zzb",
        c=C,
        h_ext=SCALAR_EXT,
        hbar_ext=SCALAR_EXT,
        spin=SPIN,
        q_order=8,
        m_max=2,
        n_max=2,
    )
    assert "npoles" not in fname
    assert fname.endswith(".m")


def test_mma_filename_lambda_is_max_of_m_n():
    fname = mathematica_filename(
        "zzb",
        c=C,
        h_ext=SCALAR_EXT,
        hbar_ext=SCALAR_EXT,
        spin=SPIN,
        q_order=8,
        m_max=3,
        n_max=5,
    )
    assert "lambda5" in fname
    assert "nmax" not in fname


def test_mma_filename_c_appears():
    fname = mathematica_filename(
        "zzb",
        c=mp.mpf("25"),
        h_ext=SCALAR_EXT,
        hbar_ext=SCALAR_EXT,
        spin=0,
        q_order=10,
    )
    assert "c25" in fname


def test_mma_filename_includes_chiral_norm():
    from virasoro_blocks.recursion import DEFAULT_CHIRAL_NORM
    # Default value appears when not specified.
    fname_default = mathematica_filename(
        "zzb",
        c=C,
        h_ext=SCALAR_EXT,
        hbar_ext=SCALAR_EXT,
        spin=SPIN,
        q_order=8,
        m_max=2,
        n_max=2,
    )
    assert f"chiralNorm{float(DEFAULT_CHIRAL_NORM):g}" in fname_default

    # Explicit value overrides the default.
    fname_n32 = mathematica_filename(
        "zzb",
        c=C,
        h_ext=SCALAR_EXT,
        hbar_ext=SCALAR_EXT,
        spin=SPIN,
        q_order=8,
        m_max=2,
        n_max=2,
        chiral_norm=mp.mpf(32),
    )
    assert "chiralNorm32" in fname_n32
    assert fname_default != fname_n32


def test_mma_filename_lists_all_four_external_dimensions():
    """Two correlators with the same delta12/delta34 but different Σd_i must
    produce distinct filenames (the Virasoro prefactor depends on the sum)."""
    # Same delta12 = delta34 = 0 in both, but different total dimensions.
    h_a = [mp.mpf("0.10")] * 4
    h_b = [mp.mpf("0.30")] * 4
    fname_a = mathematica_filename(
        "zzb",
        c=C,
        h_ext=h_a,
        hbar_ext=h_a,
        spin=0,
        q_order=8,
        m_max=2,
        n_max=2,
    )
    fname_b = mathematica_filename(
        "zzb",
        c=C,
        h_ext=h_b,
        hbar_ext=h_b,
        spin=0,
        q_order=8,
        m_max=2,
        n_max=2,
    )
    assert "delta1-0.2" in fname_a
    assert "delta1-0.6" in fname_b
    assert fname_a != fname_b


def test_mma_filename_lists_external_spins_only_when_nonzero():
    """Scalar externals omit the spin labels; any nonzero spin shows all four."""
    scalar_h = [mp.mpf("0.5")] * 4
    fname_scalar = mathematica_filename(
        "zzb",
        c=C,
        h_ext=scalar_h,
        hbar_ext=scalar_h,
        spin=0,
        q_order=8,
        m_max=2,
        n_max=2,
    )
    assert "spin1" not in fname_scalar

    spinning_h = [mp.mpf("0.6"), mp.mpf("0.4"), mp.mpf("0.5"), mp.mpf("0.5")]
    spinning_hbar = [mp.mpf("0.4"), mp.mpf("0.6"), mp.mpf("0.5"), mp.mpf("0.5")]
    fname_spinning = mathematica_filename(
        "zzb",
        c=C,
        h_ext=spinning_h,
        hbar_ext=spinning_hbar,
        spin=0,
        q_order=8,
        m_max=2,
        n_max=2,
        basis="ordered_B",
    )
    # spin1 = 0.6 - 0.4 = 0.2, spin2 = -0.2, spin3 = 0, spin4 = 0.
    assert "spin1-0.2" in fname_spinning
    assert "spin2--0.2" in fname_spinning


# ---------------------------------------------------------------------------
# Cross-basis consistency at the JSON layer: parity_projected_B with scalar
# externals (h_ext == hbar_ext) and the '+' projection reproduces scalar_G at
# every spin, odd ell included. The symbolic-table identity is covered in
# test_external_weights / test_polyderivs / test_composer; this test pins it
# down to what the JSON writer actually serializes for end-users.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ell,parity", [(1, "+")])
def test_json_parity_projected_B_scalar_limit_matches_scalar_G(ell, parity):
    mp.mp.dps = 40
    b = b_from_c(C)
    q_order = 8
    m_max = n_max = 3

    g_table = G_zzb_poly_table(
        ell, SCALAR_EXT, m_max, n_max, b=b, q_order=q_order
    )
    g_serialized = serialize_deriv_tables({"zzb": g_table})
    g_doc = json.loads(
        write_json(
            g_serialized, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT,
            spin=ell, q_order=q_order,
        )
    )

    b_table = parity_projected_B_zzb_poly_table(
        ell, SCALAR_EXT, SCALAR_EXT, m_max, n_max,
        parity=parity, b=b, q_order=q_order,
    )
    b_serialized = serialize_deriv_tables({"zzb": b_table})
    b_doc = json.loads(
        write_json(
            b_serialized, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT,
            spin=ell, q_order=q_order,
            basis="parity_projected_B",
            parity_projection=parity,
            parity_external=["1", "1", "1", "1"],
            parity_exchanged="1",
        )
    )

    # Metadata: bases differ as expected, but the numeric scaffolding agrees.
    assert g_doc["basis"] == "scalar_G"
    assert b_doc["basis"] == "parity_projected_B"
    assert b_doc["parity_projection"] == parity
    assert g_doc["coordinates"] == b_doc["coordinates"]
    assert g_doc["lambda"] == b_doc["lambda"]
    assert g_doc["order"] == b_doc["order"]
    assert g_doc["spin"] == b_doc["spin"]

    # Pole list and Delta-shift agree exactly (same string representation).
    assert g_doc["pole_list_x"] == b_doc["pole_list_x"]
    assert g_doc["variable_minus_x"] == b_doc["variable_minus_x"]
    assert g_doc["exp_base"] == b_doc["exp_base"]

    # derivs[coord][m][n][k] agrees numerically (the symbolic identity is
    # tight; the writer rounds to ~30 sig figs).
    g_derivs = g_doc["derivs"][0]
    b_derivs = b_doc["derivs"][0]
    assert [len(row) for row in g_derivs] == [len(row) for row in b_derivs]
    # The scalar-limit identity is tight, but the optimized projected-B path
    # can differ from scalar_G in the last rounded digits because the flint
    # arithmetic is no longer assembled in exactly the same order.
    for m, (g_row, b_row) in enumerate(zip(g_derivs, b_derivs)):
        for n, (g_coeffs, b_coeffs) in enumerate(zip(g_row, b_row)):
            assert len(g_coeffs) == len(b_coeffs), (m, n)
            for k, (g_str, b_str) in enumerate(zip(g_coeffs, b_coeffs)):
                g_val = mp.mpf(g_str)
                b_val = mp.mpf(b_str)
                scale = max(abs(g_val), abs(b_val), mp.mpf(1))
                assert abs(g_val - b_val) <= mp.mpf("1e-28") * scale, (
                    f"(m,n,k)=({m},{n},{k}): {g_str} vs {b_str}"
                )


# ---------------------------------------------------------------------------
# End-to-end shape check: the serialized JSON's derivs row lengths must
# match the kept_pairs(coord, lambda, basis, ell, parity) policy across the
# full coord x basis x ell-parity matrix.
# ---------------------------------------------------------------------------


# The kept-pairs shape depends only on (coord, basis, ell, parity_projection),
# not on the external dimensions, so it is fine to pass scalar externals
# (h = hbar = SCALAR_EXT) into the spinning-externals builders here.
_G_BUILDERS_2D = {
    "zzb": G_zzb_poly_table,
    "yyb": G_yyb_poly_table,
    "qqb": G_qqb_poly_table,
    "xt": G_xt_poly_table,
    "ws": G_ws_poly_table,
    "qqb_ws": G_qqb_ws_poly_table,
}
_G_BUILDERS_RADIAL = {
    "xt_radial": G_xt_radial_poly_table,
    "ws_radial": G_ws_radial_poly_table,
    "qqb_ws_radial": G_qqb_ws_radial_poly_table,
}
_B_BUILDERS = {
    "zzb": B_zzb_poly_table,
    "yyb": B_yyb_poly_table,
    "qqb": B_qqb_poly_table,
}
_PB_BUILDERS_2D = {
    "zzb": parity_projected_B_zzb_poly_table,
    "yyb": parity_projected_B_yyb_poly_table,
    "qqb": parity_projected_B_qqb_poly_table,
    "xt": parity_projected_B_xt_poly_table,
    "ws": parity_projected_B_ws_poly_table,
    "qqb_ws": parity_projected_B_qqb_ws_poly_table,
}
_PB_BUILDERS_RADIAL = {
    "xt_radial": parity_projected_B_xt_radial_poly_table,
    "ws_radial": parity_projected_B_ws_radial_poly_table,
    "qqb_ws_radial": parity_projected_B_qqb_ws_radial_poly_table,
}
_F_BUILDERS = {
    "z": F_z_poly_table,
    "y": F_y_poly_table,
    "q": F_q_poly_table,
}


def _build_table(coord, basis, ell, parity, lam, *, b, q_order):
    """Dispatch to the appropriate poly_table builder for (coord, basis).

    We size the table at (m_max=n_max=lam) like the CLI does. The writer
    recovers Lambda from `max(rows, cols) - 1`, so under-sizing here (e.g.
    via `required_m_max_n_max` for folded odd ell) would yield the wrong
    Lambda downstream.
    """
    if basis == "scalar_G":
        if coord in _G_BUILDERS_2D:
            return _G_BUILDERS_2D[coord](
                ell, SCALAR_EXT, lam, lam, b=b, q_order=q_order
            )
        if coord in _G_BUILDERS_RADIAL:
            return _G_BUILDERS_RADIAL[coord](
                ell, SCALAR_EXT, lam, b=b, q_order=q_order
            )
    elif basis == "ordered_B":
        return _B_BUILDERS[coord](
            ell, SCALAR_EXT, SCALAR_EXT, lam, lam, b=b, q_order=q_order
        )
    elif basis == "parity_projected_B":
        if coord in _PB_BUILDERS_2D:
            return _PB_BUILDERS_2D[coord](
                ell, SCALAR_EXT, SCALAR_EXT, lam, lam,
                parity=parity, b=b, q_order=q_order,
            )
        if coord in _PB_BUILDERS_RADIAL:
            return _PB_BUILDERS_RADIAL[coord](
                ell, SCALAR_EXT, SCALAR_EXT, lam,
                parity=parity, b=b, q_order=q_order,
            )
    elif basis == "chiral_F":
        return _F_BUILDERS[coord](SCALAR_EXT, lam, b=b, q_order=q_order)
    raise ValueError(f"no builder for {coord!r} in basis {basis!r}")


def _write_doc(serialized, *, coord, basis, ell, parity, q_order):
    """Run the JSON writer with the metadata each basis demands."""
    if basis == "chiral_F":
        return json.loads(write_json(
            serialized, c=C, h_ext=SCALAR_EXT, hbar_ext=None,
            spin=None, q_order=q_order, basis="chiral_F",
        ))
    if basis == "ordered_B":
        return json.loads(write_json(
            serialized, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT,
            spin=ell, q_order=q_order, basis="ordered_B",
        ))
    if basis == "parity_projected_B":
        return json.loads(write_json(
            serialized, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT,
            spin=ell, q_order=q_order, basis="parity_projected_B",
            parity_projection=parity,
            parity_external=["1", "1", "1", "1"],
            parity_exchanged="1",
        ))
    return json.loads(write_json(
        serialized, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT,
        spin=ell, q_order=q_order,
    ))


# (coord, basis, ell, parity_projection). Covers each coordinate family once
# per basis it admits, and both projection signs / parities where relevant.
_SHAPE_CASES = [
    # scalar_G, non-folded coords (even and odd ell give the same shape policy
    # in the non-folded branch; one coordinate suffices).
    ("zzb", "scalar_G", 1, None),
    # scalar_G, folded coords: Lambda_+ for even ell, Lambda_- for odd ell.
    ("xt", "scalar_G", 0, None),
    ("xt", "scalar_G", 1, None),
    ("ws", "scalar_G", 2, None),
    ("qqb_ws", "scalar_G", 1, None),
    # scalar_G, radial slices: only n = 0.
    ("xt_radial", "scalar_G", 0, None),
    ("ws_radial", "scalar_G", 2, None),
    ("qqb_ws_radial", "scalar_G", 0, None),
    # ordered_B: full m + n <= Lambda triangle, both halves.
    ("yyb", "ordered_B", 1, None),
    # parity_projected_B, non-folded.
    ("zzb", "parity_projected_B", 0, "+"),
    ("yyb", "parity_projected_B", 1, "-"),
    ("qqb", "parity_projected_B", 2, "+"),
    # parity_projected_B, folded: both projection signs.
    ("xt", "parity_projected_B", 0, "+"),
    ("xt", "parity_projected_B", 0, "-"),
    ("ws", "parity_projected_B", 0, "+"),
    ("qqb_ws", "parity_projected_B", 0, "-"),
    # parity_projected_B, radial.
    ("xt_radial", "parity_projected_B", 0, "+"),
    ("ws_radial", "parity_projected_B", 0, "-"),
    ("qqb_ws_radial", "parity_projected_B", 0, "+"),
    # chiral_F: only m index, length Lambda + 1.
    ("q", "chiral_F", 0, None),
]


@pytest.mark.parametrize("coord,basis,ell,parity", _SHAPE_CASES)
def test_json_derivs_row_lengths_match_kept_pairs(coord, basis, ell, parity):
    """For every (coord, basis, ell, parity), the row lengths in the
    serialized JSON match kept_pairs(coord, Lambda, basis, ell, parity).
    """
    mp.mp.dps = 35
    b = b_from_c(C)
    q_order = 8
    lam = 3

    table = _build_table(
        coord, basis, ell, parity, lam, b=b, q_order=q_order,
    )
    serialized = serialize_deriv_tables({coord: table})
    doc = _write_doc(
        serialized, coord=coord, basis=basis, ell=ell, parity=parity,
        q_order=q_order,
    )

    pairs = kept_pairs(coord, lam, basis=basis, ell=ell, parity_projection=parity)
    expected_rows: dict[int, int] = {}
    for m, _n in pairs:
        expected_rows[m] = expected_rows.get(m, 0) + 1
    expected_lengths = [expected_rows[m] for m in sorted(expected_rows)]

    derivs = doc["derivs"][0]
    if basis == "chiral_F":
        # Chiral writer collapses the degenerate n = 0 column: derivs[coord]
        # is a list of m-rows of coefficient strings, not of n-lists.
        assert len(derivs) == len(expected_lengths), (
            f"{coord}/{basis}/ell={ell}: row count "
            f"{len(derivs)} != {len(expected_lengths)}"
        )
        assert all(isinstance(row, list) and row and isinstance(row[0], str)
                   for row in derivs)
    else:
        actual_lengths = [len(row) for row in derivs]
        assert actual_lengths == expected_lengths, (
            f"{coord}/{basis}/ell={ell}/parity={parity}: "
            f"got {actual_lengths}, expected {expected_lengths}"
        )
