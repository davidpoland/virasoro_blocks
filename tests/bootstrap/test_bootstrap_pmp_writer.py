"""Tests for `virasoro_bootstrap.pmp_writer`.

Smoke-tests the structure of the emitted JSON against pmp2sdp's schema
(`docs/json_schema/pmp_schema.json` in davidsd/sdpb), validating against
known shape constraints and exercising the full c=10 strip pipeline:
build a SerializedDerivTable, apply v^{Delta_phi}, drop complex pole
pairs, restrict to F_minus_pairs, and serialize.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    apply_prefactor_to_table,
    drop_complex_pole_pairs,
    v_prefactor_taylor,
)
from virasoro_bootstrap.pmp_writer import (
    build_pmp_doc,
    build_prefactored_pmp_doc,
    write_pmp_json,
)


C_STRIP = mp.mpf("10")
SCALAR_EXT = [mp.mpf("0.15")] * 4
ISING_C = mp.mpf("0.5")
ISING_SIGMA = mp.mpf("0.0625")


# ---------------------------------------------------------------------------
# Synthetic-block fixtures: hand-built SerializedDerivTables with simple
# polynomial entries, useful for shape and validation tests.
# ---------------------------------------------------------------------------


def _const_poly(value):
    return Polynomial([mp.mpf(value)], field=mp.mpf)


def _make_real_pole_block(rows_values, *, base="0.5", shift=0, poles=()):
    """Build a SerializedDerivTable from a 2D array of scalar polynomial values."""
    rows = [[_const_poly(v) for v in row] for row in rows_values]
    return SerializedDerivTable(
        shift=mp.mpf(shift),
        base=mp.mpf(base),
        pole_list_x=tuple(mp.mpf(p) for p in poles),
        coordinates=("zzb",),
        numerators={"zzb": rows},
    )


@pytest.fixture(scope="module")
def ising_prefactored_block():
    """The Ising sigma spin-0 block (Lambda = 4, q_order = 12), prefactored;
    built once per module (the tests only read it)."""
    lam, q_order = 4, 12
    mp.mp.dps = 80
    table = merged_delta_table(lambda c_value: G_zzb_poly_table(
            0,
            [ISING_SIGMA] * 4,
            lam,
            lam,
            c=c_value,
            q_order=q_order,
        ), c=ISING_C)
    serialized = serialize_deriv_tables({"zzb": table})
    return apply_prefactor_to_table(
        serialized,
        v_prefactor_taylor(ISING_SIGMA, lam, lam),
    )


def _has_positive_real_pole(serialized):
    tol = mp.mpf("1e-25")
    return any(
        abs(mp.im(mp.mpc(pole))) <= tol and mp.re(mp.mpc(pole)) > tol
        for pole in serialized.pole_list_x
    )


# ---------------------------------------------------------------------------
# Top-level structure.
# ---------------------------------------------------------------------------


def test_doc_has_required_top_level_fields():
    block = _make_real_pole_block(
        [[1, 2], [3, 4]], poles=(-1, -2)
    )
    doc = build_pmp_doc(
        [block],
        F_pairs=[(1, 0)],
        objective=[mp.mpf(0)],
        normalization=[mp.mpf(1)],
    )
    assert "objective" in doc
    assert "normalization" in doc
    assert "PositiveMatrixWithPrefactorArray" in doc
    # Schema mandates string entries for objective/normalization.
    assert all(isinstance(s, str) for s in doc["objective"])
    assert all(isinstance(s, str) for s in doc["normalization"])


def test_doc_omits_normalization_when_not_provided():
    block = _make_real_pole_block([[1]])
    doc = build_pmp_doc(
        [block],
        F_pairs=[(0, 0)],
        objective=[mp.mpf(0)],
    )
    assert "normalization" not in doc


def test_prefactored_doc_resizes_explicit_nodes_for_extra_polynomial_degree():
    block = _make_real_pole_block([[1]], base="0.5")
    block.sample_points_x = (mp.mpf("0.1"),)
    extra = Polynomial([mp.mpf(1), mp.mpf(2), mp.mpf(3)], field=mp.mpf)

    doc = build_prefactored_pmp_doc(
        [block],
        F_pairs=[(0, 0)],
        objective=[mp.mpf(0), mp.mpf(0)],
        normalization=[mp.mpf(1), mp.mpf(0)],
        explicit_sample_points=True,
        extra_polynomials_per_block=[(extra,)],
    ).doc

    entry = doc["PositiveMatrixWithPrefactorArray"][0]
    assert len(entry["samplePoints"]) == 3
    assert len(entry["sampleScalings"]) == 3
    assert len(entry["polynomials"][0][0][0]) == 3
    assert len(entry["polynomials"][0][0][1]) == 3


def test_doc_objective_normalization_length_matches_F_pairs():
    block = _make_real_pole_block(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    F = [(1, 0), (2, 0), (2, 1)]
    doc = build_pmp_doc(
        [block],
        F_pairs=F,
        objective=[mp.mpf(0)] * 3,
        normalization=[mp.mpf(1), mp.mpf(0), mp.mpf(0)],
    )
    assert len(doc["objective"]) == len(F)
    assert len(doc["normalization"]) == len(F)


# ---------------------------------------------------------------------------
# Per-block structure.
# ---------------------------------------------------------------------------


def test_block_polynomials_are_4_deep_with_1x1_matrix():
    """PMP polynomials shape: [matrix_row][matrix_col][polyvec_idx][monomial].
    For a 1x1 matrix the first two dims are length 1."""
    block = _make_real_pole_block([[1, 2], [3, 4]])
    F = [(0, 0), (1, 0)]
    doc = build_pmp_doc(
        [block], F_pairs=F, objective=[mp.mpf(0), mp.mpf(0)]
    )
    polys = doc["PositiveMatrixWithPrefactorArray"][0]["polynomials"]
    # Outer matrix-row dim
    assert len(polys) == 1
    # Matrix-col dim
    assert len(polys[0]) == 1
    # Polynomial-vector dim (one polynomial per F-pair)
    assert len(polys[0][0]) == len(F)
    # Each polynomial: a list of coefficient strings
    for poly in polys[0][0]:
        assert isinstance(poly, list)
        assert all(isinstance(c, str) for c in poly)


def test_block_damped_rational_shape():
    block = _make_real_pole_block(
        [[1]], base="0.25", poles=(-1, -2, -3)
    )
    doc = build_pmp_doc(
        [block], F_pairs=[(0, 0)], objective=[mp.mpf(0)]
    )
    dr = doc["PositiveMatrixWithPrefactorArray"][0]["DampedRational"]
    assert set(dr) == {"base", "constant", "poles"}
    assert isinstance(dr["base"], str)
    assert isinstance(dr["constant"], str)
    assert isinstance(dr["poles"], list)
    assert all(isinstance(p, str) for p in dr["poles"])
    assert len(dr["poles"]) == 3
    assert dr["constant"] == "1"


def test_block_polynomial_coefficients_round_trip_to_input():
    """Each emitted coefficient string parses back to the input mpf."""
    rows_values = [[7, mp.mpf("3.5")], [mp.mpf("-1.25"), mp.mpf("0")]]
    block = _make_real_pole_block(rows_values)
    F = [(0, 0), (0, 1), (1, 0), (1, 1)]
    doc = build_pmp_doc(
        [block], F_pairs=F, objective=[mp.mpf(0)] * 4
    )
    polys = doc["PositiveMatrixWithPrefactorArray"][0]["polynomials"][0][0]
    for (m, n), poly in zip(F, polys):
        expected = mp.mpf(rows_values[m][n])
        assert len(poly) == 1  # constant polynomial -> one coefficient
        assert abs(mp.mpf(poly[0]) - expected) < mp.mpf("1e-25")


def test_block_coefficients_with_tiny_imaginary_noise_emit_as_real_strings():
    # Noise must sit at the standard validation tolerance `10^-(dps - 5)`,
    # not at half precision: the writer is meant to absorb roundoff from
    # converged real-c arithmetic, not large half-precision residue. At
    # dps = 80 that is `~10^-75`, which the bootstrap-side pipeline already
    # meets (regularization converges to `10^-(dps - 10)`).
    rows = [[Polynomial([mp.mpc("1.25", "3e-78")], field=mp.mpc)]]
    block = SerializedDerivTable(
        shift=mp.mpf(0),
        base=mp.mpc("0.5", "-2e-78"),
        pole_list_x=(mp.mpc("-1.0", "1e-78"),),
        coordinates=("zzb",),
        numerators={"zzb": rows},
    )

    old_dps = mp.mp.dps
    mp.mp.dps = 80
    try:
        doc = build_pmp_doc(
            [block],
            F_pairs=[(0, 0)],
            objective=[mp.mpc("0.0", "-2e-78")],
            normalization=[mp.mpc("1.0", "2e-78")],
        )
    finally:
        mp.mp.dps = old_dps

    entry = doc["PositiveMatrixWithPrefactorArray"][0]
    assert entry["DampedRational"]["base"] == "0.5"
    assert entry["DampedRational"]["poles"] == ["-1.0"]
    assert entry["polynomials"][0][0][0] == ["1.25"]
    assert doc["objective"] == ["0.0"]
    assert doc["normalization"] == ["1.0"]


def test_block_coefficients_with_half_precision_imaginary_noise_rejected():
    """Half-precision imag residue is no longer absorbed: it signals a real
    bug upstream (e.g. a complex chiral input where a real one was expected),
    so the writer raises rather than silently emitting the real part.
    """
    rows = [[Polynomial([mp.mpc("1.25", "3e-38")], field=mp.mpc)]]
    block = SerializedDerivTable(
        shift=mp.mpf(0),
        base=mp.mpf("0.5"),
        pole_list_x=(mp.mpf("-1.0"),),
        coordinates=("zzb",),
        numerators={"zzb": rows},
    )

    old_dps = mp.mp.dps
    mp.mp.dps = 80
    try:
        with pytest.raises(ValueError, match="PMP entries must be real"):
            build_pmp_doc(
                [block],
                F_pairs=[(0, 0)],
                objective=[mp.mpf(0)],
            )
    finally:
        mp.mp.dps = old_dps


def test_multi_block_emits_one_entry_per_input():
    blocks = [_make_real_pole_block([[1]]) for _ in range(4)]
    doc = build_pmp_doc(
        blocks, F_pairs=[(0, 0)], objective=[mp.mpf(0)]
    )
    assert len(doc["PositiveMatrixWithPrefactorArray"]) == 4


# ---------------------------------------------------------------------------
# Validation: input integrity.
# ---------------------------------------------------------------------------


def test_complex_pole_in_block_raises():
    """drop_complex_pole_pairs must be called upstream; a complex pole here
    indicates a pipeline error."""
    poles = (mp.mpc(mp.mpf("-1"), mp.mpf("2")), mp.mpc(mp.mpf("-1"), mp.mpf("-2")))
    rows = [[_const_poly(1)]]
    block = SerializedDerivTable(
        shift=mp.mpf(0),
        base=mp.mpf("0.5"),
        pole_list_x=poles,
        coordinates=("zzb",),
        numerators={"zzb": rows},
    )
    with pytest.raises(ValueError, match="drop_complex_pole_pairs"):
        build_pmp_doc(
            [block], F_pairs=[(0, 0)], objective=[mp.mpf(0)]
        )


def test_positive_real_pole_in_block_raises():
    """A real pole inside x > 0 is a singularity on the positivity interval."""
    block = _make_real_pole_block([[1]], poles=(mp.mpf("0.25"),))
    with pytest.raises(ValueError, match="x > 0"):
        build_pmp_doc(
            [block], F_pairs=[(0, 0)], objective=[mp.mpf(0)]
        )


def test_objective_length_mismatch_raises():
    block = _make_real_pole_block([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="objective length"):
        build_pmp_doc(
            [block],
            F_pairs=[(0, 0), (1, 0)],
            objective=[mp.mpf(0)],  # length 1, need 2
        )


def test_normalization_length_mismatch_raises():
    block = _make_real_pole_block([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="normalization length"):
        build_pmp_doc(
            [block],
            F_pairs=[(0, 0), (1, 0)],
            objective=[mp.mpf(0), mp.mpf(0)],
            normalization=[mp.mpf(1)],  # length 1, need 2
        )


def test_unknown_coordinate_raises():
    block = _make_real_pole_block([[1]])
    with pytest.raises(KeyError, match="qqb"):
        build_pmp_doc(
            [block],
            F_pairs=[(0, 0)],
            coordinate="qqb",
            objective=[mp.mpf(0)],
        )


def test_F_pair_out_of_range_raises():
    block = _make_real_pole_block([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="exceeds numerator row count"):
        build_pmp_doc(
            [block],
            F_pairs=[(5, 0)],  # m=5 exceeds 2-row table
            objective=[mp.mpf(0)],
        )
    with pytest.raises(ValueError, match="exceeds numerator column count"):
        build_pmp_doc(
            [block],
            F_pairs=[(0, 5)],  # n=5 exceeds 2-col table
            objective=[mp.mpf(0)],
        )


def test_empty_positivity_blocks_raises():
    with pytest.raises(ValueError, match="positivity_blocks must be non-empty"):
        build_pmp_doc([], F_pairs=[(0, 0)], objective=[mp.mpf(0)])


def test_empty_F_pairs_raises():
    block = _make_real_pole_block([[1]])
    with pytest.raises(ValueError, match="F_pairs must be non-empty"):
        build_pmp_doc([block], F_pairs=[], objective=[])


# ---------------------------------------------------------------------------
# File output round-trip.
# ---------------------------------------------------------------------------


def test_write_pmp_json_round_trip(tmp_path, monkeypatch):
    block = _make_real_pole_block(
        [[1, 2], [3, 4]], base="0.5", poles=(-1, -2)
    )
    doc = build_pmp_doc(
        [block],
        F_pairs=[(0, 0), (1, 0)],
        objective=[mp.mpf(0), mp.mpf(-1)],
        normalization=[mp.mpf(1), mp.mpf(0)],
    )
    out_path = tmp_path / "pmp.json"
    write_pmp_json(doc, out_path)
    assert out_path.exists()

    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded == doc

    import os
    import stat

    original = out_path.read_bytes()
    for destination in ("", str(out_path) + os.sep,
                        str(tmp_path / "missing") + os.sep + ".." + os.sep + "pmp.json"):
        with pytest.raises(OSError):
            write_pmp_json(doc, destination)
        assert out_path.read_bytes() == original
    malformed = {**doc, "unsupported": object()}
    for destination in (out_path, tmp_path / "new.json"):
        with pytest.raises(TypeError):
            write_pmp_json(malformed, destination)
        assert out_path.read_bytes() == original
        assert not (tmp_path / "new.json").exists()
        assert not list(tmp_path.glob(".vb-pmp-*"))

    def interrupted_dump(doc, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr("virasoro_bootstrap.pmp_writer.json.dump", interrupted_dump)
        with pytest.raises(OSError, match="simulated write failure"):
            write_pmp_json(doc, out_path)
    assert out_path.read_bytes() == original
    assert not list(tmp_path.glob(".vb-pmp-*"))

    if os.name == "posix":
        broken = tmp_path / "broken.json"
        broken.symlink_to("missing/../pmp.json")
        with pytest.raises(FileNotFoundError):
            write_pmp_json(doc, broken)
        assert broken.is_symlink()
        assert out_path.read_bytes() == original
        out_path.chmod(0o640)
        alias = tmp_path / "alias.json"
        alias.symlink_to(out_path.name)
        updated = {**doc, "objective": ["1", "0"]}
        write_pmp_json(updated, os.fsencode(alias))
        assert alias.is_symlink()
        assert json.loads(alias.read_text()) == updated
        assert json.loads(out_path.read_text()) == updated
        assert stat.S_IMODE(out_path.stat().st_mode) == 0o640
        if os.geteuid() != 0:
            out_path.chmod(0o440)
            with pytest.raises(PermissionError):
                write_pmp_json(doc, alias)
            assert json.loads(out_path.read_text()) == updated
    write_pmp_json(doc, os.devnull)
    descriptor_path = tmp_path / "descriptor.json"
    descriptor = os.open(descriptor_path, os.O_WRONLY | os.O_CREAT, 0o666)
    write_pmp_json(doc, descriptor)
    assert json.loads(descriptor_path.read_text()) == doc


# ---------------------------------------------------------------------------
# End-to-end strip pipeline: build -> prefactor -> drop complex poles -> emit.
# ---------------------------------------------------------------------------


def test_build_prefactored_pmp_doc_matches_manual_strip_pipeline():
    """The full c=10 pipeline (build -> prefactor -> drop complex poles ->
    emit) agrees with build_prefactored_pmp_doc and emits a PMP with a
    real-pole prefactor and the right F-vector dimension."""
    mp.mp.dps = 50
    b = b_from_c(C_STRIP)
    lam = 3
    Delta_phi = mp.mpf("0.15")

    prefactored_blocks = []
    manual_blocks = []
    for ell in (0, 2):
        table = G_zzb_poly_table(ell, SCALAR_EXT, lam, lam, b=b, q_order=8)
        serialized = serialize_deriv_tables({"zzb": table})
        prefactored = apply_prefactor_to_table(
            serialized,
            v_prefactor_taylor(Delta_phi, lam, lam),
        )
        prefactored_blocks.append(prefactored)
        manual_blocks.append(drop_complex_pole_pairs(prefactored))

    F = F_minus_pairs(lam)
    K = len(F)
    assert K > 0
    objective = [mp.mpf(0)] * K
    normalization = [mp.mpf(1)] + [mp.mpf(0)] * (K - 1)

    manual_doc = build_pmp_doc(
        manual_blocks,
        F_pairs=F,
        objective=objective,
        normalization=normalization,
    )
    built = build_prefactored_pmp_doc(
        prefactored_blocks,
        F_pairs=F,
        objective=objective,
        normalization=normalization,
    )

    assert built.doc == manual_doc
    assert built.positivity_blocks == tuple(manual_blocks)
    assert all(lowering is not None for lowering in built.positive_pole_lowerings)
    assert all(lowering.selected_groups == () for lowering in built.positive_pole_lowerings)

    # One positivity block per spin, each polynomial vector with K entries.
    entries = built.doc["PositiveMatrixWithPrefactorArray"]
    assert len(entries) == len(manual_blocks)
    for entry in entries:
        assert len(entry["polynomials"][0][0]) == K
        # All emitted poles are real: `_num_str` emits real-only strings as
        # plain decimal (no "j" suffix); complex would be "(re+imj)".
        for pole_str in entry["DampedRational"]["poles"]:
            assert "j" not in pole_str, pole_str
            assert "+" not in pole_str.lstrip("-+"), pole_str


def test_build_prefactored_pmp_doc_lowers_ising_positive_poles(ising_prefactored_block):
    # "residue" selection is covered by test_bootstrap_crossing.py's
    # test_lower_positive_real_poles_interpolated_remainder_is_pmp_admissible.
    pole_selection = "rightmost"
    mp.mp.dps = 80
    prefactored = ising_prefactored_block
    F = F_minus_pairs(4)
    objective = [mp.mpf(0)] * len(F)
    normalization = [mp.mpf(1)] + [mp.mpf(0)] * (len(F) - 1)

    with pytest.raises(ValueError, match="x > 0"):
        build_pmp_doc(
            [prefactored],
            F_pairs=F,
            objective=objective,
            normalization=normalization,
        )

    built = build_prefactored_pmp_doc(
        [prefactored],
        F_pairs=F,
        objective=objective,
        normalization=normalization,
        positive_pole_npoles=6,
        positive_pole_selection=pole_selection,
    )

    assert len(built.positivity_blocks) == 1
    assert len(built.positive_pole_lowerings) == 1
    lowering = built.positive_pole_lowerings[0]
    assert lowering is not None
    assert len(lowering.selected_groups) > 0
    assert lowering.requested_npoles == 6
    assert lowering.kept_npoles is not None
    assert 0 < lowering.kept_npoles < lowering.requested_npoles
    assert lowering.pole_selection == pole_selection
    assert not _has_positive_real_pole(built.positivity_blocks[0])
    assert built.positivity_blocks[0] == lowering.cleared_interpolated_block
    assert len(built.doc["PositiveMatrixWithPrefactorArray"]) == 1


def test_lowered_block_emits_explicit_sample_scalings(ising_prefactored_block):
    """When positive real poles are lowered (cleared into the numerator), the
    explicit-sample-scaling path must still fire: the cleared block gets
    Bohr-Sommerfeld nodes matching its (raised) degree, with positive scalings.
    Without this pmp2sdp falls back to the raw high-degree coefficients and SDPB
    diverges (the Ising c=1/2 conditioning bug)."""
    mp.mp.dps = 80
    prefactored = ising_prefactored_block
    F = F_minus_pairs(4)
    objective = [mp.mpf(0)] * len(F)
    normalization = [mp.mpf(1)] + [mp.mpf(0)] * (len(F) - 1)

    built = build_prefactored_pmp_doc(
        [prefactored],
        F_pairs=F,
        objective=objective,
        normalization=normalization,
        positive_pole_npoles=6,
        explicit_sample_points=True,
    )
    pmp = built.doc["PositiveMatrixWithPrefactorArray"][0]
    assert "sampleScalings" in pmp and "samplePoints" in pmp

    # num sample points must equal max polynomial degree + 1 (pmp2sdp asserts
    # this), and all scalings must be positive (lowered poles excluded).
    max_deg = max(
        len(vec) - 1
        for mat in pmp["polynomials"] for row in mat for vec in row
    )
    assert len(pmp["samplePoints"]) == max_deg + 1
    assert len(pmp["sampleScalings"]) == max_deg + 1
    assert all(mp.mpf(s) > 0 for s in pmp["sampleScalings"])


def test_build_prefactored_pmp_doc_requires_lowering_for_positive_pole_npoles():
    block = _make_real_pole_block([[1]], poles=(-1,))
    with pytest.raises(ValueError, match="requires lower_positive_real=True"):
        build_prefactored_pmp_doc(
            [block],
            F_pairs=[(0, 0)],
            objective=[mp.mpf(0)],
            lower_positive_real=False,
            positive_pole_npoles=1,
        )
