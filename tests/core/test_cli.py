"""Tests for the virasoro_blocks CLI (__main__.py)."""

from __future__ import annotations

import json
import os
import tempfile

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import _parse_spin_token, _parse_spins, main
from virasoro_blocks.parity import normalize_external_parity, normalize_parity_phase


# ---------------------------------------------------------------------------
# _parse_spins helpers
# ---------------------------------------------------------------------------


def test_parse_spin_single_integer():
    assert _parse_spin_token("4") == [4]


def test_parse_spin_single_half_integer():
    assert _parse_spin_token("1/2") == [0.5]


def test_parse_spin_range_start_stop():
    assert _parse_spin_token("0:4") == [0, 1, 2, 3, 4]


def test_parse_spin_range_start_step_stop():
    assert _parse_spin_token("0:2:10") == [0, 2, 4, 6, 8, 10]


def test_parse_spin_half_integer_range():
    assert _parse_spin_token("1/2:1:5/2") == [0.5, 1.5, 2.5]


def test_parse_spin_range_step_larger_than_interval():
    assert _parse_spin_token("0:5:3") == [0]


def test_parse_spins_deduplicates_and_sorts():
    assert _parse_spins(["4", "0:2:4", "2"]) == [0, 2, 4]


def test_parse_spin_invalid_raises():
    with pytest.raises((ValueError, TypeError)):
        _parse_spin_token("a:b:c")


def test_parse_parity_phase_tokens():
    assert normalize_parity_phase("+1") == "1"
    assert normalize_parity_phase("-1") == "-1"
    assert normalize_parity_phase("I") == "i"
    assert normalize_parity_phase("-i") == "-i"


def test_parse_parity_phase_invalid_raises():
    with pytest.raises(ValueError):
        normalize_parity_phase("exp(i*pi/3)")


def test_normalize_external_parity_none():
    assert normalize_external_parity(None) is None


# ---------------------------------------------------------------------------
# Integration tests — use tiny parameters to stay fast
# ---------------------------------------------------------------------------

_TINY = [
    "--central-charge", "10",
    "--spin-ranges", "0",
    "--lambda", "2",
    "--delta-ext", "0.15", "0.15", "0.15", "0.15",
    "--order", "6",
    "--precision", "30",
]


def test_main_json_content_valid():
    """One tiny JSON build: the output file, its record fields, the jagged derivs
    shape, the default output name and the recorded args/precision."""
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + ["--format", "json", "--npoles", "4", "-o", d])
        files = os.listdir(d)
        assert any(f.endswith(".json") for f in files)
        # Default JSON template matches blocks_3d's plural `derivs_{}.json`.
        assert "derivs_0.json" in files
        assert "deriv_0.json" not in files
        fpath = next(os.path.join(d, f) for f in files if f.endswith(".json"))
        doc = json.loads(open(fpath).read())
        assert doc["spin"] == 0
        assert doc["order"] == 6
        assert "derivs" in doc
        assert doc["coordinates"] == ["zzb"]
        # Scalar G with even ell on zzb truncates to the blocks_3d-style jagged
        # shape: row m has length min(m, Lambda - m) + 1.
        derivs = doc["derivs"]
        assert len(derivs) == 1        # one coordinate
        # Lambda = 2 -> jagged row lengths [1, 2, 1].
        assert [len(row) for row in derivs[0]] == [1, 2, 1]
        # The `args` field starts with the program name, mirroring blocks_3d
        # (which records `build/blocks_3d -o ...`).
        assert doc["args"].startswith("virasoro_blocks ")
        # Both precision_input and precision_actual are recorded.
        assert "precision_input" in doc
        assert "precision_actual" in doc


def test_main_mma_content_valid():
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + ["--format", "m", "--npoles", "4", "-o", d])
        files = os.listdir(d)
        assert any(f.endswith(".m") for f in files)
        fpath = next(os.path.join(d, f) for f in files if f.endswith(".m"))
        text = open(fpath).read()
        assert "zzbDeriv[0,0]" in text
        assert "shiftedPoles" in text
        assert "samplePointsX" in text
        assert "precisionInput" in text
        assert "precisionActual" in text
        assert 'basis -> "scalar_G"' in text
        assert 'coordinate -> "zzb"' in text
        assert text.strip().startswith("{")
        assert text.strip().endswith("}")


def test_main_both_format_writes_both():
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + ["--format", "both", "--npoles", "4", "-o", d])
        files = os.listdir(d)
        assert any(f.endswith(".json") for f in files)
        assert any(f.endswith(".m") for f in files)


def test_main_full_method():
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + ["--format", "json", "--npoles", "all", "-o", d])
        fpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
        doc = json.loads(open(fpath).read())
        assert doc["spin"] == 0
        assert "derivs" in doc


def test_main_multiple_spins_writes_multiple_files():
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10",
            "--spin-ranges", "0", "2",
            "--lambda", "1",
            "--delta-ext", "0.1", "0.1", "0.1", "0.1",
            "--order", "6",
            "--precision", "30",
            "--format", "json",
            "--npoles", "4",
            "-o", d,
        ])
        json_files = [f for f in os.listdir(d) if f.endswith(".json")]
        assert len(json_files) == 2


def test_main_multiple_coordinates():
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + [
            "--format", "json", "--npoles", "4",
            "--coordinates", "zzb", "qqb",
            "-o", d,
        ])
        fpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
        doc = json.loads(open(fpath).read())
        assert set(doc["coordinates"]) == {"zzb", "qqb"}
        assert len(doc["derivs"]) == 2


def test_main_output_template():
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + [
            "--format", "json", "--npoles", "4",
            "--output-template", "block_spin{spin}.json",
            "-o", d,
        ])
        assert os.path.exists(os.path.join(d, "block_spin0.json"))


def test_main_mma_filename_contains_spin():
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + ["--format", "m", "--npoles", "4", "-o", d])
        mfiles = [f for f in os.listdir(d) if f.endswith(".m")]
        assert len(mfiles) == 1
        assert "L0" in mfiles[0]


def test_main_projected_spinning_mma_filename_and_metadata():
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "1/2",
            "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
            "--spin-ext", "1/2", "-1/2", "1/2", "-1/2",
            "--parity-project", "-",
            "--parity-ext", "1", "1", "1", "1",
            "--parity-exchanged", "-1",
            "--coordinates", "ws",
            "--format", "m", "--npoles", "4",
            "--order", "6", "--precision", "30", "-o", d,
        ])
        mfiles = [f for f in os.listdir(d) if f.endswith(".m")]
        assert len(mfiles) == 1
        assert "basisParityMinus" in mfiles[0]
        text = open(os.path.join(d, mfiles[0])).read()
        assert 'basis -> "parity_projected_B"' in text
        assert 'parityProjection -> "-"' in text
        assert 'parityExchanged -> "-1"' in text


def test_main_pole_selection_residue():
    """--pole-selection residue runs and produces valid JSON."""
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + ["--format", "json", "--npoles", "4", "--pole-selection", "residue", "-o", d])
        fpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
        doc = json.loads(open(fpath).read())
        assert "derivs" in doc


def test_main_defaults_to_contribution_optimal_for_virasoro():
    """The public Virasoro CLI follows the production selector."""
    with tempfile.TemporaryDirectory() as default_dir, tempfile.TemporaryDirectory() as explicit_dir:
        common = _TINY + ["--format", "json", "--npoles", "4"]
        main(common + ["-o", default_dir])
        main(common + ["--pole-selection", "contribution_optimal", "-o", explicit_dir])
        default_path = next(
            os.path.join(default_dir, name)
            for name in os.listdir(default_dir)
            if name.endswith(".json")
        )
        explicit_path = next(
            os.path.join(explicit_dir, name)
            for name in os.listdir(explicit_dir)
            if name.endswith(".json")
        )
        default_doc = json.loads(open(default_path).read())
        explicit_doc = json.loads(open(explicit_path).read())
        assert default_doc["pole_list_x"] == explicit_doc["pole_list_x"]
        assert default_doc["sample_points_x"] == explicit_doc["sample_points_x"]


def test_main_sample_via_numeric_records_metadata():
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + [
            "--format", "json", "--npoles", "4", "--sample-via", "numeric", "-o", d,
        ])
        fpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
        doc = json.loads(open(fpath).read())
        assert doc["sample_via"] == "numeric"


def test_main_sample_via_numeric_requires_interpolation():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(_TINY + [
                "--format", "json", "--npoles", "all",
                "--sample-via", "numeric", "-o", d,
            ])


def _numbers_from_json(obj, out):
    import mpmath as mp
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_from_json(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_from_json(v, out)
    elif isinstance(obj, str):
        try:
            out.append(mp.mpf(obj))
        except (ValueError, TypeError):
            pass
    elif isinstance(obj, (int, float)):
        out.append(mp.mpf(obj))


@pytest.mark.parametrize("basis_args", [
    pytest.param([], id="ordered_B"),
    pytest.param(["--parity-project", "-", "--parity-ext", "1", "1", "1", "1"],
                 id="parity_projected_B"),
])
def test_sample_via_numeric_matches_symbolic_spinning(basis_args):
    """--sample-via numeric on the spinning ordered_B / parity_projected_B bases
    reproduces the symbolic interpolant (the numeric block equals the symbolic
    block value, so the interpolant is the same)."""
    import mpmath as mp
    mp.mp.dps = 40

    def run(sample_via, d):
        main([
            "--central-charge", "10", "--spin-ranges", "1", "--lambda", "2",
            "--delta-ext", "0.5", "0.6", "0.5", "0.6",
            "--spin-ext", "1", "-1", "0", "0", *basis_args,
            "--coordinates", "zzb", "--format", "json", "--npoles", "4", "--order", "8", "--precision", "40",
            "--sample-via", sample_via, "-o", d,
        ])
        fpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
        return json.loads(open(fpath).read())

    with tempfile.TemporaryDirectory() as dn, tempfile.TemporaryDirectory() as ds:
        jn, js = run("numeric", dn), run("symbolic", ds)
    # The algorithms can retain different amounts of construction guard
    # precision while representing the same numerical table.
    assert jn.pop("precision_actual") >= 40
    assert js.pop("precision_actual") >= 40
    an, as_ = [], []
    _numbers_from_json(jn, an)
    _numbers_from_json(js, as_)
    assert len(an) == len(as_) and an
    # relative: the files carry the block values at the nodes too (magnitudes up to ~1e8 here)
    assert max(abs(a - b) / max(1, abs(a)) for a, b in zip(an, as_)) < mp.mpf(10) ** -25


# ---------------------------------------------------------------------------
# 2d external-weights parametrization: (--delta-ext, --spin-ext) vs
# (--h-ext, --hbar-ext) -- both are accepted, but they are mutually exclusive.
# ---------------------------------------------------------------------------


def _read_only_json(directory):
    fpath = next(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")
    )
    return json.loads(open(fpath).read())


def _read_json_by_spin(directory):
    docs = {}
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        doc = json.loads(open(path).read())
        docs[doc["spin"]] = doc
    return docs


def test_main_2d_accepts_h_hbar_ext_scalar():
    """Scalar externals via --h-ext / --hbar-ext (h_i = hbar_i)."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--h-ext", "0.075", "0.075", "0.075", "0.075",
            "--hbar-ext", "0.075", "0.075", "0.075", "0.075",
            "--lambda", "2", "--order", "6", "--precision", "30",
            "--format", "json", "--npoles", "4",
            "-o", d,
        ])
        doc = _read_only_json(d)
        assert doc["basis"] == "scalar_G"
        assert doc["spin_external"] == [0, 0, 0, 0]


def test_main_scalar_G_rejects_asymmetric_zbar_shift():
    """scalar_G is z<->zbar symmetric (single chiral expansion driven by
    --z-shift); a --zbar-shift that differs from --z-shift cannot be applied and
    must be rejected loudly rather than silently ignored."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10", "--spin-ranges", "0",
                "--delta-ext", "0.25", "0.25", "0.25", "0.25",
                "--lambda", "2", "--order", "6", "--precision", "30",
                "--format", "json", "--npoles", "4",
                "--z-shift", "-0.25", "--zbar-shift", "0",
                "-o", d,
            ])


def test_main_scalar_G_accepts_symmetric_u_shift():
    """--u-shift (z_shift == zbar_shift) is the symmetric path scalar_G accepts;
    an equal explicit --zbar-shift is likewise fine."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--delta-ext", "0.25", "0.25", "0.25", "0.25",
            "--lambda", "2", "--order", "6", "--precision", "30",
            "--format", "json", "--npoles", "4",
            "--u-shift", "-0.5",
            "-o", d,
        ])
        doc = _read_only_json(d)
        assert doc["basis"] == "scalar_G"


@pytest.mark.parametrize("extra", [
    ["--npoles", "all"],
    ["--pole-selection", "nearest"],
    ["--regularize", "off"],
    ["--spin-ranges", "2", "4"],
])
def test_main_large_spin_tower_rejects_fixed_spin_flags(extra):
    """The tower interpolates in the continuous spin L with its own pole
    selector and internal regularization, so the fixed-spin Delta-path flags are
    inert; passing them explicitly is rejected rather than silently ignored."""
    base = [
        "--large-spin-lmin", "10", "--family", "virasoro",
        "--central-charge", "5", "--order", "8", "--lambda", "2",
        "--npoles", "6", "--delta-ext", "0.1", "0.2", "0.15", "0.25",
        "--large-spin-twists", "0.7", "--coordinates", "zzb",
    ]
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(base + extra + ["-o", d])


_LS_TOWER_BASE = [
    "--large-spin-lmin", "10", "--family", "virasoro",
    "--central-charge", "5", "--order", "8", "--lambda", "2",
    "--npoles", "5", "--large-spin-twists", "0.7", "--coordinates", "zzb",
    "--precision", "40",
]


@pytest.mark.parametrize("extra,expect_basis", [
    (["--delta-ext", "0.2", "0.2", "0.2", "0.2"], "scalar_G"),
    (["--h-ext", "1.0", "0.6", "0.5", "1.1",
      "--hbar-ext", "0.0", "0.6", "0.5", "0.1"], "ordered_B"),
    (["--h-ext", "1.0", "0.6", "0.5", "1.1",
      "--hbar-ext", "0.0", "0.6", "0.5", "0.1", "--parity-project", "+"],
     "parity_projected_B"),
])
def test_main_large_spin_tower_basis_auto_resolves(extra, expect_basis):
    """The tower mirrors the fixed-spin --basis auto: scalar externals -> scalar_G,
    spinning -> ordered_B, spinning + --parity-project -> parity_projected_B."""
    with tempfile.TemporaryDirectory() as d:
        main(_LS_TOWER_BASE + extra + ["-o", d])
        assert _read_only_json(d)["basis"] == expect_basis


def test_main_large_spin_global_family_is_canonical():
    """The table family flag selects the exact global large-spin tower."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--family", "global", "--basis", "scalar_G",
            "--large-spin-lmin", "10", "--lambda", "2", "--npoles", "5",
            "--large-spin-double-twists", "1",
            "--delta-ext", "0.2", "0.2", "0.2", "0.2",
            "--coordinates", "zzb", "--precision", "40", "-o", d,
        ])
        document = _read_only_json(d)
        assert document["basis"] == "global_scalar_G"
        assert document["central_charge"] is None
        assert document["order"] == 30
        assert "--family global" in document["args"]
        assert "--large-spin-mode" not in document["args"]


@pytest.mark.parametrize("family", ["virasoro", "global"])
@pytest.mark.parametrize("parity", ["1", "-1"])
@pytest.mark.parametrize("projected", [False, True], ids=["scalar", "projected"])
def test_main_large_spin_radial_tables(family, parity, projected, tmp_path):
    """Radial and full tower tables share a file and reproduce fixed-spin blocks.

    For scalar_G ``parity`` picks an even or odd checked spin of the one (z <-> zbar even) tower; for the
    projection it is the projection sign."""
    from virasoro_blocks import derivs
    from virasoro_blocks.serialization import serialized_table_from_json

    parents = ["xt", "ws", "qqb_ws"]
    radials = [coord + "_radial" for coord in parents]
    options = ["--central-charge", "32"] if family == "virasoro" else []
    if projected:
        options += ["--h-ext", *["0.0625"] * 4, "--hbar-ext", *["1.0625"] * 4,
                    "--parity-project", "+" if parity == "1" else "-",
                    "--parity-ext", "1", "1", "1", "1"]
    else:
        options += ["--delta-ext", *["0.125"] * 4] + (["--parity-exchanged", "1"] if parity == "1" else [])
    main([
        "--family", family, *options, "--order", "20", "--lambda", "2",
        "--large-spin-lmin", "14", "--large-spin-twists", "0.25",
        "--npoles", "8", "--precision", "60",
        "--chiral-norm", "4", "--u-shift", "-0.125",
        "--coordinates", *radials, *parents, "--format", "json", "-o", str(tmp_path),
    ])
    doc = _read_only_json(tmp_path)
    assert doc["variable"] == "L"
    assert doc["coordinates"] == radials + parents
    assert float(doc["verified_digits"]) > 6
    for grid in doc["derivs"][:len(radials)]:
        assert len(grid) == 3
        assert all(len(row) == 1 for row in grid)
    with mp.workdps(80):
        table = serialized_table_from_json(doc)
        spin = 16 if parity == "1" else 17
        for parent, radial in zip(parents, radials):
            name = ("global_" if family == "global" else "")
            name += ("parity_projected_B_" if projected else "G_") + parent + "_table"
            builder = getattr(derivs, name)
            kwargs = dict(c=32, q_order=20) if family == "virasoro" else {}
            externals = [[mp.mpf("0.0625")] * 4]
            if projected:
                externals.append([mp.mpf("1.0625")] * 4)
                kwargs.update(parity="+" if parity == "1" else "-",
                              parity_external=["1"] * 4, zbar_shift=mp.mpf("-0.125"))
            ref = builder(
                mp.mpf(spin) + mp.mpf("0.25"), spin, *externals,
                2, 0, chiral_norm=4, z_shift=mp.mpf("-0.125"), **kwargs,
            )
            got = table.evaluate(radial, mp.mpf(spin))
            folded = table.evaluate(parent, mp.mpf(spin))
            for m in range(3):
                assert abs(got[m][0] - ref[m][0]) < mp.mpf("1e-8") * abs(ref[m][0])
                if m <= (1 if projected and parity == "-1" else 2):
                    assert mp.almosteq(got[m][0], folded[m][0], rel_eps=mp.mpf("1e-45"))


def test_main_large_spin_tower_fermionic_parity_projected():
    """Fermionic externals (half-integer external spins) + Z4 parity phases build
    a parity_projected_B tower."""
    with tempfile.TemporaryDirectory() as d:
        main(_LS_TOWER_BASE + [
            "--h-ext", "0.5", "0.5", "0.7", "0.7",
            "--hbar-ext", "0.0", "0.0", "0.2", "0.2",
            "--parity-project", "+", "--parity-ext", "i", "-i", "i", "-i",
            "-o", d,
        ])
        assert _read_only_json(d)["basis"] == "parity_projected_B"


def test_main_large_spin_tower_scalar_G_rejects_spinning():
    """scalar_G is the scalar-external tower; spinning externals must route to
    ordered_B / parity_projected_B, not scalar_G."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(_LS_TOWER_BASE + [
                "--basis", "scalar_G",
                "--h-ext", "1.0", "0.6", "0.5", "1.1",
                "--hbar-ext", "0.0", "0.6", "0.5", "0.1", "-o", d,
            ])


def test_main_2d_accepts_h_hbar_ext_spinning():
    """Spinning externals via --h-ext / --hbar-ext: s_i = h_i - hbar_i = 1 for
    op 1 selects the ordered-B path."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--h-ext", "1.0", "0.075", "0.075", "0.075",
            "--hbar-ext", "0.0", "0.075", "0.075", "0.075",
            "--lambda", "2", "--order", "6", "--precision", "30",
            "--coordinates", "zzb",
            "--format", "json", "--npoles", "4",
            "-o", d,
        ])
        doc = _read_only_json(d)
        assert doc["basis"] == "ordered_B"
        assert doc["spin_external"] == [1, 0, 0, 0]


def test_main_2d_parametrizations_agree_on_output():
    """The (Delta, spin) parametrization and the (h, hbar) parametrization
    should produce byte-identical JSON for the same physical setup."""
    delta_ext = ["0.5", "0.5", "0.5", "0.5"]
    spin_ext = ["0", "0", "0", "0"]
    # h_i = (Delta + s)/2, hbar_i = (Delta - s)/2 -> all 0.25 for scalars.
    h_ext = ["0.25"] * 4
    hbar_ext = ["0.25"] * 4
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--delta-ext", *delta_ext, "--spin-ext", *spin_ext,
            "--lambda", "2", "--order", "6", "--precision", "30",
            "--format", "json", "--npoles", "4",
            "-o", d1,
        ])
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--h-ext", *h_ext, "--hbar-ext", *hbar_ext,
            "--lambda", "2", "--order", "6", "--precision", "30",
            "--format", "json", "--npoles", "4",
            "-o", d2,
        ])
        doc_ds = _read_only_json(d1)
        doc_hh = _read_only_json(d2)
        # Compare everything except the args field (which records the CLI
        # invocation verbatim, so it will of course differ).
        for k in ("derivs", "h_external", "hbar_external", "delta_external",
                  "spin_external", "basis", "label", "exp_base", "pole_list_x"):
            assert doc_ds[k] == doc_hh[k], f"mismatch on {k}"


def test_main_rejects_h_ext_combined_with_delta_ext():
    """The two parametrizations are mutually exclusive."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10", "--spin-ranges", "0",
                "--delta-ext", "0.5", "0.5", "0.5", "0.5",
                "--h-ext", "0.25", "0.25", "0.25", "0.25",
                "--hbar-ext", "0.25", "0.25", "0.25", "0.25",
                "--lambda", "2", "--order", "6", "--precision", "30",
                "--format", "json", "-o", d,
            ])


def test_main_rejects_hbar_ext_without_h_ext():
    """--hbar-ext alone is meaningless without --h-ext."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10", "--spin-ranges", "0",
                "--hbar-ext", "0.25", "0.25", "0.25", "0.25",
                "--lambda", "2", "--order", "6", "--precision", "30",
                "--format", "json", "-o", d,
            ])


def test_main_rejects_non_half_integer_h_minus_hbar():
    """The exchanged-spin column of the JSON requires integer or half-integer
    spins; reject a (h, hbar) pair whose difference is neither."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10", "--spin-ranges", "0",
                "--h-ext", "0.3", "0.25", "0.25", "0.25",
                "--hbar-ext", "0.0", "0.25", "0.25", "0.25",
                "--lambda", "2", "--order", "6", "--precision", "30",
                "--format", "json", "-o", d,
            ])


def test_main_default_externals_remain_all_zero():
    """When neither parametrization is supplied, the 2d path still defaults
    to all-zero delta_ext / spin_ext (backwards-compatible behavior)."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--lambda", "2", "--order", "6", "--precision", "30",
            "--format", "json", "--npoles", "4",
            "-o", d,
        ])
        doc = _read_only_json(d)
        assert doc["delta_external"] == ["0.0", "0.0", "0.0", "0.0"]
        assert doc["spin_external"] == [0, 0, 0, 0]


def test_main_num_threads_parallel_matches_serial():
    base = [
        "--central-charge", "10",
        "--spin-ranges", "0", "2",
        "--lambda", "2",
        "--delta-ext", "0.15", "0.15", "0.15", "0.15",
        "--order", "6",
        "--precision", "30",
        "--format", "json",
        "--npoles", "4",
    ]
    with tempfile.TemporaryDirectory() as serial_dir, tempfile.TemporaryDirectory() as parallel_dir:
        main(base + ["--num-threads", "1", "-o", serial_dir])
        main(base + ["--num-threads", "2", "-o", parallel_dir])

        serial_docs = _read_json_by_spin(serial_dir)
        parallel_docs = _read_json_by_spin(parallel_dir)
        assert serial_docs.keys() == parallel_docs.keys() == {0, 2}
        for spin in serial_docs:
            serial_doc = dict(serial_docs[spin])
            parallel_doc = dict(parallel_docs[spin])
            serial_doc.pop("args")
            parallel_doc.pop("args")
            assert parallel_doc == serial_doc


def test_main_rejects_nonpositive_num_threads():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(_TINY + [
                "--format", "json",
                "--npoles", "4",
                "--num-threads", "0",
                "-o", d,
            ])


# ---------------------------------------------------------------------------
# --regularize {auto, on, off} -- the tri-state dispatch.
# ---------------------------------------------------------------------------


_REGTEST_BASE = [
    "--spin-ranges", "0",
    "--lambda", "2",
    "--delta-ext", "0.1", "0.1", "0.1", "0.1",
    "--order", "6", "--precision", "30",
    "--format", "json", "--npoles", "all",
]


def test_regularize_auto_enables_at_ising(capsys):
    """At c = 1/2 the default 'auto' mode should detect the Ising minimal
    model and enable regularization, with a printed notice."""
    with tempfile.TemporaryDirectory() as d:
        main(["--central-charge", "1/2", *_REGTEST_BASE, "-o", d])
        out = capsys.readouterr().out
        assert "singular locus" in out
        assert "Ising" in out
        # On the full path the exact polynomial coefficients can be large, but
        # the regularized CLI output should still be finite rather than the
        # NaN/Inf or obviously corrupted data we'd see from a failed path.
        doc = _read_only_json(d)
        for coord_block in doc["derivs"]:
            for m_row in coord_block:
                for entry in m_row:
                    for ck in entry:
                        v = mp.mpc(ck) if "j" in ck else mp.mpf(ck)
                        assert mp.isfinite(abs(v)), f"non-finite coefficient: {ck}"


def test_regularize_auto_does_not_trigger_at_generic_c(capsys):
    """At a non-singular c, 'auto' is a no-op: no notice, regularize off."""
    with tempfile.TemporaryDirectory() as d:
        main(["--central-charge", "10", *_REGTEST_BASE, "-o", d])
        out = capsys.readouterr().out
        assert "singular locus" not in out
        assert "Ising" not in out


def test_regularize_on_at_generic_c_runs(capsys):
    """--regularize on at non-singular c is valid (the extrapolation path is
    a superset of the direct one) and produces a sensible block."""
    with tempfile.TemporaryDirectory() as d:
        main(["--central-charge", "10", "--regularize", "on",
              *_REGTEST_BASE, "-o", d])
        doc = _read_only_json(d)
        assert "derivs" in doc


def test_regularize_bare_flag_is_on_legacy(capsys):
    """Bare --regularize (no value) is the legacy spelling of --regularize on."""
    with tempfile.TemporaryDirectory() as d:
        main(["--central-charge", "10", "--regularize",
              *_REGTEST_BASE, "-o", d])
        doc = _read_only_json(d)
        assert "derivs" in doc


@pytest.mark.parametrize("c", ["1/2", "1"])
def test_regularize_off_at_singular_c_is_refused(capsys, c):
    """Reject --regularize off at a recognized singular charge before building."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(["--central-charge", c, "--regularize", "off", *_REGTEST_BASE, "-o", d])
        err = capsys.readouterr().err
        assert "singular" in err.lower() and "--regularize auto" in err


def test_regularize_invalid_choice_errors():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(["--central-charge", "10", "--regularize", "yes",
                  *_REGTEST_BASE, "-o", d])


def test_regularize_auto_hints_at_near_singular_c(capsys):
    """At c ~= 1/2 + 1e-13, the user is floating-point-close to Ising but not
    exactly on it. The auto path should print a hint to stderr suggesting the
    user land on the locus exactly."""
    with tempfile.TemporaryDirectory() as d:
        main(["--central-charge", "0.5000000000001", *_REGTEST_BASE, "-o", d])
        captured = capsys.readouterr()
        assert "hint:" in captured.err
        assert "Ising" in captured.err
        assert "exact value" in captured.err
        # The auto path stayed in direct mode (the hint did not flip to
        # regularize-on); no auto-regularize notice on stdout.
        assert "enabling regularization" not in captured.out


def test_regularize_auto_no_hint_at_exact_locus(capsys):
    """At exact singular c, the user gets the auto-regularize notice on
    stdout, not the near-locus hint on stderr."""
    with tempfile.TemporaryDirectory() as d:
        main(["--central-charge", "1/2", *_REGTEST_BASE, "-o", d])
        captured = capsys.readouterr()
        assert "hint:" not in captured.err
        assert "enabling regularization" in captured.out


def test_regularize_auto_no_hint_at_far_c(capsys):
    """At c well away from any singular locus, neither hint nor notice fires."""
    with tempfile.TemporaryDirectory() as d:
        main(["--central-charge", "10", *_REGTEST_BASE, "-o", d])
        captured = capsys.readouterr()
        assert "hint:" not in captured.err
        assert "enabling regularization" not in captured.out


def test_regularize_on_suppresses_near_locus_hint(capsys):
    """--regularize on means the user opted in explicitly; the hint is
    redundant noise in that case."""
    with tempfile.TemporaryDirectory() as d:
        main(["--central-charge", "0.5000000000001", "--regularize", "on",
              *_REGTEST_BASE, "-o", d])
        captured = capsys.readouterr()
        assert "hint:" not in captured.err


@pytest.mark.parametrize(
    ("c_value", "extra_args"),
    [
        ("0.65", []),
        ("1/2", ["--regularize", "on"]),
    ],
)
def test_interpolation_succeeds_below_one(c_value, extra_args, capsys):
    """c < 1 interpolation is supported (force_keep_real_positive path)."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", c_value,
            *extra_args,
            "--spin-ranges", "0",
            "--lambda", "2",
            "--delta-ext", "0.1", "0.1", "0.1", "0.1",
            "--order", "6", "--precision", "30",
            "--format", "json", "--npoles", "4", "-o", d,
        ])
        # Should produce output without error
        out_files = os.listdir(d)
        assert any(f.endswith(".json") for f in out_files)


def test_chiral_F_at_singular_c_regularizes(capsys):
    """--basis chiral_F at a singular c is regularized like the 2d bases: the
    chiral F(h, z) table is a DeltaDerivTable, so regularize_delta_table /
    regularize_delta_skeleton serve it (the strips map the "chiral" provenance
    tag to h_externals). Auto-regularize kicks in and the build succeeds."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "1/2",
            "--h-ext", "0.05", "0.05", "0.05", "0.05",
            "--basis", "chiral_F", "--coordinates", "z",
            "--lambda", "2", "--order", "6", "--precision", "30",
            "--format", "json", "-o", d,
        ])
        # Auto-regularize announced it triggered at the singular locus.
        out = capsys.readouterr().out
        assert "singular locus" in out
        doc = _read_only_json(d)
        assert doc["basis"] == "chiral_F"


def test_cli_chiral_F_singular_c_uses_skeleton(monkeypatch):
    """At a singular c >= 1, chiral_F on the numeric + contribution interpolation
    path builds the numerator-free skeleton (no symbolic numerators)."""
    import virasoro_blocks.table_backends as cli

    calls = []
    original = cli._CHIRAL_POLY_TABLE_BUILDERS["z"]

    def wrapped(*args, **kwargs):
        calls.append(bool(kwargs.get("skeleton", False)))
        return original(*args, **kwargs)

    monkeypatch.setitem(cli._CHIRAL_POLY_TABLE_BUILDERS, "z", wrapped)

    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "25",
            "--basis", "chiral_F", "--coordinates", "z",
            "--h-ext", "0.1", "0.1", "0.1", "0.1",
            "--lambda", "2", "--order", "8", "--npoles", "4", "--precision", "40",
            "--sample-via", "numeric",
            "--pole-selection", "contribution", "--regularize", "on",
            "--format", "json", "-o", d,
        ])

    assert calls
    assert all(calls)


def test_main_spinning_externals_json():
    """--spin-ext selects the ordered-block path; JSON records label h_hbar."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0", "2",
            "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
            "--spin-ext", "0", "2", "0", "2",
            "--order", "6", "--precision", "30",
            "--coordinates", "zzb", "--format", "json",
            "--npoles", "4", "-o", d,
        ])
        doc = json.loads(open(os.path.join(d, "derivs_0.json")).read())
        assert doc["basis"] == "ordered_B"
        assert doc["parity_projection"] is None
        assert doc["label"] == "h_hbar"
        assert doc["spin_external"] == [0, 2, 0, 2]
        assert doc["parity_basis"] is None
        assert doc["parity_external"] is None
        assert doc["parity_exchanged"] is None
        assert "derivs" in doc


def test_main_half_integer_spinning_externals_json():
    """Half-integer external and exchanged spins are accepted on the ordered-B path."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "1/2", "3/2",
            "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
            "--spin-ext", "1/2", "-1/2", "1/2", "-1/2",
            "--order", "6", "--precision", "30",
            "--coordinates", "zzb", "--format", "json",
            "--npoles", "4", "-o", d,
        ])
        doc = json.loads(open(os.path.join(d, "derivs_0.5.json")).read())
        assert doc["basis"] == "ordered_B"
        assert doc["label"] == "h_hbar"
        assert doc["spin"] == 0.5
        assert doc["spin_external"] == [0.5, -0.5, 0.5, -0.5]
        assert "derivs" in doc


def test_main_parity_metadata_json():
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "1/2",
            "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
            "--spin-ext", "1/2", "-1/2", "1/2", "-1/2",
            "--parity-ext", "1", "-1", "i", "-i",
            "--order", "6", "--precision", "30",
            "--coordinates", "zzb", "--format", "json",
            "--npoles", "4", "-o", d,
        ])
        doc = json.loads(open(os.path.join(d, "derivs_0.5.json")).read())
        assert doc["basis"] == "ordered_B"
        assert doc["parity_basis"] == "diagonal"
        assert doc["parity_external"] == ["1", "-1", "i", "-i"]


def test_main_spinning_rejects_parity_folded_coordinate():
    """A parity-folded coordinate with spinning externals is a hard error."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10", "--spin-ranges", "0",
                "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
                "--spin-ext", "2", "0", "0", "0",
                "--coordinates", "ws", "-o", d,
            ])


def test_main_projected_spinning_allows_folded_coordinate():
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "1/2",
            "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
            "--spin-ext", "1/2", "-1/2", "1/2", "-1/2",
            "--parity-project", "+",
            "--parity-ext", "1", "1", "1", "1",
            "--coordinates", "ws",
            "--format", "json", "--npoles", "4",
            "--order", "6", "--precision", "30", "-o", d,
        ])
        doc = json.loads(open(os.path.join(d, "derivs_0.5.json")).read())
        assert doc["basis"] == "parity_projected_B"
        assert doc["parity_projection"] == "+"
        assert doc["parity_basis"] == "diagonal"
        assert doc["parity_external"] == ["1", "1", "1", "1"]
        assert doc["label"] == "parity_pm"
        assert doc["coordinates"] == ["ws"]


def test_main_projected_spinning_rejects_missing_parity_metadata():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10", "--spin-ranges", "1/2",
                "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
                "--spin-ext", "1/2", "-1/2", "1/2", "-1/2",
                "--parity-project", "+",
                "--coordinates", "ws",
                "--format", "json", "--npoles", "4",
                "--order", "6", "--precision", "30", "-o", d,
            ])


def test_main_projected_rejects_scalar_externals():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(_TINY + [
                "--parity-project", "+",
                "--parity-ext", "1", "1", "1", "1",
                "-o", d,
            ])


def test_main_parity_exchanged_requires_projection():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(_TINY + [
                "--parity-exchanged", "-1",
                "-o", d,
            ])


def test_main_parity_exchanged_even_default_token_requires_projection():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(_TINY + [
                "--parity-exchanged", "1",
                "-o", d,
            ])


def test_main_projected_folded_rejects_non_real_parity_phase():
    """Parity-folded coordinates with a non-real total parity phase are a hard
    CLI error -- the projected block has no definite z<->zbar parity."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10", "--spin-ranges", "0",
                "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
                "--spin-ext", "1", "-1", "1", "-1",
                "--parity-project", "+", "--parity-ext", "i", "1", "1", "1",
                "--coordinates", "ws", "-o", d,
            ])


def test_main_projected_folded_accepts_real_parity_phase():
    """The same setup with a real total phase (paired i's) is accepted."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "2",
            "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
            "--spin-ext", "1", "-1", "1", "-1",
            "--parity-project", "+", "--parity-ext", "i", "i", "1", "1",
            "--coordinates", "ws",
            "--order", "6", "--npoles", "4", "--precision", "30", "-o", d,
        ])
        assert any(f.endswith(".json") for f in os.listdir(d))


def test_main_projected_folded_zero_table_interpolates():
    """A projected spinning table that vanishes identically should still
    serialize on the interpolation path instead of failing with a negative
    default sample-count."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
            "--spin-ext", "1", "-1", "1", "-1",
            "--parity-project", "+", "--parity-ext", "i", "i", "1", "1",
            "--coordinates", "ws",
            "--format", "json", "--order", "6", "--npoles", "4", "--precision", "30", "-o", d,
        ])
        assert any(f.endswith(".json") for f in os.listdir(d))


def test_main_radial_coordinates():
    """Radial coordinates produce single-column (m x 1) derivative tables."""
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + [
            "--coordinates", "ws_radial", "xt_radial",
            "--format", "json", "--npoles", "4",
            "-o", d,
        ])
        fpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
        doc = json.loads(open(fpath).read())
        assert set(doc["coordinates"]) == {"ws_radial", "xt_radial"}
        # radial: every (m) row has a single n-column
        for coord_derivs in doc["derivs"]:
            assert all(len(n_row) == 1 for n_row in coord_derivs)


def test_main_radial_rejected_for_raw_ordered_B():
    """Radial coordinates are parity-folded, so the raw ordered-B (spinning,
    no projection) path rejects them."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10", "--spin-ranges", "0",
                "--lambda", "2", "--delta-ext", "0.5", "0.5", "0.5", "0.5",
                "--spin-ext", "2", "0", "0", "0",
                "--coordinates", "ws_radial", "-o", d,
            ])


def test_main_scalar_externals_label():
    """With no --spin-ext the JSON label is delta_l."""
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + ["--format", "json", "--npoles", "4", "-o", d])
        fpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
        doc = json.loads(open(fpath).read())
        assert doc["basis"] == "scalar_G"
        assert doc["parity_projection"] is None
        assert doc["label"] == "delta_l"
        assert doc["parity_basis"] is None
        assert doc["parity_external"] is None
        assert doc["parity_exchanged"] is None


def test_main_npoles_default_uses_all_poles():
    """Omitting --npoles with interpolation uses all denominator poles."""
    with tempfile.TemporaryDirectory() as d:
        main(_TINY + ["--format", "json", "-o", d])
        fpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
        doc = json.loads(open(fpath).read())
        # npoles == len(pole_list_x) when no --npoles given
        assert doc["npoles"] == len(doc["pole_list_x"])


def test_main_negative_spin_range_accepted():
    """`--spin-ranges=-1:1` should emit files for spins -1, 0, 1.

    G_{Delta,-ell} = (-1)^ell * G_{Delta,ell} -- a redundant but consistent
    labeling -- so accepting negative ell on the scalar G path is a
    user-friendliness choice, not a correctness change. The CLI uses the
    `=`-form so argparse does not confuse the negative number with a flag.
    """
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10",
            "--spin-ranges=-1:1",
            "--lambda", "2",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--order", "6",
            "--precision", "30",
            "--format", "json",
            "--npoles", "4",
            "-o", d,
        ])
        spins = sorted(
            int(json.loads(open(os.path.join(d, f)).read())["spin"])
            for f in os.listdir(d) if f.endswith(".json")
        )
        assert spins == [-1, 0, 1]


# ---------------------------------------------------------------------------
# --npoles overflow + precision-vs-degree notices
# ---------------------------------------------------------------------------


def test_npoles_overflow_notice(capsys):
    """When --npoles exceeds the actual Delta-pole count at the chosen
    --order, the CLI prints a notice naming both numbers."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--lambda", "2", "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--order", "6", "--npoles", "500", "--precision", "60", "--target-digits", "60",
            "--format", "json", "-o", d,
        ])
        out = capsys.readouterr().out
        # The order=6 pole count is 18 = 2 * 9 surviving chiral Kac pairs:
        # 14 raw pairs minus the 5 both-odd ones killed by identical-scalar
        # fusion (structural source-side strip).
        assert "keeps every" in out
        assert "500" in out
        assert "18" in out
        assert "--order" in out
        # every pole kept: the full table (--target-digits escalates it as a keep-all table), not an order reduction
        doc = json.load(open(os.path.join(d, "derivs_0.json")))
        assert int(doc["npoles"]) == 18 and float(doc["verified_digits"]) >= 60 and doc["target_digits"] == 60


def test_npoles_within_range_no_overflow_notice(capsys):
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--lambda", "2", "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--order", "6", "--npoles", "10", "--precision", "60",
            "--format", "json", "-o", d,
        ])
        out = capsys.readouterr().out
        assert "exceeds" not in out


def test_precision_below_polynomial_degree_warning(capsys):
    """The degree-D monomial form cancels on evaluation: at --precision P a keep-all table is built at P (the working
    precision) and the shortfall is reported with the precision that carries it; --target-digits builds at a working
    precision above P, verified to deliver the digits."""
    import glob as _glob
    import json as _json

    args = ["--central-charge", "10", "--spin-ranges", "0", "--lambda", "2", "--delta-ext", "0.15", "0.15", "0.15",
            "0.15", "--order", "10", "--precision", "30", "--format", "json"]
    with tempfile.TemporaryDirectory() as d:
        main([*args, "-o", d])
        err = capsys.readouterr().err
        assert "monomial form" in err and "Raise --precision to about" in err and "--target-digits 30" in err
        doc = _json.load(open(_glob.glob(os.path.join(d, "*.json"))[0]))
        assert doc["precision_input"] == 30 and 30 <= doc["precision_actual"] <= 40
        assert float(doc["verified_digits"]) < 30 and "target_digits" not in doc
    with tempfile.TemporaryDirectory() as d:
        main([*args, "--target-digits", "30", "-o", d])
        assert "built at working precision" in capsys.readouterr().err
        doc = _json.load(open(_glob.glob(os.path.join(d, "*.json"))[0]))
        assert doc["precision_input"] == 30 and doc["precision_actual"] > 30
        assert float(doc["verified_digits"]) >= 30 and doc["target_digits"] == 30


def test_precision_well_above_polynomial_degree_no_warning(capsys):
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10", "--spin-ranges", "0",
            "--lambda", "2", "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--order", "6", "--precision", "60",
            "--format", "json", "-o", d,
        ])
        err = capsys.readouterr().err
        assert "polynomial degree" not in err


# ---------------------------------------------------------------------------
# degree-only (skeleton) eligibility for the numeric-sampler path
# ---------------------------------------------------------------------------


def _skeleton_args(**overrides):
    from types import SimpleNamespace

    base = dict(
        method="interpolation",
        sample_via="numeric",
        coordinates=["zzb"],
        pole_selection="contribution",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_numeric_skeleton_eligible_cases():
    pytest.importorskip("flint")
    from virasoro_blocks.table_backends import _numeric_skeleton_eligible

    elig = _numeric_skeleton_eligible

    # Non-folded scalar_G + chiral_F numeric interpolation: eligible.
    assert elig(_skeleton_args(), "scalar_G", regularize=False)
    assert elig(_skeleton_args(coordinates=["yyb", "qqb"]), "scalar_G", regularize=False)
    assert elig(
        _skeleton_args(coordinates=["z"], pole_selection="nearest"),
        "chiral_F", regularize=False,
    )
    assert elig(
        _skeleton_args(coordinates=["z"], pole_selection="contribution"),
        "chiral_F", regularize=False,
    )

    # Parity-folded 2d coordinates ARE eligible: the fold is a Delta-independent
    # recombination over distinct (m+n) blocks, so each folded entry's degree is
    # the max of the structural source degrees (no cancellation across the fold).
    assert elig(_skeleton_args(coordinates=["ws"]), "scalar_G", regularize=False)
    assert elig(
        _skeleton_args(coordinates=["zzb", "xt"]), "scalar_G", regularize=False
    )
    # Radial slices (the n = 0 column of a folded parent) are eligible too: the
    # folded skeleton at n_max = 0 gives the (m_max + 1) x 1 radial table, and the
    # numeric sampler maps the radial coordinate to the folded parent.
    assert elig(
        _skeleton_args(coordinates=["xt_radial"]), "scalar_G", regularize=False
    )

    # chiral_F with residue-based selection: needs symbolic numerators (no spin).
    assert not elig(
        _skeleton_args(coordinates=["z"], pole_selection="residue"),
        "chiral_F", regularize=False,
    )

    # ordered_B has a structural degree (deg = cd_h[m] + cd_hbar[n]); eligible
    # even with residue-based selection (the two-sided numeric scorer handles
    # ell != 0, and the per-spin safety net rebuilds full where it declines).
    assert elig(_skeleton_args(), "ordered_B", regularize=False)
    assert elig(
        _skeleton_args(pole_selection="nearest"), "ordered_B", regularize=False
    )
    # parity_projected_B IS eligible (diagonal and off-diagonal): its swapped
    # term's leading coefficients are proportional across orders, so the degree
    # is structural -- the diagonal cancels per the build-time swap sign (like
    # scalar_G's even/odd ell), the off-diagonal never cancels.
    assert elig(_skeleton_args(), "parity_projected_B", regularize=False)
    assert elig(
        _skeleton_args(z_center=mp.mpf("0.4"), zbar_center=mp.mpf("0.55")),
        "parity_projected_B", regularize=False,
    )

    # Disqualifiers: regularize, symbolic sampling, non-interpolation.
    assert not elig(_skeleton_args(), "scalar_G", regularize=True)
    assert not elig(_skeleton_args(sample_via="symbolic"), "scalar_G", regularize=False)
    assert not elig(_skeleton_args(method="full"), "scalar_G", regularize=False)


def test_cli_off_diagonal_numeric_scalar_uses_skeleton(monkeypatch):
    import virasoro_blocks.table_backends as cli

    calls = []
    original = cli._POLY_TABLE_BUILDERS["zzb"]

    def wrapped(*args, **kwargs):
        calls.append(bool(kwargs.get("skeleton", False)))
        return original(*args, **kwargs)

    monkeypatch.setitem(cli._POLY_TABLE_BUILDERS, "zzb", wrapped)

    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "10",
            "--spin-ranges", "0",
            "--lambda", "2",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--coordinates", "zzb",
            "--z-center", "0.37",
            "--zbar-center", "0.55",
            "--order", "8",
            "--npoles", "4",
            "--precision", "40",
            "--sample-via", "numeric",
            "--pole-selection", "contribution",
            "--format", "json",
            "-o", d,
        ])

    assert calls
    assert all(calls)


def test_cli_off_diagonal_regularized_scalar_uses_skeleton(monkeypatch):
    import virasoro_blocks.table_backends as cli

    calls = []
    original = cli._POLY_TABLE_BUILDERS["zzb"]

    def wrapped(*args, **kwargs):
        calls.append(bool(kwargs.get("skeleton", False)))
        return original(*args, **kwargs)

    monkeypatch.setitem(cli._POLY_TABLE_BUILDERS, "zzb", wrapped)

    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "1",
            "--spin-ranges", "0",
            "--lambda", "2",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--coordinates", "zzb",
            "--z-center", "0.37",
            "--zbar-center", "0.55",
            "--order", "8",
            "--npoles", "4",
            "--precision", "40",
            "--sample-via", "numeric",
            "--pole-selection", "contribution",
            "--regularize", "on",
            "--format", "json",
            "-o", d,
        ])

    assert calls
    assert all(calls)


def test_cli_regularized_scalar_below_one_uses_skeleton(monkeypatch):
    """At c < 1 minimal models the regularized numeric+interpolation scalar
    path also uses the skeleton builder (collision surplus is stripped
    analytically on the skeleton)."""
    import virasoro_blocks.table_backends as cli

    calls = []
    original = cli._POLY_TABLE_BUILDERS["zzb"]

    def wrapped(*args, **kwargs):
        calls.append(bool(kwargs.get("skeleton", False)))
        return original(*args, **kwargs)

    monkeypatch.setitem(cli._POLY_TABLE_BUILDERS, "zzb", wrapped)

    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "1/2",
            "--spin-ranges", "2",
            "--lambda", "2",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--coordinates", "zzb",
            "--order", "8",
            "--npoles", "4",
            "--precision", "40",
            "--sample-via", "numeric",
            "--pole-selection", "contribution",
            "--regularize", "on",
            "--format", "json",
            "-o", d,
        ])

    assert calls
    assert all(calls)


def test_cli_off_diagonal_regularized_spin_uses_skeleton(monkeypatch):
    """Off-diagonal singular ordered_B (spinning externals) on the regularize +
    numeric + contribution path builds the numerator-free skeleton: the
    structural degrees are center-independent, and the contribution scorer reads
    block values rather than the two-sided residue series."""
    import virasoro_blocks.table_backends as cli

    calls = []
    original = cli._B_POLY_TABLE_BUILDERS["zzb"]

    def wrapped(*args, **kwargs):
        calls.append(bool(kwargs.get("skeleton", False)))
        return original(*args, **kwargs)

    monkeypatch.setitem(cli._B_POLY_TABLE_BUILDERS, "zzb", wrapped)

    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "25",
            "--spin-ranges", "2",
            "--h-ext", "1.0", "0.075", "0.075", "0.075",
            "--hbar-ext", "0.0", "0.075", "0.075", "0.075",
            "--lambda", "2",
            "--coordinates", "zzb",
            "--z-center", "0.37",
            "--zbar-center", "0.55",
            "--order", "8",
            "--npoles", "4",
            "--precision", "40",
            "--sample-via", "numeric",
            "--pole-selection", "contribution",
            "--regularize", "on",
            "--format", "json",
            "-o", d,
        ])

    assert calls
    assert all(calls)


def test_cli_regularized_projected_uses_skeleton(monkeypatch):
    """Singular off-diagonal (z != zbar) parity_projected_B on the regularize +
    numeric + contribution path builds the numerator-free skeleton: its degree
    is structural, so the symbolic numerators are never built."""
    import virasoro_blocks.table_backends as cli

    calls = []
    original = cli._PROJECTED_B_POLY_TABLE_BUILDERS["zzb"]

    def wrapped(*args, **kwargs):
        calls.append(bool(kwargs.get("skeleton", False)))
        return original(*args, **kwargs)

    monkeypatch.setitem(cli._PROJECTED_B_POLY_TABLE_BUILDERS, "zzb", wrapped)

    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "25",
            "--spin-ranges", "2",
            "--h-ext", "1.0", "0.075", "0.075", "0.075",
            "--hbar-ext", "0.0", "0.075", "0.075", "0.075",
            "--parity-project", "+", "--parity-ext", "1", "1", "1", "1",
            "--lambda", "2",
            "--coordinates", "zzb",
            "--z-center", "0.37",
            "--zbar-center", "0.55",
            "--order", "8",
            "--npoles", "4",
            "--precision", "40",
            "--sample-via", "numeric",
            "--pole-selection", "contribution",
            "--regularize", "on",
            "--format", "json",
            "-o", d,
        ])

    assert calls
    assert all(calls)


def test_cli_off_diagonal_center_metadata():
    """--z-center / --zbar-center build off-diagonal blocks and record the
    expansion centers (as z cross ratios) in the JSON; the symmetric default
    emits z_center = zbar_center = 1/2."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "2.5",
            "--spin-ranges", "0",
            "--lambda", "2",
            "--h-ext", "0.1", "0.1", "0.1", "0.1",
            "--hbar-ext", "0.1", "0.1", "0.1", "0.1",
            "--coordinates", "qqb",
            "--order", "8",
            "--precision", "40",
            "--format", "json",
            "--z-center", "0.37",
            "--zbar-center", "0.55",
            "-o", d,
        ])
        doc = json.load(open(os.path.join(d, os.listdir(d)[0])))
        assert mp.almosteq(mp.mpf(doc["z_center"]), mp.mpf("0.37"))
        assert mp.almosteq(mp.mpf(doc["zbar_center"]), mp.mpf("0.55"))


def test_cli_symmetric_default_center_metadata():
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "2.5",
            "--spin-ranges", "0",
            "--lambda", "2",
            "--delta-ext", "0.2", "0.2", "0.2", "0.2",
            "--coordinates", "qqb",
            "--order", "8",
            "--precision", "40",
            "--format", "json",
            "-o", d,
        ])
        doc = json.load(open(os.path.join(d, os.listdir(d)[0])))
        assert mp.almosteq(mp.mpf(doc["z_center"]), mp.mpf("0.5"))
        assert mp.almosteq(mp.mpf(doc["zbar_center"]), mp.mpf("0.5"))


def test_cli_chiral_center_metadata_no_zbar():
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "2.5",
            "--basis", "chiral_F",
            "--lambda", "2",
            "--h-ext", "0.1", "0.15", "0.2", "0.25",
            "--coordinates", "q",
            "--order", "8",
            "--precision", "40",
            "--format", "json",
            "--z-center", "0.3",
            "-o", d,
        ])
        doc = json.load(open(os.path.join(d, os.listdir(d)[0])))
        assert mp.almosteq(mp.mpf(doc["z_center"]), mp.mpf("0.3"))
        assert "zbar_center" not in doc


def test_cli_center_rejected_for_folded():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "2.5",
                "--spin-ranges", "0",
                "--lambda", "2",
                "--delta-ext", "0.2", "0.2", "0.2", "0.2",
                "--coordinates", "xt",
                "--order", "20",
                "--z-center", "0.3",
                "-o", d,
            ])
