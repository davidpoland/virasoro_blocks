"""Tests for the chiral-only block basis `F(h, z)` (--basis chiral_F).

The chiral builders reuse `DeltaDerivTable` with shape (m_max + 1) x 1, the
same arrangement the radial slices use. These tests cover:

  * the symbolic chiral table reproduces `F_block(h, z)` at the symmetric
    point of each chiral coordinate (z / y / q);
  * the table has poles at h = d_{r,s} (one factor each, not the 2 d_{r,s}
    +/- ell pattern of the 2d composer);
  * the truncation policy keeps `(m, 0)` for m in [0, Lambda];
  * the JSON writer flattens the redundant n = 0 index and skips the 2d-only
    metadata fields; the Mathematica writer emits `zDeriv[m]` (one index);
  * the CLI rejects incompatible flags and produces a single-file chiral
    output.
"""

from __future__ import annotations

import json
import os
import tempfile

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyderivs import (
    F_q_poly_table,
    F_y_poly_table,
    F_z_poly_table,
    interpolate_delta_table,
)
from virasoro_blocks.__main__ import main
from virasoro_blocks.coordinates import q_symmetric, y_symmetric, z_of_q, z_of_y
from virasoro_blocks.params import d_rs
from virasoro_blocks.polyderivs import DeltaDerivTable, build_numeric_sampler
from virasoro_blocks.recursion import F_block
from virasoro_blocks.serialization import serialize_deriv_tables
from virasoro_blocks.truncation import (
    kept_pairs,
    parity_smoothing_label,
    required_m_max_n_max,
)
from virasoro_blocks.writers import (
    mathematica_filename,
    write_json,
    write_mathematica,
)


C = mp.mpf("1.7")
EXTERNALS = [mp.mpf("0.1"), mp.mpf("0.15"), mp.mpf("0.125"), mp.mpf("0.175")]


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


# ---------------------------------------------------------------------------
# Symbolic chiral tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder, u_at_symmetric",
    [
        (F_z_poly_table, lambda: mp.mpf("0.5")),
        (F_y_poly_table, lambda: z_of_y(y_symmetric())),
        (F_q_poly_table, lambda: z_of_q(q_symmetric())),
    ],
)
def test_chiral_table_reconstructs_F_block(builder, u_at_symmetric):
    """At the symmetric point of each chiral coordinate, the (m=0) entry of
    the symbolic chiral table equals F_block(h, z_*) computed by direct
    recursion."""
    mp.mp.dps = 50
    b = b_from_c(C)
    table = builder(EXTERNALS, 3, b=b, q_order=8)
    assert isinstance(table, DeltaDerivTable)
    assert len(table.numerators) == 4
    assert all(len(row) == 1 for row in table.numerators)
    z_star = u_at_symmetric()
    for h in [mp.mpf("0.6"), mp.mpf("1.4"), mp.mpf("3.2")]:
        evaluated = table.evaluate(h)
        direct = F_block(h, z_star, EXTERNALS, b=b, q_order=8)
        assert _close(evaluated[0][0], direct, tol=mp.mpf("1e-30"))


def test_chiral_table_denominator_roots_are_kac_weights():
    """Chiral F(h) lives over the bare Kac denominator D(h) = prod_{r,s} (h -
    d_{r,s}). The 2d composer doubles the pole count via the (h, hbar) split;
    the chiral path keeps a single factor per Kac weight."""
    mp.mp.dps = 50
    b = b_from_c(C)
    table = F_z_poly_table(EXTERNALS, 2, b=b, q_order=6)
    expected = [
        mp.mpc(d_rs(r, s, b=b))
        for r in range(1, 7)
        for s in range(1, 7)
        if r * s <= 6
    ]
    actual = [mp.mpc(root) for root in table.denominator_roots]
    assert len(actual) == len(expected)
    # Match each computed root to a Kac weight, allowing arbitrary ordering.
    matched = [False] * len(expected)
    for root in actual:
        for j, kac in enumerate(expected):
            if not matched[j] and abs(root - kac) < mp.mpf("1e-30"):
                matched[j] = True
                break
        else:
            raise AssertionError(f"no Kac match for root {root}")
    assert all(matched)


def test_chiral_table_passes_through_interpolation():
    """The (m + 1) x 1 chiral table flows through the order-reduction
    pipeline unchanged: the interpolated table reconstructs the exact one."""
    mp.mp.dps = 60
    b = b_from_c(mp.mpf("10"))
    exact = F_z_poly_table(EXTERNALS, 3, b=b, q_order=8)
    interp = interpolate_delta_table(exact, npoles=len(exact.denominator_roots))
    assert len(interp.numerators) == 4
    assert all(len(row) == 1 for row in interp.numerators)
    for h in [mp.mpf("0.5"), mp.mpf("1.7"), mp.mpf("3.4")]:
        e = exact.evaluate(h)
        i = interp.evaluate(h)
        assert _close(e[0][0], i[0][0], tol=mp.mpf("1e-30"))


def test_chiral_numeric_sampler_matches_symbolic_interpolation():
    from virasoro_blocks.polyderivs import _NumericChiralSampler

    mp.mp.dps = 60
    c = mp.mpf("10")
    b = b_from_c(c)
    exact_sym = F_z_poly_table(EXTERNALS, 3, b=b, q_order=8)
    exact_num = F_z_poly_table(EXTERNALS, 3, b=b, q_order=8)
    exact_num.numeric_sampler = _NumericChiralSampler(
        "z",
        EXTERNALS,
        exact_num.m_max,
        b=None,
        c=c,
        q_order=8,
        chiral_norm=None,
        z_shift=None,
    )
    sym = interpolate_delta_table(exact_sym, npoles=4)
    num = interpolate_delta_table(exact_num, npoles=4)

    for h in [mp.mpf("0.5"), mp.mpf("1.7"), mp.mpf("3.4")]:
        s = sym.evaluate(h)
        n = num.evaluate(h)
        assert _close(s[0][0], n[0][0], tol=mp.mpf("1e-30"))


def test_chiral_table_passes_through_serialization():
    """`serialize_deriv_tables` accepts the chiral table unchanged."""
    mp.mp.dps = 50
    b = b_from_c(C)
    table = F_z_poly_table(EXTERNALS, 2, b=b, q_order=6)
    serialized = serialize_deriv_tables({"z": table})
    assert serialized.coordinates == ("z",)
    assert len(serialized.numerators["z"]) == 3
    assert all(len(row) == 1 for row in serialized.numerators["z"])


# ---------------------------------------------------------------------------
# Truncation policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coord", ["z", "y", "q"])
def test_chiral_kept_pairs_are_m_only(coord):
    pairs = kept_pairs(coord, 4, basis="chiral_F")
    assert pairs == [(m, 0) for m in range(5)]


def test_chiral_kept_pairs_rejects_2d_basis():
    with pytest.raises(ValueError):
        kept_pairs("z", 3, basis="scalar_G")


def test_chiral_parity_smoothing_is_none():
    """Chiral blocks live on a single complex coordinate; the p_+/- smoothing
    factor is a 2d concept and has no chiral analogue."""
    assert parity_smoothing_label("chiral_F") is None


def test_chiral_required_m_max_n_max():
    assert required_m_max_n_max("z", 4, basis="chiral_F") == (4, 0)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _serialized_chiral():
    mp.mp.dps = 40
    b = b_from_c(C)
    exact = F_z_poly_table(EXTERNALS, 2, b=b, q_order=6)
    return serialize_deriv_tables({"z": exact})


def test_json_chiral_drops_2d_only_fields():
    s = _serialized_chiral()
    doc = json.loads(
        write_json(s, c=C, h_ext=EXTERNALS, basis="chiral_F", q_order=6)
    )
    assert doc["basis"] == "chiral_F"
    assert doc["label"] == "chiral_h"
    assert doc["hbar_external"] is None
    assert doc["delta_external"] is None
    assert doc["spin_external"] is None
    assert doc["spin"] is None
    assert doc["parity_projection"] is None
    assert doc["parity_smoothing"] is None
    assert doc["index_names"] == ["coordinate_index", "deriv", "monomial_degree"]


def test_json_chiral_derivs_shape_is_flat_per_m():
    """derivs[coord][m][k] (no n axis): Lambda = 2 -> three m rows, each a
    single polynomial coefficient list."""
    s = _serialized_chiral()
    doc = json.loads(
        write_json(s, c=C, h_ext=EXTERNALS, basis="chiral_F", q_order=6)
    )
    derivs = doc["derivs"]
    assert len(derivs) == 1            # one coordinate
    assert len(derivs[0]) == 3         # m = 0, 1, 2
    for row in derivs[0]:
        # each row is a flat list of string coefficients, not [[coeffs]]
        assert isinstance(row, list)
        assert all(isinstance(c, str) for c in row)


def test_mathematica_chiral_uses_single_index():
    s = _serialized_chiral()
    text = write_mathematica(s, coordinate="z", basis="chiral_F")
    assert "zDeriv[0] ->" in text
    assert "zDeriv[1] ->" in text
    assert "zDeriv[0,0]" not in text       # no 2d-style two-index entries
    assert 'basis -> "chiral_F"' in text
    assert "paritySmoothing -> Null" in text


def test_mathematica_chiral_self_contained_omits_antiholomorphic():
    """The chiral .m carries the reconstruction data (expBase/shift) and the
    holomorphic externals, but the antiholomorphic / exchanged-spin fields are
    Null and the zbar centers/shifts are omitted (a chiral half has no zbar)."""
    s = _serialized_chiral()
    text = write_mathematica(
        s, coordinate="z", basis="chiral_F",
        c=C, h_ext=EXTERNALS, q_order=6,
    )
    assert "expBase ->" in text and "shift ->" in text
    assert "hExternal -> {" in text
    assert "hbarExternal -> Null" in text
    assert "deltaExternal -> Null" in text
    assert "spinExternal -> Null" in text
    assert "spin -> Null" in text
    assert "zbarCenter" not in text and "zbarShift" not in text


def test_mathematica_chiral_filename_uses_h_labels_and_no_spin():
    fname = mathematica_filename(
        "z",
        c=C,
        h_ext=EXTERNALS,
        hbar_ext=None,
        spin=None,
        q_order=6,
        m_max=2,
        n_max=0,
        basis="chiral_F",
    )
    assert "basisChiralF" in fname
    assert "h1-0.1" in fname
    assert "h4-0.175" in fname
    # No exchanged spin in the chiral filename.
    assert "-L" not in fname.replace("lambda", "")
    # No 2d delta tags.
    assert "delta1" not in fname


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


_CHIRAL_TINY = [
    "--central-charge", "1.7",
    "--h-ext", "0.1", "0.15", "0.125", "0.175",
    "--basis", "chiral_F",
    "--coordinates", "z", "y", "q",
    "--lambda", "2",
    "--order", "6",
    "--precision", "30",
]


def test_cli_chiral_writes_single_json():
    with tempfile.TemporaryDirectory() as d:
        main(_CHIRAL_TINY + [
            "--format", "json", "--npoles", "4",
            "-o", d,
        ])
        json_files = [f for f in os.listdir(d) if f.endswith(".json")]
        # Chiral output has no exchanged spin: exactly one JSON file.
        assert len(json_files) == 1
        doc = json.loads(open(os.path.join(d, json_files[0])).read())
        assert doc["basis"] == "chiral_F"
        assert doc["label"] == "chiral_h"
        assert set(doc["coordinates"]) == {"z", "y", "q"}
        # Lambda = 2 -> 3 m-rows per coordinate; chiral has no n axis.
        assert len(doc["derivs"]) == 3
        assert all(len(coord_block) == 3 for coord_block in doc["derivs"])


def test_cli_chiral_mathematica_filenames():
    with tempfile.TemporaryDirectory() as d:
        main(_CHIRAL_TINY + [
            "--format", "m", "--npoles", "4",
            "-o", d,
        ])
        mfiles = sorted(f for f in os.listdir(d) if f.endswith(".m"))
        assert len(mfiles) == 3          # one per chiral coordinate
        assert any(f.startswith("zDerivTable") for f in mfiles)
        assert all("basisChiralF" in f for f in mfiles)


def test_cli_chiral_sample_via_numeric_records_metadata():
    with tempfile.TemporaryDirectory() as d:
        main(_CHIRAL_TINY + [
            "--format", "json", "--npoles", "4",
            "--sample-via", "numeric", "-o", d,
        ])
        json_files = [f for f in os.listdir(d) if f.endswith(".json")]
        assert len(json_files) == 1
        doc = json.loads(open(os.path.join(d, json_files[0])).read())
        assert doc["sample_via"] == "numeric"


def test_chiral_contribution_skeleton_matches_full():
    mp.mp.dps = 70
    q_order = 10
    lam = 4

    full = F_q_poly_table(EXTERNALS, lam, c=C, q_order=q_order)
    skel = F_q_poly_table(EXTERNALS, lam, c=C, q_order=q_order, skeleton=True)
    assert skel.numerators is None

    sampler_kwargs = dict(
        coordinate="q", chiral=True, spinning=False, projected=False,
        h_externals=EXTERNALS, hbar_externals=None, ell=None,
        m_max=lam, n_max=0, b=None, c=C, q_order=q_order,
        chiral_norm=None, z_shift=None, zbar_shift=None,
        center=None, center_bar=None, parity=None,
        parity_external=None, exchanged_parity=None,
    )
    full.numeric_sampler = build_numeric_sampler(**sampler_kwargs)
    skel.numeric_sampler = build_numeric_sampler(**sampler_kwargs)

    interp_full = interpolate_delta_table(
        full, npoles=4, pole_selection="contribution", pole_ranking="numeric"
    )
    interp_skel = interpolate_delta_table(
        skel, npoles=4, pole_selection="contribution", pole_ranking="numeric"
    )

    for h in [mp.mpf("0.2"), mp.mpf("0.5"), mp.mpf("1.0")]:
        got = interp_skel.evaluate(h)
        want = interp_full.evaluate(h)
        worst = max(
            abs(mp.mpc(got[m][0]) - mp.mpc(want[m][0]))
            / max(abs(mp.mpc(want[m][0])), mp.mpf("1e-30"))
            for m in range(len(want))
        )
        assert worst < mp.mpf("1e-25"), f"h={h}: rel err {mp.nstr(worst, 5)}"


def test_cli_chiral_rejects_spin_ext():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(_CHIRAL_TINY + [
                "--spin-ext", "2", "0", "0", "0",
                "--format", "json", "-o", d,
            ])


def test_cli_chiral_rejects_parity_project():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main(_CHIRAL_TINY + [
                "--parity-project", "+", "--parity-ext", "1", "1", "1", "1",
                "--format", "json", "-o", d,
            ])


def test_cli_chiral_rejects_2d_coordinates():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "1.7",
                "--h-ext", "0.1", "0.15", "0.125", "0.175",
                "--basis", "chiral_F",
                "--coordinates", "zzb",
                "--lambda", "2", "--order", "6", "--precision", "30",
                "--format", "json", "-o", d,
            ])


def test_cli_chiral_requires_h_ext():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "1.7",
                "--basis", "chiral_F",
                "--coordinates", "z",
                "--lambda", "2", "--order", "6", "--precision", "30",
                "--format", "json", "-o", d,
            ])


def test_cli_2d_path_rejects_h_ext_without_hbar_ext():
    """On the 2d bases the (h, hbar) parametrization needs both halves; --h-ext
    alone is ambiguous and is rejected. The chiral path is the only mode
    where --h-ext stands alone."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "10",
                "--spin-ranges", "0",
                "--h-ext", "0.075", "0.075", "0.075", "0.075",
                "--lambda", "2", "--order", "6", "--precision", "30",
                "--format", "json", "-o", d,
            ])


def test_cli_chiral_auto_basis_inferred_from_chiral_coords():
    """Passing only chiral coords (no explicit --basis chiral_F) auto-resolves
    to the chiral basis."""
    with tempfile.TemporaryDirectory() as d:
        main([
            "--central-charge", "1.7",
            "--h-ext", "0.1", "0.15", "0.125", "0.175",
            "--coordinates", "z",
            "--lambda", "2", "--order", "6", "--precision", "30",
            "--format", "json", "--npoles", "4",
            "-o", d,
        ])
        fpath = next(
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")
        )
        doc = json.loads(open(fpath).read())
        assert doc["basis"] == "chiral_F"


def test_cli_rejects_mixing_chiral_and_2d_coords():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "--central-charge", "1.7",
                "--h-ext", "0.1", "0.15", "0.125", "0.175",
                "--coordinates", "z", "zzb",
                "--lambda", "2", "--order", "6", "--precision", "30",
                "--format", "json", "-o", d,
            ])
