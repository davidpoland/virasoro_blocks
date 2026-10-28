"""Round-trip tests for the JSON -> SerializedDerivTable loader.

These verify that :func:`virasoro_blocks.serialized_table_from_json` is a faithful
inverse of :func:`virasoro_blocks.writers.write_json`: building a table, writing
it to the canonical JSON, and loading it back reproduces the block derivative
values at every kept ``(m, n)`` pair to the serialized precision.
"""

from __future__ import annotations

import json

import mpmath as mp
import pytest

from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
)
from virasoro_bootstrap.large_spin import double_twist_set
from virasoro_blocks.params import b_from_c
from virasoro_blocks.polyderivs import (
    B_zzb_poly_table,
    F_z_poly_table,
    G_yyb_poly_table,
    G_zzb_poly_table,
    interpolate_delta_table,
    parity_projected_B_zzb_poly_table,
)
from virasoro_blocks.large_spin import large_spin_twist_table
from virasoro_blocks.truncation import kept_pairs


def _assert_sample_block_values_roundtrip(doc, original, reloaded, *, coordinates):
    """Compare clean node values over the kept (m, n) pairs the writer emits.

    ``sample_block_values`` is serialized in the same jagged kept-pairs shape as
    ``derivs`` (the non-kept n > m symmetry half and the m + n > lambda corner
    are not stored, as they are never read by the crossing functional), so the
    round trip is faithful exactly on the kept pairs.
    """
    tol = mp.mpf(10) ** (-(mp.mp.dps - 6))
    for coord in coordinates:
        pairs = kept_pairs(
            coord,
            doc["lambda"],
            basis=doc.get("basis"),
            ell=doc.get("spin") or 0,
            parity_projection=doc.get("parity_projection"),
        )
        for table_a, table_b in zip(original[coord], reloaded[coord]):
            for (m, n) in pairs:
                a = mp.mpc(table_a[m][n])
                b = mp.mpc(table_b[m][n])
                scale = max(mp.mpf(1), abs(a))
                assert abs(a - b) <= tol * scale
from virasoro_blocks.serialization import (
    serialize_deriv_tables,
    serialized_table_from_json,
)
from virasoro_blocks.truncation import kept_pairs
from virasoro_blocks.writers import write_json


C = mp.mpf("10")
SCALAR_EXT = [mp.mpf("0.15")] * 4
Q_ORDER = 8
LAM = 3


def _assert_tables_match(original, reloaded, *, basis, ell, parity, deltas):
    """Compare block values at every kept (m, n) pair across several Delta."""
    tol = mp.mpf(10) ** (-(mp.mp.dps - 4))
    assert reloaded.coordinates == original.coordinates
    for coord in original.coordinates:
        pairs = kept_pairs(
            coord, LAM, basis=basis, ell=ell, parity_projection=parity
        )
        for delta in deltas:
            orig_vals = original.evaluate(coord, delta)
            reload_vals = reloaded.evaluate(coord, delta)
            for (m, n) in pairs:
                a = mp.mpc(orig_vals[m][n])
                b = mp.mpc(reload_vals[m][n])
                scale = max(mp.mpf(1), abs(a))
                assert abs(a - b) <= tol * scale, (
                    f"{coord} basis={basis} ell={ell} parity={parity} "
                    f"(m,n)=({m},{n}) Delta={delta}: |{a} - {b}|"
                )


def test_roundtrip_scalar_G_even_spin():
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 2
    exact = G_zzb_poly_table(ell, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    s = serialize_deriv_tables({"zzb": exact})
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell, q_order=Q_ORDER)
    )
    reloaded = serialized_table_from_json(doc)
    deltas = [mp.mpf(ell) + d for d in (mp.mpf("0.5"), mp.mpf("1.3"), mp.mpf("3"))]
    _assert_tables_match(s, reloaded, basis="scalar_G", ell=ell, parity=None, deltas=deltas)


def test_roundtrip_scalar_G_odd_spin():
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 1
    exact = G_zzb_poly_table(ell, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    s = serialize_deriv_tables({"zzb": exact})
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell, q_order=Q_ORDER)
    )
    reloaded = serialized_table_from_json(doc)
    deltas = [mp.mpf(ell) + d for d in (mp.mpf("0.5"), mp.mpf("2.1"))]
    _assert_tables_match(s, reloaded, basis="scalar_G", ell=ell, parity=None, deltas=deltas)


def test_roundtrip_ordered_B():
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 2
    exact = B_zzb_poly_table(ell, SCALAR_EXT, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    s = serialize_deriv_tables({"zzb": exact})
    doc = json.loads(
        write_json(
            s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell,
            q_order=Q_ORDER, basis="ordered_B",
        )
    )
    reloaded = serialized_table_from_json(doc)
    deltas = [mp.mpf(ell) + d for d in (mp.mpf("0.7"), mp.mpf("2.4"))]
    _assert_tables_match(s, reloaded, basis="ordered_B", ell=ell, parity=None, deltas=deltas)


@pytest.mark.parametrize("parity", ["+", "-"])
def test_roundtrip_parity_projected_B(parity):
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 2
    exact = parity_projected_B_zzb_poly_table(
        ell, SCALAR_EXT, SCALAR_EXT, LAM, LAM, parity=parity, b=b, q_order=Q_ORDER
    )
    s = serialize_deriv_tables({"zzb": exact})
    doc = json.loads(
        write_json(
            s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell,
            q_order=Q_ORDER, basis="parity_projected_B",
            parity_projection=parity,
            parity_external=["1", "1", "1", "1"],
            parity_exchanged="1",
        )
    )
    reloaded = serialized_table_from_json(doc)
    deltas = [mp.mpf(ell) + d for d in (mp.mpf("0.6"), mp.mpf("1.9"))]
    _assert_tables_match(s, reloaded, basis="parity_projected_B", ell=ell, parity=parity, deltas=deltas)


def test_roundtrip_chiral_F():
    mp.mp.dps = 40
    b = b_from_c(C)
    exact = F_z_poly_table(SCALAR_EXT, LAM, b=b, q_order=Q_ORDER)
    s = serialize_deriv_tables({"z": exact})
    doc = json.loads(
        write_json(
            s, c=C, h_ext=SCALAR_EXT, hbar_ext=None, spin=None,
            q_order=Q_ORDER, basis="chiral_F",
        )
    )
    reloaded = serialized_table_from_json(doc)
    deltas = [mp.mpf("0.3"), mp.mpf("1.2"), mp.mpf("2.7")]
    _assert_tables_match(s, reloaded, basis="chiral_F", ell=0, parity=None, deltas=deltas)


def test_roundtrip_interpolation_method():
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 2
    exact = G_zzb_poly_table(ell, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    interp = interpolate_delta_table(exact, npoles=4)
    s = serialize_deriv_tables({"zzb": interp})
    doc = json.loads(
        write_json(
            s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell,
            q_order=Q_ORDER, npoles=4,
        )
    )
    reloaded = serialized_table_from_json(doc)
    assert len(reloaded.pole_list_x) == len(s.pole_list_x)
    deltas = [mp.mpf(ell) + d for d in (mp.mpf("0.5"), mp.mpf("1.3"))]
    _assert_tables_match(s, reloaded, basis="scalar_G", ell=ell, parity=None, deltas=deltas)


def test_roundtrip_preserves_clean_sample_values_and_sample_via():
    mp.mp.dps = 80
    spec = IdenticalScalarGapSpec(
        c=mp.mpf("4"), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf("0.5"),
        lam=3, spin_list=(0, 2), q_order=14, dps=80,
        npoles=4, sample_via="numeric", num_threads=1,
    )
    channel = build_identical_scalar_gap_problem(spec).spin_channels[0]
    serialized = channel.serialized

    doc = json.loads(
        write_json(
            serialized,
            c=spec.c,
            h_ext=[spec.h_ext] * 4,
            hbar_ext=[spec.h_ext] * 4,
            spin=channel.ell,
            q_order=spec.q_order,
            npoles=spec.npoles,
            sample_via=spec.sample_via,
        )
    )
    assert doc["sample_via"] == "numeric"
    assert doc.get("sample_block_values") is not None

    reloaded = serialized_table_from_json(doc)
    assert reloaded.sample_via == "numeric"
    assert reloaded.sample_points_x == serialized.sample_points_x
    assert reloaded.sample_block_values is not None
    assert sorted(reloaded.sample_block_values) == sorted(serialized.sample_block_values)

    _assert_sample_block_values_roundtrip(
        doc, serialized.sample_block_values, reloaded.sample_block_values,
        coordinates=serialized.coordinates,
    )


def test_roundtrip_multi_coordinate():
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 2
    exact_zzb = G_zzb_poly_table(ell, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    exact_yyb = G_yyb_poly_table(ell, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    s = serialize_deriv_tables({"zzb": exact_zzb, "yyb": exact_yyb})
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell, q_order=Q_ORDER)
    )
    reloaded = serialized_table_from_json(doc)
    assert reloaded.coordinates == ("zzb", "yyb")
    deltas = [mp.mpf(ell) + d for d in (mp.mpf("0.5"), mp.mpf("1.3"))]
    _assert_tables_match(s, reloaded, basis="scalar_G", ell=ell, parity=None, deltas=deltas)


def test_roundtrip_preserves_base_shift_poles():
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 2
    exact = G_zzb_poly_table(ell, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    s = serialize_deriv_tables({"zzb": exact})
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell, q_order=Q_ORDER)
    )
    reloaded = serialized_table_from_json(doc)
    tol = mp.mpf(10) ** (-(mp.mp.dps - 4))
    assert abs(mp.mpc(reloaded.base) - mp.mpc(s.base)) <= tol
    assert abs(mp.mpf(reloaded.shift) - mp.mpf(s.shift)) <= tol
    assert len(reloaded.pole_list_x) == len(s.pole_list_x)
    # Match each reloaded pole to its nearest original pole rather than zipping
    # the (re, im)-sorted lists: several poles share a real part (conjugate
    # families), so a serialization round-trip perturbation of ~1e-dps can flip
    # their sort order and mis-pair the zip even though the pole *set* is
    # preserved. Nearest-neighbor matching tests set equality, which is the
    # actual invariant.
    remaining = [mp.mpc(p) for p in s.pole_list_x]
    for pa in reloaded.pole_list_x:
        pa = mp.mpc(pa)
        j = min(range(len(remaining)), key=lambda i: abs(pa - remaining[i]))
        assert abs(pa - remaining[j]) <= tol * max(mp.mpf(1), abs(remaining[j]))
        remaining.pop(j)


def test_default_variable_is_delta():
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 2
    exact = G_zzb_poly_table(ell, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    s = serialize_deriv_tables({"zzb": exact})
    doc = json.loads(
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell, q_order=Q_ORDER)
    )
    assert doc["variable"] == "Delta"
    assert "twist" not in doc
    assert "large_spin_lmin" not in doc


def test_write_json_rejects_unknown_variable():
    mp.mp.dps = 40
    b = b_from_c(C)
    ell = 2
    exact = G_zzb_poly_table(ell, SCALAR_EXT, LAM, LAM, b=b, q_order=Q_ORDER)
    s = serialize_deriv_tables({"zzb": exact})
    with pytest.raises(ValueError, match="variable must be"):
        write_json(s, c=C, h_ext=SCALAR_EXT, hbar_ext=SCALAR_EXT, spin=ell,
                   q_order=Q_ORDER, variable="L_bad")


def test_roundtrip_large_spin_variable_L():
    """A fixed-twist large-spin table writes with variable='L' and reloads to
    the same block values at every kept (m, n) pair, evaluated in L."""
    coordinate = "zzb"
    mp.mp.dps = 40
    he = mp.mpf("0.0625")
    z_shift = -2 * he
    twist = mp.mpf("0.25")
    lmax = 14
    m_max = n_max = 2
    interp = large_spin_twist_table(
        twist, mode="virasoro", coordinate=coordinate,
        h_externals=[he] * 4, m_max=m_max, n_max=n_max,
        Lmax=lmax, npoles=6, c=C, q_order=12, z_shift=z_shift,
    )
    s = serialize_deriv_tables({coordinate: interp})
    doc = json.loads(
        write_json(
            s, c=C, h_ext=[he] * 4, hbar_ext=[he] * 4, spin=0,
            q_order=12, npoles=6, variable="L", twist=twist, large_spin_lmin=lmax,
        )
    )
    assert doc["variable"] == "L"
    assert doc["coordinates"] == [coordinate]
    assert mp.almosteq(mp.mpf(doc["twist"]), twist)
    assert mp.almosteq(mp.mpf(doc["large_spin_lmin"]), mp.mpf(lmax))

    reloaded = serialized_table_from_json(doc)
    pairs = kept_pairs(coordinate, max(m_max, n_max), basis="scalar_G", ell=0)
    tol = mp.mpf(10) ** (-(mp.mp.dps - 6))
    for L in (mp.mpf(16), mp.mpf(20), mp.mpf(24)):
        orig_vals = s.evaluate(coordinate, L)
        reload_vals = reloaded.evaluate(coordinate, L)
        scale = max(
            abs(mp.mpc(orig_vals[m][n])) for (m, n) in pairs
        )
        scale = max(scale, mp.mpf(1))
        for (m, n) in pairs:
            a = mp.mpc(orig_vals[m][n])
            b = mp.mpc(reload_vals[m][n])
            assert abs(a - b) <= tol * scale, (
                f"{coordinate} (m,n)=({m},{n}) L={L}: |{a} - {b}|"
            )


def test_roundtrip_large_spin_preserves_clean_sample_values_and_sample_via():
    mp.mp.dps = 40
    h_ext = mp.mpf("0.0625")
    twists = tuple(double_twist_set(h_ext, 1))
    spec = IdenticalScalarGapSpec(
        c=C, h_ext=h_ext, delta_gap=mp.mpf("1.0"),
        lam=3, q_order=10, dps=40, npoles=6,
        spin_list=(0,), large_spin_lmin=2, large_spin_twists=twists,
        num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)
    channel = problem.spin_channels[-1]

    doc = json.loads(
        write_json(
            channel.serialized,
            c=spec.c,
            h_ext=[h_ext] * 4,
            hbar_ext=[h_ext] * 4,
            spin=channel.ell,
            q_order=spec.q_order,
            npoles=spec.npoles,
            variable="L",
            twist=twists[0],
            large_spin_lmin=channel.delta_min,
        )
    )
    assert doc["sample_via"] == "numeric"
    assert doc.get("sample_block_values") is not None

    reloaded = serialized_table_from_json(doc)
    assert reloaded.sample_via == "numeric"
    assert reloaded.sample_block_values is not None

    _assert_sample_block_values_roundtrip(
        doc, channel.serialized.sample_block_values, reloaded.sample_block_values,
        coordinates=channel.serialized.coordinates,
    )


def test_roundtrip_large_spin_symbolic_preserves_sample_via_without_clean_values():
    mp.mp.dps = 40
    h_ext = mp.mpf("0.0625")
    twists = tuple(double_twist_set(h_ext, 1))
    spec = IdenticalScalarGapSpec(
        c=C, h_ext=h_ext, delta_gap=mp.mpf("1.0"),
        lam=3, q_order=10, dps=40, npoles=6,
        spin_list=(0,), sample_via="symbolic",
        large_spin_lmin=2, large_spin_twists=twists,
        num_threads=1,
    )
    problem = build_identical_scalar_gap_problem(spec)
    channel = problem.spin_channels[-1]

    doc = json.loads(
        write_json(
            channel.serialized,
            c=spec.c,
            h_ext=[h_ext] * 4,
            hbar_ext=[h_ext] * 4,
            spin=channel.ell,
            q_order=spec.q_order,
            npoles=spec.npoles,
            variable="L",
            twist=twists[0],
            large_spin_lmin=channel.delta_min,
        )
    )
    assert doc["sample_via"] == "symbolic"
    assert doc.get("sample_block_values") is None

    reloaded = serialized_table_from_json(doc)
    assert reloaded.sample_via == "symbolic"
    assert reloaded.sample_block_values is None
