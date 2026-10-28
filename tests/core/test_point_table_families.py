"""Point tables for every family: the torus characters (irreducible on a Kac weight), the torus one-point blocks and
the NS/Ramond sphere blocks.

A point table freezes a block at one exchanged value (degree-0 numerators, ``shift`` the value).  The torus character
table is the reduced character of the module (h, hbar) = ((Delta + j)/2, (Delta - j)/2) paired with its mirror, with the
irreducible character on a Kac weight: it must be ``torus_vacuum_table`` at (0, 0), ``torus_current_table`` at (j, 0),
``torus_character_table`` at a generic weight, and ``evaluate --surface torus`` (which resolves Kac weights through the
Gram ranks of L(h)) at a minimal-model primary.  The one-point and NS/Ramond point tables must agree with ``evaluate``
and with their Delta tables off the Kac weights; every file a CLI run writes must load back.
"""

from __future__ import annotations

import json
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main
from virasoro_blocks.api import build_table_request
from virasoro_blocks.evaluation import evaluate_block_request
from virasoro_blocks.modular import (
    torus_character_point_values,
    torus_character_table,
    torus_current_table,
    torus_vacuum_table,
)
from virasoro_blocks.requests import BlockRequest, NumericalPolicy, capabilities_for
from virasoro_blocks.serialization import serialized_table_from_json
from virasoro_blocks.torus_onept import irreducible_character_levels, irreducible_onept_F_series


def _partitions(n_max):
    out = [1] + [0] * n_max
    for k in range(1, n_max + 1):
        for n in range(k, n_max + 1):
            out[n] += out[n - k]
    return out


@pytest.mark.parametrize("c, h", [
    (Fraction(1, 2), Fraction(0)), (Fraction(1, 2), Fraction(1, 16)), (Fraction(1, 2), Fraction(1, 2)),
    (Fraction(7, 10), Fraction(3, 80)), (Fraction(7, 10), Fraction(3, 2)), (Fraction(-22, 5), Fraction(-1, 5)),
    (Fraction(1), Fraction(1)), (Fraction(-2), Fraction(3, 8)), (Fraction(26), Fraction(0)),
    (Fraction(1, 2), Fraction(5, 2)),
])
def test_moebius_characters_are_the_gram_rank_dimensions(c, h):
    """The embedding-poset character (braids, chains, the one null of a boundary weight) has exactly the level
    dimensions of L(h) the Gram ranks give, through level 12."""
    n_max = 12
    levels = irreducible_character_levels(c, h, n_max)
    p = _partitions(n_max)
    dims = [sum(mu * p[n - level] for level, mu in levels if level <= n) for n in range(n_max + 1)]
    assert dims == [int(x) for x in irreducible_onept_F_series(h, c, 0, n_max, exact=True)]


def test_moebius_character_recognizes_verma_modules_and_far_nulls():
    assert irreducible_character_levels(Fraction(1, 2), Fraction(3, 20), 30) is None
    # Ising vacuum: the Rocha-Caridi exponents 0, 1, 6, 11, 13, 20, ...
    assert irreducible_character_levels(Fraction(1, 2), Fraction(0), 20)[:6] == [
        (0, 1), (1, -1), (6, -1), (11, 1), (13, 1), (20, -1)]
    # a degenerate weight whose first null lies beyond the levels asked for is still degenerate
    assert irreducible_character_levels(Fraction(1, 2), Fraction(5, 2), 3) == [(0, 1)]


@pytest.mark.parametrize("c", [Fraction(5, 2), Fraction(26), "3.7"])
def test_character_point_values_are_the_vacuum_and_current_tables(c):
    """At c > 1 the vacuum (0, 0) and the currents (j, 0) are the only degenerate modules: their point tables are
    the modular bootstrap's vacuum and current tables, entry for entry."""
    mp.mp.dps = 40
    lam = 8
    table, irreducible = torus_character_point_values(c, 0, 0, lam)
    assert irreducible and table == torus_vacuum_table(c, lam)
    for spin in (1, 2, 3):
        table, irreducible = torus_character_point_values(c, spin, spin, lam)
        assert irreducible and table == torus_current_table(c, spin, lam)


@pytest.mark.parametrize("c, delta, spin", [
    (Fraction(5, 2), Fraction(7, 5), 0), (Fraction(5, 2), Fraction(13, 4), 2), (Fraction(1, 2), Fraction(3, 10), 0),
    (Fraction(7, 10), Fraction(23, 10), 1),     # (33/20, 13/20): neither is a Kac weight of c = 7/10
])
def test_character_point_values_at_generic_weights_are_the_delta_table(c, delta, spin):
    mp.mp.dps = 40
    lam = 8
    table, irreducible = torus_character_point_values(c, delta, spin, lam)
    reference = torus_character_table(c, spin, lam).evaluate("vvb", delta)
    assert not irreducible
    for row, ref_row in zip(table, reference):
        for value, ref in zip(row, ref_row):
            assert abs(value - ref) <= mp.mpf(10) ** -30 * abs(ref)


def _torus_value(family, c, delta, spin, v, vbar, *, d_ext=None, order=None, precision=30):
    request = BlockRequest(
        operation="evaluate", surface="torus", family=family, coordinate="vvb", central_charge=c,
        exchange_delta=delta, exchange_spin=spin, torus_onepoint_ext=d_ext, point={"v": v, "vbar": vbar},
        policy=NumericalPolicy(precision=precision, order=order),
    )
    result = evaluate_block_request(request)
    return result.value, result.metadata["regularization_policy"]


def _taylor_sum(table, v, vbar):
    return sum(table[m][n] * v ** m * vbar ** n for m in range(len(table)) for n in range(len(table[m])))


@pytest.mark.parametrize("delta, spin", [
    ("1/8", 0),     # (sigma, sigma): the braid characters of h = 1/16
    ("1", 0),       # (epsilon, epsilon)
    ("0", 0),       # the Ising vacuum: nulls at levels 1 and 6 (not torus_vacuum_table's 1 - q)
    ("3", 2),       # (5/2, 1/2): a Kac weight outside the Kac table (one null, level 5) with epsilon
])
def test_ising_character_point_tables_are_the_evaluated_irreducible_characters(delta, spin):
    """At c = 1/2 the point table on Kac weights is the irreducible character evaluate gives (its Gram-rank
    dimensions): the (0, 0) entry at v = 0, and the Taylor sum near it."""
    mp.mp.dps = 30
    lam = 16
    table, irreducible = torus_character_point_values(Fraction(1, 2), Fraction(delta), spin, lam)
    assert irreducible
    at_zero, policy = _torus_value("torus-character", "1/2", delta, spin, 0, 0)
    assert policy == "irreducible"
    assert abs(table[0][0] - at_zero) <= mp.mpf(10) ** -27 * abs(at_zero)
    v, vbar = mp.mpf("0.002"), mp.mpf("-0.0015")
    near, _ = _torus_value("torus-character", "1/2", delta, spin, v, vbar)
    assert abs(_taylor_sum(table, v, vbar) - near) <= mp.mpf(10) ** -27 * abs(near)
    if delta == "0":
        vacuum = torus_vacuum_table(Fraction(1, 2), lam)
        assert abs(vacuum[0][0] - table[0][0]) > mp.mpf(10) ** -20      # the level-6 null


def test_onepoint_point_table_matches_evaluate_and_the_kernel_table():
    from virasoro_blocks.torus_onept import build_torus_onept_state, torus_onept_table_values

    mp.mp.dps = 30
    lam, order = 6, 12
    for delta, spin in (("17/10", 0), ("33/10", 2)):
        request = BlockRequest(
            operation="point-table", surface="torus", family="torus-onepoint", coordinate="vvb",
            central_charge="5/2", torus_onepoint_ext="1/5", exchange_delta=delta, exchange_spin=spin,
            policy=NumericalPolicy(precision=30, order=order, lambda_order=lam),
        )
        table = build_table_request(request)
        assert table.metadata["regularization_policy"] == "direct"
        assert table.verified_digits > 25 and table.metadata["q_truncation_digits"] is not None
        values = [[poly.coeffs[0] for poly in row] for row in table.numerators["vvb"]]
        kernel = build_torus_onept_state(order, d_ext=Fraction(1, 5), c=Fraction(5, 2))
        direct = torus_onept_table_values(kernel, spin, lam, Fraction(delta))
        for row, ref_row in zip(values, direct):
            for value, ref in zip(row, ref_row):
                assert abs(value - ref) <= mp.mpf(10) ** -24 * max(1, abs(ref))
        at_zero, _ = _torus_value("torus-onepoint", "5/2", delta, spin, 0, 0, d_ext="1/5", order=order)
        assert abs(values[0][0] - at_zero) <= mp.mpf(10) ** -25 * abs(at_zero)
        # the one-point Delta table (an order-reduced fit) agrees to its verified digits
        delta_table = build_table_request(BlockRequest(
            operation="table", surface="torus", family="torus-onepoint", coordinate="vvb", coordinates=("vvb",),
            central_charge="5/2", torus_onepoint_ext="1/5",
            policy=NumericalPolicy(precision=30, order=order, lambda_order=2, npoles=6),
        ), spin=spin)
        fitted = delta_table.evaluate("vvb", Fraction(delta))
        for m in range(3):
            for n in range(min(m, 2 - m) + 1):
                assert abs(fitted[m][n] - values[m][n]) <= mp.mpf(10) ** -(delta_table.verified_digits - 1) * abs(
                    values[m][n])


def test_onepoint_point_table_on_a_kac_weight_is_the_irreducible_trace():
    """Ising <epsilon> on sigma (c = 1/2, d = 1/2, (h, hbar) = (1/16, 1/16)): the trace over L(1/16), as evaluate."""
    mp.mp.dps = 30
    order = 8
    request = BlockRequest(
        operation="point-table", surface="torus", family="torus-onepoint", coordinate="vvb", central_charge="1/2",
        torus_onepoint_ext="1/2", exchange_delta="1/8",
        policy=NumericalPolicy(precision=30, order=order, lambda_order=4),
    )
    table = build_table_request(request)
    assert table.metadata["regularization_policy"] == "irreducible"
    value = table.numerators["vvb"][0][0].coeffs[0]
    at_zero, policy = _torus_value("torus-onepoint", "1/2", "1/8", 0, 0, 0, d_ext="1/2", order=order)
    assert policy == "irreducible"
    assert abs(value - at_zero) <= mp.mpf(10) ** -27 * abs(at_zero)


@pytest.mark.parametrize("family, c", [("ns", "13/10"), ("ramond", "13/10"), ("ns", "7/10")])
def test_super_point_tables_match_evaluate_and_the_delta_table(family, c):
    """NS/Ramond point tables (generic c and the N=1 minimal model c = 7/10, where they take the c -> c0 limit) are
    the evaluated block at the expansion point and the Delta table's entries, in folded coordinates too."""
    mp.mp.dps = 30
    ext = dict(h_ext=("1/10",) * 4, hbar_ext=("1/10",) * 4)
    coordinates = ("zzb", "ws", "xt_radial")

    def request(operation, **extra):
        return BlockRequest(operation=operation, family=family, basis="scalar_G", coordinate=coordinates[0],
                            coordinates=coordinates, central_charge=c, **ext, **extra,
                            policy=NumericalPolicy(precision=30, order=10, lambda_order=3))

    point = build_table_request(request("point-table", exchange_delta="7/5", exchange_spin=2))
    assert point.metadata["block_family"] == family
    assert point.metadata["regularization_policy"] == ("merged" if c == "7/10" else "direct")
    value = evaluate_block_request(BlockRequest(
        operation="evaluate", family=family, basis="scalar_G", coordinate="zzb", central_charge=c, **ext,
        exchange_delta="7/5", exchange_spin=2, point={"z": "1/2", "zbar": "1/2"},
        policy=NumericalPolicy(precision=30, order=10),
    )).value
    assert abs(point.numerators["zzb"][0][0].coeffs[0] - value) <= mp.mpf(10) ** -27 * abs(value)
    if c == "7/10":
        return
    table = build_table_request(request("table"), spin=2)
    for coordinate in coordinates:
        got = point.evaluate(coordinate, Fraction(7, 5))
        ref = table.evaluate(coordinate, Fraction(7, 5))
        for row, ref_row in zip(got, ref):
            for x, y in zip(row, ref_row):
                assert abs(x - y) <= mp.mpf(10) ** -(table.verified_digits - 2) * max(abs(y), mp.mpf(10) ** -12)


def test_new_point_table_capabilities_are_listed(capsys):
    for family, coordinates in (("ns", "ws"), ("ramond", "z"), ("torus-character", "vvb"),
                                ("torus-onepoint", "vvb")):
        capability = capabilities_for(family)[0]
        assert capability.supports_operation("point-table")
        assert coordinates in capability.coordinates_for("point-table")
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert out.count("operations: evaluate, table, point-table") == 6


def _load(path):
    with open(path) as fh:
        doc = json.load(fh)
    return doc, serialized_table_from_json(doc)


def test_cli_torus_character_point_table_round_trip(tmp_path):
    main(["point-table", "--surface", "torus", "--family", "torus-character", "--central-charge", "5/2",
          "--exchange-delta", "2", "--exchange-spin", "2", "--lambda", "6", "--precision", "30", "-o", str(tmp_path)])
    doc, table = _load(tmp_path / "derivs_point_2.json")
    assert (doc["surface"], doc["basis"], doc["variable"], doc["point_value"]) == ("torus", "torus_G", "point", "2.0")
    assert doc["regularization_policy"] == "irreducible" and doc["kac_limit"] == "irreducible"
    assert doc["spin"] == 2 and doc["spin_pair_symmetrized"] is True
    mp.mp.dps = 30
    current = torus_current_table(Fraction(5, 2), 2, 6)
    got = table.evaluate("vvb", 2)
    for m in range(7):
        for n in range(min(m, 6 - m) + 1):
            assert abs(got[m][n] - current[m][n]) <= mp.mpf(10) ** -27 * abs(current[m][n])


def test_cli_torus_onepoint_point_table_round_trip(tmp_path):
    main(["point-table", "--family", "torus-onepoint", "--torus-onept-ext", "1/5", "--central-charge", "5/2",
          "--exchange-delta", "17/10", "--lambda", "4", "--order", "12", "--precision", "30", "-o", str(tmp_path),
          "--output-template", "onept.json"])
    doc, table = _load(tmp_path / "onept.json")
    assert (doc["basis"], doc["variable"], doc["onept_d_ext"], doc["order"]) == ("torus_onept_G", "point", "0.2", 12)
    assert doc["regularization_policy"] == "direct" and "kac_limit" not in doc
    assert float(doc["verified_digits"]) > 25 and "q_truncation_digits" in doc
    mp.mp.dps = 30
    value, _ = _torus_value("torus-onepoint", "5/2", "17/10", 0, 0, 0, d_ext="1/5", order=12)
    assert abs(table.evaluate("vvb", Fraction(17, 10))[0][0] - value) <= mp.mpf(10) ** -25 * abs(value)


@pytest.mark.parametrize("family, args", [
    ("ns", ["--delta-ext", "1/5", "1/5", "1/5", "1/5", "--exchange-delta", "7/5", "--exchange-spin", "2",
            "--coordinates", "zzb", "ws"]),
    ("ramond", ["--h-ext", "1/10", "1/10", "1/10", "1/10", "--coordinates", "z", "--exchange-h", "3/5",
                "--ramond-branch", "+-"]),
])
def test_cli_super_point_table_round_trip(tmp_path, family, args):
    main(["point-table", "--family", family, "--central-charge", "13/10", "--lambda", "3", "--order", "10",
          "--precision", "30", "--q-gate", "off", "-o", str(tmp_path), "--output-template", "p.json",
          "--format", "both", *args])
    doc, table = _load(tmp_path / "p.json")
    assert doc["block_family"] == family and doc["variable"] == "point"
    assert doc["basis"] == (f"{family}_scalar_G" if family == "ns" else "ramond_chiral_F")
    if family == "ramond":
        assert doc["ramond_branch"] == "+-"
    assert list(tmp_path.glob("p_*.m"))
    mp.mp.dps = 30
    point = Fraction(7, 5) if family == "ns" else Fraction(3, 5)
    stored = table.evaluate(doc["coordinates"][0], point)[0][0]
    assert stored == table.numerators[doc["coordinates"][0]][0][0].coeffs[0]


def test_cli_torus_point_table_refuses_sphere_flags_and_half_integer_spins(tmp_path):
    base = ["point-table", "--family", "torus-character", "--central-charge", "5/2", "--exchange-delta", "2",
            "--lambda", "2", "-o", str(tmp_path)]
    for extra in (["--coordinates", "zzb"], ["--delta-ext", "1", "1", "1", "1"], ["--order", "10"],
                  ["--regularize", "on"], ["--exchange-spin", "1/2"], ["--torus-onept-ext", "1/5"],
                  ["--format", "m"]):
        with pytest.raises(SystemExit):
            main(base + extra)
    with pytest.raises(SystemExit):
        main(["point-table", "--surface", "sphere", "--family", "torus-onepoint", "--central-charge", "5/2",
              "--exchange-delta", "2", "--lambda", "2", "-o", str(tmp_path)])


def test_torus_character_delta_table_at_c_le_1_is_written_with_a_note(tmp_path, capsys):
    assert main(["table", "--surface", "torus", "--central-charge", "1/2", "--spin-ranges", "0", "--lambda", "3",
                 "--precision", "25", "-o", str(tmp_path)]) == 0
    assert "point-table --surface torus" in capsys.readouterr().err
    _doc, table = _load(tmp_path / "derivs_0.json")
    mp.mp.dps = 25
    got = table.evaluate("vvb", Fraction(3, 10))
    ref, _ = torus_character_point_values(Fraction(1, 2), Fraction(3, 10), 0, 3)
    for m in range(4):
        for n in range(min(m, 3 - m) + 1):
            assert abs(got[m][n] - ref[m][n]) <= mp.mpf(10) ** -20 * abs(ref[m][n])
