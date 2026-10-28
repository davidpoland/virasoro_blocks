"""CLI tables deliver --precision digits, checked against independent references (2026-09-27 sweep).

A table's coefficients are a degree-D monomial form that cancels on evaluation, the minimal-model pole
multiplicities were a count that assumed simple poles, global and NS/R tables defaulted to 8 kept poles, and the
numeric recursion lost digits at large c -- none of it visible in the output.  The tables are now built at a
working precision that carries them, verified against their exact sampler (``verified_digits``), and every
default keeps what the table needs.  Each test compares the written JSON with a reference computed another way.

--precision is the working precision (2026-09-29): a keep-all table is built at it and reports its shortfall; the
tests that check a table delivers P digits ask for them with --target-digits P, which escalates the working
precision.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main
from virasoro_blocks.serialization import serialized_table_from_json


def _table(args):
    with tempfile.TemporaryDirectory() as out:
        main([*args, "--format", "json", "-o", out])
        return [json.load(open(path)) for path in sorted(glob.glob(os.path.join(out, "derivs_*.json")))]


def _kept(lam):
    return [(m, n) for m in range(lam + 1) for n in range(lam + 1) if n <= m and m + n <= lam]


def _worst(table_grid, ref_grid, lam):
    return max(abs(mp.mpc(table_grid[m][n]) - mp.mpc(ref_grid[m][n])) / abs(mp.mpc(ref_grid[m][n]))
               for m, n in _kept(lam) if ref_grid[m][n] != 0)


def test_low_precision_table_is_built_at_a_working_precision_that_delivers_it():
    """--precision 40 (lambda 2, q_order 16): the degree-66 interpolant was off by up to 5e-2; it is now built at a
    higher working precision, written with those digits, and matches the direct recursion to 40 digits."""
    from virasoro_blocks.derivs import G_zzb_table

    (doc,) = _table(["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", "3.3",
                     "--spin-ranges", "0", "--delta-ext", "0.26", "0.26", "0.26", "0.26", "--coordinates", "zzb",
                     "--lambda", "2", "--order", "16", "--precision", "40", "--target-digits", "40"])
    assert doc["precision_input"] == 40 and doc["precision_actual"] > 40
    assert float(doc["verified_digits"]) >= 40
    with mp.workdps(200):
        table = serialized_table_from_json(doc)
        for delta in ("1.31", "5.9", "25.1"):
            # against the decimal inputs themselves: the CLI converts them with guard digits
            ref = G_zzb_table(mp.mpf(delta), 0, [mp.mpf("0.13")] * 4, 2, 2, c=mp.mpf("3.3"), q_order=16)
            assert _worst(table.evaluate("zzb", mp.mpf(delta)), ref, 2) < mp.mpf("1e-40")


@pytest.mark.parametrize("c", [Fraction(1, 2), Fraction(-22, 5)])
def test_minimal_model_tables_have_the_right_pole_multiplicities(c):
    """Ising and Lee-Yang CLI tables were good to 7 and 2 digits (the skeleton dropped genuine double-pole copies);
    they now match the block at c + 1e-100 (bare recursion at 300 digits) to the requested precision.  One spin (2:
    spinning blocks carry the same-side Kac-pair collisions, i.e. genuine double poles, at c = 1/2) at a small and a
    large Delta."""
    from virasoro_blocks.derivs import G_zzb_table
    from virasoro_blocks.singularity import bare_recursion_probes

    docs = _table(["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", str(float(c)),
                   "--spin-ranges", "2", "--delta-ext", "0.26", "0.26", "0.26", "0.26", "--coordinates", "zzb",
                   "--lambda", "2", "--order", "10", "--precision", "50", "--target-digits", "50"])
    assert len(docs) == 1
    # 300 digits carry c + 1e-100 through the double-pole cancellation (c + 1e-150 needs 500)
    with mp.workdps(300):
        for doc in docs:
            table, ell = serialized_table_from_json(doc), int(float(doc["spin"]))
            for x in ("0.31", "7.9"):
                delta = mp.mpf(table.shift) + mp.mpf(x)
                with bare_recursion_probes():
                    ref = G_zzb_table(delta, ell, [mp.mpf("0.13")] * 4, 2, 2,
                                      c=mp.mpf(c.numerator) / c.denominator + mp.mpf(10) ** -100, q_order=10)
                assert _worst(table.evaluate("zzb", delta), ref, 2) < mp.mpf("1e-49")


def test_order_reduced_table_is_built_once_with_its_verified_digits(capsys):
    """An explicit --npoles is an approximation: it is not escalated or refused, and records its digits."""
    (doc,) = _table(["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", "10",
                     "--spin-ranges", "0", "--delta-ext", "0.15", "0.15", "0.15", "0.15", "--coordinates", "zzb",
                     "--lambda", "2", "--order", "10", "--npoles", "4", "--precision", "30"])
    assert float(doc["verified_digits"]) < 30
    assert "order-reduced table" in capsys.readouterr().err


def test_global_table_without_npoles_reaches_the_requested_precision():
    """Global tables defaulted to 8 kept poles (13 digits); without --npoles they now keep enough to deliver it."""
    from virasoro_blocks.point import point_block_table

    (doc,) = _table(["table", "--family", "global", "--basis", "global_scalar_G", "--spin-ranges", "0",
                     "--delta-ext", "0.26", "0.26", "0.26", "0.26", "--coordinates", "zzb", "--lambda", "2",
                     "--precision", "40", "--target-digits", "40"])
    assert float(doc["verified_digits"]) >= 40
    h_parsed = mp.mpf("0.13")
    with mp.workdps(120):
        table = serialized_table_from_json(doc)
        for x in ("0.31", "2.3"):
            delta = mp.mpf(table.shift) + mp.mpf(x)
            point = point_block_table(["zzb"], basis="global_scalar_G", ell=0, point_value=delta,
                                      h_externals=[h_parsed] * 4, hbar_externals=[h_parsed] * 4, lam=2,
                                      c=None, q_order=int(doc["order"]), chiral_norm=None, z_shift=None,
                                      zbar_shift=None, center=None, center_bar=None, parity=None,
                                      parity_external=None, exchanged_parity="1", regularize="auto")
            assert _worst(table.evaluate("zzb", delta), point.evaluate("zzb", delta), 2) < mp.mpf("1e-40")


def test_ns_table_keeps_every_pole_by_default():
    """NS tables defaulted to 8 poles (45 of 80 digits); they keep the whole pool and match numeric_full_table."""
    from virasoro_blocks.superconformal_blocks import numeric_full_table

    (doc,) = _table(["table", "--family", "ns", "--basis", "ns_scalar_G", "--super-sector", "even",
                     "--central-charge", "1.2", "--spin-ranges", "0", "--delta-ext", "0.25", "0.25", "0.25", "0.25",
                     "--coordinates", "zzb", "--lambda", "2", "--order", "6", "--precision", "40",
                     "--target-digits", "40"])
    assert int(doc["npoles"]) > 8 and float(doc["verified_digits"]) >= 40
    with mp.workdps(200):
        table = serialized_table_from_json(doc)
        for x in ("0.31", "2.3", "7.9"):
            delta = mp.mpf(table.shift) + mp.mpf(x)
            ref = numeric_full_table("ns", "ns_scalar_G", delta, 0, "zzb", [mp.mpf("0.125")] * 4,
                                     c=mp.mpf("1.2"), q_order=6, m_max=2, n_max=2)
            assert _worst(table.evaluate("zzb", delta), ref, 2) < mp.mpf("1e-40")


def test_numeric_tables_keep_their_digits_at_large_c():
    """The numeric recursion lost 42 of 60 digits at c = 1e4, q_order 30 (residues grow with c and cancel, and b's
    rounding alone cost 9 digits at c = 1e5); a verified guard and a b formed with extra digits keep them.  Checked at
    c = 1e5, where both effects are largest."""
    from virasoro_blocks.derivs import G_zzb_table

    h = [mp.mpf("0.13")] * 4
    for c in ("100000",):
        with mp.workdps(400):
            ref = G_zzb_table(mp.mpf("2.3"), 0, h, 3, 3, c=mp.mpf(c), q_order=30)
        with mp.workdps(60):
            got = G_zzb_table(mp.mpf("2.3"), 0, h, 3, 3, c=mp.mpf(c), q_order=30)
        assert max(abs(got[m][n] - ref[m][n]) / abs(ref[m][n]) for m in range(4) for n in range(4)
                   if ref[m][n] != 0) < mp.mpf("1e-58")


def test_identity_kac_root_is_exactly_zero_at_any_precision():
    """d_{1,1} vanishes for every c, but the imaginary-b path assembled it as -2e-217 at dps 171, which node
    placement rightly refuses as an unremoved Kac zero; the exported root is exactly 0."""
    from virasoro_blocks.params import b_from_c
    from virasoro_blocks.polyderivs import F_z_poly_table

    with mp.workdps(171):
        table = F_z_poly_table([mp.mpf("0.05"), mp.mpf("0.10"), mp.mpf("0.15"), mp.mpf("0.20")], 4,
                               b=b_from_c(mp.mpf("0.73")), q_order=16, skeleton=True)
        roots = [r for r, prov in zip(table.denominator_roots, table.root_provenance) if prov[1] == ((1, 1),)]
    assert roots and all(r == 0 for r in roots)


def test_collision_cluster_scan_survives_distances_below_float_range():
    """The cluster cut compared float distances; at working precisions above ~300 digits they underflowed to 0."""
    from virasoro_blocks.regularization import _collide_cluster

    with mp.workdps(600):
        poles = [mp.mpc(mp.mpf("0.125") + mp.mpf(10) ** -400), mp.mpc(mp.mpf("0.125") - mp.mpf(10) ** -401),
                 mp.mpc("0.9"), mp.mpc("1.7")]
        assert len(_collide_cluster(poles, mp.mpc("0.125"))) == 2


def test_writer_coordinate_metadata_is_tied_to_the_input_precision():
    """A 0.37 center must not read back as 0.3700...0002066 at table precision."""
    (doc,) = _table(["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", "10",
                     "--spin-ranges", "0", "--delta-ext", "0.15", "0.15", "0.15", "0.15", "--coordinates", "zzb",
                     "--lambda", "2", "--order", "6", "--precision", "30", "--z-center", "0.37",
                     "--zbar-center", "0.41"])
    assert doc["z_center"] == "0.37" and doc["zbar_center"] == "0.41"


def test_order_reduced_table_is_built_at_the_requested_precision(monkeypatch):
    """An explicit --npoles table is the code's main product: it is built at --precision as given (the automatic
    guard for a full table's degree tripled its working precision for no accuracy, 2026-09-27)."""
    from virasoro_blocks import table_backends

    build_precisions = []
    original = table_backends._process_spin_once

    def record_build(*args, **kwargs):
        build_precisions.append(mp.mp.dps)
        return original(*args, **kwargs)

    monkeypatch.setattr(table_backends, '_process_spin_once', record_build)
    (doc,) = _table(["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", "3.3",
                     "--spin-ranges", "0", "--delta-ext", "0.26", "0.26", "0.26", "0.26", "--coordinates", "zzb",
                     "--lambda", "4", "--order", "16", "--npoles", "10", "--precision", "40"])
    assert build_precisions == [40]
    # Output retains source guard digits in coefficients and saved samples;
    # its header is not the precision at which the build started.
    assert doc["precision_input"] == 40 and 40 <= doc["precision_actual"] <= 50
    assert int(doc["npoles"]) == 10


def test_json_table_evaluates_at_its_working_precision_from_any_caller_precision():
    """A 40-digit table evaluated at the default 15 digits canceled its monomial coefficients to 2e-4; the reader
    parses at precision_actual and the table evaluates there, rounding to the caller's precision."""
    from virasoro_blocks.derivs import G_zzb_table

    (doc,) = _table(["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", "3.3",
                     "--spin-ranges", "0", "--delta-ext", "0.26", "0.26", "0.26", "0.26", "--coordinates", "zzb",
                     "--lambda", "2", "--order", "16", "--precision", "40", "--target-digits", "40"])
    with mp.workdps(15):
        got = serialized_table_from_json(doc).evaluate("zzb", mp.mpf("5.9"))
    with mp.workdps(60):
        ref = G_zzb_table(mp.mpf("5.9"), 0, [mp.mpf("0.13")] * 4, 2, 2, c=mp.mpf("3.3"), q_order=16)
        assert _worst(got, ref, 2) < mp.mpf("1e-14")


def test_spinning_global_table_raises_precision_when_its_degree_needs_it():
    """A spinning global pool carries double poles (two per order), so the kept whole pool has twice the degree the
    pole count suggests; the build raised poles but never precision and levelled off at 71 of 120 digits."""
    (doc,) = _table(["table", "--family", "global", "--basis", "global_ordered_B", "--spin-ranges", "1",
                     "--delta-ext", "0.15", "1.25", "0.35", "1.45", "--spin-ext", "0", "1", "0", "1",
                     "--coordinates", "zzb", "--lambda", "2", "--precision", "40", "--target-digits", "40"])
    assert float(doc["verified_digits"]) >= 40


@pytest.mark.slow
def test_ns_table_at_an_n1_minimal_model_c_matches_the_c_limit():
    """At c = 7/10 the recursion runs at c + eps; eps was fixed at 1e-30, so every table and evaluation there was
    good to ~28 digits whatever --precision said (the verification, at the same eps, could not see it).  eps now
    sits below the requested digits, and the table matches the c -> 7/10 limit (eps 1e-150 at 300 digits)."""
    from virasoro_blocks.superconformal_blocks import numeric_full_table

    (doc,) = _table(["table", "--family", "ns", "--basis", "ns_scalar_G", "--super-sector", "odd",
                     "--central-charge", "0.7", "--spin-ranges", "0", "--delta-ext", "0.25", "0.25", "0.25", "0.25",
                     "--coordinates", "zzb", "--lambda", "2", "--order", "8", "--precision", "40",
                     "--target-digits", "40"])
    table = serialized_table_from_json(doc)
    with mp.workdps(300):
        for x in ("0.31", "2.3"):
            ref = numeric_full_table("ns", "ns_scalar_G", mp.mpf(x), 0, "zzb", [mp.mpf("0.125")] * 4,
                                     c=mp.mpf(7) / 10, q_order=8, m_max=2, n_max=2, sector="odd", eps_exp=150)
            assert _worst(table.evaluate("zzb", mp.mpf(x)), ref, 2) < mp.mpf("1e-40")


def test_c_below_one_order_reduced_table_matches_the_block_at_adequate_precision(capsys):
    """At c < 1 the Kac poles on x > 0 are always kept, so an order-reduced table keeps near-full degree and its
    monomial coefficients cancel ~0.75 digits per unit of degree: at a low --precision it does not reproduce the
    block and the CLI says so; at a precision that carries its degree it matches point-table (Ising, lambda 4)."""
    from virasoro_blocks.point import point_block_table

    args = ["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", "0.5", "--spin-ranges", "0",
            "--delta-ext", "0.26", "0.26", "0.26", "0.26", "--coordinates", "zzb", "--lambda", "4", "--order", "12",
            "--npoles", "8"]
    _table([*args, "--precision", "30"])
    assert "Raise --precision to about" in capsys.readouterr().err
    (doc,) = _table([*args, "--precision", "80"])
    table = serialized_table_from_json(doc)
    h = mp.mpf("0.13")
    with mp.workdps(120):
        for delta in ("1.37", "3.61"):
            point = point_block_table(["zzb"], basis="scalar_G", ell=0, point_value=mp.mpf(delta),
                                      h_externals=[h] * 4, hbar_externals=[h] * 4, lam=4, c=mp.mpf(1) / 2,
                                      q_order=12, chiral_norm=None, z_shift=None, zbar_shift=None, center=None,
                                      center_bar=None, parity=None, parity_external=None, exchanged_parity="1",
                                      regularize="auto")
            assert _worst(table.evaluate("zzb", mp.mpf(delta)), point.evaluate("zzb", mp.mpf(delta)), 4) < mp.mpf("1e-15")
