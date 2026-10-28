"""A table's verified digits cover the range SDPB uses it on, x = Delta - shift in [0, oo) (2026-09-29 CLI sweep).

The verification checked between the nodes and at 1.25 x the last one.  Past the nodes the monomial form
extrapolates its node values, and loses the most where the Delta-poles sit: a keep-all spinning table at
--precision 30 delivered 11 digits at x = 270 while recording 36.  The global radial slices were checked against a
sampler in the wrong variable (verified_digits -1.5 for a 28-digit table), and the order-reduced warnings quoted a
degree-based loss (115 digits, "raise --precision to 185") where the table lost 53, and the node range where the
table was good.  Each test builds a table through the CLI and checks it another way.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile

import mpmath as mp

from virasoro_blocks.__main__ import main
from virasoro_blocks.serialization import serialized_table_from_json


def _tables(args):
    with tempfile.TemporaryDirectory() as out:
        main(["table", *args, "--format", "json", "-o", out])
        return [json.load(open(path)) for path in sorted(glob.glob(os.path.join(out, "derivs_*.json")))]


def _digits(got, ref, kept):
    """Worst per-entry digits (entries more than 60 digits below the largest compared at that level)."""
    scale = max(abs(mp.mpc(ref[m][n])) for m, n in kept)
    floor = scale * mp.mpf(10) ** -60
    err = max(abs(mp.mpc(got[m][n]) - mp.mpc(ref[m][n])) / max(abs(mp.mpc(ref[m][n])), floor) for m, n in kept)
    return mp.inf if err == 0 else -mp.log10(err)


def test_keep_all_spinning_table_is_verified_past_its_nodes():
    """ordered_B spin 1 at --precision 30: the table recorded 35.9 verified digits (between the nodes) but was good
    to 11.4 at x = 270, where its poles sit, and 17 from x = 5000 on; the build now checks out to 16x the pole scale
    and (--target-digits 30) raises the working precision until it delivers 30 digits there too."""
    from virasoro_blocks.derivs import B_zzb_table
    from virasoro_blocks.truncation import kept_pairs

    (doc,) = _tables(["--central-charge", "3.3", "--lambda", "4", "--order", "24", "--precision", "30",
                      "--delta-ext", "0.15", "1.25", "0.35", "1.45", "--spin-ext", "0", "1", "0", "1",
                      "--spin-ranges", "1", "--coordinates", "zzb", "--target-digits", "30"])
    assert float(doc["verified_digits"]) >= 30
    h = [mp.mpf(v) for v in ("0.075", "1.125", "0.175", "1.225")]
    hbar = [mp.mpf(v) for v in ("0.075", "0.125", "0.175", "0.225")]
    kept = kept_pairs("zzb", 4, basis="ordered_B", ell=1)
    with mp.workdps(int(doc["precision_actual"]) + 40):
        table = serialized_table_from_json(doc)
        for x in ("270", "1000", "20000"):
            delta = mp.mpf(table.shift) + mp.mpf(x)
            ref = B_zzb_table((delta + 1) / 2, (delta - 1) / 2, h, hbar, 4, 4, c=mp.mpf("3.3"), q_order=24)
            assert _digits(table.evaluate("zzb", delta), ref, kept) > 30, x


def test_global_radial_slice_is_verified_in_its_own_variable():
    """A global xt_radial table is the n = 0 row of its folded xt parent; the point sampler built it as the
    non-folded zzb grid (d^m/dz^m, 2^m off), so a 28-digit table recorded -1.5 verified digits, and without
    --npoles the build escalated for nothing (npoles 30 -> 62, precision 80 -> 135)."""
    from virasoro_blocks.global_blocks import build_global_sampler

    h = [mp.mpf("0.1")] * 4
    common = dict(basis="global_scalar_G", ell=0, h_externals=h, hbar_externals=h, m_max=5)
    row = [(m, 0) for m in range(6)]
    (radial,), _ = build_global_sampler("xt_radial", n_max=0, **common).sample_bucket([mp.mpf("3.7")], row)
    (folded,), _ = build_global_sampler("xt", n_max=2, **common).sample_bucket([mp.mpf("3.7")], row)
    for m, _n in row:
        want = mp.mpc(folded[m][0])
        assert abs(mp.mpc(radial[m][0]) - want) <= mp.mpf(10) ** -40 * abs(want)       # (was off by 2^m)

    (doc,) = _tables(["--family", "global", "--basis", "global_scalar_G", "--lambda", "5", "--precision", "40",
                      "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--spin-ranges", "0", "--coordinates", "xt_radial",
                      "--npoles", "16"])
    assert float(doc["verified_digits"]) > 25


def test_order_reduced_c_below_one_warning_quotes_the_measured_loss(capsys):
    """c = 0.73, --npoles 14 keeps every pole on x > 0 (degree 153).  The warning said its coefficients lose ~115
    digits and to raise --precision to 185, and quoted 31.2 digits (the node range); the table loses ~53 digits
    (worst at x ~ 55) and delivers 9 over [0, oo).  At the precision the warning now recommends, it delivers the
    requested 60."""
    import re

    (doc,) = _tables(["--central-charge", "0.73", "--lambda", "5", "--order", "30", "--npoles", "14",
                      "--precision", "60", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--spin-ranges", "0",
                      "--coordinates", "zzb"])
    err = capsys.readouterr().err
    assert "warning:" in err and "cancel by ~" in err
    lost = int(re.search(r"cancel by ~(\d+) digits", err).group(1))
    advised = int(re.search(r"Raise --precision to about (\d+)", err).group(1))
    assert 40 <= lost <= 65 and advised < 130
    assert float(doc["verified_digits"]) < 20        # the far range, not the node range (31.2)
    (again,) = _tables(["--central-charge", "0.73", "--lambda", "5", "--order", "30", "--npoles", "14",
                        "--precision", str(advised), "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--spin-ranges", "0",
                        "--coordinates", "zzb"])
    assert float(again["verified_digits"]) >= 60


def test_order_reduced_near_c_one_reports_the_far_range(capsys):
    """c = 0.99999999: 22.9 digits was true on the node range [0, 2.6] only (0.3 digits at x = 25 by an independent
    recursion)."""
    (doc,) = _tables(["--central-charge", "0.99999999", "--lambda", "5", "--order", "30", "--npoles", "14",
                      "--precision", "60", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--spin-ranges", "0",
                      "--coordinates", "zzb"])
    assert float(doc["verified_digits"]) < 10
    assert "warning:" in capsys.readouterr().err


def test_order_reduced_table_off_past_its_nodes_near_c_25_is_a_warning(capsys):
    """c = 25 + 1e-12, --npoles 14: off by 1e7 from x ~ 100 on, it recorded 3.3 digits with a note."""
    (doc,) = _tables(["--central-charge", "25.000000000001", "--lambda", "5", "--order", "30", "--npoles", "14",
                      "--precision", "60", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--spin-ranges", "0",
                      "--coordinates", "zzb"])
    assert float(doc["verified_digits"]) < 0
    err = capsys.readouterr().err
    assert "warning:" in err and "does not reproduce the block" in err and "note:" not in err.split("warning:")[-1]


def test_fit_rounding_is_blamed_only_when_it_shows_between_the_nodes(monkeypatch):
    """An order-reduced table whose node fit lost most of the working digits is precision-limited only if it is
    about as good between its nodes as it fits them: there the interpolation is well conditioned, so the fit's
    rounding shows at its node level, while an order-reduction error does not vanish between the nodes."""
    from types import SimpleNamespace

    import virasoro_blocks.verified_build as vb

    table = SimpleNamespace(numerators={}, sample_points_x=[1])

    def diagnose(delivered, between, fit, work):
        monkeypatch.setattr(vb, "precision_limit_digits", lambda *a, **k: (work - 1.0, 4.5))
        monkeypatch.setattr(vb, "node_residual_digits", lambda *a, **k: fit)
        verification = vb.Verification(digits=delivered, node_digits=between, points=(1,))
        return vb.diagnose_monomial(table, verification, digits=work, work=work, order_reduced=True)

    # Lambda 19, c = 1.00001, Delta_ext 0.4, spin 0 at npoles 84: 113 digits at the nodes, -1.9 between them (and
    # the same at twice the precision): the kept poles limit it
    assert diagnose(-1.9, -1.9, 113.3, 240).cause == "npoles"
    # Lambda 5, npoles 100 at dps 80: 9.40 at the nodes, 9.41 between them, 4.5 over the range (30.9 at dps 160)
    found = diagnose(4.5, 9.41, 9.40, 80)
    assert (found.cause, found.mechanism) == ("precision", "fit")
