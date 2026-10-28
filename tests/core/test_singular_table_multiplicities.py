"""Keep-all and order-reduced tables at minimal-model and c = -2 central charges, at the precisions that failed.

The c < 1 tables' pole multiplicities were measured on the sampler, and the measurement read the leading order
only where its coefficient stood above the lower orders' at the probe offsets: Ising sigma (lambda 3, q_order 30)
failed at --precision 40 and 60 and built at 100, and every --npoles table stopped at 26-27 digits.  They are now
the recursion's exact Laurent orders (``regularization.chiral_kac_pole_orders``).  c = -2 also needed the t-window
of the principal parts split off at x > 0 widened (they came back at 1e-40).  Each table is checked against its
own sampler by the build (``verified_digits``) and, for Ising, against the exact identity and epsilon blocks.

--precision is the working precision (2026-09-29): a keep-all table of degree ~80-120 built at it loses ~25-46 digits
to the monomial form's rounding and says so, with the precision that carries the digits; the keep-all tests ask for
the digits with --target-digits, and check that the advised precision delivers them.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile

import re

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main
from virasoro_blocks.serialization import serialized_table_from_json


def _table(*args):
    with tempfile.TemporaryDirectory() as out:
        main(["table", "--family", "virasoro", "--basis", "scalar_G", "--coordinates", "zzb", "--spin-ranges", "0",
              *args, "--format", "json", "-o", out])
        (path,) = glob.glob(os.path.join(out, "derivs_*.json"))
        return json.load(open(path))


def _advised(err):
    """The --precision a shortfall message advises."""
    return int(re.search(r"Raise --precision to about (\d+)", err).group(1))


def _ising_digits(doc):
    """Worst digits of the table against the exact Ising sigma blocks at Delta = 0 (identity) and 1 (epsilon)."""
    with mp.workdps(int(doc["precision_actual"]) + 20):
        table = serialized_table_from_json(doc)

        def identity(z):
            s = mp.sqrt(1 - z)
            return mp.sqrt((1 + s) / 2) * (1 - z) ** (-mp.mpf(1) / 8)

        def epsilon(z):
            s = mp.sqrt(1 - z)
            return mp.sqrt(2 * (1 - s)) / 4 * (1 - z) ** (-mp.mpf(1) / 8)

        worst = mp.inf
        for delta, block in ((0, identity), (1, epsilon)):
            c = mp.taylor(block, mp.mpf(1) / 2, 3)
            got = table.evaluate("zzb", mp.mpf(delta))
            for m in range(4):
                for n in range(m + 1):
                    if m + n <= 3:
                        ref = c[m] * c[n]  # the spin-0 scalar_G block is F(z) F(zbar)
                        worst = min(worst, -mp.log10(abs(mp.mpc(got[m][n]) / ref - 1)))
        return worst


_ISING = ("--central-charge", "1/2", "--delta-ext", "1/8", "1/8", "1/8", "1/8", "--lambda", "3", "--order", "30")


@pytest.mark.parametrize("precision", [30, 40, 60])
def test_ising_keep_all_table_builds_at_every_precision(precision):
    doc = _table(*_ISING, "--precision", str(precision), "--target-digits", str(precision))
    assert float(doc["verified_digits"]) >= precision and int(doc["precision_actual"]) > precision
    # q_order 30 truncates the exact blocks at ~37.5 digits
    assert _ising_digits(doc) > 37


def test_ising_keep_all_table_at_the_working_precision_names_the_precision_that_carries_it(capsys):
    """At --precision 40 alone the degree-83 table delivers ~14 digits and says so; the precision it advises
    delivers 40."""
    doc = _table(*_ISING, "--precision", "40")
    assert doc["precision_input"] == 40 and 40 <= int(doc["precision_actual"]) <= 50
    assert float(doc["verified_digits"]) < 30
    advised = _advised(capsys.readouterr().err)
    doc = _table(*_ISING, "--precision", str(advised))
    assert float(doc["verified_digits"]) >= 40 and _ising_digits(doc) > 37


def test_ising_order_reduced_table_is_not_capped(capsys):
    """Not capped at 26-27 digits: 98.8 between its nodes, 84.4 over the whole range since the check covers the
    large-Delta tail (2026-09-29), limited there by its coefficients' cancellation at x ~ 34 (the kept Kac poles on
    x > 0 hold the degree at 83); the precision the note advises delivers --precision 110."""
    doc = _table(*_ISING, "--precision", "110", "--npoles", "20")
    assert float(doc["verified_digits"]) > 80
    assert _ising_digits(doc) > 37
    err = capsys.readouterr().err
    assert "cancel by ~" in err
    doc = _table(*_ISING, "--precision", str(_advised(err)), "--npoles", "20")
    assert float(doc["verified_digits"]) >= 110


def test_c_minus_two_keep_all_table_builds():
    doc = _table("--central-charge", "-2", "--delta-ext", "1/8", "1/8", "1/8", "1/8", "--lambda", "3",
                 "--order", "30", "--precision", "60", "--target-digits", "60")
    assert float(doc["verified_digits"]) >= 60 and int(doc["precision_actual"]) > 60
