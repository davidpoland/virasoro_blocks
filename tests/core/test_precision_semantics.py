"""--precision sets working digits; --target-digits requests verified accuracy.

Tables retain source guard digits when written and report measured
shortfalls. Target-driven builds increase precision by the measured
deficit, or explain why the requested accuracy cannot be reached.
"""

from __future__ import annotations

import glob
import json
import os
import re
import tempfile

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main
from virasoro_blocks.verified_build import (
    PrecisionRequest, TargetDigitsError, Verification, default_max_precision, point_agreement, run_verified_build,
)

_KEEP_ALL = ["table", "--central-charge", "10", "--spin-ranges", "0", "--lambda", "2", "--delta-ext", "0.15", "0.15",
             "0.15", "0.15", "--order", "10", "--format", "json"]


def _run(args):
    with tempfile.TemporaryDirectory() as out:
        code = main([*args, "-o", out])
        docs = [json.load(open(path)) for path in sorted(glob.glob(os.path.join(out, "derivs_*.json")))]
    return code, docs


def test_precision_is_the_working_precision_without_a_target(capsys):
    code, (doc,) = _run([*_KEEP_ALL, "--precision", "30"])
    assert code in (0, None)
    assert doc["precision_input"] == 30 and 30 <= doc["precision_actual"] <= 40
    assert "target_digits" not in doc
    assert float(doc["verified_digits"]) < 27 and "verified_digits_largest_entry" in doc
    lo, hi = (float(x) for x in doc["verified_delta_range"])
    assert hi - lo > 100          # past the nodes, to the large-Delta regime
    err = capsys.readouterr().err
    assert err.count("spin 0: the table") == 1      # one message


def test_the_suggested_precision_delivers_the_digits(capsys):
    _run([*_KEEP_ALL, "--precision", "30"])
    advised = int(re.search(r"Raise --precision to about (\d+)", capsys.readouterr().err).group(1))
    assert advised > 30
    _code, (doc,) = _run([*_KEEP_ALL, "--precision", str(advised)])
    assert doc["precision_input"] == advised and advised <= int(doc["precision_actual"]) <= advised + 10
    assert float(doc["verified_digits"]) >= 30


def test_target_digits_escalates_and_records_it(capsys):
    _code, (doc,) = _run([*_KEEP_ALL, "--precision", "30", "--target-digits", "30"])
    assert doc["precision_input"] == 30 and doc["precision_actual"] > 30
    assert float(doc["verified_digits"]) >= 30 and doc["target_digits"] == 30
    assert "was built at working precision" in capsys.readouterr().err


def test_target_digits_stops_at_the_cap(capsys):
    code, docs = _run([*_KEEP_ALL, "--precision", "30", "--target-digits", "30", "--max-precision", "36"])
    assert code == 2 and not docs
    err = capsys.readouterr().err
    assert "cannot deliver --target-digits 30" in err and "above --max-precision 36" in err


def test_target_digits_refuses_an_order_reduction_limit(capsys):
    code, docs = _run([*_KEEP_ALL, "--precision", "30", "--npoles", "4", "--target-digits", "30"])
    assert code == 2 and not docs
    err = capsys.readouterr().err
    assert "the order reduction (its kept poles) limits it" in err and "raise --npoles" in err


def test_target_digits_stall_means_precision_is_not_the_limit():
    """An escalation that closes less than a quarter of the gap stops: precision is not what limits the table."""
    request = PrecisionRequest.make(30, 30)
    builds = []

    def build(work):
        builds.append(work)
        return work

    def verify(_built, _work):
        return Verification(digits=12.0 + 0.1 * len(builds), largest_digits=14.0, where_text="here")

    def diagnose(_built, v, work):
        from dataclasses import replace

        return replace(v, cause="precision", mechanism="rounding", lost=work - v.digits, needed_precision=60, degree=9)

    with pytest.raises(TargetDigitsError, match="less than a quarter of the way"):
        run_verified_build(build, verify, request=request, what="t", order_reduced=False, diagnose=diagnose)
    assert builds[0] == 30 and len(builds) == 2       # (the first step went to the measured need)
    assert builds[1] == 60


@pytest.mark.parametrize("extra, message", [
    (["--max-precision", "90"], "caps the escalation that --target-digits asks for"),
    (["--target-digits", "50", "--max-precision", "40"], "exceeds --max-precision 40"),
    (["--target-digits", "20", "--max-precision", "25"], "is below --precision 30"),
])
def test_precision_flags_are_honoured_or_refused(extra, message, capsys):
    with pytest.raises(SystemExit):
        main([*_KEEP_ALL, "--precision", "30", *extra, "-o", tempfile.gettempdir()])
    assert message in capsys.readouterr().err


def test_default_cap():
    assert default_max_precision(60, 60) == 260 and default_max_precision(30, 100) == 400
    assert PrecisionRequest.make(80, 40).cap == 240 and PrecisionRequest.make(80).cap == 80


def test_precision_controls_reject_numpy_booleans():
    import numpy as np
    from virasoro_blocks.api import NumericalPolicy

    for value in (np.bool_(True), np.bool_(False)):
        for options, field in (((value, None, None), 'precision'),
                               ((40, value, None), 'target-digits'),
                               ((40, 20, value), 'max-precision')):
            with pytest.raises(ValueError, match=field):
                PrecisionRequest.make(*options)
        with pytest.raises(ValueError, match='target_digits'):
            NumericalPolicy(target_digits=value)


def test_largest_entry_measure_decides_whether_a_table_reproduces_the_block(capsys):
    """c = 8, lambda 11, xt, --npoles 40: the folded subleading entries lose relative accuracy past the nodes (0.2
    digits per entry) while the table holds ~16 digits against its largest entry: a note, not 'does not
    reproduce'."""
    with mp.workdps(40):
        got = [[mp.mpf(1), mp.mpf("1.5e-20")]]
        ref = [[mp.mpf(1) + mp.mpf("1e-17"), mp.mpf("1e-20")]]
        per_entry, largest = point_agreement(got, ref, [(0, 0), (0, 1)], 30)
    assert per_entry < 1 and 16.5 < largest < 17.5

    _code, (doc,) = _run(["table", "--central-charge", "8", "--lambda", "11", "--order", "40", "--precision", "80",
                          "--npoles", "40", "--delta-ext", "0.125", "0.125", "0.125", "0.125", "--spin-ranges", "0",
                          "--coordinates", "xt", "--format", "json"])
    assert float(doc["verified_digits"]) < 3 < 10 < float(doc["verified_digits_largest_entry"])
    err = capsys.readouterr().err
    assert "note: spin 0: the order-reduced table" in err and "does not reproduce" not in err


@pytest.mark.parametrize("family", ["virasoro", "global"])
def test_evaluate_target_digits_for_every_family(family, capsys):
    """The CLI honors --target-digits wherever the Python API does (it once refused it off torus-onepoint)."""
    from virasoro_blocks.cli.evaluate import main as evaluate_main

    central = [] if family == "global" else ["--central-charge", "8"]
    assert evaluate_main(["--family", family, *central, "--delta-ext", "0.2", "0.2", "0.2", "0.2",
                          "--exchange-delta", "3", "--point", "z=0.4", "--point", "zbar=0.4", "--precision", "20",
                          "--target-digits", "30"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["precision"] == 20 and doc["precision_actual"] >= 30 and doc["estimated_digits"] >= 30


def test_point_and_character_tables_are_verified_at_the_working_precision(tmp_path):
    from virasoro_blocks.cli.point_table import main as point_table_main

    assert point_table_main(["--central-charge", "8", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--exchange-delta",
                             "3", "--lambda", "2", "--order", "12", "--precision", "30", "-o", str(tmp_path)]) == 0
    doc = json.load(open(tmp_path / "derivs_point_0.json"))
    assert doc["precision_input"] == 30 and doc["precision_actual"] >= 30
    assert float(doc["verified_digits"]) > 20
    assert main(["table", "--surface", "torus", "--central-charge", "8", "--lambda", "3", "--precision", "25",
                 "--spin-ranges", "0", "-o", str(tmp_path / "t")]) == 0
    doc = json.load(open(tmp_path / "t" / "derivs_0.json"))
    assert doc["precision_actual"] == 25 and float(doc["verified_digits"]) > 20
