"""CLI flags take effect or are rejected before building a block.

Cover family-specific controls, sampling options, q-gates, chiral shifts,
Ramond sign branches, and point-table and large-spin metadata.
"""

from __future__ import annotations

import json
import os

import pytest

from virasoro_blocks.__main__ import main
from virasoro_blocks.cli import join_option_values
from virasoro_blocks.cli.evaluate import main as evaluate_main
from virasoro_blocks.cli.point_table import main as point_table_main


_DELTA = ["--delta-ext", "0.2", "0.2", "0.2", "0.2"]
_VIR = ["table", "--central-charge", "8", "--order", "8", "--lambda", "2", "--precision", "30", *_DELTA,
        "--q-gate", "off"]
_CHIRAL = ["table", "--central-charge", "8", "--order", "8", "--lambda", "2", "--precision", "30",
           "--h-ext", "0.1", "0.1", "0.1", "0.1", "--coordinates", "z", "--q-gate", "off"]
_GLOBAL = ["table", "--family", "global", "--basis", "scalar_G", "--order", "8", "--lambda", "2",
           "--precision", "30", *_DELTA, "--npoles", "6"]
_NS = ["table", "--family", "ns", "--central-charge", "1.3", "--order", "6", "--lambda", "2", "--precision", "30",
       *_DELTA, "--q-gate", "off"]
_RAMOND = ["table", "--family", "ramond", "--central-charge", "1.3", "--order", "6", "--lambda", "1",
           "--precision", "20", *_DELTA, "--q-gate", "off", "--npoles", "2"]
_TOWER = ["table", "--central-charge", "8", "--order", "8", "--lambda", "1", "--precision", "20", *_DELTA,
          "--large-spin-lmin", "10", "--large-spin-twists", "0.4", "--npoles", "3", "--q-gate", "off"]


def _refused(argv, tmp_path, capsys, needle):
    with pytest.raises(SystemExit) as exc:
        main(argv + ["-o", str(tmp_path)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert needle in err, err
    assert not os.listdir(tmp_path)


@pytest.mark.parametrize("base,extra,needle", [
    # an even scalar_G tower's L = 0 member is the doubled 2 B(h, h) under a file tagged scalar_G_spin0 "single"
    ([*_VIR, "--large-spin-lmin", "0", "--large-spin-twists", "0.4", "--npoles", "3"], [],
     "--large-spin-lmin 0 starts the scalar_G tower"),
    ([*_GLOBAL, "--large-spin-lmin", "0/1", "--large-spin-twists", "0.4"], [],
     "--large-spin-lmin 0 starts the scalar_G tower"),
    # the N=1 choices were dropped by every other family (and --ramond-branch by NS)
    (_VIR, ["--super-sector", "odd"], "--super-sector is for the N=1"),
    (_VIR, ["--super-sector", "even"], "--super-sector is for the N=1"),
    (_VIR, ["--ramond-branch", "+-"], "--ramond-branch is for the N=1"),
    (_VIR, ["--ramond-branch", "--"], "--ramond-branch is for the N=1"),
    (_GLOBAL, ["--super-sector", "odd"], "the global blocks have no sector"),
    (_TOWER, ["--ramond-branch=-+"], "--ramond-branch is for the N=1"),
    (_NS, ["--ramond-branch", "+-"], "the NS blocks have no three-point sign branch"),
    # --pole-selection with every pole kept (no --npoles, or --npoles all)
    (_VIR, ["--pole-selection", "nearest"], "--pole-selection chooses the poles"),
    (_VIR, ["--npoles", "all", "--pole-selection", "nearest"], "--npoles all keeps every pole"),
    (_CHIRAL, ["--pole-selection", "nearest"], "--pole-selection chooses the poles"),
    (_NS, ["--pole-selection", "nearest"], "--pole-selection chooses the poles"),
    # nothing is sampled
    (_VIR, ["--npoles", "all", "--shared-sample-points"], "not used with --npoles all"),
    (_VIR, ["--npoles", "all", "--sample-via", "symbolic"], "not used with --npoles all"),
    (_GLOBAL, ["--sample-via", "symbolic"], "--sample-via is not used by the global blocks"),
    (_TOWER, ["--sample-via", "numeric"], "--sample-via is not used by the large-spin tower"),
    # the global blocks have no q-series
    (_GLOBAL, ["--q-gate", "strict"], "--q-gate strict has nothing to check"),
    # a chiral table has no exchanged spin and no zbar half
    (_CHIRAL, ["--spin-ranges", "2"], "--spin-ranges does not apply"),
    (_CHIRAL, ["--zbar-shift", "0.3"], "--zbar-shift is not valid"),
    (_CHIRAL, ["--z-center", "0.4", "--zbar-center", "0.3"], "--zbar-center is not valid"),
])
def test_inapplicable_flag_is_refused(base, extra, needle, tmp_path, capsys):
    _refused(base + extra, tmp_path, capsys, needle)


@pytest.mark.parametrize("prefix", [[], ["table"]])
@pytest.mark.parametrize("option", [["--npoles", "all"], ["--npoles=all"]])
def test_torus_onepoint_refuses_all_poles_before_creating_output(prefix, option, tmp_path, capsys):
    output = tmp_path / "blocks"
    with pytest.raises(SystemExit) as exc:
        main(prefix + ["--surface", "torus", "--family", "torus-onepoint",
             "--central-charge", "2.5", "--torus-onept-ext", ".2",
             "--lambda", "0", "--order", "3", "--precision", "25",
             "--q-gate", "off", *option, "-o", str(output)])
    assert exc.value.code == 2
    assert "--npoles all is not supported" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("option", [["--npoles", "all"], ["--npoles=all"]])
def test_torus_character_refuses_all_poles(option, tmp_path, capsys):
    _refused(["table", "--surface", "torus", "--central-charge", "2.5",
              "--lambda", "0", *option], tmp_path, capsys, "--npoles")


@pytest.mark.parametrize("onepoint", [False, True])
@pytest.mark.parametrize("option", [["--coordinates", "zzb"], ["--coordinates=zzb"],
                                     ["--basis", "scalar_G"]])
def test_torus_refuses_explicit_sphere_defaults(onepoint, option, tmp_path, capsys):
    argv = ["table", "--surface", "torus", "--central-charge", "2.5",
            "--lambda", "0", "--precision", "25"]
    if onepoint:
        argv += ["--torus-onept-ext", ".2", "--order", "3", "--npoles", "1"]
    _refused(argv + option, tmp_path, capsys, option[0].split("=")[0])


@pytest.mark.parametrize("option", [["--basis", "auto"], ["--basis=auto"]])
def test_torus_accepts_explicit_auto_basis(option, tmp_path):
    """--basis auto lets the family choose (torus_G), so it is what the torus table does anyway."""
    assert main(["table", "--surface", "torus", "--central-charge", "2.5", "--lambda", "0",
                 "--precision", "25", *option, "-o", str(tmp_path)]) in (None, 0)
    assert json.loads((tmp_path / "derivs_0.json").read_text())["basis"] == "torus_G"


def test_join_option_values_rewrites_only_the_flag_value():
    argv = ["--ramond-branch", "--", "--order", "6", "--ramond-branch", "-+", "--x", "--"]
    assert join_option_values(argv, "--ramond-branch", ("++", "+-", "-+", "--")) == [
        "--ramond-branch=--", "--order", "6", "--ramond-branch=-+", "--x", "--",
    ]


@pytest.mark.parametrize("spelling", [["--ramond-branch", "--"], ["--ramond-branch", "-+"], ["--ramond-branch=--"]])
def test_ramond_branch_documented_spellings(spelling, tmp_path):
    """``--ramond-branch --`` and ``-+`` failed with "expected one argument"; every spelling now builds."""
    assert main(_RAMOND + spelling + ["-o", str(tmp_path)]) in (None, 0)
    doc = json.loads((tmp_path / "derivs_0.json").read_text())
    assert doc["ramond_branch"] == spelling[-1].split("=")[-1]


def test_evaluate_ramond_branch_and_family_flags(capsys):
    common = ["--central-charge", "1.3", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--exchange-delta", "0.8",
              "--point", "z=0.3", "--point", "zbar=0.4", "--order", "10", "--precision", "30", "--format", "json",
              "--q-gate", "off"]
    assert evaluate_main(["--family", "ramond", *common, "--ramond-branch", "--"]) == 0
    separate = json.loads(capsys.readouterr().out)
    assert evaluate_main(["--family", "ramond", *common, "--ramond-branch=--"]) == 0
    joined = json.loads(capsys.readouterr().out)
    assert separate["branch"] == joined["branch"] == "--"
    assert separate["value"] == joined["value"]
    for argv, needle in ((["--family", "ns", *common, "--ramond-branch", "++"], "--ramond-branch is for"),
                         (["--family", "virasoro", *common, "--super-sector", "even"], "--super-sector is for")):
        with pytest.raises(SystemExit):
            evaluate_main(argv)
        assert needle in capsys.readouterr().err


def test_u_shift_on_a_chiral_basis_shifts_z(tmp_path, capsys):
    """For a chiral table, --u-shift applies to its single z coordinate."""
    main(_CHIRAL + ["--order", "6", "--u-shift", "0.1", "-o", str(tmp_path / "u")])
    main(_CHIRAL + ["--order", "6", "--z-shift", "0.1", "-o", str(tmp_path / "z")])
    u = json.loads((tmp_path / "u" / "derivs_chiral.json").read_text())
    z = json.loads((tmp_path / "z" / "derivs_chiral.json").read_text())
    assert u["derivs"] == z["derivs"]
    assert u.get("zbar_shift") is None
    common = ["--central-charge", "8", "--h-ext", "0.1", "0.1", "0.1", "0.1", "--coordinate", "z",
              "--exchange-h", "0.8", "--point", "z=0.3", "--format", "text", "--q-gate", "off"]
    capsys.readouterr()
    assert evaluate_main(common + ["--u-shift", "0.1"]) == 0
    via_u = capsys.readouterr().out
    assert evaluate_main(common + ["--z-shift", "0.1"]) == 0
    assert capsys.readouterr().out == via_u


def test_point_table_records_q_truncation_digits(tmp_path, capsys):
    """The point table records the q-order gate's estimate at its exchanged weight, as the table JSON does."""
    common = ["--central-charge", "8", "--lambda", "1", "--precision", "20", *_DELTA, "--exchange-delta", "2.4",
              "--exchange-spin", "2"]
    assert point_table_main(common + ["--order", "8", "-o", str(tmp_path / "low")]) == 0
    assert "q-order gate [point table]" in capsys.readouterr().err       # order 8 is short at this c
    low = json.loads((tmp_path / "low" / "derivs_point_2.json").read_text())
    assert point_table_main(common + ["--order", "24", "-o", str(tmp_path / "high")]) == 0
    high = json.loads((tmp_path / "high" / "derivs_point_2.json").read_text())
    assert float(low["q_truncation_digits"]) < float(high["q_truncation_digits"])
    assert point_table_main(common + ["--order", "8", "--q-gate", "off", "-o", str(tmp_path / "off")]) == 0
    assert "q_truncation_digits" not in json.loads((tmp_path / "off" / "derivs_point_2.json").read_text())
    with pytest.raises(SystemExit):
        point_table_main(["--family", "global", "--lambda", "1", "--precision", "20", *_DELTA,
                          "--exchange-delta", "2.4", "--q-gate", "warn", "-o", str(tmp_path / "g")])
    assert "--q-gate warn has nothing to check" in capsys.readouterr().err


def test_large_spin_tower_records_verified_digits(tmp_path):
    """The tower is an order-reduced interpolation in L: it records the digits it reproduces between its spin
    nodes (it wrote no verified_digits at all) and the spin interval that check covers; more poles, more digits."""
    docs = {}
    for npoles in ("2", "4"):
        argv = [a if a != "3" else npoles for a in _TOWER]
        main(argv + ["-o", str(tmp_path / npoles)])
        docs[npoles] = json.loads((tmp_path / npoles / "derivs_large_spin_0.json").read_text())
    for doc in docs.values():
        lo, hi = (float(x) for x in doc["verified_spin_range"])
        assert lo == 10.0 and hi > lo
        assert 0 < float(doc["verified_digits"]) <= 40
    assert float(docs["4"]["verified_digits"]) > float(docs["2"]["verified_digits"]) + 1


@pytest.mark.parametrize("operation", ["table", "point-table", "evaluate"])
def test_fixed_count_option_with_too_few_values_names_itself(operation, capsys):
    # argparse would fill --delta-ext with the following option names and fail later with an unrelated message
    from virasoro_blocks.__main__ import main

    with pytest.raises(SystemExit):
        main([operation, "--family", "global", "--delta-ext", "0.5", "--precision", "30"])
    assert "--delta-ext expects 4 values, got 1" in capsys.readouterr().err
