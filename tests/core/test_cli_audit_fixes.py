"""Pre-release audit fixes of the blocks CLI: messages that match behavior, refusals instead of crashes or silent
misbehavior, honest point-value accuracy, collision-free Mathematica names, and real entries from real tables."""
from __future__ import annotations

import json

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main as table_main
from virasoro_blocks.cli.evaluate import main as evaluate_main
from virasoro_blocks.cli.point_table import main as point_table_main
from virasoro_blocks.requests import BlockRequest, NumericalPolicy, RequestValidationError, validate_request

_EXT = ["--delta-ext", "0.1", "0.3", "0.1", "0.3"]


def _error(capsys, function, argv):
    with pytest.raises(SystemExit) as info:
        function(argv)
    assert info.value.code == 2
    return capsys.readouterr().err


# --- a negative gap on the sphere is refused (it crashed for unequal externals, and was silently accepted below
# --- unitarity for identical ones)

@pytest.mark.parametrize("ext", [_EXT, ["--delta-ext", "0.1", "0.1", "0.1", "0.1"]])
def test_negative_sphere_gap_is_refused(capsys, tmp_path, ext):
    err = _error(capsys, table_main, ["table", "--central-charge", "8", *ext, "--gap", "-0.05", "--lambda", "2",
                                      "--order", "16", "--npoles", "4", "-o", str(tmp_path)])
    assert "--gap -0.05 is negative" in err and "below unitarity" in err
    assert not list(tmp_path.iterdir())


def test_negative_sphere_gap_is_refused_by_the_request_layer():
    def request(gap):
        return BlockRequest(operation="table", family="virasoro", coordinate="zzb", central_charge=8,
                            delta_ext=("0.1",) * 4, gap=gap, policy=NumericalPolicy(precision=30, lambda_order=2))

    validate_request(request("0"))
    validate_request(request("1/3"))
    with pytest.raises(RequestValidationError, match="gap -1/20 is negative"):
        validate_request(request("-1/20"))
    with pytest.raises(RequestValidationError, match="finite real"):
        validate_request(request("1+2j"))


# --- --chiral-norm is a positive integer on the command line (the bootstrap reuses integer norms only)

@pytest.mark.parametrize("function, argv", [
    (table_main, ["table", "--central-charge", "8", *_EXT, "--lambda", "2", "--order", "8", "--npoles", "4"]),
    (point_table_main, ["--central-charge", "8", *_EXT, "--exchange-delta", "2", "--lambda", "2"]),
    (evaluate_main, ["--central-charge", "8", *_EXT, "--exchange-delta", "2", "--point", "z=0.4",
                     "--point", "zbar=0.4"]),
])
@pytest.mark.parametrize("norm", ["16.5", "0", "-16", "33/2"])
def test_chiral_norm_must_be_a_positive_integer(capsys, tmp_path, function, argv, norm):
    out = [] if function is evaluate_main else ["-o", str(tmp_path)]
    err = _error(capsys, function, [*argv, f"--chiral-norm={norm}", *out])
    assert "--chiral-norm must be a positive integer" in err


def test_integer_chiral_norm_spellings_are_accepted(capsys):
    values = []
    for norm in ("4", "4.0", "8/2"):
        assert evaluate_main(["--central-charge", "8", *_EXT, "--exchange-delta", "2", "--point", "z=0.4",
                              "--point", "zbar=0.4", "--order", "12", "--precision", "20",
                              "--chiral-norm", norm]) == 0
        values.append(json.loads(capsys.readouterr().out)["value"])
    assert len(set(values)) == 1


# --- messages that matched no behavior

def test_tower_note_offers_the_chosen_pole_count(capsys, tmp_path):
    """A large-spin tower without --npoles chooses its pole count, so its order-reduction note offers that."""
    assert table_main(["table", "--central-charge", "8", "--delta-ext", "0.1", "0.1", "0.1", "0.1",
                       "--large-spin-lmin", "10", "--large-spin-double-twists", "1", "--lambda", "1",
                       "--npoles", "2", "--order", "8", "--precision", "30", "-o", str(tmp_path)]) in (0, None)
    err = capsys.readouterr().err
    assert "the order reduction limits it (raise --npoles, or omit it to choose the pole count)" in err


def test_keep_all_alternative_is_optional_in_the_knob_names():
    from virasoro_blocks.verified_build import BOOTSTRAP_KNOBS, CLI_KNOBS, KnobNames

    assert CLI_KNOBS.more_poles() == "raise --npoles, or omit it to keep every pole"
    assert BOOTSTRAP_KNOBS.more_poles() == 'raise npoles, or use npoles="all" to keep every pole'
    assert KnobNames(keep_all=None).more_poles() == "raise --npoles"


def test_sample_via_help_offers_only_accepted_values(capsys, tmp_path):
    from virasoro_blocks.__main__ import _build_parser

    help_text = " ".join(_build_parser().format_help().split())
    assert "default 'auto' = numeric" not in help_text and "'auto' is not a value to pass" in help_text
    err = _error(capsys, table_main, ["table", "--central-charge", "8", *_EXT, "--lambda", "2", "--npoles", "4",
                                      "--sample-via", "auto", "-o", str(tmp_path)])
    assert "invalid choice: 'auto'" in err


# --- point values report their accuracy as the block, q-truncation included

def _evaluate(capsys, *extra):
    code = evaluate_main(["--central-charge", "10", "--delta-ext", "0.15", "0.15", "0.15", "0.15",
                          "--exchange-delta", "2", "--exchange-spin", "2", *extra])
    assert code == 0
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.err


def test_estimated_digits_include_the_q_truncation(capsys):
    """At z = 0.999 order 10 leaves ~3 digits of the block; estimated_digits used to claim 57."""
    doc, err = _evaluate(capsys, "--point", "z=0.999", "--point", "zbar=0.999", "--order", "10")
    assert doc["q_truncation_digits"] < 5
    assert doc["estimated_digits"] == doc["q_truncation_digits"]
    assert "q-order gate" in err
    doc, _err = _evaluate(capsys, "--point", "z=0.3", "--point", "zbar=0.4", "--order", "30", "--precision", "30")
    assert doc["estimated_digits"] <= min(30, doc["q_truncation_digits"])
    assert doc["estimated_digits"] > 25


def test_target_digits_checks_the_q_truncation_against_the_target(capsys):
    """The q-gate threshold is the target with --target-digits (20 digits otherwise); precision cannot raise it."""
    args = ("--point", "z=0.4", "--point", "zbar=0.4", "--order", "20", "--precision", "20")
    doc, err = _evaluate(capsys, "--point", "z=0.4", "--point", "zbar=0.4", "--order", "20", "--precision", "60")
    truncation = doc["q_truncation_digits"]
    assert 20 < truncation < 45 and doc["estimated_digits"] == truncation and "q-order gate" not in err
    doc, err = _evaluate(capsys, *args, "--target-digits", "50")
    assert doc["precision_actual"] >= 50 and doc["estimated_digits"] < 50
    assert "q-order gate" in err
    with pytest.raises(SystemExit):
        evaluate_main(["--central-charge", "10", "--delta-ext", "0.15", "0.15", "0.15", "0.15",
                       "--exchange-delta", "2", "--exchange-spin", "2", *args, "--target-digits", "50",
                       "--q-gate", "strict"])


# --- Mathematica filenames name every table-changing setting

def test_mathematica_filename_tags_every_table_changing_setting():
    from virasoro_blocks.writers import mathematica_filename

    common = dict(c=8, h_ext=["0.05"] * 4, hbar_ext=["0.05"] * 4, spin=0, q_order=12, m_max=2, n_max=2,
                  npoles=4, basis="scalar_G", chiral_norm=16)
    default = mathematica_filename("zzb", **common)
    assert default == "zzbDerivTable-c8-delta1-0.1-delta2-0.1-delta3-0.1-delta4-0.1-L0-lambda2-npoles4-chiralNorm16-order12.m"
    # the defaults, spelled out, keep the short name
    assert mathematica_filename("zzb", **common, pole_selection="contribution_optimal", z_shift=0, z_center="1/2",
                                zbar_center="0.5", gap=0, parity_exchanged="1") == default
    variants = [
        dict(gap="0.3"), dict(z_shift="-0.1"), dict(z_center="0.4"), dict(z_center="0.4", zbar_center="0.45"),
        dict(pole_selection="nearest"), dict(shared_sample_points=True),
    ]
    names = {mathematica_filename("zzb", **common, **variant) for variant in variants}
    assert len(names) == len(variants) and default not in names
    assert "-polesNearest-" in mathematica_filename("zzb", **common, pole_selection="nearest")
    assert "-zCenter0.4-zbCenter0.45-" in mathematica_filename("zzb", **common, z_center="0.4", zbar_center="0.45")

    two_sided = dict(common, h_ext=["0.6", "0.2", "0.1", "0.2"], hbar_ext=["0.1", "0.2", "0.1", "0.2"], spin=1,
                     basis="parity_projected_B", parity_projection="-")
    name = mathematica_filename("zzb", **two_sided, parity_external=["1", "-1", "1", "-1"], parity_exchanged="-1",
                                z_shift="-0.3", zbar_shift="-0.2")
    assert "-basisParityMinus-parityExt+1-1+1-1-parityExch-1-chiralNorm16-zShift-0.3-zbShift-0.2-order12.m" in name
    # scalar_G carries one shift: its zbar copy does not add a tag
    assert "zbShift" not in mathematica_filename("zzb", **common, z_shift="-0.1", zbar_shift="-0.1")


def test_mathematica_runs_differing_in_gap_do_not_overwrite(tmp_path):
    base = ["table", "--central-charge", "8", "--delta-ext", "0.1", "0.1", "0.1", "0.1", "--lambda", "2",
            "--order", "12", "--npoles", "4", "--precision", "30", "--format", "m", "-o", str(tmp_path)]
    table_main(base)
    table_main([*base, "--gap", "0.3"])
    names = sorted(path.name for path in tmp_path.iterdir())
    assert len(names) == 2 and any("-gap0.3-" in name for name in names)


# --- real tables evaluate to real entries

def test_serialized_real_table_evaluates_to_mpf():
    from pathlib import Path

    from virasoro_blocks.serialization import serialized_table_from_json

    root = Path(__file__).resolve().parents[2] / "examples" / "block_samples"
    for name in ("scalar_G/derivs_0.json", "parity_projected_B/derivs_1.json", "chiral_F/derivs_chiral.json",
                 "global_scalar_G/derivs_2.json"):
        table = serialized_table_from_json(json.loads((root / name).read_text()))
        assert any(mp.im(mp.mpc(pole)) != 0 for pole in table.pole_list_x) or "global" in name
        rows = table.evaluate(table.coordinates[0], "1.7")
        assert all(isinstance(entry, mp.mpf) for row in rows for entry in row)
        float(rows[0][0])
        complex_rows = table.evaluate(table.coordinates[0], mp.mpc("1.7", "0.1"))
        assert isinstance(complex_rows[0][0], mp.mpc)
        # (a complex-typed real Delta is a real point too)
        assert isinstance(table.evaluate(table.coordinates[0], mp.mpc("1.7"))[0][0], mp.mpf)
