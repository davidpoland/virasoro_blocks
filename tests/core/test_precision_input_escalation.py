"""Verify escalated results against independently parsed exact inputs, including CLI output."""
from fractions import Fraction
import json

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request, evaluate_block_request
from virasoro_blocks.__main__ import main
from virasoro_blocks.serialization import serialized_table_from_json


def _point_request(precision, target=None, *, dimensions=False, exact_type=str):
    external = tuple(exact_type(Fraction(i, 7)) for i in range(1, 5))
    return BlockRequest(
        operation="point-table", family="global", coordinate="z",
        **({"delta_ext": external} if dimensions else {"h_ext": external}),
        exchange_h=exact_type(Fraction(2, 3)), center=exact_type(Fraction(2, 5)),
        z_shift=exact_type(Fraction(1, 9)),
        policy=NumericalPolicy(precision=precision, target_digits=target, lambda_order=2),
    )


@pytest.mark.parametrize("dimensions", [False, True])
@pytest.mark.parametrize("exact_type", [str, Fraction])
def test_point_target_preserves_exact_inputs_and_derived_weights(dimensions, exact_type):
    low = build_table_request(_point_request(20, 40, dimensions=dimensions, exact_type=exact_type))
    high = build_table_request(_point_request(90, dimensions=dimensions, exact_type=exact_type))
    assert low.working_dps > 20 and low.verified_digits >= 40
    with mp.workdps(90):
        got, ref = low.evaluate("z", low.shift), high.evaluate("z", high.shift)
        for a, b in zip(got, ref):
            assert abs((a[0] - b[0]) / b[0]) < mp.mpf("1e-40")


def test_point_cli_target_preserves_rational_inputs(tmp_path):
    main(["point-table", "--family", "global", "--basis", "chiral_F", "--coordinates", "z",
          "--h-ext", "1/7", "2/7", "3/7", "4/7", "--exchange-h", "2/3",
          "--center", "z=2/5", "--z-shift", "1/9", "--lambda", "2",
          "--precision", "20", "--target-digits", "40", "-o", str(tmp_path)])
    doc = json.loads((tmp_path / "derivs_point_0.json").read_text())
    low = serialized_table_from_json(doc)
    high = build_table_request(_point_request(90))
    with mp.workdps(90):
        assert abs(low.shift - high.shift) < mp.mpf("1e-40")
        got, ref = low.evaluate("z", low.shift), high.evaluate("z", high.shift)
        for a, b in zip(got, ref):
            assert abs((a[0] - b[0]) / b[0]) < mp.mpf("1e-40")


def _torus_request(precision, target=None):
    return BlockRequest(
        operation="evaluate", surface="torus", family="torus-onepoint", coordinate="vvb",
        central_charge="10.3", torus_onepoint_ext=0, exchange_delta="7/3", exchange_spin=1,
        point={"v": "1/7", "vbar": "1/9"},
        policy=NumericalPolicy(precision=precision, target_digits=target, order=4),
    )


def test_torus_target_keeps_derived_exchange_weights_and_emits_achieved_digits():
    low = evaluate_block_request(_torus_request(20, 40), q_gate="off")
    high = evaluate_block_request(_torus_request(90), q_gate="off")
    assert low.metadata["precision_actual"] > 20
    with mp.workdps(90):
        assert abs((low.value - high.value) / high.value) < mp.mpf("1e-40")
        assert abs((mp.mpf(low.as_dict()["value"]) - high.value) / high.value) < mp.mpf("1e-40")
        assert abs(mp.mpf(low.metadata["exchange"]["delta"]) - mp.mpf(7) / 3) < mp.mpf("1e-40")
        assert abs(mp.mpf(low.metadata["point"]["v"]) - mp.mpf(1) / 7) < mp.mpf("1e-40")


@pytest.mark.parametrize("output_format", ["json", "text"])
def test_torus_cli_emits_target_digits(output_format, capsys):
    main(["evaluate", "--surface", "torus", "--family", "torus-onepoint", "--coordinate", "vvb",
          "--central-charge", "10.3", "--torus-onept-ext", "0", "--exchange-delta", "7/3",
          "--exchange-spin", "1", "--point", "v=1/7", "--point", "vbar=1/9", "--order", "4",
          "--precision", "20", "--target-digits", "40", "--format", output_format])
    output = capsys.readouterr().out
    value = json.loads(output)["value"] if output_format == "json" else output.strip()
    high = evaluate_block_request(_torus_request(90), q_gate="off")
    with mp.workdps(90):
        assert abs((mp.mpf(value) - high.value) / high.value) < mp.mpf("1e-40")


@pytest.mark.parametrize("family, fields", [
    ("global", {"coordinate": "z", "delta_ext": ("1/7", "2/7", "3/7", "4/7")}),
    ("torus-character", {"surface": "torus", "coordinate": "vvb", "central_charge": "10.123456789123456789"}),
])
def test_ordinary_tables_escalate_for_the_original_inputs(family, fields):
    def build(precision, target=None):
        return build_table_request(BlockRequest(
            operation="table", family=family, **fields,
            policy=NumericalPolicy(precision=precision, target_digits=target, lambda_order=1)))

    low, high = build(20, 30), build(90)
    with mp.workdps(90):
        got, ref = (t.evaluate(fields["coordinate"], mp.mpf("2.3")) for t in (low, high))
        for a, b in zip(got, ref):
            for u, v in zip(a, b):
                if v:
                    assert abs((u - v) / v) < mp.mpf("1e-30")


def test_escalated_output_keeps_resolved_small_imaginary_components():
    from dataclasses import replace

    low_request = replace(_torus_request(20, 40), point={"v": "0.1+1e-30j", "vbar": 0})
    high_request = replace(_torus_request(90), point={"v": "0.1+1e-30j", "vbar": 0})
    low = evaluate_block_request(low_request, q_gate="off")
    high = evaluate_block_request(high_request, q_gate="off")
    doc = low.as_dict()
    assert isinstance(doc["value"], dict)
    assert isinstance(doc["point"]["v"], dict)
    with mp.workdps(90):
        emitted = mp.mpc(doc["value"]["real"], doc["value"]["imag"])
        assert abs((emitted - high.value) / high.value) < mp.mpf("1e-40")
