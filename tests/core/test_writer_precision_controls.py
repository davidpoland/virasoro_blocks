"""Invalid digit counts must not silently destroy serialized numerical data."""

from fractions import Fraction
import json

import numpy as np
import pytest

from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.serialization import SerializedDerivTable, serialized_table_from_json
from virasoro_blocks.writers import _positive_integer, write_json, write_mathematica, write_torus_json


def emit(kind, digits=None, **metadata):
    coord = "vvb" if kind == "torus" else "zzb"
    table = SerializedDerivTable(
        shift=0, base=Fraction(1, 2), pole_list_x=(), coordinates=(coord,),
        numerators={coord: [[Polynomial([Fraction(1, 3)])]]},
    )
    if kind == "torus":
        return write_torus_json(table, c=8, sig_figs=digits, **metadata)
    kwargs = dict(c=8, h_ext=(0,) * 4, hbar_ext=(0,) * 4,
                  q_order=0, sig_figs=digits, **metadata)
    if kind == "mathematica":
        return write_mathematica(table, coordinate=coord, **kwargs)
    return write_json(table, **kwargs)


@pytest.mark.parametrize("digits", [True, np.bool_(False), 0, -1, 2.5,
                                    float("nan"), float("inf"), "20"])
def test_writer_digit_validator_rejects_invalid_types_and_bounds(digits):
    with pytest.raises(ValueError, match="sig_figs"):
        _positive_integer("sig_figs", digits)


@pytest.mark.parametrize("kind", ["json", "mathematica", "torus"])
@pytest.mark.parametrize("digits", [0, 2.5])
def test_writer_rejects_invalid_digit_counts(kind, digits):
    with pytest.raises(ValueError, match="sig_figs"):
        emit(kind, digits)


@pytest.mark.parametrize("kind", ["json", "mathematica", "torus"])
@pytest.mark.parametrize("digits", [20.0, Fraction(20), np.int64(20)])
def test_writer_normalizes_integral_digit_counts(kind, digits):
    assert emit(kind, digits) == emit(kind, 20)


@pytest.mark.parametrize("kind", ["json", "mathematica", "torus"])
@pytest.mark.parametrize("field", ["precision_input", "precision_actual"])
@pytest.mark.parametrize("value", [0, 2.5])
def test_writer_rejects_invalid_precision_metadata(kind, field, value):
    with pytest.raises(ValueError, match=field):
        emit(kind, **{field: value})


@pytest.mark.parametrize("kind", ["json", "mathematica", "torus"])
@pytest.mark.parametrize("field", ["precision_input", "precision_actual"])
@pytest.mark.parametrize("value", [30.0, Fraction(30), np.int64(30)])
def test_precision_metadata_is_integral_and_loadable(kind, field, value):
    text = emit(kind, **{field: value})
    assert text == emit(kind, **{field: 30})
    if kind != "mathematica":
        doc = json.loads(text)
        assert type(doc[field]) is int
        restored = serialized_table_from_json(doc)
        assert restored.coordinates == (("vvb",) if kind == "torus" else ("zzb",))
