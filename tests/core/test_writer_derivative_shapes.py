"""Writers must not advertise derivative grids their payload cannot represent."""

import json

import pytest

from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.serialization import SerializedDerivTable, serialized_table_from_json
from virasoro_blocks.writers import write_json, write_mathematica


def grid(rows, cols):
    return [[Polynomial([1 + m + n]) for n in range(cols)] for m in range(rows)]


def table(numerators):
    return SerializedDerivTable(0, 1, (), tuple(numerators), numerators)


KWARGS = dict(c=8, h_ext=[0] * 4, hbar_ext=[0] * 4, q_order=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_json_requires_one_shared_coordinate_cutoff(reverse):
    entries = [("zzb", grid(2, 2)), ("qqb", grid(3, 3))]
    if reverse:
        entries.reverse()
    with pytest.raises(ValueError, match="cutoff"):
        write_json(table(dict(entries)), **KWARGS)


@pytest.mark.parametrize("coord", ["zzb", "ws"])
@pytest.mark.parametrize("writer", ["json", "mathematica"])
def test_writer_refuses_missing_required_derivative_entries(coord, writer):
    source = table({coord: grid(5, 2)})
    with pytest.raises(ValueError, match="derivative.*entries"):
        if writer == "json":
            write_json(source, **KWARGS)
        else:
            write_mathematica(source, coordinate=coord, **KWARGS)


def test_rectangular_grid_is_valid_when_all_kept_entries_exist():
    source = table({"zzb": grid(4, 2), "ws": grid(4, 2)})
    doc = json.loads(write_json(source, **KWARGS))
    restored = serialized_table_from_json(doc)
    assert doc["lambda"] == 3
    for coord in source.coordinates:
        assert restored.evaluate(coord, 2)[1][1] == 3
