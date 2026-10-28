"""A numeric string must not be mistaken for an array of digit coefficients."""

import pytest

from virasoro_blocks.serialization import serialized_table_from_json


def document(basis, coeffs):
    chiral = basis == "chiral_F"
    coord = "z" if chiral else "vvb" if basis == "torus_G" else "zzb"
    return dict(
        coordinates=[coord], basis=basis, spin=0, **{"lambda": 0},
        exp_base="1", variable_minus_x="0", pole_list_x=[],
        derivs=[[coeffs]] if chiral else [[[coeffs]]],
    )


@pytest.mark.parametrize("basis", ["scalar_G", "chiral_F", "torus_G"])
@pytest.mark.parametrize("coeffs", ["12", {"1": "ignored", "2": "ignored"}, 12, None])
def test_loader_rejects_non_array_polynomial(basis, coeffs):
    with pytest.raises(ValueError, match="coefficient.*array"):
        serialized_table_from_json(document(basis, coeffs))


@pytest.mark.parametrize("basis", ["scalar_G", "chiral_F", "torus_G"])
@pytest.mark.parametrize("coeffs,expected", [(["12"], 12), (["1", "2"], 7)])
def test_coefficient_arrays_keep_their_polynomial_meaning(basis, coeffs, expected):
    doc = document(basis, coeffs)
    table = serialized_table_from_json(doc)
    assert table.evaluate(doc["coordinates"][0], 3)[0][0] == expected


@pytest.mark.parametrize("basis", ["scalar_G", "chiral_F", "torus_G"])
def test_sample_value_string_is_not_an_array_of_digit_samples(basis):
    doc = document(basis, ["1", "2"])
    coord = doc["coordinates"][0]
    chiral = basis == "chiral_F"
    doc["sample_points_x"] = ["1", "2"]
    doc["sample_block_values_index_names"] = (
        ["coordinate_index", "deriv", "sample_point"] if chiral else
        ["coordinate_index", "deriv_0", "deriv_1", "sample_point"]
    )
    doc["sample_block_values"] = {coord: ["35"] if chiral else [["35"]]}
    with pytest.raises(ValueError, match="sample.*array"):
        serialized_table_from_json(doc)
    doc["sample_block_values"][coord] = [["3", "5"]] if chiral else [[["3", "5"]]]
    table = serialized_table_from_json(doc)
    assert [sample[0][0] for sample in table.sample_block_values[coord]] == [3, 5]


@pytest.mark.parametrize("basis", ["scalar_G", "chiral_F", "torus_G"])
def test_dense_sample_rows_require_arrays(basis):
    doc = document(basis, ["1"])
    coord = doc["coordinates"][0]
    doc["sample_points_x"] = ["1"]
    doc["sample_block_values"] = {coord: [["12"]]}
    with pytest.raises(ValueError, match="sample.*array"):
        serialized_table_from_json(doc)
    doc["sample_block_values"][coord] = [[["12"]]]
    table = serialized_table_from_json(doc)
    assert table.sample_block_values[coord][0][0][0] == 12


@pytest.mark.parametrize("field", ["pole_list_x", "sample_points_x",
                                   "verified_delta_range", "verified_spin_range"])
@pytest.mark.parametrize("value", ["12", {"1": 0, "2": 0}, 12])
def test_numeric_vector_fields_require_arrays(field, value):
    doc = document("scalar_G", ["1"])
    if field == "verified_spin_range":
        doc["variable"] = "L"
    doc[field] = value
    with pytest.raises(ValueError, match=field + ".*array"):
        serialized_table_from_json(doc)


@pytest.mark.parametrize("value", ["z", {"z": 0}])
def test_coordinate_sequence_requires_an_array(value):
    doc = document("chiral_F", ["1"])
    doc["coordinates"] = value
    with pytest.raises(ValueError, match="coordinates.*array"):
        serialized_table_from_json(doc)


def test_valid_numeric_vectors_keep_their_values():
    doc = document("scalar_G", ["1"])
    doc.update(pole_list_x=["-12"], sample_points_x=["12"],
               verified_delta_range=["1", "12"])
    table = serialized_table_from_json(doc)
    assert table.pole_list_x == (-12,)
    assert table.sample_points_x == (12,)
    assert table.verified_range == (1, 12)
    assert table.evaluate("zzb", 0)[0][0] == pytest.approx(1 / 12)
