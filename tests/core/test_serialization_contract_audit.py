"""Round-trip cutoffs and reject silently truncated clean sample payloads."""
from copy import deepcopy
import json

import pytest

from virasoro_blocks.serialization import serialized_table_from_json
from virasoro_blocks.truncation import kept_pairs
from virasoro_blocks.writers import write_json


def _derivatives_for_pairs(pairs):
    rows = {}
    for m, n in pairs:
        rows.setdefault(m, []).append([str(1 + m + n)])
    return list(rows.values())


@pytest.mark.parametrize("spin", [True, False, .25, float("nan"), float("inf")])
@pytest.mark.parametrize("coord", ["zzb", "ws"])
def test_loader_rejects_invalid_spin_before_selecting_layout(spin, coord):
    doc = {"coordinates": [coord], "lambda": 0, "basis": "scalar_G", "spin": spin,
           "exp_base": "1", "variable_minus_x": "0", "pole_list_x": [],
           "derivs": [[[['1']]]]}
    if coord == "ws" and spin is True:
        doc["derivs"] = [[]]  # the odd-spin cutoff would otherwise mask the bad label
    with pytest.raises(ValueError, match="spin"):
        serialized_table_from_json(doc)


@pytest.mark.parametrize("spin", [-1, 0, .5, -.5])
def test_loader_preserves_valid_ordered_spin_labels(spin):
    doc = {"coordinates": ["zzb"], "lambda": 0, "basis": "ordered_B", "spin": spin,
           "exp_base": "1", "variable_minus_x": "0", "pole_list_x": [],
           "derivs": [[[['2']]]]}
    assert serialized_table_from_json(doc).evaluate("zzb", 3)[0][0] == 2


@pytest.mark.parametrize("coord", ["zzb", "yyb", "qqb"])
def test_off_diagonal_files_require_the_complete_derivative_triangle(coord):
    doc = {"coordinates": [coord], "lambda": 2, "basis": "scalar_G", "spin": 0,
           "z_center": "0.4", "zbar_center": "0.3", "exp_base": "1",
           "variable_minus_x": "0", "pole_list_x": []}
    pairs = kept_pairs(coord, 2, basis="scalar_G", off_diagonal=True)
    doc["derivs"] = [_derivatives_for_pairs(pairs)]
    assert serialized_table_from_json(doc).evaluate(coord, 1)[0][1] == 2
    doc["derivs"] = [_derivatives_for_pairs(kept_pairs(coord, 2, basis="scalar_G"))]
    with pytest.raises(ValueError, match="regenerate"):
        serialized_table_from_json(doc)


@pytest.mark.parametrize("coord", ["xt", "ws", "qqb_ws"])
@pytest.mark.parametrize("projection", ["+", "-"])
def test_folded_files_require_the_cutoff_from_sign_times_phase(coord, projection):
    doc = {"coordinates": [coord], "lambda": 2, "basis": "parity_projected_B", "spin": 0,
           "parity_projection": projection, "parity_external": ["1", "1", "1", "-1"],
           "exp_base": "1", "variable_minus_x": "0", "pole_list_x": []}
    fields = dict(basis="parity_projected_B", parity_projection=projection)
    doc["derivs"] = [_derivatives_for_pairs(kept_pairs(coord, 2, parity_phase=-1, **fields))]
    assert serialized_table_from_json(doc).evaluate(coord, 1)[0][0] == 1
    doc["derivs"] = [_derivatives_for_pairs(kept_pairs(coord, 2, **fields))]
    with pytest.raises(ValueError, match="regenerate"):
        serialized_table_from_json(doc)


@pytest.mark.parametrize("lam", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("basis", ["parity_projected_B", "scalar_G"])
def test_odd_folded_table_preserves_cutoff_and_samples_when_rewritten(lam, basis):
    # an odd-parity fold (parity '-') keeps m + 2n <= lambda - 1; an odd-spin scalar_G is z <-> zbar even
    projected = dict(parity_projection="-", parity_external=["1"] * 4) if basis == "parity_projected_B" else {}
    rows = {}
    samples = {}
    for m, n in kept_pairs("xt", lam, basis=basis, ell=1, parity_projection=projected.get("parity_projection")):
        rows.setdefault(m, []).append([str(m + n + 1)])
        samples.setdefault(m, []).append([str(m + n + 1), str(m + n + 2)])
    doc = {"coordinates": ["xt"], "lambda": lam, "basis": basis, "spin": 1,
           "exp_base": "0.5", "variable_minus_x": "1", "pole_list_x": [],
           "derivs": [list(rows.values())], "sample_points_x": ["1", "2"],
           "sample_block_values": {"xt": list(samples.values())},
           "sample_block_values_index_names": ["coordinate_index", "deriv_0", "deriv_1", "sample_point"],
           **projected, **({"scalar_G_odd": "even"} if basis == "scalar_G" else {})}
    table = serialized_table_from_json(doc)
    for _ in range(2):
        emitted = json.loads(write_json(table, c=8, h_ext=[0] * 4, hbar_ext=[0] * 4,
                                       spin=1, q_order=0, basis=basis, **projected))
        assert emitted["lambda"] == lam
        assert len(emitted["derivs"][0]) == len(doc["derivs"][0])
        restored = serialized_table_from_json(emitted)
        assert restored.numerators.keys() == table.numerators.keys()
        assert restored.evaluate("xt", 3) == table.evaluate("xt", 3)
        assert restored.sample_block_values == table.sample_block_values
        table = restored


def _document(chiral):
    coord = "z" if chiral else "zzb"
    return {"coordinates": [coord], "lambda": 1, "basis": "chiral_F" if chiral else "scalar_G",
            "exp_base": "0.5", "variable_minus_x": "0", "pole_list_x": [],
            "derivs": [[["1"], ["2"]]] if chiral else [[[["1"]], [["2"]]]],
            "sample_points_x": ["1"],
            "sample_block_values": {coord: [["1"], ["2"]] if chiral else [[["1"]], [["2"]]]},
            "sample_block_values_index_names": ["coordinate_index", "deriv", "sample_point"] if chiral
            else ["coordinate_index", "deriv_0", "deriv_1", "sample_point"]}


@pytest.mark.parametrize("chiral", [False, True])
@pytest.mark.parametrize("damage", ["extra_sample", "missing_sample", "extra_row", "missing_row"])
def test_sample_layout_refuses_surplus_or_missing_data(chiral, damage):
    doc = _document(chiral)
    rows = doc["sample_block_values"][doc["coordinates"][0]]
    leaf = rows[1] if chiral else rows[1][0]
    if damage == "extra_sample":
        leaf.append("999")
    elif damage == "missing_sample":
        leaf.clear()
    elif damage == "extra_row":
        rows.append(deepcopy(rows[-1]))
    else:
        rows.pop()
    with pytest.raises(ValueError, match="sample_block_values"):
        serialized_table_from_json(doc)


def test_duplicate_coordinates_are_not_silently_overwritten():
    doc = _document(True)
    doc["coordinates"] *= 2
    doc["derivs"].append([["999"], ["999"]])
    with pytest.raises(ValueError, match="coordinates"):
        serialized_table_from_json(doc)


def test_loader_names_malformed_nested_payloads():
    for invalid in (None, [], 1):
        with pytest.raises(ValueError, match='JSON object'):
            serialized_table_from_json(invalid)
    for chiral in (False, True):
        coord = 'z' if chiral else 'zzb'
        for fields, expected in (
            ({'coordinates': [[coord]]}, 'coordinates'),
            ({'basis': []}, 'basis'),
            ({'parity_projection': []}, 'parity_projection'),
            ({'derivs': [None]}, 'derivs'),
            ({'derivs': [{'row': ['1']}]}, 'derivs'),
            ({'sample_block_values': [['1']]}, 'sample_block_values'),
            ({'sample_block_values': {coord: None}}, 'sample_block_values'),
            ({'sample_block_values': {coord: [None, None]}}, 'sample_block_values'),
            ({'parity_external': '1111'}, 'parity_external'),
            ({'parity_external': [0] * 4}, 'parity'),
        ):
            with pytest.raises(ValueError, match=expected):
                serialized_table_from_json({**_document(chiral), **fields})
    # Numeric rows need an array at both axes, rather than a string's length.
    doc = _document(False)
    doc['derivs'] = [[None, None]]
    with pytest.raises(ValueError, match='derivs'):
        serialized_table_from_json(doc)


@pytest.mark.parametrize("tables", [[], [[["1"]], [["2"]]], [[[]]], [[["1"]]]])
def test_legacy_dense_samples_validate_node_counts_and_required_entries(tables):
    doc = _document(True)
    del doc["sample_block_values_index_names"]
    doc["sample_block_values"] = {"z": tables}
    with pytest.raises(ValueError, match="sample_block_values"):
        serialized_table_from_json(doc)


def test_valid_legacy_dense_samples_still_load():
    doc = _document(True)
    del doc["sample_block_values_index_names"]
    doc["sample_block_values"] = {"z": [[["1"], ["2"]]]}
    assert serialized_table_from_json(doc).sample_block_values["z"] == [[[1], [2]]]


@pytest.mark.parametrize("field", ["exp_base", "variable_minus_x", "pole_list_x", "derivs", "sample_block_values"])
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "(nan+1j)"])
def test_nonfinite_serialized_numbers_are_refused(field, bad):
    doc = _document(True)
    if field == "pole_list_x":
        doc[field] = [bad]
    elif field == "derivs":
        doc[field][0][0][0] = bad
    elif field == "sample_block_values":
        doc[field]["z"][0][0] = bad
    else:
        doc[field] = bad
    with pytest.raises(ValueError, match="non-finite"):
        serialized_table_from_json(doc)


def _exact_table(root=-1, scale=1):
    import mpmath as mp
    from virasoro_blocks.polyderivs import DeltaDerivTable
    from virasoro_blocks.polynomial import Polynomial

    return DeltaDerivTable(mp.mpf(1), Polynomial([-root, 1]).scale(scale), [root],
                           [[Polynomial([scale])]])


def test_interpolated_coordinate_tables_require_the_same_exact_origin():
    from dataclasses import replace
    from fractions import Fraction
    import mpmath as mp
    from virasoro_blocks.interpolation import DampedRational
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.polyderivs import InterpolatedDeltaTable
    from virasoro_blocks.serialization import serialize_deriv_tables

    with mp.workdps(30):
        first = InterpolatedDeltaTable(DampedRational(1, (), mp.mpf('.5')),
            shift='.35', sample_points=(), numerators=[[Polynomial.one()]])
        equivalent = replace(first, shift=Fraction(7, 20))
        serialize_deriv_tables({'z': first, 'y': equivalent})
        shifted = replace(first, shift=Fraction(7, 20) + Fraction(1, 10**70))
        with pytest.raises(ValueError, match='shift'):
            serialize_deriv_tables({'z': first, 'y': shifted})


def test_exact_coordinate_tables_must_share_poles_not_just_pole_count():
    from virasoro_blocks.serialization import serialize_deriv_tables

    with pytest.raises(ValueError, match='poles'):
        serialize_deriv_tables({'zzb': _exact_table(-1), 'qqb': _exact_table(-2)})


def test_exact_coordinate_normalizations_survive_serialization():
    from virasoro_blocks.serialization import serialize_deriv_tables

    tables = {'zzb': _exact_table(scale=1), 'qqb': _exact_table(scale=3)}
    result = serialize_deriv_tables(tables)
    for coord, original in tables.items():
        assert result.evaluate(coord, 1)[0][0] == original.evaluate(1)[0][0]


def test_interpolated_constants_and_shared_metadata_survive_serialization():
    import mpmath as mp
    from virasoro_blocks.interpolation import DampedRational
    from virasoro_blocks.polyderivs import InterpolatedDeltaTable
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import serialize_deriv_tables

    tables = {coord: InterpolatedDeltaTable(DampedRational(scale, (-1,), mp.mpf('.5')),
                                           0, (1,), [[Polynomial([1])]], metadata={'block_family': 'ns'})
              for coord, scale in [('zzb', 1), ('qqb', 3)]}
    result = serialize_deriv_tables(tables)
    assert result.metadata == {'block_family': 'ns'}
    for coord, original in tables.items():
        assert result.evaluate(coord, 1)[0][0] == original.evaluate(1)[0][0]


def test_serialized_evaluation_accepts_exact_fraction_input():
    from fractions import Fraction
    from virasoro_blocks.serialization import serialize_deriv_tables

    result = serialize_deriv_tables({'zzb': _exact_table()})
    assert result.evaluate('zzb', Fraction(1, 7)) == result.evaluate('zzb', '1/7')


@pytest.mark.parametrize('variable', ['Delta', 'L', 'torus'])
def test_verification_context_survives_loading_and_rewriting(variable):
    from virasoro_blocks.writers import write_torus_json

    doc = _document(False)
    doc['precision_actual'] = 40
    doc['verified_digits'] = '25.0'
    doc['verified_digits_largest_entry'] = '28.0'
    doc['target_digits'] = 24
    doc['q_truncation_digits'] = '31.0'
    range_key = 'verified_spin_range' if variable == 'L' else 'verified_delta_range'
    doc[range_key] = ['0.5', '20.5']
    if variable == 'torus':
        doc['coordinates'] = ['vvb']
        doc['basis'] = 'torus_G'
        doc.pop('sample_block_values')
        doc['fit_digits'] = '27.0'
        doc['precision_digits'] = '32.0'
    else:
        doc['variable'] = variable
    table = serialized_table_from_json(doc)
    if variable == 'torus':
        rewritten = json.loads(write_torus_json(table, c=8))
    else:
        rewritten = json.loads(write_json(table, c=8, q_order=0, h_ext=(0,)*4, hbar_ext=(0,)*4,
                                          spin=0, variable=variable))
    for key in ['verified_digits', 'verified_digits_largest_entry', 'target_digits',
                'q_truncation_digits', range_key]:
        assert rewritten[key] == doc[key]
    if variable == 'torus':
        assert rewritten['fit_digits'] == '27.0'
        assert rewritten['precision_digits'] == '32.0'


def test_chiral_coordinate_relabeling_keeps_precision_and_verification():
    from virasoro_bootstrap.chiral_from_blocks import _relabel_chiral_as_zzb

    doc = _document(True)
    doc.update(precision_actual=80, verified_digits='45.0', target_digits=40,
               verified_delta_range=['0.5', '20.5'], q_truncation_digits='50.0')
    source = serialized_table_from_json(doc)
    relabeled = _relabel_chiral_as_zzb(source)
    assert relabeled.working_dps == 80
    assert relabeled.verified_digits == 45
    assert relabeled.target_digits == 40
    assert relabeled.verified_range == source.verified_range
    assert relabeled.q_truncation_digits == 50


@pytest.mark.parametrize('field', ['lambda', 'precision_actual', 'precision_input', 'target_digits'])
@pytest.mark.parametrize('bad', [True, 2.5, -1, '40'])
def test_serialized_integer_fields_are_not_silently_coerced(field, bad):
    doc = _document(True)
    doc[field] = bad
    with pytest.raises(ValueError, match=field):
        serialized_table_from_json(doc)


@pytest.mark.parametrize('field', ['index_names', 'sample_block_values_index_names'])
def test_serialized_axis_labels_must_match_the_payload(field):
    doc = _document(True)
    doc[field] = ['sample_point', 'coordinate_index', 'deriv']
    with pytest.raises(ValueError, match=field):
        serialized_table_from_json(doc)


@pytest.mark.parametrize('spelling', ['fraction', 'decimal', 'mpf'])
def test_serialized_evaluation_does_not_round_an_explicit_offset_onto_a_pole(spelling):
    from fractions import Fraction
    import mpmath as mp
    from virasoro_blocks.serialization import serialize_deriv_tables

    with mp.workdps(100):
        delta = Fraction(1) + Fraction(1, 10**80)
        if spelling == 'decimal':
            delta = '1.' + '0'*79 + '1'
        elif spelling == 'mpf':
            delta = mp.mpf('1.' + '0'*79 + '1')
        result = serialize_deriv_tables({'zzb': _exact_table(root=1)})
        expected = result.evaluate('zzb', delta)[0][0]
    with mp.workdps(15):
        value = result.evaluate('zzb', delta)[0][0]
        assert abs(value / expected - 1) < mp.mpf('1e-14')
        assert mp.mp.dps == 15


def test_direct_serialization_retains_the_precision_of_its_arithmetic():
    import mpmath as mp
    from virasoro_blocks.serialization import serialize_deriv_tables

    with mp.workdps(80):
        serialized = serialize_deriv_tables({'zzb': _exact_table()})
    with mp.workdps(15):
        assert serialized.working_dps == 80
        doc = json.loads(write_json(serialized, c=8, q_order=0,
                                    h_ext=(0,)*4, hbar_ext=(0,)*4, spin=0))
        assert doc['precision_actual'] == 80
        assert mp.mp.dps == 15


@pytest.mark.parametrize('bad', [1+2j, float('nan'), float('inf')])
def test_serializer_refuses_nonreal_or_nonfinite_shift(bad):
    from virasoro_blocks.serialization import serialize_deriv_tables

    with pytest.raises(ValueError):
        serialize_deriv_tables({'zzb': _exact_table()}, shift=bad)


def test_file_without_precision_retains_the_callers_parsing_precision():
    import mpmath as mp

    with mp.workdps(80):
        restored = serialized_table_from_json(_document(True))
    with mp.workdps(15):
        assert restored.working_dps == 80
        doc = json.loads(write_json(restored, c=8, q_order=0, h_ext=(0,)*4, basis='chiral_F'))
        assert doc['precision_actual'] == 80


@pytest.mark.parametrize("basis, projection", [
    ("unknown_basis", None), ("scalar_G", "+"),
    ("parity_projected_B", None), ("parity_projected_B", "invalid"),
    ("global_parity_projected_B", None), ("ns_parity_projected_B", None),
    ("ramond_parity_projected_B", None), ("torus_G", "+"),
])
def test_reader_rejects_inconsistent_basis_metadata(basis, projection):
    coord = "vvb" if basis == "torus_G" else "zzb"
    doc = {"coordinates": [coord], "lambda": 0, "basis": basis,
           "parity_projection": projection, "spin": 0,
           "exp_base": "0.5", "variable_minus_x": "0", "pole_list_x": [],
           "derivs": [[[["1"]]]]}
    with pytest.raises(ValueError, match="basis|parity_projection"):
        serialized_table_from_json(doc)


@pytest.mark.parametrize("shift", ["(1+2j)", "(1+1e-80j)"])
def test_reader_rejects_nonreal_positivity_anchor(shift):
    doc = _document(True)
    doc["variable_minus_x"] = shift
    with pytest.raises(ValueError, match="variable_minus_x.*real"):
        serialized_table_from_json(doc)


def test_reader_accepts_real_complex_notation_for_positivity_anchor():
    doc = _document(True)
    doc["variable_minus_x"] = "(1+0j)"
    assert serialized_table_from_json(doc).shift == 1


@pytest.mark.parametrize('coord', ['zzb', 'yyb', 'qqb'])
def test_offdiagonal_roundtrip_with_explicit_physical_context(coord):
    doc = {'coordinates': [coord], 'lambda': 2, 'basis': 'scalar_G', 'spin': 0,
           'z_center': '0.4', 'zbar_center': '0.3', 'exp_base': '0.5',
           'variable_minus_x': '1', 'pole_list_x': ['-1'], 'precision_actual': 40}
    pairs = kept_pairs(coord, 2, basis='scalar_G', off_diagonal=True)
    doc['derivs'] = [_derivatives_for_pairs(pairs)]
    restored = serialized_table_from_json(doc)
    rewritten = json.loads(write_json(restored, c=8, h_ext=[0]*4, hbar_ext=[0]*4,
        spin=0, q_order=4, basis='scalar_G', z_center=doc['z_center'],
        zbar_center=doc['zbar_center']))
    again = serialized_table_from_json(rewritten)
    for delta in [1, 2, 10]:
        first, second = restored.evaluate(coord, delta), again.evaluate(coord, delta)
        for m, n in pairs:
            assert first[m][n] == second[m][n]
    assert rewritten['z_center'] == '0.4'
    assert rewritten['zbar_center'] == '0.3'
