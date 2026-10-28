"""Serialization preserves the function when shifting or lowering precision."""

from fractions import Fraction
import json

import mpmath as mp
import pytest

from virasoro_blocks.interpolation import DampedRational
from virasoro_blocks.numeric_input import literal_number
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.polyderivs import DeltaDerivTable, InterpolatedDeltaTable
from virasoro_blocks.serialization import SerializedDerivTable, serialize_deriv_tables, serialized_table_from_json
from virasoro_blocks.writers import write_json


HE = (".1", ".2", ".3", ".4")


def test_numeric_file_data_keeps_small_real_coefficient_offsets():
    from virasoro_blocks.writers import write_mathematica

    with mp.workdps(40):
        epsilon = mp.mpf(2)**-130
        value = 1 + epsilon
        table = SerializedDerivTable(0, value, (-value,), ('z',),
            {'z': [[Polynomial([-1, value], field=mp.mpf)]]},
            sample_points_x=(0,), sample_block_values={'z': [[[value]]]})
        metadata = dict(c=32, h_ext=HE, q_order=0, basis='chiral_F')
        doc = json.loads(write_json(table, **metadata))
        restored = serialized_table_from_json(doc)
        # Base and pole labels use 40 digits, coefficients and samples 45.
        # Rounding to those precisions is permitted; trimming the offset is not.
        for actual in (restored.base, -restored.pole_list_x[0]):
            assert actual != 1
            assert abs(actual-value) < mp.mpf('1e-39')
        for actual in (restored.numerators['z'][0][0].coeffs[1],
                       restored.sample_block_values['z'][0][0][0]):
            assert abs((actual - 1)/epsilon - 1) < mp.mpf('1e-5')
        assert abs(restored.evaluate('z', 1)[0][0]/table.evaluate('z', 1)[0][0]-1) < mp.mpf('1e-5')
        text = mp.nstr(value, mp.mp.dps + 5)
        assert text in write_mathematica(table, coordinate='z', **metadata)
        assert mp.mp.dps == 40


def test_tiny_exponential_phase_survives_serialization_and_file_output():
    from virasoro_blocks.modular import TORUS_COORDINATE
    from virasoro_blocks.polyderivs import F_z_poly_table
    from virasoro_blocks.writers import write_mathematica, write_torus_json

    with mp.workdps(40):
        epsilon = mp.mpf(2)**-250
        query = 2**250
        one = Polynomial([1], field=mp.mpf)
        metadata = dict(c=32, h_ext=HE, q_order=0, basis='chiral_F')
        for sign in (1, -1):
            base = mp.mpc(1, sign*epsilon)
            exact = DeltaDerivTable(base, one, [], [[one]])
            serialized = serialize_deriv_tables({'z': exact}, shift=0)
            direct = SerializedDerivTable(0, base, (), ('z',), {'z': [[one]]})
            with mp.workdps(180):
                expected = base**query
            for source in (serialized, direct):
                restored = serialized_table_from_json(json.loads(write_json(source, **metadata)))
                assert '*I' in write_mathematica(source, coordinate='z', **metadata)
                for table in (source, restored):
                    assert abs(table.evaluate('z', query)[0][0]/expected-1) < mp.mpf('1e-37')
            torus = SerializedDerivTable(0, base, (), (TORUS_COORDINATE,), {TORUS_COORDINATE: [[one]]})
            restored = serialized_table_from_json(json.loads(write_torus_json(torus, c=32, q_order=0)))
            assert abs(restored.evaluate(TORUS_COORDINATE, query)[0][0]/expected-1) < mp.mpf('1e-37')
        # The phase also occurs in real block builders at a complex center.
        exact = F_z_poly_table(HE, 0, c=32, q_order=0, center=mp.mpc('.5', epsilon))
        serialized = serialize_deriv_tables({'z': exact})
        assert mp.im(exact.exp_base) != 0
        assert mp.im(serialized.base) == mp.im(exact.exp_base)
        assert mp.mp.dps == 40


@pytest.mark.parametrize("gap", [".35", "1/3", "1e-60", Fraction(7, 20)])
@pytest.mark.parametrize("chiral", [False, True])
def test_cli_positivity_anchor_retains_exact_input_labels(gap, chiral):
    from virasoro_blocks.table_backends import positivity_anchor

    spin = Fraction(10**80 * 2 + 1, 2)
    expected = literal_number(gap) + (0 if chiral else spin)
    for dps in (5, 40):
        with mp.workdps(dps):
            assert positivity_anchor(gap, spin, chiral) == expected
            assert mp.mp.dps == dps


@pytest.mark.parametrize("spelling", ["fraction", "text", "mpf"])
def test_interpolation_sample_offsets_survive_serialization_at_lower_precision(spelling):
    point = Fraction(1, 2)+Fraction(1, 10**70)
    with mp.workdps(180):
        if spelling == "text":
            point = f"{point.numerator}/{point.denominator}"
        elif spelling == "mpf":
            point = mp.convert(point)
        expected = literal_number(point)-Fraction(1, 2)
    with mp.workdps(40):
        table = InterpolatedDeltaTable(DampedRational(1, (), 1), Fraction(1, 2), (point,),
                                       [[Polynomial([1], field=mp.mpf)]])
        serialized = serialize_deriv_tables({"zzb": table})
        assert serialized.sample_points_x == (expected,)
        assert mp.mp.dps == 40


@pytest.mark.parametrize("interpolated", [False, True])
def test_low_precision_serialization_refuses_materially_different_prefactors(interpolated):
    with mp.workdps(5):
        if interpolated:
            first = InterpolatedDeltaTable(DampedRational(1, (), ".5"), 0, (), [[Polynomial([1])]])
            second = InterpolatedDeltaTable(DampedRational(1, (), ".6"), 0, (), [[Polynomial([1])]])
        else:
            first = DeltaDerivTable(mp.mpf(".5"), Polynomial([1]), [], [[Polynomial([1])]])
            second = DeltaDerivTable(mp.mpf(".6"), Polynomial([1]), [], [[Polynomial([1])]])
        with pytest.raises(ValueError, match="disagree"):
            serialize_deriv_tables({"zzb": first, "qqb": second})
        assert mp.mp.dps == 5


def test_default_json_rewrites_preserve_payload_text_and_recorded_precision():
    from virasoro_blocks.modular import TORUS_COORDINATE
    from virasoro_blocks.writers import write_mathematica, write_torus_json

    with mp.workdps(50):
        coefficients = [mp.mpf(mp.pi), mp.mpc(mp.e, '-1e-80'), mp.mpc(0, '1e-90')]
        base, pole = mp.mpc('.7', '1e-80'), mp.mpf('-1') / 3
        endpoints = (mp.mpf('1e-30'), mp.mpf(mp.pi))
        for coordinate in ('z', 'zzb', TORUS_COORDINATE):
            table = SerializedDerivTable(0, base, (pole,), (coordinate,),
                {coordinate: [[Polynomial(coefficients)]]}, working_dps=50,
                sample_points_x=(1, 2),
                sample_block_values={coordinate: [[[coefficients[0]]], [[coefficients[1]]]]},
                verified_digits=20, verified_range=endpoints,
                metadata={'verified_delta_range': endpoints})
            if coordinate == TORUS_COORDINATE:
                writer, metadata = write_torus_json, dict(c=32, q_order=0)
            else:
                writer = write_json
                metadata = dict(c=32, h_ext=HE, hbar_ext=HE, q_order=0)
                if coordinate == 'z':
                    metadata['basis'] = 'chiral_F'
            expected = json.loads(writer(table, **metadata))
            for caller_dps in (15, 100, 30, 50):
                with mp.workdps(caller_dps):
                    table = serialized_table_from_json(expected)
                    assert json.loads(writer(table, **metadata)) == expected
                    assert table.working_dps == expected['precision_actual']
                    if coordinate != TORUS_COORDINATE:
                        text = write_mathematica(table, coordinate=coordinate, **metadata)
                        assert '*I' in text
                        assert expected['exp_base'][1:].split('-')[0].split('+')[0] in text
                    assert mp.mp.dps == caller_dps


def test_loaded_table_edits_and_explicit_formatting_do_not_reuse_stale_text():
    from dataclasses import replace
    import pickle

    metadata = dict(c=32, h_ext=HE, q_order=0, basis='chiral_F')
    with mp.workdps(50):
        source = SerializedDerivTable(0, mp.mpf(1), (), ('z',),
            {'z': [[Polynomial([mp.pi])]]}, working_dps=50,
            sample_points_x=(1,), sample_block_values={'z': [[[mp.mpf(mp.e)]]]})
        original = json.loads(write_json(source, **metadata))
        restored = serialized_table_from_json(original)
        for clone in (replace(restored), pickle.loads(pickle.dumps(restored))):
            assert json.loads(write_json(clone, **metadata)) == original
        formatted = json.loads(write_json(restored, sig_figs=12, **metadata))
        assert formatted['derivs'][0][0][0] == mp.nstr(mp.pi, 12)
        explicit = json.loads(write_json(restored, precision_actual=20, **metadata))
        assert explicit['precision_actual'] == 20
        assert explicit['derivs'] == original['derivs']
    with mp.workdps(120):
        changed = 1 + mp.mpf('1e-80')
        restored.numerators['z'][0][0] = Polynomial([changed])
        restored.sample_block_values['z'][0][0][0] = changed
        restored.base = mp.mpf('.5') + mp.mpf('1e-90')
    with mp.workdps(15):
        updated = json.loads(write_json(restored, **metadata))
        assert updated['derivs'] != original['derivs']
        assert updated['sample_block_values'] != original['sample_block_values']
        reread = serialized_table_from_json(updated)
        assert json.loads(write_json(reread, **metadata)) == updated
        assert mp.mp.dps == 15


