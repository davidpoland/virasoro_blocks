"""A sample scan cannot substitute for an undecided polynomial check."""
import json
from types import SimpleNamespace

import mpmath as mp
import pytest

from virasoro_bootstrap.certification import certify_dual_from_sdpb_output
from virasoro_bootstrap.positivity import block_positivity
from virasoro_bootstrap.sdp_writer import write_sdp_directory


def test_certification_preserves_negative_contraction_residuals(tmp_path):
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_bootstrap.certification import (
        _direct_sdp_block_detail, _sample_value_channel_detail,
        certify_dual_functional, load_y_vector,
    )

    with mp.workdps(180):
        height = mp.mpf(2)**250
        epsilon = 1/height
        smaller = height-1
        dual_text = mp.nstr(1-epsilon, 150)
        height_text, smaller_text = mp.nstr(height, 150), mp.nstr(smaller, 150)
        one = Polynomial([height], field=mp.mpf)
        stored = SerializedDerivTable(0, mp.mpf(1), (), ('zzb',),
            {'zzb': [[one], [one]]}, working_dps=180)
    (tmp_path/'y.txt').write_text('2 1\n-1\n'+dual_text+'\n')

    class RoundedBlock:
        def __init__(self):
            self.precisions = []

        def evaluate(self, coordinate, delta):
            self.precisions.append(mp.mp.dps)
            return [[+height], [+smaller]]

    with mp.workdps(40):
        y = load_y_vector(tmp_path)
        assert y[1] != 1
        channel = SimpleNamespace(prefactored=stored, delta_min=0, ell=0)
        ok, details = certify_dual_functional(y, [channel], [(0, 0), (1, 0)],
                                             n_test_points=1, delta_range=0)
        assert not ok and details[0]['min_value'] == pytest.approx(-1)
        fresh = RoundedBlock()
        channel.prefactored = fresh
        ok, details = certify_dual_functional([-1, 1], [channel], [(0, 0), (1, 0)],
                                             n_test_points=2, delta_range=1)
        assert not ok and details[0]['min_value'] == -1
        assert fresh.precisions[0] == 40 and fresh.precisions[-1] > 100
        assert len(fresh.precisions) == 3  # reuse the measured precision on the next point
        (tmp_path/'block_data_0.json').write_text(json.dumps({'c': [smaller_text], 'B': [[height_text]]}))
        direct = _direct_sdp_block_detail([mp.mpf(1)], tmp_path, 0)
        samples = _sample_value_channel_detail([mp.mpf(1), mp.mpf(-1)],
            {'samplePoints': ['0']}, [[smaller_text, height_text]], block_index=0)
        assert not direct['passed'] and direct['min_value'] == -1
        assert not samples['passed'] and samples['min_value'] == -1
        assert mp.mp.dps == 40


def test_certification_diagnostics_preserve_extreme_and_nonfinite_values(tmp_path, capsys):
    from virasoro_bootstrap.certification import (
        _direct_sdp_block_detail, _json_safe_positivity, _normalization_violation,
        _pmp_channel_detail, _sample_value_channel_detail, certify_dual_functional,
    )
    from virasoro_bootstrap.hpc.recertify import _negative_where, _summarize

    with mp.workdps(40):
        for token in ('1e500', '-1e-500', 'nan', 'inf'):
            value = mp.mpf(token)
            channel = SimpleNamespace(ell=0, delta_min=0,
                prefactored=SimpleNamespace(evaluate=lambda coordinate, delta: [[value]]))
            passed, details = certify_dual_functional([1], [channel], [(0, 0)],
                                                      n_test_points=1, delta_range=0)
            assert passed is (token == '1e500')
            (tmp_path/'block_data_0.json').write_text(json.dumps({'c': [token], 'B': [[]]}))
            entry = {'samplePoints': ['0'], 'polynomials': [[[[token]]]],
                     'DampedRational': {'constant': '1', 'base': '0.5', 'poles': []}}
            all_details = [details[0], _direct_sdp_block_detail([], tmp_path, 0),
                _sample_value_channel_detail([1], entry, [[token]], block_index=0),
                _pmp_channel_detail([1], entry, block_index=0, n_test_points=1, delta_range=0)]
            for detail in all_details:
                restored = json.loads(json.dumps(detail, allow_nan=False))
                shown = restored['min_value']
                assert isinstance(shown, str)
                if not mp.isfinite(value):
                    assert not mp.isfinite(mp.mpf(shown))
                    assert not restored['passed']
                else:
                    assert mp.mpf(shown) == value
                assert shown in _negative_where(restored)
                assert _summarize(restored['passed'], [restored],
                    {'probe': 'lower', 'probe_value': None}) is restored['passed']
                output = capsys.readouterr().out
                if token == '-1e-500':
                    assert 'NEGATIVE' in output and shown in output
                elif token in ('nan', 'inf'):
                    assert 'NOT PROVEN' in output and 'NEGATIVE' not in output
            root_detail = _json_safe_positivity({'first_negative_value': value, 'roots': [value]})
            assert isinstance(root_detail['first_negative_value'], str)
            json.dumps(root_detail, allow_nan=False)
            problem = SimpleNamespace(pmp_build=SimpleNamespace(doc={'normalization': [token]}))
            violation = _normalization_violation([1], problem, tmp_path)
            assert isinstance(violation['normalization_value'], str)
            json.dumps(violation, allow_nan=False)
        for pole in ('1e500', '1e-500', 'nan', 'inf'):
            entry['DampedRational']['poles'] = [pole]
            result = block_positivity(entry, [1])
            assert result['reason'] == 'damped_rational_pole_not_negative'
            shown = _json_safe_positivity(result)['poles'][0]
            assert isinstance(shown, str)
            if mp.isfinite(mp.mpf(pole)):
                assert mp.mpf(shown) == mp.mpf(pole)
            json.dumps(_json_safe_positivity(result), allow_nan=False)
    out = tmp_path/'out'
    out.mkdir()
    (out/'y.txt').write_text('1 1\n1\n')
    problem = SimpleNamespace(spec=SimpleNamespace(dps=40), F_pairs=[(0, 0)],
        spin_channels=[], pmp_build=SimpleNamespace(doc={'normalization': ['1e500']}))
    with pytest.warns(UserWarning, match='normalization .*1.0e\\+500'):
        passed, details = certify_dual_from_sdpb_output(problem, out, q_gate='off')
    assert not passed
    json.dumps(details, allow_nan=False)
    sdp = tmp_path / 'sdp'
    sdp.mkdir()
    (sdp / 'control.json').write_text('{}')
    (sdp / 'normalization.json').write_text(json.dumps({'normalization': ['1', '0']}))
    (sdp / 'block_data_0.json').write_text(json.dumps({'c': ['1'], 'B': [['0']]}))
    (out / 'y.txt').write_text('1 1\n0\n')
    for exponent in (500, -500):
        with mp.workdps(40):
            # Positive at the sampled point, but negative between two roots
            # whose locations cannot be represented by a nonzero finite float.
            scale = mp.mpf(10) ** exponent
            entry = {'polynomials': [[[[mp.nstr(2*scale**2, 40), mp.nstr(-3*scale, 40), '1'], ['0']]]],
                'DampedRational': {'constant': '1', 'base': '.5', 'poles': []}}
        channel = SimpleNamespace(ell=0, prefactored=SimpleNamespace(shift=0))
        problem = SimpleNamespace(spec=SimpleNamespace(dps=40), F_pairs=[(0, 0)], spin_channels=[channel],
            pmp_metadata={'moment': {'kind': 'normalized'}}, pmp_build=SimpleNamespace(doc={
                'objective': ['0', '0'], 'normalization': ['1', '0'],
                'PositiveMatrixWithPrefactorArray': [entry]}))
        passed, details = certify_dual_from_sdpb_output(problem, out, q_gate='off', warn=False)
        assert not passed
        stored = json.loads(json.dumps(details, allow_nan=False))[0]
        assert stored['first_negative_value'] == stored['exact_positivity']['first_negative_value']
        assert stored['min_value'] > 0
        assert not _summarize(False, [stored], {'probe': 'lower', 'probe_value': None})
        output = capsys.readouterr().out
        assert 'NEGATIVE' in output and stored['first_negative_value'] in output
        for name in ('first_negative_x', 'first_negative_delta'):
            assert isinstance(stored[name], str) and mp.isfinite(mp.mpf(stored[name]))
            assert stored[name] == stored['exact_positivity'][name]
            with mp.workdps(40):
                assert abs(mp.mpf(stored[name])/scale-mp.mpf('1.5')) < mp.mpf('1e-15')
    # A later NaN minimum must not erase an earlier finite negative witness.
    (sdp / 'block_data_0.json').write_text(json.dumps({'c': ['-1e-500', 'nan'], 'B': [[], []]}))
    detail = _direct_sdp_block_detail([], sdp, 0)
    assert detail['min_value'] == 'nan'
    assert not _summarize(False, [detail], {'probe': 'lower', 'probe_value': None})
    output = capsys.readouterr().out
    assert 'NEGATIVE' in output and 'sample 0' in output and detail['first_negative_value'] in output


def test_certification_normalization_keeps_a_small_eliminated_component(tmp_path):
    from virasoro_bootstrap.certification import _normalization_violation, _reconstruct_full_y

    out, sdp = tmp_path/'out', tmp_path/'sdp'
    out.mkdir()
    sdp.mkdir()
    (sdp/'normalization.json').write_text(json.dumps({'normalization': ['1', '1']}))
    problem = SimpleNamespace(pmp_build=SimpleNamespace(doc={'normalization': ['1', '1']}))
    with mp.workdps(180):
        height = mp.mpf(2)**250
        epsilon = 1/height
        free = 1-epsilon
        normalized = [height, 1-height]
    with mp.workdps(40):
        full = _reconstruct_full_y([free], out)
        assert abs(full[0]/epsilon-1) < mp.mpf('1e-35')
        assert _normalization_violation(full, problem, out) is None
        assert _normalization_violation(normalized, problem, out) is None
        # Legacy solver files carry their rounded normalization values.
        # Comparing them must not round a nearly tied magnitude again.
        token = '1.' + '0'*69 + '1'
        for sign in (1, -1):
            raw = ['1', token if sign > 0 else '-'+token]
            (sdp/'normalization.json').write_text(json.dumps({'normalization': raw}))
            full = _reconstruct_full_y([mp.mpf(2)], out)
            assert full[0] == 2 and full[1] == -sign
        # Our writer also records the chosen pivot and construction precision:
        # at 40 digits the original coefficients tied, even if the reader can
        # subsequently resolve their input labels at higher precision.
        (sdp/'normalization.json').write_text(json.dumps({'normalization': ['1', token],
            'eliminated_index': 0, 'eval_dps': 40}))
        assert _reconstruct_full_y([mp.mpf(2)], out) == [-1, 2]
        # Older direct-writer runs retain the original precision in their log.
        (sdp/'normalization.json').write_text(json.dumps({'normalization': ['1', token]}))
        (tmp_path/'pmp2sdp.log').write_text('sdp_writer (eval_dps=40)')
        assert _reconstruct_full_y([mp.mpf(2)], out) == [-1, 2]
        assert mp.mp.dps == 40


def _entry(poles, coeffs):
    return {"polynomials": [[[coeffs, ["0"]]]],
            "DampedRational": {"constant": "1", "base": "0.5", "poles": poles}}


@pytest.mark.parametrize("poles, coeffs, positive", [
    ([], ["0.0101", "-0.201", "1"], False),
    (["0"], ["1", "1"], True),
    (["0", "0"], ["0.0101", "-0.201", "1"], False),
])
def test_positivity_decides_with_boundary_poles(poles, coeffs, positive):
    from fractions import Fraction

    with mp.workdps(50):
        entry = _entry(poles, coeffs)
        result = block_positivity(entry, [mp.mpf(1), mp.mpf(0)])
        assert result["passed"] is positive
        entry['DampedRational'] = {'constant': Fraction(1, 3), 'base': Fraction(1, 2),
                                  'poles': [Fraction(p) for p in poles]}
        assert block_positivity(entry, [1, 0])["passed"] is positive


@pytest.mark.parametrize("pole", ["0", "nan"])
def test_public_certificate_cannot_pass_negative_or_undecided_polynomial(tmp_path, pole):
    with mp.workdps(50):
        entry = _entry([pole], ["0.0101", "-0.201", "1"])
        xs = [mp.mpf(x) for x in (".05", ".2", ".4")]
        entry.update(samplePoints=[str(x) for x in xs],
                     sampleScalings=[str(mp.mpf(".5") ** x / x) for x in xs])
        doc = {"objective": ["0", "0"], "normalization": ["1", "0"],
               "PositiveMatrixWithPrefactorArray": [entry]}
        write_sdp_directory(doc, tmp_path / "sdp", eval_dps=50)
    problem = SimpleNamespace(spec=SimpleNamespace(dps=50), F_pairs=[(0, 0)], spin_channels=[],
                              pmp_build=SimpleNamespace(doc=doc), pmp_metadata={"moment": {"kind": "normalized"}})
    out = tmp_path / "out"
    out.mkdir()
    (out / "y.txt").write_text("1 1\n0\n")
    ok, details = certify_dual_from_sdpb_output(problem, out, q_gate="off", warn=False)
    assert ok is False and details[0]["passed"] is False
    assert details[0]["exact_positivity"]["passed"] is (False if pole == "0" else None)
    json.dumps(details)
    # Missing direct-SDP metadata selects the PMP fallback. The polynomial
    # remains available, so that path must not accept only-positive grid nodes.
    (tmp_path / 'sdp' / 'control.json').unlink()
    ok, details = certify_dual_from_sdpb_output(problem, out, q_gate='off', warn=False,
                                              n_test_points=2, delta_range=30)
    assert not ok and not details[0]['passed']
    assert details[0]['basis'] == 'pmp_polynomial'
    assert details[0]['exact_positivity']['passed'] is (False if pole == '0' else None)
    assert details[0]['certificate_scope'] == 'interpolated_pmp_polynomial'


def test_extra_moment_certification_checks_the_available_polynomial_between_nodes(monkeypatch):
    from virasoro_bootstrap import certification, moments

    entry = _entry([], ['0.0101', '-0.201', '1'])
    entry['samplePoints'] = ['0', '1']
    channel = SimpleNamespace(ell=2, prefactored=SimpleNamespace(shift=0))
    problem = SimpleNamespace(spec=SimpleNamespace(spin_list=(0,), large_spin_twists=()))
    extended = SimpleNamespace(spin_channels=[channel],
        pmp_build=SimpleNamespace(doc={'PositiveMatrixWithPrefactorArray': [entry]}),
        sample_values_by_block={0: [['0.0101', '0'], ['0.40455', '0']]})
    monkeypatch.setattr(moments, 'rebuild_normalized_moment_problem', lambda *args, **kw: extended)
    for builder in (certification._certify_normalized_moment_extra_pmp_blocks,
                    certification._certify_normalized_moment_extra_sample_blocks):
        options = {'n_test_points': 2, 'delta_range': 30} if builder.__name__.endswith('pmp_blocks') else {}
        ok, details = builder([1, 0], problem, extra_spins=1, extra_twists=0, **options)
        assert not ok and not details[0]['passed']
        assert details[0]['extra'] and details[0]['n_negative'] == 0
        assert details[0]['exact_positivity']['passed'] is False
        assert details[0]['first_negative_x'] == pytest.approx(.1005)
        assert details[0]['certificate_scope'] == 'interpolated_pmp_polynomial'


@pytest.mark.parametrize("data", [{}, {"c": ["inf"], "B": [[]]}])
def test_direct_sample_scan_refuses_missing_or_nonfinite_data(tmp_path, capsys, data):
    from virasoro_bootstrap.certification import _direct_sdp_block_detail
    from virasoro_bootstrap.hpc.recertify import _summarize

    (tmp_path / "block_data_0.json").write_text(json.dumps(data))
    detail = _direct_sdp_block_detail([], tmp_path, 0)
    assert detail["passed"] is False
    assert not _summarize(False, [detail], {"probe": "lower", "probe_value": None})
    output = capsys.readouterr().out
    assert "NOT PROVEN" in output and "NEGATIVE" not in output


@pytest.mark.parametrize("n_negative", [0, 1])
def test_undecided_certificate_is_reported_as_not_proven_unless_sample_is_negative(capsys, n_negative):
    from virasoro_bootstrap.hpc.recertify import _summarize

    details = [{"passed": False, "block_index": 0, "n_negative": n_negative,
                "min_value": -1 if n_negative else 1,
                "exact_positivity_undecided": "no_full_y_vector"}]
    assert not _summarize(False, details, {"probe": "lower", "probe_value": None})
    output = capsys.readouterr().out
    if n_negative:
        assert "INVALID" in output and "NEGATIVE" in output
    else:
        assert "NOT PROVEN" in output and "NEGATIVE" not in output


@pytest.mark.parametrize("field, bad", [("base", "0"), ("constant", "nan")])
def test_invalid_prefactor_cannot_certify_a_positive_numerator(field, bad):
    from fractions import Fraction

    entry = _entry([], ["1"])
    for value in [bad, Fraction(0) if bad == '0' else mp.mpf(bad), mp.mpc(1, 1), True]:
        entry["DampedRational"][field] = value
        assert block_positivity(entry, [1, 0])["passed"] is None


def test_positivity_refuses_unsupported_constraint_shapes():
    entry = _entry([], ["1"])
    entry["polynomials"] = [[[["1"]], [["0"]]], [[["0"]], [["-1"]]]]
    detail = block_positivity(entry, [1])
    assert detail == {"passed": None, "reason": "non_scalar_polynomial_matrix"}
    entry = _entry([], ['1'])
    for bad in (False, 1, 'invalid'):
        entry['DampedRational'] = bad
        assert block_positivity(entry, [1, 0]) == {
            'passed': None, 'reason': 'invalid_damped_rational_prefactor'}
    for bad in (None, '-1', {}):
        entry['DampedRational'] = {'poles': bad}
        assert block_positivity(entry, [1, 0]) == {
            'passed': None, 'reason': 'invalid_damped_rational_poles'}
    for bad in (None, mp.mpc(1, 1), True):
        entry['DampedRational'] = {'poles': [bad]}
        assert block_positivity(entry, [1, 0])['passed'] is None


@pytest.mark.parametrize("bad", ["matrix", "zero_normalization", "coefficients", "duplicate_nodes",
                                 "sample_columns"])
def test_direct_writer_rejects_invalid_constraints_before_writing(tmp_path, bad):
    entry = _entry([], ["1", "2"])
    entry.update(samplePoints=["0", "1"], sampleScalings=["1", "1"])
    doc = {"objective": ["0", "0"], "normalization": ["1", "0"],
           "PositiveMatrixWithPrefactorArray": [entry]}
    samples = None
    if bad == "matrix":
        entry["polynomials"] = [entry["polynomials"][0] * 2] * 2
    elif bad == "objective":
        doc["objective"].append("0")
    elif bad == "normalization":
        doc["normalization"][0] = "nan"
    elif bad == "zero_normalization":
        doc["normalization"] = ["0", "0"]
    elif bad == "coefficients":
        entry["polynomials"][0][0][0][0] = "inf"
    elif bad == "scalings":
        entry["sampleScalings"].append("1")
    elif bad == "nodes":
        entry["samplePoints"][0] = "-1"
    elif bad == "duplicate_nodes":
        entry["samplePoints"] = ["1", "1"]
    elif bad == "components":
        entry["polynomials"][0][0].append(["0"])
    else:
        samples = {0: [["1", "0"], ["3", "0"]]}
        if bad == "sample_rows":
            samples[0].append(["4", "0"])
        elif bad == "sample_columns":
            samples[0][0].append("0")
        else:
            samples[0][0][0] = "nan"
    destination = tmp_path / "sdp"
    with pytest.raises(ValueError):
        write_sdp_directory(doc, destination, eval_dps=40, sample_values_by_block=samples)
    assert not destination.exists()


@pytest.mark.parametrize("data", [[["1"]]])
def test_clean_samples_must_match_the_serialized_nodes(data):
    from virasoro_bootstrap.certification import _sample_value_channel_detail

    detail = _sample_value_channel_detail([1], {"samplePoints": ["0", "1"]}, data, block_index=0)
    assert detail["passed"] is False


@pytest.mark.parametrize("bad", ["nan"])
def test_sample_and_polynomial_scans_reject_nonfinite_values(bad):
    from virasoro_bootstrap.certification import _pmp_channel_detail, _sample_value_channel_detail

    entry = _entry([], [bad])
    entry["samplePoints"] = ["1"]
    assert not _pmp_channel_detail([1, 0], entry, block_index=0)["passed"]
    assert not _sample_value_channel_detail([1], entry, [[bad]], block_index=0)["passed"]


def _constant_channel(value):
    return SimpleNamespace(ell=0, delta_min=mp.mpf(0),
                           prefactored=SimpleNamespace(evaluate=lambda coord, delta: [[value]]))


@pytest.mark.parametrize("n", [0])
def test_public_certification_rejects_invalid_sample_count(n):
    from virasoro_bootstrap.certification import certify_dual_functional

    with pytest.raises(ValueError, match="n_test_points"):
        certify_dual_functional([1], [_constant_channel(-1)], [(0, 0)], n_test_points=n)


@pytest.mark.parametrize("value", [mp.nan])
def test_public_certification_rejects_nonfinite_blocks(value):
    from virasoro_bootstrap.certification import certify_dual_functional

    assert not certify_dual_functional([1], [_constant_channel(value)], [(0, 0)], n_test_points=2)[0]


@pytest.mark.parametrize("count", [5])
def test_public_scan_stays_in_requested_range_and_includes_threshold(count):
    from virasoro_bootstrap.certification import certify_dual_functional

    points = []
    channel = _constant_channel(1)
    channel.prefactored.evaluate = lambda coord, delta: points.append(delta) or [[1]]
    assert certify_dual_functional([1], [channel], [(0, 0)], n_test_points=count, delta_range=0.5)[0]
    assert min(points) == 0 and max(points) <= mp.mpf("0.5")
    assert len(points) == count


@pytest.mark.parametrize("value", [mp.nan])
def test_public_certification_rejects_invalid_range(value, tmp_path):
    from virasoro_bootstrap.certification import certify_dual_functional

    with pytest.raises(ValueError, match="delta_range"):
        certify_dual_functional([1], [_constant_channel(1)], [(0, 0)], delta_range=value)
    with pytest.raises(ValueError, match="delta_range"):
        certify_dual_from_sdpb_output(None, tmp_path, delta_range=value)


@pytest.mark.parametrize("count", [1, 4])
def test_public_scan_includes_edges_for_positive_range(count):
    from virasoro_bootstrap.certification import certify_dual_functional

    points = []
    channel = _constant_channel(1)
    channel.prefactored.evaluate = lambda coord, delta: points.append(delta) or [[1]]
    ok, details = certify_dual_functional([1], [channel], [(0, 0)], n_test_points=count, delta_range=30)
    assert ok and min(points) == 0 and len(points) == details[0]["n_test_points"] == count
    assert max(points) == (0 if count == 1 else 30)


@pytest.mark.parametrize("y, pairs", [([1], [(0, 0), (0, 1)]), ([], [])])
def test_public_scan_refuses_incomplete_functionals(y, pairs):
    from virasoro_bootstrap.certification import certify_dual_functional

    with pytest.raises(ValueError, match="vector length"):
        certify_dual_functional(y, [_constant_channel(-1)], pairs, n_test_points=2)


@pytest.mark.parametrize("pair", [(1, 0), (0, -1)])
def test_public_scan_refuses_missing_functional_entries(pair):
    from virasoro_bootstrap.certification import certify_dual_functional

    with pytest.raises(ValueError, match="missing functional entry"):
        certify_dual_functional([1], [_constant_channel(1)], [pair], n_test_points=2)


def test_scan_accepts_numpy_count_and_preserves_high_precision_threshold():
    import numpy as np
    from virasoro_bootstrap.certification import certify_dual_functional

    with mp.workdps(50):
        channel = _constant_channel(1)
        channel.delta_min = mp.mpc(mp.mpf(1) / 3, 0)
        points = []
        channel.prefactored.evaluate = lambda coord, delta: points.append(delta) or [[1]]
        assert certify_dual_functional([1], [channel], [(0, 0)], n_test_points=np.int64(2))[0]
        assert points[0] == channel.delta_min


def test_requested_precision_cannot_hide_a_narrow_negative_interval():
    entry = _entry([], ['0.'+'9'*30, '-2', '1'])
    with mp.workdps(15):
        detail = block_positivity(entry, [1, 0], dps=80)
        assert detail['passed'] is False
        assert detail['first_negative_value'] < 0
        assert mp.mp.dps == 15


def test_functional_combination_is_exact_before_sign_decisions():
    from flint import fmpq
    from virasoro_blocks.numeric_input import literal_number
    from virasoro_bootstrap.positivity import _exact_real

    entry = {'polynomials': [[[['1'], ['1.'+'0'*79+'1']]]]}
    with mp.workdps(15):
        detail = block_positivity(entry, [1, -1])
    assert detail['passed'] is False
    assert detail['degree'] == 0
    with mp.workdps(100):
        native = [mp.mpf(0), mp.mpf(2)**250+1, -(mp.mpf(2)**250+1), mp.mpf(2)**-250]
    with mp.workdps(15):
        for value in native:
            exact = literal_number(value)
            assert _exact_real(value) == fmpq(exact.numerator, exact.denominator)


@pytest.mark.parametrize('coefficients', [[], ['0', '0', '0']])
def test_zero_polynomial_is_nonnegative(coefficients):
    assert block_positivity(_entry([], coefficients), [1, 0])['passed'] is True


@pytest.mark.parametrize('roots, positive', [
    ([(0, 3), (-1, 1), (2, 2)], True), ([(1, 1), (2, 1)], False), ([(0, 2), (1, 3)], False),
])
def test_exact_positivity_matches_factored_polynomial_signs(roots, positive):
    from flint import fmpq_poly

    polynomial = fmpq_poly([1])
    for root, multiplicity in roots:
        polynomial *= fmpq_poly([-root, 1])**multiplicity
    assert block_positivity(_entry([], [str(c) for c in polynomial.coeffs()]), [1, 0])['passed'] is positive


def test_close_roots_trigger_refinement_without_losing_the_negative_interval():
    from flint import fmpq, fmpq_poly

    x = fmpq_poly([0, 1])
    polynomial = (x-1) * (x-1-fmpq(1,10**100))
    with mp.workdps(15):
        detail = block_positivity(_entry([], [str(c) for c in polynomial.coeffs()]), [1, 0])
    assert detail['passed'] is False
    assert detail['first_negative_value'] < 0


def _exact_c_certification_problem():
    from fractions import Fraction
    from virasoro_bootstrap.identical_scalar import (
        IdenticalScalarGapSpec, IdenticalScalarSpinChannel,
    )

    spec = IdenticalScalarGapSpec(
        c=Fraction(1) + Fraction(1, 10**60), h_ext=".1", delta_gap="1.1",
        lam=1, spin_list=(0,), q_order=4, dps=30, npoles=2,
        large_spin_lmin=4, large_spin_twists=(".5", "1.5"),
    )
    channel = IdenticalScalarSpinChannel(
        ell=0, delta_min=mp.mpf("1.1"), serialized=None, prefactored=None,
    )
    return SimpleNamespace(spec=spec, spin_channels=[channel], resolved_method="interpolation")


def test_exact_certification_samplers_preserve_deliberate_c_shift():
    from virasoro_bootstrap.certification import (
        _build_exact_block_channels, _build_extra_twist_channels,
    )
    from virasoro_blocks.polyderivs import _NumericBlockSampler
    from virasoro_blocks.numeric_input import literal_number

    with mp.workdps(30):
        problem = _exact_c_certification_problem()
        channels = _build_exact_block_channels(problem)
        assert isinstance(channels[0].prefactored._sampler, _NumericBlockSampler)
        channels += _build_extra_twist_channels(problem, 1)
        for channel in channels:
            assert literal_number(channel.prefactored._sampler.c) == problem.spec.c


@pytest.mark.parametrize("exact_blocks", [True])
def test_extra_spin_certification_preserves_c_at_builder_boundary(monkeypatch, exact_blocks):
    from virasoro_bootstrap.certification import _build_extra_spin_channels
    import virasoro_bootstrap.identical_scalar as scalar
    from virasoro_blocks.numeric_input import literal_number

    class ReachedBuilder(Exception):
        pass

    def check(c, *args, **kwargs):
        assert literal_number(c) == problem.spec.c
        assert kwargs["used_regularize"] is False
        raise ReachedBuilder

    monkeypatch.setattr(scalar, "_build_exact_scalar_tables", check)
    monkeypatch.setattr(scalar, "_build_exact_scalar_table", check)
    with mp.workdps(30):
        problem = _exact_c_certification_problem()
        with pytest.raises(ReachedBuilder):
            _build_extra_spin_channels(problem, 1, exact_blocks=exact_blocks)


@pytest.mark.parametrize("header", ["1 2", "one 1"])
def test_dual_loader_rejects_malformed_matrix_header(tmp_path, header):
    from virasoro_bootstrap.certification import load_y_vector

    (tmp_path / "y.txt").write_text(header + ("\n0\n" if header else ""))
    with pytest.raises(ValueError, match="y.txt.*header"):
        load_y_vector(tmp_path)


def test_zero_length_free_dual_reconstructs_single_normalized_component(tmp_path, monkeypatch):
    from virasoro_bootstrap import certification

    out = tmp_path / "out"
    out.mkdir()
    (out / "y.txt").write_text("0 1\n")
    sdp = tmp_path / "sdp"
    sdp.mkdir()
    (sdp / "normalization.json").write_text(json.dumps({"normalization": ["0.5"]}))
    problem = SimpleNamespace(spec=SimpleNamespace(dps=30), F_pairs=[(1, 0)],
                              spin_channels=["channel"], pmp_build=SimpleNamespace(doc={
                                  "objective": ["0"], "normalization": ["0.5"]}))
    seen = []
    def certify(y, channels, pairs, **kwargs):
        seen.append((y, channels, pairs))
        return True, [{"passed": True}]
    monkeypatch.setattr(certification, "certify_dual_functional", certify)
    ok, details = certification.certify_dual_from_sdpb_output(
        problem, out, q_gate="off", exact_blocks=False, warn=False)
    assert seen == [([mp.mpf(2)], ["channel"], [(1, 0)])]
    assert ok and details[0]["passed"]


@pytest.mark.parametrize("normalized_value", ["0", "1"])
def test_normalization_cancellation_does_not_relax_required_value(tmp_path, capsys, normalized_value):
    from virasoro_bootstrap.certification import _normalization_violation
    from virasoro_bootstrap.hpc.recertify import _summarize

    problem = SimpleNamespace(pmp_build=SimpleNamespace(doc={"normalization": ["1", "1"]}))
    with mp.workdps(60):
        large = mp.mpf("1e20")
        y = [large, mp.mpf(normalized_value) - large]
        result = _normalization_violation(y, problem, tmp_path)
    if normalized_value == "1":
        assert result is None
    else:
        assert result is not None
        assert result["reason"] == "normalization_violated"
        assert not _summarize(False, [result], {"probe": "lower", "probe_value": None})
        output = capsys.readouterr().out
        assert "INVALID functional" in output and "NEGATIVE" not in output
        assert "normalization_violated" in output


@pytest.mark.parametrize("imaginary,passes", [(".05", False), ("1e-50", True)])
def test_sampled_certificate_requires_finite_real_values(capsys, imaginary, passes):
    from virasoro_bootstrap.certification import certify_dual_functional
    from virasoro_bootstrap.hpc.recertify import _summarize

    class Block:
        def evaluate(self, coordinate, delta):
            return [[mp.mpc(sign, mp.mpf(imaginary))]]
    channel = SimpleNamespace(ell=0, delta_min=0, prefactored=Block())
    for dps in (3, 7, 40):
        for sign in (1, -1):
            with mp.workdps(dps):
                ok, details = certify_dual_functional([mp.mpf(1)], [channel], [(0, 0)], n_test_points=3)
            assert ok is (passes and sign == 1)
            assert details[0]["passed"] is ok
            assert _summarize(ok, details, {"probe": "lower", "probe_value": None}) is ok
            output = capsys.readouterr().out
            if not passes:
                assert "NOT PROVEN" in output and "NEGATIVE" not in output
            elif sign < 0:
                assert "INVALID (negative channel)" in output
            else:
                assert "VALID functional" in output


@pytest.mark.parametrize("basis", ["ratio"])
def test_moment_physical_rows_do_not_discard_nonreal_block_values(basis):
    from virasoro_bootstrap.certification import _normalized_moment_physical_row_from_table

    problem = SimpleNamespace(F_pairs=[(0, 0)], pmp_metadata={"moment": {"block_basis": basis}})
    channel = SimpleNamespace(delta_min=0)
    with mp.workdps(40):
        value = _normalized_moment_physical_row_from_table(
            [mp.mpf(1)], problem, channel, 0, [[mp.mpc(1, 1)]], weights=[])
    assert not mp.isfinite(value)


def test_certification_range_does_not_shrink_requested_high_precision_endpoint():
    from virasoro_bootstrap.certification import _certification_delta_range

    with mp.workdps(80):
        requested = mp.mpf(30) + mp.mpf("1e-60")
    with mp.workdps(15):
        actual = _certification_delta_range(requested, SimpleNamespace(c=8, dps=80))
    assert actual == requested


def test_certification_range_keeps_large_finite_endpoints_finite():
    from virasoro_bootstrap.certification import _certification_delta_range

    with mp.workdps(40):
        requested = mp.mpf("1e400")
        actual = _certification_delta_range(requested, SimpleNamespace(c=8, dps=40))
    assert mp.isfinite(actual)
    assert actual == requested
