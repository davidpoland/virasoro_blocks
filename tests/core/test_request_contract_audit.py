"""Public requests must not silently replace or ignore explicit inputs."""
from dataclasses import replace
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks.api import (
    BlockRequest, NumericalPolicy, RequestValidationError, build_table_request,
    evaluate_block_request, validate_request,
)
from virasoro_blocks.requests import _integer_field


# The type contract is shared by the CFT, bootstrap, and task validators.
# Callers test their own bounds and early returns rather than this full matrix.
@pytest.mark.parametrize('value', [True, np.bool_(False), '3', 1.5, Fraction(3, 2),
                                 mp.nan, mp.inf, None, 3j, object()])
def test_shared_integer_validator_rejects_noninteger_inputs(value):
    with pytest.raises(RequestValidationError, match='count'):
        _integer_field('count', value)


@pytest.mark.parametrize('value', [7, 7., Fraction(7), np.int64(7), mp.mpf(7), -3])
def test_shared_integer_validator_normalizes_integral_inputs(value):
    actual = _integer_field('count', value)
    assert type(actual) is int and actual == value


@pytest.mark.parametrize("family", ["torus-character", "torus-onepoint"])
@pytest.mark.parametrize("basis", ["scalar_G", "ordered_B", "parity_projected_B", "chiral_F",
                                   "torus-character_scalar_G", "torus-onepoint_chiral_F"])
def test_torus_rejects_sphere_basis_aliases(family, basis):
    request = BlockRequest(operation="evaluate", surface="torus", family=family,
                           basis=basis, coordinate="vvb")
    with pytest.raises(RequestValidationError, match="basis"):
        validate_request(request)


def _point_request(**fields):
    return BlockRequest(operation="point-table", family="global", coordinate="z",
                        h_ext=(0, 0, 0, 0), exchange_h=2,
                        policy=NumericalPolicy(precision=20, lambda_order=1), **fields)


@pytest.mark.parametrize("field, value", [("hbar_ext", (1,) * 4), ("delta_ext", (2,) * 4),
                                         ("spin_ext", (0,) * 4), ("exchange_delta", 7),
                                         ("exchange_hbar", 1), ("exchange_spin", 0),
                                         ("center_bar", "0.4"), ("zbar_shift", 1)])
def test_implicit_chiral_point_table_refuses_unused_fields(field, value):
    with pytest.raises(RequestValidationError, match=field):
        build_table_request(_point_request(**{field: value}))


@pytest.mark.parametrize("field, value", [("npoles", 5), ("npoles", "all"),
                                         ("pole_selection", "contribution_optimal")])
def test_point_table_refuses_interpolation_controls(field, value):
    request = _point_request()
    request = replace(request, policy=replace(request.policy, **{field: value}))
    with pytest.raises(RequestValidationError, match=field):
        build_table_request(request)


def test_point_table_refuses_unused_spin_keyword():
    for spin in (2, Fraction(1, 2)):
        with pytest.raises(RequestValidationError, match="spin"):
            build_table_request(_point_request(), spin=spin)
    assert build_table_request(_point_request(), spin=Fraction(0)).coordinates == ('z',)


def test_table_spin_rationals_follow_the_same_contract_as_native_spins():
    request = BlockRequest(
        operation='table', family='global', coordinate='zzb', delta_ext=('.2',)*4,
        policy=NumericalPolicy(precision=20, lambda_order=0, order=2, npoles=1),
    )
    for spin in (0, 1):
        expected = build_table_request(request, spin=spin)
        actual = build_table_request(request, spin=Fraction(spin))
        assert actual.evaluate('zzb', 3) == expected.evaluate('zzb', 3)
    ordered = replace(request, basis='ordered_B', delta_ext=None,
                      h_ext=('.2',)*4, hbar_ext=('.1',)*4)
    expected = build_table_request(ordered, spin=.5)
    actual = build_table_request(ordered, spin=Fraction(1, 2))
    assert actual.evaluate('zzb', 3) == expected.evaluate('zzb', 3)


def test_two_sided_point_table_refuses_chiral_exchange_label():
    request = replace(_point_request(), coordinate="zzb", hbar_ext=(0,) * 4, exchange_delta=3)
    with pytest.raises(RequestValidationError, match="exchange_h"):
        build_table_request(request)


def test_valid_chiral_fallback_still_builds():
    assert build_table_request(_point_request()).coordinates == ("z",)
    assert build_table_request(_point_request(coordinates=iter(()))).coordinates == ("z",)
    expected = build_table_request(_point_request(coordinates=("z", "y")))
    for source in (["z", "y"], iter(("z", "y"))):
        request = _point_request(coordinates=source)
        assert request.coordinates == ("z", "y")
        if isinstance(source, list):
            source[:] = ["zzb"]
        # Read-only validation must leave a repeatable request for later use.
        validate_request(request)
        validate_request(request)
        actual = build_table_request(replace(request))
        assert actual.coordinates == expected.coordinates
        for coordinate in expected.coordinates:
            assert actual.evaluate(coordinate, 2) == expected.evaluate(coordinate, 2)
    invalid = _point_request(coordinates=iter(("unsupported",)))
    for _ in range(2):
        with pytest.raises(RequestValidationError, match="unsupported"):
            validate_request(invalid)


@pytest.mark.parametrize("basis", ["auto", "torus_G"])
def test_valid_torus_basis_still_evaluates(basis):
    request = BlockRequest(operation="evaluate", family="torus-character", surface="torus",
                           basis=basis, coordinate="vvb", central_charge=8, exchange_delta=2,
                           point={"v": 0, "vbar": 0})
    assert evaluate_block_request(request, q_gate="off").metadata["basis"] == "torus_G"


def test_requests_snapshot_point_mappings_and_external_parity():
    point = {'v': mp.mpc(0), 'vbar': mp.mpc(0)}
    request = BlockRequest(operation='evaluate', family='torus-character', surface='torus',
                           coordinate='vvb', central_charge=8, exchange_delta=2, point=point)
    expected = evaluate_block_request(request, q_gate='off').as_dict()
    point['v'] = 1
    assert evaluate_block_request(request, q_gate='off').as_dict() == expected
    empty = {}
    incomplete = replace(request, point=empty)
    empty.update(v=0, vbar=0)
    with pytest.raises(RequestValidationError, match='torus evaluation needs'):
        evaluate_block_request(incomplete, q_gate='off')
    for invalid in (None, [], 1):
        with pytest.raises(RequestValidationError, match='point must be a mapping'):
            replace(request, point=invalid)

    parity = ['1'] * 4
    projected = BlockRequest(
        operation='evaluate', family='global', basis='parity_projected_B', parity='+',
        parity_external=parity, coordinate='zzb', h_ext=(0,) * 4, hbar_ext=(0,) * 4,
        exchange_delta=3, point={'z': '.3', 'zbar': '.4'},
        policy=NumericalPolicy(precision=20),
    )
    expected = evaluate_block_request(projected, q_gate='off').as_dict()
    parity[0] = '-1'
    assert evaluate_block_request(projected, q_gate='off').as_dict() == expected
    assert evaluate_block_request(replace(projected), q_gate='off').as_dict() == expected
    streamed = replace(projected, parity_external=iter(('1',) * 4))
    assert evaluate_block_request(streamed, q_gate='off').as_dict() == expected


def test_point_cli_refuses_duplicate_centers(tmp_path, capsys):
    from virasoro_blocks.cli.point_table import main

    with pytest.raises(SystemExit) as exc:
        main(["--family", "global", "--h-ext", "0", "0", "0", "0", "--coordinates", "z",
              "--exchange-h", "2", "--lambda", "0", "--center", "z=0.3", "--center", "z=0.4",
              "--precision", "20", "-o", str(tmp_path)])
    assert exc.value.code == 2
    assert "duplicate" in capsys.readouterr().err and not list(tmp_path.iterdir())


@pytest.mark.parametrize('name,value', [
    ('precision', float('inf')), ('order', float('-inf')),
    ('lambda_order', float('nan')), ('npoles', 2j),
    ('target_digits', float('inf')), ('max_precision', float('nan')),
])
def test_invalid_policy_integers_raise_validation_errors(name, value):
    with pytest.raises(RequestValidationError, match=name):
        NumericalPolicy(**{name: value})


def test_policy_precision_cannot_be_none():
    with pytest.raises(RequestValidationError, match='precision'):
        NumericalPolicy(precision=None)


@pytest.mark.parametrize('spelling', ['inf', '-inf', 'nan'])
def test_nonfinite_numpy_scalars_cannot_bypass_request_validation(spelling):
    import numpy as np

    with pytest.raises(RequestValidationError, match='finite'):
        BlockRequest(operation='evaluate', central_charge=np.float32(spelling))


def test_unsupported_numeric_objects_fail_at_the_request_boundary():
    with pytest.raises(RequestValidationError, match='finite number'):
        BlockRequest(operation='evaluate', central_charge=object())
    for field, value in (
        ('central_charge', True), ('exchange_delta', False),
        ('exchange_h', np.bool_(True)), ('h_ext', (0, None, 0, 0)),
        ('hbar_ext', (0, True, 0, 0)), ('h_ext', 1),
        ('delta_ext', '1234'), ('spin_ext', {0: 1, 1: 1, 2: 1, 3: 1}),
        ('point', {'z': None}), ('point', {'z': True}),
        ('point', {1: '.3', 2j: '.4'}), ('coordinates', ('z', None)),
        ('parity_external', (1, 1, 1, 1)),
    ):
        with pytest.raises(RequestValidationError, match=field):
            BlockRequest(operation='evaluate', **{field: value})
    # None still denotes an omitted optional field, not a numeric component.
    request = BlockRequest(operation='evaluate', central_charge=None, h_ext=None,
                           exchange_h=None, coordinates=None, parity_external=None)
    assert request.central_charge is request.h_ext is request.exchange_h is None


def test_finite_numpy_scalars_remain_accepted():
    import numpy as np

    request=BlockRequest(operation='evaluate', central_charge=np.int64(8), exchange_h=np.float32(.5))
    assert request.central_charge==8
    assert request.exchange_h==.5


@pytest.mark.parametrize("basis, parity", [
    ("scalar_G", "+"), ("ordered_B", "-"), ("chiral_F", "+"),
    ("parity_projected_B", None), ("parity_projected_B", "invalid"),
])
def test_table_basis_requires_consistent_parity_before_backend(basis, parity):
    # Parity is validated before family dispatch; the successful alias test
    # below covers every family prefix without repeating these error cases.
    request = BlockRequest(
        operation="table", family="virasoro", basis=basis, parity=parity,
        coordinate="z" if basis == "chiral_F" else "zzb",
        h_ext=(1,) * 4,
        hbar_ext=(2,) * 4 if basis in {"ordered_B", "parity_projected_B"} else None,
        policy=NumericalPolicy(precision=20, lambda_order=1),
    )
    with pytest.raises(RequestValidationError, match="parity"):
        build_table_request(request)


@pytest.mark.parametrize("family", ["virasoro", "global", "ns", "ramond"])
@pytest.mark.parametrize("parity", ["+", "-"])
def test_table_auto_projection_and_prefixed_basis_resolve_consistently(family, parity):
    from virasoro_blocks.table_adapters import _resolve_sphere_table_basis

    request = BlockRequest(
        operation="table", family=family, basis="auto", parity=parity,
        coordinate="zzb", h_ext=(1,) * 4, hbar_ext=(2,) * 4,
        policy=NumericalPolicy(precision=20, lambda_order=1),
    )
    resolved = _resolve_sphere_table_basis(request)
    assert resolved.basis.endswith("parity_projected_B")
    assert resolved.parity == parity
    assert _resolve_sphere_table_basis(
        replace(request, basis=f"{family}_parity_projected_B")
    ) == resolved


@pytest.mark.parametrize("field, value", [
    ("method", "interploation"), ("method", "numeric"), ("method", None),
    ("sample_via", "numreic"), ("sample_via", "auto"), ("sample_via", None),
])
def test_backend_options_reject_unknown_dispatch_values(field, value):
    from virasoro_blocks.table_backends import VirasoroBackendOptions

    with pytest.raises(ValueError, match=field):
        VirasoroBackendOptions(**{field: value})


@pytest.mark.parametrize("central_charge, policy, expected", [
    (1, "auto", True), ("2.5", "on", True),
    ("2.5", "auto", False), ("2.5", "off", False),
])
def test_sampling_options_preserve_request_regularization(monkeypatch, central_charge, policy, expected):
    import virasoro_blocks.table_backends as backend

    # Observe the resolved options at the numerical boundary without running
    # recursion: changing sampling alone must retain the request's c policy.
    monkeypatch.setattr(backend, "build_cli_chiral_cache", lambda *a, **k: object())
    monkeypatch.setattr(backend, "build_virasoro_table", lambda request, *, spin, options: options)
    request = BlockRequest(
        operation="table", central_charge=central_charge, h_ext=(0,) * 4,
        coordinates=("zzb",),
        policy=NumericalPolicy(precision=20, lambda_order=1, order=4, c_regularization=policy),
    )
    options = build_table_request(
        request, backend_options=backend.VirasoroBackendOptions(sample_via="numeric"),
    )
    assert options.regularize is expected


def test_singular_c_table_sampling_override_matches_default_build():
    import mpmath as mp
    from virasoro_blocks.table_backends import VirasoroBackendOptions

    request = BlockRequest(
        operation="table", central_charge=1, h_ext=("0.1",) * 4,
        coordinates=("zzb",),
        policy=NumericalPolicy(precision=30, lambda_order=1, order=4, npoles="all"),
    )
    default = build_table_request(request)
    customized = build_table_request(request, backend_options=VirasoroBackendOptions(sample_via="symbolic"))
    with mp.workdps(30):
        expected = default.evaluate("zzb", mp.mpf("3.3"))
        actual = customized.evaluate("zzb", mp.mpf("3.3"))
        for m, n in [(0, 0), (1, 0)]:
            assert mp.isfinite(actual[m][n])
            assert mp.almosteq(actual[m][n], expected[m][n], rel_eps=mp.mpf("1e-25"))


@pytest.mark.parametrize("field", ["h_ext", "hbar_ext", "delta_ext", "spin_ext"])
@pytest.mark.parametrize("length", [0, 3, 5])
def test_external_label_vectors_require_four_entries(field, length):
    # Vector arity is checked before family dispatch. The family matrix is
    # already exercised by the parity/dispatch regressions in this module.
    family = "global"
    request = BlockRequest(operation="table", family=family, **{field: (0,) * length})
    with pytest.raises(RequestValidationError, match=field):
        validate_request(request)


@pytest.mark.parametrize("operation", ["evaluate", "table"])
def test_explicit_empty_external_spins_are_not_scalar_defaults(operation):
    request = BlockRequest(
        operation=operation, family="global", delta_ext=("0.2",) * 4, spin_ext=(),
        exchange_delta=3 if operation == "evaluate" else None,
        point={"z": ".3", "zbar": ".4"} if operation == "evaluate" else {},
        policy=NumericalPolicy(precision=20, lambda_order=0, order=4, npoles=2),
    )
    with pytest.raises(RequestValidationError, match="spin_ext"):
        (evaluate_block_request if operation == "evaluate" else build_table_request)(request)


@pytest.mark.parametrize("dimension", ["4", "3." + "0" * 69 + "1"])
def test_exchange_weights_cannot_silently_override_conflicting_dimension(dimension):
    request = BlockRequest(
        operation="evaluate", family="global", delta_ext=("0.2",) * 4,
        exchange_h=2, exchange_hbar=1, exchange_delta=dimension,
        point={"z": ".3", "zbar": ".4"},
        policy=NumericalPolicy(precision=20),
    )
    with pytest.raises(RequestValidationError, match="exchange_delta"):
        evaluate_block_request(request)


def test_consistent_redundant_exchange_dimension_remains_accepted():
    from virasoro_blocks.numeric_input import literal_number

    request = BlockRequest(
        operation="evaluate", family="global", delta_ext=("0.2",) * 4,
        exchange_h=2, exchange_hbar=1, exchange_delta=3, exchange_spin=1,
        point={"z": ".3", "zbar": ".4"},
        policy=NumericalPolicy(precision=20),
    )
    assert evaluate_block_request(request).value == evaluate_block_request(
        replace(request, exchange_delta=None)
    ).value
    # Mixed exact rational and native-complex labels must be compared and
    # recorded before numerical addition rounds a nonterminating sum.
    mixed = replace(request, basis="ordered_B", exchange_h=Fraction(1, 3),
                    exchange_hbar=mp.mpc(1), exchange_delta=Fraction(4, 3), exchange_spin=None)
    actual = evaluate_block_request(mixed, q_gate="off")
    assert actual.value == evaluate_block_request(replace(mixed, exchange_delta=None), q_gate="off").value
    assert literal_number(actual.metadata["exchange"]["delta"]) == Fraction(4, 3)
    assert literal_number(actual.metadata["exchange"]["spin"]) == Fraction(-2, 3)


def test_native_mpf_exchange_labels_are_judged_within_roundoff():
    # mpf('2.65') - mpf('0.65') is not exactly 2 in binary; at the working precision it is.
    with mp.workdps(30):
        h, hbar, delta = mp.mpf("2.65"), mp.mpf("0.65"), mp.mpf("3.3")
    request = BlockRequest(
        operation="evaluate", family="global", basis="scalar_G", delta_ext=("0.2",) * 4,
        exchange_h=h, exchange_hbar=hbar, exchange_delta=delta, exchange_spin=2,
        point={"z": ".3", "zbar": ".4"}, policy=NumericalPolicy(precision=20),
    )
    exact = replace(request, exchange_h="2.65", exchange_hbar="0.65", exchange_delta="3.3")
    expected = evaluate_block_request(exact, q_gate="off").value
    assert abs(evaluate_block_request(request, q_gate="off").value - expected) < 1e-18 * abs(expected)
    evaluate_block_request(replace(request, basis="ordered_B"), q_gate="off")
    # Exact labels stay exact: a written offset below the working precision is still not an integer spin.
    with pytest.raises(RequestValidationError, match="integer spin"):
        evaluate_block_request(replace(exact, exchange_hbar="0.65" + "0" * 30 + "1", exchange_delta=None,
                                       exchange_spin=None), q_gate="off")


def test_tiny_complex_external_asymmetry_selects_ordered_basis():
    from virasoro_blocks.evaluation import _canonical_basis

    with mp.workdps(30):
        native = mp.mpc(".125", "1e-80")
    with mp.workdps(15):
        for shifted in (native, ".125+1e-80j"):
            request = BlockRequest(
                operation="evaluate", family="global", coordinate="zzb",
                h_ext=(".125",) * 4, hbar_ext=(shifted,) * 4,
                exchange_h=1, exchange_hbar=1, point={"z": ".3", "zbar": ".4"},
                policy=NumericalPolicy(precision=20),
            )
            assert _canonical_basis(request) == "global_ordered_B"
            actual = evaluate_block_request(request, q_gate="off")
            expected = evaluate_block_request(replace(request, basis="ordered_B"), q_gate="off")
            assert actual.metadata["basis"] == "global_ordered_B"
            assert actual.value == expected.value
        assert mp.mp.dps == 15


def test_complex_exchange_relations_are_checked_within_roundoff(monkeypatch):
    from virasoro_blocks import evaluation

    monkeypatch.setattr(evaluation, "global_evaluate_block",
                        lambda *a, **kw: pytest.fail("inconsistent labels reached block evaluation"))
    # Native mpmath inputs are numerical values: an offset above the working
    # precision's roundoff is inconsistent, one below it is not.
    with mp.workdps(30):
        bad_dimension = mp.mpc(2, "1e-10")
        bad_weight = mp.mpc(1, "1e-10")
    request = BlockRequest(
        operation="evaluate", family="global", basis="ordered_B", coordinate="zzb",
        h_ext=(".125",) * 4, hbar_ext=(".125",) * 4,
        exchange_h=1, exchange_hbar=1, point={"z": ".3", "zbar": ".4"},
        policy=NumericalPolicy(precision=20),
    )
    with mp.workdps(15):
        for changed, message in (
            (replace(request, exchange_delta=bad_dimension), "exchange_delta"),
            (replace(request, basis="scalar_G", exchange_h=bad_weight, exchange_hbar=0), "integer spin"),
            (replace(request, exchange_h=bad_weight, exchange_hbar=0, exchange_spin=1), "exchange_spin"),
        ):
            with pytest.raises(RequestValidationError, match=message):
                evaluate_block_request(changed, q_gate="off")
        with mp.workdps(150):
            shifted_spin = mp.mpc(mp.mpf(".5") + mp.mpf("1e-10"), 0)
            roundoff_spin = mp.mpc(mp.mpf(".5") + mp.mpf("1e-100"), 0)
        with pytest.raises(RequestValidationError, match="half-integer"):
            evaluation._resolve_exchange(replace(request, exchange_h=None, exchange_hbar=None,
                exchange_delta=3, exchange_spin=shifted_spin), chiral=False)
        with mp.workdps(20):
            h, hbar = evaluation._resolve_exchange(replace(request, exchange_h=None, exchange_hbar=None,
                exchange_delta=3, exchange_spin=roundoff_spin), chiral=False)
        assert (h, hbar) == (Fraction(7, 4), Fraction(5, 4))


def test_derived_native_external_weights_retain_both_components():
    from virasoro_blocks.evaluation import _resolve_external_h
    from virasoro_blocks.numeric_input import literal_components
    from virasoro_blocks.table_adapters import _normalize_sphere_table_request

    with mp.workdps(400):
        real = mp.mpf(".5") + mp.mpf("1e-300")
        labels = (mp.mpc(real, 0), mp.mpc(real, "1e-300"))
    with mp.workdps(15):
        for delta in labels:
            request = BlockRequest(operation="table", family="global", basis="ordered_B",
                delta_ext=(delta,) * 4, spin_ext=(".5",) * 4,
                policy=NumericalPolicy(precision=20, lambda_order=1))
            normalized = _normalize_sphere_table_request(request)
            direct = _resolve_external_h(request, chiral=False)
            real, imaginary = literal_components(delta)
            expected = (((real + Fraction(1, 2))/2, imaginary/2),
                        ((real - Fraction(1, 2))/2, imaginary/2))
            for weights in ((normalized.h_ext, normalized.hbar_ext), direct):
                assert tuple(literal_components(values[0]) for values in weights) == expected
            assert normalized.policy.precision == 20
        assert mp.mp.dps == 15


def test_torus_evaluation_uses_the_table_spin_contract():
    # Conversion must not merge non-real spins or tiny lattice offsets into
    # an integer. Generic numeric-type coverage is at request construction.
    for family in ("torus-character", "torus-onepoint"):
        request = BlockRequest(operation="evaluate", surface="torus", family=family,
            coordinate="vvb", central_charge=8, exchange_delta=3,
            torus_onepoint_ext=0 if family == "torus-onepoint" else None,
            point={"v": 0, "vbar": 0}, policy=NumericalPolicy(precision=20, order=1))
        with mp.workdps(15):
            for spin in (mp.mpc(1, "1e-80"), 1+Fraction(1, 10**100)):
                with pytest.raises(RequestValidationError, match="non-negative integers"):
                    evaluate_block_request(replace(request, exchange_spin=spin), q_gate="off")
        expected = evaluate_block_request(replace(request, exchange_spin=1), q_gate="off")
        assert evaluate_block_request(replace(request, exchange_spin=1.0), q_gate="off").value == expected.value
