"""PMP digit counts cannot silently round to a different output precision."""
from fractions import Fraction
import mpmath as mp
import pytest
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.serialization import SerializedDerivTable
from virasoro_bootstrap.pmp_writer import build_pmp_doc, build_prefactored_pmp_doc


def build(builder, digits):
    block = SerializedDerivTable(0, mp.mpf('.5'), (), ('zzb',),
        {'zzb': [[Polynomial([mp.mpf('1.23456789')])]]})
    kwargs = {'explicit_sample_points': True} if builder is build_prefactored_pmp_doc else {}
    result = builder([block], [(0, 0)], objective=[mp.mpf('.123456789')],
                     normalization=[1], sig_figs=digits, **kwargs)
    return result if isinstance(result, dict) else result.doc


def test_pmp_numeric_entries_keep_small_real_offsets():
    from decimal import Decimal

    with mp.workdps(40):
        epsilon = mp.mpf(2)**-130
        value = 1 + epsilon
        block = SerializedDerivTable(0, mp.mpf('.5'), (), ('zzb',),
            {'zzb': [[Polynomial([-1, value], field=mp.mpf)]]})
        for builder in (build_pmp_doc, build_prefactored_pmp_doc):
            kwargs = {'explicit_sample_points': True} if builder is build_prefactored_pmp_doc else {}
            result = builder([block], [(0, 0)], objective=[value],
                             normalization=[mp.mpc(value, epsilon**2)], **kwargs)
            doc = result if isinstance(result, dict) else result.doc
            matrix = doc['PositiveMatrixWithPrefactorArray'][0]
            for text in (doc['objective'][0], doc['normalization'][0],
                         matrix['polynomials'][0][0][0][1]):
                with mp.workdps(60):
                    assert abs((mp.mpf(text)-1)/epsilon-1) < mp.mpf('1e-5')
        literal = '1.' + '0'*69 + '1'
        with mp.workdps(120):
            source = mp.mpc(mp.mpf(literal), mp.mpf('1e-140'))
        for builder in (build_pmp_doc, build_prefactored_pmp_doc):
            for label in (Decimal(literal), .7):
                result = builder([block], [(0, 0)], objective=[label],
                                 normalization=[source], sig_figs=100)
                doc = result if isinstance(result, dict) else result.doc
                with mp.workdps(120):
                    expected = mp.mpf(str(label)) if not isinstance(label, Fraction) else mp.convert(label)
                    assert abs(mp.mpf(doc['objective'][0])-expected) < mp.mpf('1e-98')
                    assert abs(mp.mpf(doc['normalization'][0])-source.real) < mp.mpf('1e-98')
        assert mp.mp.dps == 40


def test_low_precision_pmp_rejects_material_complex_entries():
    from dataclasses import replace

    for dps in (3, 40):
        with mp.workdps(dps):
            block = SerializedDerivTable(0, mp.mpf('.5'), (), ('zzb',),
                {'zzb': [[Polynomial.one()]]})
            for bad in (mp.mpc(1, '.001'), complex(1, -.001)):
                for field in ('objective', 'normalization', 'base', 'pole', 'coefficient'):
                    altered = block
                    kwargs = {'objective': [bad if field == 'objective' else 0],
                              'normalization': [bad if field == 'normalization' else 1]}
                    if field == 'base':
                        altered = replace(block, base=bad)
                    elif field == 'pole':
                        altered = replace(block, pole_list_x=(-bad,))
                    elif field == 'coefficient':
                        altered = replace(block, numerators={'zzb': [[Polynomial([bad], field=mp.mpc)]]})
                    with pytest.raises(ValueError, match='(must be real|complex value)'):
                        build_pmp_doc([altered], [(0, 0)], **kwargs)
                    assert mp.mp.dps == dps
            noise = mp.mpf(10)**-max(7, dps)
            doc = build_pmp_doc([block], [(0, 0)], objective=[mp.mpc(1, noise)])
            assert mp.mpf(doc['objective'][0]) == 1


# Shared integer type coverage lives in the core validator tests.
@pytest.mark.parametrize('builder, digits', [(build_pmp_doc, True), (build_prefactored_pmp_doc, 2.5)])
def test_invalid_pmp_precision_is_rejected(builder, digits):
    with pytest.raises(ValueError, match='sig_figs'):
        build(builder, digits)


@pytest.mark.parametrize('builder', [build_prefactored_pmp_doc])
@pytest.mark.parametrize('digits', [Fraction(20)])
def test_integral_precision_types_match_integer(builder, digits):
    with mp.workdps(30):
        assert build(builder, digits) == build(builder, 20)
