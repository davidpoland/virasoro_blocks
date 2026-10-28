"""Frozen tables reuse external weights across validation and coordinates."""
import mpmath as mp
import pytest
from virasoro_blocks.point import point_block_table


@pytest.mark.parametrize('basis', ['global_ordered_B', 'global_parity_projected_B'])
def test_omitted_antiholomorphic_weights_match_explicit_weights(basis):
    with mp.workdps(30):
        ext = [mp.mpf('.1')] * 4
        kwargs = dict(basis=basis, ell=2, point_value=3,
                      h_externals=ext, lam=1, parity='+')
        expected = point_block_table(['zzb', 'yyb'], hbar_externals=ext, **kwargs)
        actual = point_block_table(['zzb', 'yyb'], **kwargs)
        for coordinate in ('zzb', 'yyb'):
            assert actual.evaluate(coordinate, 3) == expected.evaluate(coordinate, 3)


@pytest.mark.parametrize('basis', ['global_ordered_B', 'ordered_B'])
@pytest.mark.parametrize('which', ['holomorphic', 'antiholomorphic', 'both'])
def test_external_iterators_survive_pole_checks_and_multiple_coordinates(basis, which):
    with mp.workdps(30):
        ext = [mp.mpf(v) for v in ('.1', '.2', '.3', '.4')]
        anti = [mp.mpf(v) for v in ('.2', '.3', '.4', '.5')]
        kwargs = dict(basis=basis, ell=2, point_value=3, lam=1, c=32, q_order=4)
        expected = point_block_table(['zzb', 'yyb'], h_externals=ext,
                                     hbar_externals=anti, **kwargs)
        actual = point_block_table(['zzb', 'yyb'],
            h_externals=iter(ext) if which != 'antiholomorphic' else ext,
            hbar_externals=iter(anti) if which != 'holomorphic' else anti, **kwargs)
        for coordinate in ('zzb', 'yyb'):
            assert actual.evaluate(coordinate, 3) == expected.evaluate(coordinate, 3)


@pytest.mark.parametrize('count', [0, 3, 5])
@pytest.mark.parametrize('side', ['holomorphic', 'antiholomorphic'])
def test_malformed_external_lists_fail_before_pole_check(monkeypatch, count, side):
    from virasoro_blocks import point
    def unexpected(**kwargs):
        pytest.fail('malformed external weights reached the pole check')
    monkeypatch.setattr(point, '_check_point_pole', unexpected)
    with pytest.raises(ValueError, match='length 4'):
        point_block_table(['zzb'], basis='global_ordered_B', ell=0,
            point_value=1, lam=0,
            h_externals=[.1] * (count if side == 'holomorphic' else 4),
            hbar_externals=[.1] * (count if side == 'antiholomorphic' else 4))


def test_scalar_point_inputs_are_validated_without_discarding_independent_basis_inputs(monkeypatch):
    from fractions import Fraction
    from virasoro_blocks import point
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request

    with mp.workdps(120):
        near_weight = mp.mpc(mp.mpf('.7')+mp.mpf('1e-70'), 0)
        near_shift = mp.mpc(mp.mpf('-.25')+mp.mpf('1e-70'), 0)
    with mp.workdps(30):
        for family in ('', 'global_'):
            common = dict(basis=family+'scalar_G', ell=0, point_value='.2',
                h_externals=['.7']*4, c=2, q_order=4, lam=0, z_shift='-.25')
            invalid = [
                dict(hbar_externals=['.25', '.4', '.3', '.2']),
                dict(hbar_externals=[near_weight]+['.7']*3),
                dict(hbar_externals=[Fraction(7, 10)+Fraction(1, 10**70)]+['.7']*3),
                dict(zbar_shift='.7'), dict(zbar_shift=near_shift),
                dict(zbar_shift=Fraction(-1, 4)+Fraction(1, 10**70)),
            ]
            with monkeypatch.context() as patch:
                patch.setattr(point, '_check_point_pole',
                    lambda **kwargs: pytest.fail('mismatched scalar inputs reached numerical work'))
                for edit in invalid:
                    with pytest.raises(ValueError, match='scalar_G'):
                        point_block_table(['zzb'], **common, **edit)
            expected = point_block_table(['zzb'], **common).evaluate('zzb', '.2')
            for label in ('.7', 0.7, Fraction(7, 10)):
                actual = point_block_table(['zzb'], **common, hbar_externals=iter([label]*4),
                    zbar_shift=Fraction(-1, 4)).evaluate('zzb', '.2')
                assert actual == expected
            zero_shift = {**common, 'z_shift': None}
            assert (point_block_table(['zzb'], **zero_shift, zbar_shift='0').evaluate('zzb', '.2') ==
                    point_block_table(['zzb'], **zero_shift).evaluate('zzb', '.2'))

            # The independent parameters remain meaningful for spinning bases,
            # matching their public pointwise evaluator at an off-diagonal point.
            ext, anti = ['.125']*4, ['.25', '.4', '.3', '.2']
            for kind in ('ordered_B', 'parity_projected_B'):
                basis = family+kind
                parity = '+' if kind == 'parity_projected_B' else None
                table = point_block_table(['zzb'], basis=basis, ell=0, point_value='.2',
                    h_externals=ext, hbar_externals=anti, c=2, q_order=4, lam=0,
                    center='.3', center_bar='.4', z_shift='-.25', zbar_shift='.7', parity=parity)
                request = BlockRequest(operation='evaluate', family='global' if family else 'virasoro',
                    basis=basis, coordinate='zzb', central_charge=2, h_ext=ext, hbar_ext=anti,
                    exchange_delta='.2', exchange_spin=0, z_shift='-.25', zbar_shift='.7', parity=parity,
                    point={'z': '.3', 'zbar': '.4'}, policy=NumericalPolicy(precision=30, order=4))
                expected = evaluate_block_request(request, q_gate='off').value
                actual = table.evaluate('zzb', '.2')[0][0]
                assert abs(actual-expected) < mp.mpf('1e-25')*max(1, abs(expected))
        assert mp.mp.dps == 30


def test_chiral_point_tables_reject_independent_antiholomorphic_inputs_and_spin(monkeypatch):
    from fractions import Fraction
    from virasoro_blocks import point

    with mp.workdps(120):
        near_weight = mp.mpc(mp.mpf('.125')+mp.mpf('1e-70'), 0)
        near_shift = mp.mpc(mp.mpf('-.25')+mp.mpf('1e-70'), 0)
    with mp.workdps(30):
        for basis in ('chiral_F', 'global_chiral_F'):
            inputs = dict(basis=basis, ell=0, point_value='.2', h_externals=['.125']*4,
                c=2, q_order=4, lam=1, z_shift='-.25')
            invalid = [(dict(hbar_externals=['.25']*4), 'hbar_externals'),
                       (dict(hbar_externals=[near_weight]+['.125']*3), 'hbar_externals'),
                       (dict(zbar_shift='.7'), 'zbar_shift'), (dict(zbar_shift=near_shift), 'zbar_shift')]
            invalid += [(dict(ell=ell), 'ell') for ell in (1, -1, Fraction(1, 2), True, None)]
            with monkeypatch.context() as patch:
                patch.setattr(point, '_check_point_pole',
                    lambda **kwargs: pytest.fail('ignored chiral input reached numerical work'))
                for edit, name in invalid:
                    with pytest.raises(ValueError, match=name):
                        point_block_table(['z'], **{**inputs, **edit})
            expected = point_block_table(['z'], **inputs).evaluate('z', '.2')
            for label, spin in (('.125', 0), (0.125, 0.0), (Fraction(1, 8), Fraction(0)),
                                (mp.mpc('.125'), mp.mpc(0)), ('.125', '0')):
                actual = point_block_table(['z'], **{**inputs, 'ell': spin}, hbar_externals=iter([label]*4),
                    zbar_shift=Fraction(-1, 4)).evaluate('z', '.2')
                assert actual == expected
        assert mp.mp.dps == 30


def test_point_spins_follow_the_shared_exact_spin_input_policy(monkeypatch):
    from fractions import Fraction
    from virasoro_blocks import point

    with mp.workdps(120):
        shifted_half = mp.mpc(mp.mpf('.5')+mp.mpf('1e-70'), 0)
    with mp.workdps(30):
        for family in ('', 'global_'):
            for kind in ('scalar_G', 'ordered_B', 'parity_projected_B'):
                inputs = dict(basis=family+kind, point_value='3', h_externals=['.125']*4,
                    c=2, q_order=4, lam=0, parity='+' if kind == 'parity_projected_B' else None)
                spins = (2,) if kind == 'scalar_G' else (2, Fraction(1, 2))
                for spin in spins:
                    expected = point_block_table(['zzb'], ell=spin, **inputs).evaluate('zzb', 3)
                    for label in (str(spin), float(spin), mp.mpc(float(spin))):
                        assert point_block_table(['zzb'], ell=label, **inputs).evaluate('zzb', 3) == expected
                with monkeypatch.context() as patch:
                    patch.setattr(point, '_check_point_pole',
                        lambda **kwargs: pytest.fail('invalid spin reached the pole check'))
                    invalid = (True, '.7', shifted_half)
                    if kind == 'scalar_G':
                        invalid += (Fraction(1, 2),)
                    for spin in invalid:
                        with pytest.raises(ValueError, match='ell'):
                            point_block_table(['zzb'], ell=spin, **inputs)
        assert mp.mp.dps == 30
