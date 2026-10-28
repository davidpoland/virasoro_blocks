"""Global point evaluation accepts complex external parameters and checks their poles."""
import mpmath as mp
import pytest

from virasoro_blocks.api import BlockPoleError, BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.numeric_input import literal_components


@pytest.mark.parametrize('basis', ['chiral_F', 'ordered_B', 'scalar_G', 'parity_projected_B'])
@pytest.mark.parametrize('source', ['real_mpc', 'complex_mpc', 'python_complex'])
def test_identical_complex_externals_match_the_analytic_global_block(basis, source):
    with mp.workdps(100):
        ext = (mp.mpc('.3', '0' if source == 'real_mpc' else '.1'),)*4
    if source == 'python_complex':
        ext = (complex(.3, .1),)*4
    chiral = basis == 'chiral_F'
    request = BlockRequest(
        operation='evaluate', family='global', basis=basis, coordinate='z' if chiral else 'zzb',
        h_ext=ext, hbar_ext=None if chiral else ext, exchange_h='.8',
        exchange_hbar=None if chiral else '.8',
        parity='+' if basis == 'parity_projected_B' else None,
        point={'z': '.3'} if chiral else {'z': '.3', 'zbar': '.4'},
        policy=NumericalPolicy(precision=40),
    )
    actual = evaluate_block_request(request, q_gate='off').value
    with mp.workdps(80):
        def half(z):
            h=mp.mpf('.8')
            return mp.power(16, -h)*mp.power(z, h)*mp.hyp2f1(h, h, 2*h, z)
        expected = half(mp.mpf('.3'))
        if not chiral:
            expected *= half(mp.mpf('.4'))
        if basis == 'parity_projected_B':
            expected *= 2      # B + P[B]; the spin-0 scalar_G is the product block itself
        assert abs(actual-expected) < mp.mpf('1e-38')*abs(expected)


@pytest.mark.parametrize('direction', ['real', 'imag'])
def test_small_complex_external_difference_preserves_a_genuine_global_vacuum_pole(direction):
    with mp.workdps(160):
        center = mp.mpc('.3', '.1')
        shift = mp.mpf('1e-70') if direction == 'real' else mp.mpc(0, '1e-70')
        ext = (center, center+shift, center+shift, center)
    with pytest.raises(BlockPoleError):
        evaluate_block_request(BlockRequest(
            operation='evaluate', family='global', basis='chiral_F', coordinate='z',
            h_ext=ext, exchange_h=0, point={'z': '.3'}, policy=NumericalPolicy(precision=40)), q_gate='off')


def test_common_imaginary_components_leave_exact_real_external_differences():
    from virasoro_blocks.global_blocks import _externals_diffs, _genuine_chiral_poles

    with mp.workdps(100):
        ext = (mp.mpc('.25', '.3'), mp.mpc('1.25', '.3'), mp.mpc('1.75', '.3'), mp.mpc('.75', '.3'))
    with mp.workdps(15):
        assert _externals_diffs(ext) == (1, 1)
        assert _genuine_chiral_poles(1, delta12=_externals_diffs(ext)[0], delta34=_externals_diffs(ext)[1], shift=-1) == [-.5]
        assert literal_components(ext[0])[1] == literal_components(ext[1])[1]


@pytest.mark.parametrize('basis', ['chiral_F', 'ordered_B'])
def test_asymmetric_complex_external_differences_match_hypergeometric_formula(basis):
    with mp.workdps(100):
        ext = (mp.mpc('.3', '.1'), mp.mpc('.6', '.2'), mp.mpc('.4', '.3'), mp.mpc('.8', '.4'))
    chiral = basis == 'chiral_F'
    actual = evaluate_block_request(BlockRequest(
        operation='evaluate', family='global', basis=basis, coordinate='z' if chiral else 'zzb',
        h_ext=ext, hbar_ext=None if chiral else ext, exchange_h='.8',
        exchange_hbar=None if chiral else '.7',
        point={'z': '.3'} if chiral else {'z': '.3', 'zbar': '.4'},
        policy=NumericalPolicy(precision=40)), q_gate='off').value
    with mp.workdps(80):
        def half(h, z):
            return 16**(-h)*z**h*mp.hyp2f1(h+ext[1]-ext[0], h+ext[2]-ext[3], 2*h, z)
        expected = half(mp.mpf('.8'), mp.mpf('.3'))
        if not chiral:
            expected *= half(mp.mpf('.7'), mp.mpf('.4'))
        assert abs(actual-expected) < mp.mpf('1e-38')*abs(expected)
