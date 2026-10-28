"""Small imaginary offsets select the nome and fractional-power branches."""
import mpmath as mp
import pytest

from virasoro_blocks.recursion import q_from_z


@pytest.mark.parametrize('point', [(-0.5, 1e-60), (-0.5, -1e-60), (-2, -1e-130)])
def test_tiny_imaginary_offset_selects_the_nome_side(point):
    # on the negative real axis q is negative; a tiny Im z sets the side (the branch of q^h) below q's rounding
    with mp.workdps(400):
        z = mp.mpc(*point)
        reference = mp.exp(-mp.pi**2 / (2*mp.agm(1, mp.sqrt(z))*mp.ellipk(z)))
    with mp.workdps(50):
        q = q_from_z(z)
        assert mp.sign(q.imag) == mp.sign(point[1])
        assert abs(q - reference) < mp.mpf('1e-49')*abs(reference)
        assert abs(q.imag - reference.imag) < mp.mpf('1e-45')*abs(reference.imag)
