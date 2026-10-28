"""Direct numerical grids and samplers accept ordinary Fraction inputs."""

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks import derivs
from virasoro_blocks.polyderivs import build_numeric_sampler


@pytest.mark.parametrize("coordinate", ["zzb", "yyb", "xt", "ws"])
def test_direct_grid_accepts_fraction_dimension(coordinate):
    builder = getattr(derivs, f"G_{coordinate}_table")
    kwargs = dict(c=32, q_order=4, chiral_norm=4, z_shift=mp.mpf("-.25"))
    with mp.workdps(35):
        got = builder(Fraction(3, 2), 0, [mp.mpf(".125")] * 4, 1, 1, **kwargs)
        expected = builder(mp.mpf("1.5"), 0, [mp.mpf(".125")] * 4, 1, 1, **kwargs)
        assert got == expected


@pytest.mark.parametrize("basis", ["scalar", "ordered", "projected", "chiral"])
@pytest.mark.parametrize("restricted", [False, True])
def test_numeric_sampler_accepts_fraction_weights(basis, restricted):
    with mp.workdps(35):
        chiral = basis == "chiral"
        sampler = build_numeric_sampler(
            "z" if chiral else "zzb", chiral=chiral,
            spinning=basis in ("ordered", "projected"), projected=basis == "projected",
            h_externals=[mp.mpf(".125")] * 4, hbar_externals=[mp.mpf(".25")] * 4,
            ell=0, m_max=1, n_max=0 if chiral else 1,
            c=32, q_order=4, chiral_norm=4, z_shift=mp.mpf("-.25"),
            zbar_shift=mp.mpf("-.5"), parity="+", parity_external=("1",) * 4,
        )
        entries = [(0, 0)] if restricted else None
        actual, actual_arb = sampler([Fraction(3, 2)], entries=entries)
        expected, expected_arb = sampler([mp.mpf("1.5")], entries=entries)
        assert actual_arb == expected_arb
        for m, n in entries or [(m, n) for m in range(2) for n in range(1 if chiral else 2)]:
            a, e = actual[0][m][n], expected[0][m][n]
            if actual_arb:
                a, e = derivs._arb_to_mpf(a), derivs._arb_to_mpf(e)
            assert a == e
