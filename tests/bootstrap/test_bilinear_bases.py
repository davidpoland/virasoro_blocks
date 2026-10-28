"""The SDP writer's bilinear bases (`pmp_writer._compute_bilinear_bases`).

The flint Stieltjes/Lanczos construction must give, at working precision, an
orthonormal, degree-ordered basis of the weighted-monomial row space that
pmp2sdp's bilinear decomposition prescribes. Rank follows the actual weighted
support, so tiny nonzero sample scalings keep every independent row.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.interpolation import DampedRational, get_sample_points
from virasoro_bootstrap.pmp_writer import _compute_bilinear_bases, _compute_sample_scalings

DPS = 60


def _block(n):
    """Sample points and scalings of a many-pole damped-rational block (a Virasoro-like measure)."""
    base = 3 - 2 * mp.sqrt(2)
    poles = tuple(-mp.mpf(k) / 3 for k in range(1, 9))
    xs = tuple(get_sample_points(DampedRational(constant=1, poles=poles, base=base), n))
    return xs, _compute_sample_scalings(xs, base, poles)


def _reference(xs, scalings, rows, shift):
    """Orthonormal columns spanning the weighted-monomial rows, by Householder QR at 3x precision."""
    with mp.workdps(3 * DPS):
        A = mp.matrix(len(xs), rows)
        for i, (x, s) in enumerate(zip(xs, scalings)):
            for k in range(rows):
                A[i, k] = mp.sqrt(abs(s)) * mp.sqrt(x) ** shift * x ** k
        Q, _ = mp.qr(A)
        return [[Q[i, k] for i in range(len(xs))] for k in range(rows)]


def _dot(u, v):
    return mp.fsum(a * b for a, b in zip(u, v))


@pytest.mark.parametrize("n", [9, 24])
def test_bases_orthonormal_degree_ordered_and_span_reference(n):
    with mp.workdps(DPS):
        xs, sc = _block(n)
        b0, b1 = _compute_bilinear_bases(xs, sc, n - 1, DPS)
    assert (len(b0), len(b1)) == ((n - 1) // 2 + 1, (n - 2) // 2 + 1)
    tol = mp.mpf(10) ** (-(DPS - 5))
    for basis, shift in ((b0, 0), (b1, 1)):
        ref = _reference(xs, sc, len(basis), shift)
        with mp.workdps(3 * DPS):
            for a, row in enumerate(basis):
                for b in range(a + 1):
                    assert abs(_dot(row, basis[b]) - (a == b)) < tol
                # degree ordering: row a lies in the span of the first a+1 weighted monomials
                coef = [_dot(row, ref[k]) for k in range(a + 1)]
                resid = [row[i] - mp.fsum(c * ref[k][i] for k, c in enumerate(coef)) for i in range(n)]
                assert max(abs(r) for r in resid) < tol
                # and it is the Gram-Schmidt row itself (positive leading coefficient), up to the QR sign
                assert abs(abs(coef[a]) - 1) < tol


def test_tiny_scalings_keep_every_row():
    """A relative rank test: scaling the measure by 1e-200 changes no row (an absolute cutoff
    10^-(dps/4) used to drop every row whose weighted norm fell below it and zero-pad them)."""
    with mp.workdps(DPS):
        xs, sc = _block(24)
        b0, b1 = _compute_bilinear_bases(xs, sc, 23, DPS)
        t0, t1 = _compute_bilinear_bases(xs, [s * mp.mpf(10) ** -200 for s in sc], 23, DPS)
    tol = mp.mpf(10) ** (-(DPS - 8))
    for ref, tiny in ((b0, t0), (b1, t1)):
        assert len(ref) == len(tiny)
        for r, t in zip(ref, tiny):
            assert max(abs(u - v) for u, v in zip(r, t)) < tol


@pytest.mark.parametrize('repeated_nodes', [False, True])
def test_rank_deficient_measure_is_zero_padded(repeated_nodes):
    """Only two points carry weight: two independent rows, the rest padding (pmp2sdp's row count)."""
    with mp.workdps(DPS):
        xs = tuple(mp.mpf(k) / 2 for k in range(1, 7))
        sc = [mp.mpf(0), mp.mpf(2), mp.mpf(0), mp.mpf(0), mp.mpf(3), mp.mpf(0)]
        if repeated_nodes:
            xs = (mp.mpf(1), mp.mpf(2)) * 3
            sc = [mp.mpf(1)] * 6
        b0, b1 = _compute_bilinear_bases(xs, sc, 5, DPS)
    for basis in (b0, b1):
        assert len(basis) == 3
        nonzero = [any(v != 0 for v in row) for row in basis]
        assert nonzero == [True, True, False]


def test_negative_sample_point_rejected():
    with mp.workdps(DPS):
        with pytest.raises(ValueError, match="sample points"):
            _compute_bilinear_bases((mp.mpf(-1), mp.mpf(1)), [mp.mpf(1), mp.mpf(1)], 1, DPS)


@pytest.mark.parametrize('offset, spacing', [('1', '1e-30'), ('1e40', '1')])
def test_clustered_nodes_keep_full_polynomial_span(offset, spacing):
    """A common translation must not make independent monomials disappear."""
    with mp.workdps(100):
        xs = tuple(mp.mpf(offset) + k * mp.mpf(spacing) for k in range(5))
        sc = [mp.mpf(k + 1) for k in range(5)]
        bases = _compute_bilinear_bases(xs, sc, 4, 100)
        # QR on the original monomials is an independent, higher-precision
        # reference even when their common offset makes them ill-conditioned.
        with mp.workdps(300):
            for shift, basis in enumerate(bases):
                matrix = mp.matrix([[mp.sqrt(s * x ** shift) * x ** k
                                     for k in range(len(basis))]
                                    for x, s in zip(xs, sc)])
                reference, _ = mp.qr(matrix)
                for a, row in enumerate(basis):
                    assert abs(_dot(row, row) - 1) < mp.mpf('1e-90')
                    overlap = _dot(row, [reference[i, a] for i in range(len(xs))])
                    assert abs(abs(overlap) - 1) < mp.mpf('1e-90')


@pytest.mark.parametrize('exponent, dominant', [(40, 1), (240, 2)])
def test_direct_writer_keeps_independent_rows_with_uneven_weights(tmp_path, exponent, dominant):
    """A small nonzero residual is an independent row, not missing support."""
    import json
    from flint import ctx
    from virasoro_bootstrap.sdp_writer import write_sdp_directory

    scalings = [f'1e-{exponent}'] * 3
    scalings[dominant] = '1'
    doc = {'objective': ['0'], 'normalization': ['1'],
           'PositiveMatrixWithPrefactorArray': [{
               'polynomials': [[[['1', '0', '1']]]],
               'samplePoints': ['0', '1', '2'], 'sampleScalings': scalings}]}
    original_dps, original_bits = mp.mp.dps, ctx.prec
    output = write_sdp_directory(doc, tmp_path / 'sdp', eval_dps=DPS)
    assert (mp.mp.dps, ctx.prec) == (original_dps, original_bits)
    data = json.loads((output / 'block_data_0.json').read_text())
    with mp.workdps(DPS + exponent + 20):
        xs = [mp.mpf(k) for k in range(3)]
        for shift, name in enumerate(('bilinear_bases_even', 'bilinear_bases_odd')):
            basis = [[mp.mpf(v) for v in row] for row in data[name]]
            matrix = mp.matrix([[mp.sqrt(mp.mpf(s) * x ** shift) * x ** k
                                 for k in range(len(basis))]
                                for x, s in zip(xs, scalings)])
            reference, _ = mp.qr(matrix)
            for a, row in enumerate(basis):
                assert abs(_dot(row, row) - 1) < mp.mpf('1e-55')
                for b in range(a):
                    assert abs(_dot(row, basis[b])) < mp.mpf('1e-55')
                overlap = _dot(row, [reference[i, a] for i in range(len(xs))])
                assert abs(abs(overlap) - 1) < mp.mpf('1e-55')
