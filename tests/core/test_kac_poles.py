"""The chiral evaluators' Kac pole list carries the genuine (complex) Kac weights."""

from __future__ import annotations

import mpmath as mp

from virasoro_blocks import derivs


def test_strip_kac_poles_are_the_complex_conjugate_pairs():
    """For 1 < c < 25 the weights of r != s are complex conjugate pairs; their real parts alone made each pair a
    spurious double real pole in the singular-c regularizer's cluster matching."""
    mp.mp.dps = 30
    evaluate = derivs.chiral_z_series_evaluator([mp.mpf(1) / 16] * 4, 4, c=mp.mpf(8), q_order=6)
    poles = [mp.mpc(p) for p in evaluate.kac_poles()]
    complex_ones = [p for p in poles if p.imag != 0]
    assert complex_ones, "no complex Kac weight at c = 8"
    for p in complex_ones:
        assert min(abs(q - mp.conj(p)) for q in poles) < mp.mpf(10) ** -25
    assert any(p.imag == 0 for p in poles)           # r = s weights are returned real
