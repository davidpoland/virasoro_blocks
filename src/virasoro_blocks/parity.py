"""Parity metadata helpers.

External and exchanged parity phases, made explicit at the CLI/output boundary
and used by the parity-projected blocks (`composer.parity_reversed_B_block`,
the projected and folded tables). The model is a diagonal parity-eigenstate
basis for the four external operators, encoded by per-operator unit phases.
"""

from __future__ import annotations

from typing import Sequence

import mpmath as mp


_PARITY_PHASE_MAP = {
    "+1": "1",
    "1": "1",
    "-1": "-1",
    "+i": "i",
    "i": "i",
    "-i": "-i",
}

_PARITY_PHASE_COMPLEX = {
    "1": mp.mpc(1),
    "-1": mp.mpc(-1),
    "i": mp.j,
    "-i": -mp.j,
}


def normalize_parity_phase(token: str) -> str:
    """Normalize one external parity-eigenvalue token.

    The initial metadata model is intentionally narrow: a diagonal basis with
    per-operator unit phases in {1, -1, i, -i}. These are stored as strings so
    later projection code can attach convention-specific meaning without a JSON
    float/complex encoding choice.
    """
    if not isinstance(token, str):
        raise ValueError("external parity phases must be one of 1, -1, i, -i")
    key = token.strip().lower()
    if key not in _PARITY_PHASE_MAP:
        raise ValueError(
            "external parity phases must be one of 1, -1, i, -i"
        )
    return _PARITY_PHASE_MAP[key]


def normalize_external_parity(phases: Sequence[str] | None):
    """Return normalized parity metadata for the four externals, or None."""
    if phases is None:
        return None
    if len(phases) != 4:
        raise ValueError("external parity data must have exactly 4 entries")
    return [normalize_parity_phase(phase) for phase in phases]


def parity_phase_value(token: str) -> mp.mpc:
    """Return the unit complex number for one normalized parity token."""
    return _PARITY_PHASE_COMPLEX[normalize_parity_phase(token)]


def parity_reversal_phase(
    parity_external: Sequence[str] | None = None,
    *,
    exchanged_parity: str = "1",
) -> mp.mpc:
    """Return the phase multiplying the parity-reversed ordered block.

    In the diagonal-parity basis, the overall phase is the product of the four
    external parity phases, times an optional exchanged-operator parity phase. When parity metadata is
    absent, the phase defaults to 1.
    """
    phase = parity_phase_value(exchanged_parity)
    normalized = normalize_external_parity(parity_external)
    if normalized is None:
        return phase
    for token in normalized:
        phase *= parity_phase_value(token)
    return phase


def parity_projection_is_zero(coefficient, h, hbar, z, zbar, h_ext, hbar_ext, z_shift=None, zbar_shift=None,
                              *, global_block=False):
    """Recognize symmetry-enforced zeros using exact labels, including stored mpmath bits.

    ``coefficient`` includes both the projection sign and the parity phase. Shifts must already have their
    family-specific defaults resolved. Distinct labels that round to the same working value are not equal here.
    """
    from .numeric_input import literal_components

    def same(left, right):
        return literal_components(left) == literal_components(right)

    if coefficient != -1:
        return False
    if same(z, zbar):
        return True
    if not (same(h, hbar) and same(0 if z_shift is None else z_shift, 0 if zbar_shift is None else zbar_shift)):
        return False
    if global_block:
        # The 2F1 depends only on h2-h1 and h3-h4, symmetrically in its two numerator parameters.
        def differences(ext):
            parts = tuple(literal_components(value) for value in ext)
            return sorted((parts[a][0] - parts[b][0], parts[a][1] - parts[b][1])
                          for a, b in ((1, 0), (2, 3)))
        return differences(h_ext) == differences(hbar_ext)
    return all(same(a, b) for a, b in zip(h_ext, hbar_ext))
