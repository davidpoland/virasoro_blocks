"""Conventions of the scalar ``G`` bases (internal; not part of the public API).

The scalar ``G`` is the z <-> zbar-even (parity-even) block of scalar externals,

    G_{Delta, ell} = [B(h, hbar) + B(hbar, h)] / (1 + delta_{ell, 0})

at every integer ``ell``.  For scalar externals the correlator is symmetric under z <-> zbar, so ``B(h, hbar)``
and ``B(hbar, h)`` always enter with equal coefficients; the z <-> zbar-odd structure ``B(h, hbar) - B(hbar, h)``
belongs to parity-odd correlators and is the ``parity_projected_B`` basis with ``eta = -1``.  Written scalar-``G``
tables and towers record ``scalar_G_odd = "even"`` (:data:`SCALAR_G_ODD_KEY`), and the readers refuse odd-``ell``
data that lacks it.

The division by ``1 + delta_{ell, 0}`` makes the spin-0 block ``B(h, h) = F(h, z) F(h, zbar)``, so that with the
identity coefficient 1 the sum ``sum_O p_O G_O`` with ``p_O = N^Delta lambda_O^2`` is the physical correlator in
every channel.  The spin-0 normalization is a process-wide setting (``"single"`` by default): forked workers inherit
it, the process pools pass it to their workers explicitly (:func:`pool_initializer_kwargs`), and every written
scalar-``G`` table records it (``scalar_G_spin0``).

Only the scalar-``G`` builders read this setting: ordered ``B``, parity-projected ``B``, chiral ``F``, the
large-spin towers, and the torus blocks do not depend on it.
"""

from __future__ import annotations

from contextlib import contextmanager

SCALAR_G_SPIN0_SINGLE = "single"
SCALAR_G_SPIN0_DOUBLE = "double"
SCALAR_G_SPIN0_CONVENTIONS = (SCALAR_G_SPIN0_SINGLE, SCALAR_G_SPIN0_DOUBLE)
# The metadata key under which written tables and manifests record the convention.
SCALAR_G_SPIN0_KEY = "scalar_G_spin0"
# The metadata key (and its only value) recording that odd-ell blocks are z <-> zbar even: B(h, hbar) + B(hbar, h).
SCALAR_G_ODD_KEY = "scalar_G_odd"
SCALAR_G_ODD_EVEN = "even"

_convention = SCALAR_G_SPIN0_SINGLE


def _checked(convention) -> str:
    if convention not in SCALAR_G_SPIN0_CONVENTIONS:
        raise ValueError(f"scalar_G spin-0 convention must be one of {SCALAR_G_SPIN0_CONVENTIONS}, "
                         f"got {convention!r}")
    return convention


def scalar_G_spin0_convention() -> str:
    """The process's scalar-``G`` spin-0 convention (``"single"`` by default)."""
    return _convention


def set_scalar_G_spin0_convention(convention) -> str:
    """Select the process-wide convention and return the previous one (internal)."""
    global _convention
    previous, _convention = _convention, _checked(convention)
    return previous


@contextmanager
def scalar_G_spin0_scope(convention):
    """Run a block of code under ``convention``, restoring the previous selection afterwards (internal)."""
    previous = set_scalar_G_spin0_convention(convention)
    try:
        yield
    finally:
        set_scalar_G_spin0_convention(previous)


def select_double_scalar_G_convention() -> None:
    """Select the undivided spin-0 normalization for the whole process (internal)."""
    set_scalar_G_spin0_convention(SCALAR_G_SPIN0_DOUBLE)


def scalar_G_swap_coefficient(ell) -> int:
    """The coefficient ``sigma`` of the reversed product ``B(hbar, h)`` in the scalar ``G`` of spin ``ell``.

    ``+1`` for every ``ell != 0``, even or odd: the scalar ``G`` is z <-> zbar even at every spin.  At ``ell = 0``,
    where the reversed product equals the forward one, it is ``0`` (``G = B(h, h)``).
    """
    ell = int(ell)
    if ell == 0:
        return 1 if _convention == SCALAR_G_SPIN0_DOUBLE else 0
    return 1


def scalar_G_fold_parity(ell) -> int:
    """The z <-> zbar parity marker of the scalar ``G`` of spin ``ell`` (0 even, 1 odd): 0 at every spin.

    The parity-folded coordinates (``xt``, ``ws``, ``qqb_ws``) strip one ``r = (u - ubar)/2`` from an odd block;
    a scalar ``G`` never needs it.
    """
    return 0 if scalar_G_swap_coefficient(ell) >= 0 else 1


def odd_scalar_G_data_without_record(doc) -> bool:
    """Whether a scalar-``G`` table or tower document holds odd-``ell`` data without the ``scalar_G_odd`` record.

    It does when it lacks ``scalar_G_odd = "even"`` and is an odd tower (``parity_exchanged = -1``) or has an odd
    ``spin``.  Even-spin data without the record is unaffected.
    """
    if doc.get(SCALAR_G_ODD_KEY) == SCALAR_G_ODD_EVEN:
        return False
    from .parity import normalize_parity_phase

    try:
        exchanged = normalize_parity_phase(str(doc.get("parity_exchanged") or "1"))
    except ValueError:
        exchanged = None        # an invalid phase is reported with the other metadata problems
    if exchanged == "-1":
        return True
    from fractions import Fraction

    try:
        spin = Fraction(str(doc.get("spin")))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False        # an invalid spin is reported with the other metadata problems
    return spin.denominator == 1 and spin.numerator % 2 == 1


def check_scalar_G_odd_record(doc, where="block file") -> None:
    """Refuse odd-``ell`` scalar-``G`` data without the record (see :func:`odd_scalar_G_data_without_record`)."""
    if odd_scalar_G_data_without_record(doc):
        raise ValueError(
            f"{where} holds odd-spin scalar_G data without {SCALAR_G_ODD_KEY} = {SCALAR_G_ODD_EVEN!r}, so its "
            "odd spins are not known to be the parity-even B(h, hbar) + B(hbar, h); regenerate it with the current "
            "virasoro_blocks")


def scalar_G_spin0_weight() -> int:
    """The coefficient of ``B(h, h)`` in the spin-0 scalar ``G`` (1 by default)."""
    return 1 + scalar_G_swap_coefficient(0)


def _initialize_worker(convention, initializer=None, initargs=()):
    set_scalar_G_spin0_convention(convention)
    if initializer is not None:
        initializer(*initargs)


def pool_initializer_kwargs(initializer=None, initargs=()) -> dict:
    """``initializer`` / ``initargs`` for a process pool whose workers follow this process's convention.

    A worker's own ``initializer(*initargs)``, if any, runs after the convention is set.
    """
    return {"initializer": _initialize_worker, "initargs": (_convention, initializer, tuple(initargs))}
