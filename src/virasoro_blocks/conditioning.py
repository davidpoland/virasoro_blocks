"""Digits lost near poles, endpoints, or a cancellation between block components.

Every computation runs at the working precision ``P``.  An exchanged weight ``x`` near a pole loses about
``L = log10(max(|x|, |pole|) / |x - pole|)`` digits to the cancellation ``x - pole``; a point near ``z = 1`` loses
about ``log10(1/|1 - z|)`` in the nome. Adding block components loses another
``log10(max(|a|, |b|)/|a + b|)`` digits when they cancel. The evaluators report that measured conditioning:

* ``L > P/2``: a :class:`PrecisionLossWarning` naming the digits left and the precision that carries ``P`` of them;
* ``L >= P - 1`` (no correct digit): :class:`InsufficientPrecisionError`.

An input exactly on a pole or on ``z = 1`` is left to the evaluator's own handling (a pole error, a removable limit,
the OPE-point refusal).  A request with ``target_digits = D`` uses the same measurement to run once at
``max(P, D + L)`` (:func:`target_precision`).
"""
from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import mpmath as mp


class PrecisionLossWarning(UserWarning):
    """An ill-conditioned evaluation keeps fewer than half of the working digits."""


class InsufficientPrecisionError(ValueError):
    """An ill-conditioned evaluation keeps no correct digit at the working precision."""

    def __init__(self, message, *, digits_lost, precision):
        super().__init__(message)
        self.digits_lost = digits_lost
        self.precision = precision
        self.needed_precision = int(math.ceil(precision + digits_lost))


_HANDLED = ContextVar("vb_conditioning_handled", default=False)
_CAPTURE = ContextVar("vb_conditioning_capture", default=None)


@contextmanager
def handled():
    """Inside, the evaluators do not report: the caller (a request with its own report) measured it already."""
    token = _HANDLED.set(True)
    try:
        yield
    finally:
        _HANDLED.reset(token)


@contextmanager
def capture(*, silent: bool):
    """Record the largest loss reported inside (``state["loss"]``); ``silent`` also suppresses the warning/error,
    for a caller that escalates the precision itself (``target_digits``)."""
    state = {"loss": 0.0, "silent": silent}
    token = _CAPTURE.set(state)
    try:
        yield state
    finally:
        _CAPTURE.reset(token)


def quiet(function):
    """Decorator: a table build (verified on its own) evaluates near poles on purpose; no reports inside."""
    from functools import wraps

    @wraps(function)
    def wrapped(*args, **kwargs):
        with handled():
            return function(*args, **kwargs)
    return wrapped


def _convert(value):
    from .numeric_input import parse_number

    value = parse_number(value)
    return value if isinstance(value, (mp.mpf, mp.mpc)) else mp.convert(value)


def pole_loss(x, poles, exact_pole=None) -> float:
    """Digits lost to ``x - pole`` for the nearest of ``poles``.

    ``poles`` is a sequence of numbers, or a float64 complex array with ``exact_pole(i)`` giving pole ``i`` at the
    working precision.  A float64 scan picks the nearest pole; only a candidate within ``1e-12`` relative is measured
    at the working precision.  A pole at 0 costs nothing.  Exact labels exactly on an exact pole give 0 (the
    evaluator's own pole handling decides); an input that rounds onto a computed pole has lost every digit."""
    if _HANDLED.get():
        return 0.0                  # (a quiet table build: not measured, so not paid for)
    import numpy as np

    approx = np.asarray(poles if exact_pole is not None else [complex(_convert(p)) for p in poles], dtype=complex)
    if approx.size == 0:
        return 0.0
    try:
        x64 = complex(_convert(x))
    except (OverflowError, ValueError, TypeError):
        return 0.0
    if not math.isfinite(abs(x64)):
        return 0.0
    scale = np.maximum(abs(x64), np.abs(approx))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(scale > 0, np.abs(approx - x64) / np.where(scale > 0, scale, 1.0), np.inf)
    i = int(np.argmin(relative))
    if relative[i] > 1e-12:
        return max(0.0, -math.log10(relative[i])) if math.isfinite(relative[i]) else 0.0
    from fractions import Fraction
    from .numeric_input import parse_number

    pole_label = exact_pole(i) if exact_pole is not None else parse_number(poles[i])
    label = parse_number(x)
    if isinstance(label, (int, Fraction)) and isinstance(pole_label, (int, Fraction)):
        if label == pole_label:
            return 0.0              # exactly on the pole: the evaluator's own handling decides
    value, pole = _convert(label), _convert(pole_label)
    if value == 0 and pole == 0:
        return 0.0
    distance = abs(value - pole)
    if distance == 0:
        return float(mp.mp.dps)     # on a computed pole to roundoff: no digit of the offset survives
    return max(0.0, float(mp.log10(max(abs(value), abs(pole)) / distance)))


def endpoint_loss(z) -> float:
    """Digits lost in the nome at a point near ``z = 1`` (``log10(1/|1 - z|)``); 0 away from it and at z = 1."""
    if _HANDLED.get():
        return 0.0
    from .numeric_input import parse_number

    z = parse_number(z)
    try:
        coarse = abs(1 - complex(_convert(z)))
    except (OverflowError, ValueError, TypeError):
        return 0.0
    if coarse >= 0.1:
        return 0.0
    distance = mp.convert(abs(1 - z))
    if distance == 0:
        return 0.0
    return max(0.0, float(-mp.log10(distance)))


def _carries_extra_bits(value) -> bool:
    """An mpmath input stored at more bits than the working precision: some evaluators resolve its offset."""
    from .numeric_input import parse_number

    value = parse_number(value)
    if isinstance(value, mp.mpf):
        return value._mpf_[3] > mp.mp.prec
    if isinstance(value, mp.mpc):
        return max(value._mpc_[0][3], value._mpc_[1][3]) > mp.mp.prec
    return False


def report(loss: float, *, what: str, source=None) -> float:
    """Warn or raise for ``loss`` digits lost at the working precision (unless :func:`handled`); returns ``loss``.

    ``source`` is the input whose offset is measured: one stored at more bits than the working precision may still
    be resolved by the evaluator, so it gets the warning rather than the error."""
    if _HANDLED.get() or loss <= 0:
        return loss
    state = _CAPTURE.get()
    if state is not None:
        state["loss"] = max(state["loss"], loss)
        if state["silent"]:
            return loss
    precision = mp.mp.dps
    if loss >= precision - 1 and not (source is not None and _carries_extra_bits(source)):
        raise InsufficientPrecisionError(
            f"{what}: no correct digit at precision {precision} (~{loss:.0f} digits lost to cancellation); "
            f"raise the precision to about {int(math.ceil(precision + loss))} digits, or pass "
            "target_digits", digits_lost=loss, precision=precision)
    if loss > precision / 2:
        warnings.warn(PrecisionLossWarning(
            f"{what}: ~{loss:.0f} of {precision} digits lost to cancellation (about "
            f"{precision - loss:.0f} correct); precision {int(math.ceil(precision + loss))} carries {precision}"),
            stacklevel=3)
    return loss


def conditioned_sum(first, second, *, exact_zero=False):
    """Add two block components and account for cancellation of their significant digits.

    ``exact_zero`` must follow from the input labels and a symmetry, never from equality of rounded values.
    Otherwise a vanishing sum of nonzero components means the current precision cannot resolve the answer.
    The composition amplifies the conditioning loss already captured from its components.
    """
    if exact_zero:
        return mp.mpf(0)
    value = first + second
    if _HANDLED.get():
        return value
    scale = max(abs(first), abs(second))
    if scale:
        loss = float(mp.mp.dps) if not value else max(0.0, float(mp.log10(scale / abs(value))))
        if loss > 0:
            state = _CAPTURE.get()
            report(loss + (state["loss"] if state is not None else 0.0), what="block composition")
    return value


def target_precision(loss: float, target_digits: int, precision: int, max_precision: int) -> int:
    """The single working precision that delivers ``target_digits`` despite ``loss``: ``max(P, D + L)``."""
    from .verified_build import TargetDigitsError

    need = max(int(precision), int(math.ceil(target_digits + loss)))
    if need > max_precision:
        raise TargetDigitsError(
            f"{target_digits} digits need precision {need} here (~{loss:.0f} lost to the nearby pole or endpoint), "
            f"above max_precision {max_precision}; raise max_precision")
    return need
