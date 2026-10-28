"""Working precision and accuracy verification for table-writing paths.

``--precision P`` sets the table's working and output precision. Verification
compares it with numeric samples or a higher-precision construction and records:

* ``verified_digits``: the worst per-entry relative agreement over the check
  points. Entries smaller than the reference's resolution are compared using
  that absolute resolution.
* ``verified_digits_largest_entry``: the worst agreement relative to the
  largest entry at each point. This measures the table's overall accuracy when
  subleading entries, such as folded t-derivatives at large Delta, lose relative
  accuracy.

Check ranges depend on the family. They include points between interpolation
nodes and, where supported, extend into the large-Delta regime used by SDPB.
Exact tables are checked against the same construction at higher precision.

A shortfall produces one message identifying its cause and remedy
(:func:`shortfall_message`): increase working precision for cancellation,
``--npoles`` for interpolation error, or ``--order`` for q-series truncation.
A keep-all table with an incorrect pole structure cannot be repaired by
increasing numerical precision.

``--target-digits D`` requests at least D verified digits. The build starts at
P and increases working precision according to the measured deficit, up to
``--max-precision M``. It stops with an explanatory error if the target cannot
be reached (:func:`run_verified_build`).
"""

from __future__ import annotations

import sys
from math import hypot, isfinite
from dataclasses import dataclass, replace
from typing import Any, Callable

import mpmath as mp

from .conditioning import quiet as _quiet_conditioning


# ---------------------------------------------------------------------------------------------------------------
# The precision request
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KnobNames:
    """How the user spells each knob a message names (the CLI's flags, the bootstrap spec's fields)."""

    precision: str = "--precision"
    target: str = "--target-digits"
    max_precision: str = "--max-precision"
    npoles: str = "--npoles"
    order: str = "--order"
    #: the alternative to raising the pole count, or None where there is none
    keep_all: str | None = "omit it to keep every pole"
    assign: str = "{name} {value}"      # how a value is given: "--precision 90", "dps=90"

    def set(self, name: str, value) -> str:
        return self.assign.format(name=name, value=value)

    def more_poles(self) -> str:
        """``raise --npoles[, or <keep_all>]``: what fixes a table limited by its kept poles."""
        return f"raise {self.npoles}" + ("" if self.keep_all is None else f", or {self.keep_all}")


CLI_KNOBS = KnobNames()
BOOTSTRAP_KNOBS = KnobNames(precision="dps", target="target_digits", max_precision="max_dps", npoles="npoles",
                            order="q_order", keep_all='use npoles="all" to keep every pole', assign="{name}={value}")


def default_max_precision(precision: int, target_digits: int) -> int:
    """The escalation cap when ``--max-precision`` is not given: ``max(4 D, 2 P, D + 200)``.

    The escalation stops by itself when precision is not the limit (an attempt that closes less than a quarter of
    the remaining gap), so the cap only bounds the cost of a genuine precision shortfall: a monomial-form table loses
    ~0.6-0.75 digits per unit of degree, so 4 D carries D digits through a table of degree up to ~4 D, and D + 200
    through one of degree ~270 at any D (the keep-all tables built here reach degree ~250 at lambda 19, order 60;
    a keep-all lambda 5 table at dps 80 needs ~170 for 40 digits); a build at 4 D costs about 10x one at D.
    (2 P: a start above 4 D still leaves room to escalate.)
    """
    return max(4 * int(target_digits), 2 * int(precision), int(target_digits) + 200)


def verification_margin(digits: int) -> int:
    """Digits the check points must show above ``digits``: they sample the table, and between them it was up to ~2
    digits worse (40-digit tables at 38.5-39.5 elsewhere)."""
    return max(3, int(digits) // 20)


@dataclass(frozen=True)
class PrecisionRequest:
    """``precision`` is the working precision (with a target, where the build starts); ``target_digits`` the opt-in
    number of verified digits to deliver; ``max_precision`` the escalation cap (default
    :func:`default_max_precision`)."""

    precision: int
    target_digits: int | None = None
    max_precision: int | None = None
    knobs: KnobNames = CLI_KNOBS

    @classmethod
    def make(cls, precision, target_digits=None, max_precision=None, *, knobs: KnobNames = CLI_KNOBS):
        """A validated request (``ValueError`` names the offending knob): ``max_precision`` needs a target, and the
        target must fit under the cap, which must not be below the starting precision."""
        from .requests import _integer_field as integer

        precision = integer(knobs.precision, precision)
        if target_digits is not None:
            target_digits = integer(knobs.target, target_digits)
            if target_digits < 1:
                raise ValueError(f"{knobs.target} must be >= 1, got {target_digits}")
        if max_precision is not None:
            max_precision = integer(knobs.max_precision, max_precision)
            if target_digits is None:
                raise ValueError(
                    f"{knobs.max_precision} caps the escalation that {knobs.target} asks for; without "
                    f"{knobs.target} the table is built at {knobs.set(knobs.precision, precision)} and nothing "
                    "escalates")
            if max_precision < precision:
                raise ValueError(f"{knobs.set(knobs.max_precision, max_precision)} is below "
                                 f"{knobs.set(knobs.precision, precision)}, "
                                 "where the build starts")
            if target_digits > max_precision:
                raise ValueError(
                    f"{knobs.set(knobs.target, target_digits)} exceeds {knobs.set(knobs.max_precision, max_precision)}: "
                    "a table cannot "
                    "deliver more digits than its working precision carries")
        return cls(precision, target_digits, max_precision, knobs)

    @property
    def digits(self) -> int:
        """The digits the verification measures against: the target, else the working precision."""
        return int(self.precision if self.target_digits is None else self.target_digits)

    @property
    def cap(self) -> int:
        if self.target_digits is None:
            return int(self.precision)
        if self.max_precision is not None:
            return int(self.max_precision)
        return default_max_precision(self.precision, self.target_digits)

    @property
    def input_dps(self) -> int:
        """Digits for parsing inputs and deriving labels, not for building blocks.

        Keep decimal/rational inputs above every permitted working precision,
        with guard digits for verification. Otherwise escalation verifies a block at the
        initially rounded labels instead of the labels the caller supplied.
        """
        return max(3 * self.precision + 50, self.cap + 50)

    @property
    def goal(self) -> int:
        """Digits the check points must show for the target to count as delivered."""
        return self.digits + verification_margin(self.digits)

    def short(self, digits) -> bool:
        """Whether ``digits`` verified falls short: below the goal with a target, else more than the margin below
        the working precision (a table at P digits cannot show P at its check points)."""
        if digits is None:
            return False
        if self.target_digits is not None:
            return digits < self.goal
        return digits < self.precision - verification_margin(self.precision)


def precision_request_of(obj, *, knobs: KnobNames = CLI_KNOBS) -> PrecisionRequest:
    """The :class:`PrecisionRequest` of a CLI namespace (``args.precision_request``) or a numerical policy (its
    ``precision`` / ``target_digits`` / ``max_precision``); the current precision when it carries none."""
    request = getattr(obj, "precision_request", None)
    if isinstance(request, PrecisionRequest):
        return request
    precision = getattr(obj, "precision", None) or mp.mp.dps
    return PrecisionRequest.make(precision, getattr(obj, "target_digits", None), getattr(obj, "max_precision", None),
                                 knobs=knobs)


# ---------------------------------------------------------------------------------------------------------------
# Check points and agreement measures
# ---------------------------------------------------------------------------------------------------------------


def check_points_x(sample_points_x, *, beyond_last=True):
    """Points between interpolation nodes (and, with ``beyond_last``, past the last) where a table is checked.

    About one gap per eighth of the node range (5 points missed a 39.5-digit spot in a 40-digit table).  A table
    without nodes (an exact one) is checked at 0.31 and 2.3.  (The Virasoro tables are also checked past their nodes
    out to the large-Delta regime: :func:`far_check_points_x`.)
    """
    xs = sorted(mp.re(mp.mpc(x)) for x in sample_points_x)
    if len(xs) < 2:
        return [mp.mpf("0.31"), mp.mpf("2.3")]
    last_gap = len(xs) - 2
    picks = sorted({min(last_gap, (k * len(xs)) // 8) for k in range(8)} | {last_gap})
    points = [(xs[i] + xs[i + 1]) / 2 for i in picks]
    if beyond_last:
        points.append(xs[-1] * mp.mpf(5) / 4)
    return points


def far_check_points_x(sample_points_x, poles_x):
    """Check points past the last node, out to where the table is in its large-Delta regime.

    SDPB imposes positivity for every x >= 0, so a table is used far past its last node L.  There the monomial
    form is an extrapolation of its node values: their rounding is amplified most where the Delta-poles sit (a
    spinning c = 3.3 table, last node 62, poles to |x| = 269: 51 digits at 60, 11 at 270, 17 from 5000 on -- the
    leading coefficient's accuracy, which fixes every larger x), while its values between the nodes were fine.
    The points double from 1.25 L to past 16 R, R = max(L, |poles|): past ~16 R the error has settled to that
    large-Delta plateau (within half a digit up to x = 1e7 R in the measured zzb tables; a folded coordinate's
    t-derivative entries, which the fold makes subleading at large Delta, drift on by 2-4 digits a decade).  A point
    landing on (or next to) a pole is moved off it.
    """
    nodes = [mp.mpc(x) for x in sample_points_x]
    poles = [mp.mpc(p) for p in poles_x]
    if any(not mp.isfinite(value) for value in (*nodes, *poles)):
        raise ValueError("verification nodes and poles must be finite")
    xs = sorted(value.real for value in nodes)
    # A constant/degenerate node set can end at zero. Start beyond it on the
    # positive half-line, where doubling makes progress toward the pole scale.
    last = xs[-1] if xs and xs[-1] > 0 else mp.mpf("2.3")
    native_poles = [complex(pole) for pole in poles]
    magnitudes = [hypot(pole.real, pole.imag) for pole in native_poles]
    native_range = all(isfinite(magnitude) and (magnitude != 0 or pole == 0)
                       for magnitude, pole in zip(magnitudes, poles))
    scale = max([last] + ([mp.mpf(value) for value in magnitudes] if native_range
                         else [abs(pole) for pole in poles]))

    def near_pole(value):
        # A 1e-3 proximity check needs only coarse accuracy, but float must
        # retain the magnitudes. Keep the common case fast and fall back for
        # large/tiny finite inputs, including an underflowed check radius.
        native = float(value)
        if native_range and isfinite(native) and native / 1000 > 0:
            radius = native / 1000
            return any(hypot(native - pole.real, pole.imag) < radius for pole in native_poles)
        radius = value / 1000
        return any(abs(value - pole) < radius for pole in poles)

    points, x = [], last * mp.mpf(5) / 4
    while True:
        while near_pole(x):
            advanced = x * mp.mpf("1.01")
            if advanced <= x:
                raise ValueError("increase working precision to choose verification points off the poles")
            x = advanced
        points.append(x)
        if x >= 16 * scale:
            return points
        x *= 2


def _flint_bits(dps: int) -> int:
    return int(int(dps) * 3.33) + 20


def _as_acb(value):
    from flint import acb, arb

    from .polynomial import _mpf_to_acb

    if isinstance(value, acb):
        return value
    if isinstance(value, arb):
        return acb(value)
    return _mpf_to_acb(value)


def _log10_abs(value) -> float:
    """log10 |value| of an acb (its midpoint), -inf for zero; in arb, so no float underflow (a block damped by
    base^x is ~1e-13000 at x = 1e4)."""
    from flint import arb

    magnitude = abs(value.mid())
    if magnitude == 0:
        return float("-inf")
    return float((magnitude.log() / arb(10).log()).mid())


def _entry(table, m, n):
    return table[(m, n)] if isinstance(table, dict) else table[m][n]


def point_agreement(got, ref, kept, floor_digits: int):
    """``(per-entry digits, largest-entry digits)`` of one table ``got`` against ``ref`` at one check point over the
    entries ``kept``, or None (both rows are zero, or no entries are selected).

    Per entry: relative, except that entries more than ``floor_digits`` below the largest are compared at that
    absolute level.  Largest entry: the worst error relative to the largest entry of the point.  The tables are
    nested ``[m][n]`` lists or ``{(m, n): value}`` dicts of mpmath or flint values; the comparison runs in flint at
    the current precision. When the whole reference row is zero, use absolute
    error. Non-finite entries are errors, never omitted from the comparison.
    """
    from flint import ctx

    ctx.prec = max(ctx.prec, _flint_bits(mp.mp.dps))
    logs = []
    for m, n in kept:
        g, r = _as_acb(_entry(got, m, n)), _as_acb(_entry(ref, m, n))
        if not g.is_finite() or not r.is_finite():
            raise ValueError(f"non-finite verification entry ({m}, {n})")
        logs.append((_log10_abs(g - r), _log10_abs(r)))
    if not logs:
        return None
    scale = max(lr for _ld, lr in logs)
    if scale == float("-inf"):
        error = max(ld for ld, _lr in logs)
        return None if error == float("-inf") else (-error, -error)
    floor = scale - int(floor_digits)
    exact = float(10 * int(floor_digits))
    per_entry = max(ld - max(lr, floor) for ld, lr in logs)
    largest = max(ld for ld, _lr in logs) - scale
    return (exact if per_entry == float("-inf") else -per_entry), (exact if largest == float("-inf") else -largest)


def table_values(serialized, coordinate, deltas, kept, *, dps: int):
    """``[{(m, n): acb}]``: the stored ``serialized`` table's entries ``kept`` at each Delta of ``deltas``, evaluated
    from its coefficients at ``dps`` digits in flint (the consumer's evaluation, :meth:`SerializedDerivTable.
    evaluate`, without its per-entry mpmath round trips)."""
    from flint import acb, ctx

    from .polynomial import _mpf_to_acb

    with mp.workdps(int(dps)):
        ctx.prec = _flint_bits(dps)
        rows = serialized.numerators[coordinate]
        polys = [((m, n), rows[m][n]._get_acb()) for m, n in kept]
        ctx.prec = _flint_bits(dps)
        base = _mpf_to_acb(mp.mpc(serialized.base))
        poles = [_mpf_to_acb(mp.mpc(pole)) for pole in serialized.pole_list_x]
        shift = mp.mpc(serialized.shift)
        out = []
        for delta in deltas:
            x = _mpf_to_acb(mp.mpc(delta) - shift)
            denominator = acb(1)
            for pole in poles:
                denominator *= x - pole
            prefactor = base ** x / denominator
            out.append({entry: prefactor * poly(x) for entry, poly in polys})
    return out


def agreement_digits(gots, refs, kept, floor_digits: int):
    """Worst per-entry digits of agreement of the tables ``gots`` with ``refs`` (paired, one per check point), or
    None (both tables zero at every point); see :func:`point_agreement`."""
    worst = None
    for got, ref in zip(gots, refs):
        measured = point_agreement(got, ref, kept, floor_digits)
        if measured is not None and (worst is None or measured[0] < worst):
            worst = measured[0]
    return worst


@dataclass(frozen=True)
class Verification:
    """What a table delivers, measured at its check points.

    ``digits`` / ``largest_digits``: the worst per-entry and largest-entry agreement (``verified_digits`` and
    ``verified_digits_largest_entry``).  ``node_digits``: the worst between the nodes.  ``worst_x``, ``far_x``,
    ``last_node``: the point where it is worst, the farthest check point and the last node, in the table's variable
    less its shift.  ``span``: the range of the table's own variable the check covered (recorded as
    ``verified_delta_range`` / ``verified_spin_range``).  The diagnosis of a shortfall (:func:`diagnose_monomial`)
    fills ``cause`` ("precision", "npoles", "poles", "order", "offset"), the measured cancellation ``lost`` (at
    ``lost_x``) and the ``needed_precision`` that carries the reference digits through it; ``detail`` is extra
    family text for the message.
    """

    digits: float | None
    largest_digits: float | None = None
    node_digits: float | None = None
    worst_x: Any = None
    far_x: Any = None
    last_node: Any = None
    points: tuple = ()
    span: tuple | None = None
    variable: str = "x = Delta - shift"
    cause: str | None = None
    lost: float | None = None
    lost_x: Any = None
    needed_precision: int | None = None
    mechanism: str | None = None
    degree: int | None = None
    fit_digits: float | None = None
    offset_eps: int | None = None       # (an offset singular c: the exponent of c + 10^-e, and what it leaves)
    offset_digits: float | None = None
    detail: str = ""
    where_text: str | None = None

    @property
    def where(self) -> str:
        if self.where_text is not None:
            return self.where_text
        if self.far_x is None:
            return "at its check points"
        text = f"over {self.variable} in [0, {mp.nstr(self.far_x, 3)}]"
        name = self.variable.split(" ")[0]
        if self.node_digits is not None and self.digits is not None and self.node_digits > self.digits + 1:
            if self.last_node is None:      # (a table without nodes: checked near x = 0 and far out)
                text += f" (worst at {name} = {mp.nstr(self.worst_x, 3)})"
            else:
                text += (f" (between its nodes {self.node_digits:.1f}; worst at {name} = "
                         f"{mp.nstr(self.worst_x, 3)}, past the last node {mp.nstr(self.last_node, 3)})")
        return text


def verification_from_points(per_point, points, *, n_inner: int, last_node=None, shift=0, variable=None,
                             far_x=None) -> Verification | None:
    """A :class:`Verification` from ``per_point`` ``(per-entry, largest-entry)`` digits (None: nothing measured
    there) at ``points`` (x = variable - shift), the first ``n_inner`` of them between the nodes."""
    measured = [(d, x) for d, x in zip(per_point, points) if d is not None]
    if not measured:
        return None
    (digits, largest), worst_x = min(measured, key=lambda item: item[0][0])
    largest = min(d[1] for d, _x in measured)
    inner = [d[0] for d in per_point[:n_inner] if d is not None]
    far = far_x if far_x is not None else max(mp.re(mp.mpc(x)) for x in points)
    shift = mp.re(mp.mpc(shift))
    kwargs = {} if variable is None else {"variable": variable}
    return Verification(digits=digits, largest_digits=largest, node_digits=min(inner) if inner else None,
                        worst_x=worst_x, far_x=far, last_node=last_node, points=tuple(points),
                        span=(shift, shift + far), **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# Diagnosis: what limits a table
# ---------------------------------------------------------------------------------------------------------------


def precision_limit_digits(serialized, points, floor_digits: int, work: int):
    """Digits the table's monomial coefficients can carry at the check ``points`` at working precision ``work``,
    and the x where that is fewest: ``work`` less the cancellation ``log10(sum_k |c_k| |x|^k / |P(x)|)`` of its
    worst entry (entries more than ``floor_digits`` below the largest at that level, as the verification does).

    Measured from the stored coefficients alone (no reference): a table short of this is short for another reason
    (its kept poles), one at it is limited by the precision it was built at.
    """
    from flint import arb, arb_poly

    from .polynomial import _mpf_to_arb

    xs = [mp.re(mp.mpc(x)) for x in points]
    worst, worst_x = None, None
    with mp.workdps(max(mp.mp.dps, int(work)) + 10):
        at = [_mpf_to_arb(x) for x in xs]
        tenth = arb(10) ** (-int(floor_digits))
        for rows in serialized.numerators.values():
            # per point: (sum_k |c_k| |x|^k, |P(x)|) of every entry (the prefactor is common to them and cancels)
            pairs = [[] for _ in xs]
            for row in rows:
                for poly in row:
                    acb_p = poly._get_acb()
                    abs_p = arb_poly([abs(coeff) for coeff in acb_p.coeffs()])
                    for i, x in enumerate(at):
                        pairs[i].append((abs_p(x), abs(acb_p(x))))
            for x, entries in zip(xs, pairs):
                scale = max((value for _bound, value in entries), default=arb(0))
                if not scale > 0:
                    continue
                floor = scale * tenth
                ratio = max(bound / max(value, floor) for bound, value in entries)
                lost = float((ratio.log() / arb(10).log()).mid()) if ratio > 1 else 0.0
                digits = int(work) - lost
                if worst is None or digits < worst:
                    worst, worst_x = digits, x
    return worst, worst_x


def node_residual_digits(serialized, floor_digits: int, kept_by_coord=None):
    """Digits to which a table's stored coefficients reproduce the values it was fit to at its own nodes (its
    recorded ``sample_block_values``), or None (none recorded).

    Every entry's recorded values are its own interpolant at the nodes, so in exact arithmetic the table matches
    them whatever its order reduction: what it misses is the rounding of the fit (a degree-d monomial form built
    from node values loses ~0.6-0.75 digits per unit of degree) and of its evaluation.  Evaluated at twice the
    working precision, as the verification does, at about one node per eighth of the node range and the last.
    """
    sbv = getattr(serialized, "sample_block_values", None)
    xs = getattr(serialized, "sample_points_x", None)
    if not sbv or not xs:
        return None
    worst = None
    for coord, tables in sbv.items():
        if coord not in serialized.numerators:
            continue
        kept = list((kept_by_coord or {}).get(coord) or ())
        if not kept:
            rows = serialized.numerators[coord]
            kept = [(m, n) for m, row in enumerate(rows) for n, poly in enumerate(row)
                    if any(coeff != 0 for coeff in poly.coeffs)]
        # about one node per eighth of the range, and the last (the fit's rounding varies slowly along the nodes)
        picks = sorted({(k * len(xs)) // 8 for k in range(8)} | {len(xs) - 1})
        gots = table_values(serialized, coord, [mp.mpf(serialized.shift) + xs[i] for i in picks], kept,
                            dps=2 * mp.mp.dps)
        for got, ref in zip(gots, [tables[i] for i in picks]):
            measured = point_agreement(got, ref, kept, floor_digits)
            if measured is not None and (worst is None or measured[0] < worst):
                worst = measured[0]
    return worst


def _max_degree(serialized) -> int:
    return max((poly.degree for rows in serialized.numerators.values() for row in rows for poly in row), default=0)


def diagnose_monomial(serialized, verification: Verification, *, digits: int, work: int,
                      order_reduced: bool, kept_by_coord=None) -> Verification:
    """Name what limits a monomial-form table that falls short of ``digits``.

    A keep-all table represents the (q-truncated) block exactly, so what it misses is the working precision's
    rounding -- the fit of its degree-d monomial form and its evaluation, ~0.6-0.75 digits per unit of degree, most
    of it amplified past the nodes -- and ``needed_precision`` adds the measured shortfall to the digits asked for.
    (A pole structure that disagrees with the block would look the same; only an escalation that stops helping
    tells them apart, and :func:`run_verified_build` says so.)

    An order-reduced table is precision-limited when it is within 3 digits of what its coefficients carry, measured
    two ways: their cancellation on evaluation at the check points (:func:`precision_limit_digits`) and how well
    they reproduce the values they were fit to at the nodes (:func:`node_residual_digits`); ``needed_precision`` =
    digits + that loss + the margin.  Otherwise its kept poles limit it ("npoles").
    """
    delivered = verification.digits
    margin = verification_margin(digits)
    degree = _max_degree(serialized)
    if not order_reduced:
        lost = float(int(work) - delivered)
        needed = int(mp.ceil(digits + lost + margin))
        return replace(verification, cause="precision", lost=lost, lost_x=verification.worst_x,
                       needed_precision=needed, mechanism="rounding", degree=degree)
    points = verification.points or tuple(check_points_x(getattr(serialized, "sample_points_x", None) or ()))
    measured = []
    limit, limit_x = precision_limit_digits(serialized, points, digits, work)
    if limit is not None:
        measured.append((limit, "cancellation", limit_x))
    fit = node_residual_digits(serialized, digits, kept_by_coord)
    if fit is not None:
        measured.append((min(fit, float(work)), "fit", None))
    if measured and delivered is not None:
        limit, mechanism, limit_x = min(measured, key=lambda item: item[0])
        lost = None
        if limit < 3:
            # the coefficients are rounding noise and the measured loss saturates at the working precision: the
            # shortfall itself is then the better measure
            lost = max(float(int(work) - limit), float(int(work) - delivered))
        elif limit < digits and delivered >= limit - 3:
            # what the coefficients carry explains what the table delivers
            lost = float(int(work) - limit)
        elif fit is not None and fit < digits / 2 and (
            verification.node_digits is None or verification.node_digits >= fit - 3
        ):
            # the fit lost most of the working digits to rounding at its own nodes, and past them that rounding is
            # amplified (npoles 100 at dps 80: 9.4 digits at the nodes, 5.0 over the range): all of the shortfall is
            # taken as the precision's (an escalation that stops helping says otherwise).  Between the nodes the
            # interpolation is well conditioned, so there the rounding shows at about its node level (that case: 9.41
            # between them); a table far worse between its nodes than it fits them is limited by its kept poles
            # (Lambda 19, c = 1.00001, Delta_ext 0.4, spin 0: 113 digits at the nodes, -1.9 between them, the same at
            # twice the precision).
            lost, mechanism, limit_x = float(int(work) - delivered), "fit", None
        if lost is not None:
            needed = int(mp.ceil(digits + lost + margin))
            return replace(verification, cause="precision", lost=lost, lost_x=limit_x, needed_precision=needed,
                           mechanism=mechanism, degree=degree, fit_digits=fit)
    return replace(verification, cause="npoles", degree=degree)


# ---------------------------------------------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------------------------------------------


class TargetDigitsError(ValueError):
    """A table that cannot deliver ``--target-digits``: the message names why (the cap, a stall, the order
    reduction, the q-order, the pole structure) and the knob that would help."""


def _shown(v: Verification) -> str:
    digits = v.digits
    text = "fewer than 1 digit" if digits < 1 else f"{digits:.1f} digits"
    if v.largest_digits is not None and v.largest_digits > digits + 1:
        text += f" per entry ({v.largest_digits:.1f} against the largest entry)"
    return text


def reproduces(v: Verification) -> bool:
    """Whether the table reproduces the block at all: 3 digits against the largest entry at every check point
    (the per-entry measure when no largest-entry one was taken)."""
    measure = v.largest_digits if v.largest_digits is not None else v.digits
    return measure is None or measure >= 3


def shortfall_message(what: str, v: Verification, request: PrecisionRequest, work: int) -> str:
    """The one line reporting a table short of ``request``'s digits, with its cause and fix.

    Severity, the same for every family: a *warning* when the table does not reproduce the block (under 3 digits
    against the largest entry), when a keep-all table disagrees with its block beyond the precision, or when the
    precision is the limit and less than half the working digits survive; a *note* otherwise (the order reduction,
    the q-series truncation, a precision loss that leaves most of the digits).
    """
    k = request.knobs
    ref = request.digits
    shown = _shown(v)
    of = f"of {k.set(k.precision, work)}" if request.target_digits is None else f"of {k.set(k.target, ref)}"
    tail = f" {v.detail}" if v.detail else ""
    if v.cause == "precision":
        severe = not reproduces(v) or v.digits < ref / 2
        different = "" if reproduces(v) else ", and a bootstrap built on it imposes positivity on a different function"
        if v.mechanism == "cancellation":
            why = (f"its monomial coefficients cancel by ~{v.lost:.0f} digits on evaluation (at x = "
                   f"{mp.nstr(v.lost_x, 3)})")
        elif v.mechanism == "offset":
            why = (f"its c-offset c + 1e-{v.offset_eps} (the regularization at this singular c, which follows the "
                   f"working precision: half its digits) matches the c -> c0 limit to only {v.offset_digits:.1f} "
                   "digits")
        elif v.mechanism == "exact":
            why = f"it loses ~{v.lost:.0f} digits to rounding at working precision {work}"
        elif v.mechanism == "recursion":
            why = f"the one-point recursion cancels ~{v.lost:.0f} digits at working precision {work}"
        elif v.mechanism == "fit":
            fit = v.fit_digits if v.fit_digits is not None else int(work) - v.lost
            why = (f"its degree-{v.degree} monomial coefficients reproduce the values they were fit to at its nodes "
                   f"to only {fit:.1f} digits (the fit's rounding), amplified past them")
        else:
            why = (f"its degree-{v.degree} monomial form loses ~{v.lost:.0f} digits to rounding at working precision "
                   f"{work} (the fit and its evaluation lose ~0.6-0.75 digits per unit of degree, most of it past "
                   "the nodes)")
        return (f"{'warning' if severe else 'note'}: {what} reproduces the block to {shown} {of} {v.where}: {why}, "
                f"which is what limits it{different}.  Raise {k.precision} to about {v.needed_precision} (which "
                f"carries {ref} digits through it), or pass {k.set(k.target, ref)} to raise it automatically.{tail}")
    if v.cause == "npoles":
        if not reproduces(v):
            return (f"warning: {what} does not reproduce the block ({shown} {of} {v.where}): a bootstrap built on it "
                    f"imposes positivity on a different function.  R{k.more_poles()[1:]}.{tail}")
        return (f"note: {what} reproduces the block to {shown} {of} {v.where}; the order reduction limits it "
                f"({k.more_poles()}).{tail}")
    if v.cause == "order":
        severity = "note" if reproduces(v) else "warning"
        return (f"{severity}: {what} reproduces the block to {shown} {of} {v.where}; the q-series truncation limits "
                f"it (raise {k.order}).{tail}")
    if v.cause == "offset":
        severity = "note" if reproduces(v) else "warning"
        return (f"{severity}: {what} reproduces the block to {shown} {of} {v.where}; limited by the c-offset "
                f"c + 1e-{v.offset_eps} it was given (eps_exp), which matches the c -> c0 limit to only "
                f"{v.offset_digits:.1f} digits (leave eps_exp unset: the offset then follows the working "
                f"precision).{tail}")
    # "poles": a keep-all table that disagrees with its block where the precision is not the limit
    carried = "" if v.lost is None else f" (its coefficients carry ~{int(work) - v.lost:.0f} digits there)"
    return (f"warning: {what} matches its sampler to only {shown} {of} {v.where}, and not because of the "
            f"precision{carried}: the table's pole structure disagrees with the block.{tail}")


def unreachable_message(what: str, v: Verification, request: PrecisionRequest, work: int, *, why: str) -> str:
    """The error for a table that cannot reach ``--target-digits``."""
    k = request.knobs
    margin = ""
    if v.digits >= request.target_digits:
        # (its check points sample the table, which is up to ~2 digits worse between them)
        margin = (f", short of the {request.goal} its check points must show "
                  f"({request.target_digits} + a {request.goal - request.target_digits}-digit margin)")
    return (f"{what} cannot deliver {k.set(k.target, request.target_digits)}: it reproduces the block to {_shown(v)} "
            f"{v.where} at working precision {work}{margin}; {why}")


def _cause_hint(v: Verification, request: PrecisionRequest, *, order_reduced: bool) -> str:
    k = request.knobs
    if v.cause == "order":
        return f"the q-series truncation limits it, not the precision: raise {k.order}"
    if v.cause == "npoles" or (v.cause != "poles" and order_reduced):
        return f"the order reduction (its kept poles) limits it, not the precision: {k.more_poles()}"
    if v.cause == "offset":
        return "the c-offset it was given (eps_exp) limits it: leave eps_exp unset to let it follow the precision"
    return "the table's pole structure disagrees with the block, which no precision fixes"


# ---------------------------------------------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Knob:
    """A family-owned knob the loop may raise when the diagnosis names ``cause``: ``advance(verification, work)``
    raises it and returns the working precision the next build needs (``work`` if unchanged), or None when it
    cannot help any more.  ``needs_target``: raised only for a ``--target-digits`` build (a knob that also raises
    the precision)."""

    cause: str
    advance: Callable[[Verification, int], int | None]
    needs_target: bool = False


@_quiet_conditioning
def run_verified_build(build: Callable[[int], Any], verify: Callable[[Any, int], Verification | None], *,
                       request: PrecisionRequest, what: str, order_reduced: bool,
                       diagnose: Callable[[Any, Verification, int], Verification] | None = None,
                       knobs: tuple = (), report: bool = True, stream=None):
    """Build a table at ``request.precision``, verify it, and -- with a target only -- escalate; report the result.

    ``build(work)`` returns the table (or a tuple the other callbacks understand) built at working precision
    ``work``; ``verify(built, work)`` its :class:`Verification` (None: nothing to check against);
    ``diagnose(built, verification, work)`` names the cause of a shortfall (``cause`` and, for a precision
    shortfall, ``needed_precision``).  Without a target the table is built once at ``request.precision`` (a family
    :class:`Knob` that does not raise the precision, the global tables' automatic pole count, may still grow) and a
    shortfall is reported in one line (:func:`shortfall_message`).  With ``request.target_digits`` a precision
    shortfall raises the working precision -- the first step to the precision the measured cancellation needs, later
    ones from the measured digits-per-working-digit rate -- until the table delivers the target, and the build stops
    with :class:`TargetDigitsError` when the cap is hit, when an attempt closes less than a quarter of the remaining
    gap (precision is not the limit), or when the diagnosis names another cause.  Returns ``(built, verification,
    work)``: the table, what it delivers, and the working precision it was built (and must be written) at.
    """
    stream = sys.stderr if stream is None else stream
    target = request.target_digits
    goal = request.goal
    cap = request.cap
    by_cause = {knob.cause: knob for knob in knobs}
    work = int(request.precision)
    previous = None     # (digits, work) of the previous precision step
    clamped = False     # whether the cap cut that step short
    best = None         # (built, verification, work) with the most digits (a knob-grown table can get worse)
    while True:
        built = build(work)
        v = verify(built, work)
        if v is None or v.digits is None:
            return built, v, work
        if mp.isnan(v.digits):
            raise ValueError(f"{what}: verification returned NaN digits")
        if best is None or v.digits > best[1].digits:
            best = (built, v, work)
        if not request.short(v.digits):
            if report and target is not None and work > request.precision:
                print(f"note: {what} was built at working precision {work} to deliver "
                      f"{request.knobs.set(request.knobs.target, target)} ({v.digits:.1f} digits verified {v.where})", file=stream, flush=True)
            return built, v, work
        if diagnose is not None:
            v = diagnose(built, v, work)
        knob = by_cause.get(v.cause)
        if knob is not None and (target is not None or not knob.needs_target):
            advanced = knob.advance(v, work)
            if advanced is not None:
                if advanced > cap:
                    raise TargetDigitsError(unreachable_message(
                        what, v, request, work,
                        why=f"it needs working precision about {advanced}, above "
                            f"{request.knobs.set(request.knobs.max_precision, cap)}; raise {request.knobs.max_precision}"))
                work, previous = max(work, int(advanced)), None
                continue
        if target is None:
            built, v_best, work_best = best
            if v_best is not v and diagnose is not None:
                v_best = diagnose(built, v_best, work_best)
            if report:
                print(shortfall_message(what, v_best, request, work_best), file=stream, flush=True)
            return built, v_best, work_best
        if v.cause != "precision" and knob is None:
            raise TargetDigitsError(unreachable_message(what, v, request, work,
                                                        why=_cause_hint(v, request, order_reduced=order_reduced)))
        # (a family knob that stopped helping: the precision gets its turn, and the stall test below ends the build
        # when neither helps)
        # (a step the cap cut short was not sized to close the gap: at the cap the cap is the reason)
        if previous is not None and not clamped and v.digits - previous[0] < (goal - previous[0]) / 4:
            raise TargetDigitsError(unreachable_message(
                what, v, request, work,
                why=f"raising the working precision from {previous[1]} to {work} took it from {previous[0]:.1f} to "
                    f"{v.digits:.1f} digits, less than a quarter of the way to {goal}: precision is not what limits it "
                    f"({_cause_hint(replace(v, cause=None), request, order_reduced=order_reduced)})"))
        if work >= cap:
            raise TargetDigitsError(unreachable_message(
                what, v, request, work,
                why=f"it needs working precision about {max(v.needed_precision or 0, work + 1)}, above "
                    f"{request.knobs.set(request.knobs.max_precision, cap)}; raise {request.knobs.max_precision}"))
        pad = max(10, request.digits // 10)
        step = v.needed_precision or 0
        if previous is not None:
            # a working digit buys one delivered digit between the nodes but fewer past them (0.75 at the poles of a
            # spinning c = 3.3 table, 1 at c = 1000): later steps follow the measured rate
            rate = min(mp.mpf(1), max(mp.mpf(1) / 4, (v.digits - previous[0]) / (work - previous[1])))
            step = max(step, work + int(mp.ceil((goal - v.digits) / rate)) + pad)
        elif not step:
            step = work + int(mp.ceil((goal - v.digits) * mp.mpf(4) / 3)) + pad
        previous = (v.digits, work)
        clamped = max(step, work + 1) > cap
        work = min(max(step, work + 1), cap)


def record_verification(table, v: Verification | None, work: int, request: PrecisionRequest) -> None:
    """Record on ``table`` what the writers put in its file: ``working_dps`` (the JSON's ``precision_actual``),
    ``verified_digits``, ``verified_digits_largest_entry``, ``target_digits`` and the covered range."""
    table.working_dps = int(work)
    table.target_digits = request.target_digits
    table.verified_digits = None if v is None else v.digits
    table.verified_digits_largest_entry = None if v is None else v.largest_digits
    if v is not None and v.span is not None:
        table.verified_range = v.span


__all__ = [
    "BOOTSTRAP_KNOBS",
    "CLI_KNOBS",
    "Knob",
    "KnobNames",
    "PrecisionRequest",
    "TargetDigitsError",
    "Verification",
    "agreement_digits",
    "check_points_x",
    "default_max_precision",
    "diagnose_monomial",
    "far_check_points_x",
    "point_agreement",
    "precision_limit_digits",
    "precision_request_of",
    "record_verification",
    "reproduces",
    "run_verified_build",
    "shortfall_message",
    "verification_from_points",
    "verification_margin",
]
