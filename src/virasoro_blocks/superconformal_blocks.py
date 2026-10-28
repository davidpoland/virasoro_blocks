"""Common numerical and interpolated N=1 superconformal block backends.

This module is the adapter layer between the NS/Ramond recurrences and the
repository's existing global-block interpolation contract.  It deliberately
does not implement another recurrence: ``recursion_ns`` and ``recursion_r``
remain the authoritative elliptic engines, while this module supplies the
same coordinate, left/right assembly, interpolation, and singular-parameter
interfaces used by the Virasoro and global families.

The public API has two levels:

``ns_chiral_block`` / ``ramond_chiral_block`` and ``numeric_*_table``
    Direct numerical blocks and Taylor tables.

``*_interpolated_table``
    ``InterpolatedDeltaTable`` objects using the shared Bohr--Sommerfeld node
    and pole-selection machinery.

Both paths accept generic parameters.  ``regularize='auto'`` recognizes the
N=1 singular central charges.  Direct evaluations there run the exact
Laurent-in-t recursion at c itself (:mod:`virasoro_blocks.super_laurent`; a
genuine pole is refused), and interpolated tables sample its merged-pole
form at the same charge. The sampler records the input, evaluated charge,
and selected policy for serialization layers and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Sequence

from contextlib import nullcontext

import mpmath as mp
from .truncation import is_off_diagonal
from .precision import taylor_coefficients
from .numeric_input import _integer_spin, _exchange_weight, literal_number
from .super_kac import _super_cutoff, guard_super_inputs, matches_kac_weight

from .coordinates import COORDINATES
from .derivs import (
    _assemble_block_grid,
    _fold_parity,
    fold_parity_marker,
    folded_entry_power,
    folded_source_order,
)
from .global_blocks import (
    _interpolate_global_table,
    _make_residue_scorer,
)
from .interpolation import pole_selection_score_entries
from .parity import parity_reversal_phase
from ._conventions import scalar_G_spin0_weight, scalar_G_swap_coefficient
from .recursion import _resolve_chiral_norm, q_from_z
from .recursion_ns import (
    F_block_ns,
    NSFamily,
    SECTOR_EVEN,
    SECTOR_ODD,
    beta_from_c,
    enumerate_ns_poles,
    ns_order_for_q_order,
)
from .recursion_r import (
    BRANCH_MM,
    BRANCH_MP,
    BRANCH_PM,
    BRANCH_PP,
    F_block_r,
    RFamily,
    SIGN_MINUS,
    SIGN_PLUS,
)
from .requests import BlockPoleError


FAMILY_NS = "ns"
FAMILY_RAMOND = "ramond"
FAMILY_R = FAMILY_RAMOND
SECTOR_NAMES = {"even": SECTOR_EVEN, "odd": SECTOR_ODD}
BRANCH_NAMES = {
    "++": BRANCH_PP,
    "+-": BRANCH_PM,
    "-+": BRANCH_MP,
    "--": BRANCH_MM,
}

_CHIRAL_COORDINATES = frozenset(
    name for name, spec in COORDINATES.items() if spec.kind == "chiral"
)
_NON_FOLDED_2D = frozenset(
    name for name, spec in COORDINATES.items() if spec.kind == "non_folded_2d"
)
_FOLDED_2D = frozenset(
    name for name, spec in COORDINATES.items() if spec.kind == "folded_2d"
)
_RADIAL_2D = frozenset(
    name for name, spec in COORDINATES.items() if spec.kind == "radial"
)


class SuperconformalPoleError(BlockPoleError):
    """A direct super-block evaluation landed on an exchanged Kac pole.

    A finite-part continuation at a genuine exchanged-module pole is not a
    physical block.  The interpolation path samples away from these points,
    while callers asking for a direct value should get an actionable error
    instead of a huge, precision-dependent number.  The interpolation path
    uses a private off-pole probe and does not expose a finite-part bypass.
    """

    def __init__(
        self, message=None, *, family=None, basis=None, exchanged=None,
        pole=None, side=None, sector=None, branch=None,
    ):
        self.sector = sector
        self.branch = branch
        super().__init__(
            message, family=family, basis=basis, exchanged=exchanged,
            pole=pole, side=side,
        )


def super_offset_eps_exp(digits=None) -> int:
    """The c-offset exponent for results good to ``digits`` digits (default: the working precision).

    At an N=1 minimal-model c the recurrence runs at ``c + 10^-eps_exp``, and that differs from the c -> c0 limit
    by O(10^-eps_exp): 9.4e-29 at c = 7/10 and 5.2e-32 at c = 3/2 for eps 1e-30 (2026-09-27).  A fixed eps_exp = 30
    capped every singular-c super block at ~28-32 digits whatever the precision (and the verification, run at
    the same offset, could not see it).  The offset sits ``max(10, digits // 10)`` digits below the target.
    """
    digits = int(mp.mp.dps if digits is None else digits)
    return digits + max(10, digits // 10)


@dataclass(frozen=True)
class SuperBlockPolicy:
    """Shared singular-parameter policy for NS and Ramond block families.

    ``regularize`` controls the central-charge path. A public value at a
    recognized singular c is computed at c itself by the
    exact Laurent-in-t recursion (:mod:`virasoro_blocks.super_laurent`): the
    c -> c0 limit at fixed weight, also on a Kac weight, and a genuine pole
    there is refused. Singular-c interpolation samples the merged-pole form
    of that recursion. The private offset fallback uses ``c + 10**(-eps_exp)``;
    it is a central-charge offset, not a finite-part prescription for an
    exchanged Kac pole. Output metadata records the selected table policy.
    """

    regularize: str = "auto"
    eps_exp: int | None = None      # None: tied to the working precision (super_offset_eps_exp)

    def __post_init__(self):
        from .requests import _integer_field

        if self.regularize not in {"auto", "on", "off", True, False}:
            raise ValueError("regularize must be 'auto', 'on', or 'off'")
        if self.eps_exp is not None:
            exponent = _integer_field("eps_exp", self.eps_exp)
            if exponent < 8:
                raise ValueError("eps_exp must be >= 8")
            object.__setattr__(self, "eps_exp", exponent)

    def effective_eps_exp(self) -> int:
        return super_offset_eps_exp() if self.eps_exp is None else int(self.eps_exp)

    def resolve_c(self, c):
        return resolve_super_c(
            c, regularize=self.regularize, eps_exp=self.effective_eps_exp()
        )

    def metadata(self, family, *, c_input, c_eval, regularized, sector, branch):
        return {
            "block_family": family,
            "super_sector": "odd" if normalize_sector(sector) else "even",
            "ramond_branch": (
                next(name for name, value in BRANCH_NAMES.items()
                     if value == normalize_branch(branch))
                if family == FAMILY_RAMOND else None
            ),
            "central_charge_input": c_input,
            "central_charge_evaluated": c_eval,
            "regularization_policy": "c+eps" if regularized else "direct",
            "regularization_eps_exp": self.effective_eps_exp(),
            # Retain the established metadata key, but do not make the
            # unsafe alternative part of the public policy object.
            "collision_policy": "raise",
        }


DEFAULT_SUPER_BLOCK_POLICY = SuperBlockPolicy()


def normalize_sector(sector) -> int:
    from .requests import _integer_field

    if isinstance(sector, str):
        key = sector.strip().lower()
        if key not in SECTOR_NAMES:
            raise ValueError("sector must be 'even' or 'odd'")
        return SECTOR_NAMES[key]
    value = _integer_field("sector", sector)
    if value not in (SECTOR_EVEN, SECTOR_ODD):
        raise ValueError("sector must be 0/even or 1/odd")
    return value


def normalize_branch(branch) -> int:
    from .requests import _integer_field

    if isinstance(branch, str):
        if branch not in BRANCH_NAMES:
            raise ValueError("Ramond branch must be one of ++, +-, -+, --")
        return BRANCH_NAMES[branch]
    value = _integer_field("branch", branch)
    if value not in range(4):
        raise ValueError("Ramond branch must be 0 (++), 1 (+-), 2 (-+), or 3 (--)")
    return value


def resolve_super_policy(
    policy: SuperBlockPolicy | None = None,
    *,
    regularize="auto",
    eps_exp: int | None = None,
) -> SuperBlockPolicy:
    """Normalize the public policy arguments used by every super-block path.

    The keyword arguments remain for backwards compatibility with the first
    adapter API.  New code can pass one :class:`SuperBlockPolicy` object so the
    c-offset and exchanged-pole choices cannot drift between chiral and full
    calls.
    """
    if policy is not None:
        if not isinstance(policy, SuperBlockPolicy):
            raise TypeError("policy must be a SuperBlockPolicy")
        nondefault = regularize != "auto" or eps_exp is not None
        if nondefault:
            raise ValueError("pass either policy or regularize/eps_exp, not both")
        # the offset is fixed here, at the precision the call runs at
        return policy if policy.eps_exp is not None else replace(policy, eps_exp=super_offset_eps_exp())
    return SuperBlockPolicy(regularize=regularize, eps_exp=super_offset_eps_exp() if eps_exp is None else eps_exp)


def _offset_guard_dps(regularized, q_order, eps_exp=None) -> int:
    """Extra working digits for an evaluation at ``c + 10^-eps_exp``.

    The coincident Kac poles of an N=1 minimal-model c cancel as ``1/eps^k`` with a collision order ``k`` that
    grows with the level: the loss measured at c = 3/2 is 74, 132, 164, 203 digits at q_order 8, 16, 24, 32
    (about 2.5-7 eps_exp), 26 digits at c = 7/10.  Without the guard a dps-60 evaluation returned garbage.
    """
    if not regularized:
        return 0
    eps_exp = super_offset_eps_exp() if eps_exp is None else int(eps_exp)
    return eps_exp * (2 + max(int(q_order), 1) // 4) + 20


_MEASURED_GUARDS: dict = {}
# generic chiral weights and cross-ratio for the guard probes (away from every Kac pole the pools reach)
_GUARD_PROBES = ("0.4137", "2.7183", "7.3891")
_GUARD_PROBE_Z = "0.4"


def _measured_offset_guard(family, c_eval, h_ext_sets, *, q_order, sector, branch, eps_exp, regularized=True) -> int:
    """Guard digits an evaluation at the offset ``c + 10^-eps_exp`` needs, measured (and cached).

    The colliding Kac poles of an N=1 minimal-model c cancel as ``1/eps^k``; how many digits that costs depends on
    the family and the collision structure, not on ``q_order`` alone.  The fixed formula
    (:func:`_offset_guard_dps`, ``eps_exp (2 + q/4) + 20``) needed 1.0 eps_exp at c = 3/2 (NS) and 1.7-2.4 at 7/10
    against its 4-6, yet fell short for Ramond at c = 3/2, q_order 16 (2026-09-28).  Here the chiral block is
    evaluated at three generic probes with the guard raised by ``eps_exp/2`` from ``eps_exp + 10`` (the least that
    represents the offset) until two successive guards agree to the working precision; ten digits are added.
    ``h_ext_sets`` lists the external sets the caller evaluates with (both sides of a full block).

    Unregularized (a c the recursion takes as it is) it is 0: near (not on) a singular c or at large c the bare
    NS/Ramond recursion cancels too (1.5 + 1e-20 at 30 digits: no correct digit), but its coefficients and their sum
    are certified from their own balls (:func:`poles.certified_level_series`,
    :func:`recursion_ns.certified_half_value`).
    """
    if not regularized:
        # a c the recursion takes as it is: its coefficients and their half-power sum are certified from their own
        # balls (poles.certified_level_series, recursion_ns.certified_half_value), so no guard is measured
        return 0
    eps_exp = int(eps_exp)
    key = (family, _series_key(c_eval), tuple(tuple(_series_key(h) for h in ext) for ext in h_ext_sets),
           int(q_order), int(sector), branch, eps_exp, bool(regularized), mp.mp.prec)
    cached = _MEASURED_GUARDS.get(key)
    if cached is not None:
        return cached
    step = max(20, eps_exp // 2)
    cap = 4 * _offset_guard_dps(True, q_order, eps_exp)
    tol = mp.mpf(10) ** (-(mp.mp.dps - 2))

    def values(guard):
        out = []
        for ext in h_ext_sets:
            for probe in _GUARD_PROBES:
                out.append(_chiral_evaluator(
                    family, mp.mpf(probe), mp.mpf(_GUARD_PROBE_Z), list(ext), c=c_eval, q_order=q_order,
                    chiral_norm=None, z_shift=None, sector=sector, branch=branch, guard_dps=guard,
                ))
        return out

    guard = eps_exp + 10
    previous = values(guard)
    while guard < cap:
        current = values(guard + step)
        if all(abs(a - b) <= tol * abs(b) for a, b in zip(previous, current)):
            break
        guard, previous = guard + step, current
    result = guard + 10
    if len(_MEASURED_GUARDS) >= 256:
        _MEASURED_GUARDS.pop(next(iter(_MEASURED_GUARDS)))
    _MEASURED_GUARDS[key] = result
    return result


def _guarded(guard_dps):
    return mp.workdps(mp.mp.dps + guard_dps) if guard_dps else nullcontext()


def _check_chiral_collision(family, h, h_ext, *, c, q_order, sector, branch, offset_eps_exp=None,
                            c_input=None, report=False):
    """Raise :class:`SuperconformalPoleError` when ``h`` is on an exchanged Kac pole (``report``: return whether it
    is, for a caller that takes the c -> c0 limit there)."""
    family = _resolve_family(family)
    h = literal_number(h)
    c = c if c_input is None else c_input
    order = max(1, 2 * int(q_order) + 4)
    # under the offset guard: below eps_exp digits c + eps rounds back to the singular c
    with _guarded(_offset_guard_dps(offset_eps_exp is not None, order, offset_eps_exp)):
        poles = _chiral_poles(
            family, c, h_ext, q_order=order,
            sector=sector, branch=branch, with_labels=True,
        )
    for label, pole in poles:
        if matches_kac_weight(h, c, *label[:2], pole):
            if report:
                return True
            raise SuperconformalPoleError(
                f"{family} exchanged chiral weight h={mp.nstr(h, 12)} "
                f"is on the Kac pole h={mp.nstr(pole, 12)}; "
                "a finite-part continuation is diagnostic only. Use "
                "the private interpolation probe path or choose a physical "
                "irreducible sector.",
                family=family, exchanged=h, pole=pole,
                sector=sector, branch=branch,
            )
    if not report:
        from .conditioning import pole_loss, report as report_loss

        report_loss(pole_loss(h, [pole for _label, pole in poles]), what=f"{family} block", source=h)
    return False


def _exact_singular(policy, c):
    """``(c, True)`` when a value is computed at the N=1 singular ``c`` itself -- the exact Laurent-in-t recursion
    (:mod:`virasoro_blocks.super_laurent`; the c -> c0 limit at fixed weight, the physical block also on a Kac weight,
    and a genuine pole there refused) -- else ``(the c to evaluate at, False)``."""
    from .super_laurent import exact_singular_c

    c_input, c_eval, regularized = policy.resolve_c(c)
    if regularized and exact_singular_c(c_input) is not None:
        return c_input, True
    return c_eval, False


def _exact_value(family, exchanged, compute):
    """``compute()`` at the singular c, its genuine-pole error as :class:`SuperconformalPoleError`."""
    from .super_laurent import SuperLaurentPoleError

    try:
        return compute()
    except SuperLaurentPoleError as exc:
        raise SuperconformalPoleError(str(exc), family=family, exchanged=exchanged, pole=exchanged) from exc


def is_superconformal_singular_c(c) -> bool:
    """Whether ``c`` is a recognized N=1 recursion singularity.

    Real literals use exact rational membership. Native mpmath values are
    checked at the precision they carry, including the final charge check
    in :func:`virasoro_blocks.super_laurent.exact_singular_c`. A nearby charge
    that passes candidate screening alone must retain its supplied offset.
    """
    from .numeric_input import parse_number
    from .super_laurent import exact_singular_c

    return exact_singular_c(parse_number(c)) is not None


def _candidate_superconformal_singular_c(c) -> bool:
    """Cheap candidate screening for the Laurent engine's final locus check.

    This half-precision proximity test does not establish that the supplied
    central charge is singular and must not select a public offset policy.
    """
    from .numeric_input import exact_real, exact_b_squared, parse_number, _explicit_nonreal_input

    c = parse_number(c)
    if _explicit_nonreal_input(c):
        return False
    if exact_real(c) is not None:
        return exact_b_squared(c, superconformal=True) is not None
    value = mp.convert(c)
    tol = mp.mpf(10) ** (-max(8, mp.mp.dps // 2)) * max(abs(value), 1)
    if isinstance(value, mp.mpc):
        # The final locus check rejects any nonzero imaginary component.
        if abs(value.imag) > tol:
            return False
        value = value.real
    if abs(value - mp.mpf(3) / 2) <= tol:
        return True
    for m in range(3, 256):
        cm = mp.mpf(3) / 2 - mp.mpf(12) / (m * (m + 2))
        if abs(value - cm) <= tol:
            return True
    # Non-unitary minimal models and the Virasoro-singular rational points are
    # equally singular for the N=1 recurrence.  In the beta parametrization
    # this is precisely rational beta^2.  ``limit_denominator`` is used only as
    # an exact-locus detector: a generic value must agree to many working
    # digits before it is classified as rational.
    try:
        beta_sq = beta_from_c(value) ** 2
        if isinstance(beta_sq, mp.mpc) and abs(beta_sq.imag) > tol:
            return False
        beta_sq = mp.re(beta_sq)
        rational = Fraction(str(beta_sq)).limit_denominator(512)
        if abs(beta_sq - mp.mpf(rational.numerator) / rational.denominator) <= tol:
            return True
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return False


def resolve_super_c(
    c, *, regularize="auto", eps_exp: int | None = None,
    policy: SuperBlockPolicy | None = None,
):
    """Return ``(c_input, c_eval, regularized)`` for a super block call.

    ``policy`` is a convenience for callers that do not need the exchanged
    pole checks but still want one c-offset convention shared with a table
    builder.
    """
    if policy is not None:
        policy = resolve_super_policy(policy)
        regularize = policy.regularize
        eps_exp = policy.eps_exp
    eps_exp = super_offset_eps_exp() if eps_exp is None else int(eps_exp)
    from .singularity import snap_float_c

    # complex c is allowed (mpc with zero imaginary part is demoted to mpf); a float 0.7 means 7/10
    c_input = snap_float_c(c, family="ns")
    if isinstance(c_input, mp.mpc) and c_input.imag == 0:
        c_input = c_input.real
    if regularize in (True, "on"):
        use_offset = True
    elif regularize in (False, "off"):
        use_offset = False
    elif regularize == "auto":
        use_offset = is_superconformal_singular_c(c_input)
    else:
        raise ValueError("regularize must be auto, on, or off")
    if use_offset:
        # the offset must survive the working precision (at dps <= eps_exp it rounded away entirely): convert
        # and offset c at dps + eps_exp; the evaluation itself runs under _offset_guard_dps
        with mp.workdps(mp.mp.dps + int(eps_exp) + 10):
            c_eval = mp.convert(c_input) + mp.mpf(10) ** (-int(eps_exp))
    else:
        c_eval = c_input
    return c_input, c_eval, use_offset


def _family_block(family: str):
    family = family.lower()
    if family == FAMILY_NS:
        return F_block_ns
    if family in {FAMILY_RAMOND, "r"}:
        return F_block_r
    raise ValueError("family must be 'ns' or 'ramond'")


@guard_super_inputs("ns")
def ns_chiral_block(delta, z, h_ext: Sequence, *, c, q_order: int,
                    chiral_norm=None, z_shift=None, sector=SECTOR_EVEN,
                    regularize="auto", eps_exp: int | None = None,
                    policy: SuperBlockPolicy | None = None):
    """Direct numerical NS chiral block with a named ``q_order`` unit.

    The public evaluator is fail-closed at an exchanged Kac pole.  Use a
    :class:`SuperBlockPolicy` when the same central-charge regularization
    decision must be shared by several chiral or full evaluations.
    """
    if isinstance(delta, mp.mpc) and delta.imag == 0:
        delta = mp.mpf(delta.real)
    policy = resolve_super_policy(policy, regularize=regularize, eps_exp=eps_exp)
    c_value, exact = _exact_singular(policy, c)
    if exact:
        return _exact_value(FAMILY_NS, delta, lambda: F_block_ns(
            delta, z, h_ext, c=c_value, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
            sector=normalize_sector(sector)))
    c_input, c_eval, regularized = policy.resolve_c(c)
    _check_chiral_collision(
        FAMILY_NS, delta, h_ext, c=c_eval, q_order=q_order,
        sector=normalize_sector(sector), branch=BRANCH_PP,
        offset_eps_exp=policy.eps_exp if regularized else None,
        c_input=c_input if regularized else None,
    )
    with _guarded(_measured_offset_guard(FAMILY_NS, c_eval, [h_ext], q_order=q_order, sector=normalize_sector(sector),
                                         branch=BRANCH_PP, eps_exp=policy.eps_exp, regularized=regularized)):
        return F_block_ns(
            delta, z, h_ext, c=c_eval, q_order=q_order,
            chiral_norm=chiral_norm, z_shift=z_shift,
            sector=normalize_sector(sector),
        )


@guard_super_inputs("ramond")
def ramond_chiral_block(delta, z, h_ext: Sequence, *, c, q_order: int,
                        chiral_norm=None, z_shift=None, sector=SECTOR_EVEN,
                        branch="++", regularize="auto", eps_exp: int | None = None,
                        policy: SuperBlockPolicy | None = None):
    """Direct numerical Ramond-external chiral block.

    ``branch`` is part of the block identity, not just output metadata: all
    four HJS sign branches are available and are checked against their own
    exchanged pole pool.
    """
    if isinstance(delta, mp.mpc) and delta.imag == 0:
        delta = mp.mpf(delta.real)
    policy = resolve_super_policy(policy, regularize=regularize, eps_exp=eps_exp)
    sector = normalize_sector(sector)
    branch = normalize_branch(branch)
    c_value, exact = _exact_singular(policy, c)
    if exact:
        return _exact_value(FAMILY_RAMOND, delta, lambda: F_block_r(
            delta, z, h_ext, c=c_value, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
            sector=sector, branch=branch))
    c_input, c_eval, regularized = policy.resolve_c(c)
    _check_chiral_collision(
        FAMILY_RAMOND, delta, h_ext, c=c_eval, q_order=q_order,
        sector=sector, branch=branch,
        offset_eps_exp=policy.eps_exp if regularized else None,
        c_input=c_input if regularized else None,
    )
    with _guarded(_measured_offset_guard(FAMILY_RAMOND, c_eval, [h_ext], q_order=q_order, sector=sector,
                                         branch=branch, eps_exp=policy.eps_exp, regularized=regularized)):
        return F_block_r(
            delta, z, h_ext, c=c_eval, q_order=q_order,
            chiral_norm=chiral_norm, z_shift=z_shift,
            sector=sector, branch=branch,
        )


def _chiral_evaluator(family, delta, z, h_ext, *, c, q_order, chiral_norm,
                      z_shift, sector, branch, guard_dps=0):
    """Evaluate one chiral block for an internal off-pole sample.

    Callers that expose a physical value perform the exchanged-pole check
    before reaching this helper.  Interpolation samplers call it directly at
    their deliberately off-pole nodes, so they never need a public pole
    bypass.
    """
    with _guarded(guard_dps):
        if family == FAMILY_NS:
            return F_block_ns(
                delta, z, h_ext, c=c, q_order=q_order,
                chiral_norm=chiral_norm, z_shift=z_shift, sector=sector,
            )
        return F_block_r(
            delta, z, h_ext, c=c, q_order=q_order,
            chiral_norm=chiral_norm, z_shift=z_shift, sector=sector,
            branch=branch,
        )


def F_block(family, delta, z, h_ext, *, c, q_order, chiral_norm, z_shift, sector, branch):
    """The bare NS / Ramond chiral block (at a singular c: the exact Laurent recursion's)."""
    if family == FAMILY_NS:
        return F_block_ns(delta, z, h_ext, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
                          sector=sector)
    return F_block_r(delta, z, h_ext, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift, sector=sector,
                     branch=branch)


def _center_for_coordinate(coordinate: str, center):
    spec = COORDINATES[coordinate]
    return spec.u_symmetric() if center is None else spec.u_of_z(center)


_SUPER_SERIES_CACHE: dict = {}
_SUPER_SERIES_CACHE_MAXSIZE = 64


def _series_key(value):
    if value is None:
        return None
    from .numeric_input import parse_number

    value = parse_number(value)
    if isinstance(value, Fraction):
        return ("rational", value.numerator, value.denominator)
    if isinstance(value, mp.mpc):
        return value._mpc_
    if isinstance(value, mp.mpf):
        from mpmath.libmp import fzero

        return value._mpf_, fzero
    value = mp.mpc(value)
    return (value.real._mpf_, value.imag._mpf_)


def _super_series_geometry(coordinate, h_ext, *, c, q_order, n_terms, chiral_norm, z_shift, center):
    """The Delta-independent flint series of the NS/Ramond chiral block about the coordinate's center, or None.

    ``F(Delta, z(u)) = (16q/N)^Delta * BASE(z(u)) * sum_k H_k(Delta) q^(k/2)``, where BASE is the Virasoro prefactor
    base with ``c - 1`` replaced by ``c - 3/2`` (Ramond shares it). BASE, ``log(16q/N)`` and the powers of ``q^(1/2)`` depend only on
    the geometry, so they are built once (as for the Virasoro table sampler, ``derivs._chiral_series_evaluator``);
    each Delta then requires only the H recursion and a few series products.
    """
    key = (mp.mp.prec, coordinate, int(n_terms), int(q_order), _series_key(c),
           tuple(_series_key(h) for h in h_ext), _series_key(chiral_norm), _series_key(z_shift), _series_key(center))
    if key in _SUPER_SERIES_CACHE:
        return _SUPER_SERIES_CACHE[key]
    from flint import acb, acb_mat, acb_series, ctx as flint_ctx

    from .derivs import _flint_geometry_series_coeffs, _flint_prec_bits
    from .polynomial import _mpf_to_acb
    from .recursion_ns import _resolve_ns_order

    geometry = None
    spec = COORDINATES[coordinate]
    u0 = _center_for_coordinate(coordinate, center)
    d1, d2, d3, d4 = (mp.convert(h) for h in h_ext)
    cs = mp.convert(c) - mp.mpf(3) / 2
    sigma = cs / 24
    exp_z = sigma + (mp.mpf(0) if z_shift is None else mp.convert(z_shift))
    coeffs = _flint_geometry_series_coeffs(
        n_terms, sigma, exp_z, sigma - d2 - d3, cs / 2 - 4 * (d1 + d2 + d3 + d4),
        z_of_u=spec.z_of_u, u_symmetric=u0,
    )
    if coeffs is not None:
        q_coeffs, base_coeffs = coeffs
        old_cap = flint_ctx.cap
        flint_ctx.prec, flint_ctx.cap = _flint_prec_bits(), n_terms
        try:
            q = acb_series([_mpf_to_acb(v) for v in q_coeffs])
            half = q.sqrt()                       # the principal branch, as _horner_half_series' mp.sqrt
            order = _resolve_ns_order(None, q_order)
            power, rows = acb_series([acb(1)]), []
            for _k in range(order + 1):
                row = list(power.coeffs())[:n_terms]
                rows.append(row + [acb(0)] * (n_terms - len(row)))
                power = power * half
            log_q16N = (acb(16) * q).log() - acb_series([_mpf_to_acb(mp.log(_resolve_chiral_norm(chiral_norm)))])
            geometry = (acb_mat(rows), acb_series([_mpf_to_acb(v) for v in base_coeffs]), log_q16N, order)
        finally:
            flint_ctx.cap = old_cap
    if len(_SUPER_SERIES_CACHE) >= _SUPER_SERIES_CACHE_MAXSIZE:
        _SUPER_SERIES_CACHE.pop(next(iter(_SUPER_SERIES_CACHE)))
    _SUPER_SERIES_CACHE[key] = geometry
    return geometry


def _super_series_taylor(geometry, family, delta, h_ext, *, c, q_order, n_terms, sector, branch):
    """Taylor coefficients of one chiral NS/Ramond block from its cached geometry series."""
    from .derivs import _acb_to_scalar

    coeffs = _super_series_taylor_acb(geometry, family, delta, h_ext, c=c, q_order=q_order, n_terms=n_terms,
                                      sector=sector, branch=branch)
    return [_acb_to_scalar(coeffs[i]) if i < len(coeffs) else mp.mpf(0) for i in range(n_terms)]


def _super_series_taylor_acb(geometry, family, delta, h_ext, *, c, q_order, n_terms, sector, branch):
    """:func:`_super_series_taylor` as the flint coefficients (at most ``n_terms``)."""
    from flint import acb_mat, acb_series, ctx as flint_ctx

    from .derivs import _flint_prec_bits
    from .polynomial import _mpf_to_acb
    from .recursion_ns import H_q_series_ns
    from .recursion_r import H_q_series_r
    from .singularity import snap_float_c

    Q, base, log_q16N, order = geometry
    from .numeric_input import parse_number

    delta = parse_number(delta)
    c = snap_float_c(c, family="ns")
    h_ext = [parse_number(value) for value in h_ext]
    if family == FAMILY_NS:
        H = H_q_series_ns(delta, h_ext, c=c, q_order=q_order, sector=sector, as_acb=True)
    else:
        H = H_q_series_r(delta, h_ext, c=c, q_order=q_order, sector=sector, branch=branch, as_acb=True)
    old_cap = flint_ctx.cap
    flint_ctx.prec, flint_ctx.cap = _flint_prec_bits(), n_terms
    try:
        row = acb_mat([list(H)]) * Q
        H_series = acb_series([row[0, i] for i in range(n_terms)])
        F = (_mpf_to_acb(delta) * log_q16N).exp() * base * H_series
        return list(F.coeffs())
    finally:
        flint_ctx.cap = old_cap


def _super_H_laurent(family, loc, h_ext, *, c, q_order, sector, branch, principal_order, regular_order,
                     allow_regular_point=False):
    """The Laurent data in the exchanged weight of the NS / Ramond ``H`` coefficients at ``loc`` (the
    :func:`H_q_series_ns` / :func:`H_q_series_r` series: at an N=1 singular c the merged form of the exact Laurent
    recursion), certified to the working precision, as :func:`poles.laurent_rows_acb`; None off its poles unless
    ``allow_regular_point`` (then the Taylor rows there)."""
    from . import poles as _poles
    from .numeric_input import parse_number
    from .recursion_ns import ns_level_table, ns_order_for_q_order
    from .recursion_r import _state, r_level_table
    from .singularity import snap_float_c
    from .super_laurent import exact_singular_c, merged_laurent_rows, merged_state

    c = snap_float_c(c, family="ns")
    h_ext = [parse_number(value) for value in h_ext]
    order = ns_order_for_q_order(q_order)
    if family == FAMILY_NS:
        state, field, table_at = sector, mp.mpf, (lambda: ns_level_table(c, h_ext, order))
    else:
        state, field, table_at = _state(sector, SIGN_PLUS, branch), mp.mpc, (lambda: r_level_table(c, h_ext, order))
    what = f"the {family} Laurent data at {mp.nstr(loc, 12)}"
    c0 = exact_singular_c(c)
    if c0 is not None:
        merged = dict(sector=state // 4, branch=state % 4) if family == FAMILY_RAMOND else dict(sector=state)
        tol = _poles.location_tolerance(loc)
        return _poles._certified_ladder(
            lambda: merged_laurent_rows(merged_state(family, h_ext, c0=c0, order=order, **merged), loc,
                                        principal_order=principal_order, regular_order=regular_order, tol=tol,
                                        allow_regular_point=allow_regular_point),
            _poles._laurent_deficit, order, what)
    family_rules = table_at().family
    return _poles.certified_level_laurent(
        table_at, loc, order, family_rules.seed(state, order, field), couples=family_rules.couples, sector=state,
        principal_order=principal_order, regular_order=regular_order, allow_regular_point=allow_regular_point,
        what=what)


class _SuperPoleEvaluator:
    """One chiral NS / Ramond half as the shared analytic principal-part machinery reads it
    (:func:`polyderivs._analytic_chiral_factor`): Taylor rows at a regular weight (``evaluate(h, return_acb=True)``)
    and the Laurent rows at a pole (``pole_laurent``), from the coordinate's geometry series and the recursion's own
    pole data (:func:`_super_H_laurent`).  ``kac_locations`` lists the real chiral arguments it is asked about (it has
    no batched ``many`` for the factor to prefetch with).

    Built for one principal-part extraction (the samplers' ``_analytic_principal_parts``, once per table, in the
    parent process) and never stored on a sampler: flint values do not pickle.
    """

    def __init__(self, family, coordinate, h_ext, *, c, q_order, n_terms, chiral_norm, z_shift, sector, branch,
                 center, locations):
        self.family = family
        self.h_ext = list(h_ext)
        self.c = c
        self.q_order = q_order
        self.n_terms = int(n_terms)
        self.sector = sector
        self.branch = branch
        self.locations = sorted(mp.mpf(mp.re(p)) for p in locations if mp.im(mp.mpc(p)) == 0)
        self.geometry = _super_series_geometry(
            coordinate, h_ext, c=c, q_order=q_order, n_terms=self.n_terms, chiral_norm=chiral_norm,
            z_shift=z_shift, center=center,
        )
        self.real_analytic = all(mp.im(mp.mpc(v)) == 0 for v in [c, *h_ext, z_shift or 0, center or 0,
                                                                     chiral_norm or 16])

    def kac_locations(self):
        return None if self.geometry is None else list(self.locations)

    def __call__(self, h, return_acb=True):
        from flint import acb

        coeffs = _super_series_taylor_acb(
            self.geometry, self.family, h, self.h_ext, c=self.c, q_order=self.q_order, n_terms=self.n_terms,
            sector=self.sector, branch=self.branch,
        )
        return [coeffs[i] if i < len(coeffs) else acb(0) for i in range(self.n_terms)]

    def pole_laurent(self, x, *, principal_order=1, regular_order=0, allow_regular_point=False):
        """``(principal, regular)`` Taylor rows of the block's Laurent expansion in the weight at the pole ``x``
        (:func:`derivs.compose_prefactor_laurent`), or None when ``x`` is not a pole (unless ``allow_regular_point``:
        then its Taylor expansion there)."""
        from flint import acb_mat, acb_series, ctx as flint_ctx

        from .derivs import compose_prefactor_laurent
        from .polynomial import _flint_prec_bits, _mpf_to_acb

        if self.geometry is None:
            return None
        H = _super_H_laurent(self.family, x, self.h_ext, c=self.c, q_order=self.q_order, sector=self.sector,
                             branch=self.branch, principal_order=principal_order, regular_order=regular_order,
                             allow_regular_point=allow_regular_point)
        if H is None:
            return None
        Q, base, log_q16N, _order = self.geometry
        n = self.n_terms
        old_cap = flint_ctx.cap
        flint_ctx.prec, flint_ctx.cap = _flint_prec_bits(), n
        try:
            def series(row):
                product = acb_mat([list(row)]) * Q
                return acb_series([product[0, i] for i in range(n)])

            H_pr, H_rg = ([series(row) for row in part] for part in H)
            P = (_mpf_to_acb(mp.mpc(x)) * log_q16N).exp() * base
            return compose_prefactor_laurent(P, log_q16N, H_pr, H_rg, n)
        finally:
            flint_ctx.cap = old_cap

    def pole_data(self, x):
        out = self.pole_laurent(x)
        return None if out is None else (out[0][0], out[1][0])


@guard_super_inputs()
def _chiral_taylor(
    family, delta, coordinate, h_ext, *, c, q_order, n_terms,
    chiral_norm, z_shift, sector, branch, center=None, guard_dps=0,
):
    if coordinate not in _CHIRAL_COORDINATES:
        raise ValueError(f"unknown chiral coordinate {coordinate!r}")
    with _guarded(guard_dps):
        geometry = _super_series_geometry(
            coordinate, h_ext, c=c, q_order=q_order, n_terms=n_terms,
            chiral_norm=chiral_norm, z_shift=z_shift, center=center,
        )
        if geometry is not None:
            return _super_series_taylor(
                geometry, family, delta, h_ext, c=c, q_order=q_order,
                n_terms=n_terms, sector=sector, branch=branch,
            )
    # where the flint geometry does not resolve: numerical derivatives of the block
    spec = COORDINATES[coordinate]
    u0 = _center_for_coordinate(coordinate, center)

    def value(u):
        return _chiral_evaluator(
            family, delta, spec.z_of_u(u), h_ext, c=c, q_order=q_order,
            chiral_norm=chiral_norm, z_shift=z_shift, sector=sector,
            branch=branch, guard_dps=guard_dps,
        )

    return taylor_coefficients(value, u0, n_terms - 1)


@guard_super_inputs()
def numeric_chiral_table(
    family: str, h, coordinate: str, h_ext: Sequence, *, c, q_order: int,
    m_max: int, chiral_norm=None, z_shift=None, sector=SECTOR_EVEN,
    branch="++", regularize="auto", eps_exp: int | None = None, center=None,
    policy: SuperBlockPolicy | None = None,
):
    """Return the direct numerical Taylor table ``[[F_m]]`` for a chiral block.

    The result has the same one-axis Taylor-coefficient shape as the Virasoro
    ``F_z/F_y/F_q`` builders.  ``h`` is the exchanged chiral weight.
    """
    family = _resolve_family(family)
    sector = normalize_sector(sector)
    branch = normalize_branch(branch)
    policy = resolve_super_policy(policy, regularize=regularize, eps_exp=eps_exp)
    c_value, exact = _exact_singular(policy, c)
    if exact:
        coeffs = _exact_value(family, h, lambda: _chiral_taylor(
            family, h, coordinate, h_ext, c=c_value, q_order=q_order, n_terms=m_max + 1, chiral_norm=chiral_norm,
            z_shift=z_shift, sector=sector, branch=branch, center=center))
        return [[value] for value in coeffs]
    c_input, c_eval, regularized = policy.resolve_c(c)
    _check_chiral_collision(
        family, h, h_ext, c=c_eval, q_order=q_order, sector=sector,
        branch=branch, offset_eps_exp=policy.eps_exp if regularized else None,
        c_input=c_input if regularized else None,
    )
    coeffs = _chiral_taylor(
        family, h, coordinate, h_ext, c=c_eval, q_order=q_order,
        n_terms=m_max + 1, chiral_norm=chiral_norm, z_shift=z_shift,
        sector=sector, branch=branch, center=center,
        guard_dps=_measured_offset_guard(family, c_eval, [h_ext], q_order=q_order, sector=sector, branch=branch,
                                         eps_exp=policy.eps_exp, regularized=regularized),
    )
    return [[value] for value in coeffs]


def _ramond_state(sector: int, branch: int) -> int:
    return 4 * normalize_sector(sector) + normalize_branch(branch)


_POLE_FAMILIES: dict = {}


def _pole_family(family, c, h_ext):
    """One NSFamily / RFamily per (family, c, externals, precision), so its residue memo is shared by every pool."""
    key = (family, _series_key(c), tuple(_series_key(h) for h in h_ext), mp.mp.prec)
    cached = _POLE_FAMILIES.get(key)
    if cached is None:
        if len(_POLE_FAMILIES) >= 64:
            _POLE_FAMILIES.pop(next(iter(_POLE_FAMILIES)))
        cached = _POLE_FAMILIES[key] = (NSFamily if family == FAMILY_NS else RFamily)(c, h_ext)
    return cached


def _chiral_poles(family, c, h_ext, *, q_order, sector, branch, with_labels=False):
    """Genuine chiral pole locations for one selected block state.

    Only the structural zero Delta_{1,1} = 0 is put at the origin.  At an offset minimal-model c + eps other poles
    sit O(eps) from it (label (2,2) at c = 3/2, (2,4) at c = 7/10): those are genuine poles of the offset block
    with large residues, and moving them to 0 cost K eps / x digits (a table at eps 1e-60 was good to 45).  Node
    placement treats them as at the origin (global_blocks._global_bucket_task); the table keeps them where they are.
    """
    rec_family = _pole_family(family, c, h_ext)
    cache = getattr(rec_family, "_pole_pool_cache", None)
    if cache is None:
        cache = rec_family._pole_pool_cache = {}
    key = (q_order, sector, branch, mp.mp.prec)
    if key in cache:
        rows = cache[key]
        return list(rows) if with_labels else [pole for _label, pole in rows]
    if family == FAMILY_NS:
        def build_state(label):
            # ``eval_level_series`` asks for an inner row whose sector is
            # outer_sector XOR (rs mod 2).  Looking only at rows with the
            # selected outer sector misses the universal h=0 odd-sector pole:
            # its residue is stored on the even row and feeds H^odd through the
            # (1,1) flip.
            return rec_family.build_pole(label, sector ^ (label[2] % 2))
    else:
        state = _ramond_state(sector, branch)
        def build_state(label):
            # Ramond odd poles flip both the parity and sign bits.
            return rec_family.build_pole(
                label, state if label[2] % 2 == 0 else state ^ 7
            )
    roots = []
    for label in enumerate_ns_poles(max(0, int(q_order))):
        row = build_state(label)
        # Only proven fusion zeros make a candidate removable. A nonzero
        # residue remains a pole even when it is much smaller than mp.eps.
        if row["K"] != 0:
            # Delta_11 = 0 is exact, but the alpha/beta parametrization reaches
            # it through a cancellation and can leave a tiny numerical image.
            # The Bohr--Sommerfeld node code distinguishes an origin pole from
            # a near-origin pole, so canonicalize this structural value (by its
            # label: a magnitude test also caught the eps images above).
            d = mp.mpf(0) if tuple(label[:2]) == (1, 1) else row["d"]
            roots.append((label, d))
    cache[key] = tuple(roots)
    return roots if with_labels else [pole for _label, pole in roots]


def _half_exp_base(coordinate, center, chiral_norm):
    spec = COORDINATES[coordinate]
    u0 = _center_for_coordinate(coordinate, center)
    z0 = spec.z_of_u(u0)
    return 16 * q_from_z(z0) / _resolve_chiral_norm(chiral_norm)


def _super_laurent_orders(family, c, ext_sets, *, q_order, sector, branch, exact):
    """``(principal_order, regular_order)`` the chiral factors of an exact principal-part split need: ``(1, 0)`` at a
    generic c (simple Kac poles); at an exact singular c the highest merged pole order ``a`` of the halves and
    ``a - 1`` (a product of two halves of order ``a`` reads their regular rows through ``(h - h0)^(a-1)``)."""
    if not exact:
        return 1, 0
    top = 1
    for ext in ext_sets:
        counts = {}
        for pole in _merged_chiral_pool(family, c, ext, q_order=q_order, sector=sector, branch=branch):
            key = mp.nstr(pole, 30)
            counts[key] = counts.get(key, 0) + 1
        top = max([top, *counts.values()])
    return top, top - 1


def _unit_factor(_loc):
    """The chiral factor of a lone holomorphic half's missing antiholomorphic side: no pole, value 1."""
    from flint import acb

    return [acb(0)], [acb(1)]


class SuperChiralSampler:
    """Picklable numeric sampler used by the common interpolation core."""

    def __init__(self, family, coordinate, h_ext, *, c, q_order, m_max,
                 chiral_norm, z_shift, sector, branch, center,
                 c_input=None, c_eval=None, regularized=False, eps_exp=None, exact=False):
        self.family = family
        self.coordinate = coordinate
        self.h_ext = list(h_ext)
        self.c = c
        self.c_input = c if c_input is None else c_input
        self.c_eval = c if c_eval is None else c_eval
        self.q_order = _super_cutoff("q_order", q_order)
        self.m_max = _super_cutoff("m_max", m_max)
        self.chiral_norm = chiral_norm
        self.z_shift = z_shift
        self.sector = normalize_sector(sector)
        self.branch = normalize_branch(branch)
        self.center = center
        self.regularized = bool(regularized)
        self.exact = bool(exact)
        # at the singular c itself the exact Laurent recursion runs, which verifies its own digits: no offset guard
        self.guard_dps = 0 if exact else _measured_offset_guard(
            self.family, self.c, [self.h_ext], q_order=self.q_order, sector=self.sector, branch=self.branch,
            eps_exp=super_offset_eps_exp() if eps_exp is None else eps_exp, regularized=self.regularized,
        )

    def _analytic_principal_parts(self, roots, entries, exp_base):
        """Exact principal parts of the chiral table at ``roots`` (the shared
        :func:`polyderivs._analytic_principal_parts_axis` with a unit antiholomorphic factor, as the Virasoro chiral
        sampler), from the recursion's pole data.  At an offset c they are those of the block at ``c + eps``, the
        function the sampler evaluates, under the same guard."""
        with _guarded(self.guard_dps):
            return self._principal_parts(roots, entries, exp_base)

    def _principal_parts(self, roots, entries, exp_base):
        from .polyderivs import _analytic_chiral_factor, _analytic_principal_parts_axis

        order, regular = _super_laurent_orders(self.family, self.c, [self.h_ext], q_order=self.q_order,
                                               sector=self.sector, branch=self.branch, exact=self.exact)
        n_terms = self.m_max + 1
        evaluator = _SuperPoleEvaluator(
            self.family, self.coordinate, self.h_ext, c=self.c, q_order=self.q_order, n_terms=n_terms,
            chiral_norm=self.chiral_norm, z_shift=self.z_shift, sector=self.sector, branch=self.branch,
            center=self.center, locations=roots,
        )
        factor = _analytic_chiral_factor(evaluator, n_terms, deriv_factor=1, principal_order=order,
                                         regular_order=regular)
        if factor is None:
            return None
        return _analytic_principal_parts_axis(roots, entries, exp_base, factor_m_plus=factor,
                                              factor_m_minus=_unit_factor, factor_n_plus=_unit_factor,
                                              factor_n_minus=_unit_factor, sigma=None)

    def __call__(self, h):
        coeffs = _chiral_taylor(
            self.family, h, self.coordinate, self.h_ext, c=self.c,
            q_order=self.q_order, n_terms=self.m_max + 1,
            chiral_norm=self.chiral_norm, z_shift=self.z_shift,
            sector=self.sector, branch=self.branch, center=self.center,
            guard_dps=self.guard_dps,
        )
        return [[value] for value in coeffs]

    def sample_bucket(self, values, entries):
        # The generic core accepts this optional hook for batched/arb paths.
        # The mpmath implementation is intentionally simple and remains fully
        # compatible with ProcessPoolExecutor.
        entries = tuple(entries) if entries is not None else None
        tables = [self(value) for value in values]
        if entries is not None:
            tables = [
                [[table[m][n] if (m, n) in entries else 0
                  for n in range(len(table[m]))]
                 for m in range(len(table))]
                for table in tables
            ]
        return tables, False


class SuperBlockSampler2d:
    """Numerical ordered/scalar/parity-projected NS or Ramond 2d sampler."""

    def __init__(self, family, coordinate, ell, h_ext, hbar_ext, *, c,
                 q_order, m_max, n_max, sigma, fold_parity, chiral_norm,
                 z_shift, zbar_shift, sector, branch, center, center_bar,
                 c_input=None, c_eval=None, regularized=False, eps_exp=None, exact=False, scalar=False):
        self.family = family
        # a scalar_G sampler: its public spin-0 block (sigma = 0) keeps the scalar assembly path
        self.scalar = bool(scalar)
        self.coordinate = coordinate
        self.ell = _integer_spin(ell)
        self.h_ext = list(h_ext)
        self.hbar_ext = list(hbar_ext)
        self.c = c
        self.c_input = c if c_input is None else c_input
        self.c_eval = c if c_eval is None else c_eval
        self.q_order = _super_cutoff("q_order", q_order)
        self.m_max = m_max = _super_cutoff("m_max", m_max)
        self.n_max = n_max = _super_cutoff("n_max", n_max)
        self.sigma = sigma
        self.fold_parity = fold_parity
        self.chiral_norm = chiral_norm
        self.z_shift = z_shift
        self.zbar_shift = zbar_shift
        self.sector = normalize_sector(sector)
        self.branch = normalize_branch(branch)
        self.center = center
        # An unset zbar-center follows the z-center (the diagonal), exactly as
        # the Virasoro samplers do; ``None`` alone would mean the point 1/2.
        self.center_bar = center if center_bar is None else center_bar
        self.regularized = bool(regularized)
        self.exact = bool(exact)
        # (sigma = 0 with a fold parity is the public spin-0 scalar block: one ordered product, folded even)
        if coordinate in _FOLDED_2D | _RADIAL_2D and (
                fold_parity is None or sigma not in (-1, 0, 1)):
            raise ValueError(
                "folded/radial super blocks require a scalar or parity-projected "
                "basis with a real (+/-1) total parity phase"
            )
        # at the singular c itself the exact Laurent recursion runs, which verifies its own digits: no offset guard
        self.guard_dps = 0 if exact else _measured_offset_guard(
            self.family, self.c, [self.h_ext, self.hbar_ext], q_order=self.q_order, sector=self.sector,
            branch=self.branch, eps_exp=super_offset_eps_exp() if eps_exp is None else eps_exp,
            regularized=self.regularized,
        )
        if coordinate in _RADIAL_2D:
            self.parent_coordinate = coordinate[:-7]
            self.folded = True
            self.out_n_max = 0
            self.src_m = folded_source_order(m_max, 0, fold_parity)
            self.src_n = self.src_m
        elif coordinate in _FOLDED_2D:
            self.parent_coordinate = COORDINATES[coordinate].base_2d
            self.folded = True
            self.out_n_max = n_max
            self.src_m = folded_source_order(m_max, n_max, fold_parity)
            self.src_n = self.src_m
        elif coordinate in _NON_FOLDED_2D:
            self.parent_coordinate = coordinate
            self.folded = False
            self.out_n_max = n_max
            self.src_m = m_max
            self.src_n = n_max
        else:
            raise ValueError(f"unknown 2d coordinate {coordinate!r}")
        if self.folded and (center is not None or center_bar is not None):
            raise ValueError("folded/radial super blocks require the symmetric center")

    def _analytic_principal_parts(self, roots, entries, exp_base):
        """Exact principal parts of the block at ``roots`` (:func:`polyderivs._analytic_principal_parts_axis`, or
        ``_folded`` on a folded / radial coordinate), from the four chiral streams the block is assembled from (the
        m-axis halves at ``center``, the n-axis halves at ``center_bar``) and the recursion's pole data (at an offset c
        those of the block at ``c + eps``, under the sampler's guard); None for a merged pole of order > 1 on a folded
        coordinate (the folded scorer reads simple halves)."""
        with _guarded(self.guard_dps):
            return self._principal_parts(roots, entries, exp_base)

    def _principal_parts(self, roots, entries, exp_base):
        from .polyderivs import (
            _analytic_chiral_factor, _analytic_principal_parts_axis, _analytic_principal_parts_folded,
        )

        order, regular = _super_laurent_orders(self.family, self.c, [self.h_ext, self.hbar_ext],
                                               q_order=self.q_order, sector=self.sector, branch=self.branch,
                                               exact=self.exact)
        if self.folded and order > 1:
            return None
        ell = self.ell
        n_terms = max(self.src_m, self.src_n) + 1
        coordinate = COORDINATES[self.parent_coordinate].chiral_var
        locations = [(mp.mpc(r) + half * ell) / 2 for r in roots for half in (1, -1)]
        made = {}

        def factor(ext, center, shift):
            key = (tuple(_series_key(e) for e in ext), _series_key(center), _series_key(shift))
            if key not in made:
                evaluator = _SuperPoleEvaluator(
                    self.family, coordinate, ext, c=self.c, q_order=self.q_order, n_terms=n_terms,
                    chiral_norm=self.chiral_norm, z_shift=shift, sector=self.sector, branch=self.branch,
                    center=center, locations=locations,
                )
                made[key] = _analytic_chiral_factor(evaluator, n_terms, deriv_factor=2, principal_order=order,
                                                    regular_order=regular)
            return made[key]

        # B[m][n] = c_h(m-center)[m] c_hbar(n-center)[n] + sigma c_hbar(m-center)[m] c_h(n-center)[n] (__call__)
        f_m_plus = factor(self.h_ext, self.center, self.z_shift)
        f_n_minus = factor(self.hbar_ext, self.center_bar, self.zbar_shift)
        f_m_minus = factor(self.hbar_ext, self.center, self.zbar_shift)
        f_n_plus = factor(self.h_ext, self.center_bar, self.z_shift)
        if None in (f_m_plus, f_n_minus, f_m_minus, f_n_plus):
            return None
        factors = dict(
            factor_m_plus=lambda loc: f_m_plus((loc + ell) / 2),
            factor_m_minus=lambda loc: f_m_minus((loc - ell) / 2),
            factor_n_plus=lambda loc: f_n_plus((loc + ell) / 2),
            factor_n_minus=lambda loc: f_n_minus((loc - ell) / 2),
            sigma=None if self.sigma == 0 else self.sigma,
        )
        if self.folded:
            return _analytic_principal_parts_folded(roots, entries, exp_base, fold_marker=self.fold_parity,
                                                    m_max=self.m_max, n_max=self.out_n_max, **factors)
        return _analytic_principal_parts_axis(roots, entries, exp_base, **factors)

    def _half(self, h, ext, *, center, shift, n_terms):
        coordinate = COORDINATES[self.parent_coordinate].chiral_var
        return _chiral_taylor(
            self.family, h, coordinate, ext, c=self.c, q_order=self.q_order,
            n_terms=n_terms, chiral_norm=self.chiral_norm, z_shift=shift,
            sector=self.sector, branch=self.branch, center=center,
            guard_dps=self.guard_dps,
        )

    def __call__(self, delta):
        delta, h, hbar = _split_super_exchange(delta, self.ell)
        n_top = max(self.src_m, self.src_n) + 1
        # the four halves coincide in pairs at a diagonal geometry (equal centers, externals and shifts), and all
        # four at spin 0: each distinct one is computed once
        memo = {}

        def half(weight, ext, center, shift):
            key = (_series_key(weight), tuple(_series_key(e) for e in ext), _series_key(center), _series_key(shift))
            if key not in memo:
                memo[key] = self._half(weight, ext, center=center, shift=shift, n_terms=n_top)
            return memo[key]

        cm = half(h, self.h_ext, self.center, self.z_shift)
        cn = half(hbar, self.hbar_ext, self.center_bar, self.zbar_shift)
        if self.sigma == 0 and self.scalar:
            # the spin-0 scalar block B(h, h): the same elementwise assembly as (1 + 1) B(h, h), so it is exactly
            # half of it
            grid = _assemble_block_grid(cm, cn, self.src_m, self.src_n, sigma=0, c_swap_m=cm, c_swap_n=cn)
        elif self.sigma == 0:
            grid = _assemble_block_grid(cm, cn, self.src_m, self.src_n, sigma=0)
        else:
            csm = half(hbar, self.hbar_ext, self.center, self.zbar_shift)
            csn = half(h, self.h_ext, self.center_bar, self.z_shift)
            grid = _assemble_block_grid(
                cm, cn, self.src_m, self.src_n, sigma=self.sigma,
                c_swap_m=csm, c_swap_n=csn,
            )
        if self.folded:
            return _fold_parity(
                grid, self.fold_parity, self.m_max, self.out_n_max
            )
        return grid

    def sample_bucket(self, values, entries):
        entries = tuple(entries) if entries is not None else None
        tables = [self(value) for value in values]
        if entries is not None:
            tables = [
                [[table[m][n] if (m, n) in entries else 0
                  for n in range(len(table[m]))]
                 for m in range(len(table))]
                for table in tables
            ]
        return tables, False


def _resolve_family(family):
    key = family.lower()
    if key == "r":
        key = FAMILY_RAMOND
    if key not in {FAMILY_NS, FAMILY_RAMOND}:
        raise ValueError("family must be 'ns' or 'ramond'")
    return key


def _resolve_2d_sigma_basis(basis, ell, *, parity=None, parity_external=None,
                            exchanged_parity="1"):
    ell = _integer_spin(ell)
    if basis in {"scalar_G", "ns_scalar_G", "r_scalar_G", "ramond_scalar_G"}:
        # +1 at every ell != 0 (z <-> zbar even); 0 for the public spin-0 block B(h, h) (see _conventions)
        return (mp.mpf(scalar_G_swap_coefficient(ell)),
                fold_parity_marker(ell=ell), "scalar")
    if basis in {"ordered_B", "ns_ordered_B", "r_ordered_B", "ramond_ordered_B"}:
        return mp.mpf(0), None, "ordered"
    if basis in {"parity_projected_B", "ns_parity_projected_B",
                 "r_parity_projected_B", "ramond_parity_projected_B"}:
        if parity not in {"+", "-"}:
            raise ValueError("parity-projected super blocks require parity +/-")
        phase = parity_reversal_phase(
            parity_external, exchanged_parity=exchanged_parity
        )
        sigma = (mp.mpf(1) if parity == "+" else mp.mpf(-1)) * phase
        # Complex phases remain valid for non-folded coordinates. They have
        # no even/odd marker; the sampler rejects them only if a fold is used.
        fold = fold_parity_marker(parity=parity, phase=phase) if phase.imag == 0 else None
        return sigma, fold, "parity"
    raise ValueError(f"unknown super 2d basis {basis!r}")


def _split_super_exchange(delta, ell):
    """Preserve supplied exchange components before either chiral evaluation."""
    delta = literal_number(delta)
    if isinstance(delta, complex):
        delta = mp.mpc(delta)
    return delta, _exchange_weight(delta, ell), _exchange_weight(delta, -ell)


@guard_super_inputs()
def numeric_full_table(
    family: str, basis: str, delta, ell, coordinate: str, h_ext: Sequence,
    *, hbar_ext=None, c, q_order: int, m_max: int, n_max: int = None,
    chiral_norm=None, z_shift=None, zbar_shift=None, sector=SECTOR_EVEN,
    branch="++", parity=None, parity_external=None, exchanged_parity="1",
    regularize="auto", eps_exp: int | None = None, center=None, center_bar=None,
    policy: SuperBlockPolicy | None = None,
):
    """Direct numerical 2d super block Taylor table in any Virasoro coordinate.

    ``delta`` and ``ell`` are the ordinary exchanged 2d labels.  Both chiral
    weights are checked against the family pole pool before the common
    left/right Taylor assembly is evaluated.
    """
    ell = _integer_spin(ell)
    family = _resolve_family(family)
    if n_max is None:
        n_max = m_max
    if hbar_ext is None:
        hbar_ext = h_ext
    if coordinate not in _NON_FOLDED_2D | _FOLDED_2D | _RADIAL_2D:
        raise ValueError(f"unknown full-block coordinate {coordinate!r}")
    sector = normalize_sector(sector)
    branch = normalize_branch(branch)
    policy = resolve_super_policy(policy, regularize=regularize, eps_exp=eps_exp)
    c_input, c_eval, regularized = policy.resolve_c(c)
    c_value, exact = _exact_singular(policy, c)
    delta, h, hbar = _split_super_exchange(delta, ell)
    if not exact:
        for weight, ext in ((h, h_ext), (hbar, hbar_ext)):
            _check_chiral_collision(family, weight, ext, c=c_eval, q_order=q_order, sector=sector, branch=branch,
                                    offset_eps_exp=policy.eps_exp if regularized else None,
                                    c_input=c_input if regularized else None)
    sigma, fold, kind = _resolve_2d_sigma_basis(
        basis, ell, parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity,
    )
    if coordinate in _NON_FOLDED_2D and fold is None:
        fold = 0
    sampler = SuperBlockSampler2d(
        family, coordinate, ell, h_ext, hbar_ext, c=c_value,
        q_order=q_order, m_max=m_max, n_max=n_max, sigma=sigma,
        fold_parity=fold, chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=z_shift if zbar_shift is None else zbar_shift,
        sector=sector, branch=branch, center=center, center_bar=center_bar,
        c_input=c_input, c_eval=c_value, regularized=regularized and not exact,
        eps_exp=policy.eps_exp, exact=exact, scalar=kind == "scalar",
    )
    return _exact_value(family, h, lambda: sampler(delta)) if exact else sampler(delta)


@guard_super_inputs()
def super_ordered_B_block(
    family: str, h, hbar, z, zbar, h_ext: Sequence, *,
    hbar_ext=None, c, q_order: int, chiral_norm=None, z_shift=None,
    zbar_shift=None, sector=SECTOR_EVEN, branch="++", regularize="auto",
    eps_exp: int | None = None, policy: SuperBlockPolicy | None = None,
):
    """Direct ordered full block ``B(h,hbar;z,zbar)`` for NS or Ramond.

    This is the numeric counterpart of Virasoro ``composer.B_block``.  It is
    intentionally primitive: parity projection and scalar ``G`` are built from
    it below, so the CLI, Taylor-table builder, and direct API share exactly
    one left/right convention.
    """
    family = _resolve_family(family)
    if hbar_ext is None:
        hbar_ext = h_ext
    sector = normalize_sector(sector)
    branch = normalize_branch(branch)
    policy = resolve_super_policy(policy, regularize=regularize, eps_exp=eps_exp)
    c_value, exact = _exact_singular(policy, c)
    if exact:
        def both():
            left = F_block(family, h, z, h_ext, c=c_value, q_order=q_order, chiral_norm=chiral_norm,
                           z_shift=z_shift, sector=sector, branch=branch)
            right = F_block(family, hbar, zbar, hbar_ext, c=c_value, q_order=q_order, chiral_norm=chiral_norm,
                            z_shift=z_shift if zbar_shift is None else zbar_shift, sector=sector, branch=branch)
            return left * right

        return _exact_value(family, h, both)
    c_input, c_eval, regularized = policy.resolve_c(c)
    offset = policy.eps_exp if regularized else None
    for weight, ext in ((h, h_ext), (hbar, hbar_ext)):
        _check_chiral_collision(family, weight, ext, c=c_eval, q_order=q_order, sector=sector, branch=branch,
                                offset_eps_exp=offset, c_input=c_input if regularized else None)
    guard = _measured_offset_guard(family, c_eval, [h_ext, hbar_ext], q_order=q_order, sector=sector, branch=branch,
                                   eps_exp=policy.eps_exp, regularized=regularized)
    left = _chiral_evaluator(
        family, h, z, h_ext, c=c_eval, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, sector=sector, branch=branch,
        guard_dps=guard,
    )
    right = _chiral_evaluator(
        family, hbar, zbar, hbar_ext, c=c_eval, q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift if zbar_shift is None else zbar_shift,
        sector=sector, branch=branch, guard_dps=guard,
    )
    return left * right


def super_parity_reversed_B_block(
    family: str, h, hbar, z, zbar, h_ext: Sequence, *,
    hbar_ext=None, parity_external=None, exchanged_parity="1", c,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
    sector=SECTOR_EVEN, branch="++", regularize="auto", eps_exp: int | None = None,
    policy: SuperBlockPolicy | None = None,
):
    """Parity-reversed ordered super-block, with the Virasoro phase convention."""
    if hbar_ext is None:
        hbar_ext = h_ext
    phase = parity_reversal_phase(
        parity_external, exchanged_parity=exchanged_parity
    )
    return phase * super_ordered_B_block(
        family, hbar, h, z, zbar, hbar_ext, hbar_ext=h_ext, c=c,
        q_order=q_order, chiral_norm=chiral_norm,
        z_shift=zbar_shift if zbar_shift is not None else z_shift,
        zbar_shift=z_shift, sector=sector, branch=branch,
        regularize=regularize, eps_exp=eps_exp,
        policy=policy,
    )


def super_parity_projected_B_block(
    family: str, h, hbar, z, zbar, h_ext: Sequence, *,
    hbar_ext=None, parity: str, parity_external=None, exchanged_parity="1",
    c, q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
    sector=SECTOR_EVEN, branch="++", regularize="auto", eps_exp: int | None = None,
    policy: SuperBlockPolicy | None = None,
):
    """Parity ``+``/``-`` combination of ordered super-blocks."""
    if parity not in {"+", "-"}:
        raise ValueError("parity must be '+' or '-'")
    first = super_ordered_B_block(
        family, h, hbar, z, zbar, h_ext, hbar_ext=hbar_ext, c=c,
        q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=zbar_shift, sector=sector, branch=branch,
        regularize=regularize, eps_exp=eps_exp,
        policy=policy,
    )
    second = super_parity_reversed_B_block(
        family, h, hbar, z, zbar, h_ext, hbar_ext=hbar_ext,
        parity_external=parity_external, exchanged_parity=exchanged_parity,
        c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=zbar_shift, sector=sector, branch=branch,
        regularize=regularize, eps_exp=eps_exp,
        policy=policy,
    )
    sign = mp.mpf(1) if parity == "+" else mp.mpf(-1)
    from .conditioning import conditioned_sum
    from .parity import parity_projection_is_zero

    phase = parity_reversal_phase(parity_external, exchanged_parity=exchanged_parity)
    return conditioned_sum(first, sign * second, exact_zero=parity_projection_is_zero(
        sign * phase, h, hbar, z, zbar, h_ext, h_ext if hbar_ext is None else hbar_ext,
        z_shift, z_shift if zbar_shift is None else zbar_shift))


@guard_super_inputs()
def super_scalar_G_block(
    family: str, delta, ell: int, z, zbar, h_ext: Sequence, *, c,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
    sector=SECTOR_EVEN, branch="++", regularize="auto", eps_exp: int | None = None,
    policy: SuperBlockPolicy | None = None,
):
    """Scalar-external parity-symmetrized super-block ``G_{Delta,ell}``.

    ``[B(h, hbar) + B(hbar, h)] / (1 + delta_{ell,0})``, z <-> zbar even at every spin: the spin-0 block is
    ``B(h, h)``.  The z <-> zbar-odd
    ``B(h, hbar) - B(hbar, h)`` is :func:`super_parity_projected_B_block` with ``parity="-"``.
    """
    ell = _integer_spin(ell)
    delta, h, hbar = _split_super_exchange(delta, ell)
    value = super_parity_projected_B_block(
        family, h, hbar, z, zbar, h_ext, hbar_ext=h_ext,
        parity="+", c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        sector=sector, branch=branch, regularize=regularize,
        eps_exp=eps_exp, policy=policy,
    )
    # B + P[B] = 2 B(h, h) at ell = 0
    return value / 2 if ell == 0 and scalar_G_spin0_weight() == 1 else value


def ns_B_block(*args, **kwargs):
    return super_ordered_B_block(FAMILY_NS, *args, **kwargs)


def ramond_B_block(*args, **kwargs):
    return super_ordered_B_block(FAMILY_RAMOND, *args, **kwargs)


def ns_G_block(*args, **kwargs):
    return super_scalar_G_block(FAMILY_NS, *args, **kwargs)


def ramond_G_block(*args, **kwargs):
    return super_scalar_G_block(FAMILY_RAMOND, *args, **kwargs)


def super_chiral_block(family: str, *args, **kwargs):
    """Family-dispatching alias for the numeric chiral block API."""
    family = _resolve_family(family)
    evaluator = ns_chiral_block if family == FAMILY_NS else ramond_chiral_block
    return evaluator(*args, **kwargs)


def super_full_block(family: str, basis: str, *args, **kwargs):
    """Family/basis-dispatching alias for direct numeric full blocks."""
    family = _resolve_family(family)
    prefix = "ramond_" if family == FAMILY_RAMOND else "ns_"
    short = basis[len(prefix):] if basis.startswith(prefix) else basis
    if short == "scalar_G":
        return super_scalar_G_block(family, *args, **kwargs)
    if short == "ordered_B":
        return super_ordered_B_block(family, *args, **kwargs)
    if short == "parity_projected_B":
        return super_parity_projected_B_block(family, *args, **kwargs)
    raise ValueError(f"unknown super full-block basis {basis!r}")


def _limit_pole_positions(pool, eps_exp):
    """At an offset minimal-model c + 10^-eps_exp, put the pairs the offset split off real poles back on the axis.

    A real double pole of the c -> c0 limit (at c = 3/2 all of Delta = n^2/8) appears at c + eps as a complex pair
    O(sqrt eps) off the axis with canceling residues of size 1/sqrt(eps); the numerators then cancel those
    digits (a dps-60 table had none left).  The limit function has the real double pole, and differs from the
    offset one by O(eps), so the pair is represented by two copies of its real part.  Genuine complex poles are
    O(1) off the axis; the band is eps^(1/4).
    """
    band = mp.mpf(10) ** (-(int(eps_exp) // 4))
    return [mp.mpf(mp.re(p)) if 0 < abs(mp.im(mp.mpc(p))) < band else p for p in pool]


def _pool_for_full(family, c, h_ext, hbar_ext, *, q_order, sector, branch,
                   ell, level=None):
    """The full block's pole pool from both chiral pools, through the half-level ``level`` (default the candidate
    pool's ``2 q_order + 4``)."""
    order = max(1, 2 * int(q_order) + 4) if level is None else int(level)
    left = _chiral_poles(
        family, c, h_ext, q_order=order, sector=sector, branch=branch
    )
    right = _chiral_poles(
        family, c, hbar_ext, q_order=order, sector=sector, branch=branch
    )
    return [2 * d - ell for d in left] + [2 * d + ell for d in right]


def _merged_chiral_pool(family, c_value, h_ext, *, q_order, sector, branch, level=None):
    """The chiral pole pool at an N=1 singular c: the exact merged poles of the Laurent table (each location by its
    order, :func:`super_laurent.merged_pole_pool`), through the half-level ``level`` (default the pool's
    ``2 q_order + 4``)."""
    from .super_laurent import exact_singular_c, merged_pole_pool, merged_state

    c0 = exact_singular_c(c_value)
    order = max(1, 2 * int(q_order) + 4) if level is None else int(level)
    return merged_pole_pool(merged_state(family, h_ext, c0=c0, order=order, sector=sector, branch=branch))


def _stamp_super_table(table, *, policy, family, c_input, c_eval,
                       regularized, sector, branch, exact=False):
    """Attach non-numerical provenance without changing the table schema."""
    metadata = policy.metadata(
        family, c_input=c_input, c_eval=c_eval, regularized=regularized,
        sector=sector, branch=branch,
    )
    if exact:
        # built at the singular c itself from the exact Laurent table, its merged poles the skeleton: no offset
        metadata["regularization_policy"] = "merged"
        metadata["central_charge_evaluated"] = c_input
        metadata["regularization_eps_exp"] = None
    table.metadata = metadata
    # Keep the descriptive alias for callers of the first adapter revision.
    table.super_metadata = metadata
    return table


@guard_super_inputs()
def super_chiral_F_interpolated_table(
    family: str, coordinate: str, h_ext: Sequence, *, c, m_max: int,
    q_order: int, chiral_norm=None, z_shift=None, sector=SECTOR_EVEN,
    branch="++", npoles: int = 8, pole_selection="contribution_optimal", shift=None,
    center=None, regularize="auto", eps_exp: int | None = None, num_threads: int = 1,
    shared_sample_points=False, kept_entries=None,
    policy: SuperBlockPolicy | None = None,
):
    """Build an interpolated NS/Ramond chiral ``F(h)`` table.

    Interpolation probes around genuine poles through a private sampler and
    retains the same fail-closed policy in serialized metadata.
    """
    family = _resolve_family(family)
    sector = normalize_sector(sector)
    branch = normalize_branch(branch)
    policy = resolve_super_policy(policy, regularize=regularize, eps_exp=eps_exp)
    c_input, c_eval, regularized = policy.resolve_c(c)
    c_value, exact = _exact_singular(policy, c)
    base = _half_exp_base(coordinate, center, chiral_norm)
    shift = mp.mpf(0) if shift is None else mp.mpf(shift)
    sampler = SuperChiralSampler(
        family, coordinate, h_ext, c=c_value, q_order=q_order, m_max=m_max,
        chiral_norm=chiral_norm, z_shift=z_shift, sector=sector,
        branch=branch, center=center, c_input=c_input, c_eval=c_value,
        regularized=regularized and not exact, eps_exp=policy.eps_exp, exact=exact,
    )
    # The selector must rank against one q-order-defined pool so that changing
    # npoles means taking a longer prefix of the same nested ordering.
    pool_order = max(1, 2 * int(q_order) + 4)
    # the block's own poles (its recursion stops at half-level 2 q_order, two levels short of the pool): a forced
    # pole past them has no principal part and is left out of the table
    level = ns_order_for_q_order(q_order)
    if exact:
        pool = _merged_chiral_pool(family, c_value, h_ext, q_order=q_order, sector=sector, branch=branch)
        block_pool = _merged_chiral_pool(family, c_value, h_ext, q_order=q_order, sector=sector, branch=branch,
                                         level=level)
    else:
        with _guarded(_offset_guard_dps(regularized, pool_order, policy.eps_exp)):
            pool, block_pool = (_chiral_poles(family, c_eval, h_ext, q_order=order, sector=sector, branch=branch)
                                for order in (pool_order, level))
        if regularized:
            pool, block_pool = (_limit_pole_positions(p, policy.eps_exp) for p in (pool, block_pool))
    decay = int(sector)

    def entry_power(m, n):
        return int(m) - decay

    table = _interpolate_global_table(
        sampler=sampler, m_max=m_max, n_max=0, base=base, shift=shift,
        pool=pool, residue_scorer=_make_residue_scorer(
            sampler, base, m_max, 0,
            score_entries=pole_selection_score_entries(pole_selection),
        ),
        entry_power=entry_power, npoles=npoles, pole_selection=pole_selection,
        num_threads=num_threads, shared_sample_points=shared_sample_points,
        kept_entries=kept_entries, force_keep_real_positive=True,
        fixed_candidate_pool=True, block_pool=block_pool,
    )
    return _stamp_super_table(
        table, policy=policy, family=family, c_input=c_input, c_eval=c_eval,
        regularized=regularized, sector=sector, branch=branch, exact=exact,
    )


@guard_super_inputs()
def super_2d_interpolated_table(
    family: str, basis: str, coordinate: str, ell: int, h_ext: Sequence, *,
    hbar_ext=None, c, m_max: int, n_max: int = None, q_order: int,
    chiral_norm=None, z_shift=None, zbar_shift=None, sector=SECTOR_EVEN,
    branch="++", parity=None, parity_external=None, exchanged_parity="1",
    npoles: int = 8, pole_selection="contribution_optimal", shift=None, center=None,
    center_bar=None, regularize="auto", eps_exp: int | None = None,
    num_threads: int = 1, shared_sample_points=False, kept_entries=None,
    policy: SuperBlockPolicy | None = None,
):
    """Build an interpolated full NS/Ramond scalar, ordered, or projected block."""
    ell = _integer_spin(ell)
    family = _resolve_family(family)
    if n_max is None:
        n_max = m_max
    if hbar_ext is None:
        hbar_ext = h_ext
    if coordinate not in _NON_FOLDED_2D | _FOLDED_2D | _RADIAL_2D:
        raise ValueError(f"unknown full-block coordinate {coordinate!r}")
    sector = normalize_sector(sector)
    branch = normalize_branch(branch)
    policy = resolve_super_policy(policy, regularize=regularize, eps_exp=eps_exp)
    c_input, c_eval, regularized = policy.resolve_c(c)
    c_value, exact = _exact_singular(policy, c)
    sigma, fold, kind = _resolve_2d_sigma_basis(
        basis, ell, parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity,
    )
    folded = coordinate in _FOLDED_2D | _RADIAL_2D
    sampler = SuperBlockSampler2d(
        family, coordinate, ell, h_ext, hbar_ext, c=c_value,
        q_order=q_order, m_max=m_max, n_max=n_max, sigma=sigma,
        fold_parity=fold, chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=z_shift if zbar_shift is None else zbar_shift,
        sector=sector, branch=branch, center=center, center_bar=center_bar,
        c_input=c_input, c_eval=c_value, regularized=regularized and not exact,
        eps_exp=policy.eps_exp, exact=exact, scalar=kind == "scalar",
    )
    if folded:
        center_bar = None
    base_left = _half_exp_base(
        COORDINATES[coordinate].chiral_var if folded else coordinate,
        center, chiral_norm,
    )
    base_right = _half_exp_base(
        COORDINATES[coordinate].chiral_var if folded else coordinate,
        center_bar if center_bar is not None else center, chiral_norm,
    )
    base = mp.sqrt(base_left * base_right)
    shift = mp.mpf(abs(int(ell))) if shift is None else mp.mpf(shift)
    # the pool is computed under the sampler's offset guard: at a working precision below eps_exp, c + eps
    # rounded back to the singular c (NSDegenerateCError)
    # and the block's own poles (its recursion stops two levels short of the pool; see the chiral table)
    levels = (None, ns_order_for_q_order(q_order))
    if exact:
        def merged_pool(level):
            left = _merged_chiral_pool(family, c_value, h_ext, q_order=q_order, sector=sector, branch=branch,
                                       level=level)
            right = _merged_chiral_pool(family, c_value, hbar_ext, q_order=q_order, sector=sector, branch=branch,
                                        level=level)
            return [2 * d - ell for d in left] + [2 * d + ell for d in right]

        pool, block_pool = (merged_pool(level) for level in levels)
    else:
        with _guarded(_offset_guard_dps(regularized, 2 * int(q_order) + 4, policy.eps_exp)):
            pool, block_pool = (_pool_for_full(family, c_eval, h_ext, hbar_ext, q_order=q_order, sector=sector,
                                               branch=branch, ell=ell, level=level) for level in levels)
        if regularized:
            pool, block_pool = (_limit_pole_positions(p, policy.eps_exp) for p in (pool, block_pool))
    # Both odd-sector chiral blocks begin one inverse power of the exchanged
    # weight down.  The antisymmetric parity '-' combination (sigma = -1; never
    # the scalar G, whose sigma is +1) has one further cancellation when the two
    # halves are identical.
    decay = 2 * int(sector)
    # (off the diagonal the halves carry different base weights and nothing cancels -- as for Virasoro scalar_G)
    equal_halves = list(h_ext) == list(hbar_ext) and z_shift == (
        z_shift if zbar_shift is None else zbar_shift
    ) and not is_off_diagonal(center, center_bar)
    drop = 1 if equal_halves and sigma == -1 else 0

    def entry_power(m, n):
        if folded:
            return folded_entry_power(m, n, fold) - decay - drop
        return int(m) + int(n) - decay - drop

    table = _interpolate_global_table(
        sampler=sampler, m_max=m_max, n_max=(0 if coordinate in _RADIAL_2D else n_max),
        base=base, shift=shift, pool=pool,
        residue_scorer=_make_residue_scorer(
            sampler, base, m_max,
            0 if coordinate in _RADIAL_2D else n_max,
            score_entries=pole_selection_score_entries(pole_selection),
        ),
        entry_power=entry_power, npoles=npoles, pole_selection=pole_selection,
        num_threads=num_threads, shared_sample_points=shared_sample_points,
        kept_entries=kept_entries, force_keep_real_positive=True,
        fixed_candidate_pool=True, block_pool=block_pool,
    )
    return _stamp_super_table(
        table, policy=policy, family=family, c_input=c_input, c_eval=c_eval,
        regularized=regularized, sector=sector, branch=branch, exact=exact,
    )


def ns_chiral_F_interpolated_table(*args, **kwargs):
    return super_chiral_F_interpolated_table(FAMILY_NS, *args, **kwargs)


def ramond_chiral_F_interpolated_table(*args, **kwargs):
    return super_chiral_F_interpolated_table(FAMILY_RAMOND, *args, **kwargs)


__all__ = [
    "FAMILY_NS", "FAMILY_RAMOND", "FAMILY_R", "SECTOR_EVEN", "SECTOR_ODD",
    "BRANCH_PP", "BRANCH_PM", "BRANCH_MP", "BRANCH_MM",
    "SuperconformalPoleError", "SuperBlockPolicy", "DEFAULT_SUPER_BLOCK_POLICY",
    "normalize_sector", "normalize_branch", "resolve_super_policy",
    "is_superconformal_singular_c", "resolve_super_c",
    "ns_chiral_block", "ramond_chiral_block", "super_ordered_B_block",
    "super_parity_reversed_B_block", "super_parity_projected_B_block",
    "super_scalar_G_block", "ns_B_block", "ramond_B_block", "ns_G_block",
    "ramond_G_block", "super_chiral_block", "super_full_block",
    "numeric_chiral_table", "numeric_full_table", "SuperChiralSampler",
    "SuperBlockSampler2d", "super_chiral_F_interpolated_table",
    "ns_chiral_F_interpolated_table", "ramond_chiral_F_interpolated_table",
    "super_2d_interpolated_table",
]
