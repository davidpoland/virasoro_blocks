"""Moment-bound builders for identical-scalar Virasoro bootstrap problems.

The moment bootstrap bounds linear functionals of the positive self-dual block
measure by adding moment-polynomial components to the usual crossing-vector
PMP. This module implements the construction on the fixed-spin channels plus,
by default, the large-spin tower slices.

A moment of a weight ``W`` is

    M[W] = sum_O lambda_O^2 G_O * W(Delta_O, ell_O),

summed over the exchanged Virasoro primaries, where ``G_O`` is the operator's
block at the crossing-symmetric point (the ``(0, 0)`` entry of its table, in
the problem's normalization, where the identity is ``F_0 F_0`` with coefficient
1 and ``lambda_O^2`` stands for ``p_O = N^Delta lambda_O^2`` in every channel)
and ``W`` is a polynomial in ``Delta``, ``L``
(the spin), ``h`` and ``hbar`` (:class:`MomentWeight`).  With
``identity_mode="include"`` the Virasoro vacuum term is added; ``"exclude"``
leaves it out.

Entry points, all taking an :class:`IdenticalScalarGapSpec` (whose gap and
spins set the positivity assumptions) and an optional ``prepared=`` build:

- :func:`build_identical_scalar_normalized_moment_lower_problem` /
  :func:`build_identical_scalar_normalized_moment_upper_problem`: bounds on
  the ratio ``R = (M[numerator] + beta) / (M[denominator] + alpha)``, for
  example ``<Delta> = M[Delta] / M[1]``, in one SDPB solve.  Convert SDPB's
  ``dualObjective`` with :func:`normalized_moment_lower_bound_from_objective`
  / :func:`normalized_moment_upper_bound_from_objective`.  The denominator
  ``M[denominator] + alpha`` must be strictly positive on every admissible
  spectrum (see :func:`build_identical_scalar_normalized_moment_lower_problem`).
- :func:`build_identical_scalar_moment_lower_problem` /
  :func:`build_identical_scalar_moment_upper_problem`: bounds on one
  unnormalized ``M[W]``; convert with :func:`moment_lower_bound_from_objective`
  / :func:`moment_upper_bound_from_objective`.
- :func:`build_identical_scalar_normalized_ratio_probe_problem` and the two
  ``bound_..._by_bisection`` helpers: the same ratio bounds reached through
  unnormalized probes, kept as an independent cross-check.

Solve them with :func:`virasoro_bootstrap.run_identical_scalar_gap_problem`
in optimization mode (no ``--findPrimalFeasible`` / ``--findDualFeasible``),
read the objective with :func:`virasoro_bootstrap.read_sdpb_objective`, and
check the functional with
:func:`virasoro_bootstrap.certify_dual_from_sdpb_output`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace as _dataclass_replace
from fractions import Fraction
from typing import Callable, Mapping

import mpmath as mp
from flint import arb_poly, ctx, fmpq, fmpq_poly

from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.numeric_input import (
    ExactRational, _integer_control, literal_number, parse_number, _evaluation_number,
)
from virasoro_blocks.writers import _metadata_num_str
from virasoro_blocks.interpolation import (
    DampedRational,
    get_sample_points,
    lagrange_interpolation_polynomial,
)

from .identical_scalar import (
    IdenticalScalarGapProblem,
    IdenticalScalarGapSpec,
    _as_real_mpf,
    _as_real_label,
    _convert_number,
    build_identical_scalar_gap_problem,
    self_dual_identity_measure_value,
)
from .pmp_writer import (
    PrefactoredPmpBuild,
    _pmp_real_str,
    _pmp_real_value,
    _polynomial_to_str_list,
    _resolve_sig_figs,
    _sampling_point_labels,
    _saved_sample_tables_at_nodes,
    build_prefactored_pmp_doc,
)
from .crossing import _at_block_precision


_MOMENT_VARIABLES = frozenset({"Delta", "L", "h", "hbar"})
_RATIO_BLOCK_BASIS = "ratio"
_NONRATIO_BLOCK_BASIS = "nonratio"


def _moment_real_label(value, *, name):
    """A validated rational source, including the stored bits of an mpf."""
    label = literal_number(value)
    if isinstance(label, (int, Fraction)) and not isinstance(label, bool):
        return ExactRational(label)
    return _as_real_label(label, name=name)


def _moment_input_text(value, *, name):
    return _metadata_num_str(_moment_real_label(value, name=name), mp.mp.dps + 5)


def _moment_rational(value, *, name):
    label = _moment_real_label(value, name=name)
    return fmpq(label.numerator, label.denominator)


def _validated_moment_powers(powers):
    cleaned = {}
    for name, power in dict(powers or {}).items():
        if name not in _MOMENT_VARIABLES:
            raise ValueError(f"unknown moment variable {name!r}; expected one of {sorted(_MOMENT_VARIABLES)}")
        try:
            integer = _integer_control('moment power', power)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("moment powers must be non-negative integers") from None
        cleaned[name] = integer
    return cleaned


@dataclass(frozen=True)
class MomentTerm:
    """One monomial ``coefficient * prod_v v**powers[v]`` of a moment weight.

    ``coefficient``
        A number (``int``, ``str``, ``Fraction`` or ``mpf``); default 1.
    ``powers``
        ``{variable: power}`` with variables ``"Delta"``, ``"L"`` (the spin),
        ``"h"`` and ``"hbar"`` and non-negative integer powers; ``None`` or
        ``{}`` is the constant monomial.  ``MomentTerm(2, {"Delta": 1, "L": 2})``
        is ``2 Delta L^2``.
    """

    coefficient: object = 1
    powers: Mapping[str, int] | None = None

    def to_json(self) -> dict:
        return {
            "coefficient": _moment_input_text(self.coefficient, name="MomentTerm coefficient"),
            "powers": _validated_moment_powers(self.powers),
        }

    @staticmethod
    def from_json(data: dict) -> "MomentTerm":
        return MomentTerm(
            # Parsing precedes the manifest's working-precision context.
            # Keep every stored digit until the numerical builder consumes it.
            coefficient=parse_number(data["coefficient"]),
            powers=_validated_moment_powers(data.get("powers")),
        )


@dataclass(frozen=True)
class MomentWeight:
    """A polynomial weight ``W(Delta, L, h, hbar)``: a sum of :class:`MomentTerm`.

    ``terms``
        The monomials.  Build common weights with :meth:`one` (``W = 1``, the
        zeroth moment) and :meth:`monomial` (``MomentWeight.monomial(Delta=1)``
        is ``W = Delta``), and combine them with :meth:`plus`, :meth:`minus`,
        :meth:`scale` and unary minus.
    ``spin_filter``
        ``None`` (default) weights every exchanged spin.  An even integer
        ``ell`` restricts the weight to the fixed-spin channel of that spin
        (and to the identity when ``ell == 0``); it cannot select a spin inside
        the large-spin tower.  A filter that selects no channel of the problem
        (a spin not in ``spin_list``, an odd or negative spin) is refused: the
        moment would be identically zero.
    """

    terms: tuple[MomentTerm, ...]
    spin_filter: int | None = None

    def __post_init__(self) -> None:
        # int() truncated a filter of 2.5 to spin 2; whether it selects a channel is checked against the problem
        # (_moment_channels), which knows the spins
        spin = self.spin_filter
        if spin is None:
            return
        try:
            from virasoro_blocks.requests import _integer_field

            spin = _integer_field('spin_filter', spin)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"spin_filter must be an integer spin or None, got {spin!r}") from None
        object.__setattr__(self, "spin_filter", spin)

    @staticmethod
    def one() -> "MomentWeight":
        return MomentWeight((MomentTerm(1, {}),))

    @staticmethod
    def monomial(**powers: int) -> "MomentWeight":
        return MomentWeight((MomentTerm(1, powers),))

    def scale(self, scalar) -> "MomentWeight":
        scalar = _moment_real_label(scalar, name="MomentWeight.scale factor")
        return MomentWeight(
            tuple(
                MomentTerm(scalar * _moment_real_label(term.coefficient, name="MomentTerm coefficient"), dict(term.powers or {}))
                for term in self.terms
            ),
            spin_filter=self.spin_filter,
        )

    def __neg__(self) -> "MomentWeight":
        return self.scale(-1)

    def plus(self, other: "MomentWeight") -> "MomentWeight":
        if self.spin_filter != other.spin_filter:
            raise ValueError(
                "cannot combine MomentWeight values with different spin_filter"
            )
        return MomentWeight(self.terms + other.terms, spin_filter=self.spin_filter)

    def minus(self, other: "MomentWeight") -> "MomentWeight":
        return self.plus(other.scale(-1))

    def to_json(self) -> dict:
        return {
            "terms": [term.to_json() for term in self.terms],
            "spin_filter": None if self.spin_filter is None else int(self.spin_filter),
        }

    @staticmethod
    def from_json(data: dict) -> "MomentWeight":
        sf = data.get("spin_filter")
        return MomentWeight(
            tuple(MomentTerm.from_json(t) for t in data["terms"]),
            spin_filter=sf,
        )


@dataclass(frozen=True)
class MomentBoundSpec:
    """A record of one moment bound: what is bounded, under which assumptions.

    The builders take these fields as arguments; this record keeps them
    together (for example to label a result).

    ``base_spec``
        The :class:`IdenticalScalarGapSpec` whose spins, gap and truncation
        define the positivity assumptions.
    ``numerator``, ``denominator``
        The weights of ``R = (M[numerator] + beta) / (M[denominator] + alpha)``;
        ``denominator=None`` means the unnormalized ``M[numerator]``.
    ``alpha``, ``beta``
        Constants added to the denominator and numerator (default 0).
    ``identity_mode``
        ``"include"`` (default) adds the Virasoro vacuum to both moments;
        ``"exclude"`` sums over non-identity primaries only.
    ``direction``
        ``"upper"`` (default) or ``"lower"``.
    """

    base_spec: IdenticalScalarGapSpec
    numerator: MomentWeight
    denominator: MomentWeight | None = None
    alpha: object = 0
    beta: object = 0
    identity_mode: str = "include"
    direction: str = "upper"


def _at_spec_dps(builder):
    """Run a moment builder at the spec's working precision, not the ambient one.

    The gap builder enters ``mp.workdps(spec.dps)`` itself, but the moment layer
    parses weights, identity shifts and PMP coefficients around it; at a lower
    ambient precision those were silently rounded (a dps=60 spec built at
    ambient 15 digits differed in the 17th digit).  The first positional
    argument is either the spec or a problem carrying one.
    """
    @functools.wraps(builder)
    def wrapper(first, *args, **kwargs):
        first, kwargs = at_working_precision(first, kwargs)
        spec = getattr(first, "spec", first)
        with mp.workdps(int(spec.dps)):
            return builder(first, *args, **kwargs)

    return wrapper


def at_working_precision(spec, kwargs):
    """A ``target_digits`` spec at the working precision its blocks are built at, with the ``prepared`` build that
    chose it (``(spec, kwargs)`` unchanged otherwise).

    The problem builders parse weights and assemble the PMP around the gap build at ``spec.dps``; for a
    ``target_digits`` spec the blocks are built at a raised working precision
    (:func:`~virasoro_bootstrap.identical_scalar.prepare_identical_scalar_gap_build`), and a PMP assembled at the
    requested ``dps`` would round them back.  An injected ``base_problem`` or ``prepared`` build fixes it.
    """
    from dataclasses import replace

    from .identical_scalar import IdenticalScalarGapSpec, prepare_identical_scalar_gap_build

    if not isinstance(spec, IdenticalScalarGapSpec) or spec.target_digits is None:
        return spec, kwargs
    base_problem, prepared = kwargs.get("base_problem"), kwargs.get("prepared")
    if base_problem is not None:
        work = int(base_problem.spec.dps)
    else:
        if prepared is None:
            prepared = prepare_identical_scalar_gap_build(spec)
            kwargs = dict(kwargs, prepared=prepared)
        work = int(prepared.working_dps or spec.dps)
    return replace(spec, dps=max(int(spec.dps), work)), kwargs


def _poly_const(value, *, field=mp.mpf) -> Polynomial:
    return Polynomial([field(value)], field=field)


def _channel_variable_polys(channel) -> dict[str, fmpq_poly]:
    """Return Delta/L/h/hbar as polynomials in the channel's positivity variable x.

    Fixed-spin channels vary Delta at fixed spin: x = Delta - shift, L = ell.
    Large-spin tower slices invert this -- the positivity variable is the spin
    L (x = L - Lmax) at fixed twist tau, so Delta = tau + L and hbar = tau/2.
    """

    shift = _moment_rational(channel.prefactored.shift, name="channel shift")
    x = fmpq_poly([0, 1])
    twist = getattr(channel, "twist", None)
    if twist is not None:
        tau = _moment_rational(twist, name="twist")
        L = x + shift
        Delta = L + tau
    else:
        Delta = x + shift
        L = fmpq_poly([_moment_rational(channel.ell, name="spin")])
    h = (Delta + L) / 2
    hbar = (Delta - L) / 2
    return {"Delta": Delta, "L": L, "h": h, "hbar": hbar}


def _weight_polynomial_from_variables(
    weight: MomentWeight, variables: dict[str, fmpq_poly]
) -> Polynomial:
    """Evaluate a moment weight on a given Delta/L/h/hbar polynomial basis."""

    # Combine the complete weight before rounding: separately rounded
    # monomials can destroy small coefficients or exact cancellations.
    total = fmpq_poly([])
    for term in weight.terms:
        coeff = _moment_rational(term.coefficient, name="MomentTerm coefficient")
        mono = fmpq_poly([coeff])
        powers = _validated_moment_powers(term.powers)
        for name, power in powers.items():
            mono *= variables[name] ** power
        total = total + mono
    # fmpq is already reduced. Convert it in flint before extracting numeric
    # midpoints: re-reducing large Fraction coefficients in mpmath is costly.
    old_prec = ctx.prec
    try:
        ctx.prec = mp.mp.prec
        coefficients = [mp.make_mpf(c._mpf_) for c in arb_poly(total).coeffs()]
    finally:
        ctx.prec = old_prec
    return Polynomial._from_coeffs_trusted(coefficients, field=mp.mpf)


def moment_weight_polynomial(weight: MomentWeight, channel) -> Polynomial:
    """Convert a moment weight to a polynomial in a channel's variable x."""

    return _weight_polynomial_from_variables(weight, _channel_variable_polys(channel))


def _channel_variable_polys_in_delta(channel) -> dict[str, fmpq_poly]:
    """Return Delta/L/h/hbar as polynomials in Delta itself (base variable Delta).

    A public helper (the builders do not use it): the moment weight in Delta
    rather than in the positivity variable x = Delta - shift. For a fixed-spin
    channel L = ell is constant; for a large-spin tower slice at twist tau the
    spin is L = Delta - tau.
    """

    Delta = fmpq_poly([0, 1])  # the polynomial variable *is* Delta here
    twist = getattr(channel, "twist", None)
    if twist is not None:
        L = Delta - _moment_rational(twist, name="twist")
    else:
        L = fmpq_poly([_moment_rational(channel.ell, name="spin")])
    h = (Delta + L) / 2
    hbar = (Delta - L) / 2
    return {"Delta": Delta, "L": L, "h": h, "hbar": hbar}


def moment_weight_polynomial_in_delta(weight: MomentWeight, channel) -> Polynomial:
    """Convert a moment weight to a polynomial in Delta (for weighted blocks)."""

    return _weight_polynomial_from_variables(
        weight, _channel_variable_polys_in_delta(channel)
    )


def _grid_value_to_mpc(value):
    """Coerce a block table entry (mpf/mpc or flint arb/acb) to mpmath mpc."""

    if isinstance(value, (mp.mpf, mp.mpc, int, float)):
        return mp.mpc(value)
    real = getattr(value, "real", None)
    imag = getattr(value, "imag", None)
    if real is not None and imag is not None and hasattr(real, "_mpf_"):
        return mp.make_mpc((real._mpf_, imag._mpf_))  # flint acb
    if hasattr(value, "_mpf_"):  # flint arb (real)
        return mp.mpc(mp.make_mpf(value._mpf_))
    return mp.mpc(complex(value))


def _moment_channels(problem: IdenticalScalarGapProblem, *, include_tower: bool, weights=()):
    """Positivity channels the moment insertion attaches to.

    Always the fixed-spin channels; with ``include_tower`` the large-spin tower
    slices are appended so their continuous-in-L positivity covers the moment's
    large-spin contribution (Delta = tau + L at each twist). Point-constraint
    channels are not supported for moments.  Each of ``weights`` with a
    ``spin_filter`` must select one of the fixed-spin channels.
    """
    n_fixed = len(problem.spec.spin_list)
    if problem.point_channels:
        raise NotImplementedError(
            "moment insertions for point-constraint channels are not implemented yet"
        )
    channels = list(problem.spin_channels[:n_fixed])
    if include_tower and problem.large_spin_channels:
        if _as_real_label(problem.spec.large_spin_lmin, name="large_spin_lmin") == 0:
            # the tower is analytic in L, so its L = 0 member is the doubled block 2 B(h, h), while a fixed spin-0
            # channel is B(h, h); a moment would count that member twice.  Compared by value, so "0", "0.0" and
            # "0/1" are refused too (the spec validation refuses a zero floor; this guards a problem assembled
            # without it)
            raise ValueError(
                "moments need large_spin_lmin >= 1 (list spin 0 in spin_list instead): the tower's L = 0 member "
                "carries the doubled spin-0 block"
            )
        channels.extend(problem.large_spin_channels)
    _check_spin_filters(channels, weights)
    return tuple(channels)


def _check_spin_filters(channels, weights) -> None:
    """Refuse a ``spin_filter`` that selects none of ``channels`` (every moment component would be zero)."""
    fixed_spins = sorted({int(ch.ell) for ch in channels if getattr(ch, "twist", None) is None})
    tower_floors = [ch.delta_min for ch in channels if getattr(ch, "twist", None) is not None]
    for weight in weights:
        spin = None if weight is None else weight.spin_filter
        if spin is None or spin in fixed_spins:
            continue
        if tower_floors and spin >= min(mp.mpf(floor) for floor in tower_floors):
            continue    # inside the tower: _channel_weight_poly says why a slice cannot isolate it
        tower = (f"; the large-spin tower covers L >= {mp.nstr(min(mp.mpf(f) for f in tower_floors), 6)}"
                 if tower_floors else "")
        raise ValueError(
            f"spin_filter = {spin} selects no channel of the problem, whose fixed spins are {fixed_spins}{tower}: "
            "the filtered moment would be identically zero (identical scalars exchange even spins >= 0 only)"
        )


def _self_dual_measure_poly(problem: IdenticalScalarGapProblem, channel) -> Polynomial:
    """Return the self-dual measure numerator polynomial for a fixed-spin channel."""

    rows = channel.prefactored.numerators[problem.spec.coordinate]
    return rows[0][0]


def _validate_identity_mode(identity_mode: str) -> None:
    if identity_mode not in {"include", "exclude"}:
        raise ValueError("identity_mode must be 'include' or 'exclude'")


def _identity_weight_label(weight: MomentWeight):
    """Evaluate a moment weight on the identity quantum numbers."""

    if weight.spin_filter is not None and int(weight.spin_filter) != 0:
        return ExactRational(0)

    value = ExactRational(0)
    for term in weight.terms:
        powers = _validated_moment_powers(term.powers)
        if any(power > 0 for power in powers.values()):
            continue
        value += _moment_real_label(term.coefficient, name="MomentTerm coefficient")
    return value


def _identity_weight_value(weight: MomentWeight) -> mp.mpf:
    return mp.convert(_identity_weight_label(weight))


def _identity_moment_label(problem: IdenticalScalarGapProblem, weight: MomentWeight):
    """Return the Virasoro vacuum contribution to a moment weight.

    The positivity blocks in this module encode non-vacuum operator sums.  This
    helper supplies the separate Virasoro vacuum self-dual measure value when a
    caller wants full, identity-including moments.
    """

    weight_value = _identity_weight_label(weight)
    if weight_value == 0:
        return ExactRational(0)
    # Prefer an explicit self-dual vacuum measure stored on the problem (e.g. the
    # global control's c-independent global vacuum) over recomputing the Virasoro
    # vacuum from spec.c, so the identity term matches the positivity block family.
    stored = (problem.pmp_metadata or {}).get("self_dual_identity_measure")
    if stored is not None:
        vacuum_value = _moment_real_label(stored, name="identity measure")
    else:
        vacuum_value = self_dual_identity_measure_value(
            problem.spec.c,
            problem.spec.h_ext,
            problem.spec.lam,
            problem.spec.q_order,
            coordinate=problem.spec.coordinate,
            chiral_norm=problem.spec.chiral_norm,
        )
    return weight_value * _moment_real_label(vacuum_value, name="identity measure")


def _identity_moment_contribution(problem: IdenticalScalarGapProblem, weight: MomentWeight):
    return mp.convert(_identity_moment_label(problem, weight))


def _effective_unnormalized_identity_shift(
    problem: IdenticalScalarGapProblem,
    weight: MomentWeight,
    *,
    identity_mode: str,
) -> mp.mpf:
    _validate_identity_mode(identity_mode)
    if identity_mode == "exclude":
        return mp.mpf(0)
    return mp.mpf(_identity_moment_contribution(problem, weight))


def _effective_ratio_constants(
    problem: IdenticalScalarGapProblem,
    numerator: MomentWeight,
    denominator: MomentWeight,
    *,
    alpha,
    beta,
    identity_mode: str,
) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    _validate_identity_mode(identity_mode)
    alpha = _moment_real_label(alpha, name="alpha")
    beta = _moment_real_label(beta, name="beta")
    identity_numerator = ExactRational(0)
    identity_denominator = ExactRational(0)
    if identity_mode == "include":
        identity_numerator = _identity_moment_label(problem, numerator)
        identity_denominator = _identity_moment_label(problem, denominator)
        beta += identity_numerator
        alpha += identity_denominator
    _refuse_nonpositive_denominator(denominator, alpha, identity_mode)
    return tuple(mp.convert(value) for value in (alpha, beta, identity_numerator, identity_denominator))


def _refuse_nonpositive_denominator(denominator: MomentWeight, alpha_eff, identity_mode: str) -> None:
    """Refuse a ratio whose denominator ``M[denominator] + alpha_eff`` is never positive.

    Every channel has ``Delta, L, h, hbar >= 0``, so a weight with no positive coefficient has ``M[W] <= 0``; with
    ``alpha_eff <= 0`` the denominator is then nonpositive on every spectrum, and the linear-fractional program would
    bound a different quantity (a constant ratio ``1 / (-1)`` came out with a lower bound of 10).  Other signed
    denominators cannot be decided here and stay the caller's obligation."""
    # like monomials combined first: 1 - 1 is the zero weight, not one positive and one negative term
    combined: dict = {}
    for term in denominator.terms:
        key = tuple(sorted((var, power) for var, power in _validated_moment_powers(term.powers).items() if power))
        combined[key] = combined.get(key, 0) + _moment_real_label(term.coefficient, name="MomentTerm coefficient")
    if mp.convert(alpha_eff) <= 0 and all(mp.convert(value) <= 0 for value in combined.values()):
        shift = "alpha plus the identity's contribution" if identity_mode == "include" else "alpha"
        raise ValueError(
            "the ratio's denominator M[denominator] + alpha is never positive: the denominator weight has no positive "
            f"coefficient (and Delta, L, h, hbar >= 0 in every channel), and {shift} is "
            f"{mp.nstr(mp.convert(alpha_eff), 8)} <= 0; normalized moment bounds need a strictly positive denominator")


def _resolve_base_problem(spec, prepared, base_problem):
    """Return the positivity problem the moment components attach to.

    ``base_problem`` lets callers inject an already-built problem on the same PMP
    surface -- e.g. a global-block control from
    ``build_global_identical_scalar_control_problem`` -- so the moment encoding
    can be validated against the global-block ``<Delta>`` bounds of 2603.18140.
    Its specification must match, allowing the worker count and an implicit
    versus explicit crossing prefactor (both validated to equal h_ext) or
    default chiral normalization.
    """
    if base_problem is not None:
        from dataclasses import fields
        from virasoro_blocks.recursion import _resolve_chiral_norm
        from .identical_scalar import (_validate_gap_spec_common, _resolved_delta_min_by_ell,
                                       _canonical_non_scalar_delta_min_overrides)

        _validate_gap_spec_common(spec)

        def comparable(value_spec):
            # Match the builder's canonical labels, including lower edges.
            # The domain and CFT labels stay exact: rounding them could
            # accept a different solved problem after a deliberate small shift.
            with mp.workdps(value_spec.dps):
                values = {f.name: getattr(value_spec, f.name) for f in fields(value_spec)
                          if f.name not in {"num_threads", "prefactor_exponent"}}
                values["h_ext"] = _as_real_label(value_spec.h_ext, name="h_ext")
                if values["large_spin_lmin"] is not None:
                    values["large_spin_lmin"] = _as_real_label(values["large_spin_lmin"], name="large_spin_lmin")
                values["chiral_norm"] = _resolve_chiral_norm(value_spec.chiral_norm)
                values["large_spin_twists"] = tuple(
                    _as_real_label(t, name="large_spin_twists") for t in value_spec.large_spin_twists or ()
                )
                values["point_constraints"] = tuple(
                    (ell, _as_real_label(delta, name="point constraint"))
                    for ell, delta in value_spec.point_constraints or ()
                )
                edges = _resolved_delta_min_by_ell(value_spec, value_spec.spin_list)
                values["delta_gap"] = edges[0]
                values["delta_min_by_spin"] = _canonical_non_scalar_delta_min_overrides(edges)
                return values

        from virasoro_blocks._conventions import scalar_G_spin0_convention

        if base_problem.scalar_G_spin0 != scalar_G_spin0_convention():
            raise ValueError(
                f"base_problem was built with scalar_G_spin0 = {base_problem.scalar_G_spin0!r}, but this process uses "
                f"{scalar_G_spin0_convention()!r}: its blocks and the identity term built here would differ by the "
                "spin-0 factor 2; rebuild it in this process"
            )
        requested, supplied = comparable(spec), comparable(base_problem.spec)
        mismatched = [name for name, value in requested.items() if value != supplied[name]]
        if mismatched:
            raise ValueError("base_problem specification differs in: " + ", ".join(mismatched))
        return base_problem
    return build_identical_scalar_gap_problem(spec, prepared=prepared)


@_at_spec_dps
def build_identical_scalar_moment_upper_problem(
    spec: IdenticalScalarGapSpec,
    weight: MomentWeight,
    *,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
) -> IdenticalScalarGapProblem:
    """Build a PMP whose optimum is an upper bound on an unnormalized moment."""

    gap_problem = _resolve_base_problem(spec, prepared, base_problem)
    channels = _moment_channels(gap_problem, include_tower=include_tower, weights=(weight,))
    identity_shift = _effective_unnormalized_identity_shift(
        gap_problem, weight, identity_mode=identity_mode
    )
    n = len(gap_problem.F_pairs)
    # Operator sum rule sum_{O!=id} lambda^2 F_O = -b. Positivity
    # z.F_O - G_self W_O >= 0 gives M_op <= -z.b, so the tightest upper bound is
    # -max_z(z.b) + I[W]. SDPB *maximizes* objective.dual (calibrated against
    # OPE-max: dualObjective = max(objective.dual)); with objective b-part = +b
    # and s-part = -I[W], dualObjective = max(z.b) - I[W], and the upper bound is
    # -dualObjective (see moment_upper_bound_from_objective).
    objective = [mp.mpf(v) for v in gap_problem.identity_vector] + [
        -identity_shift
    ]
    normalization = [mp.mpf(0)] * n + [mp.mpf(1)]
    pmp_build, sample_values_by_block = _sampled_moment_pmp_doc(
        gap_problem,
        channels=channels,
        extra_components_by_channel=(
            lambda channel: (
                _channel_weight_poly(weight, channel) * mp.mpf(-1),
            )
        ),
        objective=objective,
        normalization=normalization,
        sig_figs=None,
        block_basis=block_basis,
    )
    basis_meta = _moment_basis_metadata(block_basis, include_tower)
    return _dataclass_replace(
        gap_problem,
        spin_channels=channels,
        pmp_build=pmp_build,
        sample_values_by_block=sample_values_by_block,
        problem_kind="moment_upper",
        objective_vector=tuple(objective),
        normalization_vector=tuple(normalization),
        pmp_metadata={
            **(gap_problem.pmp_metadata or {}),
            "moment": {
                "kind": "unnormalized",
                "direction": "upper",
                "numerator": weight.to_json(),
                "identity_mode": identity_mode,
                "identity_shift": mp.nstr(identity_shift, mp.mp.dps + 5),
                **basis_meta,
            }
        },
    )


@_at_spec_dps
def build_identical_scalar_moment_lower_problem(
    spec: IdenticalScalarGapSpec,
    weight: MomentWeight,
    *,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
) -> IdenticalScalarGapProblem:
    """Build the direct dual whose optimum is a lower bound on a moment."""

    gap_problem = _resolve_base_problem(spec, prepared, base_problem)
    channels = _moment_channels(gap_problem, include_tower=include_tower, weights=(weight,))
    identity_shift = _effective_unnormalized_identity_shift(
        gap_problem, weight, identity_mode=identity_mode
    )
    n = len(gap_problem.F_pairs)
    # Positivity z.F_O + G_self W_O >= 0 gives M_op >= z.b, so the tightest lower
    # bound is max_z(z.b) + I[W]. SDPB maximizes objective.dual, so with objective
    # b-part = +b and s-part = +I[W], dualObjective = max(z.b) + I[W] is the lower
    # bound on M_all directly (identity conversion).
    objective = [mp.mpf(v) for v in gap_problem.identity_vector] + [
        identity_shift
    ]
    normalization = [mp.mpf(0)] * n + [mp.mpf(1)]
    pmp_build, sample_values_by_block = _sampled_moment_pmp_doc(
        gap_problem,
        channels=channels,
        extra_components_by_channel=(
            lambda channel: (
                _channel_weight_poly(weight, channel),
            )
        ),
        objective=objective,
        normalization=normalization,
        sig_figs=None,
        block_basis=block_basis,
    )
    basis_meta = _moment_basis_metadata(block_basis, include_tower)
    return _dataclass_replace(
        gap_problem,
        spin_channels=channels,
        pmp_build=pmp_build,
        sample_values_by_block=sample_values_by_block,
        problem_kind="moment_lower",
        objective_vector=tuple(objective),
        normalization_vector=tuple(normalization),
        pmp_metadata={
            **(gap_problem.pmp_metadata or {}),
            "moment": {
                "kind": "unnormalized",
                "direction": "lower",
                "numerator": weight.to_json(),
                "identity_mode": identity_mode,
                "identity_shift": mp.nstr(identity_shift, mp.mp.dps + 5),
                **basis_meta,
            }
        },
    )


def _constant_scalar_block(vector, *, sig_figs: int | None = None) -> dict:
    sig_figs = _resolve_sig_figs(sig_figs)
    poly_vector = [
        [_pmp_real_str(c, sig_figs, where="normalized_moment.scalar_block")]
        for c in vector
    ]
    return {
        "DampedRational": {
            "base": "0.5",
            "constant": "1",
            "poles": [],
        },
        "polynomials": [[poly_vector]],
        "samplePoints": ["0"],
        "sampleScalings": ["1"],
    }


def _pmp_poly_coeffs(poly: Polynomial, *, sig_figs: int, n_sample_points: int | None = None) -> list[str]:
    coeffs = _polynomial_to_str_list(poly, sig_figs)
    if n_sample_points is None:
        return coeffs
    if len(coeffs) > n_sample_points:
        raise ValueError("too few ratio sample points for the polynomial degree")
    # pmp2sdp infers the sampling degree from coefficient-array lengths, even
    # when interpolation simplifies to an exact lower-degree polynomial.
    return coeffs + ["0"] * (n_sample_points - len(coeffs))


def _chebyshev_lobatto_nodes(n: int, a, b) -> tuple:
    n = _integer_control("number of ratio sample nodes", n, minimum=1)
    # A single-node support has equal endpoints. Validate each separately.
    a_label = _sampling_point_labels((a,), where="ratio sample interval")[0]
    b_label = _sampling_point_labels((b,), where="ratio sample interval")[0]
    a, b = (_evaluation_number(parse_number(x)) for x in (a_label, b_label))
    if b < a or (n > 1 and b == a):
        raise ValueError("ratio sample interval must have positive length")
    if n == 1:
        return (a_label,)
    if n == 2:
        return (a_label, b_label)
    nodes = [
        a + (b - a) / 2 if 2 * k == n - 1 else
        a + (b - a) * (1 - mp.cos(mp.pi * k / (n - 1))) / 2
        for k in range(1, n - 1)
    ]
    # Keep exact endpoints as source labels, rather than their approximations.
    return (a_label, *nodes, b_label)


def _sampled_ratio_nodes(channel, *, min_points: int) -> tuple:
    min_points = _integer_control("minimum ratio sample nodes", min_points, minimum=1)
    sample_x = getattr(channel.prefactored, "sample_points_x", None)
    old_nodes = None if sample_x is None else _sampling_point_labels(sample_x, where="ratio sampling")
    if old_nodes is not None and len(old_nodes) >= min_points:
        numeric_nodes = [_evaluation_number(parse_number(x)) for x in old_nodes]
        order = sorted(range(len(old_nodes)), key=numeric_nodes.__getitem__)
        # The exponential and pole prefactors cancel in the F/G_self basis.
        # Use flat-measure Chebyshev nodes for the ratio, retaining the input
        # nodes' support interval and boundary offset.
        return _chebyshev_lobatto_nodes(len(old_nodes), old_nodes[order[0]], old_nodes[order[-1]])

    prefactor = DampedRational(
        constant=mp.mpf(1),
        poles=(),
        base=mp.mpf("0.5"),
    )
    support_nodes = tuple(get_sample_points(prefactor, min_points))
    return _chebyshev_lobatto_nodes(min_points, support_nodes[0], support_nodes[-1])


def _clean_sample_tables_at_nodes(channel, coordinate: str, xs) -> list | None:
    return _saved_sample_tables_at_nodes(channel.prefactored, coordinate, xs)


def _direct_control_table_at(problem: IdenticalScalarGapProblem, channel, x):
    # Tower channels parameterize the block by spin L at fixed twist, not by
    # Delta, so the fixed-spin control re-evaluators do not apply. They carry
    # their own clean sample-block values (attached at build time); force the
    # stored-values fallback by reporting "no direct table" here.
    if getattr(channel, "twist", None) is not None:
        return None
    coordinate = problem.spec.coordinate
    delta = _moment_real_label(channel.delta_min, name="lower edge") + _moment_real_label(x, name="sample offset")
    h_ext = _moment_real_label(problem.spec.h_ext, name="h_ext")

    if problem.resolved_method == "global_control":
        from virasoro_bootstrap.controls.global_control import _GLOBAL_CONTROL_BUILDERS
        from virasoro_blocks.recursion import _resolve_chiral_norm

        builder = _GLOBAL_CONTROL_BUILDERS[coordinate]
        return builder(
            delta,
            channel.ell,
            [h_ext] * 4,
            problem.spec.lam,
            problem.spec.lam,
            chiral_norm=_resolve_chiral_norm(problem.spec.chiral_norm),
            z_shift=-2 * h_ext,
        )

    if problem.resolved_method == "numeric_virasoro_control":
        from virasoro_bootstrap.controls.global_control import _NUMERIC_CONTROL_BUILDERS
        from virasoro_blocks.recursion import _resolve_chiral_norm

        builder = _NUMERIC_CONTROL_BUILDERS[coordinate]
        return builder(
            delta,
            channel.ell,
            [h_ext] * 4,
            problem.spec.lam,
            problem.spec.lam,
            c=problem.spec.c,
            q_order=problem.spec.q_order,
            chiral_norm=_resolve_chiral_norm(problem.spec.chiral_norm),
            z_shift=-2 * h_ext,
        )

    return None


def _normalize_moment_block_basis(block_basis: str) -> str:
    basis = str(block_basis).lower().replace("-", "_")
    if basis in {
        _RATIO_BLOCK_BASIS,
        "f_over_self_dual_measure",
        "f_over_g",
        "f_over_g_self",
    }:
        return _RATIO_BLOCK_BASIS
    if basis in {
        _NONRATIO_BLOCK_BASIS,
        "damped_rational",
        "f_plus_g_self_w",
        "f_with_self_dual_measure",
        "f_plus_self_dual_measure_moments",   # the name the problem metadata records
    }:
        return _NONRATIO_BLOCK_BASIS
    raise ValueError("block_basis must be 'ratio' or 'nonratio'")


def _inner_basis_metadata(problem) -> dict:
    inner = (problem.pmp_metadata or {}).get("moment", {})
    return {key: inner.get(key) for key in ("block_basis", "sample_value_basis", "include_tower")}


def _moment_basis_metadata(block_basis: str, include_tower: bool) -> dict:
    basis = _normalize_moment_block_basis(block_basis)
    # include_tower: whether the tower channels are in the problem (a manifest reload needs it)
    if basis == _RATIO_BLOCK_BASIS:
        return {
            "block_basis": "F_over_self_dual_measure",
            "sample_value_basis": "direct_ratio_at_sdp_nodes",
            "include_tower": bool(include_tower),
        }
    return {
        "block_basis": "F_plus_self_dual_measure_moments",
        "sample_value_basis": "direct_nonratio_at_damped_nodes",
        "include_tower": bool(include_tower),
    }


def _sampled_nonratio_values_by_block(
    problem: IdenticalScalarGapProblem,
    pmp_doc: dict,
    *,
    channels,
    extra_components_by_channel,
    sampling_blocks=None,
) -> dict[int, list]:
    """Clean sample values for ``F`` plus ``G_self * W`` blocks: numeric block values at the nodes times the exact
    weight (never a reduced table's polynomial re-evaluated at the node)."""

    coordinate = problem.spec.coordinate
    blocks = pmp_doc["PositiveMatrixWithPrefactorArray"]
    out: dict[int, list] = {}
    for block_index, channel in enumerate(channels):
        if block_index >= len(blocks):
            break
        block_doc = blocks[block_index]
        if "samplePoints" not in block_doc or "sampleScalings" not in block_doc:
            continue

        # The writer exposes the actual prepared grid alongside its rounded
        # numeric PMP representation. Use that provenance when available;
        # an independently supplied PMP document instead defines its own nodes.
        source_nodes = (None if sampling_blocks is None else
                        sampling_blocks[block_index].sample_points_x)
        xs = _sampling_point_labels(
            block_doc["samplePoints"] if source_nodes is None else source_nodes,
            where="moment PMP sampling",
        )
        scalings = [mp.mpf(s) for s in block_doc["sampleScalings"]]
        direct_tables = [_direct_control_table_at(problem, channel, x) for x in xs]
        if any(table is None for table in direct_tables):
            direct_tables = None
        clean_tables = None
        if direct_tables is None:
            clean_tables = _clean_sample_tables_at_nodes(
                channel,
                coordinate,
                xs,
            )
        if direct_tables is None and clean_tables is None:
            continue

        extra_components = tuple(extra_components_by_channel(channel))
        rows = []
        for k, x in enumerate(xs):
            table = direct_tables[k] if direct_tables is not None else clean_tables[k]
            measure_value = _pmp_real_value(table[0][0], where="moment.nonratio.measure")
            row = [
                _pmp_real_value(table[m][n], where="moment.nonratio.F") / scalings[k]
                for m, n in problem.F_pairs
            ]
            for poly in extra_components:
                row.append(_pmp_real_value(measure_value * poly.evaluate(x),
                    where="moment.nonratio.insertion") / scalings[k])
            rows.append(row)
        out[block_index] = rows
    return out


def _at_moment_block_precision(builder):
    """Moment sampling and fitting run at the blocks' working precision (or the caller's, if higher)."""
    @functools.wraps(builder)
    def wrapped(problem, *, channels, **kwargs):
        channels = tuple(channels)

        @_at_block_precision
        def from_sources(*tables):
            return builder(problem, channels=channels, **kwargs)

        before = ctx.prec
        try:
            return from_sources(*(channel.prefactored for channel in channels))
        finally:
            ctx.prec = before
    return wrapped


@_at_moment_block_precision
def _sampled_nonratio_pmp_doc(
    problem: IdenticalScalarGapProblem,
    *,
    channels,
    extra_components_by_channel,
    objective,
    normalization,
    sig_figs: int | None,
) -> tuple[PrefactoredPmpBuild, dict[int, list]]:
    """Build moment blocks as ``F`` plus ``G_self * W`` in the old PMP basis.

    Each moment component is the reduced self-dual measure times the exact weight ``W`` (degree deg G_00 + deg W,
    within the block's own degree), and its SDP data are the measure's numeric values at the block's nodes times
    ``W`` there.  (Re-reducing ``G_self * W`` on weight-adapted nodes -- the former "method B" -- moved the bounds by
    ~1e-25, far below the npoles truncation, measured against npoles 20-24 references for W = Delta and Delta^3;
    it needed the exact tables, so a problem read from block files could not reproduce it, and its values at the
    nodes came from its monomial numerator.)
    """
    coordinate = problem.spec.coordinate
    extras = []
    for channel in channels:
        measure = _self_dual_measure_poly(problem, channel)
        extras.append(tuple(measure * poly for poly in extra_components_by_channel(channel)))

    pmp_build = build_prefactored_pmp_doc(
        [channel.prefactored for channel in channels],
        list(problem.F_pairs),
        coordinate=coordinate,
        objective=objective,
        normalization=normalization,
        sig_figs=sig_figs,
        positive_pole_npoles=problem.spec.positive_pole_npoles,
        positive_pole_selection=problem.spec.positive_pole_selection,
        explicit_sample_points=problem.spec.explicit_sample_points,
        extra_polynomials_per_block=extras,
    )
    sample_values_by_block = _sampled_nonratio_values_by_block(
        problem,
        pmp_build.doc,
        channels=channels,
        extra_components_by_channel=extra_components_by_channel,
        sampling_blocks=pmp_build.positivity_blocks,
    )
    return pmp_build, sample_values_by_block


def _sampled_moment_pmp_doc(
    problem: IdenticalScalarGapProblem,
    *,
    channels,
    extra_components_by_channel,
    objective,
    normalization,
    sig_figs: int | None,
    block_basis: str,
) -> tuple[PrefactoredPmpBuild, dict[int, list]]:
    basis = _normalize_moment_block_basis(block_basis)
    if basis == _RATIO_BLOCK_BASIS:
        # The ratio basis appends the exact low-degree weight directly.
        return _sampled_self_dual_ratio_pmp_doc(
            problem,
            channels=channels,
            extra_components_by_channel=extra_components_by_channel,
            objective=objective,
            normalization=normalization,
            sig_figs=sig_figs,
        )
    return _sampled_nonratio_pmp_doc(
        problem,
        channels=channels,
        extra_components_by_channel=extra_components_by_channel,
        objective=objective,
        normalization=normalization,
        sig_figs=sig_figs,
    )


@_at_moment_block_precision
def _sampled_self_dual_ratio_pmp_doc(
    problem: IdenticalScalarGapProblem,
    *,
    channels,
    extra_components_by_channel,
    objective,
    normalization,
    sig_figs: int | None,
) -> tuple[PrefactoredPmpBuild, dict[int, list]]:
    """Build moment blocks in the paper's sampled ``F/G_self`` basis.

    The moment duals in arXiv:2603.18140 use
    ``N_l(Delta)=F_l(Delta)/G_self,l(Delta)`` and moment insertions ``V_l``
    directly.  Encoding it as ``F_l`` plus ``G_self,l V_l`` is algebraically
    equivalent for exact blocks, but a polynomial interpolant for ``G_self`` can
    develop negative tails and destroy moment positivity.  Interpolate the ratio
    itself at the block's existing sample nodes (or no-pole ratio nodes if the
    block has none), then append the low-degree moment insertions directly.
    """

    sig_figs = _resolve_sig_figs(sig_figs)
    blocks = []
    clean_values_by_block: dict[int, list] = {}
    for block_index, channel in enumerate(channels):
        rows = channel.prefactored.numerators[problem.spec.coordinate]
        measure = rows[0][0]
        extra_components = tuple(extra_components_by_channel(channel))
        max_extra_degree = max(
            (poly.degree for poly in extra_components),
            default=0,
        )
        max_row_degree = max(
            rows[m][n].degree
            for m, n in problem.F_pairs
        )
        min_points = max(max_extra_degree, max_row_degree) + 1
        xs = _sampled_ratio_nodes(channel, min_points=min_points)
        sample_point_texts = [
            _pmp_real_str(x, sig_figs, where="normalized_moment.ratio_node")
            for x in xs
        ]
        # An explicit text-precision cap can merge otherwise distinct nodes.
        _sampling_point_labels(sample_point_texts, where="normalized moment ratio export")
        clean_tables = _clean_sample_tables_at_nodes(
            channel,
            problem.spec.coordinate,
            xs,
        )
        direct_tables = [
            _direct_control_table_at(problem, channel, x)
            for x in xs
        ]
        if any(table is None for table in direct_tables):
            # Use the direct control tables only when EVERY node has one; a
            # partially-None list would later index None[0][0]. This matches the
            # stricter all-or-nothing convention in
            # `_sampled_nonratio_values_by_block`. Behavior-preserving today,
            # since `_direct_control_table_at` is None per-channel-constant.
            direct_tables = None

        components = []
        component_values = []
        sampled_tables = direct_tables if direct_tables is not None else clean_tables
        measure_values = [
            _pmp_real_value(sampled_tables[k][0][0] if sampled_tables is not None else measure.evaluate(x),
                where="moment.ratio.measure")
            for k, x in enumerate(xs)
        ]
        if any(denom <= 0 for denom in measure_values):
            raise ValueError(
                "self-dual measure must be positive at all sampled-ratio "
                "ratio interpolation nodes"
            )
        for m, n in problem.F_pairs:
            values = [
                _pmp_real_value(sampled_tables[k][m][n] if sampled_tables is not None else rows[m][n].evaluate(x),
                    where="moment.ratio.F") / measure_values[k]
                for k, x in enumerate(xs)
            ]
            components.append(lagrange_interpolation_polynomial(xs, values, shift=0))
            component_values.append(values)

        components.extend(extra_components)
        for poly in extra_components:
            component_values.append([_pmp_real_value(poly.evaluate(x), where="moment.ratio.insertion") for x in xs])
        clean_values_by_block[block_index] = [
            [component_values[a][k] for a in range(len(component_values))]
            for k in range(len(xs))
        ]

        blocks.append(
            {
                "DampedRational": {
                    "base": "0.5",
                    "constant": "1",
                    "poles": [],
                },
                "polynomials": [[[_pmp_poly_coeffs(p, sig_figs=sig_figs, n_sample_points=len(xs))
                                  for p in components]]],
                "samplePoints": sample_point_texts,
                "sampleScalings": ["1" for _ in xs],
            }
        )

    doc = {
        "objective": [
            _pmp_real_str(v, sig_figs, where="normalized_moment.objective")
            for v in objective
        ],
        "normalization": [
            _pmp_real_str(v, sig_figs, where="normalized_moment.normalization")
            for v in normalization
        ],
        "PositiveMatrixWithPrefactorArray": blocks,
    }
    return (
        PrefactoredPmpBuild(
            doc=doc,
            positivity_blocks=tuple(channel.prefactored for channel in channels),
            positive_pole_lowerings=tuple(None for _ in channels),
        ),
        clean_values_by_block,
    )


def _channel_weight_applies(weight: MomentWeight, channel) -> bool:
    if weight.spin_filter is not None:
        spin = int(weight.spin_filter)
        if getattr(channel, "twist", None) is not None:
            # a tower slice is continuous in L >= its floor: it cannot isolate one spin, and its label ell = int(Lmax)
            # is not a spin (matching it put the whole tower into a spin filter at Lmax, and none at Lmax + 2)
            if spin >= _as_real_label(channel.delta_min, name="tower floor"):
                raise ValueError(
                    f"spin_filter = {spin} lies inside the large-spin tower (L >= {mp.nstr(channel.delta_min, 6)}), "
                    "which cannot isolate a single spin: build the moment with include_tower=False, or with "
                    "large_spin_lmin above the filtered spin"
                )
            return False
        if channel.ell != spin:
            return False
    return True


def _channel_weight_poly(weight: MomentWeight, channel) -> Polynomial:
    if not _channel_weight_applies(weight, channel):
        return _poly_const(0)
    return moment_weight_polynomial(weight, channel)


def _channel_ratio_probe_poly(numerator, denominator, r, channel):
    # Filters belong to the two moments independently. Apply them before
    # combining terms, rather than requiring both moments to use one filter.
    terms = numerator.terms if _channel_weight_applies(numerator, channel) else ()
    if _channel_weight_applies(denominator, channel):
        terms += denominator.scale(-_moment_real_label(r, name="r")).terms
    return moment_weight_polynomial(MomentWeight(terms), channel)


def _channel_weight_value(weight: MomentWeight, channel, value):
    """A physical moment weight evaluated before quantum-number roundoff.

    Fixed-spin queries supply Delta; a tower query supplies L at fixed twist.
    Exact arithmetic retains small hbar and Delta - L cancellations and the
    same spin-filter convention as the inserted SDP polynomial.
    """
    if not _channel_weight_applies(weight, channel):
        return literal_number(0)
    value = _moment_real_label(value, name="moment query")
    twist = getattr(channel, "twist", None)
    if twist is None:
        delta, spin = value, _moment_real_label(channel.ell, name="spin")
    else:
        spin = value
        delta = spin + _moment_real_label(twist, name="twist")
    variables = {"Delta": delta, "L": spin, "h": (delta+spin)/2, "hbar": (delta-spin)/2}
    result = literal_number(0)
    for term in weight.terms:
        monomial = _moment_real_label(term.coefficient, name="MomentTerm coefficient")
        for name, power in _validated_moment_powers(term.powers).items():
            monomial *= variables[name]**power
        result += monomial
    return result


@_at_spec_dps
def build_identical_scalar_normalized_moment_lower_problem(
    spec: IdenticalScalarGapSpec,
    numerator: MomentWeight,
    denominator: MomentWeight,
    *,
    alpha=0,
    beta=0,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
) -> IdenticalScalarGapProblem:
    """Build the direct normalized lower-bound dual.

    This is the PRODUCTION path: a single SDP, no bisection.  ``r`` (here
    ``y = -r``) enters the positivity constraint linearly, multiplying the known
    weight ``V0``, so it is just another dual variable and SDPB returns the
    bound in one solve.  The published moment grids were produced this way.

    The SDPB objective maximizes ``-y``, so ``dualObjective`` is the ratio lower
    bound directly (:func:`normalized_moment_lower_bound_from_objective` is the
    identity).  Signs were validated against
    :func:`bound_identical_scalar_normalized_moment_lower_by_bisection`, which
    remains available as an independent cross-check rather than as the
    preferred path.

    The ratio's denominator ``M[denominator] + alpha`` (the identity's share included) must be strictly positive
    on every admissible spectrum: the linear-fractional formulation assumes it, and a negative one makes the
    program bound something else.  A denominator that is provably never positive is refused; for any other signed
    weight or ``alpha`` the positivity is the caller's to ensure.
    """

    gap_problem = _resolve_base_problem(spec, prepared, base_problem)
    channels = _moment_channels(gap_problem, include_tower=include_tower, weights=(numerator, denominator))
    alpha_eff, beta_eff, identity_numerator, identity_denominator = (
        _effective_ratio_constants(
            gap_problem,
            numerator,
            denominator,
            alpha=alpha,
            beta=beta,
            identity_mode=identity_mode,
        )
    )
    n = len(gap_problem.F_pairs)
    # Dual var order (z_0..z_{n-1}, s, y); s is fixed to 1 by normalization.
    # The paper maximizes -y; SDPB maximizes objective.dual, so the objective
    # picks -y (y-slot = -1) and dualObjective = max(-y) is the ratio lower
    # bound directly (normalized_moment_lower_bound_from_objective is identity).
    objective = [mp.mpf(0)] * n + [mp.mpf(0), mp.mpf(-1)]
    normalization = [mp.mpf(0)] * n + [mp.mpf(1), mp.mpf(0)]
    pmp_build, sample_values_by_block = _sampled_moment_pmp_doc(
        gap_problem,
        channels=channels,
        extra_components_by_channel=(
            lambda channel: (
                _channel_weight_poly(numerator, channel),
                _channel_weight_poly(denominator, channel),
            )
        ),
        objective=objective,
        normalization=normalization,
        sig_figs=None,
        block_basis=block_basis,
    )
    # Scalar inequality  beta + alpha*y + b.z >= 0  (paper eq.; the paper's
    # vacuum b is -identity_vector in this repo's sum-rule convention, so the
    # paper's "-b.z" becomes "+b.z" here). With V0=0, alpha=1, beta=0 this
    # collapses the normalized dual onto the verified unnormalized lower bound.
    # Dual variable order is (z_0..z_{n-1}, s, y): s is fixed to 1 by the
    # normalization vector and carries beta_eff; y carries alpha_eff.
    scalar_vector = [mp.mpf(v) for v in gap_problem.identity_vector] + [
        beta_eff,
        alpha_eff,
    ]
    scalar_block = _constant_scalar_block(
        scalar_vector,
        sig_figs=None,
    )
    doc = dict(pmp_build.doc)
    doc["PositiveMatrixWithPrefactorArray"] = (
        list(doc["PositiveMatrixWithPrefactorArray"]) + [scalar_block]
    )
    pmp_build = PrefactoredPmpBuild(
        doc=doc,
        positivity_blocks=pmp_build.positivity_blocks,
        positive_pole_lowerings=pmp_build.positive_pole_lowerings,
    )
    basis_meta = _moment_basis_metadata(block_basis, include_tower)
    return _dataclass_replace(
        gap_problem,
        spin_channels=channels,
        pmp_build=pmp_build,
        sample_values_by_block=sample_values_by_block,
        problem_kind="normalized_moment_lower",
        objective_vector=tuple(objective),
        normalization_vector=tuple(normalization),
        pmp_metadata={
            **(gap_problem.pmp_metadata or {}),
            "moment": {
                "kind": "normalized",
                "direction": "lower",
                "numerator": numerator.to_json(),
                "denominator": denominator.to_json(),
                "alpha": _moment_input_text(alpha, name="alpha"),
                "beta": _moment_input_text(beta, name="beta"),
                "alpha_effective": mp.nstr(alpha_eff, mp.mp.dps + 5),
                "beta_effective": mp.nstr(beta_eff, mp.mp.dps + 5),
                "identity_mode": identity_mode,
                "identity_numerator": mp.nstr(identity_numerator, mp.mp.dps + 5),
                "identity_denominator": mp.nstr(
                    identity_denominator, mp.mp.dps + 5
                ),
                **basis_meta,
            }
        },
    )


@_at_spec_dps
def build_identical_scalar_normalized_moment_upper_problem(
    spec: IdenticalScalarGapSpec,
    numerator: MomentWeight,
    denominator: MomentWeight,
    *,
    alpha=0,
    beta=0,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
) -> IdenticalScalarGapProblem:
    """Build the direct normalized upper-bound problem.

    This is the PRODUCTION path: a single SDP, no bisection.  See
    :func:`build_identical_scalar_normalized_moment_lower_problem` for the
    encoding.

    Implemented as a lower bound on ``-R``:

        -R = (M[-V1] - beta) / (M[V0] + alpha).

    The returned SDPB objective is converted with
    :func:`normalized_moment_upper_bound_from_objective`, which negates the
    minimized objective back to an upper bound on ``R``.
    :func:`bound_identical_scalar_normalized_moment_by_bisection` remains
    available as an independent cross-check.
    """

    problem = build_identical_scalar_normalized_moment_lower_problem(
        spec,
        -numerator,
        denominator,
        alpha=alpha,
        beta=-_moment_real_label(beta, name="beta"),
        identity_mode=identity_mode,
        prepared=prepared,
        base_problem=base_problem,
        block_basis=block_basis,
        include_tower=include_tower,
    )
    # Compute the reported effective constants from the base problem, including
    # any explicit self-dual identity measure supplied by a control builder.
    constants_problem = _resolve_base_problem(spec, prepared, base_problem)
    alpha_eff, beta_eff, identity_numerator, identity_denominator = (
        _effective_ratio_constants(
            constants_problem,
            numerator,
            denominator,
            alpha=alpha,
            beta=beta,
            identity_mode=identity_mode,
        )
    )
    return _dataclass_replace(
        problem,
        problem_kind="normalized_moment_upper",
        pmp_metadata={
            **(problem.pmp_metadata or {}),
            "moment": {
                "kind": "normalized",
                "direction": "upper",
                "numerator": numerator.to_json(),
                "denominator": denominator.to_json(),
                "alpha": _moment_input_text(alpha, name="alpha"),
                "beta": _moment_input_text(beta, name="beta"),
                "alpha_effective": mp.nstr(alpha_eff, mp.mp.dps + 5),
                "beta_effective": mp.nstr(beta_eff, mp.mp.dps + 5),
                "identity_mode": identity_mode,
                "identity_numerator": mp.nstr(identity_numerator, mp.mp.dps + 5),
                "identity_denominator": mp.nstr(
                    identity_denominator, mp.mp.dps + 5
                ),
                # the inner build's basis record, include_tower with it (a manifest reload and a certification
                # rebuild need it)
                **_inner_basis_metadata(problem),
            }
        },
    )


_NORMALIZED_MOMENT_BUILDERS = {
    "lower": build_identical_scalar_normalized_moment_lower_problem,
    "upper": build_identical_scalar_normalized_moment_upper_problem,
}


@_at_spec_dps
def rebuild_normalized_moment_problem(
    problem: IdenticalScalarGapProblem,
    *,
    spin_list=None,
    n_twists=None,
    twists=None,
) -> IdenticalScalarGapProblem:
    """Rebuild a solved normalized-moment problem with a wider positivity set.

    Reconstructs the moment weights / direction / identity_mode / alpha / beta
    from ``problem.pmp_metadata['moment']`` and rebuilds with an extended
    ``spin_list`` and/or a different large-spin tower -- either the first
    ``n_twists`` canonical double-twist slices or an explicit ``twists`` list
    (which can retain custom slices and place lines BETWEEN them). A count
    replaces the slice list; certification supplies an explicit union when
    extending a custom or gap-selected tower. The dual dimension is spin-independent, so a dual vector ``y`` from
    the original solve applies unchanged to every block of the rebuilt problem
    -- this is what lets certification evaluate the *solved* functional on spins
    and twists that were never in the SDP. Only the block set changes; the
    moment insertion (``G_self * W``) is rebuilt identically per channel.
    """
    import dataclasses
    from virasoro_blocks.requests import _integer_field
    from .identical_scalar import _validate_even_spins
    from .large_spin import double_twist_set

    meta = (problem.pmp_metadata or {}).get("moment")
    if meta is None or meta.get("kind") != "normalized":
        raise ValueError(
            "rebuild_normalized_moment_problem requires a normalized-moment "
            f"problem; got metadata {meta!r}"
        )
    if getattr(problem, "resolved_method", None) in ("global_control", "numeric_virasoro_control"):
        # the builders below are the Virasoro interpolation path: a global-control solve would be certified on
        # Virasoro blocks with the Virasoro vacuum constants (alpha_eff 2.63902 -> 2.63875 in a check)
        raise NotImplementedError(
            f"rebuilding a {problem.resolved_method} moment problem is not supported: its blocks are not the "
            "Virasoro interpolation surface the rebuild uses"
        )
    direction = meta["direction"]
    builder = _NORMALIZED_MOMENT_BUILDERS.get(direction)
    if builder is None:
        raise ValueError(f"unknown normalized-moment direction {direction!r}")

    numerator = MomentWeight.from_json(meta["numerator"])
    denominator = MomentWeight.from_json(meta["denominator"])
    alpha = _moment_real_label(meta.get("alpha", "0"), name="alpha")
    beta = _moment_real_label(meta.get("beta", "0"), name="beta")
    identity_mode = meta.get("identity_mode", "include")

    spec = problem.spec
    replace_kwargs = {}
    if spin_list is not None:
        replace_kwargs["spin_list"] = _validate_even_spins(spin_list)
    has_tower = getattr(spec, "large_spin_lmin", None) is not None
    if twists is not None and has_tower:
        # An explicit twist list can retain custom/gap-selected slices. A
        # canonical double-twist count cannot express
        # twists BETWEEN the slices -- and unitarity permits operators at any
        # tau >= 0 at large spin, so those interstitial lines are exactly where
        # an optimized functional is free to dip negative.
        replace_kwargs["large_spin_twists"] = tuple(_as_real_label(t, name="twists entry") for t in twists)
    elif n_twists is not None and has_tower:
        replace_kwargs["large_spin_twists"] = tuple(
            double_twist_set(spec.h_ext, _integer_field("n_twists", n_twists))
        )
    new_spec = dataclasses.replace(spec, **replace_kwargs) if replace_kwargs else spec

    # Positivity blocks depend only on the spec. Reuse them across bound
    # directions and identity modes; moment weights enter through the
    # per-channel insertion assembled above.
    return builder(
        new_spec, numerator, denominator,
        alpha=alpha, beta=beta, identity_mode=identity_mode,
        block_basis=meta.get("block_basis") or _NONRATIO_BLOCK_BASIS,
        include_tower=bool(meta.get("include_tower", True)),
        base_problem=_cached_base_problem(new_spec),
    )


#: Prepared positivity problems keyed by spec, for reuse across the several
#: certification rebuilds of one solve. Small because a task touches one or two
#: specs; cleared wholesale rather than evicted one at a time since the entries
#: are large and a task never revisits an old spec.
_BASE_PROBLEM_CACHE: dict = {}
_BASE_PROBLEM_CACHE_MAXSIZE = 4


def _spec_cache_value(value):
    """Hashable snapshot that retains numeric input types and nested values."""
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return (type(value), frozenset(
            (_spec_cache_value(k), _spec_cache_value(v)) for k, v in value.items()
        ))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_spec_cache_value(v) for v in value))
    return (type(value), value)


def _cached_base_problem(spec):
    """Memoize builds without hashing mutable fields or conflating input types."""
    from copy import deepcopy
    from dataclasses import fields

    from virasoro_blocks._conventions import scalar_G_spin0_convention

    snapshot = deepcopy(spec)
    # the spin-0 blocks and identity vector depend on the process's scalar_G convention too
    key = (scalar_G_spin0_convention(),) + tuple(_spec_cache_value(getattr(snapshot, f.name)) for f in fields(snapshot))
    hit = _BASE_PROBLEM_CACHE.get(key)
    if hit is not None:
        return hit
    built = build_identical_scalar_gap_problem(snapshot)
    if len(_BASE_PROBLEM_CACHE) >= _BASE_PROBLEM_CACHE_MAXSIZE:
        _BASE_PROBLEM_CACHE.clear()
    _BASE_PROBLEM_CACHE[key] = built
    return built


@_at_spec_dps
def build_identical_scalar_normalized_ratio_probe_problem(
    spec: IdenticalScalarGapSpec,
    numerator: MomentWeight,
    denominator: MomentWeight,
    r,
    *,
    alpha=0,
    beta=0,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
) -> IdenticalScalarGapProblem:
    """Build the signed unnormalized probe for a candidate ratio bound.

    For ``R = (M[V1] + beta_eff) / (M[V0] + alpha_eff)``, where the effective
    constants include the requested identity-mode shifts, the operator-dependent
    part of the upper-bound probe is ``M[V1 - r V0]``.  The caller compares the
    resulting upper bound to ``r * alpha_eff - beta_eff``.  The denominator must be
    strictly positive, as for :func:`build_identical_scalar_normalized_moment_lower_problem`.
    """

    return _build_normalized_ratio_probe_problem(
        spec, numerator, denominator, r, direction="upper", alpha=alpha, beta=beta,
        identity_mode=identity_mode, prepared=prepared, base_problem=base_problem,
        block_basis=block_basis, include_tower=include_tower,
    )


def _build_normalized_ratio_probe_problem(
    spec: IdenticalScalarGapSpec,
    numerator: MomentWeight,
    denominator: MomentWeight,
    r,
    *,
    direction,
    alpha=0,
    beta=0,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
) -> IdenticalScalarGapProblem:
    """Assemble either signed probe with one shared insertion and metadata path."""

    gap_problem = _resolve_base_problem(spec, prepared, base_problem)
    channels = _moment_channels(gap_problem, include_tower=include_tower, weights=(numerator, denominator))
    alpha_eff, beta_eff, identity_numerator, identity_denominator = (
        _effective_ratio_constants(
            gap_problem,
            numerator,
            denominator,
            alpha=alpha,
            beta=beta,
            identity_mode=identity_mode,
        )
    )
    n = len(gap_problem.F_pairs)
    # Both probes have objective +b and no identity insertion. The upper
    # probe appends -G*(V1-rV0), yielding -dualObjective as its bound; the
    # lower probe appends +G*(V1-rV0), yielding dualObjective. Identity terms
    # enter the comparison threshold instead.
    objective = [mp.mpf(v) for v in gap_problem.identity_vector] + [mp.mpf(0)]
    normalization = [mp.mpf(0)] * n + [mp.mpf(1)]
    r = _moment_real_label(r, name="r")
    pmp_build, sample_values_by_block = _sampled_moment_pmp_doc(
        gap_problem,
        channels=channels,
        extra_components_by_channel=(
            lambda channel: (
                _channel_ratio_probe_poly(numerator, denominator, r, channel) * (-1 if direction == "upper" else 1),
            )
        ),
        objective=objective,
        normalization=normalization,
        sig_figs=None,
        block_basis=block_basis,
    )
    basis_meta = _moment_basis_metadata(block_basis, include_tower)
    return _dataclass_replace(
        gap_problem,
        spin_channels=channels,
        pmp_build=pmp_build,
        sample_values_by_block=sample_values_by_block,
        problem_kind=("normalized_ratio_probe" if direction == "upper" else "normalized_ratio_lower_probe"),
        objective_vector=tuple(objective),
        normalization_vector=tuple(normalization),
        pmp_metadata={
            **(gap_problem.pmp_metadata or {}),
            "moment": {
                "kind": "normalized_ratio_probe",
                **({"direction": "lower"} if direction == "lower" else {}),
                **({"operator_weight": numerator.minus(denominator.scale(r)).to_json()}
                   if direction == "lower" and numerator.spin_filter == denominator.spin_filter else {}),
                "candidate": _moment_input_text(r, name="r"),
                "numerator": numerator.to_json(),
                "denominator": denominator.to_json(),
                "alpha": _moment_input_text(alpha, name="alpha"),
                "beta": _moment_input_text(beta, name="beta"),
                "alpha_effective": mp.nstr(alpha_eff, mp.mp.dps + 5),
                "beta_effective": mp.nstr(beta_eff, mp.mp.dps + 5),
                "identity_mode": identity_mode,
                "identity_numerator": mp.nstr(identity_numerator, mp.mp.dps + 5),
                "identity_denominator": mp.nstr(
                    identity_denominator, mp.mp.dps + 5
                ),
                **basis_meta,
            }
        },
    )


@_at_spec_dps
def build_identical_scalar_normalized_ratio_lower_probe_problem(
    spec: IdenticalScalarGapSpec,
    numerator: MomentWeight,
    denominator: MomentWeight,
    r,
    *,
    alpha=0,
    beta=0,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
) -> IdenticalScalarGapProblem:
    """Build the unnormalized lower probe for ``R >= r``.

    For ``R = (M[V1] + beta_eff) / (M[V0] + alpha_eff)``, a candidate lower
    bound ``r`` is certified when a lower bound on ``M[V1 - r V0]`` is at least
    ``r * alpha_eff - beta_eff``.  The probe itself always bounds only the
    non-identity/operator contribution; identity terms are carried entirely by
    the effective constants in the comparison threshold.  The denominator must be
    strictly positive, as for :func:`build_identical_scalar_normalized_moment_lower_problem`.
    """

    return _build_normalized_ratio_probe_problem(
        spec, numerator, denominator, r, direction="lower", alpha=alpha, beta=beta,
        identity_mode=identity_mode, prepared=prepared, base_problem=base_problem,
        block_basis=block_basis, include_tower=include_tower,
    )


def moment_upper_bound_from_objective(objective_value) -> mp.mpf:
    # SDPB maximizes; upper builder has objective [+b, -I[W]], so dualObjective
    # = max(z.b) - I[W] and the upper bound on M_all is -dualObjective.
    value = objective_value if isinstance(objective_value, mp.mpf) else mp.mpf(_convert_number(objective_value))
    return mp.fneg(value, exact=True)


def moment_lower_bound_from_objective(objective_value) -> mp.mpf:
    # Lower builder has objective [+b, +I[W]], so dualObjective = max(z.b) + I[W]
    # is the lower bound on M_all directly.
    return objective_value if isinstance(objective_value, mp.mpf) else mp.mpf(_convert_number(objective_value))


def normalized_moment_lower_bound_from_objective(objective_value) -> mp.mpf:
    # Direct lower dual: objective picks -y, so dualObjective = max(-y) is the
    # ratio lower bound directly (SDPB maximizes objective.dual).
    return objective_value if isinstance(objective_value, mp.mpf) else mp.mpf(_convert_number(objective_value))


def normalized_moment_upper_bound_from_objective(objective_value) -> mp.mpf:
    # The direct upper formulation is encoded as a lower bound on -R:
    # lower(-R) = dualObjective, and upper(R) = -lower(-R).
    value = objective_value if isinstance(objective_value, mp.mpf) else mp.mpf(_convert_number(objective_value))
    return mp.fneg(value, exact=True)


def normalized_ratio_probe_threshold(r, *, alpha=0, beta=0) -> mp.mpf:
    """Right-hand side for certifying ``R <= r`` with a ratio probe."""

    return mp.convert(_moment_real_label(r, name="r") * _moment_real_label(alpha, name="alpha")
                      - _moment_real_label(beta, name="beta"))


@_at_spec_dps
def bound_identical_scalar_normalized_moment_by_bisection(
    spec: IdenticalScalarGapSpec,
    numerator: MomentWeight,
    denominator: MomentWeight,
    *,
    lo,
    hi,
    tol,
    solve_probe_upper_bound: Callable[[IdenticalScalarGapProblem, mp.mpf], object],
    alpha=0,
    beta=0,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
    max_iterations: int = 100,
) -> dict:
    """Certify an upper bound on a normalized moment by ratio-probe bisection.

    Cross-check path, NOT the production one: the shipped bounds come from the
    single-SDP :func:`build_identical_scalar_normalized_moment_upper_problem`.
    This routine costs one solve per bisection step and is kept because it
    reaches the same bound through unnormalized probes only, which is what
    pinned down the direct dual's signs.

    ``solve_probe_upper_bound(problem, r)`` must solve the returned ratio-probe
    problem and return an upper bound on ``M[V1 - r V0]``.  A candidate ``r`` is
    certified when that upper bound is at most ``r * alpha - beta``.
    """

    lo = _as_real_mpf(lo, name="lo")
    hi = _as_real_mpf(hi, name="hi")
    tol = _as_real_mpf(tol, name="tol")
    if not lo < hi:
        raise ValueError("lo must be less than hi")
    if tol <= 0:
        raise ValueError("tol must be positive")

    constants_problem = _resolve_base_problem(spec, prepared, base_problem)
    alpha_eff, beta_eff, identity_numerator, identity_denominator = (
        _effective_ratio_constants(
            constants_problem,
            numerator,
            denominator,
            alpha=alpha,
            beta=beta,
            identity_mode=identity_mode,
        )
    )
    probes = []

    def probe(r):
        problem = build_identical_scalar_normalized_ratio_probe_problem(
            spec,
            numerator,
            denominator,
            r,
            alpha=alpha,
            beta=beta,
            identity_mode=identity_mode,
            prepared=prepared,
            base_problem=constants_problem,
            block_basis=block_basis,
            include_tower=include_tower,
        )
        bound = mp.mpf(solve_probe_upper_bound(problem, r))
        if mp.isnan(bound):
            raise ValueError("moment bisection solver returned NaN")
        threshold = normalized_ratio_probe_threshold(
            r, alpha=alpha_eff, beta=beta_eff
        )
        certified = bound <= threshold
        probes.append(
            {
                "r": mp.nstr(r, mp.mp.dps + 5),
                "upper_bound": mp.nstr(bound, mp.mp.dps + 5),
                "threshold": mp.nstr(threshold, mp.mp.dps + 5),
                "certified_upper": certified,
            }
        )
        return certified

    if probe(lo):
        raise ValueError("lo is already certified as an upper bound")
    if not probe(hi):
        raise ValueError("hi is not certified as an upper bound")

    iterations = 0
    while hi - lo > tol:
        if iterations >= max_iterations:
            raise RuntimeError("normalized moment bisection exceeded max_iterations")
        mid = (lo + hi) / 2
        if not lo < mid < hi:
            raise RuntimeError("moment bisection cannot resolve the requested tolerance; increase spec.dps")
        if probe(mid):
            hi = mid
        else:
            lo = mid
        iterations += 1

    return {
        "lower_uncertified": mp.nstr(lo, mp.mp.dps + 5),
        "upper_certified": mp.nstr(hi, mp.mp.dps + 5),
        "tolerance": mp.nstr(tol, mp.mp.dps + 5),
        "iterations": iterations,
        "alpha": _moment_input_text(alpha, name="alpha"),
        "beta": _moment_input_text(beta, name="beta"),
        "numerator": numerator.to_json(),
        "denominator": denominator.to_json(),
        "alpha_effective": mp.nstr(alpha_eff, mp.mp.dps + 5),
        "beta_effective": mp.nstr(beta_eff, mp.mp.dps + 5),
        "identity_mode": identity_mode,
        "identity_numerator": mp.nstr(identity_numerator, mp.mp.dps + 5),
        "identity_denominator": mp.nstr(identity_denominator, mp.mp.dps + 5),
        **_moment_basis_metadata(block_basis, include_tower),
        "probes": probes,
    }


@_at_spec_dps
def bound_identical_scalar_normalized_moment_lower_by_bisection(
    spec: IdenticalScalarGapSpec,
    numerator: MomentWeight,
    denominator: MomentWeight,
    *,
    lo,
    hi,
    tol,
    solve_probe_lower_bound: Callable[[IdenticalScalarGapProblem, mp.mpf], object],
    alpha=0,
    beta=0,
    identity_mode: str = "include",
    prepared=None,
    base_problem=None,
    block_basis: str = _NONRATIO_BLOCK_BASIS,
    include_tower: bool = True,
    max_iterations: int = 100,
) -> dict:
    """Certify a lower bound on a normalized moment by ratio-probe bisection.

    Cross-check path, NOT the production one; see
    :func:`bound_identical_scalar_normalized_moment_by_bisection`.
    """

    lo = _as_real_mpf(lo, name="lo")
    hi = _as_real_mpf(hi, name="hi")
    tol = _as_real_mpf(tol, name="tol")
    if not lo < hi:
        raise ValueError("lo must be less than hi")
    if tol <= 0:
        raise ValueError("tol must be positive")

    constants_problem = _resolve_base_problem(spec, prepared, base_problem)
    alpha_eff, beta_eff, identity_numerator, identity_denominator = (
        _effective_ratio_constants(
            constants_problem,
            numerator,
            denominator,
            alpha=alpha,
            beta=beta,
            identity_mode=identity_mode,
        )
    )
    probes = []

    def probe(r):
        problem = build_identical_scalar_normalized_ratio_lower_probe_problem(
            spec,
            numerator,
            denominator,
            r,
            alpha=alpha,
            beta=beta,
            identity_mode=identity_mode,
            prepared=prepared,
            base_problem=constants_problem,
            block_basis=block_basis,
            include_tower=include_tower,
        )
        bound = mp.mpf(solve_probe_lower_bound(problem, r))
        if mp.isnan(bound):
            raise ValueError("moment bisection solver returned NaN")
        threshold = normalized_ratio_probe_threshold(
            r, alpha=alpha_eff, beta=beta_eff
        )
        certified = bound >= threshold
        probes.append(
            {
                "r": mp.nstr(r, mp.mp.dps + 5),
                "lower_bound": mp.nstr(bound, mp.mp.dps + 5),
                "threshold": mp.nstr(threshold, mp.mp.dps + 5),
                "certified_lower": certified,
            }
        )
        return certified

    if not probe(lo):
        raise ValueError("lo is not certified as a lower bound")
    if probe(hi):
        raise ValueError("hi is already certified as a lower bound")

    iterations = 0
    while hi - lo > tol:
        if iterations >= max_iterations:
            raise RuntimeError(
                "normalized moment lower bisection exceeded max_iterations"
            )
        mid = (lo + hi) / 2
        if not lo < mid < hi:
            raise RuntimeError("moment bisection cannot resolve the requested tolerance; increase spec.dps")
        if probe(mid):
            lo = mid
        else:
            hi = mid
        iterations += 1

    return {
        "lower_certified": mp.nstr(lo, mp.mp.dps + 5),
        "upper_uncertified": mp.nstr(hi, mp.mp.dps + 5),
        "tolerance": mp.nstr(tol, mp.mp.dps + 5),
        "iterations": iterations,
        "alpha": _moment_input_text(alpha, name="alpha"),
        "beta": _moment_input_text(beta, name="beta"),
        "numerator": numerator.to_json(),
        "denominator": denominator.to_json(),
        "alpha_effective": mp.nstr(alpha_eff, mp.mp.dps + 5),
        "beta_effective": mp.nstr(beta_eff, mp.mp.dps + 5),
        "identity_mode": identity_mode,
        "identity_numerator": mp.nstr(identity_numerator, mp.mp.dps + 5),
        "identity_denominator": mp.nstr(identity_denominator, mp.mp.dps + 5),
        **_moment_basis_metadata(block_basis, include_tower),
        "probes": probes,
    }


def rebuild_moment_problem_from_blocks(spec, meta: Mapping, problem_kind: str, blocks, *, base_problem=None,
                                     _block_reader=None):
    """The moment problem ``problem_kind`` described by ``meta`` (a problem's ``pmp_metadata['moment']``), its
    positivity blocks read from ``blocks`` (:func:`prepare_identical_scalar_gap_build_from_blocks`).

    The moment insertion is rebuilt per channel exactly as the builder assembled it; the tables come from the files.
    """
    from .identical_scalar import prepare_identical_scalar_gap_build_from_blocks

    numerator = MomentWeight.from_json(meta["numerator"])
    common = dict(
        identity_mode=meta.get("identity_mode", "include"),
        block_basis=meta.get("block_basis", _NONRATIO_BLOCK_BASIS),
        include_tower=bool(meta.get("include_tower", True)),
    )
    if not common["include_tower"] and spec.large_spin_lmin is not None:
        # the tower was left out of the problem (and so out of its block files)
        spec = _dataclass_replace(spec, large_spin_lmin=None, large_spin_twists=())
    with mp.workdps(spec.dps):
        if base_problem is None:
            common["prepared"] = prepare_identical_scalar_gap_build_from_blocks(spec, blocks,
                                                                              _block_reader=_block_reader)
        else:
            common["base_problem"] = base_problem
        if problem_kind in ("moment_upper", "moment_lower"):
            builder = (build_identical_scalar_moment_upper_problem if problem_kind == "moment_upper"
                       else build_identical_scalar_moment_lower_problem)
            return builder(spec, numerator, **common)
        denominator = MomentWeight.from_json(meta["denominator"])
        ratio = dict(alpha=_moment_real_label(meta.get("alpha", "0"), name="alpha"),
                     beta=_moment_real_label(meta.get("beta", "0"), name="beta"))
        if problem_kind in ("normalized_moment_lower", "normalized_moment_upper"):
            builder = _NORMALIZED_MOMENT_BUILDERS[problem_kind.rsplit("_", 1)[1]]
            return builder(spec, numerator, denominator, **ratio, **common)
        if problem_kind in ("normalized_ratio_probe", "normalized_ratio_lower_probe"):
            builder = (build_identical_scalar_normalized_ratio_probe_problem if problem_kind == "normalized_ratio_probe"
                       else build_identical_scalar_normalized_ratio_lower_probe_problem)
            return builder(spec, numerator, denominator, _moment_real_label(meta["candidate"], name="r"), **ratio, **common)
    raise ValueError(f"unknown moment problem kind {problem_kind!r}")
