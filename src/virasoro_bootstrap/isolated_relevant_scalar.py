"""Builders for the isolated relevant scalar Z + four-point experiment.

This module implements convex constraints that couple torus one-point and
sphere four-point data:

* modular crossing with a fixed scalar primary at ``Delta_phi`` and continuum
  gap ``Delta_gap``;
* identical-scalar four-point crossing with the same ``phi`` allowed as an
  isolated exchanged scalar and the rest of ``phi x phi`` gapped.

The exact common-heavy-support test is not a single convex SDP in the current
variables, because modular degeneracies and four-point OPE-squared weights
share locations but not coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from virasoro_blocks.numeric_input import literal_number

from .identical_scalar import (
    IdenticalScalarGapProblem,
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
)
from .modular_crossing import (
    ModularGapProblem,
    ModularGapSpec,
    build_modular_gap_problem,
)


@dataclass(frozen=True)
class IsolatedRelevantScalarSpec:
    """Shared parameters for the isolated relevant scalar diagnostic."""

    c: object
    delta_phi: object
    delta_gap: object = 2
    degeneracy_phi: object = 1
    modular_lam: int = 5
    four_point_lam: int = 5
    modular_spin_list: Sequence[int] = tuple(range(6))
    four_point_spin_list: tuple[int, ...] = (0, 2)
    modular_dps: int = 80
    four_point_dps: int = 80
    modular_scalar_gap_only: bool = False
    # "fixed": modular degeneracy d_phi is fixed to degeneracy_phi.
    # "allowed": d_phi is an unknown nonnegative variable at Delta_phi.
    modular_phi_mode: str = "fixed"
    # chiral currents (j, 0) + (0, j) with their irreducible characters (ModularGapSpec.chiral_currents)
    modular_chiral_currents: bool = True
    q_order: int = 12
    npoles: int | str = 8
    chiral_norm: int | None = None

    def __post_init__(self):
        for name in ("modular_spin_list", "four_point_spin_list"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True)
class IsolatedRelevantScalarProblems:
    spec: IsolatedRelevantScalarSpec
    modular: ModularGapProblem
    four_point: IdenticalScalarGapProblem


def modular_isolated_spec(spec: IsolatedRelevantScalarSpec) -> ModularGapSpec:
    """Return the Phase-1 modular spec for this experiment point."""
    if not isinstance(spec.modular_scalar_gap_only, bool):
        raise ValueError("modular_scalar_gap_only must be True or False")
    if not isinstance(spec.modular_phi_mode, str) or spec.modular_phi_mode not in {"fixed", "allowed"}:
        raise ValueError("modular_phi_mode must be 'fixed' or 'allowed'")
    fixed_scalars = (
        ((spec.delta_phi, spec.degeneracy_phi),)
        if spec.modular_phi_mode == "fixed"
        else ()
    )
    allowed_scalars = (
        (spec.delta_phi,)
        if spec.modular_phi_mode == "allowed"
        else ()
    )
    if spec.modular_scalar_gap_only:
        return ModularGapSpec(
            c=spec.c,
            delta_gap=0,
            lam=spec.modular_lam,
            spin_list=tuple(spec.modular_spin_list),
            dps=spec.modular_dps,
            delta_min_by_spin={0: spec.delta_gap},
            isolated_scalars=fixed_scalars,
            allowed_isolated_scalars=allowed_scalars,
            chiral_currents=spec.modular_chiral_currents,
        )
    return ModularGapSpec(
        c=spec.c,
        delta_gap=spec.delta_gap,
        lam=spec.modular_lam,
        spin_list=tuple(spec.modular_spin_list),
        dps=spec.modular_dps,
        isolated_scalars=fixed_scalars,
        allowed_isolated_scalars=allowed_scalars,
        chiral_currents=spec.modular_chiral_currents,
    )


def four_point_isolated_spec(spec: IsolatedRelevantScalarSpec) -> IdenticalScalarGapSpec:
    """Return the identical-scalar spec for ``phi x phi = 1 + phi + heavy``."""
    # Divide the input label exactly; neither decimal literals nor stored mpf
    # values should be rounded at the caller's ambient precision here.
    h_phi = literal_number(spec.delta_phi) / 2
    return IdenticalScalarGapSpec(
        c=spec.c,
        h_ext=h_phi,
        delta_gap=spec.delta_gap,
        lam=spec.four_point_lam,
        spin_list=tuple(spec.four_point_spin_list),
        q_order=spec.q_order,
        dps=spec.four_point_dps,
        npoles=spec.npoles,
        chiral_norm=spec.chiral_norm,
        point_constraints=((0, spec.delta_phi),),
    )


def build_isolated_relevant_scalar_problems(
    spec: IsolatedRelevantScalarSpec,
) -> IsolatedRelevantScalarProblems:
    """Build the modular and four-point SDPs for one ``(c, Delta_phi)`` point."""
    modular = build_modular_gap_problem(modular_isolated_spec(spec))
    four_point = build_identical_scalar_gap_problem(four_point_isolated_spec(spec))
    return IsolatedRelevantScalarProblems(
        spec=spec,
        modular=modular,
        four_point=four_point,
    )
