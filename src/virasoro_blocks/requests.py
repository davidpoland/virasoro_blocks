"""Shared request, policy, and capability records for the block API.

The numerical backends predate the CLI unification work and expose family-
specific call signatures.  These small records provide a stable vocabulary at
the boundary without forcing a rewrite of the recurrence or interpolation
modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import mpmath as mp


#: Smallest working precision (decimal digits) for sphere table builds: below
#: it the interpolation / pole-recognition tolerances (``10**-(dps - 10)``)
#: leave no digits and the builds fail with internal errors.
MIN_TABLE_PRECISION = 20


class BlockPoleError(ValueError):
    """Common fail-closed error for a genuine exchanged-module pole."""

    def __init__(
        self, message=None, *, family=None, basis=None, exchanged=None,
        pole=None, side=None,
    ):
        self.family = family
        self.basis = basis
        self.exchanged = exchanged
        self.pole = pole
        self.side = side
        if message is None:
            where = f" ({side} chiral half)" if side else ""
            message = (
                f"{family} {basis} exchanged value {mp.nstr(exchanged, 12)} "
                f"is on the Kac/SL(2) pole {mp.nstr(pole, 12)}{where} at precision {mp.mp.dps}; "
                "direct point evaluation is undefined there (an offset below this precision needs more of it: "
                "raise the precision, or pass target_digits)"
            )
        super().__init__(message)


class RequestValidationError(ValueError):
    """A user-facing block request is inconsistent or incomplete."""


def _integer_field(name, value) -> int:
    """An integral policy value as ``int``; 2.7, -0.5, "3" or True are errors."""
    import numpy as np

    if isinstance(value, (bool, np.bool_, str)):
        raise RequestValidationError(f"{name} must be an integer, got {value!r}")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RequestValidationError(f"{name} must be an integer, got {value!r}") from exc
    if integer != value:
        raise RequestValidationError(f"{name} must be an integer, got {value!r}")
    return integer


_REQUEST_NUMBER_FIELDS = (
    "central_charge", "exchange_h", "exchange_hbar", "exchange_delta", "exchange_spin",
    "torus_onepoint_ext", "torus_scalar_shift", "gap", "chiral_norm",
    "z_shift", "zbar_shift", "center", "center_bar",
)
_REQUEST_TUPLE_FIELDS = ("h_ext", "hbar_ext", "delta_ext", "spin_ext")


def _coerce_number(value, name, *, optional=False):
    """Preserve exact real literals until the numerical backend needs them.

    An mpmath value is a numerical value at its precision, not an exact
    label: it passes through (checked finite) rather than becoming the
    rational of its stored bits.
    """
    from .numeric_input import literal_number

    if optional and value is None:
        return None
    try:
        if value is None or isinstance(value, bool):
            raise ValueError("a numeric component cannot be null or boolean")
        if isinstance(value, mp.mpf):
            if not mp.isfinite(value):
                raise ValueError("numeric inputs must be finite")
            return value
        value = literal_number(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise RequestValidationError(f"{name}: {value!r} is not a finite number") from exc
    from fractions import Fraction

    if isinstance(value, (int, float, complex, Fraction, mp.mpf, mp.mpc)):
        return value
    try:
        converted = mp.mpmathify(value)
        if not mp.isfinite(converted):
            raise ValueError("numeric inputs must be finite")
        return converted
    except (TypeError, ValueError, OverflowError) as exc:
        raise RequestValidationError(f"{name}: {value!r} is not a finite number") from exc


@dataclass(frozen=True)
class NumericalPolicy:
    """Numerical controls shared by direct and table block operations.

    ``precision`` sets the table's working and output precision, like the CLI's
    ``--precision``. Set ``target_digits`` to request a minimum number of
    verified digits. The build then starts at ``precision`` and increases it
    by the measured deficit, up to ``max_precision`` (default
    ``max(4 target_digits, 2 precision, target_digits + 200)``).

    ``order`` controls the q-series truncation, ``lambda_order`` the Taylor
    order, and ``npoles`` the rational interpolation. Public requests support
    only ``pole_policy="raise"``. Internal interpolation probes use separate
    controls.
    """

    precision: int = 60
    order: int | None = 30
    lambda_order: int | None = None
    npoles: int | str | None = None
    pole_selection: str | None = None
    c_regularization: str = "auto"
    pole_policy: str = "raise"
    target_digits: int | None = None
    max_precision: int | None = None

    def __post_init__(self):
        precision = _integer_field("precision", self.precision)
        if precision < 10:
            raise RequestValidationError("precision must be >= 10")
        object.__setattr__(self, "precision", precision)
        from .verified_build import KnobNames, PrecisionRequest

        try:
            checked = PrecisionRequest.make(
                precision, self.target_digits, self.max_precision,
                knobs=KnobNames(precision="precision", target="target_digits", max_precision="max_precision",
                                assign="{name}={value}"),
            )
        except ValueError as exc:
            raise RequestValidationError(str(exc)) from None
        object.__setattr__(self, "target_digits", checked.target_digits)
        object.__setattr__(self, "max_precision", checked.max_precision)
        for name in ("order", "lambda_order"):
            value = getattr(self, name)
            if value is None:
                continue
            value = _integer_field(name, value)
            if value < 0:
                raise RequestValidationError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)
        if isinstance(self.npoles, str):
            if self.npoles.strip().lower() != "all":
                raise RequestValidationError(
                    f"npoles must be a non-negative integer or 'all', got {self.npoles!r}"
                )
            object.__setattr__(self, "npoles", "all")
        elif self.npoles is not None:
            npoles = _integer_field("npoles", self.npoles)
            if npoles < 0:
                raise RequestValidationError("npoles must be >= 0 or 'all'")
            object.__setattr__(self, "npoles", npoles)
        if self.c_regularization not in {"auto", "on", "off"}:
            raise RequestValidationError(
                "c_regularization must be 'auto', 'on', or 'off'"
            )
        if self.pole_policy != "raise":
            raise RequestValidationError(
                "public block requests only support pole_policy='raise'"
            )


@dataclass(frozen=True)
class BlockRequest:
    """Canonical request shared by the CLI and supported Python API."""

    operation: str
    surface: str = "sphere"
    family: str = "virasoro"
    basis: str = "auto"
    coordinate: str = "zzb"
    coordinates: tuple[str, ...] = ()
    h_ext: tuple[Any, ...] | None = None
    hbar_ext: tuple[Any, ...] | None = None
    delta_ext: tuple[Any, ...] | None = None
    spin_ext: tuple[Any, ...] | None = None
    central_charge: Any | None = None
    exchange_h: Any | None = None
    exchange_hbar: Any | None = None
    exchange_delta: Any | None = None
    exchange_spin: int | None = None
    torus_onepoint_ext: Any | None = None
    torus_scalar_shift: Any | None = None
    #: Positivity anchor raised this far above the basis unitarity bound
    #: (|ell| for 2d, 0 for chiral). The block's anchor IS its positivity
    #: domain, so this is how a gap is expressed in a block file.
    gap: Any | None = None
    point: Mapping[str, Any] = field(default_factory=dict)
    sector: str = "even"
    branch: str = "++"
    parity: str | None = None
    parity_external: tuple[str, ...] | None = None
    parity_exchanged: str | None = None
    chiral_norm: Any | None = None
    z_shift: Any | None = None
    zbar_shift: Any | None = None
    center: Any | None = None
    center_bar: Any | None = None
    policy: NumericalPolicy = field(default_factory=NumericalPolicy)

    def __post_init__(self):
        if self.operation not in {"evaluate", "table", "point-table"}:
            raise RequestValidationError(
                "operation must be evaluate, table, or point-table"
            )
        if self.surface not in {"sphere", "torus"}:
            raise RequestValidationError("surface must be sphere or torus")
        if self.family not in {
            "virasoro", "global", "ns", "ramond", "torus-character",
            "torus-onepoint",
        }:
            raise RequestValidationError(f"unknown block family {self.family!r}")
        if self.basis == "auto" and self.family == "torus-character":
            object.__setattr__(self, "basis", "torus_G")
        if self.basis == "auto" and self.family == "torus-onepoint":
            object.__setattr__(self, "basis", "torus_onept_G")
        # Validation and the backend each inspect these sequences. Own one
        # snapshot so a generator cannot disappear, or a caller's list change
        # the request between those passes.
        for name in ("coordinates", "parity_external"):
            values = getattr(self, name)
            if values is not None and not isinstance(values, tuple):
                try:
                    snapshot = tuple(values)
                except TypeError as exc:
                    raise RequestValidationError(f"{name} must be an iterable of labels") from exc
                object.__setattr__(self, name, snapshot)
            if values is not None and any(not isinstance(value, str) for value in getattr(self, name)):
                raise RequestValidationError(f"{name} must contain string labels")
        if not isinstance(self.point, Mapping):
            raise RequestValidationError("point must be a mapping of coordinate names to numbers")
        if any(not isinstance(key, str) for key in self.point):
            raise RequestValidationError("point coordinate names must be strings")
        # Real literals remain rational. Complex text starts above the
        # escalation cap and guards its written component offsets before
        # conversion; existing mp values keep their supplied value.
        from .verified_build import precision_request_of

        with mp.workdps(precision_request_of(self.policy).input_dps):
            for name in _REQUEST_NUMBER_FIELDS:
                value = getattr(self, name)
                coerced = _coerce_number(value, name, optional=True)
                if name == 'central_charge' and isinstance(coerced, (complex, mp.mpc)):
                    from .numeric_input import _complex_charge_input

                    coerced = _complex_charge_input(coerced)
                if coerced is not value:
                    object.__setattr__(self, name, coerced)
            for name in _REQUEST_TUPLE_FIELDS:
                values = getattr(self, name)
                if values is None:
                    continue
                try:
                    if isinstance(values, (str, bytes, Mapping)):
                        raise TypeError
                    values = tuple(values)
                except TypeError as exc:
                    raise RequestValidationError(f"{name} must be a sequence of four numbers") from exc
                coerced = tuple(_coerce_number(value, f"{name}[{i}]") for i, value in enumerate(values))
                object.__setattr__(self, name, coerced)
            # Copy even empty mappings and already normalized components:
            # their source dictionary remains mutable after construction.
            point = {key: _coerce_number(value, f"point[{key!r}]") for key, value in self.point.items()}
            object.__setattr__(self, "point", point)


def validate_request(request: BlockRequest) -> BlockCapabilities:
    """Validate operation-level compatibility and return its capabilities.

    Family adapters still validate their numerical parameters (exchanged
    labels, parity data, and so on), but this common gate makes the external
    vector lengths and operation/surface/basis/coordinate contract uniform
    before a backend is entered.
    """
    capability = capabilities_for(request.family)[0]
    if not capability.supports_operation(request.operation):
        raise RequestValidationError(
            f"family {request.family!r} does not support operation "
            f"{request.operation!r}"
        )
    if request.surface not in capability.surfaces:
        raise RequestValidationError(
            f"family {request.family!r} does not support surface "
            f"{request.surface!r}"
        )
    if request.surface == "sphere":
        for name in _REQUEST_TUPLE_FIELDS:
            values = getattr(request, name)
            if values is not None and len(values) != 4:
                raise RequestValidationError(f"{name} must contain exactly four entries")
    allowed_bases = set(capability.bases) | {"auto"}
    if request.basis not in allowed_bases:
        prefix = f"{request.family}_"
        short_bases = {
            "scalar_G", "ordered_B", "parity_projected_B", "chiral_F",
        }
        if not (request.surface == "sphere" and (
            request.basis in short_bases
            or
            request.basis in {
                f"{prefix}scalar_G", f"{prefix}ordered_B",
                f"{prefix}parity_projected_B", f"{prefix}chiral_F",
            }
        )):
            raise RequestValidationError(
                f"basis {request.basis!r} is not supported by family "
                f"{request.family!r}"
            )
    coordinates = request.coordinates or (request.coordinate,)
    allowed_coordinates = capability.coordinates_for(request.operation)
    unsupported = [coord for coord in coordinates if coord not in allowed_coordinates]
    if unsupported:
        raise RequestValidationError(
            f"family {request.family!r} does not support {request.operation!r} "
            f"coordinates {unsupported}; choose from {allowed_coordinates}"
        )
    if request.parity_exchanged is not None and request.surface == "sphere":
        from .api import _is_scalar_G
        from .parity import normalize_parity_phase

        if _is_scalar_G(request) and normalize_parity_phase(str(request.parity_exchanged)) != "1":
            raise RequestValidationError(
                scalar_G_exchanged_parity_message(request.parity_exchanged, "parity_exchanged"))
    if request.policy.c_regularization == "off" and request.central_charge is not None:
        locus = _singular_locus_label(request.family, request.central_charge)
        if locus is not None:
            raise RequestValidationError(
                f"c is a recognized singular central charge ({locus}), where the bare recursion is singular: "
                "regularization 'off' cannot evaluate it; use 'auto' (the default)"
            )
    if request.gap is not None and request.surface == "sphere":
        # (a torus one-point table checks its own spin-0 anchor)
        problem = sphere_gap_problem(request.gap)
        if problem is not None:
            raise RequestValidationError(problem.replace("{name}", "gap"))
    if request.torus_scalar_shift is not None:
        if request.family != "torus-onepoint" or request.operation != "table":
            # a table's positivity anchor: an evaluation takes the exchanged Delta itself, a character table
            # has its own anchor rule -- accepting it would ignore it
            raise RequestValidationError(
                "torus_scalar_shift is the spin-0 anchor of torus one-point TABLES; "
                f"{request.family} {request.operation} does not use it"
            )
        if request.gap is not None:
            raise RequestValidationError(
                "gap and torus_scalar_shift both set the spin-0 anchor: gap G anchors every spin at |j| + G "
                "(spin 0 at G), torus_scalar_shift X anchors spin 0 alone at X; pass one"
            )
    return capability


def scalar_G_exchanged_parity_message(value, name) -> str:
    """Why an exchanged parity other than +1 does not apply to a scalar_G block (``name`` spells the option;
    a ``--`` flag gets the CLI's pointer)."""
    where = ("; the CLI builds parity_projected_B for spinning externals only, so for scalar externals use the "
             "Python API (a BlockRequest with basis 'parity_projected_B' and parity '-')" if name.startswith("--")
             else "")
    return (f"{name} {value} does not apply to scalar_G: the scalar G = B(h, hbar) + B(hbar, h) is z <-> zbar even "
            "at every spin (the only exchanged parity it admits is +1). The z <-> zbar-odd structure "
            f"B(h, hbar) - B(hbar, h) is parity_projected_B with parity '-' (eta = -1){where}")


def sphere_gap_problem(gap):
    """Why ``gap`` cannot raise a sphere table's positivity anchor, or None when it can.

    The anchor is ``|ell| + gap`` (``gap`` for a chiral table), so ``gap`` measures the distance above the
    unitarity bound: it must be a finite real number ``>= 0``.  A negative gap would anchor the table below
    unitarity (and an anchor below ``0`` breaks the interpolation's sample-point construction).  The message
    spells the option ``{name}``, for the caller to fill in.
    """
    from .numeric_input import literal_components

    try:
        real, imaginary = literal_components(gap)
        finite = mp.isfinite(mp.convert(real))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError):
        return f"{{name}} must be a finite real number, got {gap!r}"
    if imaginary != 0 or not finite:
        return f"{{name}} must be a finite real number, got {gap!r}"
    if real < 0:
        return (f"{{name}} {gap} is negative: the gap raises the positivity anchor above the unitarity bound "
                "(|ell| + gap for the 2d bases, gap for chiral), so a negative gap would anchor the table below "
                "unitarity; pass a gap >= 0, or omit it for the unitarity bound itself")
    return None


def _singular_locus_label(family, central_charge):
    """The recognized singular locus the family's bare recursion cannot evaluate at ``central_charge``, or None."""
    from .numeric_input import parse_number

    try:
        c = parse_number(central_charge)
    except (TypeError, ValueError):
        return None
    if family in ("virasoro", "torus-onepoint"):
        from .singularity import recognized_singular_locus
        return recognized_singular_locus(c)
    if family in ("ns", "ramond"):
        from .superconformal_blocks import is_superconformal_singular_c
        return "an N=1 singular central charge" if is_superconformal_singular_c(c) else None
    return None


@dataclass(frozen=True)
class BlockValue:
    """A pointwise block value with reproducibility metadata."""

    value: Any
    metadata: Mapping[str, Any]

    @staticmethod
    def _number(value, digits=None):
        from fractions import Fraction

        digits = mp.mp.dps if digits is None else int(digits)
        if isinstance(value, Fraction):
            return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        if isinstance(value, mp.mpf):
            # nstr reads the stored binary value: no re-rounding at the
            # (possibly lower) ambient precision.
            return mp.nstr(value, digits)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            with mp.workdps(max(mp.mp.dps, digits) + 10):
                return mp.nstr(mp.mpf(value), digits)
        if isinstance(value, mp.mpc):
            # A relative test: an imaginary part is dropped only when it is
            # rounding noise against the whole value, so a genuinely complex
            # value of small modulus keeps its imaginary part.
            # Chop only at rounding scale. A half-precision threshold discarded
            # resolved components (e.g. Im/Re ~ 1e-30 in a 40-digit result).
            imag_tol = mp.mpf(10) ** (-max(12, digits - 2)) * abs(value)
            if abs(value.imag) <= imag_tol:
                return mp.nstr(value.real, digits)
            return {
                "real": mp.nstr(value.real, digits),
                "imag": mp.nstr(value.imag, digits),
            }
        if isinstance(value, complex):
            with mp.workdps(max(mp.mp.dps, digits) + 10):
                return BlockValue._number(mp.mpc(value), digits)
        return value

    def as_dict(self) -> dict[str, Any]:
        digits = self.metadata.get("precision_actual") or self.metadata.get("precision", mp.mp.dps)
        return {
            # a spin-0 scalar_G value is B(h, h) under the default normalization; the metadata records it as
            # ``scalar_G_spin0``
            "schema": "block-value/v2",
            **dict(self.metadata),
            "value": self._number(self.value, digits),
        }


@dataclass(frozen=True)
class BlockCapabilities:
    """What a family adapter can implement."""

    family: str
    operations: tuple[str, ...]
    surfaces: tuple[str, ...]
    bases: tuple[str, ...]
    coordinates: tuple[str, ...]
    supports_order: bool
    supports_lambda: bool
    supports_npoles: bool
    supports_precision: bool = True
    evaluate_coordinates: tuple[str, ...] | None = None
    point_table_coordinates: tuple[str, ...] | None = None
    output_formats: tuple[str, ...] = ("json",)

    def supports_operation(self, operation: str) -> bool:
        return operation in self.operations

    def coordinates_for(self, operation: str) -> tuple[str, ...]:
        if operation == "evaluate":
            return self.coordinates if self.evaluate_coordinates is None else self.evaluate_coordinates
        if operation == "point-table":
            return self.coordinates if self.point_table_coordinates is None else self.point_table_coordinates
        return self.coordinates

    def as_dict(self) -> dict[str, Any]:
        evaluate_coordinates = (
            self.coordinates
            if self.evaluate_coordinates is None else self.evaluate_coordinates
        )
        point_table_coordinates = (
            self.coordinates
            if self.point_table_coordinates is None else self.point_table_coordinates
        )
        return {
            "family": self.family,
            "operations": list(self.operations),
            "surfaces": list(self.surfaces),
            "bases": list(self.bases),
            "coordinates": list(self.coordinates),
            "evaluate_coordinates": list(evaluate_coordinates),
            "point_table_coordinates": list(point_table_coordinates),
            "output_formats": list(self.output_formats),
            "supports_order": self.supports_order,
            "supports_lambda": self.supports_lambda,
            "supports_npoles": self.supports_npoles,
            "supports_precision": self.supports_precision,
        }


_SPHERE_2D = ("zzb", "yyb", "qqb")
_SPHERE_CHIRAL = ("z", "y", "q")
_SPHERE_FOLDED = (
    "ws", "xt", "qqb_ws", "ws_radial", "xt_radial", "qqb_ws_radial"
)


CAPABILITIES: dict[str, BlockCapabilities] = {
    "virasoro": BlockCapabilities(
        family="virasoro", operations=("evaluate", "table", "point-table"),
        surfaces=("sphere",),
        bases=("scalar_G", "ordered_B", "parity_projected_B", "chiral_F"),
        coordinates=_SPHERE_2D + _SPHERE_CHIRAL + _SPHERE_FOLDED,
        evaluate_coordinates=_SPHERE_2D + _SPHERE_CHIRAL,
        point_table_coordinates=_SPHERE_2D + _SPHERE_CHIRAL + _SPHERE_FOLDED,
        output_formats=("json", "m", "both", "text"),
        supports_order=True, supports_lambda=True, supports_npoles=True,
    ),
    "global": BlockCapabilities(
        family="global", operations=("evaluate", "table", "point-table"),
        surfaces=("sphere",),
        bases=(
            "global_scalar_G", "global_ordered_B",
            "global_parity_projected_B", "global_chiral_F",
        ),
        coordinates=_SPHERE_2D + _SPHERE_CHIRAL + _SPHERE_FOLDED,
        evaluate_coordinates=_SPHERE_2D + _SPHERE_CHIRAL,
        point_table_coordinates=_SPHERE_2D + _SPHERE_CHIRAL + _SPHERE_FOLDED,
        output_formats=("json", "m", "both", "text"),
        # --order sets the size of the SL(2) pole pool for the global family
        # (the 2F1 itself is evaluated exactly), so the control is meaningful.
        supports_order=True, supports_lambda=True, supports_npoles=True,
    ),
    "ns": BlockCapabilities(
        family="ns", operations=("evaluate", "table", "point-table"), surfaces=("sphere",),
        bases=("ns_scalar_G", "ns_ordered_B", "ns_parity_projected_B", "ns_chiral_F"),
        coordinates=_SPHERE_2D + _SPHERE_CHIRAL + _SPHERE_FOLDED,
        evaluate_coordinates=_SPHERE_2D + _SPHERE_CHIRAL,
        point_table_coordinates=_SPHERE_2D + _SPHERE_CHIRAL + _SPHERE_FOLDED,
        output_formats=("json", "m", "both", "text"),
        supports_order=True, supports_lambda=True, supports_npoles=True,
    ),
    "ramond": BlockCapabilities(
        family="ramond", operations=("evaluate", "table", "point-table"), surfaces=("sphere",),
        bases=(
            "ramond_scalar_G", "ramond_ordered_B",
            "ramond_parity_projected_B", "ramond_chiral_F",
        ),
        coordinates=_SPHERE_2D + _SPHERE_CHIRAL + _SPHERE_FOLDED,
        evaluate_coordinates=_SPHERE_2D + _SPHERE_CHIRAL,
        point_table_coordinates=_SPHERE_2D + _SPHERE_CHIRAL + _SPHERE_FOLDED,
        output_formats=("json", "m", "both", "text"),
        supports_order=True, supports_lambda=True, supports_npoles=True,
    ),
    "torus-character": BlockCapabilities(
        family="torus-character", operations=("evaluate", "table", "point-table"),
        surfaces=("torus",), bases=("torus_G",), coordinates=("vvb",),
        evaluate_coordinates=("vvb",),
        point_table_coordinates=("vvb",),
        output_formats=("json",),
        supports_order=False, supports_lambda=True, supports_npoles=False,
    ),
    "torus-onepoint": BlockCapabilities(
        family="torus-onepoint", operations=("evaluate", "table", "point-table"),
        surfaces=("torus",), bases=("torus_onept_G",), coordinates=("vvb",),
        evaluate_coordinates=("vvb",),
        point_table_coordinates=("vvb",),
        output_formats=("json",),
        supports_order=True, supports_lambda=True, supports_npoles=True,
    ),
}


def capabilities_for(family: str | None = None) -> list[BlockCapabilities]:
    """Return capabilities in stable family order."""
    if family is None or family == "auto":
        return list(CAPABILITIES.values())
    try:
        return [CAPABILITIES[family]]
    except KeyError as exc:
        raise RequestValidationError(f"unknown block family {family!r}") from exc
