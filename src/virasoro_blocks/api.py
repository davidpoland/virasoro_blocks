"""Supported block-engine API, with imports loaded on demand.

The package initializer re-exports additional names for compatibility.
New applications should import from this module; internal recursion helpers
may change independently of the supported API.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "FChiralH": (".chiral", "FChiralH"),
    "F_block_chiral_h": (".chiral", "F_block_chiral_h"),
    "B_block": (".composer", "B_block"),
    "G_block": (".composer", "G_block"),
    "parity_projected_B_block": (".composer", "parity_projected_B_block"),
    "parity_reversed_B_block": (".composer", "parity_reversed_B_block"),
    "q_of_z": (".coordinates", "q_of_z"),
    "q_symmetric": (".coordinates", "q_symmetric"),
    "z_of_q": (".coordinates", "z_of_q"),
    "z_symmetric": (".coordinates", "z_symmetric"),
    "BlockPoleError": (".evaluation", "BlockPoleError"),
    "evaluate_block_request": (".evaluation", "evaluate_block_request"),
    "F_block": (".recursion", "F_block"),
    "H_q_series": (".recursion", "H_q_series"),
    "block_prefactor": (".recursion", "block_prefactor"),
    "q_from_z": (".recursion", "q_from_z"),
    "BlockCapabilities": (".requests", "BlockCapabilities"),
    "BlockRequest": (".requests", "BlockRequest"),
    "BlockValue": (".requests", "BlockValue"),
    "NumericalPolicy": (".requests", "NumericalPolicy"),
    "RequestValidationError": (".requests", "RequestValidationError"),
    "capabilities_for": (".requests", "capabilities_for"),
    "validate_request": (".requests", "validate_request"),
    "SerializedDerivTable": (".serialization", "SerializedDerivTable"),
    "serialize_deriv_tables": (".serialization", "serialize_deriv_tables"),
    "serialized_table_from_json": (".serialization", "serialized_table_from_json"),
}


def build_table_request(request: BlockRequest, *, spin=0, **backend_options):
    """Build a derivative table from a canonical table request.

    ``request.operation`` selects between the ordinary ``table`` backend and
    the explicit frozen ``point-table`` backend.  Family-specific execution
    options are accepted as keyword arguments for advanced callers, while
    ordinary users can rely on the defaults used by the CLI.
    """
    import mpmath as mp

    from .requests import RequestValidationError
    from .table_adapters import table_adapter_for

    adapter = table_adapter_for(request.family)
    if request.operation not in ("table", "point-table"):
        raise RequestValidationError(
            "build_table_request requires operation='table' or 'point-table'"
        )
    if request.operation == "point-table" and backend_options:
        names = ", ".join(sorted(backend_options))
        raise TypeError(
            f"point-table requests do not accept backend options: {names}"
        )
    _refuse_unknown_options(adapter, backend_options)
    if spin is not None:
        if request.family.startswith("torus"):
            from .table_adapters import _torus_spin

            spin = _torus_spin(spin)
        else:
            from .numeric_input import _signed_half_integer

            try:
                spin = _signed_half_integer(spin)
            except ValueError as exc:
                raise RequestValidationError(f"spin: {exc}") from exc
    exchanged = request.exchange_spin if request.operation == "point-table" else spin
    if exchanged is not None and exchanged != int(exchanged) and _is_scalar_G(request):
        # G = B(h, hbar) + B(hbar, h) is the block of an integer-spin exchange between scalars.
        raise RequestValidationError(
            f"scalar_G needs an integer exchanged spin, got {exchanged}; use ordered_B or parity_projected_B"
        )
    _refuse_unused_table_fields(request, spin=spin)
    # Use the request's precision for both CLI and Python callers.
    with mp.workdps(int(request.policy.precision)):
        if request.operation == "point-table":
            return adapter.build_verified_point_table(request)
        return adapter.build_table(
            request, spin=spin, **backend_options
        )


def _refuse_unused_table_fields(request, *, spin) -> None:
    """Reject fields that do not apply to the requested table family and basis."""
    import mpmath as mp

    from .requests import RequestValidationError

    def given(name):
        return getattr(request, name) is not None

    torus = request.family.startswith("torus")
    chiral = request.basis.endswith("chiral_F") or (
        request.basis == "auto" and any(c in ("z", "y", "q")
                                      for c in (request.coordinates or (request.coordinate,))))
    unused = []
    if request.family not in ("ns", "ramond") and request.sector != "even":
        unused.append("sector")
    if request.family != "ramond" and request.branch != "++":
        unused.append("branch")
    # (parity data without a projection is recorded as the table's parity_basis metadata)
    if request.operation == "table":
        unused += [name for name in ("exchange_h", "exchange_hbar", "exchange_delta", "exchange_spin") if given(name)]
    else:
        unused += [name for name in ("gap",) if given(name)]
        unused += [name for name in (("exchange_delta", "exchange_hbar", "exchange_spin") if chiral
                                    else ("exchange_h", "exchange_hbar")) if given(name)]
        unused += [f"policy.{name}" for name in ("npoles", "pole_selection")
                   if getattr(request.policy, name) is not None]
        if spin is not None and mp.mpf(spin) != 0:
            unused.append("spin (point-table uses request.exchange_spin)")
    if request.point:
        unused.append("point")
    if torus:
        unused += [name for name in ("chiral_norm", "z_shift", "zbar_shift", "center", "center_bar", "parity",
                                     "h_ext", "hbar_ext", "delta_ext", "spin_ext") if given(name)]
        if request.policy.pole_selection is not None:
            unused.append("policy.pole_selection")
        if request.family == "torus-character":
            unused += [name for name in ("gap", "torus_onepoint_ext") if given(name)]
    else:
        unused += [name for name in ("torus_onepoint_ext",) if given(name)]
        if given("h_ext"):
            unused += [name for name in ("delta_ext", "spin_ext") if given(name)]
        elif given("hbar_ext"):
            unused.append("hbar_ext (without h_ext)")
    if chiral:
        unused += [name for name in ("center_bar", "zbar_shift", "hbar_ext") if given(name)]
        if request.operation == "table" and spin is not None and mp.mpf(spin) != 0:
            unused.append(f"spin={spin} (a chiral table has no spin)")
    if _is_scalar_G(request) and given("zbar_shift") and (
            request.zbar_shift != (request.z_shift if given("z_shift") else 0)):
        unused.append("zbar_shift (scalar_G carries one shift, z_shift; the CLI refuses it too)")
    if unused:
        raise RequestValidationError(
            f"{', '.join(unused)} do(es) not apply to a {request.family} {request.basis} {request.operation}")


def _refuse_unknown_options(adapter, options) -> None:
    """Reject backend options that the selected adapter does not accept."""
    import inspect

    parameters = inspect.signature(adapter.build_table).parameters
    known = {name for name, p in parameters.items()
             if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD) and name not in ("self", "request", "spin")}
    unknown = sorted(set(options) - known)
    if unknown:
        raise TypeError(f"unknown table option(s) {unknown} for the {type(adapter).__name__}; it takes "
                        f"{sorted(known) or 'none'}")


def _is_scalar_G(request) -> bool:
    """Whether ``request`` builds a scalar_G table (explicitly, or under basis='auto': scalar externals, 2d)."""
    if request.basis != "auto":
        return request.basis.endswith("scalar_G")
    if request.family.startswith("torus") or any(c in ("z", "y", "q")
                                               for c in (request.coordinates or (request.coordinate,))):
        return False
    spinning = any(s != 0 for s in (request.spin_ext or ())) or (
        request.hbar_ext is not None and request.h_ext is not None and tuple(request.hbar_ext) != tuple(request.h_ext))
    return not spinning and request.parity is None


def __getattr__(name):
    """Load one supported facade symbol on first access."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    module = import_module(module_name, __package__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "B_block",
    "BlockCapabilities",
    "BlockPoleError",
    "BlockRequest",
    "BlockValue",
    "FChiralH",
    "F_block",
    "F_block_chiral_h",
    "G_block",
    "H_q_series",
    "NumericalPolicy",
    "RequestValidationError",
    "SerializedDerivTable",
    "block_prefactor",
    "build_table_request",
    "capabilities_for",
    "evaluate_block_request",
    "parity_projected_B_block",
    "parity_reversed_B_block",
    "q_from_z",
    "q_of_z",
    "q_symmetric",
    "serialize_deriv_tables",
    "serialized_table_from_json",
    "z_of_q",
    "z_symmetric",
    "validate_request",
]
