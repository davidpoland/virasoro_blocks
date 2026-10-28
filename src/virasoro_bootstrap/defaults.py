"""Production defaults from this package's ``default_params.json``.

``$VB_DEFAULT_PARAMS`` names a replacement file (see
:func:`virasoro_bootstrap.paths.research_config_path`); without it only the
shipped copy is read, never a file found relative to the current directory.

The ``defaults`` section supplies parameters shared across derivative orders;
``lambdaN`` sections supply the HPC profiles. Each profile may override
the shared values.

Read the configuration once at import so dataclass fields can use module-level
constants and malformed configuration is reported before a solve begins.
"""

from __future__ import annotations

import json

from .paths import research_config_path

DEFAULT_PARAMS_PATH = research_config_path()


def load_default_params() -> dict:
    """Read one configuration snapshot; later calls may observe file updates."""
    try:
        data = json.loads(DEFAULT_PARAMS_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{DEFAULT_PARAMS_PATH}: configuration is not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise ValueError(f"{DEFAULT_PARAMS_PATH}: configuration must be a JSON object")
    return data


def _defaults_block(data: dict | None = None) -> dict:
    if data is None:
        data = load_default_params()
    block = data.get("defaults")
    if not isinstance(block, dict):
        raise ValueError(
            f"{DEFAULT_PARAMS_PATH} has no top-level 'defaults' object; it is "
            "the source of truth for the cross-Lambda defaults"
        )
    return block


def _required(name: str, block: dict) -> str:
    value = block.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{DEFAULT_PARAMS_PATH}: 'defaults.{name}' is missing or not a "
            "non-empty string"
        )
    return value


def _profile_from_snapshot(data: dict, lam: int) -> dict:
    """Merge a profile and its defaults from the same supplied snapshot."""
    prof = data.get(f"lambda{lam}")
    if not isinstance(prof, dict):
        raise ValueError(
            f"{DEFAULT_PARAMS_PATH} has no 'lambda{lam}' object"
        )
    merged = dict(_defaults_block(data))
    merged.update(prof)
    return merged


def profile(lam: int) -> dict:
    """The ``lambdaN`` profile, with ``defaults`` filled in where absent."""
    return _profile_from_snapshot(load_default_params(), lam)


_IMPORT_DEFAULTS = _defaults_block()

#: Fixed-spin kept-pole selector (``defaults.pole_selection`` in the config).
DEFAULT_POLE_SELECTION = _required("pole_selection", _IMPORT_DEFAULTS)

#: Large-spin TOWER selector. As of 2026-07-30 this has the SAME value as
#: :data:`DEFAULT_POLE_SELECTION`, but it remains a separate key on purpose: the
#: tower has its own conditioning failure mode (inverted geometry, where keeping
#: more poles explodes the order-reduced coefficients) and must stay
#: independently settable. Do not collapse the two into one constant just because
#: they currently agree.
DEFAULT_LARGE_SPIN_POLE_SELECTION = _required("large_spin_pole_selection", _IMPORT_DEFAULTS)

#: Selector for fixed-spin GLOBAL blocks.  It remains separately configurable
#: because the global SL(2) pool is distinct from the Virasoro Kac pool, but
#: both are q-order-defined finite universes and now share the recommended
#: ``contribution_optimal`` default.
DEFAULT_GLOBAL_POLE_SELECTION = _required("global_pole_selection", _IMPORT_DEFAULTS)
