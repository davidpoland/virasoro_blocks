"""Derive LAMBDA_PRESETS from default_params.json for every configured order.

The preset table and direct profile loading must agree. The Lambda=11
profile must also retain its minimum pole count.
"""
from __future__ import annotations

import re

import pytest

from virasoro_bootstrap.defaults import load_default_params
from virasoro_bootstrap.hpc.tasks import LAMBDA_PRESETS, _load_default_params_profile

# Keys the task layer reads off a preset. Any task doing preset["<key>"] needs
# its key listed here, or the agreement check below will not cover it.
_PRESET_KEYS = (
    "q_order",
    "npoles",
    "dps",
    "large_spin_npoles",
    "large_spin_double_twists",
)


def _config_lambdas() -> set[int]:
    return {
        int(m.group(1))
        for m in (re.fullmatch(r"lambda(\d+)", key) for key in load_default_params())
        if m is not None
    }


def test_presets_cover_every_lambda_profile_in_the_config():
    """A new lambdaN rung in the config is reachable from the task layer."""
    assert set(LAMBDA_PRESETS) == _config_lambdas()


@pytest.mark.parametrize("lam", sorted(_config_lambdas()))
@pytest.mark.parametrize("key", _PRESET_KEYS)
def test_preset_agrees_with_the_config_profile(lam, key):
    """The preset table and the direct profile read cannot disagree."""
    expected = _load_default_params_profile(lam).get(key)
    expected = None if expected is None else int(expected)
    assert LAMBDA_PRESETS[lam][key] == expected


def test_lambda11_npoles_is_not_the_drifted_value():
    """The Lambda=11 profile keeps at least 32 poles."""
    assert LAMBDA_PRESETS[11]["npoles"] >= 32
