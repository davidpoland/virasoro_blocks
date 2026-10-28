"""Cross-check / debugging controls for the identical-scalar bootstrap.

These are NOT the production bound driver (that is
`virasoro_bootstrap.identical_scalar`). They reuse the main driver's
normalization / PMP / certification machinery but swap the Virasoro positivity
blocks for an independently-trusted baseline:

- `global_control` — 2d global SL(2) scalar blocks, plus a direct-numeric
  Virasoro path (no symbolic/interpolation pipeline).
- `chiral_control` — a cheap 1d chiral-block cross-check.

At large `c` the Virasoro positivity blocks should approach these controls, so
disagreement localizes a bug to the blocks rather than the crossing/PMP/SDPB
plumbing. None of these produce a rigorous, publishable bound.
"""

from __future__ import annotations

from .global_control import (
    build_global_identical_scalar_control_problem,
    build_global_scalar_spin_channel,
    build_numeric_virasoro_identical_scalar_control_problem,
    build_numeric_virasoro_scalar_spin_channel,
)

__all__ = [
    "build_global_identical_scalar_control_problem",
    "build_global_scalar_spin_channel",
    "build_numeric_virasoro_identical_scalar_control_problem",
    "build_numeric_virasoro_scalar_spin_channel",
]
