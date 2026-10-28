"""Importable process-pool workers for the table CLI.

Worker functions must live in an importable package module.  Defining them in
``virasoro_blocks.__main__`` works for the installed console entry point but
fails under Python's spawn context when the CLI is launched as
``python -m virasoro_blocks``: the child then sees a fresh ``__main__`` module
and cannot unpickle the functions.  This module keeps the worker state local
to the child while calling the package-owned numerical backend.
"""

from __future__ import annotations

import mpmath as mp


_WORKER_STATE = None


def init_spin_worker(state):
    """Initialize one fixed-spin table worker in an importable module."""
    global _WORKER_STATE

    from ..table_backends import build_cli_chiral_cache

    mp.mp.dps = state["args"].precision      # the working precision (--target-digits escalates per table)
    from ..recursion import set_kac_limit

    set_kac_limit(getattr(state["args"], "kac_limit", None))     # --kac-limit (a spawned worker starts at the default)
    worker_state = dict(state)
    if worker_state["b"] is not None and worker_state["table_request"].central_charge is not None:
        from ..params import b_from_c

        worker_state["b"] = b_from_c(worker_state["table_request"].central_charge)   # at the working precision
    if worker_state["table_request"].family == "global":
        worker_state["chiral_cache"] = None
    else:
        worker_state["chiral_cache"] = build_cli_chiral_cache(
            worker_state["args"],
            worker_state["h_ext"],
            worker_state["hbar_ext"],
            worker_state["b"],
            regularize=worker_state["regularize"],
            basis=worker_state["table_request"].basis,
        )
    _WORKER_STATE = worker_state


def process_spin_worker(ell):
    """Build one fixed-spin table from state installed by the initializer."""
    if _WORKER_STATE is None:
        raise RuntimeError("spin worker was not initialized")

    from ..api import build_table_request
    from ..table_backends import VirasoroBackendOptions

    state = _WORKER_STATE
    mp.mp.dps = state["args"].precision      # the working precision (--target-digits escalates per table)
    from ..recursion import set_kac_limit

    set_kac_limit(getattr(state["args"], "kac_limit", None))     # --kac-limit (a spawned worker starts at the default)
    if state["table_request"].family == "global":
        serialized = build_table_request(
            state["table_request"], spin=ell,
            shared_sample_points=state["args"].shared_sample_points,
        )
    else:
        backend_options = VirasoroBackendOptions(
            method=state["args"].method,
            sample_via=state["args"].sample_via,
            shared_sample_points=state["args"].shared_sample_points,
            regularize=state["regularize"],
            chiral_cache=state["chiral_cache"],
        )
        serialized = build_table_request(
            state["table_request"],
            spin=ell,
            backend_options=backend_options,
        )
    return ell, serialized


__all__ = ["init_spin_worker", "process_spin_worker"]
