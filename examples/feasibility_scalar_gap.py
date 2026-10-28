"""Minimal scalar-gap feasibility test at Lambda=7, c=32.

Builds the identical-scalar crossing problem for a single scalar gap with a
large-spin double-twist positivity tower, writes the SDP data directly, and
runs SDPB. Reports whether the gap is allowed (primal feasible) or excluded
(dual feasible) from the solver status.

This is the *uncertified* runner: it trusts SDPB's verdict as-is. For an
exclusion that is additionally checked against the exact Virasoro blocks
(including extra spins and intermediate twists), see
``feasibility_scalar_gap_certified.py``.

Usage:
    python examples/feasibility_scalar_gap.py [GAP] [SPINS]

    GAP    scalar (spin-0) lower edge Delta_min(0)  (default 1.0)
    SPINS  comma-separated even spins               (default 0,2,4,6,8,10)

Requires SDPB through Docker, Apptainer, or a native installation. Set
``VB_SDPB_MODE`` to select the runner; see ``docs/CLUSTER.md``.
"""
import sys

import mpmath as mp

from virasoro_bootstrap.defaults import (
    DEFAULT_LARGE_SPIN_POLE_SELECTION,
    DEFAULT_POLE_SELECTION,
)
from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
    run_identical_scalar_gap_problem_in_docker,
)
from virasoro_bootstrap.large_spin import double_twist_set
from virasoro_bootstrap.paths import dated_output_dir

# --- SDPB solver options -----------------------------------------------------
# One dict holds every solver knob, keyed by its sdpb command-line flag. A True
# value emits a bare flag (e.g. --noFinalCheckpoint); a string/number emits
# `flag value`; False/None omits it. See `sdpb --help` for the full list.
#
# Two keys are special: --precision sets SDPB's bit precision, and --mpiProcs
# wraps sdpb in `mpirun -n N` (omit it for a single rank). The SDP data are
# written at spec.dps decimal digits.
SDPB_OPTIONS: dict[str, str | bool | int] = {
    "--precision": 768,
    # "--mpiProcs": 4,
    # Termination: stop at the first feasible point on either side, or at a
    # primal or dual feasible jump (a full step), whichever comes first.
    "--findPrimalFeasible": True,
    "--findDualFeasible": True,
    "--detectPrimalFeasibleJump": True,
    "--detectDualFeasibleJump": True,
    "--writeSolution": "x,y",
    "--noFinalCheckpoint": True,
    # Convergence thresholds (SDPB default 1e-30 each):
    # "--dualityGapThreshold": "1e-30",
    # "--primalErrorThreshold": "1e-30",
    # "--dualErrorThreshold": "1e-30",
    # Iteration / step controls:
    # "--maxIterations": "500",
    # "--maxRuntime": "86400",
    # "--maxComplementarity": "1e+100",
    # "--stepLengthReduction": "0.7",
    # "--feasibleCenteringParameter": "0.1",
    # "--infeasibleCenteringParameter": "0.3",
    # "--initialMatrixScalePrimal": "1e+20",
    # "--initialMatrixScaleDual": "1e+20",
    # "--verbosity": "1",
}


def main() -> None:
    mp.mp.dps = 120     # before parsing the inputs, so that they keep their digits
    gap = mp.mpf(sys.argv[1]) if len(sys.argv) > 1 else mp.mpf("1.0")
    spins = tuple(
        int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,2,4,6,8,10").split(",")
    )

    h_ext = mp.mpf("0.0625")  # Delta_ext = 2*h_ext = 1/8
    # Large-spin tower: continuous-in-L positivity for L >= Lmax at each
    # double-twist slice, picking up where the discrete fixed spins stop.
    lmax = max(spins) + 2
    twists = tuple(double_twist_set(h_ext, 5))

    spec = IdenticalScalarGapSpec(
        c=mp.mpf(32),
        h_ext=h_ext,
        delta_gap=gap,
        lam=7,
        spin_list=spins,
        q_order=40,
        dps=120,
        npoles=10,
        pole_selection=DEFAULT_POLE_SELECTION,
        large_spin_lmin=lmax,
        large_spin_twists=twists,
        large_spin_pole_selection=DEFAULT_LARGE_SPIN_POLE_SELECTION,
        num_threads=4,
    )
    problem = build_identical_scalar_gap_problem(spec)

    out_dir = dated_output_dir("feasibility_l7_c32") / f"gap_{mp.nstr(gap, 5)}"
    run = run_identical_scalar_gap_problem_in_docker(
        problem,
        out_dir,
        sdpb_options=SDPB_OPTIONS,
        pmp2sdp_log_name="pmp2sdp.log",
        sdpb_log_name="sdpb.log",
    )

    print(f"c=32  Lambda=7  Delta_ext=1/8  spins={spins}")
    print(f"large-spin tower: Lmax={lmax}  twists={[mp.nstr(t, 4) for t in twists]}")
    print(f"scalar gap Delta_min(0) = {mp.nstr(gap, 6)}")
    print(f"SDPB status: {run.sdpb_status}")

    if run.sdpb_status == "primal_feasible":
        print("-> ALLOWED (primal feasible)")
    elif run.sdpb_status in ("dual_feasible", "optimal"):
        print("-> EXCLUDED (dual feasible)")
    else:
        print(f"-> INCONCLUSIVE ({run.sdpb_status})")

    print(f"artifacts in {out_dir}")


if __name__ == "__main__":
    main()
