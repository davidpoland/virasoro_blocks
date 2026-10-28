"""Minimal OPE-coefficient maximization at Lambda=7, c=4.

Bounds the squared OPE coefficient of the *gap-saturating* scalar -- the lowest
spin-0 operator above the identity, placed at Delta = delta_star (the continuum
bottom). Unlike the gap feasibility test (which returns a binary allowed/excluded
verdict), this returns a continuous extremal number, which is a far more
sensitive probe when tuning block-table parameters.

The setup reuses the identical-scalar gap build verbatim and only swaps the PMP
vectors: objective = identity crossing vector, normalization = the distinguished
operator's vector V_*. SDPB then runs in *optimization* mode (no
--findPrimalFeasible/--findDualFeasible), and the bound is read off the optimum:

    OPE^2 bound = -dualObjective   (ope_squared_bound_from_objective)

The bound is in the library normalization: it bounds the coefficient of the
scalar block G, which is 16^Delta* lambda^2 for the squared OPE coefficient
lambda^2 of the standard block z^{h-h1-h2}(1 + O(z)) (the default
chiral_norm 16; see "Block normalization" in docs/CLI.md). The script prints both.

Behavior to expect: a finite positive bound for delta_star at or below the
scalar gap bound Delta*(0); above it the
operator can no longer carry positive OPE weight and SDPB's objective runs away
(status max_complementarity), which is itself a (coarse) locator of Delta*(0).

Usage:
    python examples/ope_max_scalar.py [DELTA_STAR] [C] [SPINS]

    DELTA_STAR  dimension of the distinguished scalar  (default 0.8)
    C           central charge                          (default 4)
    SPINS       comma-separated even spins               (default 0,2,4,6,8,10)

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
    build_identical_scalar_ope_max_problem,
    ope_squared_bound_from_objective,
    read_sdpb_objective,
    run_identical_scalar_gap_problem_in_docker,
)
from virasoro_bootstrap.large_spin import double_twist_set
from virasoro_bootstrap.paths import dated_output_dir

# Optimization mode: omit --find*Feasible so SDPB drives to a primal-dual
# optimum instead of stopping at the first feasible point. Unlike a feasibility
# run, this needs an explicit, *reachable* --dualityGapThreshold: the default
# (1e-30) sits below the conditioning-limited duality-gap floor (~1e-12 at
# 768-bit), so SDPB would never terminate and would instead grind until its
# Schur-complement Cholesky hits a sub-precision negative pivot and MPI_ABORTs.
# The objective is stable to ~12 digits well before that, so 1e-10 stops cleanly
# at `optimal` with a ~10-digit bound. (Raise --precision for more digits, but
# also keep the threshold above that precision's gap floor.)
SDPB_OPTIONS: dict[str, str | bool | int] = {
    "--precision": 768,
    # "--mpiProcs": 4,
    "--writeSolution": "x,y",
    "--maxIterations": 1000,
    "--noFinalCheckpoint": True,
    "--dualityGapThreshold": "1e-10",
}


def main() -> None:
    mp.mp.dps = 120
    # the inputs go to the library as exact labels (a decimal rounded to a binary float could move a deliberately
    # nearby c onto a singular one); their mpf values serve display and arithmetic
    delta_star_label = sys.argv[1] if len(sys.argv) > 1 else "0.8"
    c_label = sys.argv[2] if len(sys.argv) > 2 else "4"
    delta_star, c = mp.mpf(delta_star_label), mp.mpf(c_label)
    spins = tuple(
        int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0,2,4,6,8,10").split(",")
    )

    h_ext = mp.mpf("0.0625")  # Delta_ext = 2*h_ext = 1/8
    lmax = max(spins) + 2
    twists = tuple(double_twist_set(h_ext, 5))

    # delta_gap = delta_star: the distinguished scalar sits at the bottom of the
    # spin-0 continuum, and that is also the operator whose OPE we maximize.
    spec = IdenticalScalarGapSpec(
        c=c_label,
        h_ext=h_ext,
        delta_gap=delta_star_label,
        lam=7,
        spin_list=spins,
        q_order=40,
        dps=120,
        npoles=12,
        pole_selection=DEFAULT_POLE_SELECTION,
        large_spin_lmin=lmax,
        large_spin_twists=twists,
        large_spin_pole_selection=DEFAULT_LARGE_SPIN_POLE_SELECTION,
        num_threads=4,
    )
    problem = build_identical_scalar_ope_max_problem(
        spec, delta_star=delta_star_label, spin_star=0
    )

    out_dir = dated_output_dir("ope_max_l7") / f"c_{mp.nstr(c, 5)}__dstar_{mp.nstr(delta_star, 5)}"
    run = run_identical_scalar_gap_problem_in_docker(
        problem,
        out_dir,
        sdpb_options=SDPB_OPTIONS,
        pmp2sdp_log_name="pmp2sdp.log",
        sdpb_log_name="sdpb.log",
    )

    print(f"c={mp.nstr(c, 5)}  Lambda=7  Delta_ext=1/8  spins={spins}")
    print(f"distinguished scalar at Delta* = {mp.nstr(delta_star, 6)} (spin 0)")
    print(f"SDPB status: {run.sdpb_status}")

    obj = read_sdpb_objective(run.sdpb_out_dir) if run.sdpb_out_dir else None
    if run.sdpb_status == "optimal" and obj and "dual" in obj:
        bound = ope_squared_bound_from_objective(obj["dual"])
        print(f"-> max OPE^2 bound = {mp.nstr(bound, 8)}")
        # (the library's scalar block is 16^-Delta times the standard one, at the default chiral_norm 16)
        print(f"   (16^Delta* lambda^2; standard lambda^2 <= {mp.nstr(bound / mp.mpf(16) ** delta_star, 8)})")
    elif run.sdpb_status == "max_complementarity":
        print("-> objective ran away (max_complementarity): delta_star is above the")
        print("   scalar gap bound Delta*(0); no finite OPE bound here.")
    else:
        print(f"-> INCONCLUSIVE ({run.sdpb_status})")

    print(f"artifacts in {out_dir}")


if __name__ == "__main__":
    main()
