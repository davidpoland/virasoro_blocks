"""Scalar-gap bound via binary search at Lambda=7, c=4.

Locates the critical scalar (spin-0) gap Delta*(0): the largest lower edge that
is still allowed. Gaps below Delta* are primal feasible (ALLOWED); gaps above
are dual feasible (EXCLUDED). Feasibility is monotonic in the gap, so a bisection
between a known-allowed lower bracket and a known-excluded upper bracket converges
on the bound.

The gap-independent block tables are built once via
``prepare_identical_scalar_gap_build`` and reused for every probe, so each
bisection step only re-assembles the cheap gap-dependent part and re-runs SDPB.
Each solve is hotstarted from the previous probe's final SDPB checkpoint, which
is valid because only the gap changes between probes.

Usage:
    python examples/scalar_gap_binary_search.py [LO] [HI] [TOL] [SPINS]

    LO     lower bracket, expected ALLOWED   (default 0.5)
    HI     upper bracket, expected EXCLUDED   (default 3.0)
    TOL    stop when HI - LO < TOL            (default 0.05)
    SPINS  comma-separated even spins         (default 0,2,4,6,8,10)

Requires SDPB through Docker, Apptainer, or a native installation. Set
``VB_SDPB_MODE`` to select the runner; see ``docs/CLUSTER.md``.
"""
from __future__ import annotations

import sys

import mpmath as mp

from virasoro_bootstrap.defaults import (
    DEFAULT_LARGE_SPIN_POLE_SELECTION,
    DEFAULT_POLE_SELECTION,
)
from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
    prepare_identical_scalar_gap_build,
    run_identical_scalar_gap_problem_in_docker,
)
from virasoro_bootstrap.large_spin import double_twist_set
from virasoro_bootstrap.paths import dated_output_dir

# --- SDPB solver options -----------------------------------------------------
# See feasibility_scalar_gap.py for the full option reference. One dict holds
# every solver knob: --precision sets SDPB's bit precision (the SDP data use spec.dps),
# --mpiProcs wraps sdpb in `mpirun -n N` (omit for a single rank), and the rest
# are sdpb flags. A binary search only needs the allowed/excluded verdict, so
# stop at the first feasible point on either side or at a primal or dual
# feasible jump. The final checkpoint is kept for hotstarting the next probe.
SDPB_OPTIONS: dict[str, str | bool | int] = {
    "--precision": 768,
    # "--mpiProcs": 4,
    "--findPrimalFeasible": True,
    "--findDualFeasible": True,
    "--detectPrimalFeasibleJump": True,
    "--detectDualFeasibleJump": True,
    "--writeSolution": "x,y",
}


def main() -> None:
    mp.mp.dps = 120     # before parsing the inputs, so that they keep their digits
    gap_lo = mp.mpf(sys.argv[1]) if len(sys.argv) > 1 else mp.mpf("0.5")
    gap_hi = mp.mpf(sys.argv[2]) if len(sys.argv) > 2 else mp.mpf("3.0")
    tol = mp.mpf(sys.argv[3]) if len(sys.argv) > 3 else mp.mpf("0.05")
    spins = tuple(
        int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else "0,2,4,6,8,10").split(",")
    )
    if not (mp.isfinite(tol) and tol > 0):
        raise SystemExit(f"TOL must be finite and positive, got {sys.argv[3]}")
    if not gap_lo < gap_hi:
        raise SystemExit(f"need LO < HI, got [{mp.nstr(gap_lo, 8)}, {mp.nstr(gap_hi, 8)}]")
    h_ext = mp.mpf("0.0625")  # Delta_ext = 2*h_ext = 1/8
    # Large-spin tower: continuous-in-L positivity for L >= Lmax at each
    # double-twist slice, picking up where the discrete fixed spins stop.
    lmax = max(spins) + 2
    twists = tuple(double_twist_set(h_ext, 5))

    def make_spec(gap: mp.mpf) -> IdenticalScalarGapSpec:
        return IdenticalScalarGapSpec(
            c=mp.mpf(4),
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

    # Build the gap-independent block tables once; every probe reuses them.
    prepared = prepare_identical_scalar_gap_build(make_spec(gap_lo))

    out_root = dated_output_dir("binary_search_l7_c4")
    # keyed by the exact gap (a rounded label could merge nearby probes), each probe in its own directory
    cache: dict[mp.mpf, bool] = {}
    checkpoint = None   # the previous probe's final checkpoint, which seeds the next solve

    def is_allowed(gap: mp.mpf) -> bool:
        """True if the gap is primal feasible (ALLOWED), False if excluded."""
        nonlocal checkpoint
        if gap not in cache:
            probe_dir = out_root / f"probe{len(cache):02d}_gap_{mp.nstr(gap, 6)}"
            problem = build_identical_scalar_gap_problem(make_spec(gap), prepared=prepared)
            run = run_identical_scalar_gap_problem_in_docker(
                problem,
                probe_dir,
                sdpb_options=SDPB_OPTIONS,
                pmp2sdp_log_name="pmp2sdp.log",
                sdpb_log_name="sdpb.log",
                initial_checkpoint_dir=checkpoint,
            )
            checkpoint = run.checkpoint_dir or checkpoint
            if run.sdpb_status not in ("primal_feasible", "dual_feasible"):
                # an inconclusive solve (iteration limit, solver error) is neither verdict: counting it as an
                # exclusion would silently move the bound
                raise SystemExit(f"gap={mp.nstr(gap, 6)}: SDPB status {run.sdpb_status!r} is inconclusive; see "
                                 f"{probe_dir}")
            allowed = run.sdpb_status == "primal_feasible"
            cache[gap] = allowed
            verdict = "ALLOWED" if allowed else f"EXCLUDED ({run.sdpb_status})"
            print(f"  gap={mp.nstr(gap, 6)}  ->  {verdict}")
        return cache[gap]

    print(f"c=4  Lambda=7  Delta_ext=1/8  spins={spins}")
    print(f"large-spin tower: Lmax={lmax}  twists={[mp.nstr(t, 4) for t in twists]}")
    print(f"binary search: [{mp.nstr(gap_lo, 5)}, {mp.nstr(gap_hi, 5)}]  tol={mp.nstr(tol, 3)}")

    # Verify the initial bracket: lower allowed, upper excluded.
    if not is_allowed(gap_lo):
        print(f"-> lower bracket {mp.nstr(gap_lo, 5)} is already EXCLUDED; lower LO and retry")
        return
    if is_allowed(gap_hi):
        print(f"-> upper bracket {mp.nstr(gap_hi, 5)} is still ALLOWED; raise HI and retry")
        return

    lo, hi = gap_lo, gap_hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if not lo < mid < hi:
            raise SystemExit(f"TOL {mp.nstr(tol, 3)} is below the resolution of the working precision")
        if is_allowed(mid):
            lo = mid
        else:
            hi = mid

    print(f"-> Delta*(0) in [{mp.nstr(lo, 8)}, {mp.nstr(hi, 8)}]  (width {mp.nstr(hi - lo, 3)})")
    print(f"artifacts in {out_root}")


if __name__ == "__main__":
    main()
