"""Genus-1 modular bootstrap: gap bound Delta_1(c) via bisection.

For each central charge, bisect the uniform gap (Delta >= max(gap, |j|) in every
integer-spin sector) between an ALLOWED value (SDPB primal-feasible) and an
EXCLUDED value (dual-feasible). The exact character tables are built once
per (c, lambda, spins) and re-anchored per gap probe, so each bisection step
costs one SDPB solve (~seconds at lambda <= 19), hotstarted from the previous
probe's final checkpoint when the two SDPs have the same block structure (the
number of sectors above the gap changes with the gap, and SDPB refuses a
checkpoint of a different shape).

Anchors to check against:
  * c = 8: the E8 level-1 lattice CFT has gap Delta = 1 (the 248 currents at
    j = 1), so no gap below 1 is ever excluded; the converged bound has a
    kink AT 1 (Collier-Lin-Yin 1608.06241). Finite-lambda bounds approach it
    from above.
  * Hellerman: at low derivative order the bound sits near c/6 + 0.474.
  * c = 24: the Monster CFT has gap 2; converged numerics give ~2.2.

Usage:
    python examples/modular_gap_bound_vs_c.py [--lambda 11] [--spins 20]
        [--c 4 8 12 24] [--dps 100] [--precision 448] [--tol 0.005]
        [--mpi-procs N]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import mpmath as mp

from virasoro_bootstrap.modular_crossing import (
    ModularGapSpec,
    build_modular_gap_problem,
    prepare_modular_gap_build,
    run_modular_gap_problem,
)
from virasoro_bootstrap.paths import runs_root

# A bisection only needs the allowed/excluded verdict; terminate on the first
# feasibility signal rather than polishing to optimality. The tiny
# --dualityGapThreshold is load-bearing: with objective == 0 the duality gap
# equals the primal objective and collapses to ~0 early, so the (first-checked)
# "optimal" criterion would fire before either feasibility check gets a look
# in (observed at lambda = 19; the jump detectors alone also miss when the
# solver converges smoothly without ever taking a full step).
SDPB_OPTIONS_TEMPLATE = {
    "--findPrimalFeasible": True,
    "--findDualFeasible": True,
    "--detectPrimalFeasibleJump": True,
    "--detectDualFeasibleJump": True,
    "--dualityGapThreshold": "1e-200",
}


def _structure(problem):
    """The SDP block shapes, which a hotstart checkpoint must match."""
    return tuple(
        (len(block["polynomials"]), len(block["polynomials"][0]), len(block["polynomials"][0][0]),
         max(len(v) for row in block["polynomials"] for entry in row for v in entry))
        for block in problem.pmp_doc["PositiveMatrixWithPrefactorArray"]
    )


def _probe(prep, spec_kwargs, gap, run_dir, sdpb_options, seed=(None, None)):
    """The SDPB status at ``gap`` and ``(checkpoint, structure)`` for the next probe.

    ``seed`` is the previous probe's ``(checkpoint, structure)``; it hotstarts this solve when the structures agree.
    """
    spec = ModularGapSpec(delta_gap=gap, **spec_kwargs)
    problem = build_modular_gap_problem(spec, prepared=prep)
    structure = _structure(problem)
    checkpoint = seed[0] if seed[1] == structure else None
    run = run_modular_gap_problem(problem, run_dir, sdpb_options=sdpb_options,
                                  initial_checkpoint_dir=checkpoint)
    status = run.sdpb_status
    if status not in ("primal_feasible", "dual_feasible"):
        raise RuntimeError(
            f"unexpected SDPB status {status!r} at gap {mp.nstr(gap, 8)} "
            f"(log: {run.sdpb_log_path})"
        )
    return status, ((run.checkpoint_dir, structure) if run.checkpoint_dir else seed)


def gap_bound(
    c,
    lam: int,
    spins: int,
    dps: int,
    tol,
    out_root: Path,
    sdpb_options,
    log=print,
):
    """Bisect the gap at central charge ``c`` (an exact label such as ``"8"`` or ``"25/2"``, passed to the
    library as given) to within ``tol``; the bracket arithmetic runs at ``dps`` digits."""
    with mp.workdps(dps):
        spec_kwargs = dict(c=c, lam=lam, spin_list=range(spins + 1), dps=dps)
        prep = prepare_modular_gap_build(
            ModularGapSpec(delta_gap=0, **spec_kwargs)
        )
        tol = mp.mpf(tol)
        if not (mp.isfinite(tol) and tol > 0):
            raise ValueError(f"tol must be finite and positive, got {tol}")

        lo = mp.mpf("0.05")  # expected ALLOWED
        hi = mp.mpf(c) / 6 + mp.mpf(2)  # expected EXCLUDED (above Hellerman-ish)
        n_run = 0
        seed = (None, None)

        def probe(gap):
            nonlocal n_run, seed
            n_run += 1
            run_dir = out_root / f"c{c}_probe{n_run:03d}"
            status, seed = _probe(prep, spec_kwargs, gap, run_dir, sdpb_options, seed)
            verdict = "ALLOWED" if status == "primal_feasible" else "EXCLUDED"
            log(f"  gap={mp.nstr(gap, 8)}  ->  {verdict}")
            return status

        if probe(lo) != "primal_feasible":
            raise RuntimeError(f"lower bracket gap={mp.nstr(lo, 6)} not allowed")
        while probe(hi) != "dual_feasible":
            lo = hi
            hi = hi * mp.mpf("1.5")
            if hi > mp.mpf(c) * 10:
                raise RuntimeError("no excluded gap found; check the setup")

        while hi - lo > tol:
            mid = (lo + hi) / 2
            if not lo < mid < hi:
                raise ValueError(f"tol {mp.nstr(tol, 3)} is below the resolution of dps={dps}")
            if probe(mid) == "primal_feasible":
                lo = mid
            else:
                hi = mid
        return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lambda", dest="lam", type=int, default=11)
    parser.add_argument("--spins", type=int, default=20,
                        help="max spin sector (all integer spins 0..N)")
    parser.add_argument("--c", nargs="+", type=str, default=["4", "8", "12", "24"])
    parser.add_argument("--dps", type=int, default=100)
    parser.add_argument("--precision", type=int, default=448)
    parser.add_argument("--tol", type=str, default="0.005")
    parser.add_argument("--mpi-procs", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out) if args.out else (
        runs_root() / f"{stamp}-modular_gap_bound_l{args.lam}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sdpb_options = dict(SDPB_OPTIONS_TEMPLATE)
    sdpb_options["--precision"] = args.precision
    if args.mpi_procs:
        sdpb_options["--mpiProcs"] = args.mpi_procs

    print(
        f"modular gap bound: lambda={args.lam}, spins 0..{args.spins}, "
        f"dps={args.dps}, precision={args.precision}, tol={args.tol}"
    )
    results = []
    for c_str in args.c:
        print(f"c = {c_str}:")
        lo, hi = gap_bound(
            c_str, args.lam, args.spins, args.dps, args.tol, out_root, sdpb_options
        )
        with mp.workdps(args.dps):
            bound = (lo + hi) / 2
            reference = mp.mpf(c_str) / 6 + mp.mpf("0.474")
        results.append((c_str, lo, hi, bound))
        print(
            f"  => Delta_1 bound in [{mp.nstr(lo, 8)}, {mp.nstr(hi, 8)}]"
            f"  (c/6 + 0.474 = {mp.nstr(reference, 6)})"
        )

    print("\nsummary (lambda=%d, spins<=%d):" % (args.lam, args.spins))
    for c_str, lo, hi, bound in results:
        print(f"  c={c_str:>6}: Delta_1 <= {mp.nstr(hi, 8)}")


if __name__ == "__main__":
    main()
