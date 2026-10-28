# Examples

Runnable bootstrap scripts and committed sample block tables. Every
script needs SDPB; see [Requirements](../README.md#requirements).  Docker with
`bootstrapcollaboration/sdpb:master` works out of the box, and
`VB_SDPB_MODE` selects Apptainer or a native build
([cluster guide](../docs/CLUSTER.md#pointing-at-your-sdpb)).  Start with the
[walkthrough in the README](../README.md#a-first-bound).

Four of the scripts bound four-point functions of identical scalars with
`Delta_phi = 1/8`.  They use derivative order `Lambda = 7`, fixed spins 0 to
10, and large-spin towers starting at spin 12. The fifth works on the torus
partition function.

| script | what it computes | runtime |
|---|---|---|
| `feasibility_scalar_gap.py [GAP] [SPINS]` | one allowed/excluded verdict for a scalar gap at `c = 32`, read from SDPB's status | 10 s |
| `feasibility_scalar_gap_certified.py [GAP] [SPINS]` | the same, and an exclusion is certified on exact blocks with 8 extra spins and 2 extra twists | 10 s (allowed, the default `GAP = 1`); 35 s for an excluded gap such as `2` |
| `scalar_gap_binary_search.py [LO] [HI] [TOL] [SPINS]` | the scalar-gap bound by bisection, reusing one block build for every probe | 2 min (8 probes) |
| `ope_max_scalar.py [DELTA_STAR] [C] [SPINS]` | an upper bound on the squared OPE coefficient of the scalar at the bottom of the spin-0 continuum, by optimization | 1 min |
| `modular_gap_bound_vs_c.py [--lambda N] [--c C ...] ...` | the modular-bootstrap bound on the gap, `Lambda = 11`, spins 0 to 20, by bisection at each `c` | 2.5 min per `c`; the default four values take about 10 min |

Runtimes are wall-clock times measured on a busy 8-core laptop with Docker
SDPB on one MPI rank.  With the defaults, the scripts print:

- `feasibility_scalar_gap.py`: `SDPB status: primal_feasible`,
  `-> ALLOWED (primal feasible)`.
- `feasibility_scalar_gap_certified.py 2.0`: `-> EXCLUDED (dual feasible,
  CERTIFIED on exact blocks +8 extra spins, +2 extra twists)`.
- `scalar_gap_binary_search.py`: `-> Delta*(0) in [0.96875, 1.0078125]` (at `c = 4`).
- `ope_max_scalar.py`: `SDPB status: optimal`, `-> max OPE^2 bound = 2.7323205`
  (at `c = 4`, `Delta* = 0.8`).  The bound is on the coefficient `p` of the
  library's scalar block, which is `16^Delta* lambda^2` in terms of the squared
  OPE coefficient `lambda^2` of the standard normalization
  ([block normalization](../docs/CLI.md#--chiral-norm-n)).  The script prints
  that conversion on the next line, `standard lambda^2 <= 0.29732789`.
- `modular_gap_bound_vs_c.py --c 8`: `c=     8: Delta_1 <= 1.5473796`.  The
  E8 lattice theory has gap 1 at `c = 8`, so no bound can fall below 1.

Output goes under `artifacts/` in the working directory:
`artifacts/experiments/<date>/...` for the four-point scripts, and
`artifacts/runs/<timestamp>-modular_gap_bound_l<Lambda>/` for the modular one
(or `--out DIR`).  Set `VB_EXPERIMENT_ROOT` or `VB_ARTIFACT_ROOT` to move it.
Each probe directory holds the manifest, the block tables, the SDPB input, and
the solver output, as described in the
[walkthrough](../README.md#3-reading-the-result).

The four-point scripts parse their arguments positionally.  `SPINS` is a
comma-separated list of even spins (`0,2,4,6,8,10`). The SDPB options are defined at
the top of each script as one dictionary keyed by SDPB flag.

## Block-table samples

[block_samples/](block_samples/) holds small committed tables for representative
sphere and torus bases, as references for the JSON and Mathematica formats. The
[CLI reference](../docs/CLI.md#json-table-schema) documents the JSON keys.
