# virasoro_blocks

`virasoro_blocks` computes two-dimensional conformal blocks as high-precision
derivative tables for the numerical conformal bootstrap, and turns them into
SDPB problems.  The sphere Virasoro blocks use Zamolodchikov's elliptic `h`-recursion:
each derivative of a block at the crossing-symmetric point is written as a
polynomial in the exchanged dimension times a damped-rational prefactor, with
the number of retained poles reduced by optimal-node interpolation. Output is JSON or
`scalar_blocks`-style Mathematica tables, or a bootstrap problem solved
directly with SDPB.

## What's in the box

- **`virasoro_blocks`**: the block engine and the `virasoro_blocks` command
  (`evaluate`, `table`, `point-table`, `info`).
  - Sphere four-point Virasoro blocks at any central charge, including the
    recursion-singular values (`c = 1`, `c = 25`, minimal models), which are
    regularized automatically.
  - Global SL(2)×SL(2) blocks, N=1 superconformal NS and Ramond blocks,
    torus character tables, and torus one-point blocks at any real central
    charge, including interpolation across positive Kac poles; point tables
    for every family, with the irreducible torus characters of degenerate
    modules.
  - Scalar, ordered (with scalar or spinning externals), and parity-projected
    bases, chiral halves, off-diagonal expansion points, and large-spin towers.
- **`virasoro_bootstrap`**: the bootstrap layer for identical scalars.
  - Scalar-gap bounds, OPE-coefficient bounds, and bounds on moments of the
    OPE spectral density, plus the modular (torus) gap bootstrap.
  - SDPB through Docker, Apptainer, or a local build.
  - Certification of SDPB's dual functional against exact blocks, with
    extra spins and twists.
  - Self-describing run manifests.
- **Cluster tools**: `vb-submit`, `vb-status`, `vb-collect`, `vb-list`, and
  `vb-certify` run sweeps of bootstrap problems on Slurm or locally
  ([cluster guide](docs/CLUSTER.md)).
- **Data**: the computed moments of known CFTs shown in the accompanying paper,
  in [data/known_cft_moments](data/known_cft_moments/).

## Requirements

- Linux or macOS (the code is tested on both).
- Python 3.10 or later.
- `mpmath`, `numpy`, and `python-flint` (0.7 or later), installed
  automatically.  `python-flint` ships binary wheels for Linux (glibc) and
  macOS. On musl-based systems such as Alpine, pip builds it from source,
  which needs the FLINT C library. Numerically intensive calculations use its
  ball arithmetic (Arb).
- For bounds, [SDPB](https://github.com/davidsd/sdpb) 3.x.  The simplest way
  to get it is Docker with the image `bootstrapcollaboration/sdpb:master`
  (`docker pull bootstrapcollaboration/sdpb:master`).  A native
  `sdpb`/`pmp2sdp` on `PATH`, or an Apptainer image, also works; see
  [Pointing at your SDPB](docs/CLUSTER.md#pointing-at-your-sdpb).

## Install

Clone the repository and install it in a virtual environment:

```sh
git clone https://github.com/davidpoland/virasoro_blocks.git
cd virasoro_blocks
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
virasoro_blocks --version
```

The packages live in `src/`, so run the code from an installed copy.  An
editable install (`-e`) picks up edits to the checkout.  Optional extras:

```sh
python -m pip install -e ".[test]"   # pytest, to run the test suite
python -m pip install -e ".[plot]"   # matplotlib, for the interpolation-error plots
```

The bootstrap and cluster layers need no extra Python packages.


## A first bound

This walkthrough computes a block table from the command line, then an upper
bound on the scalar gap of a four-point function of identical scalars, and
checks the result.

### 1. A block table

```sh
virasoro_blocks table --central-charge 8 --delta-ext 0.125 0.125 0.125 0.125 \
  --spin-ranges 0:2:10 --lambda 7 --order 40 --npoles 20 --precision 120 \
  -o tables/
```

This takes a few seconds.  It writes one file per spin,
`tables/derivs_0.json` to `tables/derivs_10.json`, and prints one note per
spin:

```text
note: spin 0: the order-reduced table (--npoles 20) reproduces the block to 10.2 digits of --precision 120 over x = Delta - shift in [0, 1.33e+4] (between its nodes 15.8; worst at x = 1.33e+4, past the last node 10.4); the order reduction limits it (raise --npoles, or omit it to keep every pole).
```

An integer `--npoles` makes an order-reduced table: an approximation with a
low polynomial degree, which keeps SDPB well conditioned.  The note says how
closely it matches the full block, between its nodes and past them, where SDPB
still imposes positivity.  `--precision` is the working precision: every table
is built and written at that many digits, then verified. Any shortfall is reported
with its cause.

For Virasoro, NS, and Ramond tables, leaving out `--npoles` gives the
`--order`-truncated block itself.  Its high-degree monomial form loses digits
to rounding, so pass `--target-digits D` to have the working precision raised
until the table delivers `D` verified digits.  Global tables, torus one-point
tables, and large-spin towers instead choose the pole count when `--npoles`
is omitted.
[Accuracy presets](docs/CLI.md#accuracy-presets) lists tested combinations,
and [the JSON schema](docs/CLI.md#json-table-schema) documents the file.
`virasoro_blocks info` lists the families, bases, and coordinates.

### 2. A gap bound

The script imposes positivity at even spins 0 to 10, and on large-spin towers
covering every spin from 12 up, with a scalar gap `Delta_gap`.  Bisection then
finds the largest allowed gap for external dimension `Delta_phi = 1/8` at
`c = 8` and derivative order `Lambda = 7`.  Save it as `first_bound.py`:

```python
import mpmath as mp

from virasoro_bootstrap import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
    certify_dual_from_sdpb_output,
    double_twist_set,
    prepare_identical_scalar_gap_build,
    run_identical_scalar_gap_problem,
    worst_certification_detail,
)

H_EXT = mp.mpf(1) / 16            # identical scalars, Delta_phi = 2 h_ext = 1/8
SDPB_OPTIONS = {
    "--precision": 768,            # SDPB bits
    "--findPrimalFeasible": True,  # stop at the first allowed / excluded verdict
    "--findDualFeasible": True,
    "--detectPrimalFeasibleJump": True,  # ... or at a feasible jump
    "--detectDualFeasibleJump": True,
    "--writeSolution": "x,y",      # y.txt holds the dual functional
}


def spec(gap):
    return IdenticalScalarGapSpec(
        c=8, h_ext=H_EXT, delta_gap=gap, lam=7,
        spin_list=(0, 2, 4, 6, 8, 10),          # positivity at these spins ...
        large_spin_lmin=12,                      # ... and for all spins L >= 12
        large_spin_twists=tuple(double_twist_set(H_EXT, 5)),
        q_order=40, npoles=20, dps=120,       # the Lambda = 7 preset
    )


def main():
    mp.mp.dps = 120
    prepared = prepare_identical_scalar_gap_build(spec(1))  # the blocks, built once
    checkpoint = None   # each solve starts from the previous one's final checkpoint

    def probe(gap):
        nonlocal checkpoint
        problem = build_identical_scalar_gap_problem(spec(gap), prepared=prepared)
        run = run_identical_scalar_gap_problem(
            problem, f"first_bound/gap_{mp.nstr(gap, 6)}", sdpb_options=SDPB_OPTIONS,
            initial_checkpoint_dir=checkpoint)
        checkpoint = run.checkpoint_dir or checkpoint
        print(f"  Delta_gap = {mp.nstr(gap, 6):8s} {run.sdpb_status}")
        if run.sdpb_status not in ("primal_feasible", "dual_feasible"):
            raise RuntimeError(f"inconclusive solve at gap {gap}: {run.sdpb_status}")
        return problem, run

    lo, hi = mp.mpf("0.5"), mp.mpf(3)   # candidate bracket; check both endpoints
    _, run = probe(lo)
    if run.sdpb_status != "primal_feasible":
        raise RuntimeError("lower endpoint is excluded; lower lo and retry")
    excluded = probe(hi)
    if excluded[1].sdpb_status != "dual_feasible":
        raise RuntimeError("upper endpoint is allowed; raise hi and retry")
    while hi - lo > 0.01:
        mid = (lo + hi) / 2
        problem, run = probe(mid)
        if run.sdpb_status == "primal_feasible":
            lo = mid
        else:
            hi, excluded = mid, (problem, run)
    print(f"scalar gap bound: Delta_gap < {mp.nstr(hi, 4)} (allowed at {mp.nstr(lo, 4)})")

    # Check SDPB's functional on exact blocks, 8 more spins, and twists
    # between the tower slices.
    problem, run = excluded
    ok, details = certify_dual_from_sdpb_output(
        problem, run.sdpb_out_dir, extra_spins=8, extra_twists=2)
    worst = worst_certification_detail(details)
    print(f"certified: {ok} ({len(details)} channels checked; "
          f"worst channel minimum {float(worst['min_value_text']):.2e})")


if __name__ == "__main__":
    main()
```

Run it with `python first_bound.py`.  With Docker it prints:

```text
  Delta_gap = 0.5      primal_feasible
  Delta_gap = 3.0      dual_feasible
  Delta_gap = 1.75     dual_feasible
  Delta_gap = 1.125    dual_feasible
  Delta_gap = 0.8125   primal_feasible
  Delta_gap = 0.96875  primal_feasible
  Delta_gap = 1.04688  dual_feasible
  Delta_gap = 1.00781  dual_feasible
  Delta_gap = 0.988281 primal_feasible
  Delta_gap = 0.998047 primal_feasible
scalar gap bound: Delta_gap < 1.008 (allowed at 0.998)
certified: True (29 channels checked; worst channel minimum 3.08e-34)
```

The ten SDPB solves account for most of the runtime; each uses one MPI rank
with the default Docker setup and starts from the previous solve's checkpoint. The block tables are built once and reused at
every gap. Raising `lam` tightens the bound, with `q_order`, `npoles`, and `dps`
raised to match (see the presets).

The block build uses up to four worker processes, so a script must keep its
work under `if __name__ == "__main__":`.  Without that guard it runs
serially and warns.  `IdenticalScalarGapSpec` documents every field
(`help(IdenticalScalarGapSpec)`).

### 3. Reading the result

`run.sdpb_status` classifies the solve:

- `primal_feasible`: SDPB found a solution to crossing, so the gap is allowed.
- `dual_feasible`: SDPB found a functional that excludes the gap.
- `optimal`: an optimization problem (OPE or moment bound) converged.
- `max_complementarity`, `max_iterations`, `max_runtime`: the solve reached
  that SDPB limit without a verdict.
- `primal_step_too_small`, `dual_step_too_small`: SDPB stalled without a verdict.
- `solver_error`: SDPB (or the runner around it) failed.
- `None`: the log names no terminate reason, or it was streamed to the
  terminal (`sdpb_log_name=None`).

`run.sdpb_summary` is SDPB's terminate line, or for `solver_error` the exit
code and the log's last error line (for example a stopped Docker daemon).

Each probe's directory holds the whole problem, so it can be reproduced or
rebuilt:

```text
first_bound/gap_1.00781/
  manifest.json   the spec, block hashes, and solver settings
  blocks/         the block tables the problem was built from
  pmp.json        the polynomial matrix program
  sdp/            the SDPB input
  pmp2sdp.log     how the SDPB input was written
  sdpb.log        the solver log
  sdpb_out/       out.txt (status, objectives), y.txt (the functional), x_*.txt
```

For OPE and moment bounds, `read_sdpb_objective(run.sdpb_out_dir)` returns
the primal and dual objectives.  Convert them to a bound with
`ope_squared_bound_from_objective` or the `moment_*_from_objective` helpers,
all importable from `virasoro_bootstrap` (see [examples](examples/)).

### 4. Certifying

SDPB's verdict rests on the interpolated blocks at the spins in `spin_list`.
`certify_dual_from_sdpb_output` reads the functional from `y.txt` and checks
that it is nonnegative:

- on the exact blocks (numeric recursion, and the exact large-spin tower)
  rather than their interpolants;
- on `extra_spins` more even spins beyond `spin_list`;
- on `extra_twists` intermediate twists between the tower slices.

Each channel is sampled on `n_test_points` (default 100) evenly spaced points
from its lower edge to `delta_range` (default 30) above it, in Δ for a fixed
spin and in the spin L for a tower slice.  At c > 1 the range widens to
(c − 1)/4 + 10 when that is larger.  Positivity beyond the range, the
asymptotic tail, is not checked, nor between the grid points.

It returns `(ok, details)`, with one record per channel (`ell`, `twist`,
`passed`, `min_value`, `min_delta`, and `min_spin` for a tower slice).
`min_value` is a float when the double range holds it and a decimal string
otherwise.  `min_value_text` is always the string, and
`worst_certification_detail(details)` prioritizes failed channels, then the
smallest functional value, treating missing or nonfinite minima as worst.
It returns `None` for empty details. Treat an exclusion as established only
once it is certified.

The certification also checks that `q_order` is high enough for the blocks
and warns if not (`q_gate="strict"` raises instead).  `vb-certify` does the
same for cluster runs.

## Python API

The public import paths are `virasoro_blocks.api` (block requests,
evaluation, tables, and their serialization) and the `virasoro_bootstrap`
package (problem specs and builders, the SDPB runner, result readers, and
certification).  `virasoro_bootstrap.api` is its core subset.  See
[the API reference](docs/API.md).

```python
from virasoro_blocks.api import BlockRequest, evaluate_block_request
from virasoro_bootstrap import IdenticalScalarGapSpec, run_identical_scalar_gap_problem
```

## Documentation

- [CLI reference](docs/CLI.md): commands, flags, accuracy presets, the JSON
  table schema.
- [Python API](docs/API.md): public import paths and supported functions.
- [Architecture](docs/ARCHITECTURE.md): package boundaries.
- [Cluster runs](docs/CLUSTER.md): sweeps, SDPB modes, and the `vb-*` tools.
- [Examples](examples/): runnable bootstrap scripts and sample tables.

## Testing

```sh
python -m pip install -e ".[test]"
pytest -q
```

`pytest -q` runs in parallel and skips the slow tier; the summary reports what
it skipped.  Run `pytest -q --slow` before a release or after changing the
recursion or precision code.  [tests/README.md](tests/README.md) describes the
tiers.  `tools/check_public_wheel.sh` builds the wheel and checks that it
installs and runs outside the checkout.


## Citing

If you use this software in published work, please cite the accompanying
paper, *virasoro_blocks: Software for the 2d bootstrap* (D. Poland,
[arXiv:2610.39106](https://arxiv.org/abs/2610.39106)).
[CITATION.cff](CITATION.cff) has the metadata.  Please also cite
[SDPB](https://github.com/davidsd/sdpb) for the bounds.

## Author

David Poland (david.poland@yale.edu)

## License

MIT.  See [LICENSE](LICENSE).
