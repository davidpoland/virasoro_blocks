# Running sweeps on your own cluster

The `virasoro_bootstrap.hpc` layer expands a sweep of bootstrap problems into
one job spec per task and runs them through Slurm or in-process. You need two
small TOML files, and neither has to live in this repository.

## The two files

A **sweep** describes the problems:

```toml
name = "gap_vs_c"
task = "scalar_gap_bisection"      # one of the tasks listed at the end of this page

[resources]              # sbatch sizing
cpus_per_task = 8
time = "02:00:00"

[params]                 # shared by every task
lam = 7
gap_lo = "0.5"
gap_hi = "3.0"
tol = "0.05"

[sweep]                  # every list-valued key is swept (Cartesian product)
c = ["2", "4", "8", "16", "32"]
```

A **cluster profile** describes the machine:

```toml
kind = "slurm"                     # or "local" to run in-process
partition = "compute"
runs_root = "~/vb_runs"            # where run directories are written
python = "~/.conda/envs/vb/bin/python"

modules = ["gcc/12", "openmpi"]    # module load, in order, before each job
setup = ["source ~/vb/bin/activate"]   # arbitrary shell lines, after modules

[env]                              # exported into each job
VB_SDPB_MODE = "apptainer"
VB_SDPB_IMAGE = "~/images/sdpb.sif"
```

## Submit a sweep

Save the examples as `sweep.toml` and `mycluster.toml` and adjust the profile's
partition, paths, modules, and setup commands. Install the package in the
environment named by `python`, and configure [SDPB](#pointing-at-your-sdpb).
Then run this command on the cluster:

```sh
vb-submit sweep.toml --cluster mycluster.toml
```

`--dry-run` validates the configuration and writes the job files without
submitting anything. A profile with `kind = "local"` runs the same specs
in-process, which is the easiest way to try a small sweep before it reaches a
scheduler. Local mode uses the interpreter running `vb-submit` and applies
the profile's `[env]`. Activate the intended environment and load any modules
in your shell first; the profile's `python`, `modules`, and `setup` configure
Slurm jobs and are not executed in local mode.

## Sweep parameters

`[params]` and `[sweep]` keys are the task's parameters. The tables at the end
of this page list them per task, as does `vb-submit --list-tasks [TASK]`.

Use top-level `[[sweep_rows]]` entries for settings that must vary together.
For example, append these rows to the sweep above:

```toml
[[sweep_rows]]
q_order = 60
npoles = 20

[[sweep_rows]]
q_order = 80
npoles = 60
```

Each row overrides `[params]` and is crossed with every `[sweep]` combination.
These two pairs and the five central charges above produce ten tasks. Keep
correlated keys in the rows; a key also present in `[sweep]` takes its value
from `[sweep]`.

Numbers keep their written value. Decimals such as `0.7` mean `7/10` exactly,
even when written with more digits than a float holds, and rationals are
written as strings (`"8/7"`). `lam` must be one of the packaged derivative
orders (7, 11, 15, 19), whose settings come from the `default_params.json`
shipped in `virasoro_bootstrap`. To use your own profiles, set
`VB_DEFAULT_PARAMS` to a complete replacement file, both where you submit and,
through the profile's `[env]`, in the jobs.

The optional `[resources]` table sizes each array element:

| Key | Meaning | Default |
|---|---|---|
| `cpus_per_task` | CPUs per task (`#SBATCH --cpus-per-task`) | `8` |
| `time` | wall time (`#SBATCH --time`) | `"23:00:00"` |
| `mem_per_cpu` | memory per CPU (`#SBATCH --mem-per-cpu`) | `"4G"` |
| `partition` | the Slurm partition, overriding the profile's | the profile's |

A `[params.sdpb_options]` table is merged flag by flag over the task's default
SDPB flags. The task tables name the default set, defined in
[`hpc/tasks.py`](../src/virasoro_bootstrap/hpc/tasks.py):

| Default set | `--precision` (bits) | Other flags |
|---|---|---|
| `DEFAULT_FEASIBILITY_SDPB_OPTIONS`, `DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS` | `768` | `--findPrimalFeasible`, `--findDualFeasible`, `--detectPrimalFeasibleJump`, `--detectDualFeasibleJump`, `--dualityGapThreshold=1e-200` |
| `DEFAULT_OPE_MAX_SDPB_OPTIONS` | `768` | `--maxIterations=1000`, `--dualityGapThreshold=1e-10` |
| Moment bounds | the task's `precision`, from the lam profile by default | `--maxIterations=1000`, `--dualityGapThreshold=1e-10` |

All four sets also request `--writeSolution=x,y` and `--noFinalCheckpoint`.
Scalar-gap bisection with `hotstart=true` and moment bounds with
`keep_checkpoint=true` remove `--noFinalCheckpoint` so a later solve can use
the saved checkpoint. Flags absent from these sets use SDPB's own defaults.

A value you write replaces that flag's default, defaults you do not mention
are kept, and `false` removes a default:

```toml
[params.sdpb_options]
"--precision" = 1024              # replaces the default 768
"--mpiProcs" = 8                  # see "MPI ranks inside a job" below
"--findPrimalFeasible" = false    # drops this default flag
```

A flag written in the equals form, such as `"--maxIterations=500"`, also
replaces that flag's default.

Bisection tasks stop with an error asking for a higher `dps` when the working
precision cannot resolve the requested tolerance.

For `scalar_ope_max_scan` with `all_spins_twist_gap = true`, the scanned value
is the twist `tau`, and the distinguished operator sits at the lower edge of
its channel, `Delta = spin_star + tau` (recorded as `operator_delta`).

## Cluster profile

Every key is optional, and `kind` defaults to `"slurm"`. Any other key, such as
a misspelled `partiton`, is an error.

| Key | Meaning |
|---|---|
| `kind` | `"slurm"` to submit an array, `"local"` to run in-process |
| `partition`, `account` | passed straight through to `sbatch` |
| `runs_root` | where run directories go; required for remote submission |
| `python` | the interpreter that runs each task, as a path to `<env>/bin/python`. It needs no activation, which avoids conda's fragile shell hooks in batch scripts. The default is whatever `python` is on `PATH`, which may not have `virasoro_bootstrap` installed |
| `modules` | `module load` lines emitted before each job |
| `setup` | arbitrary shell lines emitted after the modules |
| `sbatch_extra` | extra directives, each written as `#SBATCH <string>` (for example `"--constraint=avx512"` or `"--qos=long"`) |
| `[env]` | exported into each job |
| `remote_repo` | optional checkout path for `--host` submission (see below) |
| `max_array_throttle` | cap on concurrent array elements, for a resource Slurm cannot see (e.g. shared filesystem capacity) |
| `host` | not used by the `vb-*` commands; select the remote host with `--host` |

Paths and `[env]` values expand `~` and `$VARS` on the host where the job runs.

## Pointing at your SDPB

SDPB can be reached three ways, chosen by `VB_SDPB_MODE`, either as a shell
variable locally or in the profile's `[env]` for cluster jobs.

| Mode | Needs | For |
|---|---|---|
| `native` | `sdpb` and `pmp2sdp` on `PATH`, or `VB_SDPB_BIN_DIR` pointing at their directory | a self-built or module-provided SDPB |
| `apptainer` | `VB_SDPB_IMAGE` = path to a `.sif` | most HPC sites, where Docker is unavailable |
| `docker` | Docker installed | a laptop |

With no mode selected, the runner uses Docker if it is installed and otherwise
native executables on `PATH`. An explicit mode is used without fallback. For
example, a self-built SDPB in a nonstandard location:

```toml
[env]
VB_SDPB_MODE = "native"
VB_SDPB_BIN_DIR = "$HOME/sdpb/build"
```

### MPI ranks inside a job

`"--mpiProcs" = N` in `sdpb_options` runs `sdpb` on `N` MPI ranks. It is read
by this package, not passed to SDPB. The ranks are started by the template in
`VB_MPI_LAUNCHER`, where `{n}` becomes `N`. The default is `mpirun -n {n}`, or
`mpirun -n {n} --allow-run-as-root` in Docker mode, because the upstream image
runs as root. In the container modes the launcher runs inside the image.

The CPU controls are independent: `cpus_per_task` reserves CPUs,
`num_threads` sets block-building workers for tasks that expose it, and
`--mpiProcs` sets solver ranks. A larger CPU allocation does not change either
worker count. Without `--mpiProcs`, SDPB runs on one rank.

**On Slurm, the default launcher usually needs changing.** Each array element
is one Slurm task with `cpus_per_task` CPUs. Open MPI's `mpirun` counts
allocated tasks, not CPUs, so it refuses a second rank ("not enough slots"),
in Apptainer mode as well. Allow oversubscription in the profile. For Open MPI:

```toml
[env]
VB_MPI_LAUNCHER = "mpirun --oversubscribe -n {n}"
```

Keep worker counts at or below `cpus_per_task`. `vb-submit` warns when a Slurm
sweep requests multiple MPI ranks without a launcher override, or when
`--mpiProcs` exceeds `cpus_per_task`.

## Checks before anything is queued

`vb-submit` refuses a sweep that cannot run as written and lists every problem
at once, before anything is written or copied. This catches unknown tasks and
keys, misspelled parameters (with the closest real name suggested), missing
required parameters, values the task cannot convert, a `lam` with no profile,
nonpositive tolerances, inverted gap brackets, empty sweeps, and invalid
profiles.

It also warns when the setup looks unreachable: a missing Apptainer image, no
`sdpb`/`pmp2sdp` in native mode, or a `python` that does not exist or cannot
import `virasoro_bootstrap`. The interpreter and toolchain checks are skipped
when the profile sets `modules` or `setup`, since those may only take effect
on compute nodes.

## Two ways to run

### On the cluster itself

Install the package in the environment your `python` key points at, then:

```sh
export VB_CLUSTER="$PWD/mycluster.toml"
vb-submit sweep.toml
vb-list                                  # the runs and their ids, newest first
vb-status  <run-id>
vb-collect <run-id> -o results.json
vb-certify <run-id> --all-tasks --last-probe
```

A run can be named by its run id (as printed by `vb-submit` and `vb-list`), by
its directory, or by a Slurm job id. Run ids are looked up under
`--runs-root`, else the profile's `runs_root`, else `VB_RUNS_ROOT` or
`./artifacts/runs`. The exported `VB_CLUSTER` selects the profile for every
command above; you can set it in `~/.bashrc` to keep that default across shells.

`vb-certify` rebuilds each probe's problem from its saved `manifest.json` and
block files, currently for `scalar_gap_bisection` runs. It evaluates the dual
functional on the true q-series blocks at finite test grids over dimension,
for the selected spins and tower twists. `--extra-spins N` adds even spins
beyond the solved list, `--extra-twists N` adds checks between tower slices,
and `--num-threads N` sets the certification worker count.

Choose `--task N` or
`--all-tasks`, and optionally one of `--probe NAME`, `--last-probe`, or
`--all-probes` (the default). `--last-probe` is the largest saved gap, which is
not necessarily the final bisection bracket. `VALID functional` means the
configured finite checks passed. A witnessed negative value is printed as
`NEGATIVE` and gives an `INVALID (negative channel)` verdict; a failed
normalization or nonfinite dual vector gives `INVALID functional`. Incomplete
checks give `NOT PROVEN`, and unavailable probes are reported as `SKIPPED`.
The exit status is zero only when every requested check passes.

### From your laptop, submitting over SSH

Add `--host <ssh alias>`, where the alias is one your `~/.ssh/config`
resolves:

```sh
vb-submit sweep.toml --cluster mycluster.toml --host mycluster
```

This copies both files to the remote run directory, runs the same `vb-submit`
there, and copies the resulting `manifest.json` back, so your laptop keeps a
record of every run it launched. `vb-status`, `vb-collect`, `vb-list`, and
`vb-submit --retry` accept `--host` as well.

Without `remote_repo` the remote runs its installed package. With
`remote_repo` set to a checkout on that machine, and a local Git checkout that
is clean and pushed, `--host` first runs `git fetch && git checkout <your local
commit>` there, and the manifest records that commit. Tasks import the code
from this shared checkout when they start, so leave it unchanged while the
run's jobs are queued or running.


## When a task fails

A task that raises, or that returns an error result (`"reason": "solve_error"`
or `"prepare_error:..."`), is recorded as `failed`. `vb-status` shows the
cause, for an SDPB failure the error line from `sdpb.log` (e.g. a Docker
`pull access denied`).

`vb-submit --retry <run-dir>` takes the run directory (e.g.
`<runs_root>/<run-id>`) and reruns every task that is failed, missing, or
stale (its parameters, or the defaults that fill the ones it leaves out,
changed since it ran). Completed tasks are skipped otherwise, and on Slurm a
task that is still queued or running is not resubmitted. A retry
resubmits the run's saved `job.sbatch` with the profile recorded in its
manifest, so an explicit `--cluster` must match that profile (with `--host` it
only says how to reach the cluster). To change the partition or environment,
submit the sweep again as a new run.

Tasks journal their progress, so a rerun reuses completed probes and solves
and repeats only the incomplete ones. The worker captures the defaults used
by the task and records their effective values after applying the sweep's
overrides in `progress.json` and `result.json`. Saved work with different
effective values is archived and recomputed. Changes to overridden defaults,
unused profile fields, and other `lambdaN` profiles do not trigger a rerun.
When `lam` is omitted, the check uses the task's default derivative order.
Moment tasks also compare their resolved problem, solver, and certification
settings, including `large_spin_lmin`.

Older settings formats are compared using their recorded parameter values;
their code fingerprints are ignored. Work saved before settings were recorded
is adopted, so submit a new run if its omitted parameters may have changed.
Code itself is not fingerprinted, and changes to the SDPB installation and
input file contents are not detected. When a numerical fix requires fresh
results with the same parameters, submit a new run.
Historical results remain available to `vb-collect` and `vb-status`.

Generated Slurm scripts request
`SIGUSR1` 600 seconds before the wall-time limit and enable requeue. On
`SIGUSR1` or `SIGTERM`, the worker sets a stop flag that tasks check between
solver calls. When a task stops, the worker records `interrupted` and attempts
to requeue its array element. `vb-status` reports repeated starts; if requeue
fails, use `vb-submit --retry` after checking the reported cause.

An OPE scan whose points classify
inconsistently returns `reason="inconsistent_scan"` with the individual points
kept for inspection.

## What a run leaves behind

```text
<runs_root>/<run-id>/
  manifest.json    what was submitted: sweep and profile contents, Git commit, job id
  inputs/          original files under sweep/ and cluster/ (--host submissions only)
  specs/NNNN.json  one job spec per task
  tasks/NNNN/      per-task output (result.json, status.json, progress.json)
  job.sbatch       the rendered array script
  logs/            Slurm stdout per array element (task-<array index>.out)
```

Slurm starts each job in its run directory. Profile `sbatch_extra` entries can
override the working directory and log pattern.

The manifest embeds the sweep and profile **by value**, not by path, so a run
stays self-describing even if you later edit the files it came from. A run id
names a single new directory. Use `--retry` for an existing run, or a fresh
`--run-id` for a new submission.

After a task succeeds, the worker deletes the large generated SDPB inputs
(`block_data_*.json` and `block_info_*.json`). It keeps their exact sample
constraints in one checksummed `certification_samples.json.gz` per solve, which
certification reads automatically. Block tables, PMP documents, solver outputs,
checkpoints, journals, and results are kept, as are the inputs of failed,
interrupted, and preparation-only tasks. To keep everything for debugging, set
this in the profile:

```toml
[env]
VB_KEEP_SOLVER_INPUTS = "1"
```

The same cleanup can be previewed or applied to existing runs, and a cleaned
solve's inputs can be regenerated without a block recursion or SDPB solve:

```sh
python -m virasoro_bootstrap.hpc.cleanup /path/to/vb_runs
python -m virasoro_bootstrap.hpc.cleanup /path/to/vb_runs --execute --workers 4
python -m virasoro_bootstrap.hpc.cleanup --restore /path/to/solve
```

Without `--execute` it only reports what it would delete. It skips active runs,
failed, incomplete, and stale tasks, and runs whose profile keeps inputs with
`VB_KEEP_SOLVER_INPUTS`.

Each solver output records its SDP input directory and checksums in
`vb_sdp_source.json`. Before new inputs are written, the runners remove old
solution files from the output directory, so a dual vector is only certified
against the inputs that produced it. If `--writeSolution` omits `y`, the run
has no dual solution to certify.

Set `VB_SCRATCH_ROOT` in the profile's `[env]` to place new task directories
under `<VB_SCRATCH_ROOT>/<run-id>/<task-index>`, with symlinks from `tasks/`.
An existing task directory is kept in place. Workers copy the small
`result.json` files into `results_mirror/` beside the specs, and `vb-status`,
`vb-collect`, and `vb-submit --retry` fall back on that copy if scratch purges
a task directory. Block files, dual vectors, journals, and checkpoints remain
on scratch; retain the task directories for later recertification, progress
recovery, or hotstarting.

Workers for the same output directory share a lock, so a duplicate attempt
exits without touching the active task's files and attempts to requeue itself
on Slurm.

## Tasks and their parameters

Every parameter is optional unless marked required. "From the lam profile"
means the packaged profile for that `lam` supplies it. The tables are generated
from the task implementations (`python -m virasoro_bootstrap.hpc.params`), and
`vb-submit --list-tasks TASK` prints the same list with the task's full
description.

An OPE value, the `ope_squared` a fixed-OPE task takes or the
`ope_squared_bound` an OPE task reports, is the block coefficient
`p = N^Δ λ²` of the distinguished operator, in the same normalization at
every spin (the spin-0 block is `B(h, h)`).

`moment_recertify` rechecks a solved `moment_nu1_over_nu0` result without a new
solve. Each source directory must contain its original `manifest.json` and
block files alongside `sdpb_out/`, and the requested settings must match the
saved problem.

<!-- BEGIN GENERATED TASK TABLE -->
### `isolated_relevant_scalar_degeneracy_intervals`

Refine modular allowed intervals in fixed isolated-scalar degeneracy.

| Parameter | Type | Default |
|---|---|---|
| `c` | number | required |
| `chiral_currents` | true/false | `true` |
| `degeneracy_grid` | list of numbers | required |
| `degeneracy_tol` | number | `"0.125"` |
| `delta_gap` | number | `"2"` |
| `delta_phi` | number | required |
| `dps` | integer | from the lam profile |
| `lam` | integer | `11` |
| `modular_dps` | integer | same as dps |
| `modular_spin_max` | integer | `20` |
| `sdpb_options` | table | merged over `tasks.DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS` |

### `isolated_relevant_scalar_delta_phi_intervals`

Refine the modular allowed intervals in `Delta_phi` at fixed `c`.

| Parameter | Type | Default |
|---|---|---|
| `c` | number | required |
| `chiral_currents` | true/false | `true` |
| `delta_gap` | number | `"2"` |
| `delta_phi_grid` | list of numbers | required |
| `delta_phi_tol` | number | `"0.025"` |
| `dps` | integer | from the lam profile |
| `lam` | integer | `11` |
| `modular_dps` | integer | same as dps |
| `modular_spin_max` | integer | `20` |
| `sdpb_options` | table | merged over `tasks.DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS` |

### `isolated_relevant_scalar_grid_point`

Allowed/excluded test at one `(c, Delta_phi)` grid point.

| Parameter | Type | Default |
|---|---|---|
| `backend` | string | `"sdp"` |
| `c` | number | required |
| `chiral_currents` | true/false | `true` |
| `chiral_norm` | any | - |
| `degeneracy_phi` | number | `"1"` |
| `delta_gap` | number | `"2"` |
| `delta_phi` | number | required |
| `dps` | integer | from the lam profile |
| `four_point_dps` | integer | same as dps |
| `four_point_large_spin` | true/false | `true` |
| `four_point_spins` | list of integers | `[0, 2, 4, 6, 8, 10]` |
| `lam` | integer | `11` |
| `modular_dps` | integer | same as dps |
| `modular_phi_mode` | string | `"fixed"` |
| `modular_scalar_gap_only` | true/false | `false` |
| `modular_spin_max` | integer | `20` |
| `modular_spins` | list of integers | - |
| `npoles` | integer | from the lam profile |
| `q_order` | integer | from the lam profile |
| `run_four_point` | true/false | `true` |
| `sdpb_options` | table | merged over `tasks.DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS` |

### `isolated_scalar_four_point_fixed_ope`

Feasibility test with an isolated operator's p = N^Delta lambda^2 fixed to `ope_squared`.

| Parameter | Type | Default |
|---|---|---|
| `backend` | string | `"sdp"` |
| `c` | number | required |
| `chiral_norm` | any | - |
| `delta_ext` | number | required |
| `delta_gap` | number | `"2"` |
| `delta_star` | number | `delta_ext` |
| `dps` | integer | from the lam profile |
| `four_point_large_spin` | true/false | `true` |
| `four_point_spins` | list of integers | `[0, 2, 4, 6, 8, 10]` |
| `lam` | integer | `11` |
| `large_spin_double_twists` | integer | `5` |
| `npoles` | integer | from the lam profile |
| `num_threads` | integer | `4` |
| `ope_squared` | number | required |
| `pole_selection` | string | `"contribution_optimal"` |
| `q_order` | integer | from the lam profile |
| `sdpb_options` | table | merged over `tasks.DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS` |
| `spin_star` | integer | `0` |

### `isolated_scalar_four_point_ope_bounds`

Upper/lower four-point bounds on an isolated operator's p = N^Delta lambda^2 (`ope_squared_bound`).

| Parameter | Type | Default |
|---|---|---|
| `backend` | string | `"sdp"` |
| `c` | number | required |
| `chiral_norm` | any | - |
| `delta_ext` | number | required |
| `delta_gap` | number | `"2"` |
| `delta_star` | number | `delta_ext` |
| `directions` | list | `["lower", "upper"]` |
| `dps` | integer | from the lam profile |
| `four_point_large_spin` | true/false | `true` |
| `four_point_spins` | list of integers | `[0, 2, 4, 6, 8, 10]` |
| `lam` | integer | `11` |
| `large_spin_double_twists` | integer | `5` |
| `npoles` | integer | from the lam profile |
| `num_threads` | integer | `4` |
| `pole_selection` | string | `"contribution_optimal"` |
| `q_order` | integer | from the lam profile |
| `sdpb_options` | table | merged over `tasks.DEFAULT_OPE_MAX_SDPB_OPTIONS` |
| `spin_star` | integer | `0` |

### `modular_scalar_gap_bisection`

Modular bootstrap bound on the scalar gap at one central charge.

| Parameter | Type | Default |
|---|---|---|
| `c` | number | required |
| `chiral_currents` | true/false | `true` |
| `dps` | integer | from the lam profile |
| `gap_hi` | number | `c / 6 + 2` |
| `gap_lo` | number | `"0.05"` |
| `lam` | integer | `11` |
| `modular_spin_max` | integer | `20` |
| `sdpb_options` | table | merged over `tasks.DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS` |
| `tol` | number | `"0.05"` |

### `moment_nu1_over_nu0`

nu_1/nu_0 normalized-moment upper/lower bounds at one (c, Delta_ext).

| Parameter | Type | Default |
|---|---|---|
| `adaptive_cert_extra_spins` | integer | from the lam profile |
| `adaptive_cert_extra_twists` | integer | from the lam profile |
| `all_spins_twist_gap` | true/false | `false` |
| `backend` | string | `"sdp"` |
| `c` | number | required |
| `cert_delta_range` | number | `30.0` |
| `cert_extra_spins` | integer | from the lam profile |
| `cert_extra_twists` | integer | from the lam profile |
| `cert_log_near_points` | integer | `0` |
| `cert_n_test_points` | integer | `100` |
| `delta_ext` | number | required |
| `delta_gap` | number | `"0"` |
| `directions` | list | `["lower", "upper"]` |
| `dps` | integer | from the lam profile |
| `force_exact_included` | true/false | `false` |
| `identity_modes` | list | `["include", "exclude"]` |
| `initial_checkpoint_source_dir` | any | - |
| `interstitial_twists` | integer | `0` |
| `keep_checkpoint` | true/false | `false` |
| `lam` | integer | `19` |
| `large_spin_double_twists` | integer | from the lam profile |
| `large_spin_lmin` | integer | `max(spins) + 2` |
| `large_spin_low_twist_step` | any | - |
| `large_spin_npoles` | integer | from the lam profile |
| `large_spin_pole_selection` | string | from the lam profile |
| `large_spin_twists` | any | - |
| `moment_cert_q_order` | integer | from the lam profile |
| `npoles` | integer | from the lam profile |
| `num_threads` | integer | `4` |
| `physical_moment_blocks` | true/false | `true` |
| `pole_selection` | string | from the lam profile |
| `precision` | integer | from the lam profile |
| `q_order` | integer | from the lam profile |
| `sdpb_options` | table | merged over the task's defaults |
| `shared_sample_points` | true/false | `false` |
| `spins` | list of integers | from the lam profile |
| `spot_check` | true/false | `false` |

### `moment_recertify`

Recheck a saved moment_nu1_over_nu0 result without another SDPB solve.

| Parameter | Type | Default |
|---|---|---|
| `adaptive_cert_extra_spins` | integer | from the lam profile |
| `adaptive_cert_extra_twists` | integer | from the lam profile |
| `all_spins_twist_gap` | true/false | `false` |
| `c` | number | required |
| `cert_delta_range` | number | `30.0` |
| `cert_extra_spins` | integer | from the lam profile |
| `cert_extra_twists` | integer | from the lam profile |
| `cert_log_near_points` | integer | `0` |
| `cert_n_test_points` | integer | `100` |
| `delta_ext` | number | required |
| `delta_gap` | number | `"0"` |
| `directions` | list | `["lower", "upper"]` |
| `dps` | integer | from the lam profile |
| `force_exact_included` | true/false | `false` |
| `identity_modes` | list | `["include", "exclude"]` |
| `interstitial_twists` | integer | `0` |
| `lam` | integer | `19` |
| `large_spin_double_twists` | integer | from the lam profile |
| `large_spin_lmin` | integer | `max(spins) + 2` |
| `large_spin_low_twist_step` | any | - |
| `large_spin_npoles` | integer | from the lam profile |
| `large_spin_pole_selection` | string | from the lam profile |
| `large_spin_twists` | any | - |
| `moment_cert_q_order` | integer | required |
| `npoles` | integer | from the lam profile |
| `num_threads` | integer | `4` |
| `physical_moment_blocks` | true/false | `true` |
| `pole_selection` | string | from the lam profile |
| `q_order` | integer | from the lam profile |
| `shared_sample_points` | true/false | `false` |
| `source_output_dir` | path | required |
| `spins` | list of integers | from the lam profile |
| `spot_check` | true/false | `true` |

### `scalar_gap_bisection`

Critical scalar-gap bound Delta*(0) at one central charge, by bisection.

| Parameter | Type | Default |
|---|---|---|
| `all_spins_twist_gap` | true/false | `false` |
| `backend` | string | `"sdp"` |
| `bound_direction` | string | `"upper"` |
| `c` | number | required |
| `cert_extra_spins` | integer | `0` |
| `cert_extra_twists` | integer | `0` |
| `decide_by` | string | `"status"` |
| `dps` | integer | from the lam profile |
| `gap_hi` | number | `"3.0"` |
| `gap_lo` | number | `"0.5"` |
| `h_ext` | number | `"0.0625"` |
| `hotstart` | true/false | `true` |
| `lam` | integer | `7` |
| `large_spin_double_twists` | integer | from the lam profile |
| `large_spin_npoles` | integer | from the lam profile |
| `large_spin_pole_selection` | string | `"contribution_optimal"` |
| `max_hi` | number | `"64"` |
| `npoles` | integer | from the lam profile |
| `num_threads` | integer | `4` |
| `pole_selection` | string | `"contribution_optimal"` |
| `q_order` | integer | from the lam profile |
| `sdpb_options` | table | merged over `tasks.DEFAULT_FEASIBILITY_SDPB_OPTIONS` |
| `self_consistent_h_ext` | true/false | `false` |
| `spins` | list of integers | `[0, 2, 4, 6, 8, 10]` |
| `tol` | number | `"0.05"` |
| `tower` | true/false | `true` |
| `twist_gap` | none, match_gap, or number | `"none"` |

### `scalar_only_gap_probe`

Single-point ALLOWED/EXCLUDED classification for the scalar-only gap.

| Parameter | Type | Default |
|---|---|---|
| `backend` | string | `"sdp"` |
| `c` | number | required |
| `delta` | number | required |
| `dps` | integer | from the lam profile |
| `lam` | integer | `11` |
| `num_threads` | integer | `4` |
| `sdpb_options` | table | merged over `tasks.DEFAULT_FEASIBILITY_SDPB_OPTIONS` |

### `scalar_ope_max_scan`

Locate the scalar gap bound by maximizing an OPE coefficient.

| Parameter | Type | Default |
|---|---|---|
| `all_spins_twist_gap` | true/false | `false` |
| `backend` | string | `"sdp"` |
| `c` | number | required |
| `delta_stars` | list of numbers | `["0.5", "1.0"]` |
| `dps` | integer | from the lam profile |
| `h_ext` | number | `"0.0625"` |
| `lam` | integer | `11` |
| `large_spin_double_twists` | integer | from the lam profile |
| `large_spin_npoles` | integer | from the lam profile |
| `npoles` | integer | from the lam profile |
| `num_threads` | integer | `4` |
| `q_order` | integer | from the lam profile |
| `sdpb_options` | table | merged over `tasks.DEFAULT_OPE_MAX_SDPB_OPTIONS` |
| `spin_star` | integer | `0` |
| `spins` | list of integers | `[0, 2, 4, 6, 8, 10]` |
| `tower` | true/false | `true` |
<!-- END GENERATED TASK TABLE -->
