# Public Python API

Import the supported block API from `virasoro_blocks.api` and the bootstrap
API from the `virasoro_bootstrap` package root. The core bootstrap functions
are also available through `virasoro_bootstrap.api`. Lower-level modules are
implementation details.

```python
from virasoro_blocks.api import (
    BlockRequest,
    NumericalPolicy,
    build_table_request,
    evaluate_block_request,
)
from virasoro_bootstrap import (
    IdenticalScalarGapSpec,
    certify_dual_from_sdpb_output,
    read_sdpb_objective,
    run_identical_scalar_gap_problem,
)
```

## `virasoro_blocks.api`

Use `BlockRequest` to specify a block evaluation or derivative table.
`validate_request` checks whether the requested operation is supported for
the selected family. Pass the request to `evaluate_block_request` or
`build_table_request` to perform the evaluation or build the table.

The main exports are:

- Requests and policy: `BlockRequest`, `BlockValue`, `BlockCapabilities`,
  `NumericalPolicy`, `RequestValidationError`, `capabilities_for`, and
  `validate_request`.
- Evaluation: `evaluate_block_request`, `F_block`, `G_block`, `B_block`,
  `F_block_chiral_h`, `parity_projected_B_block`, and
  `parity_reversed_B_block`.
- Table construction and serialization: `build_table_request`,
  `SerializedDerivTable`, `serialize_deriv_tables`, and
  `serialized_table_from_json`. A table loaded from JSON retains its stored
  precision (`precision_actual`). `evaluate` uses at least that precision to
  limit cancellation among the monomial coefficients, then rounds to the
  caller's precision. Files contain only the retained `(m, n)` entries;
  `evaluate` returns zero for omitted entries. At a diagonal center, a
  `scalar_G` table stores `n <= m`; the remaining entries follow from
  `G[n][m] = G[m][n]`. Folded coordinates store the parity-allowed entries.
  Ordered tables retain the full `m + n <= lambda_order` triangle, including
  for scalar externals.
- Coordinate helpers: `q_from_z`, `q_of_z`, `q_symmetric`, `z_of_q`, and
  `z_symmetric`.

For sphere tables, an explicit `basis="ordered_B"` keeps a single product of
chiral blocks for scalar or spinning externals. With scalar externals,
`basis="auto"` selects the parity combination `scalar_G`.
Ordered tables require `zzb`, `yyb`, or `qqb`; they support independent shifts
and centers and retain both derivative orientations. The same rule applies
to the family-prefixed global, NS, and Ramond bases, and to Virasoro/global
point tables and large-spin towers where those operations are available.
Large-spin towers use a common shift for the two chiral halves.

The scalar block is normalized as
`G_{Δ,ℓ} = [B(h, h̄) + B(h̄, h)] / (1 + δ_{ℓ,0})` for every integer `ℓ`.
A parity-even four-point function of scalars is symmetric under `z ↔ z̄`,
so the two orderings enter with equal weight. For the parity-odd structure
`B(h, h̄) − B(h̄, h)`, use `parity_projected_B_block` or a
`parity_projected_B` table with `parity="-"` (`η = −1`). A `scalar_G` request
or tower requires `parity_exchanged="1"`. One scalar-`G` large-spin tower
covers every integer `L ≥ Lmax`.

The spin-0 block in `G_block`, every `*scalar_G` table, and
`evaluate_block_request` is the product `B(h, h) = F(h, z) F(h, z̄)`.
The identity contribution is `F_0 F_0` with coefficient 1. Each channel enters
the four-point function with coefficient `p = N^Δ λ²`, where `N` is the chiral
normalization (see [block normalization](CLI.md#--chiral-norm-n)). In a mixed
correlator, the s-channel coefficient is `N^Δ λ_{12O} λ_{43O}`. This gives
`N^Δ λ_{12O}²` in `⟨φ1 φ2 φ2 φ1⟩` and `(−1)^ℓ N^Δ λ_{12O}²` in
`⟨φ1 φ2 φ1 φ2⟩`, since `λ_{21O} = (−1)^ℓ λ_{12O}`. The blocks contain no
overall `(−1)^ℓ` factor.

Serialized scalar-`G` tables and towers record `scalar_G_odd: "even"`.
Readers reject odd-spin scalar-`G` files and odd towers
(`parity_exchanged: "-1"`) without this field. Scalar-`G` tables also record
`scalar_G_spin0: "single"`; the bootstrap rejects spin-0 tables without
`scalar_G_spin0`. `BlockValue.as_dict()` uses schema `block-value/v2`;
scalar-`G` values include both `scalar_G_spin0` and `scalar_G_odd`.

For scalar and parity-projected towers, `xt_radial`, `ws_radial`, and
`qqb_ws_radial` retain derivatives with `n = 0` and `0 <= m <= lambda_order`
of the parity-smoothed block, for either parity. The `lambda_order - 1`
cutoff for odd parity applies only to the two-variable folded tables.

`BlockPoleError` is raised at a genuine exchanged-weight pole. The numerical
policy controls regularization at recognized singular central charges. At a
Kac weight and a singular `c`, Virasoro value evaluators return the block of the
irreducible module `L(h)`. This is the `kac_limit="irreducible"` default in
`F_block`, `G_block`, `B_block`, their `_regularized` forms, and
`H_q_series_regularized`. Set `kac_limit="c_limit"` to take the `c -> c0`
limit at fixed `h`. This differs from the irreducible block where Kac poles
at the same level merge, including every Kac weight at `c = 1` and `25`.
Evaluators without a `kac_limit` argument,
including those used by the table builders and `evaluate_block_request`, use
`virasoro_blocks.recursion.kac_limit_mode()`. Set this mode with
`kac_limit_context(mode)` or the CLI's `--kac-limit` option.

`NumericalPolicy.precision` sets the working precision, like the CLI's
`--precision`. Tables are built and written at that precision, then verified
against their numeric samplers between and beyond the interpolation nodes
(global tables and large-spin towers are checked between nodes only). They
record `verified_digits`, `verified_digits_largest_entry`, `verified_range`,
and `working_dps` (the JSON field `precision_actual`). Any shortfall is reported
on stderr with its cause.

Set `NumericalPolicy.target_digits` to request a minimum number of verified
digits. The build starts at `precision` and increases the working precision
by the measured deficit, up to `max_precision` (default
`max(4 target_digits, 2 precision, target_digits + 200)`). If it cannot reach
the target, `virasoro_blocks.verified_build.TargetDigitsError` (a `ValueError`)
reports the reason. `max_precision` requires `target_digits`.

Point evaluations near a pole or near `z = 1` lose digits through cancellation in
`h - pole` (about `log10(max(|h|, |pole|)/|h - pole|)`) or `1 - z`. Combining two
block components `a` and `b` can lose another `log10(max(|a|, |b|)/|a + b|)` digits,
as in a parity-odd projection near `z = zbar`. At working precision `P`, the
evaluators measure this loss `L` and issue a
`virasoro_blocks.conditioning.PrecisionLossWarning` when `L > P/2`.
If no reliable digits remain, they raise `InsufficientPrecisionError`, a
`ValueError` that reports the precision needed to retain `P` digits.
`evaluate_block_request` records
`estimated_digits = min(P - L, q_truncation_digits)` (for torus one-point blocks,
`min(verified_digits, q_truncation_digits)`). Table builds use their own
verification procedure instead of these point-value warnings.

With `target_digits = D`, `evaluate_block_request` starts at `max(P, D)` and
measures the loss `L`. If needed, it raises the precision to
`max(P, ceil(D + L))` and repeats until the target is met. It raises
`TargetDigitsError` if the required precision exceeds `max_precision` or
the calculation stops converging. The result records `precision_actual`.
A `torus-onepoint` value is verified against a deeper reference instead.
Symmetry-enforced zeros are recognized from the input labels and returned
exactly. Distinct exact inputs that round to the same working value still
require enough precision to resolve their difference.

Decimal and rational strings are interpreted exactly: `"0.7"`, `".7"`, and
`"7/10"` all represent seven tenths. The same applies to `int`, `Fraction`,
and `Decimal` inputs. Python floats are interpreted using their shortest
decimal representation. For these inputs, Kac-pole membership, singular
central charges, and spin integrality are checked exactly. Mpmath values
have finite precision and are compared within roundoff. Computations use
the working precision.

Requests reject unused or conflicting fields, such as `exchange_*` on a
table, `policy.npoles` on a point table, or `delta_ext` together with `h_ext`.

A Python float within `2^-50 max(|c|, 1)` of a singular central charge is
treated as that exact value. For example, `c=6/7` as a float is evaluated
like `Fraction(6, 7)` and produces one `SnappedCentralChargeWarning`;
`c=0.7` is interpreted as 7/10 from its decimal representation.
To evaluate near a singular value without this adjustment, pass an `mpf`,
a string, or a `Fraction`. Within `1e-10` of a singular value, colliding Kac
poles cause cancellation in the bare recursion. Evaluations use extra
working precision, verified at each call site by comparing results at two
precisions until they agree. The verified precision is cached.

## Torus families

Use `surface="torus"`, `family="torus-onepoint"`, coordinate `"vvb"`, and
`torus_onepoint_ext` for the external scalar's chiral weight. Interpolated
tables support any real central charge; recognized singular charges use the
exact Laurent kernel. Every real pole at positive `x = Delta - shift` is
split off before interpolation and recombined afterward, including merged
pole orders. `NumericalPolicy.npoles` caps only the remaining nonpositive or
complex poles. If omitted, the cap is chosen from the measured accuracy and
recorded as `npoles_chosen`. Positive poles are always included, even when
`npoles=0`, so the retained count (the table's `npoles`) can exceed the cap.
Each remainder entry uses a node set matched to its degree.

Verification compares the table with a deeper, higher-precision reference
between nodes, on both sides of representative positive poles, and in the
far tail. Metadata records `fit_digits`, `precision_digits`, and
`q_truncation_digits` separately, with `verified_digits` as the overall
accuracy measure. `target_digits` raises the working precision when it
limits accuracy. The anchor is `abs(spin)` by default, with `gap` and
`torus_scalar_shift` as described in
the [CLI reference](CLI.md#torus-tables). Torus character tables
(`family="torus-character"`) are the Verma characters at any real `c`.

`operation="point-table"` builds a torus table on `"vvb"` at fixed
`exchange_delta` and `exchange_spin`, with derivative order
`policy.lambda_order`. The spin must be an integer `j >= 0`; the module
`(h, h̄) = ((Δ + j)/2, (Δ − j)/2)` is paired with its mirror.
One-point tables also use `policy.order` and `torus_onepoint_ext`.

A character point table uses the irreducible character at a Kac weight:
`torus_vacuum_table` at `(0, 0)`, `torus_current_table` at
`(j, 0)`, and the minimal-model and `c = 1` characters at a recognized
singular `c`. `virasoro_blocks.modular.torus_character_point_values` returns
the table and a flag indicating whether it is irreducible.
For one-point blocks, `virasoro_blocks.torus_onept.torus_onept_point_table`
uses the irreducible-module trace at Kac weights of recognized singular
central charges, as
`evaluate_block_request` does. It records `verified_digits`, measured against
a deeper reference, and `q_truncation_digits`. For these Kac weights, the table's
`metadata["regularization_policy"]` is `"irreducible"`.

## Superconformal families

The request interface and CLI support the `ns` and `ramond` families. They
use the same table schema, coordinates, and interpolation format as
the ordinary Virasoro family. Recognized singular charges, including the N=1
minimal models and `c = 3/2`, are evaluated at `c` itself by the exact Laurent
recursion, and interpolated tables use its merged poles. For nonsingular
charges near a singular value, `NumericalPolicy.c_regularization="on"`
selects the `c+eps` evaluation path, with configurable offsets.
Select the family, sector, and Ramond branch through `BlockRequest` or the
CLI; see the [CLI reference](CLI.md) for valid combinations.

`ns` has four NS externals and NS exchange; `ramond` has four Ramond externals
and NS exchange. Mixed NS/R external configurations and Ramond intermediate
modules are not implemented. Full blocks use the same elliptic sector on the
two chiral halves and currently require integer exchanged spins, including
in the ordered and parity-projected bases. `operation="point-table"` evaluates
an NS or Ramond block at `exchange_delta` / `exchange_spin` (or `exchange_h`
for a chiral basis) in every coordinate, including folded coordinates, as
for the Virasoro family. At an N=1 singular `c`, the table uses the
`c -> c0` limit. It records the same family metadata as other NS/Ramond tables.


## `virasoro_bootstrap.api`

The bootstrap API provides `IdenticalScalarGapSpec` and
`IdenticalScalarGapProblem`, their corresponding build functions, manifest
read/write helpers, crossing helpers, and moment-bound records. These tools
depend on `virasoro_blocks`, which can also be used independently.

Gap, OPE, and moment problem builders accept `prepared=` to reuse block
tables across a scan. Create these tables with
`prepare_identical_scalar_gap_build(spec)`, or load them with
`prepare_identical_scalar_gap_build_from_blocks(spec, blocks)`. The latter
accepts CLI `table` output written with `--u-shift −2h_ext` or the `blocks/`
directory saved with a manifest.

- Each file's metadata is checked against `spec`: central charge,
  externals, crossing shift, λ, order, npoles, regularization, and tower
  twists. Rounded labels, such as a crossing shift of −2/15, are compared at
  the file's `precision_input`. This allows a 30-digit table to match a spec
  whose labels were entered at higher precision.
- If a spin's requested edge differs from the file's anchor, the block
  engine rebuilds that channel at the new edge, as it does for tables built
  in memory. SDP data use numeric block samples at the channel's nodes:
  directly for the highest-degree entries, and through interpolants
  evaluated in product form for lower-degree entries. The saved polynomial's
  coefficients are not reevaluated to obtain these data.

`build_identical_scalar_problem_from_manifest` rebuilds gap, OPE, and moment
problems from a saved manifest. For moment problems, it reconstructs the
insertion from the recorded weights, α, β, identity mode, and basis, using
the saved blocks. At each node, the SDP data are the measure's numeric value
multiplied by the exact weight.

Manifest edits must be compatible with the saved block data. The reader
rejects changes to the recorded block settings (c, h_ext, λ, order, N,
npoles, pole selection, and tower), `point_constraints`,
`shared_sample_points`, `sample_via`, or `dps`, and identifies the affected
field. The files retain the manifest's precision; set SDPB's precision
separately through `sdpb_options`. Changing a gap or per-spin edge rebuilds
the affected channels. Changing `target_digits` is allowed only if every
fixed-spin block file has recorded `verified_digits` at least that large.

**Precision.** `IdenticalScalarGapSpec.dps` sets the working precision for
building blocks and writing SDP data. To write the SDP data at another
precision, use `run_identical_scalar_gap_problem(..., eval_dps=...)`.
SDPB's `--precision` is separate: it defaults to `⌈dps log₂ 10⌉ + 32` bits,
with a minimum of 448, and can be overridden in `sdpb_options`.

Each fixed-spin block is verified against its numeric sampler between the
interpolation nodes and beyond them at large Δ. The file records
`verified_digits`, `verified_digits_largest_entry`, and `precision_actual`.
A `RuntimeWarning` reports precision-limited blocks with fewer than half
the requested digits, or blocks with fewer than three verified digits on
either an entry's own scale or the scale of the largest entry. For
precision-limited blocks, the warning gives the required `dps`. Pole
reduction is an approximation; its accuracy loss alone does not trigger a
warning if every entry retains at least three verified digits on both scales.

Set `target_digits`, optionally with `max_dps`, to require that many verified
digits from every fixed-spin block. The builder increases the working
precision by the measured deficit and rebuilds the whole problem. If it
cannot reach the target, `TargetDigitsError` reports the cause: the precision
cap, stalled convergence, or pole reduction with an integer `npoles`.
The resulting `spec.dps` records the working precision used to write the
problem data. The manifest stores `target_digits` and `max_dps` in its spec,
along with `provenance["precision"]` = `{working_dps, dps_requested,
target_digits, max_dps, verified_digits, verified_digits_largest_entry}`.
Prepared builds require matching block settings. Without `target_digits`,
`dps` must also match exactly. With `target_digits`, keep it and `max_dps`
unchanged and request `dps` no greater than the build's `working_dps`.

### Solving, reading, and certifying

The package root, `virasoro_bootstrap`, re-exports the core API and adds tools
for solving problems, reading results, and checking functionals:

- Problem builders: `build_identical_scalar_gap_problem`,
  `build_identical_scalar_ope_max_problem` (maximize the squared OPE
  coefficient of the operator at the bottom of a spin's continuum),
  `build_identical_scalar_ope_bound_problem` (upper or lower bound on an
  isolated operator below the gap), `build_identical_scalar_fixed_ope_problem`
  (feasibility with one OPE coefficient fixed), and the moment builders. A
  normalized moment ratio `(M[numerator] + beta) / (M[denominator] + alpha)`
  requires a strictly positive denominator on every admissible spectrum.
  The builder rejects denominators that are provably never positive. For
  other signed weights, the caller must establish positivity.
  For the modular (torus) gap bootstrap, use `ModularGapSpec`,
  `prepare_modular_gap_build`, `build_modular_gap_problem`,
  and `run_modular_gap_problem`. Modular problems require `c > 1`. The
  certification functions below support sphere problems only. Check a
  modular bound for convergence by including more spins in `spin_list`.
- Large-spin towers: `double_twist_set(h_ext, n)` returns the twists
  `2 Delta_phi + 2k`, `k < n`, for `IdenticalScalarGapSpec.large_spin_twists`.
  With a twist gap, `twists_above_gap(twists, gap)` returns the gap and all
  twists above it. Set `large_spin_lmin` and `large_spin_twists` together.
  The tower floor `large_spin_lmin` must be an integer from 1 to
  `max(spin_list) + 2`. Cluster tasks default to `max(spin_list) + 2`.
  A lower floor also imposes the tower over part of the fixed-spin range,
  where it nearly duplicates the fixed-spin constraints at those spins, but
  is generally harmless.
  A tower floor of zero is rejected because the tower's analytic continuation
  to `L = 0` gives the doubled block `2 B(h, h)`, whereas the spin-0 block is
  `G = B(h, h)`. Include spin 0 in `spin_list`.
  `virasoro_blocks.large_spin.large_spin_twist_table` also rejects
  `Lmax = 0` for an even `scalar_G` tower for the same reason.
- Running SDPB: `run_identical_scalar_gap_problem(problem, output_dir,
  sdpb_options=...)` writes the problem, runs SDPB, and returns a
  `DockerBootstrapRun`, with `sdpb_status`, `sdpb_summary`, `sdpb_out_dir`,
  and log paths. Possible statuses are `primal_feasible`, `dual_feasible`,
  `optimal`, `max_complementarity`, `max_iterations`, `max_runtime`,
  `primal_step_too_small`, and `dual_step_too_small`, based on SDPB's
  termination reason. A nonzero exit gives `solver_error`, with the log's
  last error line in the summary. The status is `None` if the log is missing
  or contains no termination reason.
  This function runs gap, OPE, and moment problems. Use `SdpbRunner` and
  `resolve_sdpb_runner` to select Docker, Apptainer, or a native SDPB
  ([cluster guide](CLUSTER.md#pointing-at-your-sdpb)).
  `classify_sdpb_log_text` turns a log into the same status string.
- Reading results: `read_sdpb_objective(sdpb_out_dir)` returns
  `{"primal": ..., "dual": ...}` from `out.txt`. Convert these objectives to
  bounds with
  `ope_squared_bound_from_objective` / `ope_squared_lower_bound_from_objective`
  or the `moment_*_from_objective` / `normalized_moment_*_from_objective`
  helpers. `load_y_vector(sdpb_out_dir)` reads the functional written by SDPB,
  with the normalization pivot omitted.
  `load_y_vector(sdpb_out_dir, full=True)` restores that component from the
  SDP directory's `normalization.json`, as does `reconstruct_full_y(y,
  sdpb_out_dir)`.
- Certification: `certify_dual_from_sdpb_output(problem, sdpb_out_dir,
  extra_spins=..., extra_twists=...)` returns `(ok, details)`. For Virasoro gap
  and OPE problems, the default `exact_blocks=True` checks the functional on
  blocks evaluated directly from the recursion, rather than their interpolants.
  Normalized-moment problems written directly to SDP have a different default:
  included channels are checked against the saved SDP interpolants, which
  does not independently test their block interpolation error. Set
  `force_exact_included=True` with `physical_moment_blocks=True` (the latter
  is the default) to evaluate those blocks directly on a finite grid.
  `exact_blocks` does not control this moment path. Requested extra or adaptive
  moment coverage uses directly evaluated blocks by default, with
  `moment_cert_q_order` selecting the q-series order. For moment towers,
  `extra_twists` extends the double-twist sequence above its top slice;
  `interstitial_twists` adds checks between slices. For gap/OPE towers,
  `extra_twists` itself adds intermediate slices.

  To check a full-length functional already in memory, use
  `certify_dual_functional(y, spin_channels, F_pairs)` with the desired
  channels. For example, for a gap problem, call
  `certify_dual_functional(load_y_vector(out, full=True), problem.spin_channels, problem.F_pairs)`
  with `F_pairs` specifying the functional's components.
  `certify_dual_functional` samples each channel from its edge to
  `delta_range` above it. For `c > 1`, `certify_dual_from_sdpb_output` extends
  this range to `max(delta_range, (c - 1)/4 + 10)`. These finite grids do not
  check the asymptotic tail. `worst_certification_detail(details)` prioritizes
  failed channels, then the smallest functional value, treating missing or
  nonfinite minima as worst. It returns `None` for empty details.
  The record's `min_value` is a float, or a string if outside the
  double-precision range; `min_value_text` is always a string.

The [README walkthrough](../README.md#a-first-bound) works through a complete solve,
and the [examples](../examples/) cover the OPE and modular problems.

The optional HPC commands manage local and cluster jobs. Their Python
interfaces are not part of the stable API. See the [cluster guide](CLUSTER.md)
for configuration and usage.
