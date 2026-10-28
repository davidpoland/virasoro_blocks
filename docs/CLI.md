# `virasoro_blocks` CLI reference

This reference describes each `table` flag, when to use it, and how it
interacts with other flags. `evaluate` and `point-table` have their own sections
([below](#evaluate-and-point-table)).
Source: [`src/virasoro_blocks/__main__.py`](../src/virasoro_blocks/__main__.py).
For a worked example, see the [README](../README.md).

The normal entry points are `virasoro_blocks evaluate`, `virasoro_blocks table`,
`virasoro_blocks point-table`, and `virasoro_blocks info`. Running
`virasoro_blocks` with table flags and no command is an alias for `table`.
Use `info` to list the bases and coordinates supported by each family.

Real numeric flags accept exact decimal and rational text: `.7`, `0.7`, `7e-1`,
and `7/10` all mean exactly seven tenths. Parsing is independent of
`--precision`; an explicitly written offset such as `0.7000000000000001`
is retained when deciding whether an input is singular. Chiral weights derived
from exact dimensions and spins remain exact as well.
Spin quantization is checked exactly: a value such as `0.5000000000000001`
is rejected rather than rounded to a half-integer.

## Cheat sheet

| flag | purpose | default |
|------|---------|---------|
| `--central-charge C` | Virasoro central charge `c` | *required*, except for explicit `global_*` bases |
| `--lambda N` | total derivative bound Λ | *required* |
| `--spin-ranges SPIN ...` | exchanged spins (2d only) | `0` |
| `--delta-ext D D D D` | external dimensions (2d) | all zeros |
| `--spin-ext S S S S` | external spins (2d) | all zeros |
| `--h-ext H H H H` | external chiral weights | none |
| `--hbar-ext HB HB HB HB` | external antichiral weights (2d) | none |
| `--basis BASIS` | block basis; see `virasoro_blocks info` for valid bases by family | `auto` |
| `--family FAMILY` | block family (`auto`, `virasoro`, `global`, `ns`, `ramond`, `torus-character`, `torus-onepoint`); use `virasoro_blocks info` for valid family/basis pairs | `auto` |
| `--surface {sphere,torus}` | four-point sphere blocks, or torus character and one-point tables ([Torus tables](#torus-tables)) | `sphere` |
| `--torus-onept-ext D` | with `--surface torus`: torus one-point tables for a scalar external of chiral weight `D` | none (character tables) |
| `--torus-scalar-shift X` | positivity anchor of the spin-0 torus one-point table alone (exclusive with `--gap`) | none |
| `--gap G` | raise the table's positivity anchor to `G` above the unitarity bound | none (unitarity bound) |
| `--parity-project {+,-}` | project to `B_±` (spinning only) | none |
| `--parity-ext P P P P` | external parity phases | none |
| `--parity-exchanged P` | exchanged-op parity phase | none |
| `--coordinates COORD ...` | which derivative tables to emit | `zzb` |
| `--order N` | q-expansion truncation `q_order` | `30` |
| `--npoles K` | poles kept in interpolation prefactor; fixed-spin/chiral Virasoro tables also accept `all` for symbolic output without samples | Virasoro/NS/Ramond: every pole; global, large-spin towers: chosen (as many as the precision carries); torus one-point: chosen (from the measured digits) |
| `--precision P` | working precision: every table is built and written at `P` digits, verified, and a shortfall reported | `60` |
| `--target-digits D` | opt in: raise the working precision from `P` by the measured deficit until the table delivers `D` verified digits (an error when it cannot) | off |
| `--max-precision M` | cap on that escalation (needs `--target-digits`) | `max(4 D, 2 P, D + 200)` |
| `--num-threads N` | worker count for parallel spin sweeps | `1` |
| `--chiral-norm N` | chiral block normalization `N` (a positive integer; the block is `N^{-h}` times the standard one) | `16` |
| `--z-shift S` | holomorphic z-exponent shift | `0` |
| `--zbar-shift S` | antiholomorphic z̄-exponent shift (two-sided bases only) | `0` |
| `--u-shift S` | set both `--z-shift` and `--zbar-shift` (on a chiral basis, `--z-shift` alone) | none |
| `--z-center Z0` | off-symmetric expansion center (a z cross ratio; for 2d this is the diagonal `z = z̄ = Z0`) | `1/2` |
| `--zbar-center ZB0` | independent z̄-center for an off-diagonal 2d block (`z = --z-center ≠ z̄ = --zbar-center`); requires `--z-center`; not valid for chiral bases | diagonal (`= --z-center`) |
| `--sample-via {numeric,symbolic}` | on the Virasoro integer-`--npoles` interpolation path, how sample-point block values are computed: `numeric` (well-conditioned recursion/Taylor builders) vs `symbolic` (evaluate the exact table directly). Refused off that path (`--npoles all`, the global and NS/Ramond families, the large-spin tower) | chosen automatically (`numeric` where supported, else `symbolic`; `auto` is not an accepted value) |
| `--q-gate {off,warn,strict}` | q-order convergence gate: estimate the block accuracy implied by `--order` at this `(c, externals)` and warn/error when inadequate (global blocks: only `off`) | `warn` |
| `--pole-selection SELECTOR` | which poles an integer `--npoles` keeps (refused without one, except on the fixed-spin global bases) | `contribution_optimal` for Virasoro, global, and superconformal fixed-spin blocks |
| `--shared-sample-points` / `--no-shared-sample-points` | one shared max-degree node set for all `(m,n)` (padded) vs per-entry degree-matched sets (refused with `--npoles all`, which has no nodes) | off (per-entry) |
| `--regularize {auto,on,off}` | the block at a recursion-singular `c` from its merged Kac poles; `auto` triggers at recognized singular `c` | `auto` |
| `--kac-limit {irreducible,c_limit}` | the Virasoro block ON a Kac weight of a singular `c`: the irreducible module's, or the `c -> c0` limit at fixed `h` (Virasoro family only) | `irreducible` |
| `--format {m,json,both}` | output format | `json` |
| `-o DIR` / `--output DIR` | output directory | `.` |
| `--output-template TEMPLATE` | JSON filename pattern | `derivs_{spin}.json` |
| `--large-spin-lmin LMIN` | emit fixed-twist large-spin (`variable="L"`) tower instead of fixed-spin blocks; LMIN is the lower spin edge (above 0 for an even `scalar_G` tower) | off |
| `--large-spin-double-twists K` | auto-build the leading `K` double-twist twists from the externals | none |
| `--large-spin-twists TAU ...` | explicit isolated twist values (combined with the double-twist set) | none |
| `--large-spin-pole-selection SELECTOR` | tower pole selection (separate from `--pole-selection`) | `contribution_optimal` |

---

## Accuracy presets

Four numbers set a table's cost and accuracy: `--lambda`, `--order`,
`--npoles`, and `--precision`.  The combinations below are the ones the
packaged bootstrap profiles use (`lam`, `q_order`, `npoles`, and `dps` of
`IdenticalScalarGapSpec`).  Measured at `c = 8`, external dimensions `1/8`,
one process on an 8-core Apple M3 laptop.  Each row gives one spin 0 table and
one at the profile's highest fixed spin.

| use | `--lambda` | `--order` | `--npoles` | `--precision` | seconds per spin | `verified_digits` | degree |
|---|---|---|---|---|---|---|---|
| format check, smoke tests | 4 | 20 | 6 | 60 | 0.5 | 4.3 | 10 |
| Λ = 7 bounds | 7 | 40 | 20 | 120 | 1.3–1.6 | 9.3–10.2 | 27 |
| Λ = 7, every pole (reference) | 7 | 40 | omitted, `--target-digits 120` | 120 (built at 312) | 10 | 127 | 217 |
| Λ = 11 bounds | 11 | 60 | 32 | 160 | 3.7–4.2 | 13.0–13.6 | 43 |
| Λ = 15 bounds | 15 | 80 | 60 | 180 | 13 | 20.6–21.6 | 75 |
| Λ = 19 bounds | 19 | 80 | 60 | 200 | 13–15 | 20.6–21.4 | 79 |

`verified_digits` is how closely the table reproduces the `--order`-truncated
block over `x = Δ - shift` from 0 to far past its last node (16 times the pole
scale, where SDPB still imposes positivity); between the nodes the order-reduced
rows hold 6.5, 15.8–17.5, 22–25, 37–41, and 35–40 digits, and their worst point is
the large-Δ tail.  "Degree" is the highest polynomial degree in `derivs`.  The
reference row keeps every pole (210 of them): at `--precision 120` alone its
degree-217 monomial form loses all its digits to rounding (the build says so and
names the precision that fixes it), so it asks for `--target-digits 120`, and is
built and written at 312 digits (`precision_actual`).

- The order-reduced rows are approximations by design.  Their low degree keeps
  SDPB well conditioned.  Certification
  (`virasoro_bootstrap.certify_dual_from_sdpb_output`) re-checks gap
  functionals on exact blocks by default. For normalized-moment direct-SDP
  problems, included channels use the serialized interpolant unless
  `force_exact_included=True`; enable that option to check their interpolation
  error against freshly evaluated blocks.
- `--order` must grow with `c` and with `--lambda`.  The q-order gate (below)
  measures this and suggests an order.  At `c = 8` every row above passes it
  except the first, which is at about 20 digits.
- At `c <= 1` every Kac pole at positive `x` is kept, so a table's degree
  stays near the full table's, whatever `--npoles` is.  The build then prints the
  `--precision` its measured cancellation needs (or pass `--target-digits`).
- `--num-threads N` builds `N` spins at once.  A single Virasoro spin does
  not parallelize further.

### The digits notes

Every table is verified, and one that falls short of `--precision` prints one
line naming the cause and the fix (on stderr; the singular-`c` notice goes to
stdout).  The severity is the same for every family: a **warning** when the
table does not reproduce the block (under 3 digits against its largest entry),
when a keep-all table disagrees with its own block beyond the precision, or
when the precision is the limit and less than half the working digits survive;
a **note** otherwise.

- `note: spin S: the order-reduced table (--npoles K) reproduces the block to X
  digits of --precision P over x = Delta - shift in [0, R] (...); the order
  reduction limits it (raise --npoles, or omit it to keep every pole)` — the
  approximation asked for.  `X` is the table's `verified_digits`; a value below
  `P` is expected.  The global tables say the same against the exact ₂F₁ block
  (about 2 digits per kept pole).  The global tables and large-spin towers
  choose the pole count when `--npoles` is omitted, so their note says
  `raise --npoles, or omit it to choose the pole count`.
- `warning: spin S: the table reproduces the block to X digits of --precision P
  ...: its degree-D monomial form loses ~L digits to rounding at working
  precision P ..., which is what limits it.  Raise --precision to about W (which
  carries P digits through it), or pass --target-digits P to raise it
  automatically.` — a keep-all table (`--npoles` omitted or `all`): its fit and
  evaluation lose about 0.6–0.75 digits per unit of degree.  An order-reduced
  table can be precision-limited too, when its coefficients cancel on
  evaluation (`... cancel by ~L digits on evaluation (at x = ...)`, typically
  at `c <= 1`) or cannot reproduce the values they were fit to at the nodes
  (`... reproduce the values they were fit to at its nodes to only F digits`).
  `W` estimates the precision needed to recover `P` digits from the measured
  loss. Check the rebuilt table's `verified_digits`; further increases may be
  needed. `--target-digits P` repeats this check and raises the precision as
  needed, subject to `--max-precision`.
- `note: spin S: the table was built at working precision W to deliver
  --target-digits D (X digits verified ...)` — with `--target-digits`.
- `warning: q-order gate [...]: estimated block accuracy ~X digits at
  q_order=N (...); suggest q_order >= M` — the `q`-series itself is
  truncated too early.  Neither `--precision` nor `--npoles` can fix this;
  raise `--order`.  See [`--q-gate`](#--q-gate-off-warn-strict).
- `note: c is at a recognized Virasoro singular locus (...); enabling
  regularization` — see [`--regularize`](#--regularize-auto-on-off).

The accuracy of a table against the true block is the smaller of its
`verified_digits` and the q-order gate's estimate; the files record both
(`verified_digits` and `q_truncation_digits`), for the Virasoro and the
NS/Ramond families alike.  A `warning:` line saying a table "does not reproduce
the block" means a bootstrap built on it constrains a different function:
raise `--npoles`.  That test uses `verified_digits_largest_entry`, the error
relative to the largest entry at each point: a folded coordinate's subleading
t-derivative entries lose relative accuracy at large Δ while the table as a
whole is right (`c = 8`, `--lambda 11`, `xt`, `--npoles 40`: 0.2 digits per
entry, 16 against the largest entry, which is a note).

---

## Required flags

### `--central-charge C`

Virasoro central charge `c`. Required for every basis except the explicit
`global_*` bases, which are independent of `c` and ignore it if given. Accepts
exact decimal and rational strings. The Liouville
parameter `b` is derived from `c` via `b² − Q·b + 1 = 0` with `Q² = (c − 1)/6`.

The recursion is singular at `c = 1`, `c = 25`, and the minimal-model values `c = 1 −
6(p−q)²/(pq)`. The default `--regularize auto` detects these and builds the block at the
singular `c` itself, from the merged Kac poles of the Laurent expansion of the
recursion, and prints a notice. `--regularize off` selects the direct recursion at
nonsingular inputs; at a recognized singular value the CLI refuses the request and
recommends `--regularize auto`. See [`--regularize`](#--regularize-auto-on-off) below.

### `--lambda N`

The total derivative bound Λ. The shape of the kept `(m, n)` pairs depends on
the coordinate and basis (see [truncation table](#coordinates)). Same
semantics as `blocks_3d`'s `--lambda`.

---

## External operators

The 2d bases accept either of two parametrizations; they are equivalent and
mutually exclusive within one invocation.

### Spin & parity conventions

Each operator carries holomorphic / antiholomorphic weights `(h, h̄)`, with

- scaling dimension `Δ = h + h̄`,
- **spin `ℓ = h − h̄`** — in 2d the rotation little group is `SO(2) ≅ U(1)`, so
  spin is a single **signed charge** (the eigenvalue of `J = L₀ − L̄₀`, the
  `L₃`/`Jz`-type generator), **not** a nonnegative Casimir. It runs over all of
  `ℤ` (or `ℤ + ½` for fermionic operators).

There is **no `h ≥ h̄` restriction**: signed spins are allowed throughout, and
`(h, h̄)` may be supplied directly (`--h-ext` / `--hbar-ext`) or as `(Δ, ℓ)`
(`--delta-ext` / `--spin-ext`). Parity (the `z ↔ z̄` swap) acts as `ℓ → −ℓ`, i.e.
`(h, h̄) → (h̄, h)`; the parity-definite bases (`scalar_G`, `parity_projected_B`)
combine `±ℓ` into one block, so they are naturally labeled by the **magnitude**
`s = |ℓ| ≥ 0` (and `G_{Δ,−ℓ} = G_{Δ,ℓ}` is a redundant relabeling). The
raw `ordered_B` block keeps a definite ordering, where the sign of `ℓ` matters.

By convention one usually takes `h ≥ h̄` (so `ℓ = s ≥ 0`) and lets the
parity-definite block supply the `−ℓ` partner. The large-spin tower is the
`ℓ = L → +∞` family (it grows the holomorphic weight at fixed antiholomorphic
twist `τ = 2h̄`); the negative-spin family is its parity image.

### Parametrization A — `(Δ, spin)`

#### `--delta-ext D D D D`

External scaling dimensions `Δ_1 Δ_2 Δ_3 Δ_4`. Used together with `--spin-ext`
(both default to all zeros). The chiral halves are then `h_i = (Δ_i + s_i)/2`,
`h̄_i = (Δ_i − s_i)/2`. Defaults to all zeros when neither parametrization is
supplied.

Mutually exclusive with `--h-ext` / `--hbar-ext`. Rejected with `--basis
chiral_F`.

#### `--spin-ext S S S S`

External operator spins `s_i = h_i − h̄_i`. Accepts integers or half-integers
(decimal, e.g. `0.5`, or fraction form, e.g. `1/2`; negative values allowed).
Any nonzero entry selects the spinning-external code path, which restricts the
admissible coordinates (see [truncation table](#coordinates)).

Mutually exclusive with `--h-ext` / `--hbar-ext`. Rejected with `--basis
chiral_F`.

### Parametrization B — `(h, h̄)`

#### `--h-ext H H H H`

External holomorphic weights `h_1 h_2 h_3 h_4`. On the 2d bases, must be
paired with `--hbar-ext`. For `--basis chiral_F`, `--h-ext` is the only
external-weight input (no antiholomorphic half exists).

Mutually exclusive with `--delta-ext` / `--spin-ext`. Per-operator
`h_i − h̄_i` must be an integer or half-integer.

#### `--hbar-ext HB HB HB HB`

External antiholomorphic weights `h̄_1 h̄_2 h̄_3 h̄_4`. Required together with
`--h-ext` on the 2d bases; rejected with `--basis chiral_F`.

Mutually exclusive with `--delta-ext` / `--spin-ext`.

---

## Block basis

### `--basis BASIS`

Which block representation to emit. Default `auto` infers from the externals and
parity flags (it never selects a `global_*` basis — those are opt-in only):

Run `virasoro_blocks info` to list the complete set of family-prefixed bases.

- All-scalar externals (no spin) → `scalar_G` (the `G_{Δ, ℓ}` parity
  eigenstate).
- Any nonzero external spin, no parity projection → `ordered_B` (the raw
  ordered `B(h, h̄; z, z̄)` block).
- Any nonzero external spin + `--parity-project` → `parity_projected_B` (the
  `B_+ / B_-` eigenstate).
- Only chiral coordinates (`z`, `y`, `q`) requested → `chiral_F` (the chiral
  half `F(h, z)`).

You can pass the basis explicitly to override the inference; the CLI then
validates that the inputs are consistent (e.g. `chiral_F` requires `--h-ext`
and rejects `--delta-ext`/`--spin-ext`/`--parity-*`).

Explicit `ordered_B` also accepts scalar externals, for fixed-spin tables,
point tables, and large-spin towers where the family supports them. It keeps
one spin orientation, while `scalar_G` combines the two. Ordered tables retain
both `(m, n)` and `(n, m)` derivatives and require the non-folded coordinates
`zzb`, `yyb`, or `qqb`. This applies to the family-prefixed ordered bases too.

### Global (SL(2)×SL(2)) bases — `global_scalar_G`, `global_ordered_B`, `global_parity_projected_B`, `global_chiral_F`

These emit the **2d global (SL(2)×SL(2)) block** — the `c → ∞` limit, built
from the chiral 2F1 `k_h(z) = z^h ₂F₁(h+h₂−h₁, h+h₃−h₄; 2h; z)` — as the same
`derivs[m][n] -> poly(x)` output the Virasoro bases produce, so downstream
tooling is unchanged. They are the global counterparts of the four Virasoro
bases (same externals/coordinates/parity rules), with these differences:

- The block is built by **optimal-node interpolation of the exact numeric 2F1**.
  `--order` does not truncate that evaluation; it defines the finite SL(2)
  pole universe. A chiral pole `2h = -p` first appears at elliptic q-level
  `p + 1`, so `--order Q` includes the genuine left/right poles with `p + 1 ≤ Q`.
- For generic external weights the global block has infinitely many SL(2)
  poles and no exact finite rational form. The decay base is
  `exp_base = 4 ρ(½)/N` (`= (3 − 2√2)/4 ≈ 0.043`
  for the default `N = 16`),
  and `--npoles` keeps that many **genuine** SL(2) Δ-poles (Δ = ℓ−n, −ℓ−n);
  removable poles are stripped exactly from the external differences (the set
  depends on whether the externals are pairwise-equal). Each `(m, n)` entry's
  degree is tied to `--npoles` so the large-Δ tail power is preserved — so
  `--npoles` is the precision knob (raise it, and `--precision`, for accuracy).
  At coincident centers, antiparity cancels the leading power, giving
  `m+n-1` even for unequal holomorphic and antiholomorphic external weights.
  At distinct centers the leading power is `m+n`.
  This per-entry tail matching is the **default** (`--no-shared-sample-points`);
  passing `--shared-sample-points` pads every entry to the max degree, which
  improves the in-window fit but lets the far tail grow (see that flag below).
- `--pole-selection` defaults to `contribution_optimal`. It ranks against the
  one q-order-defined pool, so increasing `--npoles` extends the same nested
  greedy selection. If `--npoles` exceeds the genuine poles available at that
  q-order, raise `--order`. The other selectors remain available for numerical
  comparisons, and `real_first` / `real_deep` differ on double-pole
  multiplicity.
- `global_chiral_F` builds the one-variable global 2F1 directly in `h`
  (chiral coordinates `z`/`y`/`q`, `--h-ext` only, single output file).
- **Coordinates match the Virasoro side.** `global_scalar_G` and
  `global_parity_projected_B` accept the non-folded `zzb`/`yyb`/`qqb`, the
  parity-folded `xt`/`ws`/`qqb_ws`, and the radial slices `xt_radial`/
  `ws_radial`/`qqb_ws_radial` — the interpolation core applies the same
  Δ-independent `_fold_parity` recombination as the fixed-spin tables (even
  smoothing parity for `global_scalar_G` at every spin, sign × phase for
  `global_parity_projected_B`). `global_ordered_B` has no parity partner, so it
  keeps the non-folded coordinates only.

The numeric 2F1 Taylor samplers run on `python-flint` `acb_series` (compiled C).

### N=1 superconformal bases — `--family {ns, ramond}`

The N=1 super-blocks are a separate family, selected explicitly:

```text
virasoro_blocks table --family ns --basis ns_chiral_F ...
virasoro_blocks table --family ramond --basis ramond_scalar_G ...
```

The short bases (`chiral_F`, `scalar_G`, `ordered_B`, `parity_projected_B`)
are accepted as aliases once `--family` is set. Two further flags select the
superconformal sector and Ramond branch:

- **`--super-sector {even, odd}`** — the elliptic sector. Both families expose
  both.
- **`--ramond-branch {++, +-, -+, --}`** — the HJS sign branch. Ramond only.
  Every branch can be written as a separate value (`--ramond-branch --`,
  `--ramond-branch -+`) or with `=` (`--ramond-branch=--`).

The Virasoro, global, and torus families reject both flags, even when passed
their default values. The `ns` family also rejects `--ramond-branch`.

`ns` has four NS externals and NS exchange; `ramond` has four Ramond externals
and NS exchange. Mixed NS/R externals and Ramond intermediate modules are not
implemented. Full blocks use the same elliptic sector on both chiral halves.

`--order` is the q-order for the Virasoro and the NS/Ramond recurrences alike;
the NS/R recurrence's internal half-levels need no second public order.

At `c ≤ 3/2` NS/Ramond Kac poles sit at positive `x = Δ − ℓ` (inside the
positivity region; below `c = 3/2` they are most of the pool) and are always
kept. As for Virasoro at `c ≤ 1` (see "How a c ≤ 1 table is built"), they are
split off as exact principal parts, taken from the NS/Ramond recursion's pole
data (at a singular c, from the merged Laurent data of the exact recursion;
under `--regularize on`, at the offset `c + 10^-eps` the block is evaluated
at). The candidate pool runs two levels (q^2) past the block's own recursion;
a positive pole of those deeper levels is no pole of the block and is not
written. The order reduction then fits only the regular remainder, whose poles all lie
at `x ≤ 0`. `--npoles` counts the kept `x ≤ 0` poles; the positive ones are
exact. The written table still carries the positive poles in its prefactor, so
its degree stays near the full table's and its monomial form can cancel more
digits than `--precision` holds; the build then quotes the precision that
carries them.

JSON output records `block_family`, `super_sector`, `ramond_branch`,
`central_charge_evaluated`, `regularization_policy`, `regularization_eps_exp`,
and `collision_policy`; the Mathematica writer carries the same as
`blockFamily`, `superSector`, `ramondBranch`, `centralChargeEvaluated`,
`regularizationPolicy`, `regularizationEpsExp`, and `collisionPolicy`.

For a direct point value rather than a table:

```text
virasoro_blocks evaluate --family ns --basis ns_scalar_G \
  --central-charge 1.3 --delta-ext 0.2 0.2 0.2 0.2 \
  --coordinate zzb --exchange-delta 0.8 --exchange-spin 0 \
  --point z=0.3 --point zbar=0.4 --order 20 --precision 80
```

Its `value` is the order-20 block to 80 digits, but at this point order 20
leaves only about 33 digits of the untruncated block, and the JSON says so:
`q_truncation_digits` and `estimated_digits` are both `33.5`. `--order 50`
raises both to `76.8`.

The Python entry points behind these are described in [API.md](API.md).

### `--parity-project {+, -}`

For spinning externals, project the ordered block onto its parity eigenstates:
`B_± = B(h, h̄) ± phase · B(h̄, h)`. Selects `parity_projected_B`. Required
when requesting parity-folded coordinates (`xt`, `ws`, `qqb_ws`, `*_radial`)
with spinning externals, since the folded coordinates need a `z ↔ z̄`
eigenstate. The total phase (from `--parity-ext` and `--parity-exchanged`)
must be real (`±1`) when used with folded coordinates.

Requires `--parity-ext`. Rejected on scalar externals (which already use the
scalar parity eigenstate `G`). Rejected with `--basis chiral_F`.

### `--parity-ext P P P P`

External parity eigenphases for the four operators in the diagonal-parity
basis. Each `P` must be one of `1`, `-1`, `i`, `-i` (string tokens like `+1`,
`I` are normalized). Required when using `--parity-project`. Without
`--parity-project` it is metadata only (recorded in the output as
`parity_external`; does not alter the computed block).

Rejected with `--basis chiral_F` (parity is a 2d concept).

### `--parity-exchanged P`

Parity phase of the exchanged primary, in the same diagonal-basis convention.
Used with `--parity-project` (tables and `parity_projected_B` towers). The
scalar `G` is `z ↔ z̄` even at every spin, `G = k_h k_hbar + k_hbar k_h`, so it
takes no exchanged parity: a fixed-spin `scalar_G` table refuses the flag, and
its large-spin tower accepts only `+1` (the default), as one tower covers every
integer `L ≥ LMIN`. `-1` is refused with a pointer to the `z ↔ z̄`-odd
structure `k_h k_hbar − k_hbar k_h`, which is `parity_projected_B` with
`--parity-project -` (`η = −1`). The CLI builds that basis for spinning
externals only; for scalar externals use the Python API.

---

## Spin range

### `--spin-ranges SPIN [SPIN ...]`

Exchanged spins ℓ. Accepts:

- single integer or half-integer: `0`, `2`, `1/2`, `0.5`;
- `start:stop` (step = 1): `0:4` → `[0, 1, 2, 3, 4]`;
- `start:step:stop`: `0:2:10`, `1/2:1:5/2`.

Multiple tokens are unioned, deduplicated, and sorted. One JSON / `.m` file
is emitted per spin.

Half-integer exchanged spins need `ordered_B` or `parity_projected_B`:
`scalar_G` requires integer spins and rejects a
half-integer `--spin-ranges` entry.

NS and Ramond full-block bases currently require integer exchanged spins,
including their ordered and parity-projected bases.

Ignored on the chiral path (`--basis chiral_F` produces a single file; ℓ is
not a property of a chiral block).

---

## Coordinates

### `--coordinates COORD [COORD ...]`

Which derivative tables to compute. One or more of:

| family | members | admissible bases |
|---|---|---|
| non-folded 2d | `zzb`, `yyb`, `qqb` | scalar_G, ordered_B, parity_projected_B, global_scalar_G, global_ordered_B, global_parity_projected_B |
| parity-folded | `xt`, `ws`, `qqb_ws` | scalar_G, parity_projected_B |
| radial slice | `xt_radial`, `ws_radial`, `qqb_ws_radial` | scalar_G, parity_projected_B |
| chiral | `z`, `y`, `q` | chiral_F, global_chiral_F |

Coordinates written together must share a positivity anchor, exponential base,
and pole multiset, since one JSON file stores a common prefactor. Order
reduction can select different poles for different coordinates. If the CLI
reports that their interpolation prefactors disagree, request each coordinate
in a separate invocation and output directory.

The `global_*` bases take the same coordinates as their Virasoro counterparts.
`global_scalar_G` and `global_parity_projected_B` support the parity-folded
coordinates and radial slices as well (the interpolation core folds via the
shared `_fold_parity` recombination); `global_ordered_B`, like the Virasoro
`ordered_B`, keeps the non-folded coordinates only.

The parity-folded coordinates `xt`/`ws` are threaded through the full crossing
stack: the fixed-spin path (`crossing.F_minus_pairs` builds the folded `F^-`
selector — odd `a` over `a + 2b ≤ Λ`, vs the non-folded `m + n` odd; both the
Virasoro and global control builders accept them) and the large-spin tower
(`large_spin_twist_table`, which folds the non-folded outer-product grid via the
same L-independent `_fold_parity` recombination). Folded towers require basis
`scalar_G` or `parity_projected_B` (`ordered_B` has no parity fold).

The sphere tables retain the following nonnegative integer `(m, n)` pairs,
sorted lexicographically. A diagonal center means `z_center = zbar_center`;
it need not be `1/2`.

| coordinate and center | scalar_G / parity_projected_B | ordered_B | chiral_F |
|---|---|---|---|
| `zzb`, `yyb`, `qqb`, diagonal center | `m+n ≤ Λ` and `n ≤ m` | `m+n ≤ Λ`, both orientations | rejected |
| `zzb`, `yyb`, `qqb`, unequal centers | `m+n ≤ Λ`, both orientations | `m+n ≤ Λ`, both orientations | rejected |
| `xt`, `ws`, `qqb_ws` | `m + 2n ≤ Λ_±` (jagged) | rejected | rejected |
| `*_radial` | `n = 0`, `m ∈ [0, Λ]` | rejected | rejected |
| `z`, `y`, `q` | rejected | rejected | `m ∈ [0, Λ]` |

At unequal centers, swapping the derivative indices also swaps the expansion
point, so `n > m` entries cannot be recovered from the other half of the table.
For example, at `Λ = 4`, a non-folded scalar table stores nine entries at a
diagonal center and 15 at unequal centers. Ordered tables always store all 15.
These are storage rules; entries that vanish by symmetry may still be stored.

`Λ_+ = Λ` for even-parity outputs (scalar G at every ℓ, or a projection whose
sign × phase is `+1`); `Λ_- = Λ − 1` for odd-parity outputs (sign × phase `−1`).
An odd-parity folded table at `Λ = 0` has no entries. The radial cutoff remains
`m ≤ Λ` for both parities; the `Λ − 1` rule applies to the two-variable folded
tables. Folded and radial coordinates use the crossing-symmetric center and
reject center overrides.

Chiral and 2d coordinates cannot be mixed within one invocation. The
family-prefixed global, NS, and Ramond bases use the corresponding rules above
where the operation is supported. Torus `vvb` tables retain `m+n ≤ Λ` and
`n ≤ m`.

---

## Numerical control

### `--order N`

q-expansion truncation in the Zamolodchikov h-recursion. Larger `N` improves
convergence at `z = z̄ = 1/2` (each `q^k` is multiplied by `(q(1/2))^k ≈
0.043^k`). Defaults to `30`. Costs grow like O(N²) memory and O(N²-N³) time
through the polynomial-in-h recursion.

### `--q-gate {off, warn, strict}`

Checks convergence of the q-series. The truncation error grows with the
derivative order Λ and with `c` (roughly
additive: adequate `--order` ≈ Λ + an offset that grows from ~15 at small `c`
to hundreds at `c ≳ 300`), and raising `--precision` cannot fix it. The gate
measures the actual truncation tail of the chiral series (differences of
evaluator builds at three q-orders; the recursion is triangular in the
q-power, so the differences are exactly the tail windows) and estimates the
block accuracy in decimal digits at this `(c, externals, --order)`.

The omitted tail is estimated by geometric extrapolation of finite windows
at selected exchange weights. This is not a rigorous bound on the full tail
or every table entry; cancellations in folded coordinates can amplify errors.

- `warn` (default): one stderr line with the measured digits, the decay rate
  (digits per unit of `--order`), and a suggested `--order`, whenever the
  estimate is below 20 digits **or** the series has not yet entered its
  geometric regime at this `c`, where a digit estimate can overstate
  convergence. The latter is reported as unconverged.
- `strict`: raise an error instead of warning (recommended for production bound
  runs; certification also runs this gate via
  `certify_dual_from_sdpb_output(..., q_gate=...)`).
- `off`: skip the probe.

The probe uses a few chiral-evaluator calls, with results cached by the inputs,
order, and precision. Its cost depends on those inputs. The global blocks have
no q-series, so they accept only `--q-gate off` (an explicit `warn` or
`strict` is refused rather than silently skipped). At a
recognized singular `c` it probes the merged Laurent block at that charge.
Exact nonzero offsets are retained, with additional working precision when
needed to resolve them. The estimate is calibrated to track the true truncation
error within ~2 digits across `c ∈ [1.1, 3000]`, small and large externals.
NS/Ramond tables get the same gate, on their numeric Taylor coefficients at
three q-orders (their `verified_digits` compare at one order, so they cannot see
truncation). Every Virasoro and NS/Ramond table records the estimate as
`q_truncation_digits`. Torus one-point tables apply the same modes to their
measured `q_truncation_digits` ([Torus tables](#torus-tables)); the exact
character tables refuse the flag.

### `--precision P`

The **baseline working precision** in decimal digits (default `60`): the CLI
sets `mp.mp.dps = P`, and the whole-table build starts at that precision.
Internal routines may temporarily use extra guard precision to resolve inputs
and protect against cancellation. FLINT measures precision in binary bits;
the factor `3.33` approximates `log2(10)`, converting decimal digits to bits.
It does not multiply the requested number of decimal digits by 3.33.

Without `--target-digits`, verification reports a shortfall rather than
rebuilding the whole table at a higher precision. The table is serialized and
written at its working precision (`P`, unless `--target-digits` raised it).

Tables are stored as polynomials in the monomial basis, whose fit and
evaluation lose digits to rounding: a degree-`D` table loses about 0.6–0.75
digits per unit of `D`, most of it past its last node.  So every table is
**verified** against its own exact numeric sampler at points between its
interpolation nodes and past them, doubling out to 16 times the larger of the
last node and the largest `|pole|`, where the error has settled to its
large-Δ value: SDPB imposes positivity at every `x >= 0`.  (Global tables and
large-spin towers are checked between their nodes only: past them an
order-reduced global table degrades where the block is exponentially damped.)
The JSON records:

- `precision_input` — `P`, as given;
- `precision_actual` — the working precision the coefficients were computed and
  written at (`P`, unless `--target-digits` raised it);
- `verified_digits` — the worst per-entry relative agreement with the block over
  the checked range, `verified_delta_range`;
- `verified_digits_largest_entry` — the worst agreement relative to the largest
  entry at each point (what decides whether the table reproduces the block);
- `target_digits` — the `--target-digits` asked for (absent without it).

A table short of `P` prints one line with the cause and the fix
([The digits notes](#the-digits-notes)): the precision (the monomial form's
measured rounding or cancellation: `Raise --precision to about W`), the order
reduction (`raise --npoles`), or the `--order` truncation.  A keep-all table
(`--npoles` omitted) of high degree is precision-limited at any `P` it is not
built for: the reference row of the presets loses all 120 of its digits.

**`--target-digits D`** (with **`--max-precision M`**, default
`max(4 D, 2 P, D + 200)`) asks for at least `D` verified digits.  `P` is then the
starting working precision: a precision-limited table is rebuilt at the
precision its measured loss needs, then by the measured digits-per-working-digit
rate, until it delivers `D` (plus a small margin), and is written at that
precision (`precision_actual`).
The build stops with an error instead of
writing a table when `M` is reached (raise `--max-precision`), when an attempt
closes less than a quarter of the remaining gap (precision is not the limit:
the kept poles or the table's pole structure are), or when an order-reduced
table is limited by its order reduction (raise `--npoles`) or the q-series by
`--order`.  `--max-precision` without `--target-digits`, `D > M`, and `M < P` are
refused.  The cap only bounds the cost of a genuine precision shortfall: 4 `D`
(or `D` + 200) carries `D` digits through a table of degree up to ~4 `D`
(~270), and a build at 4 `D` costs about 10 times one at `D`.

**Order-reduced tables** (an explicit `--npoles`) are approximations by design:
their `verified_digits` below `P` is expected (a note), and `--target-digits`
can only raise their precision, never their kept poles.  Torus one-point tables
are always order-reduced and follow the same rules, with a check against a
deeper, higher-precision reference ([Torus tables](#torus-tables)).

At c ≤ 1 (NS/Ramond: c ≤ 3/2) the Kac poles at positive x (inside the positivity region) are always
kept, so the degree stays close to the full table's whatever `--npoles` is, and
the coefficients' cancellation on evaluation can exceed `P`: the warning then
quotes the measured loss and the precision that carries it.

**How a c ≤ 1 table is built.** The poles at positive x are split off as
exact principal parts, Σ_a g_a (Δ − p)^−a, taken from the recursion's Laurent
data (at a singular c, from its merged Laurent data). The order reduction then
fits only the regular remainder, whose poles all lie at x ≤ 0, exactly as at
c > 1. The written table is the usual prefactor-and-numerator form with both
parts recombined; its `sample_block_values` give the written interpolant at
the shared writer nodes, using recursion samples or the entry's Lagrange
interpolant as described below. `--npoles` counts the kept x ≤ 0 poles; the
positive ones are exact.
At singular c (minimal models, c = −2, ...) each colliding pole's order is
measured on the regularized block, because a collision can form a
higher-order pole: Δ = −1/4 at c = −2 is order 6.

Global tables without `--npoles` choose the kept poles (about 1.6-2 digits per
pole): at first the count the working precision carries (each pole adds ~1.6
digits and each unit of degree costs ~0.75 of `P`), grown by the measured
deficit while the order reduction is the limit and more poles help.  With
`--target-digits` the working precision grows as well. A spinning global pool
carries double poles, so its degree is about twice its pole count.

At an N=1 singular central charge (rational `beta^2`: the N=1 minimal models,
`c = 3/2`, ...), NS and Ramond tables are built at `c` itself. The shared
`virasoro_blocks.laurent_levels` engine computes the recursion as a Laurent
series in `t`. Samples use the block at that charge, and the prefactor retains
each merged pole at its surviving order (`regularization_policy: "merged"`).
No central-charge offset is needed. A `c` near a singular value is evaluated
as supplied, with the recursion's guard precision.

**Reading a table back.** `virasoro_blocks.serialization.serialized_table_from_json`
parses the coefficients at `precision_actual`, and `evaluate` runs at that
precision whatever `mp.mp.dps` the caller has set (or at the caller's, if
higher). It rounds the result to the caller's precision. Evaluating the
monomial coefficients at fewer digits cancels them away (a 50-digit table
evaluated at 15 digits is off by about 2e-4), so a reader of your own should
also work at `precision_actual`. Default rewrites preserve the loaded numeric
text and recorded precision, so repeated load/write cycles do not append
conversion roundoff digits; changed values are written at the working
precision, and explicit `sig_figs` reformats the data.

Decimal inputs (`--central-charge`, the externals, centers, and shifts) denote
their exact values (`0.7` is 7/10), so the table is accurate for the decimals as
typed.


### `--num-threads N`

blocks_3d-style worker count for sweeps over multiple exchanged spins.
Parallelism is only across independent spins: one worker builds one spin's
serialized derivative tables at a time, and the parent process keeps the file
writing order deterministic.

The CLI flag `--num-threads` controls worker processes. Processes bypass
the GIL on the mpmath-heavy build path and give each worker its own
`mp.mp.dps` precision context. `--num-threads 1` is the default serial path.
Worker counts must be positive integers. Python interpolation APIs also accept
integral numeric values such as `2.0`, normalize them to integers, and reject
booleans, fractional counts, and strings without truncation.
Python interpolation calls inside a daemon worker run serially, since that
worker cannot start child processes.

For the `global_*` bases the flag does double duty: with multiple spins it
parallelizes across spins as above; with a **single** spin it instead
parallelizes *within* the build, across the per-derivative-degree node-count
buckets (the optimal-node search + block sampling + Lagrange fit for each
bucket are independent). So a single high-`--lambda` global build still uses
all the workers. The output is bit-identical regardless of `--num-threads`.
(The Virasoro bases parallelize across spins only; a single Virasoro spin
ignores the flag.)

### `--shared-sample-points` / `--no-shared-sample-points`

Controls how the per-`(m,n)` numerator polynomials of an interpolated block are
fit. It applies to every interpolated basis — the Virasoro integer-`--npoles`
bases (`scalar_G`, `ordered_B`, `parity_projected_B`,
`chiral_F`) **and** the `global_*` bases — and is **off (per-entry) by
default**.

- **Off (default, per-entry).** Each entry gets its **own degree-matched**
  Bohr–Sommerfeld node set (degree tied to `--npoles`, so the reduced rational's
  large-Δ tail power is preserved). The serialized block still carries a single
  shared sample set (the max-degree nodes, `sample_points_x`), with each
  entry's value there in `sample_block_values`. For the max-degree entries
  that value is the numeric block sample itself. A lower-degree entry, fit on
  its own nodes, gets its Lagrange interpolant at the shared node, formed from
  its fitted node values by products (never a monomial polynomial evaluated,
  whose coefficients cancel about 0.7 digits per unit of degree). The SDP
  writer uses these values directly.
- **On (`--shared-sample-points`).** Every entry is fit on the **single, shared
  max-degree node set**: the lowest-degree entries are padded up to the largest
  entry's degree, so the interpolation nodes coincide exactly with the serialized
  `sample_points_x`. More accurate *strictly inside* the sampling window.

**Why per-entry nodes are the default.** Padding a low-degree entry increases
its polynomial tail power, which can spoil the approximation beyond the node
range. SDPB imposes positivity throughout the half-line, so this can change a
bound even when the fit is accurate between nodes. Use
`--shared-sample-points` only when accuracy within the node range is sufficient
and you have checked that the tail does not affect the result. The bootstrap
layer exposes the same option as `IdenticalScalarGapSpec.shared_sample_points`.

### `--chiral-norm N`

Selects `N` in the library's chiral rescaling

```
F(h, z) = N^{-h} (16 q)^h · (h-independent factors) · H(q).
```

Default `N = 16` gives base `q(1/2) = e^{-π} ≈ 0.043` at the crossing-symmetric
point. This base is matched to the physical large-weight decay of `OPE² · block`
(`~ q^Δ`, `~ q^ℓ`, from Virasoro Mean Field Theory; see
Virasoro Mean Field Theory, [arXiv:1811.05710](https://arxiv.org/abs/1811.05710)), so the Bohr–Sommerfeld
sample points concentrate in the region of dominant contributions. The
normalization propagates consistently through the chiral block, the 2d
composer, the symbolic Δ-tables, and the output metadata.

For the self-dual-point base `4 q(1/2) ≈ 0.173 ≈ ρ(1/2)` (Hogervorst–Rychkov
radial), pass `--chiral-norm 4`.

`N` must be a positive integer on the command line: the bootstrap specs and
block manifests that reuse CLI tables accept integer normalizations only. (The
Python block functions take any positive real `chiral_norm`.)

**Block normalization.** Write the standard chiral block as
`F_std(h, z) = z^{h − h₁ − h₂} (1 + O(z))`. The library block is

```
F(h, z) = N^{-h} z^{h₁ + h₂ + z_shift} F_std(h, z),
```

so with `--z-shift=-h1-h2` it is `N^{-h} F_std`: at the default `N = 16`,
`16^{-h}` times the standard block. A 2d block is a product of two chiral
halves, `B(h, h̄) = F(h, z) F(h̄, z̄)`, and carries `N^{-Δ}`. The
parity-symmetric scalar block is

```
G_{Δ,ℓ} = [B(h, h̄) + B(h̄, h)] / (1 + δ_{ℓ,0}),
```

at every integer `ℓ`, odd included (it is `z ↔ z̄` even, as the correlator of
scalars in a parity-invariant theory is). At `ℓ = 0` it is the product block `F(h, z) F(h, z̄)` itself, and the
identity block `G_{0,0}` is `F_0(z) F_0(z̄)` with coefficient 1. The four-point
function is `G_{0,0} + Σ p G_{Δ,ℓ}` with

```
p = N^Δ λ²
```

in every channel: `λ²` is the standard-normalization squared OPE coefficient of
the scalar at `ℓ = 0`, and of each member of the parity pair `(h, h̄)`,
`(h̄, h)` at `ℓ ≠ 0`. A z-shift shared by every block, identity included, does
not enter `p`. An OPE-coefficient bound computed from these tables bounds `p`;
divide by `N^Δ` for the standard `λ²`.

Every scalar-`G` table records its spin-0 normalization as `scalar_G_spin0`
(`"single"`; `scalarGSpin0` in the `.m` files), and the bootstrap refuses a
spin-0 table without it.

Odd spins of a scalar `G` are the `z ↔ z̄`-even `B(h, h̄) + B(h̄, h)`; the
`z ↔ z̄`-odd structure is the `parity_projected_B` block with `η = −1`.
Scalar-`G` tables and towers record `scalar_G_odd: "even"` (`scalarGOdd` in the
`.m` files), and the table readers and the bootstrap refuse an odd-spin
scalar-`G` table or an odd tower (`parity_exchanged: "-1"`) without it.

### `--z-shift S`

Shift in the holomorphic z-exponent of the chiral block. The full block
prefactor includes a factor `z^{σ + z_shift}` (where `σ = (c−1)/24`), so the
leading small-z behavior is `N^{-h} z^{h + z_shift}` (see
[block normalization](#--chiral-norm-n)). Default `0` gives the leading power
`z^h`: the standard block `z^{h − h₁ − h₂}(1 + O(z))` times `z^{h₁ + h₂}`,
divided by `N^h`.

The most common nonzero value is `--z-shift=-h1-h2`, which gives the
standard power `z^{h − h₁ − h₂}` (still divided by `N^h`). For identical-scalar crossing equations the bootstrap passes
`z_shift = zbar_shift = −2 h_ext`.

The CLI can generate block files for the same bootstrap problem with matching
kinematics and numerical options:

- `--u-shift −2h_ext`;
- `--gap Δ_gap` on the spin-0 table only (other spins anchor at their
  unitarity bound);
- the manifest's `--lambda`, `--order`, `--npoles`, and `--precision`;
- the same family, basis, coordinates, normalization, pole selection, and sampling
  options.

A problem rebuilt from such files gives the same `pmp.json` to working precision.

A block set is reusable for any problem it describes: another gap, another
objective (OPE maximization, moments), or a subset of its spins.
`virasoro_bootstrap.api.prepare_identical_scalar_gap_build_from_blocks(spec, directory)` reads CLI
output or a manifest directory, checks each file's metadata against `spec`,
and returns a prepared build to pass as `prepared=` to any problem builder.
A spin whose requested edge differs from its file's anchor (another gap, above
or below) is rebuilt at the new edge by the block engine, exactly as an
in-memory `prepared=` build does. Its SDP data are then computed at its own new
nodes from numeric block samples (see `sample_block_values` above); the file's
polynomial is never re-evaluated elsewhere. Every
other spin, the tower, and the point constraints come from the files. A whole
binary search therefore runs from one CLI call per spin set, rebuilding only
spin 0 at each probe. Editing the gap in a `manifest.json` behaves the same
way.

**Crossing equation.** With the default `z_shift = zbar_shift = 0`, the
block sum satisfies

```
G(z, z̄) = ((z z̄) / ((1−z)(1−z̄)))^{Δ_ext} · G(1−z, 1−z̄)
```

where `Δ_ext = h_ext + h̄_ext` is the external scaling dimension. The
prefactor arises because the block definition includes a `(1−z)^{σ − d₂ − d₃}`
factor that is not symmetric under `z → 1−z`. Setting `z_shift = zbar_shift =
−Δ_ext` absorbs this prefactor into the block, making the crossing equation
simply `G(z, z̄) = G(1−z, 1−z̄)`.

### `--zbar-shift S`

Shift in the antiholomorphic z̄-exponent, analogous to `--z-shift` for the
`z̄` half. **Default `0`, independent of `--z-shift`** (each flag shifts only
its own exponent; use `--u-shift` to shift both).

Only the two-sided bases (`ordered_B`, `parity_projected_B`, and their global
variants) carry independent z/z̄ shifts. The `scalar_G` block is
z↔z̄-symmetric and is built from a *single* chiral expansion driven by
`--z-shift`, so it has no separate z̄ shift: a `--zbar-shift` that differs from
`--z-shift` is **rejected**: use `--u-shift` for symmetric conditioning;
independent z/z̄ shifts need `ordered_B` or
`parity_projected_B`. The large-spin tower is likewise single-shift.

### `--u-shift S`

Convenience shorthand: sets both `--z-shift` and `--zbar-shift` to the same
value `S`. Overridden by an explicit `--z-shift` or `--zbar-shift` on the same
command line. A chiral basis has no z̄ half, so there `--u-shift` sets
`--z-shift` alone (and an explicit `--zbar-shift` or `--zbar-center` is refused). The name "`u`" refers to the cross-ratio product `u = z z̄`; the
common crossing-equation convention uses `u_shift = −Δ_ext`.

### `--regularize {auto, on, off}`

Tri-state. Controls whether the block at a recursion-singular `c` (`c = 1`,
`c = 25`, minimal models, and the positive-rational `b² = p/q` branch for
`c > 25`) is built at that `c` from the merged poles of the Laurent recursion
in `t` (`c = c₀ + t²` at the branch points `c₀ = 1, 25`, `c = c₀ + t`
elsewhere): colliding Kac poles merge into higher-order poles, each at its
order after every residue cancellation, so the table is exact at `c₀`, with no
extrapolation in `c` and no pole stripping. Default `auto`. The table records
`regularization_policy` (`merged`, else `direct`).

- **`auto`** (default): the CLI inspects `c` via
  `singularity.recognized_singular_locus` and turns regularization **on** iff
  `c` is at one of the exact recognized loci. When auto-enabled the CLI
  prints a notice to stdout naming the locus (e.g. `note: c is at a
  recognized Virasoro singular locus (minimal model M(3,4) (Ising)); enabling
  regularization`). Detection is exact-only —
  a `c` *numerically near* a singular locus but not exactly on it is not
  recognized, and `auto` keeps the direct recursion. **However**, if `c` is
  within `10⁻¹⁰` of a recognized locus, the CLI prints a one-line hint to
  stderr noting the proximity and suggesting passing the exact value
  to land on the locus. `--regularize on` does not change the supplied value.
  Typical triggers: a typo (`0.5000000000001`), an
  IEEE-double residual from upstream Python, or an optimizer that converged
  near a singular CFT. The hint does not change the algorithm.
- **`on`**: always take the regularized route; at a `c` that is not a
  recognized locus it is the direct recursion.
- **`off`**: use the direct recursion. At a recognized singular `c`, the CLI
  rejects the request before building and recommends `--regularize auto`;
  the bare recursion is undefined there.

The bare flag `--regularize` (no value) is preserved as a backwards-compatible
spelling of `--regularize on`.

### `--kac-limit {irreducible, c_limit}`

Which Virasoro block a value takes ON a Kac weight of a recognized singular
`c`, where Kac poles collide and the block is discontinuous in `(h, c)`:

- **`irreducible`** (default): the block of the irreducible module `L(h)`, the
  one whose OPE coefficients a compact CFT at that `c` has (the free-boson
  vertex correlator at `c = 1` decomposes into its exact Fock couplings with
  it). It is read on the Kac curve `h = d_{r,s}(c)` with the externals kept on
  the fusion rule that decouples the null vector; a channel whose externals do
  not decouple it has no irreducible block and is refused as a genuine pole.
- **`c_limit`**: the `c -> c0` limit at fixed `h`. It equals the irreducible
  block except where Kac poles of one level merge -- every Kac weight at
  `c = 1` and `c = 25` -- where it keeps finite admixtures of the null
  submodule's composition factors (at `c = 1`, `h = m²/4`: of `L((m+4)²/4)`
  and `L((m+8)²/4)`, from level `2(m+2)` on; `3.35e-5` at level 8 for `h = 1`
  with `25/16` externals). At the minimal models and at `h = 0` (the vacuum)
  the two agree.

Off the Kac weights both are the block at `c0`, continuous in `Delta`, which
is also the function a `Delta` table carries; the option matters for values
and point tables at an exact Kac weight. Tables and values at a singular `c`
record it as `kac_limit`. Virasoro family only: it is refused for the global,
NS/Ramond and torus families (the N=1 blocks keep the `c -> c0` limit, see
`evaluate` below).

**Recognized singular loci.** Auto-detection covers `c = 1`, `c = 25`, the
full minimal-model series `c = 1 − 6(p−q)²/(pq)`, and the positive-rational
`b² = p/q` branch. Common nicknames (Ising, Yang-Lee, tricritical Ising,
etc.) are surfaced in the notice/warning text; everything else falls back to
the generic `minimal model M(p,q)` label plus the unitary index `m` when
applicable.

The chiral basis is covered too: `--basis chiral_F` at a singular `c` flows
through the same regularization machinery (the chiral `F(h, z)` table is a
one-index derivative table with "Δ" = `h`, and the removable-pole / collision
strips key on provenance), so no chiral-specific handling is needed.

---

## Block fidelity (numerator polynomial)

The per-`(m, n)` numerator polynomial is controlled by `--npoles`. The
following two paths describe fixed-spin and chiral Virasoro sphere tables:

- an **integer** `--npoles K` — order-reduced: sample the exact table at optimal
  nodes ([arXiv:2509.14307](https://arxiv.org/abs/2509.14307)) and interpolate,
  keeping `K` poles; produces a much lower-degree polynomial representing the
  same function to bootstrap accuracy. This is the production path.
- `--npoles all` — exact: keep every pole, no order reduction: the exact
  symbolic table written at `--precision`, with no sample points or node values
  (verified against the numeric sampler at 0.31, 2.3, and the far points like
  every table). Omit `--npoles` for the same poles as an interpolated table
  (sample points and node values). This is the conditioning-fragile extreme — at high `--order`
  the degree is enormous (e.g. ~232 at λ=11) and the SDP it feeds is badly
  conditioned; it is mainly a no-approximation reference for cross-checks.

At `c <= 1`, all Kac poles are real. The interpolation path keeps every
pole above the unitarity bound (`x = Delta - spin > 0`), split off as an exact
principal part (see "How a c ≤ 1 table is built"); `--npoles` then
covers the remaining poles at `x <= 0`. The selectable count depends on
**both** `--spin-ranges` (the shift ℓ) and `--order`:

- At spin 0 (shift = 0), it equals the number of Kac pairs (r,s) with
  r·s ≤ `--order` whose weight d_{r,s} < 0. For `c = 0.9` the first such
  pair is (8,7) at r·s = 56, so spin-0 has 0 selectable poles for order < 56.
- At higher spin ℓ (shift = ℓ), all poles at Δ ∈ (0, ℓ) also become
  selectable; `c = 0.9` spin-2 already has **15** selectable poles at order 20.

Keeping more poles reduces the order reduction, reaching the full truncated
block at `--npoles all`. Higher degree also increases cancellation, so measured
accuracy at fixed working precision need not improve monotonically. At the
keep-all limit an integer `--npoles`
equal to the full pole count and `--npoles all` produce the **same rational
function at the same degree** (same shifted variable `x = Δ − shift`); the
coefficients agree to the working precision but are not bit-identical at high
degree. The point of an integer `--npoles` is `npoles < all` (order reduction
for SDPB), where its low degree makes it cheap.

### `--sample-via {numeric, symbolic}`

Controls **only** how the Virasoro integer-`--npoles` path computes the
block values at its optimal sample points. Elsewhere it does nothing, so
passing it (either value) where it cannot apply is an error:

- `--npoles all` does **no order reduction** and no sampling (it also refuses
  `--shared-sample-points`).
- the **global** bases interpolate directly from the numeric 2F1 block.
- the **NS/Ramond** tables sample their block numerically and are verified
  against it.
- the **large-spin** tower interpolates its own numeric block sampler.

On the Virasoro interpolation path the two choices interpolate the **same
function** — only the conditioning of the sample-value step differs:

- `numeric` (default) — evaluate the block at each sample Δ via the direct
  recursion / Taylor builders. Well-conditioned at the working precision (the
  symbolic table's `P(Δ)/D(Δ)` coefficients reach ~1e500 and lose accuracy to
  cancellation when evaluated directly).
- `symbolic` — evaluate the exact symbolic table at the sample points; needs
  much higher precision to be safe.

The default resolves to `numeric`, except for a coordinate the numeric sampler
does not cover, where it is `symbolic`. Recursion-singular `c` is fully supported under `numeric` — the numeric
sampler built at the singular `c` from its merged poles supplies the node
values, and under `symbolic` the table is the exact rational function of those
poles — as are all the 2d and chiral coordinates and the four Virasoro bases.
Passing `numeric` *explicitly* in an unsupported case is an error (with the
reason), not a silent fallback.

### `--z-center Z0` / `--zbar-center ZB0`

Move the Taylor expansion center off the crossing-symmetric point `z = z̄ = 1/2`.
Both are given as **z cross ratios** (even for the `y`/`q` coordinates, whose
expansion variable is internally `u(z)`):

- `--z-center Z0` — the holomorphic center. For a 2d block this is the diagonal
  point `z = z̄ = Z0`; for a chiral block (`chiral_F`, which has no z̄) it is the
  single expansion point.
- `--zbar-center ZB0` — an independent antiholomorphic center for an
  **off-diagonal** 2d block (`z = Z0 ≠ z̄ = ZB0`). Requires `--z-center`; not
  valid for a chiral basis. Default: the diagonal (`z̄-center = z-center`).

Supported for the non-folded 2d (`zzb`/`yyb`/`qqb`) and chiral (`z`/`y`/`q`)
coordinates, and for the large-spin tower; the folded/radial coordinates reject
centers (their fold geometry is defined relative to the symmetric point). The
chosen centers are recorded in the JSON as `z_center` / `zbar_center` (always
emitted, default `1/2`), so every block self-documents the point its derivatives
are taken around — a value `!= 1/2` means each `derivs[m][n]` coefficient is a
derivative about `Z0` (and `ZB0`), not about `1/2`.

### `--npoles K`

An **integer** `K` is the maximum
number of actual Δ-poles kept in the damped-rational prefactor (the rest are
dropped by `--pole-selection` rank). It controls the interpolation polynomial
degree and file size.

When `--npoles` is omitted, the Virasoro and superconformal (NS, Ramond) bases keep
every pole, which represents the `--order`-truncated block exactly; the table is built
at the working precision `--precision`, verified, and written with its node values (its
high degree loses digits to rounding: pass `--target-digits` to have the precision
raised until it delivers them). The interpolated keep-all form reports a node-placement
failure instead of dropping poles to find an easier interpolation problem. An integer
`K` remains a maximum cap and can retain fewer poles when node placement requires it.

The `global_*` bases have infinitely many genuine poles. When `--npoles` is omitted, the
kept count, and the `--order` that supplies it, are chosen: the count the working
precision carries, grown while the order reduction limits the table (about 1.6–2 digits
per pole; with `--target-digits`, until the table delivers them). An explicit `K` is
always honored and recorded with its `verified_digits`.

The literal **`all`** keeps the same Δ-poles as the omitted Virasoro form but writes the
exact high-degree symbolic block at `P` without node values; it is available for
fixed-spin/chiral Virasoro sphere tables only. Global, NS/Ramond, torus one-point, and
large-spin tower tables use interpolation and refuse `all`; omit `--npoles` to keep
every pole on NS/Ramond tables.

**Ceiling.** The candidate Δ-pole count is determined by `--order`. For
ordinary full Virasoro blocks, the unfiltered candidate pool at `q_order = N`
has `2 × #{(r, s) : r·s ≤ N}` Δ-poles (e.g. 28 at order=6, 54 at order=10,
132 at order=20); a chiral block has one copy. Vanishing residues can remove
candidates, and singular-locus collisions require the merged pole orders.
NS/Ramond pools follow their own recurrence, sector, and residue rules. For
global tables, the pool instead contains genuine SL(2) poles with
chiral labels `0 ≤ p < N`: both left/right chiral factors for fixed-spin
blocks, or the holomorphic factor for a global large-spin tower. Coincident
locations remain repeated. If `--npoles` exceeds the available candidate pool,
increase `--order`; a Virasoro `--npoles` at or above the pool keeps every pole,
so the table is built as with `--npoles` omitted (a keep-all table).
The numerator degree also includes the derivative entry’s large-dimension
power; it can exceed the kept pole count (for example, torus one-point entry
`(m, n)` has degree `npoles + m + n`).

### `--pole-selection SELECTOR`

Which `K` poles `--npoles` keeps. The accepted values are listed by
`virasoro_blocks table --help`. It needs an integer `--npoles`: without one (or
with `--npoles all`) a Virasoro or NS/Ramond table keeps every pole and the flag
is refused. The fixed-spin global bases are the exception: without `--npoles`
they choose and grow the kept poles, and an explicit selector ranks that pool. The large-spin tower uses
`--large-spin-pole-selection` instead.

- `contribution_optimal` — **the Virasoro, global, and superconformal default, including large-spin towers.** First computes each
  candidate pole's full-principal-part contribution at the evaluation point,
  maximized across the retained derivative entries. It then estimates that
  pole's interpolation remainder using the optimal-node,
  damped-rational measure and builds a nested greedy order. “Optimal” refers
  to the interpolation nodes and measure, not to a global search over all
  size-`K` pole sets. Complex-conjugate pairs are kept together, and increasing
  `--npoles` extends the earlier selection.

- `contribution` — ranks each pole by the magnitude of its **full principal
  part** at the evaluation point: `|Σ_{j<m} a_j d^j| / |d|^m` for a pole of
  multiplicity `m` at distance `d`, i.e. every Laurent coefficient with the
  distance to the correct power. This is a direct score ranking, unlike the
  optimal-node remainder estimate above. A pole sitting exactly at the
  evaluation point is given an infinite (must-keep) score.
- `contribution_leading` — the leading-residue variant: the `residue` score
  (undifferentiated `(0, 0)` block value) divided by distance^1 only. Drops the
  sub-leading principal-part coefficients and uses the wrong distance power for
  double+ poles; kept for comparison (at a singular c it reads the full
  symbolic table of the merged poles).
- `residue` — ranks poles by their residue in the **undifferentiated `(0, 0)`
  block value**. Keeps the poles most
  relevant to the physical block rather than being dominated by the
  highest-derivative residues. Empirically ~100–1000× more accurate than
  `residue_all` at fixed `--npoles` for the identical-scalar fixed-spin
  Δ-channels; `contribution` is its distance-weighted refinement.
- `nearest` — smallest `|Δ_pole|`. Monotonically improves with
  `--npoles`, no even/odd accuracy discontinuity, and no admissibility failures
  at strip `c ∈ (1, 25)`. The robust geometric fallback (and essentially tied
  with `contribution` on the large-spin tower).
- `residue_all` — largest residue magnitude taken as the **max over every kept
  `(m, n)` derivative entry**. Can beat `nearest` by 0.5–1.6
  decades at specific odd `--npoles` values, but has a catastrophic even/odd
  accuracy trough at `c ∈ (1, 25)` (up to 2.4 decades worse at even counts).
  Safe only with manually chosen odd `--npoles`.
- `contribution_all` — like `contribution` but the full-principal-part score is
  maximized over every kept `(m, n)` derivative entry rather than just `(0, 0)`.
  Generally inferior to `contribution` (the high-derivative entries overweight
  deep poles); kept for comparison.
- `contribution_leading_all` — the `residue_all` score (max over every
  kept `(m, n)` derivative) divided by distance^1.
- `rightmost` — largest-Re poles; useful as a comparison baseline when broad
  node support matters more than fixed-window accuracy.
- `real_first` / `real_deep` — prioritise real poles over complex pairs; they
  differ only in how they handle ℓ = 0 double-pole multiplicity (breadth-first
  vs depth-first).

`contribution_optimal` requires a candidate pool independent of `--npoles`.
That holds for Virasoro fixed-spin blocks and towers, whose Kac pool is fixed
by q-order, and for superconformal blocks, whose recursion pole pool is
likewise fixed by q-order. Both fixed-spin global blocks and global large-spin
towers use the finite set of SL(2) poles first appearing through q-order.

> **Related bootstrap-side knobs (not CLI flags).** The bootstrap pipeline has
> two *separate* pole selectors on `IdenticalScalarGapSpec`: `positive_pole_selection`
> (`rightmost`/`nearest`/`residue`/`residue_all`, default `rightmost`) governs the
> remainder interpolation after genuine positive-real poles are cleared (a `c < 1`
> phenomenon), and `large_spin_pole_selection` (default `contribution_optimal`) governs the
> large-spin tower blocks. Both accept the same `residue` (value) / `residue_all`
> (all-derivative) semantics as this flag.


---

## Output

### `--format {m, json, both}`

`m` is the `scalar_blocks`-style Mathematica association; `json` is the
`blocks_3d`-style schema with all numeric values as strings; `both` emits
both. Default `json`. The association holds the polynomial coefficients as
`{coord}Deriv[m,n]`, with `shiftedPoles`, `expBase`, `shift`, and the JSON
metadata in camelCase (for example `verifiedDigits`, `samplePointsX`,
`scalarGSpin0`, `scalarGOdd`). It omits the node values
(`sample_block_values`).

### `-o DIR`, `--output DIR`

Output directory. Created if it does not exist. Default `.`.

### `--output-template TEMPLATE`

JSON filename pattern (also the stem of the NS, Ramond, and `point-table`
Mathematica files). `{spin}` is replaced by the exchanged spin (an int or
`.5` float); on the chiral path it is replaced by the literal `chiral`.
Default `derivs_{spin}.json` (matches `blocks_3d`'s `derivs_{}.json`
convention).

Mathematica filenames are not configurable. The Virasoro and global bases
follow the long `{coord}DerivTable-c{c}-...{basis}-...-order{O}.m` pattern
(global files omit `-c{c}`), which names every setting that changes the table,
so two runs that differ in one of them never overwrite each other in one `-o`
directory:

```
{coord}DerivTable[-c{c}]-delta1-{Δ1}-...-delta4-{Δ4}[-spin1-{s1}-...-spin4-{s4}]
    -L{ℓ}-lambda{Λ}-npoles{K}[-poles{Selection}][-sharedNodes]
    [-basis{Basis}][-parityExt{p1}{p2}{p3}{p4}][-parityExch{p}]-chiralNorm{N}
    [-zShift{S}][-zbShift{S}][-zCenter{z0}][-zbCenter{zb0}][-gap{G}]-order{O}.m
```

A bracketed tag appears only away from its default: a `--pole-selection`
other than `contribution_optimal` (for example `-polesNearest`),
`--shared-sample-points`, external parities other than `1 1 1 1` (written with
signs, `-parityExt+1-1+1-1`), an exchanged parity other than `1`, a nonzero
shift (`-zbShift` only for the two-sided `ordered_B` and `parity_projected_B`
bases, whose z̄ shift is independent), an expansion center other than `1/2`
(`-zbCenter` when it differs from the z center), and a nonzero `--gap`. A
default table keeps the short name, for example
`zzbDerivTable-c10-delta1-0.15-delta2-0.15-delta3-0.15-delta4-0.15-L0-lambda4-npoles6-chiralNorm16-order20.m`.
Numbers are written to 6 significant figures. `--precision`,
`--target-digits`, and `--sample-via` change only the digits of the same table,
so a rerun that differs only in them replaces the file (its header records
them). The NS and Ramond families name their Mathematica files after the JSON
file, `derivs_{spin}_{coord}.m` with the default template (a template without
`{spin}` gets the spin appended); pass an `--output-template` that
distinguishes the runs to keep them apart.

### JSON table schema

A `derivs_*.json` file holds one exchanged spin (or one large-spin twist
slice) for every requested coordinate. High-precision numbers are strings.
The labels that identify the block (central charge, external weights, centers,
the positivity anchor `variable_minus_x`, a tower's `twist`, a point block's
`point_value`) and the interpolation sample coordinates are exact: terminating
decimals, or `"p/q"` for a nonterminating rational. Other metadata (`z_shift`,
`zbar_shift`, `chiral_normalization_N`, `central_charge_evaluated`,
`onept_d_ext`, the verified range) are decimals at the metadata digits, as are
all Mathematica metadata. Readers accept either spelling of every label, and
compare a block file's labels with a problem's up to roundoff. Computed
coefficients, poles, and bases remain decimal strings, and complex numbers are
written `"(a+bj)"`. Spin labels remain JSON integers or half-integer floats; a
half-integer too large to represent exactly as a float is rejected.
Read a file back with
`virasoro_blocks.api.serialized_table_from_json(json.load(f))`, which parses
at `precision_actual`.
Its `evaluate(coordinate, Delta)` returns the `[m][n]`
table at one `Delta`.
Rewriting a loaded table preserves its stored working precision and verification
diagnostics. The loader rejects inconsistent axis labels and noninteger cutoff
or precision fields instead of guessing their meaning.

**The payload.**  With `x = Delta - variable_minus_x` (`x = L - variable_minus_x`
on a tower),

```text
block[c][m][n](Delta) = exp_base**x * sum_d derivs[c][m][n][d] * x**d / prod_j (x - pole_list_x[j])
```

where `c` indexes `coordinates`. The entry is the Taylor coefficient
`(1/(m! n!)) d_u^m d_v^n` in the selected coordinates at the expansion center
(the `blocks_3d` / `scalar_blocks` convention). For `zzb`, `(u, v) = (z, zbar)`;
`yyb` and `qqb` differentiate in `(y, ybar)` and `(q, qbar)`, respectively.
Folded coordinates differentiate the parity-smoothed block in their folded
variables, and radial slices keep its `n = 0` column. Torus `vvb` uses
`(v, vbar)`. `derivs` is jagged: row `m` lists
only the kept `n = 0, 1, ...` in increasing order (see the
[truncation table](#coordinates)).  A chiral table drops the `n` axis:
`derivs[c][m][d]`, with derivatives in `z`, `y`, or `q` as selected.
SDPB's positivity region is `x >= 0`.

| key | meaning |
|---|---|
| `derivs` | numerator coefficients, indexed as `index_names` says: `[coordinate_index][deriv_0][deriv_1][monomial_degree]` |
| `index_names` | the axis names of `derivs` |
| `exp_base` | base of the damped exponential; use the stored value. Sphere bases depend on the family, center, and table type as described below. Torus tables use `e^-2pi` |
| `pole_list_x` | the kept poles, in `x` |
| `variable`, `variable_minus_x` | the positivity variable (`"Delta"`; `"L"` on a tower; `"point"` for a `point-table`) and the shift that makes it `x` (the unitarity bound, raised by `--gap`) |
| `sample_points_x` | the interpolation nodes, in `x` |
| `sample_block_values` | `{coordinate: [m][n][k]}`: the written interpolant at `sample_points_x[k]`, from recursion samples or per-entry Lagrange interpolation |
| `sample_block_values_index_names` | its axis names |
| `sample_via` | how the node values were computed (`numeric` or `symbolic`) |
| `coordinates` | the coordinate of each `coordinate_index` |
| `basis`, `label` | the block basis (`scalar_G`, `ordered_B`, `global_scalar_G`, `ns_scalar_G`, ...) and its operator-labeling scheme |
| `scalar_G_spin0` | the spin-0 normalization of a scalar-`G` table (`*scalar_G` bases only): `"single"`, `G_{Δ,0} = F(h, z) F(h, z̄)` (see [block normalization](#--chiral-norm-n)). A scalar-`G` tower carries it too, but never holds a spin-0 member (its `LMIN = 0` is refused) |
| `scalar_G_odd` | `"even"` on every scalar-`G` table and tower (`*scalar_G` bases only): odd spins are `B(h, h̄) + B(h̄, h)`, `z ↔ z̄` even. Odd-spin scalar-`G` files and odd towers without it are refused |
| `central_charge` | `c` (`null` for the `global_*` bases) |
| `h_external`, `hbar_external`, `delta_external`, `spin_external` | the four external operators (on a chiral table only `h_external` is set; the others are `null`) |
| `spin` | the exchanged spin (the lower edge `LMIN` on a tower; `null` on a chiral table) |
| `parity_projection`, `parity_basis`, `parity_external`, `parity_exchanged`, `parity_smoothing` | parity data; `null` where it does not apply |
| `z_center`, `zbar_center` | the expansion point (default `0.5`; a chiral table has only `z_center`) |
| `z_shift`, `zbar_shift` | the kinematic exponent shifts (`--z-shift`, `--zbar-shift`) |
| `chiral_normalization_N` | the block normalization `N` |
| `lambda`, `order`, `npoles` | `--lambda`, `--order`, and the sphere pole cap (without a cap, the number kept); forced positive poles come on top. Torus one-point `npoles` counts all kept poles and `npoles_requested` records the cap on the remainder |
| `pole_selection` | the selector that chose them (order-reduced tables only) |
| `precision_input` | `--precision` |
| `precision_actual` | the working precision the coefficients were computed and written at (`--precision`, or above it with `--target-digits`) |
| `verified_digits` | digits to which the table matches its reference (worst entry, relative to itself) over `verified_delta_range`; the same `--order` except for torus one-point tables, which use a deeper reference |
| `verified_digits_largest_entry` | the same, relative to the largest entry at each check point |
| `verified_delta_range` | the Δ interval checked: between the nodes and past them to the large-Δ regime (global tables and towers: between the nodes) |
| `target_digits` | `--target-digits` (absent without it) |
| `q_truncation_digits` | the estimate of the `--order` truncation (absent for the global family, which has no q-series); torus one-point tables record it even with `--q-gate off` |
| `output_template`, `args` | the filename template and the shell-quoted command line, using the canonical `virasoro_blocks` entry point |

For sphere `table` output, define `E(z) = 16 q(z) / N` for Virasoro, NS, and
Ramond blocks, or `E(z) = 4 rho(z) / N` for global blocks. A fixed-spin table
in `Delta` uses `exp_base = sqrt(E(z_center) E(zbar_center))`; a chiral table
uses `E(z_center)` per unit of `h`. A large-spin tower instead uses a base per
unit of `L`: `E(z_center)` for `ordered_B`, and whichever of the two bases has
the larger magnitude for scalar and parity-projected towers. All reduce to
`E(1/2)` at the default center (`e^-pi` for Virasoro, NS, and Ramond at `N = 16`).
Sphere `point-table` output uses the global base for every family; this sets
the positive continuation away from the frozen point, where the table does
not represent the block at another exchanged weight.

Some tables carry more:

- A large-spin tower adds `twist` and `large_spin_lmin`, and records its
  `verified_digits` with the spin interval they cover, `verified_spin_range`
  (see [Large-spin tower](#large-spin-tower)).
- A `point-table` adds `point_value`, the frozen exchanged dimension (the
  chiral weight `h` on a chiral basis), with `variable: "point"`, degree-0
  `derivs`, an empty `pole_list_x`, and `sample_points_x: ["0.0"]`.  On the
  torus it is a `vvb` torus table of the module at `point_value` and `spin`
  (paired with its mirror); a character or one-point block on a Kac weight
  adds `kac_limit: "irreducible"` and `regularization_policy: "irreducible"`.
- NS and Ramond tables add `block_family`, `super_sector`, and
  `central_charge_evaluated`, plus `ramond_branch` on Ramond.  They also add
  `regularization_policy`, `regularization_eps_exp`, and `collision_policy`.
- Torus tables add `surface`, `spin_input`, `spin_pair_symmetrized`,
  `nome_convention`, and `crossing_point_tau`. They carry no sphere external
  weights or parity fields. One-point tables add `onept_d_ext`,
  `npoles_requested`, `fit_digits`, and `precision_digits`; their
  `q_truncation_digits` compares the requested q-order with a deeper
  reference. Their positive poles are recombined into the ordinary
  `pole_list_x` and numerator representation.

`--npoles all` tables have no sample points or node values; they are verified
like every table.

---

## Large-spin tower

The CLI can emit fixed-twist, **variable-spin-`L`** blocks — the same
`variable="L"` schema the bootstrap manifest writes — so CLI- and Python-API-
produced tower blocks are interchangeable. Each twist slice is one
`derivs_large_spin_{index}.json` file (the index runs over the resolved twist
list). This bypasses the fixed-spin Δ path entirely.

Setting **`--large-spin-lmin LMIN`** switches the CLI into tower mode. `LMIN` is
the lower spin edge: the spin-`L` interpolation is valid for `L ≥ LMIN`, and
`LMIN` becomes the block shift, recorded as `large_spin_lmin` (with
`variable="L"`). The block files also carry `twist`.

A `scalar_G` tower (either family) refuses `LMIN = 0`. The tower is
analytic in `L`, so its `L = 0` member is the doubled block `2 B(h, h)`, while
the spin-0 scalar `G` is `B(h, h)` (see the `scalar_G_spin0` field).  Spin 0
belongs in a fixed-spin table, and the tower starts above it, usually at the
first spin past the fixed-spin list.  The `scalar_G` tower is `z ↔ z̄` even and
covers every integer `L ≥ LMIN`; its files are labeled by the smallest integer
spin `≥ LMIN`. The `ordered_B` and `parity_projected_B` towers accept
`LMIN = 0`.

Twists come from two combinable sources (at least one is required):

- **`--large-spin-double-twists K`** — the leading `K` double-twist twists
  `τ_n^{12} = 2(h̄₁+h̄₂)+2n` and `τ_n^{34} = 2(h̄₃+h̄₄)+2n` for
  `n = 0 … K−1`, built from the antiholomorphic external weights. For scalar
  externals these are `Δ₁+Δ₂+2n` and `Δ₃+Δ₄+2n`, coinciding as
  `2Δ_ext + 2n` when the externals are identical.
- **`--large-spin-twists TAU ...`** — explicit isolated twist values, merged
  with the double-twist set. Only identical exact labels are de-duplicated:
  `0.7`, `.7`, and `7/10` identify the same twist, while a deliberate nearby
  shift produces a separate file. Generated twists use exact external-weight
  labels before numerical evaluation.

The normal family selector also applies to the tower: use
**`--family virasoro`** (which needs `--central-charge` and `--order`) or
**`--family global`** (exact SL(2)×SL(2) 2F1, independent of `c`; `--order`
sets the SL(2) pole pool). **`--large-spin-pole-selection`** is the tower's
own pole selector, separate from `--pole-selection`. Both tower families
default to `contribution_optimal`, using their fixed q-order candidate pool and
the optimal-node interpolation measure. The tower reuses
`--coordinates` (the non-folded `zzb` / `yyb` / `qqb` and the parity-folded
`xt` / `ws` / `qqb_ws`, including their `*_radial` slices), optional `--npoles`,
`--chiral-norm`, `--num-threads`, and `--shared-sample-points`. Emitted as
JSON only (`--format json`; the Mathematica writer has no `variable="L"`
schema). It refuses the fixed-spin flags it does not read: `--spin-ranges`,
`--pole-selection`, `--regularize`, `--sample-via`, `--gap`.

Radial coordinates retain derivatives with `n = 0` and `m = 0, ..., Lambda`
of the parity-smoothed block, for either parity. They support `scalar_G` and
`parity_projected_B` and can be written alongside the other coordinates in one
JSON file.

Twists and the `LMIN` anchor retain their exact input labels. At a singular
central charge, surviving pole orders come from the merged Laurent table.
Genuine poles inside the positivity region are kept at their full order;
small principal coefficients do not make them removable.

A tower is an order-reduced interpolation in `L`. Without `--npoles` its pole
count is chosen as for the global tables: it starts at the count the working
precision carries and grows by the measured deficit while the kept poles limit
the tower (a global tower enlarges its `--order` pole pool to match, recorded
as `order_used`). With `--npoles`, like a fixed-spin `--npoles` table, it is
built once. Either way it is checked against the
block it interpolates (the fixed-twist block, evaluated directly at spins
between its interpolation nodes): the file records `verified_digits` and the
interval checked, `verified_spin_range = [LMIN, LMIN + last node]`, with
`verified_digits_largest_entry`, and a note is printed when it falls short of
`--precision` (`--target-digits` raises its working precision when that is the
limit, and is refused when the order reduction is). Past the last node the tower
extrapolates and is not checked; there it can hold fewer digits (a
`c = 8`, `τ = 0.4`, `--npoles 4` tower verified to 13.6 digits on
`[10, 12.9]` agreed with direct `point-table` values to 12 digits at `L = 14` and
9.5 at `L = 60`). Virasoro towers also record `q_truncation_digits` from the
q-order gate.

**Block basis (spinning / fermionic externals).** The tower mirrors the
fixed-spin `--basis auto`: scalar externals give `scalar_G`; spinning externals
(`h_i != hbar_i`, integer or half-integer spin — fermionic operators are
half-integer) give `ordered_B`, or `parity_projected_B` with `--parity-project`
`+`/`-` (and optional `--parity-ext` Z4 phases). Scalar externals also admit
an explicit `--basis ordered_B` to keep a single spin orientation.
The holomorphic half carries `L`, the antiholomorphic half is the fixed
twist; `ordered_B` / `parity_projected_B`
admit half-integer exchanged `L`. The double-twist auto-tower
(`--large-spin-double-twists`) uses the *antiholomorphic* external weights,
`tau_n^{ij} = 2(hbar_i + hbar_j) + 2n` (= `Delta_i + Delta_j + 2n` for scalars).
The tower currently uses a single symmetric shift, so a `--zbar-shift` differing
from `--z-shift` is rejected.

**`--parity-exchanged`** for `scalar_G` accepts only `+1` (the default): the
scalar `G` is `z ↔ z̄` even at every spin, so one tower covers the even and odd
spins `L ≥ LMIN` alike (odd-spin operators appear only for nonidentical
externals), and `-1` is refused (the `z ↔ z̄`-odd structure is
`parity_projected_B` with projection `-`). For `parity_projected_B` it feeds
the reversed-block phase and may be any Z4 phase (`i`/`-i` are the
fermionic-exchanged phases).

```bash
# Virasoro tower at c = 2.6, three double-twist slices, L ≥ 12
virasoro_blocks table --family virasoro --basis scalar_G \
  --central-charge 2.6 --lambda 4 \
  --delta-ext 0.125 0.125 0.125 0.125 --order 30 --npoles 12 \
  --large-spin-lmin 12 --large-spin-double-twists 3 --coordinates zzb -o out/
# -> out/derivs_large_spin_{0,1,2}.json
```

---

## Positivity anchor: `--gap G`

Raises the emitted table's positivity anchor to `G` above the basis unitarity
bound (`|ℓ| + G` for the 2d bases, `G` for chiral). The polynomial variable is
`x = Δ − shift` and SDPB imposes `x ≥ 0`, so this is how a gap is written into
a block file. It is unrelated to `--z-shift`, which moves the kinematic
z-exponent. Honored on the Virasoro, global, NS, and Ramond sphere families and
on torus one-point tables (every spin at `|j| + G`; exclusive with
`--torus-scalar-shift`); rejected for torus character tables and the
large-spin tower (whose positivity variable is `L`). Default: no gap.

On the sphere `G` must be a real number `>= 0`: a negative gap would anchor the
table below unitarity and is refused with an error. (A torus one-point table
refuses a spin-0 anchor below `0`.)

---

## Torus tables

`--surface torus` emits genus-1 tables instead of four-point blocks, one
JSON file per integer spin `j ≥ 0` on the `vvb` coordinate (JSON only).
`--family torus-character` (the default on the torus) gives exact reduced-character
tables at any real `c` (basis `torus_G`); `--central-charge`, `--spin-ranges`, `--lambda`,
`--precision`, `-o`, and `--output-template` apply, and the sphere-only flags
(`--order`, `--npoles`, external weights, parity, `--basis`, `--coordinates`,
shifts and centers, the large-spin tower) are rejected.  A table is the
character of the non-degenerate (Verma) module, continuous in `Δ`.  A module
on a Kac weight has the irreducible character instead, written by
`point-table --surface torus` ([below](#torus-point-tables)): the vacuum at
every `c`, the currents `(j, 0)`, and at `c <= 1` the Kac weights on the
half-line (the minimal-model primaries, ...), which is why a `c <= 1`
character table prints a note saying so.

`--torus-onept-ext D` (or `--family torus-onepoint`, which requires it) gives
torus one-point tables (basis `torus_onept_G`) for a scalar external of chiral
weight `D` (`Δ_Φ = 2D`), and re-enables `--order` (the q-series truncation),
`--npoles` and `--q-gate`. These tables support any real `c`.
Recognized singular charges (`c = 1`, minimal models, `25`, `26`, `33`,
`50`, ...) use the exact Laurent kernel. As on the sphere, every real Kac pole
at positive `x` is split off as an exact principal part before interpolation
and recombined into the stored rational function. Repeated poles use their
merged orders. `--npoles` caps the remaining poles at nonpositive `x` (or
complex locations), nearest the positivity half-line first, with conjugate
pairs together; all positive poles come on top, even for `--npoles 0`.
Without `--npoles` the cap is chosen per spin, as for the global tables: the
first build keeps 4 remainder poles, and the count grows by the measured
deficit while the kept poles limit the table. The JSON records it as
`npoles_chosen` (pass it as `--npoles` to rebuild the same table).

The anchor (the positivity edge `x = Δ − shift = 0`): `--gap G` anchors every
spin at `|j| + G`, as on the sphere; `--torus-scalar-shift X` anchors spin 0
alone at `X` (the other spins at `|j|`). Both set the spin-0 anchor, so passing
both is refused, as is `--torus-scalar-shift` without spin 0 in
`--spin-ranges`. The spin-0 anchor defaults to 0 and must be `>= 0`; its double
Kac pole at `Δ = 0` stays in the prefactor (the first node is lifted off it).

One-point tables are order-reduced interpolations, built at the working
precision `--precision` (as a sphere `--npoles` table is; `--target-digits`
raises it when the precision is the limit). As on the sphere, entry `(m, n)`'s
remainder has numerator degree `K + m + n`, where `K` is the number of kept
remainder poles, and is fit on its own Bohr–Sommerfeld node set. After
recombining the positive principal parts its numerator degree is the total
kept pole count plus `m + n`, preserving the block's large-Δ power `Δ^(m+n)`.
Sampling nodes are moved off positive poles.

Each table is verified against a deeper (higher `--order`), higher-precision
reference at points between its
nodes, on both sides of representative positive poles, and far past the last
node (out to 1024 times the larger of that node and the largest positive
pole), since a bootstrap reads it at every `Δ ≥ shift`. The file records what
it delivers: `verified_digits`
overall, and its parts `fit_digits` (the order reduction and the fit's
cancellation — raise `--npoles`, or `--precision` when far kept poles cost
digits past the nodes), `precision_digits` (the recursion's cancellation,
large for large `D` — raise `--precision`) and `q_truncation_digits` (the
`--order` tail — raise `--order`); a shortfall is printed as one line naming the
limiting part, with the same note/warning split as the sphere tables.  Character
tables are exact: they are verified against the same table at more digits.
The check records its span as `verified_delta_range`;
past the nodes the relative error settles to a constant, which the farthest
points measure. `--q-gate` (default `warn`) warns when `q_truncation_digits`
falls below 20 (or `--precision`); `strict` refuses the table.

`npoles` is the total number of poles the table keeps, including the forced
positive poles
(`npoles_requested` is the requested cap on the remainder poles). The
node values are written as `sample_block_values` at `sample_points_x` (the
highest-degree entries' nodes): the block's values for the `m + n = λ`
entries, and each lower entry's own interpolant there, as on the sphere (see
`--shared-sample-points`). At spin `j ≥ 1` the default anchor `Δ = j` carries
the `h̄ = 0` Kac pole at `x = 0`; it is kept in the prefactor with the lowest
node lifted off it, as the sphere keeps a conserved-current pole.

On the torus, `--pole-selection`, `--sample-via`, `--shared-sample-points`,
`--regularize`, `--num-threads`, the super-sector/Ramond-branch flags, and the
large-spin flags are refused rather than ignored; character tables also refuse
`--q-gate` and `--torus-scalar-shift`.

```bash
virasoro_blocks table --surface torus --central-charge 10 --lambda 2 \
  --spin-ranges 0 2 -o out/torus_character/
virasoro_blocks table --surface torus --torus-onept-ext 0.2 \
  --torus-scalar-shift 0.1 --central-charge 10 --lambda 2 \
  --spin-ranges 0 1 --order 8 --npoles 4 -o out/torus_onept/
```

### Torus point tables

`virasoro_blocks point-table --surface torus` freezes a torus block at one
module: `--exchange-delta D` and `--exchange-spin j` (an integer `j >= 0`)
give `(h, h̄) = ((D + j)/2, (D − j)/2)`, paired with its mirror `(h̄, h)` for
`j > 0` exactly as the `torus_G` tables pair `j` with `−j`.  The file is a
`vvb` torus table with `variable: "point"` and `point_value: D`, degree-0
numerators and no poles, written as `derivs_point_{spin}.json` by default
(JSON only).

- `--family torus-character` (the default with `--surface torus`) writes the
  reduced character of that module, with the **irreducible** character
  whenever `h` or `h̄` is a Kac weight: the vacuum `(0, 0)` (degenerate at
  every `c`) is the modular bootstrap's identity vector
  (`torus_vacuum_table`), a current pair `(j, 0) + (0, j)` is its current
  constraint (`torus_current_table`, whose vacuum half carries `1 − q̄`), and
  at a recognized singular `c` (minimal models, `c = 1`, `c = −2`, the
  `c >= 25` rational loci) every Kac weight, inside the Kac table or on its
  boundary, has its exact chain or braid (Rocha-Caridi) character, cut where
  the null series falls below the working precision.  Off the Kac weights
  the point table is the `torus_G` table evaluated at `D`.  The character is
  exact, so `--order`, `--q-gate`, and `--regularize` do not apply.
- `--family torus-onepoint` (or `--torus-onept-ext D_ext` with
  `--surface torus`) writes the one-point block, with `--torus-onept-ext` and
  `--order` as for the tables.  On a Kac weight of a recognized singular `c`
  it is the trace over the irreducible module, as `evaluate` gives it (an
  external the fusion rules forbid is refused); a Kac weight at a generic
  `c` is a pole of the block and is refused.  The table is verified against
  a deeper, higher-precision reference, which also measures the `--order`
  truncation (`q_truncation_digits`, gated by `--q-gate`).

The file records `regularization_policy` (`irreducible` on a Kac weight,
else `direct`, or `laurent` for a one-point block at a singular `c`) and,
on a Kac weight, `kac_limit: "irreducible"`.  Sphere flags (externals,
`--coordinates`, `--basis` other than the family's own, parity, shifts,
centers, `--kac-limit`, `--super-sector`, `--ramond-branch`) are refused, as
is a half-integer spin.  `--precision` may go down to 10 here.

```bash
# the Ising sigma character (h, hbar) = (1/16, 1/16): -> out/derivs_point_0.json
virasoro_blocks point-table --surface torus --central-charge 1/2 \
  --exchange-delta 1/8 --lambda 10 -o out/
# the spin-2 current pair at c = 12 (torus_current_table): -> out/derivs_point_2.json
virasoro_blocks point-table --surface torus --central-charge 12 \
  --exchange-delta 2 --exchange-spin 2 --lambda 10 -o out/
# a one-point block: -> out/onept.json
virasoro_blocks point-table --surface torus --family torus-onepoint \
  --torus-onept-ext 1/5 --central-charge 5/2 --exchange-delta 17/10 \
  --lambda 6 --order 20 -o out/ --output-template onept.json
```

---

## `evaluate` and `point-table`

`virasoro_blocks evaluate` evaluates one block at an explicit cross-ratio point
rather than emitting derivative tables:

- `--family` (`virasoro` by default, or `global`, `ns`, `ramond`,
  `torus-character`, `torus-onepoint` with `--surface torus`), `--basis`, and
  one `--coordinate` (`zzb` by default; `z`/`y`/`q` for chiral bases, `vvb` on
  the torus);
- `--point NAME=VALUE`, repeated once per component (e.g.
  `--point z=0.3 --point zbar=0.4`);
- the exchanged operator as `--exchange-delta D` / `--exchange-spin L`, or
  `--exchange-h H` / `--exchange-hbar HB`;
- externals, parity, super-sector, `--z-shift`/`--zbar-shift`/`--u-shift`,
  `--order`, `--precision`, `--regularize`, and `--kac-limit` as for `table` (scalar_G carries
  one shift, so `--z-shift S` alone gives `(z zbar)^S G`, the value its tables
  expand); on the torus, `--torus-onept-ext D` for `torus-onepoint` (the
  table-only anchor `--torus-scalar-shift` does not exist here);
- `--format {json,text}` (default `json`) and `-o FILE` (default `-`, stdout).
  The JSON records the shifts and `--chiral-norm` alongside the value.

An option the chosen block does not read (a torus shift, `--parity-exchanged`
without a parity projection, `--super-sector` off NS/Ramond, `--ramond-branch` off Ramond, a point name the
coordinate does not use) is refused rather than ignored. A real point on the cut
`z >= 1` is taken on the Euclidean section `zbar = z*`: `z = x + i0`,
`zbar = x - i0`, so `z = zbar = x > 1` gives the real Euclidean value.

At a recognized singular `c`, an exchanged weight exactly on a Kac weight where
poles collide takes the block of the irreducible Virasoro module, the one a CFT
at that `c` uses (`--kac-limit irreducible`, the default). The limit in the
weight instead is the Fock-module block at `c = 1`, off by the null submodule
at `h = 4, 9, ...`; the `c -> c0` limit at fixed weight (`--kac-limit c_limit`)
agrees with the irreducible block at the minimal models and at `h = 0`, but at
`c = 1` and `c = 25` keeps admixtures of the null submodule from level
`2(m+2)` on at `h = m²/4`. Off the Kac weights the value is the block at `c0`,
continuous in `Delta`, which is the function the tables carry; the tables
therefore do not represent the jump at an exact Kac weight. The bootstrap's
identity contribution is the vacuum block (`h = 0`, where all three agree).

NS and Ramond values on a Kac weight of an N=1 singular `c` take the
`c -> c0` limit at fixed weight, read exactly off the Laurent recursion; a
block whose limit diverges there (a genuine pole for these externals) is
refused. That limit is checked against the free boson + fermion at `c = 3/2`
only at the vacuum (`Delta = 0`); whether it is the irreducible
superconformal block on a nonzero Kac weight is unverified (the Virasoro
analogue keeps null-submodule admixtures there), and `--kac-limit` does not
apply to these families.
Torus values on a Kac weight of a singular `c` are the trace over the
irreducible module, which neither limit of the recursion gives (the
near-null states keep a finite diagonal matrix element; at `D = 0` the
recursion is the Verma character). `torus-onepoint` builds that trace exactly
(Ising `<epsilon>` on `sigma` is `q^{1/24} prod(1 - q^n)`) and refuses an
external that does not decouple from the null states (a fusion rule forbids
the block); `torus-character` gives the irreducible character on any Kac
weight (the minimal-model and `c = 1` characters, the vacuum's `1 - q`).
The metadata then records `regularization_policy: "irreducible"`.

Virasoro, NS, and Ramond values are checked for q-truncation at the evaluation point: the JSON
records `q_truncation_digits`, the digits the `--order` truncation leaves of the
untruncated block, and `--q-gate {off,warn,strict}` (default `warn`) warns when
they are fewer than 20 (fewer than `D` with `--target-digits D`) or when the series
has not converged; `strict` makes either an error and `off` skips the check. The
order needed grows with c (at c = 10^5, order 80 is still short at z = 0.4), and
raising `--precision` cannot compensate.

`--precision` is the working precision here too: every value is the `--order`
block computed at `P` digits, and `value` prints all of them. How many of them
are digits of the block itself is `estimated_digits`, which every result
records: `P` less the measured conditioning loss (a value near a pole or
`z = 1`, or cancellation between block components), and never more than
`q_truncation_digits`. Such a value also warns when it keeps fewer than `P/2`
digits and is refused when it keeps none (naming the precision that fixes it). A `torus-onepoint` value is
also checked against a deeper, higher-precision reference: the result records
`verified_digits` and `precision_actual`, a shortfall is reported with the
precision that fixes it, and its `estimated_digits` is the smaller of
`verified_digits` and `q_truncation_digits`. A torus one-point value whose
q-truncation leaves fewer than `P` digits (fewer than `D` with
`--target-digits D`) gets a stderr warning.
An exact symmetry zero, such as a parity-odd block at `z = zbar` inside `(0, 1)`,
is returned as zero without precision escalation. Nearby points are resolved
from their original input labels when precision is increased.

`--target-digits D` (and `--max-precision M`) work for every family: they raise
the working precision by the measured deficit until the `--order` block holds
`D` digits (`precision_actual` records where it stopped). The q-truncation is
not a precision deficit, so it is not escalated: the q-gate checks it against
`D` instead (raise `--order`). For torus one-point values, `--order` must be at
least 1, and a recognized singular charge (`c = 1`, minimal models, `25`, `26`,
...) uses the exact Laurent kernel (`regularization_policy` `laurent`).

`virasoro_blocks point-table` builds a frozen derivative table for a single
exchanged dimension — the bootstrap point-constraint operation, not a
cross-ratio evaluation. Every family has one. On the sphere it takes the
`table` kinematics flags (`--family virasoro|global|ns|ramond`, `--basis`,
`--coordinates`, `--lambda`, externals, parity, shifts, `--chiral-norm`,
`--format`, `-o`; `--super-sector` and `--ramond-branch` for the N=1
families) with the exchanged
operator given as `--exchange-delta` / `--exchange-spin` (or `--exchange-h`
for a chiral basis), and an off-symmetric center as `--center z=VALUE`. It
honors `--regularize` and `--output-template`; the default output is
`derivs_point_{spin}.json`, with `{spin}` the exchanged spin (`0` on the chiral
path). A Virasoro point table runs the q-order gate at its own exchanged
weight(s) (`--q-gate {off,warn,strict}`, default `warn`, as for `table`), an
NS or Ramond one the probe their tables run, and each records the estimate as
`q_truncation_digits`; the global family accepts only
`--q-gate off`.  An NS or Ramond point table at an N=1 singular `c` is the
`c -> c0` limit at fixed weight (`regularization_policy: "merged"`; there is
no `--kac-limit` for these families) and records the NS/Ramond family
metadata of their tables.  A point table is built at the working precision `--precision`
and verified against the same table at more digits (`verified_digits`);
`--target-digits` / `--max-precision` work as for `table`.  The torus families
have their own point tables ([Torus point tables](#torus-point-tables)).

The stored numerator polynomials are constant in the exchanged dimension.
The reconstructed table equals the requested derivative grid at `point_value`;
elsewhere it equals that same grid multiplied by
`base^(Delta - point_value)` (use `h` for a chiral table). This positive factor
preserves the point constraint's inequality. Evaluating the frozen table at
another dimension does not give the conformal block there; build another
point table or use an ordinary interpolated table for that purpose.

```bash
virasoro_blocks evaluate --central-charge 10 --delta-ext 0.15 0.15 0.15 0.15 \
  --exchange-delta 2 --exchange-spin 2 --point z=0.3 --point zbar=0.4
virasoro_blocks point-table --central-charge 10 --lambda 2 \
  --delta-ext 0.15 0.15 0.15 0.15 --exchange-delta 2 --exchange-spin 2 -o out/
# -> out/derivs_point_2.json
```

Run `virasoro_blocks evaluate --help` or `virasoro_blocks point-table --help`
for the complete lists.

---

## Compatibility rules (summary)

The CLI rejects combinations that violate these rules:

| rule | rationale |
|---|---|
| `--delta-ext`/`--spin-ext` ⊥ `--h-ext`/`--hbar-ext` | two parametrizations of the same data |
| `--h-ext` on 2d bases needs `--hbar-ext` | the 2d block has two halves |
| `--parity-project` needs `--parity-ext` | projection is convention-dependent on operator phases |
| `--parity-project` only with spinning externals | scalar externals already use `G` |
| `--parity-exchanged` only with `--parity-project` (on the large-spin `scalar_G` tower only `+1`, the default) | otherwise it has nothing to attach to; the scalar `G` is `z ↔ z̄` even at every spin |
| folded coordinates with spinning externals need `--parity-project` | parity-folded coordinates require a `z ↔ z̄` eigenstate |
| folded coordinates need a real (`±1`) total parity phase | non-real total phase has no definite `z ↔ z̄` parity |
| chiral coordinates require `--basis chiral_F` or `global_chiral_F` | the rest of the bases consume 2-axis tables |
| `--basis chiral_F` rejects `--spin-ext`, `--parity-*`, `--hbar-ext`, `--zbar-shift`, `--zbar-center`, a nonzero `--spin-ranges`, 2d coordinates | chiral block has no antiholomorphic half / no exchanged spin |
| `--super-sector` only with `--family ns`/`ramond`; `--ramond-branch` only with `ramond` | the other families have no sector or sign branch |
| `--pole-selection` needs an integer `--npoles` (fixed-spin global bases excepted) | with every pole kept there is nothing to select |
| `h_i − h̄_i` must be int/half-int | required by the JSON `spin_external` serialization |

Auto-inference (when `--basis auto`): only chiral coords → `chiral_F`; with
`--parity-project` → `parity_projected_B`; with any nonzero external spin →
`ordered_B`; otherwise → `scalar_G`.
