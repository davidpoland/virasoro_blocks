# Known-CFT moments

This directory contains the normalized moments of known 2d CFT four-point
functions that the accompanying paper compares with bootstrap bounds. They were
computed with research code that is not part of this release, following the
paper's appendix on known models. The bounds themselves can be reproduced with
`virasoro_bootstrap`.

All ratios are `nu_1/nu_0` in the paper's normalization, reported with the
Virasoro identity included and excluded. Central charges and external dimensions
are exact rationals written as `p/q`. Irrational parameters such as crossing
radii are decimal strings.

## Files

| File | Contents |
|---|---|
| `points.json` | every registered isolated point: `c`, `delta_ext`, `nu1_over_nu0_include_identity`, `nu1_over_nu0_exclude_identity`, `label`, `series`, `status`, `crossing_residual` |
| `mixture_intervals.json` | estimated moment ranges over real mixtures of degenerate scalars, in the two lists described below |
| `overlays/liouville_scalar_moments_l19.json` | Liouville theory (DOZZ) scalar moments on the bound grids |
| `overlays/c1_*.json` | the `c = 1` catalog: Runkel–Watts continuum, the finite-`c` DOZZ reference, and the model catalog used near `c = 1` |
| `overlays/ising_x_*.json`, `overlays/*_mixtures*.json` | the continuous families and mixture families at `c = 3/2` and `c = 6/5` |
| `overlays/heavy_external_moments.json` | model rows with `2 < delta_ext <= 4` at every central charge, and all rows at `c = 2, 4, 8`, grouped as `by_c` → series → `rows` |
| `overlays/c1_heavy_external_overlays.json` | `c = 1` above `delta_ext = 2`: the untwisted vertex curve, Runkel–Watts, and the momentum–winding mixtures |
| `manifest.json` | schema, generation date, and normalization, with the SHA-256, row count, and source of every JSON data file |

`mixture_intervals.json` contains two lists:

- `mixture_intervals`: `include` and `exclude` are `[min, max]`. Each entry also
  records `theory`, `c`, `delta_ext`, `series`, `parameter`, `status`, and the
  quartic-form fit's `holdout_residual`.
- `n2_k2_mixture_intervals`: two-field mixtures of the `N = 2`, `k = 2` theory
  at `delta_ext` = 1/8, 9/8, and 2. Here `include` and `exclude` are
  `{min, max, t_min, t_max}`, where `t` is the first basis field's squared
  amplitude at the extremum. These entries omit `series`, `parameter`, `status`,
  and `holdout_residual`.

In `heavy_external_moments.json`, pure rows carry `ratio_include` and
`ratio_exclude`. Degenerate spaces (`"kind": "interval"`) carry `include` and
`exclude` as `{min, max}`.

At `c = 2, 4, 8`, free-boson curve blocks include `curve_accuracy`: the domain,
interpolation tolerance, and cutoff tolerance. Vertex curves extend to the
limit at `delta_ext = 0`, and current dressings start at `delta_ext = 2`.
An endpoint row marked `right_limit` is the limit from above and keeps the
exchanges whose dimensions tend to zero separate from the identity. An `included`
endpoint is evaluated at the endpoint itself. Figures draw `right_limit`
endpoints as open circles and `included` endpoints as filled circles. A
block's `mixture_partner` names the other boundary of a continuous mixture
band, drawn on the union of the two curve grids.

Each Ising block has an `operator_spaces` inventory through `delta_ext = 4`,
checked against the character decomposition. `full` identifies a computed
interval over the complete space, `attained_families` an interval spanned by
the attained endpoints of compact operator families in the complete space,
and `single_operator` a one-dimensional space.

The paper's c = 2, 4, and 8 Ising overlays show one bar for each degenerate
space and a star for an isolated operator. Individual reference operators
remain in the data even when their markers are absorbed into a range, and every
reference point lies within its range. Independent character counts verify all
222 primaries in 18 spaces of Ising⁴ and all 22,161 primaries in 32 spaces of
Ising⁸ through dimension 4. Ising¹⁶ has ranges in all 32 spaces through
dimension 4. Above Δ = 9/4 they are labeled `attained_families`. Their
endpoints are the most extreme values attained by compact operator families,
so each bar lies inside the full range. These rows have
`backend: compact_endpoint_families`, and `endpoint_sources` names the family.

Each interval row records how it was computed:

- `backend` names the operator space that was searched, for example
  `sigma_contractions` (every coefficient of the pure-sigma space), the
  `*_descendants` backends (product and descendant primaries in one mixture),
  or `chiral_descendants` (kernels built from chiral series). A `full` interval
  varies every coefficient of its space, and `contraction_geometry`, where
  present, records how symmetry reduces the evaluation.
- `optimizer` records the starting points, seed, tolerances, and iteration
  limit, and `gates` the cutoff, crossing, normalization, positivity, and
  gradient checks. `status: shipped` means the row passed them.
- `endpoint_checks` records spectral positivity and full-space curvature at
  the extrema, and `independent_search_agreement` the spread between separate
  searches.
- `endpoint_enclosures`, where present, gives `[lower, upper]` bounds on an
  endpoint from its attained value and a numerical sum-of-squares bound, with
  exact rational Gram witnesses in `global_bound_sources` where recorded.
  `endpoint_estimates` gives the plotted midpoint of an enclosure, and
  `figure_accuracy` its uncertainty on the paper's axes (`error_pixels`, with
  `reference_kind` saying how it was measured). Endpoints without a global
  enclosure are plotted at their attained values. The `include` and `exclude`
  fields always hold the attained values.

## Status and numerical checks

Every point has `status: certified`, meaning that the construction is exact and
the cutoff ladder has converged. For mixtures, `status` is `full` if the
backend evaluates every real mixture in the degenerate space, or `reduced` if a
stated continuous symmetry maps every mixture into the evaluated subspace. `crossing_residual`
records the independent crossing check, or the ladder and fit diagnostics when
the spectrum was fitted.

Mixture extrema are estimated numerically using multiple starts, gradient
residuals, and dense sampling checks. The `full` and `reduced` statuses describe
coverage of the operator space. The numerical checks do not prove global
optimality. Similarly, `certified` records the construction and convergence
checks and does not give a rigorous bound on the omitted spectral tail.

