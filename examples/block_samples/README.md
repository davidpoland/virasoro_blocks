# Block-table samples

These are small, committed outputs from `virasoro_blocks table`. They are
reference files for consumers of the JSON and Mathematica formats, not
production-accuracy tables.

| Directory | Family and basis | What it demonstrates |
|---|---|---|
| `scalar_G/` | Virasoro `scalar_G` | identical scalar external operators |
| `ordered_B/` | Virasoro `ordered_B` | spinning external operators |
| `parity_projected_B/` | Virasoro `parity_projected_B` | parity projection |
| `chiral_F/` | Virasoro `chiral_F` | one-variable chiral tables |
| `ns_scalar_G/` | N=1 NS `scalar_G` | superconformal table schema |
| `ramond_scalar_G/` | N=1 Ramond `scalar_G` | Ramond-sector tables |
| `offdiagonal_scalar_G/` | Virasoro `scalar_G` | independent `z` and `zbar` centers |
| `torus_G/` | torus `torus_G` | reduced character table |
| `torus_onept_G/` | torus `torus_onept_G` | one-point tables, with the pole count chosen |
| `global_scalar_G/` | global `global_scalar_G` | SL(2)×SL(2) scalar limit |
| `global_ordered_B/` | global `global_ordered_B` | spinning global blocks |
| `global_parity_projected_B/` | global `global_parity_projected_B` | global parity projection |
| `global_chiral_F/` | global `global_chiral_F` | global chiral tables |
| `large_spin_scalar_G/` | Virasoro large-spin tower | tables with `variable = "L"` |
| `global_large_spin_scalar_G/` | global large-spin tower | global variable-spin tables |
| `point_table/` | several families | blocks at a single exchanged weight (`virasoro_blocks point-table`) |

The committed ordered examples use spinning externals. The ordered bases also
accept scalar externals when selected explicitly: they retain one spin
orientation and both derivative orientations on `zzb`, `yyb`, or `qqb`.
`auto` chooses `scalar_G` for scalar externals. Torus one-point tables are
supported at any real central charge, including positive-pole interpolation
at `c <= 1`. See the [CLI reference](../../docs/CLI.md#torus-tables).

`point_table/` holds one table at a single exchanged weight for each kind of
point table: a Virasoro `scalar_G` block (JSON and Mathematica), a torus
one-point block, and three torus characters. The characters are irreducible
and are chosen from the quantum numbers: the vacuum (`torus_vacuum.json`), the
current pair `(1, 0) + (0, 1)` (`torus_current.json`), and the Ising spin field
on a Kac weight at `c = 1/2` (`torus_ising_sigma.json`).

The Virasoro sphere samples use `c = 10`, derivative bound `--lambda 4`,
`--order 20`, `--npoles 6`, and 60 decimal digits. They are expanded at the
crossing-symmetric point unless the directory name says otherwise. The fixed-
spin global samples and global large-spin towers omit central charge; their
`--order 20` defines the finite SL(2) pole universe; the 2F1 evaluations are not
truncated at that order. The NS, Ramond, torus, and point-table examples are
compact samples with their own settings, recorded in `args`.

Sphere JSON tables record their basis, kinematics, coordinates, precision,
interpolation points (`sample_points_x`), the block values at those points
(`sample_block_values`), pole selection, prefactor data, and command-line
provenance. Torus character tables are constructed without interpolation and
omit the sample-point and sampled-value fields. High-precision numerical values
are strings so that precision survives serialization. The Mathematica files
contain the corresponding polynomial data and metadata.

## Regenerating

`tools/regen_block_samples.sh` defines and regenerates these examples:

```sh
tools/regen_block_samples.sh
tools/regen_block_samples.sh scalar_G ordered_B
```

Regeneration overwrites the selected sample directories. Review the resulting
diff before committing: a change in a sample is a change in the user-visible
file format or its numerical reference data.
Only the selected directories are validated. Unknown target names are rejected
before any sample is regenerated.

## Reading a sample

For a fixed-spin JSON table, the derivative data are stored under `derivs` and
the polynomial variable, shift, poles, and exponential base are recorded in
the metadata. A large-spin sample instead sets `variable` to `"L"` and stores a
twist slice. Consult [the CLI reference](../../docs/CLI.md) for the controls
that determine a table's shape.
