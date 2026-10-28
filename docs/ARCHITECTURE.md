# Architecture

The project has two importable packages.  The dependency arrow points toward
the package being used:

```text
virasoro_bootstrap ──> virasoro_blocks
```

The block engine must not import the bootstrap layer.

| Package | Responsibility | Supported entry point |
|---|---|---|
| `virasoro_blocks` | blocks, coordinates, recursion, interpolation, serialization, CLI | `virasoro_blocks.api` |
| `virasoro_bootstrap` | crossing equations, positivity problems, SDP/PMP I/O, and optional HPC helpers | `virasoro_bootstrap.api` |

The computed moments of known CFTs shown in the accompanying paper are provided as
data in [data/known_cft_moments](../data/known_cft_moments/).

The supported Python entry points are `virasoro_blocks.api` and the
`virasoro_bootstrap` package root (with its `virasoro_bootstrap.api` subset).
The `virasoro_blocks` package-root names are kept for compatibility only, and
implementation modules may change without a deprecation cycle.

## Repository layout

- `src/` contains the installable packages.
- `examples/` contains runnable, user-facing demonstrations and committed
  block-table samples.
- `tests/` contains the library test suite.
- `docs/` contains release documentation.
- `tools/` contains the small set of helpers required to validate a wheel and
  regenerate committed samples.

Generated runs, caches, and local environments are not versioned.
