# Test tiers

The tests are grouped by ownership:

- `tests/core/` — the public `virasoro_blocks` engine and its block tables,
  recursion, coordinates, interpolation, serialization, and CLI.
- `tests/bootstrap/` — `virasoro_bootstrap` crossing, positivity, manifests,
  moment bounds, and problem-construction code.
- `tests/integration/` — HPC transport and tasks.
- `tests/oracles/` — independent first-principles oracles (for example a direct
  Verma-module trace for the torus one-point block) that other tests import;
  it holds no tests of its own and has no ownership marker.

## The slow tier is opt-in

`pytest -q` runs everything **except** the slow tier.
The slow tier adds high-precision block and bootstrap checks. Use the default
run for ordinary edits; its summary lists the skipped tests.

Opt in deliberately, with either spelling:

```sh
pytest -q --slow            # everything
VB_TEST_SLOW=1 pytest -q    # same, for wrappers and CI that cannot pass argv
```

Tests are distributed with `--dist worksteal` (set in `pyproject.toml`), so an
idle worker takes queued tests instead of waiting on a busy one.  An autouse
fixture resets `mp.mp.prec` for every test, so a test may set precision with
`mp.mp.dps = ...` without leaking it into the next one.


Run the slow tier before a release or a broad merge, and after changing the
recursion or the precision layer.

If a test outside the slow tier takes more than 20 seconds, the run ends with
an `unmarked slow tests` report naming it.  That means a new expensive module
needs adding to `SLOW_TEST_FILES` in `tests/conftest.py`; the list is
hand-maintained to keep the default run fast.

The ownership markers mirror those directories: `blocks`, `bootstrap`, and
`integration`. Slow and integration are independent overlays, so a slow
bootstrap test can be selected with
`-m "bootstrap and slow" --slow` when needed.

The markers are assigned by `tests/conftest.py` at collection time from the
ownership directories.  Run a focused tier when changing a specific part, for
example:

```sh
pytest -q -m blocks
pytest -q -m bootstrap
pytest -q -m slow --slow
pytest -q -m integration
```

The ownership groups can also be run directly, for example
`pytest -q tests/core/`.

The fast tier is a feedback loop, not a substitute for the full suite: the
slow tier contains the checks that catch normalization, crossing-residual, and
high-cutoff regressions.
