"""Test-tier and ownership classification for the numerical suite.

The slow tier is opt-in: a bare ``pytest`` skips it and reports how to get it
back.  That tier runs the high-precision block and bootstrap checks, so running
it is a deliberate choice rather than the cost of every edit-loop check.  Pass
``--slow`` (or set ``VB_TEST_SLOW=1``) for release validation, or after touching
the recursion or precision layers.

Ownership markers make it possible to run the reusable block engine
independently from the bootstrap layer; the integration overlay reserves
external-workflow checks.
"""

import os
from pathlib import Path

import mpmath as mp
import pytest

# A replacement production-parameter file is read when virasoro_bootstrap is imported, before any fixture runs, so
# drop it here: the suite pins the shipped default_params.json (tests that need an override set it themselves).
os.environ.pop("VB_DEFAULT_PARAMS", None)


@pytest.fixture(autouse=True)
def _restore_mpmath_precision():
    """Give every test the default working precision and restore it afterwards.

    Hundreds of tests set ``mp.mp.dps`` globally; without this a test's result depended on which test
    ran before it on the same worker (import-time constants, cached rows keyed on precision).
    """
    prec = mp.mp.prec
    mp.mp.prec = 53
    yield
    mp.mp.prec = prec


@pytest.fixture(autouse=True)
def _no_ambient_cluster_profile(monkeypatch):
    """Run every test without the shell's ``$VB_CLUSTER`` (the default ``--cluster`` profile of the hpc CLIs).

    On the cluster ``~/.bashrc`` exports it, and a batch job inherits it: the hpc CLI tests then read that profile
    instead of their fixtures (28 failures on cluster compute nodes, 2026-10-12, all ``cannot read`` the profile file).
    A test that needs the variable sets it with ``monkeypatch.setenv``.

    The same holds for the output roots a cluster profile exports (its ``[env]`` table). With
    ``$VB_SCRATCH_ROOT`` set, the worker tests put their task directories on shared scratch under names taken from
    pytest's temporary directories, so a second session found the first session's results (4 failures on cluster
    compute nodes, 2026-10-26).
    """
    for name in ("VB_CLUSTER", "VB_ARTIFACT_ROOT", "VB_CERTIFICATES_ROOT", "VB_EXPERIMENT_ROOT", "VB_RESULTS_ROOT",
                 "VB_RUNS_ROOT", "VB_SCRATCH_ROOT", "VB_TEST_OUTPUT_ROOT", "VB_KEEP_SOLVER_INPUTS",
                 "VB_DEFAULT_PARAMS"):
        monkeypatch.delenv(name, raising=False)




# These modules contain end-to-end bootstrap calculations or deliberately
# high-precision numerical probes.  They are useful on every release, but are
# poor feedback loops for a small refactor.
SLOW_TEST_FILES = frozenset(
    {
        "test_irreducible_kac_weights.py",
        "test_table_verification_range.py",
        "test_super_singular_tables.py",
        "test_singular_kac_limits.py",
        "test_singular_table_multiplicities.py",
        "test_bootstrap_identical_scalar.py",
        "test_derivs.py",
        "test_folded_coordinate_bootstrap.py",
        "test_laurent.py",
        "test_moments.py",
        "test_point_constraint.py",
        "test_point_value_exact_metadata.py",
        "test_recursion_certified_precision.py",
        "test_recursion_ns.py",
        "test_release_hardening.py",
        "test_torus_positive_poles.py",
    }
)


# These exercise transport, cluster-task, or production orchestration
# boundaries.  They only gain the ``integration`` marker (an overlay for
# ``-m integration`` / ``-m "not integration"`` selection); nothing skips them,
# so they run in the default loop like any other fast test.
INTEGRATION_TEST_FILES = frozenset(
    {
        "test_hpc_host_transport.py",
        "test_hpc_moment_recertify_task.py",
        "test_hpc_moment_task.py",
        "test_hpc_runner.py",
        "test_hpc_worker_submit.py",
    }
)


# Directory ownership is deliberately explicit.  A test may carry more than
# one overlay (for example ``cft`` and ``slow``), but it has one primary owner.
# Keeping this derived from the directory layout means new tests cannot silently
# fall into the block fast tier just because they were not added to a filename
# list.
OWNER_MARKERS = {
    "core": pytest.mark.blocks,
    "bootstrap": pytest.mark.bootstrap,
    "integration": pytest.mark.integration,
}


# Individual slow tests inside files that otherwise belong to the fast tier,
# keyed by filename.  A file listed in SLOW_TEST_FILES is already slow as a
# whole and needs no entry here.  (The test_cli.py entry is not a model
# recomputation: that case deliberately starts a second process pool, a
# platform-sensitive comparison kept out of the fast loop.)
SLOW_TEST_NAMES = {
    "test_cli.py": frozenset({"test_main_num_threads_parallel_matches_serial"}),
    "test_global_blocks.py": frozenset({"test_num_threads_is_bit_identical"}),
}


# A test that is not in the slow tier but runs longer than this is almost
# certainly a new expensive module that nobody added to SLOW_TEST_FILES, so the
# suite says so at the end of the run instead of quietly getting slower.  Timing is machine-dependent, so this reports rather than fails.
#
# The threshold is set well above the fast tier's natural ceiling (a handful of
# block-table and regularization checks legitimately sit near 10s) so that it
# names genuine outliers instead of producing a list nobody reads.
UNMARKED_SLOW_SECONDS = 20.0


_slow_unmarked: dict[str, float] = {}


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="also run the slow tier (high-precision model recomputations and "
             "sewing). Off by default; VB_TEST_SLOW=1 has the same effect.",
    )


def _slow_requested(config):
    return config.getoption("--slow") or os.environ.get("VB_TEST_SLOW") == "1"




@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    skip_slow = None
    if not _slow_requested(config):
        skip_slow = pytest.mark.skip(
            reason="slow tier is opt-in: rerun with --slow (or VB_TEST_SLOW=1)"
        )
    for item in items:
        relative = Path(item.fspath).relative_to(Path(__file__).parent)
        owner = OWNER_MARKERS.get(relative.parts[0]) if relative.parts else None
        if owner is not None:
            item.add_marker(owner)
        filename = Path(str(item.fspath)).name
        slow_names = SLOW_TEST_NAMES.get(filename, ())
        # a test decorated with @pytest.mark.slow is slow too
        is_slow = (filename in SLOW_TEST_FILES or item.name in slow_names
                   or getattr(item, "originalname", None) in slow_names
                   or item.get_closest_marker("slow") is not None)
        if is_slow:
            item.add_marker(pytest.mark.slow)
            if skip_slow is not None:
                item.add_marker(skip_slow)
        if filename in INTEGRATION_TEST_FILES:
            item.add_marker(pytest.mark.integration)


def pytest_runtest_logreport(report):
    # logreport (not makereport) so this also works under xdist: workers send
    # their reports to the controller, where the terminal summary is written.
    if report.when != "call" or report.duration < UNMARKED_SLOW_SECONDS:
        return
    keywords = getattr(report, "keywords", {})
    if "slow" not in keywords and "audit" not in keywords:
        _slow_unmarked[report.nodeid] = report.duration


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    del exitstatus, config
    if not _slow_unmarked:
        return
    files = sorted({nodeid.split("::")[0].split("/")[-1] for nodeid in _slow_unmarked})
    terminalreporter.write_sep("=", "unmarked slow tests", yellow=True)
    for nodeid, seconds in sorted(_slow_unmarked.items(), key=lambda kv: -kv[1]):
        terminalreporter.write_line(f"{seconds:7.1f}s  {nodeid}")
    terminalreporter.write_line(
        f"add to SLOW_TEST_FILES in tests/conftest.py: {', '.join(files)}"
    )
