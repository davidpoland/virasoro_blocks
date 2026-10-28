from datetime import date
from pathlib import Path

import pytest

from virasoro_bootstrap.paths import (
    artifact_dir,
    artifact_root,
    dated_artifact_dir,
    dated_output_dir,
    experiment_root,
    resolve_existing_path,
    result_file,
    research_config_path,
    research_path,
    resolve_research_path,
    results_root,
    runs_root,
    scratch_root,
    scratch_path,
    test_output_root as _test_output_root,
)


@pytest.fixture(autouse=True)
def _default_roots(monkeypatch):
    """Cluster profiles export output roots; these tests start from the defaults."""
    for name in ("VB_ARTIFACT_ROOT", "VB_CERTIFICATES_ROOT", "VB_EXPERIMENT_ROOT", "VB_RESEARCH_ROOT",
                 "VB_RESULTS_ROOT", "VB_RUNS_ROOT", "VB_SCRATCH_ROOT", "VB_TEST_OUTPUT_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_new_artifact_defaults_are_consolidated():
    assert artifact_root() == Path("artifacts")
    assert experiment_root() == Path("artifacts/experiments")
    assert _test_output_root() == Path("artifacts/experiments")
    assert runs_root() == Path("artifacts/runs")
    assert results_root() == Path("artifacts/results")
    assert scratch_root() == Path(".local/scratch")
    assert research_path("sweeps", "demo.toml").name == "demo.toml"
    assert scratch_path("work") == Path(".local/scratch/work")
    # The shipped default config is package data, so this holds for an
    # installed wheel as well as a checkout.
    assert research_config_path().name == "default_params.json"
    assert research_config_path().is_file()
    assert dated_output_dir("demo", on=date(2026, 8, 10)) == Path(
        "artifacts/experiments/2026-08-10/demo"
    )
    assert dated_artifact_dir(
        "figures", "demo", on=date(2026, 8, 10)
    ) == Path("artifacts/figures/2026-08-10/demo")
    assert result_file("sweep") == Path("artifacts/results/sweep-results.json")


def test_artifact_environment_overrides_are_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("VB_ARTIFACT_ROOT", str(tmp_path / "artifact-root"))
    assert artifact_dir("runs") == tmp_path / "artifact-root" / "runs"
    assert experiment_root() == tmp_path / "artifact-root" / "experiments"
    assert _test_output_root() == tmp_path / "artifact-root" / "experiments"

    legacy = tmp_path / "legacy-test-output"
    monkeypatch.setenv("VB_TEST_OUTPUT_ROOT", str(legacy))
    assert _test_output_root() == legacy
    assert dated_output_dir("demo", on=date(2026, 8, 10)) == (
        legacy / "2026-08-10" / "demo"
    )


def test_explicit_legacy_paths_remain_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = Path("test_output/2026-08-10/old.json")
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")

    assert resolve_existing_path(legacy) == legacy
    assert resolve_existing_path("test_output/2026-08-10/new.json") == Path(
        "artifacts/experiments/2026-08-10/new.json"
    )


def test_explicit_root_preserves_dated_output_compatibility(tmp_path):
    assert dated_output_dir(
        "demo", root=tmp_path, on=date(2026, 8, 10)
    ) == tmp_path / "2026-08-10" / "demo"


def test_legacy_research_spellings_resolve_to_the_canonical_layout(
    tmp_path, monkeypatch
):
    """Old command lines keep working after the compatibility symlinks went.

    Driven off a temporary research root rather than the repository's own
    files: this behavior is about the mapping, and a library test that reads
    research/ data cannot run in a library-only checkout.
    """
    monkeypatch.setenv("VB_RESEARCH_ROOT", str(tmp_path))
    (tmp_path / "sweeps").mkdir()
    (tmp_path / "cluster").mkdir()
    (tmp_path / "sweeps" / "phase0.toml").write_text("")
    (tmp_path / "cluster" / "mycluster.toml").write_text("")

    assert resolve_research_path("sweeps/phase0.toml") == (
        tmp_path / "sweeps" / "phase0.toml"
    )
    # clusters/ (plural) was the old top-level name for research/cluster.
    assert resolve_research_path("clusters/mycluster.toml") == (
        tmp_path / "cluster" / "mycluster.toml"
    )
    # An already-canonical path is returned untouched.
    assert resolve_research_path("research/cluster/mycluster.toml") == Path(
        "research/cluster/mycluster.toml"
    )
    # An input with no canonical counterpart is left alone rather than guessed.
    assert resolve_research_path("sweeps/absent.toml") == Path(
        "sweeps/absent.toml"
    )


def test_existing_literal_path_beats_the_legacy_research_alias(tmp_path, monkeypatch):
    """A caller's own clusters/ directory is not replaced by research/cluster."""
    research = tmp_path / "research"
    (research / "cluster").mkdir(parents=True)
    (research / "cluster" / "mycluster.toml").write_text("canonical")
    monkeypatch.setenv("VB_RESEARCH_ROOT", str(research))
    work = tmp_path / "work"
    (work / "clusters").mkdir(parents=True)
    (work / "clusters" / "mycluster.toml").write_text("mine")
    monkeypatch.chdir(work)
    assert resolve_research_path("clusters/mycluster.toml") == Path("clusters/mycluster.toml")
    # A compatibility link to the canonical file keeps its canonical name.
    (work / "clusters" / "mycluster.toml").unlink()
    (work / "clusters" / "mycluster.toml").symlink_to(research / "cluster" / "mycluster.toml")
    assert resolve_research_path("clusters/mycluster.toml") == research / "cluster" / "mycluster.toml"
