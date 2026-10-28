"""Pre-release regressions for the cluster tooling: inputs that were silently ignored.

No SDPB runs and no scheduler: runs are created with ``--dry-run`` or a local
profile, and solver launches are inspected as argv.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from virasoro_bootstrap import identical_scalar
from virasoro_bootstrap.hpc import collect, recertify, submit
from virasoro_bootstrap.hpc.inputs import HpcInputError
from virasoro_bootstrap.hpc.journal import Journal
from virasoro_bootstrap.hpc.runner import SdpbRunner, resolve_sdpb_runner
from virasoro_bootstrap.hpc.tasks import (
    DEFAULT_FEASIBILITY_SDPB_OPTIONS, TaskContext, _merged_sdpb_options, scalar_only_gap_probe,
)

SWEEP = '''name = "probe"
task = "scalar_only_gap_probe"

[params]
c = "2"
lam = 7

[sweep]
delta = ["0.5", "0.75"]
'''


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def dry_run(tmp_path, capsys):
    """A dry-run SLURM submission: specs, job.sbatch, and manifest, nothing queued."""
    sweep = _write(tmp_path / "sweep.toml", SWEEP)
    profile = _write(tmp_path / "profile.toml", f'partition = "compute"\nruns_root = "{tmp_path / "runs"}"\n')
    assert submit.main([str(sweep), "--cluster", str(profile), "--run-id", "r1", "--dry-run"]) == 0
    capsys.readouterr()
    return SimpleNamespace(run=tmp_path / "runs" / "r1", sweep=sweep, profile=profile)


# --- 16. --retry must not ignore options that name a different run or profile -------

@pytest.mark.parametrize("extra", [["--run-id", "other"], ["--runs-root", "elsewhere"],
                                   ["SWEEP"], ["--allow-dirty"]])
@pytest.mark.parametrize("host", [False, True])
def test_retry_refuses_options_it_would_ignore(dry_run, monkeypatch, capsys, extra, host):
    monkeypatch.setattr(submit, "retry_run", lambda *a, **k: pytest.fail("retry dispatched"))
    monkeypatch.setattr(submit, "retry_remote", lambda *a, **k: pytest.fail("retry dispatched"))
    extra = [str(dry_run.sweep) if word == "SWEEP" else word for word in extra]
    argv = ["--retry", str(dry_run.run), *extra, "--cluster", str(dry_run.profile)]
    with pytest.raises(SystemExit) as exc:
        submit.main(argv + (["--host", "login"] if host else []))
    assert exc.value.code == 2
    assert "--retry" in capsys.readouterr().err


def test_retry_checks_an_explicit_profile_against_the_recorded_one(dry_run, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(submit, "_queued_array_indices", lambda manifest: set())
    assert submit.main(["--retry", str(dry_run.run), "--cluster", str(dry_run.profile), "--dry-run"]) == 0
    assert "dry run: sbatch --array=0,1" in capsys.readouterr().out

    changed = _write(tmp_path / "changed.toml",
                     f'partition = "other"\nruns_root = "{tmp_path / "runs"}"\n')
    with pytest.raises(SystemExit) as exc:
        submit.main(["--retry", str(dry_run.run), "--cluster", str(changed), "--dry-run"])
    assert exc.value.code == 2
    assert "differs" in capsys.readouterr().err

    # $VB_CLUSTER names this machine's current profile, not the run's.
    monkeypatch.setenv("VB_CLUSTER", str(changed))
    assert submit.main(["--retry", str(dry_run.run), "--dry-run"]) == 0


# --- 17. misspelled profile keys are errors -----------------------------------------

@pytest.mark.parametrize("key, suggestion", [("partiton", "partition"), ("acount", "account")])
def test_unknown_profile_key_is_rejected_with_the_valid_keys(key, suggestion):
    with pytest.raises(HpcInputError) as exc:
        submit._validate_cluster_profile({key: "x"})
    message = str(exc.value)
    assert repr(key) in message and f"did you mean {suggestion!r}" in message
    assert "valid keys:" in message and "sbatch_extra" in message


def test_unknown_profile_key_stops_submission_before_writing(tmp_path, capsys):
    sweep = _write(tmp_path / "sweep.toml", SWEEP)
    profile = _write(tmp_path / "profile.toml", f'partiton = "gpu"\nruns_root = "{tmp_path / "runs"}"\n')
    with pytest.raises(SystemExit) as exc:
        submit.main([str(sweep), "--cluster", str(profile), "--dry-run"])
    assert exc.value.code == 2 and "partiton" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_invalid_sdpb_mode_in_profile_is_an_error_and_missing_image_a_warning(monkeypatch):
    with pytest.raises(HpcInputError, match="VB_SDPB_MODE"):
        submit._validate_cluster_profile({"env": {"VB_SDPB_MODE": "apptainr"}})
    monkeypatch.delenv("VB_SDPB_IMAGE", raising=False)
    warnings = submit.sdpb_preflight({"env": {"VB_SDPB_MODE": "apptainer"}})
    assert len(warnings) == 1 and "image" in warnings[0]


# --- 18. sdpb_options merge over the task defaults ------------------------------------

def test_sdpb_options_merge_over_defaults():
    defaults = {"--precision": 768, "--findPrimalFeasible": True, "--maxIterations": 1000}
    merged = _merged_sdpb_options(defaults, {"--precision": 1024, "--mpiProcs": 8,
                                             "--findPrimalFeasible": False,
                                             "--maxIterations=500": True})
    assert merged == {"--precision": 1024, "--mpiProcs": 8, "--maxIterations=500": True}
    assert _merged_sdpb_options(defaults, None) == defaults
    assert _merged_sdpb_options(defaults, {}) == defaults


def test_task_keeps_default_flags_beside_sweep_options(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(identical_scalar, "build_identical_scalar_gap_problem", lambda spec: object())

    def run(problem, output_dir, *, sdpb_options, **kwargs):
        seen.update(sdpb_options)
        return SimpleNamespace(sdpb_status="primal_feasible", sdpb_log_path=None)

    monkeypatch.setattr(identical_scalar, "run_identical_scalar_gap_problem", run)
    ctx = TaskContext(params={"c": "2", "delta": "0.5", "lam": 7,
                              "sdpb_options": {"--precision": 1024, "--mpiProcs": 8}},
                      output_dir=tmp_path, journal=Journal(tmp_path / "progress.json"))
    assert scalar_only_gap_probe(ctx)["allowed"] is True
    assert seen["--precision"] == 1024 and seen["--mpiProcs"] == 8
    for flag in ("--findPrimalFeasible", "--findDualFeasible", "--writeSolution"):
        assert seen[flag] == DEFAULT_FEASIBILITY_SDPB_OPTIONS[flag]


# --- 19. MPI ranks in a one-task allocation -----------------------------------------

def test_mpi_preflight_names_the_launcher_setting():
    params = [{"sdpb_options": {"--mpiProcs": 8}}]
    warnings = submit.mpi_preflight({}, params, {"cpus_per_task": 4})
    assert any("exceeds cpus_per_task 4" in w for w in warnings)
    assert any("VB_MPI_LAUNCHER" in w and "--oversubscribe" in w for w in warnings)
    configured = {"env": {"VB_MPI_LAUNCHER": "mpirun --oversubscribe -n {n}"}}
    assert submit.mpi_preflight(configured, params, {"cpus_per_task": 8}) == []
    assert submit.mpi_preflight({"kind": "local"}, params, {"cpus_per_task": 8}) == []
    assert submit.mpi_preflight({}, [{"sdpb_options": {"--mpiProcs": 1}}], {}) == []
    assert submit.mpi_preflight({}, [{}], {}) == []


# --- 20. smaller items ----------------------------------------------------------------

def test_docker_mode_honors_the_mpi_launcher(monkeypatch):
    from virasoro_bootstrap.hpc import runner as runner_mod

    monkeypatch.setattr(runner_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    argv, _ = SdpbRunner(mode="docker").command_argv("sdpb", ["-s", "sdp"], Path("/w"), mpi_procs=4)
    assert argv[argv.index("mpirun"):] == ["mpirun", "-n", "4", "--allow-run-as-root", "sdpb", "-s", "sdp"]
    monkeypatch.setenv("VB_MPI_LAUNCHER", "mpiexec -np {n}")
    argv, _ = resolve_sdpb_runner(mode="docker").command_argv("sdpb", [], Path("/w"), mpi_procs=4)
    assert argv[-4:] == ["mpiexec", "-np", "4", "sdpb"]


def test_run_lookup_by_job_id_includes_retries(tmp_path):
    run = tmp_path / "runs" / "r1"
    _write(run / "manifest.json", json.dumps({"slurm_job_id": "300", "slurm_job_ids": ["100", "300"]}))
    for job in ("100", "300", "100_7"):
        assert collect.resolve_run_dir_arg(job, tmp_path / "runs") == run


def test_collect_host_help_names_the_actual_output_path(capsys):
    with pytest.raises(SystemExit):
        collect.collect_main(["--help"])
    assert "artifacts/results/<RUN>-results.json" in " ".join(capsys.readouterr().out.split())


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_manifest_commit_is_never_an_unrelated_repository(tmp_path, monkeypatch):
    from virasoro_bootstrap import paths

    outer = tmp_path / "users-project"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(["git", "-C", str(outer), "-c", "user.name=t", "-c", "user.email=t@example.test",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    monkeypatch.chdir(outer)
    monkeypatch.setattr(paths, "_source_checkout_root", lambda: None)
    assert submit._library_checkout() is None
    sweep = _write(tmp_path / "sweep.toml", SWEEP)
    profile = _write(tmp_path / "profile.toml", f'runs_root = "{tmp_path / "runs"}"\n')
    submit.create_run(sweep, profile, run_id="r", dry_run=True)
    assert json.loads((tmp_path / "runs/r/manifest.json").read_text())["git_commit"] is None


def test_certify_without_a_probe_flag_checks_every_probe(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    _write(run / "manifest.json", json.dumps({"n_tasks": 1}))
    _write(run / "specs/0000.json", json.dumps({"task": "scalar_gap_bisection", "params": {}}))
    for probe in ("gap_1p0", "gap_2p0", "gap_3p0"):
        (run / "tasks/0000" / probe / "sdpb_out").mkdir(parents=True)
    checked = []

    def certify(run_dir, task, probe, **kwargs):
        checked.append(probe)
        return True, [{"passed": True}], {"probe": probe, "probe_value": None}

    monkeypatch.setattr(recertify, "certify_probe", certify)
    assert recertify.main([str(run), "--task", "0"]) == 0
    assert checked == ["gap_1p0", "gap_2p0", "gap_3p0"]
