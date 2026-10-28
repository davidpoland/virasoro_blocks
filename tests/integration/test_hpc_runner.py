"""SdpbRunner argv construction across the three execution modes.

Pure argv-level tests: no docker/apptainer/SDPB is ever executed. Executable
lookups are monkeypatched so the suite passes on machines with none of the
tools installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from virasoro_bootstrap.hpc import runner as runner_mod
from virasoro_bootstrap.hpc.runner import SdpbRunner, resolve_sdpb_runner

WORKDIR = Path("/some/work/dir")


@pytest.fixture
def fake_which(monkeypatch):
    """Pretend every executable exists at /usr/bin/<name>."""
    monkeypatch.setattr(
        runner_mod.shutil, "which", lambda name: str(name) if Path(name).is_absolute() else f"/usr/bin/{Path(name).name}"
    )


def test_docker_sdpb_matches_historical_command(fake_which):
    argv, cwd = SdpbRunner().command_argv(
        "sdpb", ["-s", "sdp", "-o", "sdpb_out", "--precision", "768"],
        WORKDIR, mpi_procs=4,
    )
    assert argv == [
        "/usr/bin/docker", "run", "--rm", "--shm-size=2g",
        "-v", f"{WORKDIR}:/work", "-w", "/work",
        "bootstrapcollaboration/sdpb:master",
        "mpirun", "-n", "4", "--allow-run-as-root",
        "sdpb", "-s", "sdp", "-o", "sdpb_out", "--precision", "768",
    ]
    assert cwd is None


def test_docker_pmp2sdp_no_shm_no_mpi(fake_which):
    argv, _ = SdpbRunner().command_argv(
        "pmp2sdp", ["-i", "pmp.json", "-o", "sdp", "-p", "768"], WORKDIR
    )
    assert "--shm-size=2g" not in argv
    assert "mpirun" not in argv
    assert argv[-7:] == ["pmp2sdp", "-i", "pmp.json", "-o", "sdp", "-p", "768"]


def test_docker_single_rank_omits_mpirun(fake_which):
    argv, _ = SdpbRunner().command_argv("sdpb", ["-s", "sdp"], WORKDIR, mpi_procs=1)
    assert "mpirun" not in argv


def test_apptainer_mode(fake_which, tmp_path, monkeypatch):
    r = SdpbRunner(mode="apptainer", image="/images/sdpb.sif")
    argv, cwd = r.command_argv("sdpb", ["-s", "sdp"], WORKDIR, mpi_procs=8)
    assert argv == [
        "/usr/bin/apptainer", "exec",
        "--bind", f"{WORKDIR}:/work", "--pwd", "/work",
        "/images/sdpb.sif",
        "mpirun", "-n", "8",
        "sdpb", "-s", "sdp",
    ]
    assert cwd is None
    # No docker-isms leak into apptainer.
    assert "--allow-run-as-root" not in argv
    assert "--shm-size=2g" not in argv
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('VB_TEST_IMAGE_DIR', str(tmp_path / 'images'))
    for raw in ('~/images/sdpb.sif', '$VB_TEST_IMAGE_DIR/sdpb.sif'):
        r = SdpbRunner(mode='apptainer', image=raw)
        argv, _ = r.command_argv('sdpb', [], WORKDIR)
        assert argv[6] == str(tmp_path / 'images/sdpb.sif')
    for raw in ('docker://registry/image:tag', 'relative/image.sif'):
        argv, _ = SdpbRunner(mode='apptainer', image=raw).command_argv('sdpb', [], WORKDIR)
        assert argv[6] == raw
    assert resolve_sdpb_runner(mode='docker', image='~/literal:tag').image == '~/literal:tag'


def test_apptainer_requires_image():
    with pytest.raises(ValueError, match="apptainer mode needs an image"):
        SdpbRunner(mode="apptainer")


def test_native_mode_with_bin_dir_and_custom_launcher(fake_which):
    r = SdpbRunner(
        mode="native", bin_dir="/opt/sdpb/bin", mpi_launcher="srun -n {n} --mpi=pmix"
    )
    argv, cwd = r.command_argv("sdpb", ["-s", "sdp"], WORKDIR, mpi_procs=16)
    assert argv == [
        "/usr/bin/srun", "-n", "16", "--mpi=pmix", "/opt/sdpb/bin/sdpb", "-s", "sdp"
    ]
    assert cwd == WORKDIR


def test_native_mode_path_lookup(fake_which):
    argv, cwd = SdpbRunner(mode="native").command_argv("pmp2sdp", ["-i", "x"], WORKDIR)
    assert argv == ["/usr/bin/pmp2sdp", "-i", "x"]
    assert cwd == WORKDIR


def test_native_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(runner_mod.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="sdpb"):
        SdpbRunner(mode="native").command_argv("sdpb", [], WORKDIR)
    monkeypatch.setattr(runner_mod.shutil, 'which', lambda name: '/bin/sdpb' if name == 'sdpb' else None)
    with pytest.raises(FileNotFoundError, match="MPI launcher 'mpirun' not found"):
        SdpbRunner(mode='native').command_argv('sdpb', [], WORKDIR, mpi_procs=2)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown SDPB runner mode"):
        SdpbRunner(mode="podman")


def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv("VB_SDPB_MODE", "apptainer")
    monkeypatch.setenv("VB_SDPB_IMAGE", "/scratch/images/sdpb.sif")
    monkeypatch.setenv("VB_MPI_LAUNCHER", "srun -n {n}")
    r = resolve_sdpb_runner()
    assert (r.mode, r.image, r.mpi_launcher) == (
        "apptainer", "/scratch/images/sdpb.sif", "srun -n {n}"
    )


def test_resolve_explicit_args_beat_env(monkeypatch, fake_which):
    monkeypatch.setenv("VB_SDPB_MODE", "native")
    monkeypatch.setenv("VB_SDPB_IMAGE", "env-image")
    r = resolve_sdpb_runner(mode="docker", image="my/image:tag")
    assert (r.mode, r.image) == ("docker", "my/image:tag")
    monkeypatch.setenv('VB_SDPB_BIN_DIR', '/environment/bin')
    monkeypatch.setenv('VB_MPI_LAUNCHER', 'srun -n {n}')
    r = resolve_sdpb_runner(mode='native', bin_dir='', mpi_launcher='')
    argv, cwd = r.command_argv('sdpb', [], WORKDIR, mpi_procs=2)
    assert argv == ['/usr/bin/sdpb'] and cwd == WORKDIR
    with pytest.raises(ValueError, match='unknown SDPB runner mode'):
        resolve_sdpb_runner(mode='')
    with pytest.raises(ValueError, match='apptainer mode needs an image'):
        resolve_sdpb_runner(mode='apptainer', image='')


def test_resolve_defaults(monkeypatch):
    for var in ("VB_SDPB_MODE", "VB_SDPB_IMAGE", "VB_SDPB_BIN_DIR", "VB_MPI_LAUNCHER"):
        monkeypatch.delenv(var, raising=False)
    r = resolve_sdpb_runner()
    assert (r.mode, r.image, r.bin_dir) == ("docker", None, None)


@pytest.mark.parametrize("via_path", [False, True])
def test_native_relative_executable_survives_workdir_change(tmp_path, monkeypatch, via_path):
    import subprocess

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bin").mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    executable = tmp_path / "bin" / "sdpb"
    executable.write_text("#!/bin/sh\npwd\n")
    executable.chmod(0o755)
    launcher = tmp_path / 'bin/mpirun'
    launcher.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", "bin")
    runner = SdpbRunner(mode="native", bin_dir=None if via_path else "bin")
    argv, cwd = runner.command_argv("sdpb", [], Path("work"))
    result = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)
    assert Path(result.stdout.strip()).resolve() == workdir.resolve()
    for template in ('mpirun -n {n}', 'bin/mpirun -n {n}'):
        runner = SdpbRunner(mode='native', bin_dir=None if via_path else 'bin', mpi_launcher=template)
        argv, cwd = runner.command_argv('sdpb', [], Path('work'), mpi_procs=2)
        result = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)
        assert Path(result.stdout.strip()).resolve() == workdir.resolve()
    if not via_path:
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('VB_TEST_BIN_DIR', str(tmp_path / 'bin'))
        for bin_dir in ('~/bin', '$VB_TEST_BIN_DIR'):
            runner = SdpbRunner(mode='native', bin_dir=bin_dir)
            assert runner_mod.missing_sdpb_tools(runner) == ['pmp2sdp']
            argv, cwd = runner.command_argv('sdpb', [], Path('work'))
            result = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)
            assert Path(result.stdout.strip()).resolve() == workdir.resolve()


@pytest.mark.parametrize("mode", ["docker", "apptainer"])
def test_container_relative_workdir_is_an_absolute_host_bind(tmp_path, monkeypatch, fake_which, mode):
    monkeypatch.chdir(tmp_path)
    runner = SdpbRunner(mode=mode, image="image.sif")
    argv, cwd = runner.command_argv("sdpb", [], Path("work"))
    bind_flag = "-v" if mode == "docker" else "--bind"
    assert argv[argv.index(bind_flag) + 1] == f"{tmp_path.resolve() / 'work'}:/work"
    assert cwd is None


def test_failed_checkpoint_copy_preserves_source_and_existing_seed(tmp_path, monkeypatch):
    source = tmp_path / "sdp.ck"
    staged = tmp_path / "initial.ck"
    for path, text in [(source, "source"), (staged, "old seed")]:
        path.mkdir()
        (path / "checkpoint.0").write_text(text)
    def fail(*args, **kwargs):
        raise OSError("copy interrupted")
    monkeypatch.setattr(runner_mod.shutil, "copytree", fail)
    with pytest.raises(OSError, match="copy interrupted"):
        runner_mod.stage_initial_checkpoint(tmp_path, source)
    assert (source / "checkpoint.0").read_text() == "source"
    assert (staged / "checkpoint.0").read_text() == "old seed"
    assert not list(tmp_path.glob(".initial-*"))


@pytest.mark.parametrize("mode, processes", [("docker", True), ("native", 1.5), ("apptainer", 0)])
def test_runner_rejects_invalid_mpi_process_counts(fake_which, mode, processes):
    runner = SdpbRunner(mode=mode, image="image.sif")
    with pytest.raises(ValueError, match="mpi_procs must be a positive integer"):
        runner.command_argv("sdpb", [], WORKDIR, mpi_procs=processes)
    if processes is True:
        import numpy as np

        for boolean in (np.bool_(True), np.bool_(False)):
            with pytest.raises(ValueError, match="mpi_procs must be a positive integer"):
                runner.command_argv("sdpb", [], WORKDIR, mpi_procs=boolean)
        argv, _ = runner.command_argv("sdpb", [], WORKDIR, mpi_procs=np.int64(2))
        assert argv[argv.index("-n") + 1] == "2"
