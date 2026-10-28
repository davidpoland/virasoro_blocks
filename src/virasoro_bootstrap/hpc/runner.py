"""Execution backends for the SDPB toolchain: docker, apptainer, native.

Every mode honors the same contract: the tool (``pmp2sdp`` or ``sdpb``) runs
with the caller's *workdir* as its working directory -- bind-mounted at
``/work`` for the container modes, plain ``cwd`` for native -- so callers pass
paths **relative to workdir** and keep every input/output under it. That is
exactly the convention the existing Docker path established, which is what
makes the three modes drop-in interchangeable.

Mode selection is by explicit argument or environment (so scripts need no
changes to run on a cluster):

    VB_SDPB_MODE=docker|apptainer|native   (default: docker, else native if sdpb/pmp2sdp are on PATH)
    VB_SDPB_IMAGE=<docker image tag | apptainer .sif path>
    VB_SDPB_BIN_DIR=<dir containing sdpb, pmp2sdp>       (native only)
    VB_MPI_LAUNCHER="mpirun -n {n}"    (MPI prefix template, run where sdpb runs)

The launcher defaults to ``mpirun -n {n}``, and in docker mode to
``mpirun -n {n} --allow-run-as-root`` because the upstream image runs as root.
``VB_MPI_LAUNCHER`` replaces the default in every mode; in the container modes
it runs inside the image. Inside a one-task SLURM allocation Open MPI's
``mpirun`` refuses more than one rank unless oversubscription is allowed, e.g.
``VB_MPI_LAUNCHER="mpirun --oversubscribe -n {n}"`` (docs/CLUSTER.md).
Docker uses ``--shm-size=2g`` for sdpb.
Apptainer uses the caller's user namespace and the host /dev/shm.
"""
from __future__ import annotations

import os
import operator
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .inputs import HpcInputError

DOCKER_SDPB_IMAGE = "bootstrapcollaboration/sdpb:master"

ENV_MODE = "VB_SDPB_MODE"
ENV_IMAGE = "VB_SDPB_IMAGE"
ENV_BIN_DIR = "VB_SDPB_BIN_DIR"
ENV_MPI_LAUNCHER = "VB_MPI_LAUNCHER"

_MODES = ("docker", "apptainer", "native")
SDPB_MODES = _MODES
_DEFAULT_MPI_LAUNCHER = "mpirun -n {n}"
_DEFAULT_DOCKER_MPI_LAUNCHER = "mpirun -n {n} --allow-run-as-root"


def stage_initial_checkpoint(output_dir: Path, source: Path | str | None) -> list[str]:
    """Stage a warm start before the caller clears the new checkpoint target."""
    if source is None:
        return []
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"initial_checkpoint_dir {source} is not a directory")
    if source == output_dir or source in output_dir.parents:
        raise ValueError("initial_checkpoint_dir must not contain the output directory")
    staged = output_dir / "initial.ck"
    if staged.is_symlink() or source != staged.resolve():
        # Copy first, including when source is sdp.ck or inside an old initial.ck.
        # A failed copy must not destroy the previous checkpoint or staged seed.
        with tempfile.TemporaryDirectory(prefix=".initial-", dir=output_dir) as temporary:
            copied = Path(temporary) / "checkpoint"
            shutil.copytree(source, copied)
            if staged.is_symlink():
                staged.unlink()
            elif staged.exists():
                shutil.rmtree(staged)
            copied.rename(staged)
    return ["--initialCheckpointDir", "initial.ck"]


@dataclass(frozen=True)
class SdpbRunner:
    """Builds argv (and cwd) for one execution mode of the SDPB toolchain."""

    mode: str = "docker"
    image: str | None = None
    bin_dir: str | None = None
    #: MPI prefix template with ``{n}``; None selects the mode's default.
    mpi_launcher: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise HpcInputError(f"unknown SDPB runner mode {self.mode!r}; expected one of {_MODES}")
        if self.mode == "apptainer" and not self.image:
            raise HpcInputError(
                "apptainer mode needs an image (.sif path): set VB_SDPB_IMAGE or pass image="
            )
        if self.mode == "apptainer":
            # The subprocess argv has no shell to expand paths. Use string
            # expansion so relative image paths and image URIs stay intact.
            object.__setattr__(self, "image", os.path.expanduser(os.path.expandvars(self.image)))
        if self.mode == "native" and self.bin_dir:
            object.__setattr__(self, "bin_dir", os.path.expanduser(os.path.expandvars(self.bin_dir)))

    def command_argv(
        self,
        tool: str,
        tool_args: Sequence[str],
        workdir: Path,
        *,
        mpi_procs: int | None = None,
    ) -> tuple[list[str], Path | None]:
        """Return ``(argv, cwd)`` running *tool* with *tool_args* under *workdir*.

        ``cwd`` is None for the container modes (the container sets its own
        working directory to the bind target) and *workdir* for native.
        """
        workdir = Path(workdir).resolve()
        args = [str(a) for a in tool_args]
        if mpi_procs is not None:
            import numpy as np

            try:
                if isinstance(mpi_procs, (bool, np.bool_)):
                    raise TypeError
                mpi_procs = operator.index(mpi_procs)
            except TypeError:
                raise ValueError("mpi_procs must be a positive integer") from None
            if mpi_procs < 1:
                raise ValueError("mpi_procs must be a positive integer")
        use_mpi = mpi_procs is not None and mpi_procs > 1

        launcher = self.mpi_launcher
        if launcher is None:
            launcher = _DEFAULT_DOCKER_MPI_LAUNCHER if self.mode == "docker" else _DEFAULT_MPI_LAUNCHER
        mpi_prefix = shlex.split(launcher.format(n=mpi_procs)) if use_mpi else []

        if self.mode == "docker":
            docker = shutil.which("docker")
            if docker is None:
                raise FileNotFoundError(
                    "docker executable not found. Docker is only the default, "
                    "not a requirement -- set one of:\n"
                    f"  {ENV_MODE}=native      with {ENV_BIN_DIR}=<dir with sdpb, "
                    "pmp2sdp>, or both already on PATH\n"
                    f"  {ENV_MODE}=apptainer   with {ENV_IMAGE}=<path to .sif>\n"
                    f"  {ENV_MODE}=docker      and install docker"
                )
            # SDPB's MPI shared-memory windows outgrow docker's default 64m /dev/shm.
            shm = ["--shm-size=2g"] if tool == "sdpb" else []
            return (
                [
                    docker, "run", "--rm", *shm,
                    "-v", f"{workdir}:/work", "-w", "/work",
                    self.image or DOCKER_SDPB_IMAGE,
                    *mpi_prefix, tool, *args,
                ],
                None,
            )

        if self.mode == "apptainer":
            apptainer = shutil.which("apptainer") or shutil.which("singularity")
            if apptainer is None:
                raise FileNotFoundError("apptainer (or singularity) executable not found")
            return (
                [
                    apptainer, "exec",
                    "--bind", f"{workdir}:/work", "--pwd", "/work",
                    str(self.image),
                    *mpi_prefix, tool, *args,
                ],
                None,
            )

        # native
        tool_path = str(Path(self.bin_dir) / tool) if self.bin_dir else tool
        resolved_tool = shutil.which(tool_path)
        if resolved_tool is None:
            raise FileNotFoundError(
                f"{tool!r} not found ({'in ' + self.bin_dir if self.bin_dir else 'on PATH'}); "
                f"set {ENV_BIN_DIR} or load the module providing it"
            )
        # Resolve before changing cwd: both bin_dir and PATH entries may be
        # relative to the caller's directory, not the solver output directory.
        if mpi_prefix:
            resolved_launcher = shutil.which(mpi_prefix[0])
            if resolved_launcher is None:
                raise FileNotFoundError(f"MPI launcher {mpi_prefix[0]!r} not found")
            mpi_prefix[0] = str(Path(resolved_launcher).absolute())
        return ([*mpi_prefix, str(Path(resolved_tool).absolute()), *args], workdir)


SDPB_TOOLS = ("sdpb", "pmp2sdp")


def _default_mode() -> str:
    """Mode to use when nobody chose one: docker, else a native toolchain."""
    if shutil.which("docker") is not None:
        return "docker"
    if all(shutil.which(tool) is not None for tool in SDPB_TOOLS):
        return "native"
    return "docker"


def missing_sdpb_tools(runner: "SdpbRunner") -> list[str]:
    """Which of the SDPB tools this runner cannot find, for native mode.

    Container modes resolve their tools inside the image, so there is nothing
    to check here without running it.  Native mode can be checked cheaply, and
    is worth checking up front: otherwise a missing ``pmp2sdp`` only surfaces
    once a task has already been queued and scheduled.
    """
    if runner.mode != "native":
        return []
    missing = []
    for tool in SDPB_TOOLS:
        path = str(Path(runner.bin_dir) / tool) if runner.bin_dir else tool
        if shutil.which(path) is None:
            missing.append(tool)
    return missing


def resolve_sdpb_runner(
    *,
    mode: str | None = None,
    image: str | None = None,
    bin_dir: str | None = None,
    mpi_launcher: str | None = None,
) -> SdpbRunner:
    """Build an :class:`SdpbRunner` from explicit arguments, falling back to env.

    Non-None explicit arguments win over environment variables, which win
    over the defaults (upstream image, ``mpirun -n {n}``, plus
    ``--allow-run-as-root`` in docker). An empty binary directory clears the
    environment setting; an empty launcher runs sdpb without an MPI prefix.

    With no mode chosen anywhere, docker is preferred but not required: if
    docker is absent and a native toolchain is on PATH, that is used instead.
    An explicitly chosen mode is used without fallback.
    """
    return SdpbRunner(
        mode=mode if mode is not None else (os.environ.get(ENV_MODE) or _default_mode()),
        image=image if image is not None else (os.environ.get(ENV_IMAGE) or None),
        bin_dir=bin_dir if bin_dir is not None else (os.environ.get(ENV_BIN_DIR) or None),
        mpi_launcher=(mpi_launcher if mpi_launcher is not None
                      else os.environ.get(ENV_MPI_LAUNCHER)),
    )
