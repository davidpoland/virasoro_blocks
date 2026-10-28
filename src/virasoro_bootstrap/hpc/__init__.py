"""HPC execution layer: SDPB runner backends, task registry, worker, submit.

The package is deliberately masterless: a *task* (one self-contained unit,
e.g. a full gap bisection at one c) runs as a
single SLURM array element via :mod:`.worker`; :mod:`.submit` expands sweep
configs into task specs and renders/submits the array script; :mod:`.collect`
aggregates per-task results. :mod:`.runner` abstracts how the SDPB toolchain
is executed (docker / apptainer / native) so the same code runs on a laptop
and on a cluster.
"""

from .runner import DOCKER_SDPB_IMAGE, SdpbRunner, resolve_sdpb_runner

__all__ = [
    "DOCKER_SDPB_IMAGE",
    "SdpbRunner",
    "resolve_sdpb_runner",
]
