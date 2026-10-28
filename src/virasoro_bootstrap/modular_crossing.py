"""Genus-1 modular bootstrap: S-invariance crossing on torus character tables.

The crossing equation is S-invariance of the reduced partition function at the
self-dual point,
expressed in the disk coordinate v (S: (v, vbar) -> (-v, -vbar)):

    0 = (1 - S) Zhat = 2 * [ Vac_{m,n} + sum d_{Delta, j} T^{(j)}_{m,n}(Delta) ]
        over m + n odd,

with Vac the reduced vacuum-character table and T^{(j)} the (j, -j)-pair
symmetrized reduced-character tables from `virasoro_blocks.modular`. The gap
feasibility problem is the exact analogue of the identical-scalar one:
objective zero, the vacuum's odd crossing vector as the normalization, one
pole-free positivity block per spin sector j = 0, 1, 2, ... (ALL integer
spins -- there is no even-spin restriction here), positivity imposed for
Delta >= max(delta_gap, |j|). A primary at the unitarity bound of spin j >= 1 is a chiral current
(h, hbar) = (j, 0) or (0, j), whose vacuum half has the null-state factor; its irreducible character is
a separate positivity constraint wherever the spin's positivity edge is the unitarity bound
(`ModularGapSpec.chiral_currents`).

A dual-feasible SDPB verdict excludes the gap; the bound is the infimum of
excluded gaps. Blocks are exact polynomials (no q_order / npoles knobs).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import mpmath as mp

from virasoro_blocks.interpolation import DampedRational, get_sample_points
from virasoro_blocks.modular import (
    TORUS_COORDINATE,
    torus_character_table,
    torus_current_table,
    torus_exp_base,
    torus_vacuum_table,
)
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.requests import _integer_field
from virasoro_blocks.serialization import SerializedDerivTable

from .hpc.runner import DOCKER_SDPB_IMAGE as _DOCKER_SDPB_IMAGE
from .hpc.runner import SdpbRunner, resolve_sdpb_runner, stage_initial_checkpoint
from .identical_scalar import (
    _as_real_label,
    _dps_to_bits,
    _known_vector_sum,
    _run_command,
    _split_sdpb_options,
    classify_sdpb_log_text,
)
from .pmp_writer import build_prefactored_pmp_doc, write_pmp_json
from .sdp_writer import write_sdp_directory


def S_odd_pairs(lam: int) -> list:
    """The (m, n) functional index set for the S-odd sector.

    Same triangular truncation as the sphere `F_minus_pairs` on `zzb`: odd
    m + n over n <= min(m, lam - m). The n > m half is redundant because the
    j = 0 tables are (m, n)-symmetric and the j > 0 sectors are (j, -j)-pair
    symmetrized.
    """
    lam = _integer_field("lam", lam)
    if lam < 1:
        raise ValueError("lam must be >= 1 (the S-odd sector starts at order 1)")
    return [
        (m, n)
        for m in range(lam + 1)
        for n in range(min(m, lam - m) + 1)
        if (m + n) % 2 == 1
    ]


@dataclass(frozen=True)
class ModularGapSpec:
    """Gap feasibility spec for the genus-1 (partition function) bootstrap.

    `delta_gap` is imposed on every non-vacuum primary: positivity runs over
    Delta >= max(delta_gap, |j|) in each integer-spin sector of `spin_list`
    (j >= 0 labels; j > 0 means the (j, -j) pair). `delta_min_by_spin`
    optionally overrides the positivity edge for specific spins.

    The spin truncation is a *scanned* parameter: the modular bootstrap is
    known to lean on high spins, so check final bounds against a larger
    `spin_list`.  Modular problems require ``c > 1``.
    """

    c: object
    delta_gap: object
    lam: int
    spin_list: Sequence[int] = tuple(range(21))
    dps: int = 120
    delta_min_by_spin: Mapping[int, object] | None = None
    n_sample_points: int | None = None
    # Fixed scalar primaries placed on the known side of modular crossing,
    # before the continuum gap. Each entry is either Delta or
    # (Delta, degeneracy). The first isolated-relevant-scalar experiment uses
    # ((Delta_phi, 1),) with delta_gap=2.
    isolated_scalars: Sequence[object] = ()
    # Isolated scalar primaries with unknown nonnegative degeneracy. These are
    # emitted as degree-0 positivity blocks at fixed Delta rather than folded
    # into the known vector. This is the convex relaxation of "there may be a
    # scalar at Delta_phi" without fixing d_phi.
    allowed_isolated_scalars: Sequence[object] = ()
    # Chiral currents (j, 0) + (0, j), j >= 1, with their irreducible characters (the vacuum half carries the
    # null-state factor), as positivity constraints for every spin whose positivity edge is the unitarity bound
    # Delta = j. The generic table at Delta = j is the Verma continuation, not a current. False assumes the theory
    # has no chiral currents besides the stress tensor.
    chiral_currents: bool = True

    def __post_init__(self):
        # Preparation and gap builds traverse these inputs independently.
        # Capture iterables once, as IdenticalScalarGapSpec does.
        for name in ("spin_list", "isolated_scalars", "allowed_isolated_scalars"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        if not isinstance(self.chiral_currents, bool):
            raise TypeError(f"chiral_currents must be True or False, got {self.chiral_currents!r}")


@dataclass(frozen=True)
class ModularSpinChannel:
    spin: int
    delta_min: object
    block: SerializedDerivTable


@dataclass(frozen=True)
class PreparedModularGapBuild:
    """Gap-independent parts of the modular gap problem.

    The per-spin tables are anchored at the unitarity bound (shift = |j|).
    Re-anchoring at a gap is exact and cheap: with d = delta_min - j,

        N_gap(x) = base^d * N_j(x + d),

    so a gap bisection at fixed (c, lam, spins) reuses one build. Keyed by
    `reuse_key`; `build_modular_gap_problem` validates compatibility.
    """

    reuse_key: tuple
    F_pairs: tuple
    identity_vector: tuple
    unitarity_rows_by_spin: dict  # j -> trimmed [m][n] Polynomial rows


def _prepared_reuse_key(spec: ModularGapSpec, spins: tuple) -> tuple:
    return (_as_real_label(spec.c, name="c"), spec.lam, spins, spec.dps)


def _modular_c_input(value):
    c = _as_real_label(value, name="c")
    if c <= 1:
        raise ValueError(
            "modular gap problems require c > 1 (the vacuum is the only degenerate chiral module); "
            f"got c = {value!r}"
        )
    return c


def _parse_isolated_scalar(entry) -> tuple:
    if isinstance(entry, (tuple, list)):
        if len(entry) != 2:
            raise ValueError(
                "isolated scalar entries must be Delta or (Delta, degeneracy)"
            )
        delta, degeneracy = entry
    else:
        delta, degeneracy = entry, 1
    delta = _as_real_label(delta, name="isolated scalar Delta")
    degeneracy = _as_real_label(degeneracy, name="isolated scalar degeneracy")
    if delta <= 0:
        raise ValueError(f"isolated scalar Delta must be positive, got {delta}")
    if degeneracy < 0:
        raise ValueError(
            f"isolated scalar degeneracy must be nonnegative, got {degeneracy}"
        )
    return delta, degeneracy


def _modular_isolated_scalar_vector(c, lam: int, F_pairs, isolated_scalars, *, identity_vector=None) -> tuple:
    """Known modular-crossing vector from fixed isolated scalar primaries."""
    def terms():
        yield 1, (identity_vector if identity_vector is not None else [0 for _ in F_pairs])
        if isolated_scalars:
            table = torus_character_table(c, 0, lam, symmetrize=True)
            for entry in isolated_scalars:
                delta, degeneracy = _parse_isolated_scalar(entry)
                values = table.evaluate(TORUS_COORDINATE, delta)
                yield degeneracy, (values[m][n] for m, n in F_pairs)
    return _known_vector_sum(terms())


def _modular_allowed_scalar_block(c, lam: int, delta_scalar, F_pairs) -> SerializedDerivTable:
    """Degree-0 modular positivity block at a fixed scalar dimension."""
    delta = _allowed_scalar_input(delta_scalar)
    values = torus_character_table(c, 0, lam, symmetrize=True).evaluate(
        TORUS_COORDINATE, delta
    )
    rows = [[Polynomial([mp.mpf(0)], field=mp.mpf) for _ in range(lam + 1)]
            for _ in range(lam + 1)]
    for m, n in F_pairs:
        rows[m][n] = Polynomial([mp.re(values[m][n])], field=mp.mpf)
    return SerializedDerivTable(
        shift=delta,
        base=torus_exp_base(),
        pole_list_x=(),
        coordinates=(TORUS_COORDINATE,),
        numerators={TORUS_COORDINATE: rows},
        sample_points_x=None,
        working_dps=mp.mp.dps,
    )


def _modular_current_block(c, lam: int, spin: int, F_pairs) -> SerializedDerivTable:
    """Degree-0 modular positivity block of the chiral current pair (spin, 0) + (0, spin)."""
    values = torus_current_table(c, spin, lam)
    rows = [[Polynomial([mp.mpf(0)], field=mp.mpf) for _ in range(lam + 1)]
            for _ in range(lam + 1)]
    for m, n in F_pairs:
        rows[m][n] = Polynomial([values[m][n]], field=mp.mpf)
    return SerializedDerivTable(
        shift=mp.mpf(spin),
        base=torus_exp_base(),
        pole_list_x=(),
        coordinates=(TORUS_COORDINATE,),
        numerators={TORUS_COORDINATE: rows},
        sample_points_x=None,
        working_dps=mp.mp.dps,
    )


def _allowed_scalar_input(delta_scalar):
    delta = _as_real_label(delta_scalar, name="allowed isolated scalar Delta")
    if delta <= 0:
        raise ValueError(f"allowed isolated scalar Delta must be positive, got {delta}")
    return delta


def _validated_modular_controls(spec: ModularGapSpec) -> ModularGapSpec:
    """Normalize count/precision controls before numerical work or reuse."""
    controls = {}
    for name in ("lam", "dps", "n_sample_points"):
        value = getattr(spec, name)
        if name == "n_sample_points" and value is None:
            continue
        value = _integer_field(name, value)
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
        controls[name] = value
    return replace(spec, **controls)


def prepare_modular_gap_build(spec: ModularGapSpec) -> PreparedModularGapBuild:
    """Build the gap-independent tables: vacuum vector + per-spin numerators."""
    spec = _validated_modular_controls(spec)
    spins = _validate_spins(spec.spin_list)
    c = _modular_c_input(spec.c)
    # The character backends guard their own charges and anchors. Retain
    # their computed entries without rounding them in this outer context.
    with mp.workdps(spec.dps):
        F_pairs = tuple(S_odd_pairs(spec.lam))
        vac = torus_vacuum_table(c, spec.lam)
        identity_vector = tuple(vac[m][n] for (m, n) in F_pairs)
        rows_by_spin = {}
        for j in spins:
            table = torus_character_table(c, j, spec.lam, symmetrize=True)
            full_rows = table.numerators[TORUS_COORDINATE]
            # Keep only the triangular row set the S-odd functional consumes
            # (n <= min(m, lam - m), the sphere-table convention). The full
            # rectangle would drag the explicit-sampling degree match up to
            # 2*lam and double the node count for nothing.
            rows_by_spin[j] = [
                row[: min(m, spec.lam - m) + 1]
                for m, row in enumerate(full_rows)
            ]
        return PreparedModularGapBuild(
            reuse_key=_prepared_reuse_key(spec, spins),
            F_pairs=F_pairs,
            identity_vector=identity_vector,
            unitarity_rows_by_spin=rows_by_spin,
        )


@dataclass(frozen=True)
class ModularGapProblem:
    spec: ModularGapSpec
    F_pairs: tuple
    identity_vector: tuple
    spin_channels: tuple
    pmp_doc: dict


@dataclass(frozen=True)
class ModularGapRun:
    output_dir: Path
    pmp_path: Path
    sdp_dir: Path
    sdpb_out_dir: Path | None
    precision_bits: int
    sdpb_command: tuple | None
    sdpb_log_path: Path | None
    sdpb_status: str | None
    checkpoint_dir: Path | None


def _validate_spins(spin_list: Sequence[int]) -> tuple:
    spins = []
    for spin in spin_list:
        j = _integer_field("spin", spin)
        if j < 0:
            raise ValueError(f"spins must be integers >= 0, got {spin!r}")
        spins.append(j)
    if len(set(spins)) != len(spins):
        raise ValueError("spin_list contains duplicates")
    if 0 not in spins:
        raise ValueError("spin_list must include the scalar sector j = 0")
    return tuple(sorted(spins))


def _modular_delta_min_overrides(delta_min_by_spin, spins) -> dict:
    """Exact ``delta_min_by_spin`` labels with integer keys among ``spins``.

    Reject keys outside ``spin_list``, since no channel uses them.
    """
    overrides = {}
    for raw_spin, raw_delta in dict(delta_min_by_spin or {}).items():
        try:
            spin = _integer_field("delta_min_by_spin key", raw_spin)
        except ValueError:
            raise ValueError(f"delta_min_by_spin keys must be integer spins, got {raw_spin!r}") from None
        if spin not in spins:
            raise ValueError(
                f"delta_min_by_spin contains spin {spin}, which is not in spin_list {list(spins)}: no channel "
                "would carry the override"
            )
        delta = _as_real_label(raw_delta, name=f"delta_min_by_spin[{spin}]")
        if delta < spin:
            raise ValueError(f"delta_min {raw_delta!r} below unitarity |j| = {spin}")
        overrides[spin] = delta
    return overrides


def build_modular_gap_problem(
    spec: ModularGapSpec,
    *,
    prepared: PreparedModularGapBuild | None = None,
) -> ModularGapProblem:
    """Build the S-odd gap feasibility PMP from exact character tables.

    Pass `prepared` (from `prepare_modular_gap_build`) to reuse the
    gap-independent tables across a bisection at fixed (c, lam, spins).
    """
    spec = _validated_modular_controls(spec)
    spins = _validate_spins(spec.spin_list)
    c = _modular_c_input(spec.c)
    delta_gap = _as_real_label(spec.delta_gap, name="delta_gap")
    if delta_gap < 0:
        raise ValueError(f"delta_gap must be >= 0, got {spec.delta_gap!r}")
    overrides = _modular_delta_min_overrides(spec.delta_min_by_spin, spins)
    isolated = tuple(_parse_isolated_scalar(entry) for entry in spec.isolated_scalars)
    allowed = tuple(_allowed_scalar_input(delta) for delta in spec.allowed_isolated_scalars)
    spec = replace(spec, c=c, delta_gap=delta_gap, spin_list=spins, delta_min_by_spin=overrides,
                   isolated_scalars=isolated, allowed_isolated_scalars=allowed)
    if prepared is None:
        prepared = prepare_modular_gap_build(spec)
    elif prepared.reuse_key != _prepared_reuse_key(spec, spins):
        raise ValueError(
            "prepared modular gap build is incompatible with the requested spec"
        )

    with mp.workdps(spec.dps):
        F_pairs = list(prepared.F_pairs)
        max_degree = max(m + n for (m, n) in F_pairs)
        sample_points = None
        if spec.n_sample_points is not None:
            if spec.n_sample_points < max_degree + 1:
                raise ValueError(
                    f"n_sample_points {spec.n_sample_points} below max "
                    f"polynomial degree + 1 = {max_degree + 1}"
                )
            sample_points = tuple(
                get_sample_points(
                    DampedRational(
                        constant=mp.mpf(1), poles=(), base=torus_exp_base()
                    ),
                    spec.n_sample_points,
                )
            )

        identity_vector = _modular_isolated_scalar_vector(
            c, spec.lam, F_pairs, spec.isolated_scalars, identity_vector=prepared.identity_vector
        )
        base = torus_exp_base()

        channels = []
        allowed_scalar_blocks = [
            _modular_allowed_scalar_block(c, spec.lam, delta, F_pairs)
            for delta in spec.allowed_isolated_scalars
        ]
        for j in spins:
            delta_min = overrides.get(j, max(delta_gap, j))
            rows = prepared.unitarity_rows_by_spin[j]
            d = delta_min - j
            if d != 0:
                scale = base**d
                rows = [
                    [poly.compose_linear(1, d).scale(scale) for poly in row]
                    for row in rows
                ]
            block = SerializedDerivTable(
                shift=delta_min,
                base=base,
                pole_list_x=(),
                coordinates=(TORUS_COORDINATE,),
                numerators={TORUS_COORDINATE: rows},
                sample_points_x=sample_points,
                working_dps=mp.mp.dps,
            )
            channels.append(
                ModularSpinChannel(spin=j, delta_min=delta_min, block=block)
            )
            if spec.chiral_currents and j >= 1 and delta_min == j:
                channels.append(ModularSpinChannel(
                    spin=j, delta_min=delta_min, block=_modular_current_block(c, spec.lam, j, F_pairs)))
        channels.extend(
            ModularSpinChannel(spin=0, delta_min=block.shift, block=block)
            for block in allowed_scalar_blocks
        )

        objective = [mp.mpf(0)] * len(F_pairs)
        pmp_build = build_prefactored_pmp_doc(
            [channel.block for channel in channels],
            F_pairs,
            coordinate=TORUS_COORDINATE,
            objective=objective,
            normalization=identity_vector,
            lower_positive_real=False,
            explicit_sample_points=True,
        )
        return ModularGapProblem(
            spec=spec,
            F_pairs=tuple(F_pairs),
            identity_vector=identity_vector,
            spin_channels=tuple(channels),
            pmp_doc=pmp_build.doc,
        )


def run_modular_gap_problem(
    problem: ModularGapProblem,
    output_dir,
    *,
    image: str = _DOCKER_SDPB_IMAGE,
    runner: SdpbRunner | None = None,
    run_sdpb: bool = True,
    eval_dps: int | None = None,
    sdpb_options: Sequence[str] | Mapping[str, object] | None = None,
    initial_checkpoint_dir: Path | str | None = None,
) -> ModularGapRun:
    """Write the SDP (direct sdp-writer backend) and invoke sdpb.

    A slim analogue of `run_identical_scalar_gap_problem`: the PMP is written
    for the record, the SDPB `sdp` directory is produced by
    `sdp_writer.write_sdp_directory` at the working precision `spec.dps` (or `eval_dps`), and sdpb runs
    under the resolved runner (docker by default; VB_SDPB_MODE overrides).
    `sdpb_status` is `classify_sdpb_log_text` of the sdpb log: for a gap scan,
    "dual_feasible" means the gap is EXCLUDED.
    Previous solution and objective files are removed before replacing saved inputs, including when
    ``run_sdpb=False``. Hotstart checkpoints are staged separately. Omitting ``y`` from ``--writeSolution``
    leaves no saved dual solution.
    """
    if runner is None:
        runner = resolve_sdpb_runner(
            image=image if image != _DOCKER_SDPB_IMAGE else None
        )
    precision_bits, mpi_procs, sdpb_flag_args = _split_sdpb_options(
        sdpb_options, _dps_to_bits(problem.spec.dps)
    )
    # the working precision (as for the identical-scalar runner): the data carry spec.dps digits
    resolved_eval_dps = (
        _integer_field("eval_dps", eval_dps)
        if eval_dps is not None
        else int(problem.spec.dps)
    )
    if resolved_eval_dps < 1:
        raise ValueError("eval_dps must be >= 1")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pmp_path = output_dir / "pmp.json"
    sdp_dir = output_dir / "sdp"
    sdpb_out_dir = output_dir / "sdpb_out" if run_sdpb else None
    sdpb_log_path = output_dir / "sdpb.log" if run_sdpb else None
    checkpoint_dir = output_dir / "sdp.ck"
    sdpb_command = None
    sdpb_status = None
    if run_sdpb:
        initial_ck_args = stage_initial_checkpoint(output_dir, initial_checkpoint_dir)
        sdpb_command_list, sdpb_cwd = runner.command_argv(
            "sdpb",
            [
                "-s",
                "sdp",
                "-o",
                "sdpb_out",
                "--precision",
                str(precision_bits),
                *initial_ck_args,
                *sdpb_flag_args,
            ],
            output_dir,
            mpi_procs=mpi_procs,
        )
    from .hpc.storage import record_sdp_source

    # Cover input preparation as well as the solver: an early failure must not
    # leave a completed record associating an old result with replaced inputs.
    record_sdp_source(sdp_dir, output_dir / "sdpb_out", state="preparing")
    write_pmp_json(problem.pmp_doc, pmp_path)
    write_sdp_directory(problem.pmp_doc, sdp_dir, eval_dps=resolved_eval_dps)

    if run_sdpb:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        if initial_ck_args:
            # SDPB needs an existing write target when loading a separate seed.
            checkpoint_dir.mkdir()
        source_record = record_sdp_source(sdp_dir, sdpb_out_dir, state="running")
        sdpb_rc = _run_command(
            sdpb_command_list, sdpb_log_path, check=False, cwd=sdpb_cwd
        )
        source_record.update(state="completed" if sdpb_rc == 0 else "failed")
        sdpb_command = tuple(sdpb_command_list)
        if sdpb_rc != 0:
            sdpb_status = "solver_error"
        elif sdpb_log_path is not None:
            sdpb_status = classify_sdpb_log_text(sdpb_log_path.read_text())
    else:
        record_sdp_source(sdp_dir, output_dir / "sdpb_out", state="prepared")

    return ModularGapRun(
        output_dir=output_dir,
        pmp_path=pmp_path,
        sdp_dir=sdp_dir,
        sdpb_out_dir=sdpb_out_dir,
        precision_bits=precision_bits,
        sdpb_command=sdpb_command,
        sdpb_log_path=sdpb_log_path,
        sdpb_status=sdpb_status,
        checkpoint_dir=(
            checkpoint_dir
            if run_sdpb and checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir())
            else None
        ),
    )
