"""Assemble a chiral (zbar = 1/2 slice) bootstrap problem from CLI block files.

Why this exists
---------------
Crossing maps ``(z, zbar) -> (1-z, 1-zbar)``, so the slice ``zbar = 1/2`` is
crossing INVARIANT. Restricting the 2d crossing equation there leaves a genuine
one-variable sum rule in ``z``, carrying the SAME unitarity ``lambda^2 >= 0`` --
no new positivity assumption. On that slice the parity-projected block
factorizes, so the 2d crossing vector restricted to the ``(m, 0)`` derivatives is

    F_{Delta,ell}^{(m,0)} = k_hbar(1/2) * gamma_h  +  k_h(1/2) * gamma_hbar

with ``gamma`` the chiral crossing vector and BOTH weights positive (``k_h(1/2)
> 0`` for ``h >= 0``, checked across c = 8/7..30 and h = 0..100). Verified in the
repo's own conventions by a dedicated factorization diagnostic (private research tree):
at c = 6/5 the fit residual is 1e-26..1e-31 and converges with npoles, and the
fitted weight depends only on hbar exactly as derived.

The consequence is the point of all this: a functional non-negative on the
single ray ``h >= h_gap`` is automatically non-negative on EVERY 2d operator
with ``h, hbar >= h_gap``, i.e. on every spin at twist >= ``2 h_gap``. **No spin
truncation.** That removes the failure mode that invalidated the 2d twist-gap
functionals, which went negative at ell = 26..40 and on high tower twists --
channels the SDP never enforced but certification could still see.

Blocks come from the CLI, not from a rebuild
--------------------------------------------
``virasoro_blocks table --basis chiral_F`` already emits exactly the block this
needs, with numeric node sampling (``sample_via="numeric"``, the well-conditioned
O(1) recursion rather than the ~1e500-coefficient symbolic ``P(h)/D(h)``), Kac
poles, and optimal interpolation nodes. Reading that file keeps this path in
lock-step with the tested block pipeline instead of duplicating it -- and is why
this module contains no block machinery at all.

    virasoro_blocks table --family virasoro --basis chiral_F \\
        --central-charge 1.2 --h-ext H H H H --coordinates z \\
        --z-shift <-2H> --lambda L --order Q --npoles P \\
        --sample-via numeric --format json -o DIR

SHIFT / GAP, which decides what this can be used for
----------------------------------------------------
The positivity edge is the block's ``shift``: the polynomial variable
``x = h - shift`` starts at ``x = 0`` on the SDPB positivity interval. Two cases:

  * MOMENT bounds -- want ``shift = 0``, positivity on ``h >= 0``, since a
    moment bound is over the full spectrum. That is the CLI default.
  * A TWIST GAP at ``tau > 0`` -- wants positivity on ``h >= tau/2``, i.e.
    ``shift = tau/2``. Generate the block with ``--gap tau/2``; the flag is
    wired through to the chiral basis and the value round-trips onto
    ``table.shift`` (verified: ``--gap 0.35`` yields ``table.shift == 0.35``).

Either way this module CHECKS the block's shift against ``h_gap`` and refuses a
mismatch rather than silently bounding the wrong domain.
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

from dataclasses import replace as _replace

from virasoro_blocks.serialization import _file_integer, serialized_table_from_json
from virasoro_blocks.requests import _integer_field
from virasoro_blocks.numeric_input import literal_number

from .controls.chiral_control import (
    _chiral_identity_vector,
    _chiral_symbolic_exact_table,
    _exact_identity_vector,
    _resolved_chiral_regularize,
    chiral_F_minus_pairs,
)
from .identical_scalar import (
    IdenticalScalarGapProblem,
    IdenticalScalarGapSpec,
    IdenticalScalarSpinChannel,
    _as_real_label,
    _central_charge_input,
    _clean_sample_values_by_block,
    _same_file_label,
)
from .pmp_writer import build_prefactored_pmp_doc


def load_chiral_block(path) -> tuple[object, dict]:
    """Load a CLI ``derivs_chiral.json`` -> (SerializedDerivTable, metadata)."""
    doc = json.loads(Path(path).read_text())
    basis = doc.get("basis")
    if basis != "chiral_F":
        raise ValueError(f"expected basis 'chiral_F', got {basis!r}")
    table = serialized_table_from_json(doc)
    return _relabel_chiral_as_zzb(table), doc


def _relabel_chiral_as_zzb(table):
    """Re-key the chiral table's single 'z' axis under 'zzb'.

    The PMP writer and F_pairs machinery are written against the 2d coordinate
    name; chiral_control does the same thing by serializing its reduced table as
    ``{"zzb": reduced}``. This is a pure relabel -- the chiral table already has
    one Taylor axis, and the F-pairs used here are the degenerate ``(m, 0)``, so
    nothing about the data changes.
    """
    if "zzb" in table.coordinates:
        return table
    if tuple(table.coordinates) != ("z",):
        raise ValueError(
            f"expected a single 'z' axis on a chiral block, got {table.coordinates}"
        )
    numerators = {"zzb": table.numerators["z"]}
    extra = {}
    sbv = getattr(table, "sample_block_values", None)
    if isinstance(sbv, dict) and "z" in sbv:
        extra["sample_block_values"] = {"zzb": sbv["z"]}
    return _replace(table, coordinates=("zzb",), numerators=numerators, **extra)


def build_zbar_half_problem(
    block_path,
    *,
    h_gap=0,
    lam: int | None = None,
    dps: int = 120,
    identity_regularize: str = "auto",
) -> IdenticalScalarGapProblem:
    """Assemble the zbar = 1/2 chiral problem from a CLI chiral_F block.

    ``h_gap`` is the positivity edge; the resulting bound is a 2d TWIST GAP
    statement at ``tau = 2 * h_gap``. It must agree with the block's own shift
    (see the module docstring) -- a mismatch is raised, not absorbed.
    The block must be expanded at ``z_center = 1/2``, where its odd derivatives
    implement the crossing-antisymmetric equations used here.
    It must be a variable-weight table (``variable="Delta"``), not a frozen
    point constraint or a variable-spin tower.

    Assembly uses ``dps``, independently of the ambient context. The reader
    preserves the file's stored digits, and clean sample arithmetic borrows
    their precision when it exceeds ``dps``. Input labels remain exact.
    """
    dps = _integer_field("dps", dps)
    if dps < 10:
        raise ValueError("dps must be >= 10")
    with mp.workdps(dps):
        return _build_zbar_half_problem(
            block_path, h_gap=h_gap, lam=lam, dps=dps,
            identity_regularize=identity_regularize,
        )


def _build_zbar_half_problem(block_path, *, h_gap, lam, dps, identity_regularize):
    table, doc = load_chiral_block(block_path)
    if doc.get("variable", "Delta") != "Delta":
        raise ValueError("chiral continuum construction requires variable='Delta'; "
                         "generate a chiral_F table, not a point constraint or spin tower")
    # Odd derivatives are crossing-antisymmetric only at the fixed point.
    # Compare the input label before numerical rounding: an off-center table
    # cannot be paired with the symmetric-point vacuum below.
    if literal_number(doc.get("z_center", ".5")) != literal_number(".5"):
        raise ValueError(f"block z_center {doc.get('z_center')!r} != 1/2; "
                         "regenerate the chiral_F block with --z-center 0.5")
    c = _central_charge_input(doc["central_charge"])
    external_labels = doc.get("h_external")
    if not isinstance(external_labels, list) or len(external_labels) != 4:
        raise ValueError("this construction assumes four identical externals")
    externals = [literal_number(value) for value in external_labels]
    if any(value != externals[0] for value in externals[1:]):
        raise ValueError("this construction assumes four identical externals")
    h_ext = _as_real_label(externals[0], name="h_external")
    # Match the identity vector's crossing convention: z_shift = -2 h_ext
    # and chiral norm 16. Allow metadata roundoff in nonterminating labels.
    if not _same_file_label(doc.get("z_shift", 0) or 0, -2 * externals[0], dps):
        raise ValueError(
            f"block z_shift label {doc.get('z_shift', 0)!r} != -2 h_ext; regenerate the chiral_F "
            f"block with --z-shift {-2 * externals[0]} (the crossing convention the identity vector uses)"
        )
    from virasoro_blocks.recursion import _resolve_chiral_norm

    if doc.get("chiral_normalization_N") is not None and \
            not _same_file_label(doc["chiral_normalization_N"], _resolve_chiral_norm(None), dps):
        raise ValueError(f"block chiral_normalization_N {doc['chiral_normalization_N']} != the default "
                         f"{_resolve_chiral_norm(None)} the identity vector uses; regenerate it without --chiral-norm")
    # Both of these feed the identity vector and the pole count. Defaulting them
    # to 0 would not fail -- it would quietly build the vacuum at q_order=0 and
    # declare npoles=0 on the spec, i.e. a wrong problem that still solves. A
    # block doc that lacks them is malformed, so say so.
    if doc.get("order") is None:
        raise ValueError("block doc has no 'order' (q_order); cannot build the "
                         "identity vector at a known truncation")
    if doc.get("npoles") is None:
        raise ValueError("block doc has no 'npoles'; refusing to record npoles=0 "
                         "on the spec, which would misreport the block fidelity")
    q_order = _file_integer(doc, "order", 0)
    npoles = _file_integer(doc, "npoles", 0)
    # The CLI records the derivative order as "lambda" in the block doc.
    available_lam = _infer_lam(table, doc)
    lam = _integer_field("lam", lam) if lam is not None else available_lam
    if not 1 <= lam <= available_lam:
        raise ValueError(f"lam must be between 1 and the block file's lambda {available_lam}, got {lam}")

    shift = _as_real_label(table.shift, name="block shift")
    h_gap = _as_real_label(h_gap, name="h_gap")
    # Domain labels must match exactly. Python floats use their shortest
    # decimal spelling; existing mpf values denote their supplied binary bits.
    if shift != h_gap:
        raise ValueError(
            f"block shift {mp.nstr(shift, 40)} != h_gap {mp.nstr(h_gap, 40)}. The "
            "positivity edge is the block's own shift, so regenerate the block "
            f"with --gap {mp.nstr(h_gap, 8)} (shift=0, the CLI default, is the "
            "right anchor for a MOMENT bound over the full spectrum; a twist "
            "gap at tau>0 wants shift=tau/2)."
        )

    with mp.workdps(dps):
        from .point_constraint import _crossing_context_from_inputs

        F_pairs = tuple(chiral_F_minus_pairs(lam))
        channel = IdenticalScalarSpinChannel(
            ell=0, delta_min=h_gap, serialized=table, prefactored=table,
            crossing_context=_crossing_context_from_inputs(
                basis="chiral_F", coordinate="zzb", h_ext=externals[0], lam=lam,
                c=c, q_order=q_order),
        )
        identity_vector = _chiral_identity_vector(
            mode="virasoro_symbolic", h_ext=h_ext, lam=lam, c=c,
            q_order=q_order, chiral_norm=None, F_pairs=F_pairs,
            regularize=identity_regularize,
        )
        # The vacuum's own weight in a moment DENOMINATOR, for identity_mode=
        # "include". This must be the CHIRAL vacuum, not the 2d one: the moments
        # module otherwise falls back to recomputing it from spec.c through the
        # 2d table builder, which at c=6/5, h_ext=1/2 returns 5.8284 = k0^2
        # -- the 2d vacuum F_0 F_0 -- where this problem's positivity blocks and normalization
        # vector live in the chiral family and want k0 = 2.4142. Left unset, every
        # include-mode moment overstates nu_0 by that factor of ~2.41, and because alpha/beta enter the SDP objective
        # the error is in the solved bound, not merely in a post-hoc rescale.
        # `_identity_moment_contribution` prefers this stored value precisely so
        # the identity term matches the block family.
        vacuum_measure = _exact_identity_vector(
            _chiral_symbolic_exact_table(
                h_ext=h_ext, lam=lam, c=c, q_order=q_order, chiral_norm=None,
                regularize=identity_regularize,
            ),
            ((0, 0),),
        )[0]
        pmp_build = build_prefactored_pmp_doc(
            [channel.prefactored], list(F_pairs),
            objective=[mp.mpf(0)] * len(F_pairs),
            normalization=identity_vector,
            explicit_sample_points=True,
            positive_pole_npoles=(int(npoles) if npoles is not None else None),
        )
        # the SDP data at the nodes: the file's numeric block values (never the file polynomial re-evaluated)
        sample_values_by_block = _clean_sample_values_by_block(
            [channel], pmp_build.doc, F_pairs, coordinate=table.coordinates[0],
            lowerings=pmp_build.positive_pole_lowerings,
            sampling_blocks=pmp_build.positivity_blocks,
        )
        spec = IdenticalScalarGapSpec(
            c=c, h_ext=externals[0], delta_gap=h_gap, lam=lam, spin_list=(0,),
            q_order=q_order, dps=dps,
            regularize=identity_regularize,
            npoles=int(npoles) if npoles is not None else 0,
            num_threads=1,
        )
        return IdenticalScalarGapProblem(
            spec=spec,
            resolved_method="zbar_half_chiral_from_cli",
            used_regularize=_resolved_chiral_regularize(c, identity_regularize),
            prefactor_exponent=h_ext,
            F_pairs=F_pairs,
            identity_vector=identity_vector,
            spin_channels=(channel,),
            pmp_build=pmp_build,
            sample_values_by_block=sample_values_by_block,
            objective_vector=(mp.mpf(0),) * len(F_pairs),
            normalization_vector=tuple(identity_vector),
            # a decimal string, as the global control stores it (moments reads it with mp.mpf): a raw mpf made the
            # problem unwritable (write_identical_scalar_gap_problem: "mpf is not JSON serializable")
            pmp_metadata={"self_dual_identity_measure": mp.nstr(vacuum_measure, mp.mp.dps + 5)},
        )


def _infer_lam(table, doc) -> int:
    """Derivative order of the block: the CLI records it as doc["lambda"]."""
    for key in ("lambda", "lam"):
        if doc.get(key) is not None:
            return int(doc[key])
    nums = getattr(table, "numerators", None)
    if isinstance(nums, dict) and nums:
        rows = next(iter(nums.values()))
        m_max = len(rows) - 1
        return m_max if m_max % 2 else m_max - 1
    raise ValueError("cannot infer lam from the block; pass lam explicitly")
