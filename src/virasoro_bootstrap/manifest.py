"""File-backed manifests for identical-scalar bootstrap problems.

The manifest stores the typed problem spec, crossing sector, identity
normalization vector, and references to per-spin block JSON files.
Polynomial data remain in the block files.

The block files are exactly what :func:`virasoro_blocks.writers.write_json`
already emits for a :class:`~virasoro_blocks.serialization.SerializedDerivTable`,
read back via :func:`virasoro_blocks.serialization.serialized_table_from_json`.
A fixed-spin block file is anchored at the edge it was built for
(its ``shift``; the positivity variable is ``x = Delta - shift``).  Editing the
manifest's gap rebuilds just the spins whose edge moved (numerically sampled at
their new nodes); every other table is read from its file.
To reuse block files for another spec without a manifest (another objective,
spin subset, or CLI-written files) use
:func:`~virasoro_bootstrap.identical_scalar.prepare_identical_scalar_gap_build_from_blocks`.

The public API is:

- :func:`write_identical_scalar_manifest` -- export a prepared/assembled
  identical-scalar build as a directory of block JSON files plus one manifest.
- :func:`load_identical_scalar_manifest` -- parse the manifest (validating the
  schema and every referenced block's ``content_hash``).
- :func:`build_identical_scalar_problem_from_manifest` -- rebuild the
  :class:`~virasoro_bootstrap.identical_scalar.IdenticalScalarGapProblem` from
  the manifest by loading the referenced block files instead of regenerating the
  blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dataclass_replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import mpmath as mp

from virasoro_blocks.singularity import recognized_singular_locus
from virasoro_blocks.numeric_input import literal_number

from virasoro_blocks.serialization import (
    SerializedDerivTable,
    _parse_serialized_number,
    serialized_table_from_json,
)
from virasoro_blocks.recursion import _resolve_chiral_norm
from virasoro_blocks.writers import write_json, _metadata_num_str
from virasoro_blocks._conventions import SCALAR_G_SPIN0_KEY, scalar_G_spin0_convention

from .crossing import F_minus_pairs
from .identical_scalar import (
    IdenticalScalarGapProblem,
    IdenticalScalarGapSpec,
    IdenticalScalarSpinChannel,
    _canonical_non_scalar_delta_min_overrides,
    _check_block_doc,
    _check_block_target,
    _clean_sample_values_by_block,
    _fixed_spin_sampler_for,
    _normalize_npoles,
    _resolved_large_spin_npoles,
    _resolved_delta_min_by_ell,
    _resolved_method,
    _resolved_positive_pole_npoles,
    _file_backed_exact_tables,
    _spin_channels_at_edges,
    _resolved_regularize,
    _stringify_number,
    _tower_block_twist,
    _validate_gap_spec_common,
    _validate_identical_scalar_operator,
)
from .pmp_writer import build_prefactored_pmp_doc


# 1.1: identity.functional -> identity.crossing_vector (drop "stage"); add the
# explicit "pmp" block (problem_kind + objective + normalization). 1.0 manifests
# still parse (from_json accepts the old keys; absent "pmp" -> gap convention).
# 1.2: written labels are exact (p/q or terminating decimals) and every block
# hash covers the stored sample values. A 1.0/1.1 manifest may come from an older
# writer: its references may hash without the sample values
# (block_content_hash_matches(legacy=True)) and its c, h_ext, gaps and twists
# may be decimals rounded at the spec dps (_legacy_label).
# 1.3: the manifest records the spin-0 scalar_G normalization its blocks and identity vector were built in as
# "scalar_G_spin0". A scale-free manifest (gap, chiral) is rebuilt in its own normalization in any process; one
# whose answer carries the scale is rebuilt only in a process that uses it.
MANIFEST_SCHEMA_VERSION = "1.3"
MANIFEST_KIND_IDENTICAL_SCALAR = "identical_scalar_gap"
_CHIRAL_METHODS = {
    "zbar_half_chiral_from_cli", "chiral_global_control",
    "chiral_virasoro_control", "chiral_virasoro_symbolic_control",
}
_MOMENT_PROBLEM_KINDS = frozenset({
    "moment_upper", "moment_lower", "normalized_moment_lower", "normalized_moment_upper",
    "normalized_ratio_probe", "normalized_ratio_lower_probe",
})


# ---------------------------------------------------------------------------
# Content hashing over the canonical block payload.
# ---------------------------------------------------------------------------

# The payload fields that define the block table itself (and therefore its
# reuse identity). Metadata such as `args`, `output_template`, or
# `precision_actual` is excluded so a hash is stable across cosmetic re-emission.
_HASHED_BLOCK_FIELDS = (
    "exp_base",
    "variable_minus_x",
    "variable",
    "pole_list_x",
    "coordinates",
    "derivs",
    "sample_points_x",
    "sample_block_values",
    "sample_block_values_index_names",
)
# The field set hashed by manifests written before 2026-10-03 (efcb85d2), which
# did not yet cover the stored sample values.
_LEGACY_HASHED_BLOCK_FIELDS = tuple(
    key for key in _HASHED_BLOCK_FIELDS
    if key not in {"sample_block_values", "sample_block_values_index_names"}
)


def _hash_block_fields(doc: dict, fields) -> str:
    payload = {key: doc[key] for key in fields if key in doc}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def block_content_hash(doc: dict) -> str:
    """Stable hash over a block document's canonical table payload.

    Covers ``exp_base``, ``variable_minus_x``, ``variable``, ``pole_list_x``,
    ``coordinates``, the jagged ``derivs``, ``sample_points_x``, and the stored
    ``sample_block_values`` with their ``sample_block_values_index_names``.
    The sample layout marker determines how the values are read and is part
    of their meaning. Cosmetic metadata is intentionally excluded so the hash
    is a reuse identity, not a byte hash of the file. References written
    before the sample values were hashed are checked by
    :func:`block_content_hash_matches`.
    """
    return _hash_block_fields(doc, _HASHED_BLOCK_FIELDS)


def block_content_hash_matches(doc: dict, expected: str, *, legacy: bool = False) -> bool:
    """Whether ``expected`` is the content hash of ``doc``.

    With ``legacy`` (a schema 1.0/1.1 manifest), a hash over the legacy field set is
    also accepted: writers before 2026-10-03 hashed the table without the stored
    sample values (:data:`_LEGACY_HASHED_BLOCK_FIELDS`). The legacy hash is computed
    only when the current one mismatches.
    """
    if block_content_hash(doc) == expected:
        return True
    return legacy and _hash_block_fields(doc, _LEGACY_HASHED_BLOCK_FIELDS) == expected


# ---------------------------------------------------------------------------
# Labels written by older writers.
# ---------------------------------------------------------------------------

# Below this many significant digits a decimal is a typed literal, never a rounded value.
_LEGACY_LABEL_MIN_DIGITS = 20
# The block-file fields that label the block's inputs (compared against the spec).
_BLOCK_INPUT_LABELS = ("central_charge", "h_external", "hbar_external", "delta_external", "twist")


def _legacy_label(value, dps):
    """Read exact labels, recovering short rationals from legacy decimal labels.

    Schema 1.0/1.1 labels may contain mpf values printed to ``dps + 5``
    significant digits. A decimal with at least 20 significant digits reads
    as a rational if its relative distance is at most ``10 * 10^-dps`` and
    the denominator is at most ``10^(dps // 3)``. Otherwise, read its exact
    literal value. ``dps=None`` reads all labels exactly.
    """
    from decimal import Decimal, InvalidOperation

    exact = literal_number(value)
    if dps is None or not isinstance(value, str) or not isinstance(exact, Fraction) or exact == 0:
        return exact
    try:
        digits = "".join(map(str, Decimal(value.strip()).as_tuple().digits)).lstrip("0")
    except (InvalidOperation, ValueError):
        return exact
    if len(digits) < _LEGACY_LABEL_MIN_DIGITS:
        return exact
    candidate = Fraction(exact).limit_denominator(10 ** max(1, int(dps) // 3))
    if candidate != exact and abs(candidate - exact) <= abs(exact) * Fraction(10, 10 ** int(dps)):
        return literal_number(f"{candidate.numerator}/{candidate.denominator}")
    return exact


def _legacy_label_str(value, dps):
    """:func:`_legacy_label` of a label, spelled exactly; unchanged when nothing was recognized."""
    if dps is None or not isinstance(value, str):
        return value
    label = _legacy_label(value, dps)
    return value if label == literal_number(value) else _metadata_num_str(label, int(dps) + 5)


def _is_legacy_manifest(schema_version) -> bool:
    """Whether a manifest predates schema 1.2 (exact labels, sample values in every hash)."""
    try:
        minor = int(str(schema_version).split(".")[1])
    except (IndexError, ValueError):
        return False
    return minor < 2


def _legacy_label_dps(manifest) -> int | None:
    """The dps an older writer rounded the manifest's labels at, or None for a current manifest."""
    if not _is_legacy_manifest(manifest.schema_version):
        return None
    return _manifest_integer(manifest.spec.get("dps", manifest.precision_dps), "spec.dps", minimum=1)


# ---------------------------------------------------------------------------
# Manifest dataclasses.
# ---------------------------------------------------------------------------


def _manifest_integer(value, name, *, minimum=0):
    """Validate integer labels before conversion can truncate their inputs."""
    from virasoro_blocks.requests import _integer_field

    try:
        integer = _integer_field(name, literal_number(value))
        valid = integer >= minimum
    except (TypeError, ValueError, ArithmeticError):
        valid = False
    if not valid:
        raise ValueError(f"manifest {name} must be an integer >= {minimum}, got {value!r}")
    return integer


@dataclass(frozen=True)
class BlockRef:
    """A reference to a block JSON file plus the gap edge it is used at.

    ``path`` is manifest-relative (never absolute). ``variable`` mirrors the
    block file's positivity variable (``"Delta"`` for a fixed-spin gap channel,
    ``"L"`` for a large-spin tower channel, ``"point"`` for a point constraint).
    ``delta_min`` is the bootstrap lower edge for a fixed-spin channel; for a
    large-spin channel it holds the tower floor ``Lmax``; for a point channel it
    holds the frozen dimension ``point_value``. ``kind`` is ``"fixed_spin"``,
    ``"large_spin"``, ``"point"``, or ``"chiral"``. A chiral channel uses
    the block schema's ``variable="Delta"`` with a ``chiral_F`` / ``global_chiral_F``
    basis (its argument is h), a null file spin, and the placeholder ``ell=0``
    on the reference. ``twist`` and ``large_spin_lmin`` are
    recorded only for large-spin channels; ``point_value`` only for point ones.
    ``content_hash`` must match :func:`block_content_hash` of the referenced
    file.

    ``shift`` and ``npoles`` are informational diagnostics mirroring the
    serialized block's polynomial-variable origin (``serialized.shift``) and
    its post-normalization pole count (``len(prefactored.pole_list_x)``). They
    are not consumed by the reload (the block file itself carries the
    authoritative data); they exist so the manifest is a self-describing
    superset of the old ``problem.json`` summary.
    """

    path: str
    ell: int
    variable: str
    delta_min: str | None
    content_hash: str
    kind: str = "fixed_spin"
    twist: str | None = None
    large_spin_lmin: str | None = None
    point_value: str | None = None
    shift: str | None = None
    npoles: int | None = None

    def to_json(self) -> dict:
        doc = {
            "path": self.path,
            "ell": self.ell,
            "variable": self.variable,
            "delta_min": self.delta_min,
            "content_hash": self.content_hash,
            "kind": self.kind,
        }
        if self.twist is not None:
            doc["twist"] = self.twist
        if self.large_spin_lmin is not None:
            doc["large_spin_lmin"] = self.large_spin_lmin
        if self.point_value is not None:
            doc["point_value"] = self.point_value
        if self.shift is not None:
            doc["shift"] = self.shift
        if self.npoles is not None:
            doc["npoles"] = self.npoles
        return doc

    @staticmethod
    def from_json(doc: dict) -> "BlockRef":
        npoles = doc.get("npoles")
        return BlockRef(
            path=doc["path"],
            ell=_manifest_integer(doc["ell"], "block ell"),
            variable=doc["variable"],
            delta_min=doc.get("delta_min"),
            content_hash=doc["content_hash"],
            kind=doc.get("kind", "fixed_spin"),
            twist=doc.get("twist"),
            large_spin_lmin=doc.get("large_spin_lmin"),
            point_value=doc.get("point_value"),
            shift=doc.get("shift"),
            npoles=None if npoles is None else _manifest_integer(npoles, "block npoles"),
        )


@dataclass(frozen=True)
class CrossingSpec:
    """The crossing sector context a single block file cannot hold."""

    sector_kind: str
    coordinate: str
    crossing_pairs: tuple[tuple[int, int], ...]
    prefactor_exponent: str

    def to_json(self) -> dict:
        return {
            "sector_kind": self.sector_kind,
            "coordinate": self.coordinate,
            "crossing_pairs": [list(pair) for pair in self.crossing_pairs],
            "prefactor_exponent": self.prefactor_exponent,
        }

    @staticmethod
    def from_json(doc: dict) -> "CrossingSpec":
        return CrossingSpec(
            sector_kind=doc["sector_kind"],
            coordinate=doc["coordinate"],
            crossing_pairs=tuple(tuple(pair) for pair in doc["crossing_pairs"]),
            prefactor_exponent=doc["prefactor_exponent"],
        )


@dataclass(frozen=True)
class IdentityVector:
    """The identity operator's crossing vector (the flattened F^- derivatives).

    The identity contribution is finite at ``Delta = 0`` and carries no
    removable pole structure once evaluated, so unlike the positivity channels
    (stored as ``blocks/*.json`` file references) it is stored directly inline as
    the crossing-applied vector. This is provenance -- the *role* this vector
    plays in the SDP (objective vs normalization) is recorded explicitly in
    :class:`PmpVectors`, not here.
    """

    crossing_vector: tuple[str, ...]

    def to_json(self) -> dict:
        return {"crossing_vector": list(self.crossing_vector)}

    @staticmethod
    def from_json(doc: dict) -> "IdentityVector":
        # schema 1.1 uses "crossing_vector"; schema 1.0 used "functional" (with a
        # now-dropped "stage" tag). Accept both so old manifests still load.
        vec = doc.get("crossing_vector", doc.get("functional"))
        return IdentityVector(
            crossing_vector=tuple(vec),
        )


@dataclass(frozen=True)
class PmpVectors:
    """The two SDP scalar vectors and which problem they encode.

    The crossing sum rule is turned into an SDP by choosing an objective and a
    normalization vector in the F^- derivative basis:

      * ``problem_kind == "gap"``: a feasibility problem -- ``objective`` is the
        zero vector and ``normalization`` is the identity crossing vector V_id
        (the dual functional is normalized on the identity, omega . V_id = 1).
      * ``problem_kind == "ope_max"``: an extremal problem -- ``objective`` is
        the identity crossing vector V_id and ``normalization`` is the
        distinguished operator's crossing vector V_* at ``(spin_star,
        delta_star)``; the optimal objective bounds that operator's squared OPE
        coefficient.

    ``problem_kind`` is any builder's kind (``gap``, the ``ope_*`` kinds,
    ``fixed_ope_feasibility``, the moment kinds); the two above are examples.
    Both vectors are stored explicitly (rather than derived from
    ``problem_kind``) so the manifest is self-describing.
    """

    problem_kind: str
    objective: tuple[str, ...]
    normalization: tuple[str, ...]
    # Provenance of the normalization operator for ope_max (None for gap):
    # {"spin_star": int, "delta_star": str}; a fixed-OPE problem adds "ope_squared", the operator's
    # p = N^Delta lambda^2 against the identity's 1 in any spin.
    normalization_operator: dict | None = None

    def to_json(self) -> dict:
        out = {
            "problem_kind": self.problem_kind,
            "objective": list(self.objective),
            "normalization": list(self.normalization),
        }
        if self.normalization_operator is not None:
            out["normalization_operator"] = dict(self.normalization_operator)
        return out

    @staticmethod
    def from_json(doc: dict) -> "PmpVectors":
        return PmpVectors(
            problem_kind=doc.get("problem_kind", "gap"),
            objective=tuple(doc["objective"]),
            normalization=tuple(doc["normalization"]),
            normalization_operator=doc.get("normalization_operator"),
        )


@dataclass(frozen=True)
class BootstrapManifest:
    """A thin manifest referencing per-spin block files for one problem."""

    schema_version: str
    manifest_kind: str
    precision_dps: int
    spec: dict
    crossing: CrossingSpec
    identity: IdentityVector
    blocks: tuple[BlockRef, ...]
    provenance: dict
    # The SDP objective/normalization (schema 1.1+). None for schema-1.0
    # manifests, where reconstruction falls back to the gap convention
    # (objective = 0, normalization = identity).
    pmp: PmpVectors | None = None
    # The spin-0 scalar_G normalization of the blocks and identity vector (schema 1.3+; see
    # _manifest_scalar_G_spin0 for a manifest without it).
    scalar_G_spin0: str | None = None

    def to_json(self) -> dict:
        doc = {
            "schema_version": self.schema_version,
            "manifest_kind": self.manifest_kind,
            "precision_dps": self.precision_dps,
            "spec": self.spec,
            "crossing": self.crossing.to_json(),
            "identity": self.identity.to_json(),
            "blocks": [ref.to_json() for ref in self.blocks],
            "provenance": self.provenance,
        }
        if self.pmp is not None:
            doc["pmp"] = self.pmp.to_json()
        if self.scalar_G_spin0 is not None:
            doc[SCALAR_G_SPIN0_KEY] = self.scalar_G_spin0
        return doc

    @staticmethod
    def from_json(doc: dict) -> "BootstrapManifest":
        return BootstrapManifest(
            schema_version=doc["schema_version"],
            manifest_kind=doc["manifest_kind"],
            precision_dps=_manifest_integer(doc["precision_dps"], "precision_dps", minimum=1),
            spec=doc["spec"],
            crossing=CrossingSpec.from_json(doc["crossing"]),
            identity=IdentityVector.from_json(doc["identity"]),
            blocks=tuple(BlockRef.from_json(b) for b in doc["blocks"]),
            provenance=dict(doc.get("provenance", {})),
            pmp=PmpVectors.from_json(doc["pmp"]) if doc.get("pmp") is not None else None,
            scalar_G_spin0=doc.get(SCALAR_G_SPIN0_KEY),
        )


# ---------------------------------------------------------------------------
# Typed spec <-> JSON.
#
# The spec is round-tripped losslessly: every field that affects block or PMP
# assembly is stored, high-precision numbers as strings.
# ---------------------------------------------------------------------------

# Plain (JSON-native) spec fields copied verbatim in both directions.
_SPEC_PLAIN_FIELDS = (
    "lam",
    "spin_list",
    "q_order",
    "dps",
    "npoles",
    "n_sample_points",
    "pole_selection",
    "positive_pole_npoles",
    "positive_pole_selection",
    "regularize",
    "coordinate",
    "chiral_norm",
    "explicit_sample_points",
    "shared_sample_points",
    "num_threads",
    "large_spin_pole_selection",
    "large_spin_npoles",
    "sample_via",
    # (opt-in: the digits each fixed-spin block was verified to deliver, and the escalation cap)
    "target_digits",
    "max_dps",
)


def _spec_to_json(spec: IdenticalScalarGapSpec) -> dict:
    resolved_chiral_norm = _resolve_chiral_norm(spec.chiral_norm)
    if resolved_chiral_norm != mp.floor(resolved_chiral_norm):
        raise ValueError(
            "resolved chiral_norm must be integral to serialize in the manifest"
        )
    out: dict = {
        "c": _metadata_num_str(spec.c, spec.dps + 5),
        "h_ext": _metadata_num_str(literal_number(spec.h_ext), spec.dps + 5),
        "delta_gap": _metadata_num_str(literal_number(spec.delta_gap), spec.dps + 5),
        "prefactor_exponent": (
            None
            if spec.prefactor_exponent is None
            else _metadata_num_str(literal_number(spec.prefactor_exponent), spec.dps + 5)
        ),
        "large_spin_lmin": (
            None
            if spec.large_spin_lmin is None
            else _stringify_number(spec.large_spin_lmin)
        ),
        "large_spin_twists": [
            _metadata_num_str(literal_number(t), spec.dps + 5)
            for t in spec.large_spin_twists
        ],
        "point_constraints": [
            [int(ell), _metadata_num_str(literal_number(delta_point), spec.dps + 5)]
            for ell, delta_point in spec.point_constraints
        ],
    }
    for field in _SPEC_PLAIN_FIELDS:
        value = getattr(spec, field)
        if field == "chiral_norm":
            out[field] = int(resolved_chiral_norm)
            continue
        out[field] = list(value) if isinstance(value, tuple) else value
    if spec.delta_min_by_spin:
        out["delta_min_by_spin"] = {
            str(int(ell)): _metadata_num_str(literal_number(delta_min), spec.dps + 5)
            for ell, delta_min in spec.delta_min_by_spin.items()
        }
    else:
        out["delta_min_by_spin"] = None
    return out


def _spec_from_json(doc: dict, *, legacy: bool = False) -> IdenticalScalarGapSpec:
    """The spec of a manifest; ``legacy`` (schema 1.0/1.1) reads rounded labels as in :func:`_legacy_label`."""
    # Parse at the stored precision to preserve the spec's numeric labels.
    with mp.workdps(_manifest_integer(doc.get("dps", mp.mp.dps), "spec.dps", minimum=1)):
        return _spec_from_json_at_dps(doc, legacy_dps=mp.mp.dps if legacy else None)


def _spec_from_json_at_dps(doc: dict, *, legacy_dps=None) -> IdenticalScalarGapSpec:
    dps = legacy_dps
    kwargs: dict = {
        "c": _legacy_label(doc["c"], dps),
        "h_ext": _legacy_label(doc["h_ext"], dps),
        "delta_gap": _legacy_label(doc["delta_gap"], dps),
        "prefactor_exponent": (
            None
            if doc.get("prefactor_exponent") is None
            else _legacy_label(doc["prefactor_exponent"], dps)
        ),
        "large_spin_lmin": (
            None
            if doc.get("large_spin_lmin") is None
            else _manifest_integer(doc["large_spin_lmin"], "large_spin_lmin")
        ),
        "large_spin_twists": tuple(
            _legacy_label(t, dps) for t in doc.get("large_spin_twists", [])
        ),
        "point_constraints": tuple(
            (_manifest_integer(ell, "point constraint spin"), _legacy_label(delta_point, dps))
            for ell, delta_point in doc.get("point_constraints", [])
        ),
    }
    for field in _SPEC_PLAIN_FIELDS:
        if field not in doc:
            continue
        value = doc[field]
        kwargs[field] = tuple(value) if field == "spin_list" else value
    overrides = doc.get("delta_min_by_spin")
    if overrides:
        kwargs["delta_min_by_spin"] = {
            _manifest_integer(ell, "delta_min_by_spin key"): _legacy_label(delta_min, dps)
            for ell, delta_min in overrides.items()
        }
    return IdenticalScalarGapSpec(**kwargs)


def _block_settings_json(spec: IdenticalScalarGapSpec) -> dict:
    """The spec fields a manifest reload cannot honor from its block files when they are edited.

    Block-file checks cover c, h_ext, lambda, order, N, npoles, and pole
    selection. These additional settings record node sharing, point constraints,
    and the sampling method so a reload can reject incompatible spec edits.
    """
    return {
        "sample_via": spec.sample_via,
        "shared_sample_points": bool(spec.shared_sample_points),
        "point_constraints": [
            [int(ell), _metadata_num_str(literal_number(delta_point), spec.dps + 5)] for ell, delta_point in spec.point_constraints
        ],
    }


def _refuse_unhonoured_spec_edits(manifest: BootstrapManifest, spec: IdenticalScalarGapSpec, manifest_path) -> None:
    """Refuse an edit of a manifest spec field its block files cannot honor (see :func:`_block_settings_json`).

    Manifests written before 2026-09-29 record no settings and are not checked.
    """
    recorded = manifest.provenance.get("block_settings")
    if recorded is None:
        return
    legacy_dps = _legacy_label_dps(manifest)

    def same_points(a, b):
        return len(a) == len(b) and all(
            _manifest_integer(ea, "point constraint spin") == _manifest_integer(eb, "point constraint spin")
            and _legacy_label(da, legacy_dps) == _legacy_label(db, legacy_dps)
            for (ea, da), (eb, db) in zip(a, b)
        )

    with mp.workdps(spec.dps):
        current = _block_settings_json(spec)
        for name, built in recorded.items():
            now = current.get(name)
            if name == "point_constraints" and same_points(built, now):
                continue
            if name != "point_constraints" and built == now:
                continue
            raise ValueError(
                f"manifest {manifest_path}: spec.{name} was edited (the block files were built with {built!r}, the "
                f"spec now says {now!r}); the files cannot honor the edit: rebuild the problem from the edited spec"
            )


def _without_recorded_prefactor(spec: IdenticalScalarGapSpec, manifest: BootstrapManifest) -> IdenticalScalarGapSpec:
    """Drop the spec's ``prefactor_exponent`` when it is the one the manifest was written with.

    A manifest records ``prefactor_exponent = h_ext``. Clearing an unchanged
    prefactor lets the block-file check report an incompatible ``h_ext`` edit.
    An explicitly edited prefactor is retained for spec validation.
    """
    if spec.prefactor_exponent is None:
        return spec
    written = _legacy_label(manifest.crossing.prefactor_exponent, _legacy_label_dps(manifest))
    value = literal_number(spec.prefactor_exponent)
    if not isinstance(value, Fraction):
        raise ValueError("manifest spec.prefactor_exponent must be a real decimal or rational label")
    if value == written:
        return _dataclass_replace(spec, prefactor_exponent=None)
    return spec


def _check_crossing_prefactor(manifest: BootstrapManifest, block_doc: dict, block_path) -> None:
    """Match the derived crossing label against the referenced block externals.

    Older writers rounded this derived field at the manifest precision. Accept
    that exactly reproducible spelling as well as the exact external label;
    never use a proximity tolerance to erase an edit.
    """
    try:
        externals = tuple(literal_number(value) for value in block_doc["h_external"])
        written = _legacy_label(manifest.crossing.prefactor_exponent, _legacy_label_dps(manifest))
        if not isinstance(written, Fraction):
            raise ValueError("crossing labels must be real decimals or rationals")
        if len(externals) != 4 or not all(
            isinstance(value, Fraction) and value == externals[0] for value in externals
        ):
            raise ValueError("identical-scalar block externals must agree")
        external = externals[0]
        if written == external:
            return
        with mp.workdps(manifest.precision_dps):
            old_label = literal_number(_stringify_number(external))
        if written == old_label:
            return
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"manifest crossing.prefactor_exponent cannot match h_external in {block_path}") from exc
    raise ValueError(
        f"manifest crossing.prefactor_exponent disagrees with h_external in {block_path}; "
        "the identical-scalar crossing prefactor is fixed at h_ext"
    )


# ---------------------------------------------------------------------------
# Writer.
# ---------------------------------------------------------------------------


def _precision_provenance(problem, fixed_channels) -> dict:
    """The build's precision record: the working precision the data are written at (``spec.dps``), the dps asked
    for when a ``target_digits`` build raised it, the target and cap, and the fewest verified digits of any
    fixed-spin block (per entry and against the largest entry)."""
    spec = problem.spec

    def fewest(name):
        values = [getattr(channel.serialized, name, None) for channel in fixed_channels]
        values = [float(v) for v in values if v is not None]
        return None if not values else round(min(values), 1)

    requested = getattr(problem, "requested_dps", None)
    return {
        "working_dps": int(spec.dps),
        "dps_requested": int(requested if requested is not None else spec.dps),
        "target_digits": spec.target_digits,
        "max_dps": spec.max_dps,
        "verified_digits": fewest("verified_digits"),
        "verified_digits_largest_entry": fewest("verified_digits_largest_entry"),
    }


def _block_filename(ell: int) -> str:
    # Mirror the standalone block CLI's default `derivs_{spin}.json` naming so
    # CLI- and bootstrap-produced fixed-spin block files share one convention.
    return f"derivs_{ell}.json"


def _large_spin_block_filename(index: int) -> str:
    # Large-spin tower files are keyed by twist index, matching the block CLI's
    # `--large-spin-*` emit path (`derivs_large_spin_{index}.json`).
    return f"derivs_large_spin_{index}.json"


def _point_block_filename(index: int) -> str:
    # Point-constraint files are keyed by point index, matching the block CLI's
    # explicit `point-table` operation (`derivs_point_{index}.json`).
    return f"derivs_point_{index}.json"


def _chiral_storage_table(table, lam):
    """Persist the single chiral axis under z, undoing the PMP's zzb adapter."""
    rows = table.numerators["zzb"]
    if len(rows) < lam + 1 or any(len(row) != 1 for row in rows):
        raise ValueError("a chiral manifest requires a single-column table through the requested lambda")
    values = table.sample_block_values
    return _dataclass_replace(
        table, coordinates=("z",), numerators={"z": rows[:lam + 1]},
        sample_block_values=None if values is None else {"z": [row[:lam + 1] for row in values["zzb"]]},
    )


def write_identical_scalar_manifest(
    problem: IdenticalScalarGapProblem,
    output_dir,
    *,
    blocks_subdir: str = "blocks",
) -> Path:
    """(see :func:`_write_identical_scalar_manifest`), written at the problem's precision whatever the ambient one."""
    with mp.workdps(int(problem.spec.dps)):
        return _write_identical_scalar_manifest(problem, output_dir, blocks_subdir=blocks_subdir)


def _write_identical_scalar_manifest(
    problem: IdenticalScalarGapProblem,
    output_dir,
    *,
    blocks_subdir: str = "blocks",
) -> Path:
    """Write *problem* as a manifest plus a directory of block JSON files.

    Each positivity channel the problem uses (fixed-spin, tower, point) is
    emitted with :func:`virasoro_blocks.writers.write_json`; the manifest references those
    files, recording each channel's ``delta_min`` on its :class:`BlockRef`. The
    identity normalization vector is stored inline as the flattened F^- vector.

    Large-spin tower channels (when the spec requests them) are emitted as
    ``variable="L"`` block files keyed by twist index, with their twist and
    ``Lmax`` floor recorded on the :class:`BlockRef`.

    Returns the path to the written ``manifest.json``.
    """
    output_dir = Path(output_dir)
    blocks_dir = output_dir / blocks_subdir
    blocks_dir.mkdir(parents=True, exist_ok=True)

    spec = problem.spec
    h_ext = literal_number(spec.h_ext)
    chiral_norm_value = spec.chiral_norm
    block_basis = "global_scalar_G" if problem.resolved_method == "global_control" else "scalar_G"
    chiral = problem.resolved_method in _CHIRAL_METHODS
    if chiral:
        block_basis = "global_chiral_F" if problem.resolved_method == "chiral_global_control" else "chiral_F"
    sig_figs = spec.dps + 5

    # The problem orders spin_channels as the fixed-spin gap channels (in
    # spin_list order), then the gap-independent large-spin tower channels (in
    # large_spin_twists order), then the point-constraint channels. The tower
    # and point tails are mirrored on typed problem fields so each kind is
    # classified explicitly rather than by position.
    n_fixed = len(spec.spin_list)
    fixed_channels = problem.spin_channels[:n_fixed]
    if len(problem.large_spin_channels) != len(spec.large_spin_twists):
        raise ValueError(
            "large-spin channel/twist count mismatch: "
            f"{len(problem.large_spin_channels)} channels vs {len(spec.large_spin_twists)} twists"
        )
    # only the channels the problem actually uses (a moment problem built with include_tower=False leaves the
    # tower out of its positivity blocks, and so out of its block files)
    in_problem = {id(channel) for channel in problem.spin_channels}
    tower_pairs = [(channel, twist) for channel, twist in zip(problem.large_spin_channels, spec.large_spin_twists)
                   if id(channel) in in_problem]
    tower_channels = tuple(channel for channel, _ in tower_pairs)
    tower_twists = tuple(twist for _, twist in tower_pairs)
    point_channels = tuple(channel for channel in problem.point_channels if id(channel) in in_problem)
    if len(fixed_channels) + len(tower_channels) + len(point_channels) != len(
        problem.spin_channels
    ):
        raise ValueError(
            "channel partition mismatch: "
            f"{len(fixed_channels)} fixed + {len(tower_channels)} tower + "
            f"{len(point_channels)} point != {len(problem.spin_channels)} total"
        )

    block_z_shift = -2 * h_ext
    # the convention the problem's blocks and identity vector were built in (not whichever is active now)
    convention = getattr(problem, "scalar_G_spin0", None) or scalar_G_spin0_convention()
    block_refs: list[BlockRef] = []
    for channel in fixed_channels:
        doc_str = write_json(
            _chiral_storage_table(channel.serialized, spec.lam) if chiral else channel.serialized,
            c=spec.c,
            h_ext=[h_ext] * 4,
            hbar_ext=[h_ext] * 4,
            spin=channel.ell,
            q_order=spec.q_order,
            npoles=_normalize_npoles(spec.npoles),
            pole_selection=spec.pole_selection if isinstance(_normalize_npoles(spec.npoles), int) else None,
            basis=block_basis,
            chiral_norm=chiral_norm_value,
            # the identical-scalar blocks are built with z_shift = zbar_shift = -2 h_ext (the file recorded 0)
            z_shift=block_z_shift,
            zbar_shift=block_z_shift,
            precision_input=spec.dps,
            variable="Delta",
            sample_via=spec.sample_via,
            scalar_G_spin0=convention,
        )
        doc = json.loads(doc_str)
        filename = _block_filename(channel.ell)
        with open(blocks_dir / filename, "w") as f:
            f.write(doc_str)
        block_refs.append(
            BlockRef(
                path=f"{blocks_subdir}/{filename}",
                ell=channel.ell,
                variable="Delta",
                delta_min=_metadata_num_str(literal_number(channel.delta_min), spec.dps + 5),
                content_hash=block_content_hash(doc),
                kind="chiral" if chiral else "fixed_spin",
                shift=_metadata_num_str(literal_number(channel.serialized.shift), spec.dps + 5),
                npoles=len(channel.prefactored.pole_list_x),
            )
        )

    for index, (channel, twist) in enumerate(zip(tower_channels, tower_twists)):
        # Tower channels are polynomials in the spin L valid for L >= Lmax; the
        # channel's delta_min holds Lmax. Persist as a variable="L" block file
        # with its twist and lower spin edge so the reload reconstructs it.
        doc_str = write_json(
            channel.serialized,
            c=spec.c,
            h_ext=[h_ext] * 4,
            hbar_ext=[h_ext] * 4,
            spin=channel.ell,
            q_order=spec.q_order,
            # the tower's own kept-pole count (large_spin_npoles, else npoles)
            npoles=_resolved_large_spin_npoles(spec),
            pole_selection=spec.large_spin_pole_selection,
            basis=block_basis,
            chiral_norm=chiral_norm_value,
            # the identical-scalar blocks are built with z_shift = zbar_shift = -2 h_ext (the file recorded 0)
            z_shift=block_z_shift,
            zbar_shift=block_z_shift,
            precision_input=spec.dps,
            variable="L",
            twist=twist,
            large_spin_lmin=channel.delta_min,
            scalar_G_spin0=convention,
        )
        doc = json.loads(doc_str)
        filename = _large_spin_block_filename(index)
        with open(blocks_dir / filename, "w") as f:
            f.write(doc_str)
        block_refs.append(
            BlockRef(
                path=f"{blocks_subdir}/{filename}",
                ell=channel.ell,
                variable="L",
                delta_min=_stringify_number(channel.delta_min),
                content_hash=block_content_hash(doc),
                kind="large_spin",
                twist=_metadata_num_str(literal_number(twist), sig_figs),
                large_spin_lmin=_stringify_number(channel.delta_min),
                shift=_stringify_number(channel.serialized.shift),
                npoles=len(channel.prefactored.pole_list_x),
            )
        )

    for index, channel in enumerate(point_channels):
        # Point channels are degree-0 blocks frozen at delta_point (= the
        # channel's delta_min). Persist as a variable="point" block file with the
        # frozen value so the reload re-wraps the constant block.
        doc_str = write_json(
            _chiral_storage_table(channel.serialized, spec.lam) if chiral else channel.serialized,
            c=spec.c,
            h_ext=[h_ext] * 4,
            hbar_ext=[h_ext] * 4,
            spin=channel.ell,
            q_order=spec.q_order,
            npoles=len(channel.prefactored.pole_list_x),
            basis=block_basis,
            chiral_norm=chiral_norm_value,
            # the identical-scalar blocks are built with z_shift = zbar_shift = -2 h_ext (the file recorded 0)
            z_shift=block_z_shift,
            zbar_shift=block_z_shift,
            precision_input=spec.dps,
            variable="point",
            point_value=channel.delta_min,
            scalar_G_spin0=convention,
        )
        doc = json.loads(doc_str)
        filename = _point_block_filename(index)
        with open(blocks_dir / filename, "w") as f:
            f.write(doc_str)
        block_refs.append(
            BlockRef(
                path=f"{blocks_subdir}/{filename}",
                ell=channel.ell,
                variable="point",
                delta_min=doc["point_value"],
                content_hash=block_content_hash(doc),
                kind="point",
                point_value=doc["point_value"],
                shift=_stringify_number(channel.serialized.shift),
                npoles=len(channel.prefactored.pole_list_x),
            )
        )

    crossing = CrossingSpec(
        sector_kind="chiral_F_minus" if chiral else "identical_scalar_F_minus",
        coordinate=spec.coordinate,
        crossing_pairs=tuple(problem.F_pairs),
        prefactor_exponent=_metadata_num_str(literal_number(problem.prefactor_exponent), sig_figs),
    )
    identity = IdentityVector(
        crossing_vector=tuple(_stringify_number(v) for v in problem.identity_vector),
    )
    norm_op = None
    if problem.normalization_operator is not None:
        norm_op = {
            "spin_star": int(problem.normalization_operator["spin_star"]),
            "delta_star": _metadata_num_str(
                literal_number(problem.normalization_operator["delta_star"]), sig_figs
            ),
        }
        if "direction" in problem.normalization_operator:
            norm_op["direction"] = problem.normalization_operator["direction"]
        if "ope_squared" in problem.normalization_operator:
            norm_op["ope_squared"] = _metadata_num_str(
                literal_number(problem.normalization_operator["ope_squared"]), sig_figs
            )
    pmp = PmpVectors(
        problem_kind=problem.problem_kind,
        objective=tuple(_stringify_number(v) for v in problem.objective_vector),
        normalization=tuple(_stringify_number(v) for v in problem.normalization_vector),
        normalization_operator=norm_op,
    )
    provenance = {
        "resolved_method": problem.resolved_method,
        "used_regularize": problem.used_regularize,
        # at a recognized singular c the blocks are the block AT c (merged poles of the one Laurent table)
        "regularization_policy": "merged" if problem.used_regularize else "direct",
        "singular_locus": recognized_singular_locus(spec.c),
        # the spec fields the block files were built with but do not themselves record (checked on reload)
        "block_settings": _block_settings_json(spec),
        "precision": _precision_provenance(problem, fixed_channels),
    }
    if problem.pmp_metadata is not None:
        provenance["pmp_metadata"] = problem.pmp_metadata
    if chiral or problem.resolved_method in {"global_control", "numeric_virasoro_control"}:
        # Global operator blocks carry no c/order, but their stored vacuum may
        # still be Virasoro. Those inputs cannot be edited independently.
        provenance["control_identity_settings"] = {
            "c": _metadata_num_str(spec.c, spec.dps + 5),
            "q_order": spec.q_order,
            "regularize": spec.regularize,
        }
    manifest = BootstrapManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        manifest_kind=MANIFEST_KIND_IDENTICAL_SCALAR,
        precision_dps=spec.dps,
        spec=_spec_to_json(spec),
        crossing=crossing,
        identity=identity,
        blocks=tuple(block_refs),
        provenance=provenance,
        pmp=pmp,
        scalar_G_spin0=convention,
    )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_json(), f, indent=2)
    return manifest_path


# ---------------------------------------------------------------------------
# Reader.
# ---------------------------------------------------------------------------


def load_identical_scalar_manifest(manifest_path) -> BootstrapManifest:
    """Parse a manifest, validating its schema/kind and every block hash.

    Each referenced block file is read and its :func:`block_content_hash` is
    compared against the stored ``content_hash``; a mismatch raises
    ``ValueError`` (a stale or regenerated block file is a hard error, because
    the manifest's reuse guarantees would otherwise be silently violated).
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        doc = json.load(f)
    manifest = BootstrapManifest.from_json(doc)

    if manifest.manifest_kind != MANIFEST_KIND_IDENTICAL_SCALAR:
        raise ValueError(
            f"unexpected manifest_kind {manifest.manifest_kind!r}; "
            f"expected {MANIFEST_KIND_IDENTICAL_SCALAR!r}"
        )
    major = manifest.schema_version.split(".", 1)[0]
    if major != MANIFEST_SCHEMA_VERSION.split(".", 1)[0]:
        raise ValueError(
            f"unsupported manifest schema_version {manifest.schema_version!r}; "
            f"this reader supports major version "
            f"{MANIFEST_SCHEMA_VERSION.split('.', 1)[0]}"
        )

    fixed_spins = set()
    variables = {"fixed_spin": "Delta", "large_spin": "L", "point": "point", "chiral": "Delta"}
    for ref in manifest.blocks:
        if ref.kind not in variables:
            raise ValueError(f"manifest block {ref.path!r}: unknown kind {ref.kind!r}")
        if ref.variable != variables[ref.kind]:
            raise ValueError(f"manifest block {ref.path!r}: variable {ref.variable!r} disagrees with kind {ref.kind!r}")
        if ref.kind in {"fixed_spin", "chiral"}:
            if ref.ell in fixed_spins:
                raise ValueError(f"manifest has duplicate fixed-spin block for ell = {ref.ell}")
            fixed_spins.add(ref.ell)
        if Path(ref.path).is_absolute():
            raise ValueError(f"manifest block path must be relative: {ref.path!r}")
        _read_manifest_block(manifest_path, manifest, ref)
    return manifest


def _read_manifest_block(manifest_path, manifest, ref):
    """Validate the document being consumed against the original manifest, on every read."""
    block_path = Path(manifest_path).parent / ref.path
    with open(block_path) as f:
        block_doc = json.load(f)
    legacy = _is_legacy_manifest(manifest.schema_version)
    if not block_content_hash_matches(block_doc, ref.content_hash, legacy=legacy):
        raise ValueError(
            f"content_hash mismatch for {ref.path!r}: "
            f"manifest has {ref.content_hash}, file hashes to {block_content_hash(block_doc)}"
        )
    if manifest.scalar_G_spin0 is None and SCALAR_G_SPIN0_KEY not in block_doc:
        # a manifest without the record vouches for its untagged blocks, in its own normalization
        block_doc = dict(block_doc, **{SCALAR_G_SPIN0_KEY: _manifest_scalar_G_spin0(manifest)})
    dps = _legacy_label_dps(manifest)
    if legacy:
        # The input labels are compared exactly with the (identically read) spec: read an
        # older writer's rounded spelling as the rational it rounded, as the spec's is.
        block_doc = dict(block_doc)
        for key in _BLOCK_INPUT_LABELS:
            value = block_doc.get(key)
            if isinstance(value, list):
                block_doc[key] = [_legacy_label_str(item, dps) for item in value]
            elif value is not None:
                block_doc[key] = _legacy_label_str(value, dps)
        if manifest.provenance.get("resolved_method") == "global_control" and block_doc.get("basis") == "scalar_G":
            # older writers labeled every identical-scalar block scalar_G, global-control ones included
            block_doc["basis"] = "global_scalar_G"
    _check_crossing_prefactor(manifest, block_doc, block_path)
    if block_doc.get("variable", "Delta") != ref.variable:
        raise ValueError(f"manifest block {ref.path!r}: variable disagrees with the block file")
    if ref.kind == "point":
        # Older writers recorded the full-precision numerical shift on the
        # reference and a shorter point_value label. Compare labels exactly
        # against those file representations, without ambient rounding.
        try:
            anchors = tuple(literal_number(block_doc[key]) for key in ("point_value", "variable_minus_x")
                            if block_doc.get(key) is not None)
            labels = tuple(literal_number(value) for value in (ref.point_value, ref.delta_min)
                           if value is not None)
            matches = bool(labels) and all(value in anchors for value in labels)
        except (TypeError, ValueError, ArithmeticError):
            matches = False
        if not matches:
            raise ValueError(f"manifest point block {ref.path!r}: point_value/delta_min disagrees with "
                             "the frozen block; rebuild it at the requested point")
    if ref.kind == "large_spin":
        try:
            twist = literal_number(block_doc["twist"])
            floor = literal_number(block_doc["large_spin_lmin"])
            matches = (_legacy_label(ref.twist, dps) == twist
                       and literal_number(ref.large_spin_lmin) == floor
                       and literal_number(ref.delta_min) == floor and ref.ell == floor)
        except (KeyError, TypeError, ValueError, ArithmeticError):
            matches = False
        if not matches:
            raise ValueError(f"manifest tower block {ref.path!r}: twist/large_spin_lmin/"
                             "delta_min/ell disagrees with the block file")
    if manifest.crossing.sector_kind == "chiral_F_minus":
        if block_doc.get("spin") is not None or ref.ell != 0:
            raise ValueError(f"manifest chiral block {ref.path!r} cannot carry an exchanged spin")
    elif _manifest_integer(block_doc.get("spin"), f"block {ref.path!r} spin") != ref.ell:
        raise ValueError(f"manifest block {ref.path!r}: ell = {ref.ell} disagrees with the block file spin")
    return block_doc


def _load_block_table(manifest_path: Path, manifest: BootstrapManifest, ref: BlockRef, *, spec=None, c=None,
                      h_ext=None, coordinate=None, expected_basis="scalar_G",
                      verify_keep_all=True) -> tuple[SerializedDerivTable, dict]:
    """Load ``ref`` and check that its block inputs match ``spec`` when supplied."""
    block_path = Path(manifest_path).parent / ref.path
    block_doc = _read_manifest_block(manifest_path, manifest, ref)
    if spec is not None:
        _check_block_doc(block_doc, block_path, spec, c=c, h_ext=h_ext, coordinate=coordinate,
                         manifest_block=True, expected_basis=expected_basis, verify_keep_all=verify_keep_all)
    return serialized_table_from_json(block_doc), block_doc


def build_identical_scalar_problem_from_manifest(
    manifest_path,
) -> IdenticalScalarGapProblem:
    """Rebuild an identical-scalar gap problem from a manifest + block files.

    The referenced block files are loaded (hash-checked) and fed to
    :func:`build_prefactored_pmp_doc` (library-normalized, self-dual base
    ``16 q(1/2)/N``) together with the manifest's identity vector and crossing
    pairs; the SDP data at the nodes are the files' numeric block values. A spin
    whose edge moved is rebuilt at the new edge, and a moment problem is rebuilt
    by :func:`~virasoro_bootstrap.moments.rebuild_moment_problem_from_blocks`.
    Otherwise the resulting
    :class:`~virasoro_bootstrap.identical_scalar.IdenticalScalarGapProblem`
    reproduces the PMP an in-memory rebuild would produce without regenerating
    any block.
    """
    manifest = load_identical_scalar_manifest(manifest_path)
    return _build_problem_from_manifest(manifest_path, manifest)


def _manifest_scalar_G_spin0(manifest) -> str:
    """The spin-0 scalar_G convention a manifest was built in."""
    from virasoro_blocks._conventions import SCALAR_G_SPIN0_DOUBLE

    return manifest.scalar_G_spin0 or SCALAR_G_SPIN0_DOUBLE


def _scale_free_manifest(manifest) -> bool:
    """Whether the manifest's result does not depend on the spin-0 scale of the scalar ``G`` basis.

    A chiral manifest holds no scalar-``G`` data.  A gap problem's verdict, and the certification of its functional,
    are positivity statements channel by channel, which a positive rescaling of the spin-0 blocks and the identity
    vector cannot change.  OPE, fixed-OPE, and moment problems compare the scalar channel with the identity, so
    their results do depend on it.
    """
    if manifest.provenance.get("resolved_method") in _CHIRAL_METHODS:
        return True
    return manifest.pmp is None or manifest.pmp.problem_kind == "gap"


def _check_manifest_scalar_G_spin0(manifest_path, manifest):
    """Refuse to rebuild a manifest whose spin-0 normalization is not this process's.

    Its identity vector, any stored vacuum measure or normalization operator, and its spin-0 blocks share one
    convention; a spin rebuilt here would use the process's, so the two cannot be mixed.
    """
    recorded, active = _manifest_scalar_G_spin0(manifest), scalar_G_spin0_convention()
    if recorded != active:
        origin = ("records no scalar_G_spin0 normalization" if manifest.scalar_G_spin0 is None
                  else f"was built with scalar_G_spin0 = {recorded!r}")
        kind = manifest.pmp.problem_kind if manifest.pmp is not None else "gap"
        raise ValueError(f"manifest {manifest_path} {origin}, but this process uses scalar_G_spin0 = {active!r}, "
                         f"and its {kind!r} result depends on the spin-0 normalization (only gap and chiral "
                         "manifests are rebuilt across conventions); regenerate the manifest and its block files "
                         "with the current virasoro_bootstrap")


def _build_problem_from_manifest(manifest_path, manifest):
    """Rebuild from an already hash-checked manifest, including a control base for moments.

    A gap or chiral manifest built in the other spin-0 convention is rebuilt in its own convention, so its blocks,
    identity vector, and any channel rebuilt here agree (its verdict does not depend on the scale; see
    :func:`_scale_free_manifest`); any other kind is refused.
    """
    from virasoro_blocks._conventions import scalar_G_spin0_scope

    recorded = _manifest_scalar_G_spin0(manifest)
    if recorded != scalar_G_spin0_convention() and _scale_free_manifest(manifest):
        with scalar_G_spin0_scope(recorded):
            return _build_problem_from_manifest_in_convention(manifest_path, manifest)
    return _build_problem_from_manifest_in_convention(manifest_path, manifest)


def _build_problem_from_manifest_in_convention(manifest_path, manifest):
    _check_manifest_scalar_G_spin0(manifest_path, manifest)
    control_method = manifest.provenance.get("resolved_method")
    chiral = control_method in _CHIRAL_METHODS
    if not chiral and control_method not in {"global_control", "numeric_virasoro_control"}:
        control_method = None
    block_basis = "global_scalar_G" if control_method == "global_control" else "scalar_G"
    if chiral:
        block_basis = "global_chiral_F" if control_method == "chiral_global_control" else "chiral_F"
    spec = _without_recorded_prefactor(
        _spec_from_json(manifest.spec, legacy=_is_legacy_manifest(manifest.schema_version)), manifest)
    if "used_regularize" in manifest.provenance:
        stored_regularize = manifest.provenance["used_regularize"]
        if not isinstance(stored_regularize, bool):
            raise ValueError("manifest used_regularize must be a boolean")
        if not control_method and _resolved_regularize(spec.c, spec.regularize) != stored_regularize:
            raise ValueError(
                f"manifest {manifest_path}: spec.regularize changes the recorded regularization policy; "
                "rebuild the problem to change how its frozen blocks and identity are evaluated"
            )
    if control_method and manifest.provenance.get("control_identity_settings") is not None:
        settings = manifest.provenance["control_identity_settings"]
        if (_legacy_label(settings["c"], _legacy_label_dps(manifest)) != literal_number(spec.c)
                or settings["q_order"] != spec.q_order or settings["regularize"] != spec.regularize):
            raise ValueError("control manifest identity settings disagree with the edited spec; rebuild the problem")
    manifest_path = Path(manifest_path)
    _refuse_unhonoured_spec_edits(manifest, spec, manifest_path)
    if int(spec.dps) != manifest.precision_dps:
        # the block files hold precision_dps-digit data and the SDP would be written and solved at the edited dps
        # around them: a higher dps would claim digits the files do not have
        raise ValueError(
            f"manifest {manifest_path}: spec.dps was edited (the block files were written at {manifest.precision_dps} "
            f"digits, the spec now says {spec.dps}); the files cannot honor the edit: rebuild the problem from the "
            "edited spec, or set the SDPB precision with sdpb_options={'--precision': bits}"
        )
    expected_pairs = tuple(F_minus_pairs(spec.lam, coordinate=spec.coordinate))
    if chiral:
        from .controls.chiral_control import chiral_F_minus_pairs
        expected_pairs = tuple(chiral_F_minus_pairs(spec.lam))
        if spec.coordinate != "zzb" or tuple(spec.spin_list) != (0,) or spec.large_spin_lmin is not None:
            raise ValueError("a chiral manifest requires the zzb adapter, spin_list=(0,), and no 2d spin tower")
    if (manifest.crossing.coordinate != spec.coordinate
            or manifest.crossing.sector_kind != ("chiral_F_minus" if chiral else "identical_scalar_F_minus")
            or manifest.crossing.crossing_pairs != expected_pairs):
        raise ValueError(
            f"manifest {manifest_path}: crossing sector, coordinate, and pairs must agree with the spec's "
            "coordinate and lambda"
        )
    if manifest.pmp is not None and manifest.pmp.problem_kind in _MOMENT_PROBLEM_KINDS:
        # a moment problem: its positivity blocks from the (hash-checked) files, its moment insertion rebuilt from
        # the recorded weights by the builder that made it
        from .moments import rebuild_moment_problem_from_blocks

        meta = (manifest.provenance.get("pmp_metadata") or {}).get("moment")
        if meta is None:
            raise ValueError(f"moment manifest {manifest_path} records no moment metadata")
        if "include_tower" not in meta:
            # manifests written before 2026-09-28: the tower was in the problem exactly when its files are here
            meta = dict(meta, include_tower=any(ref.kind == "large_spin" for ref in manifest.blocks))
        block_paths = [manifest_path.parent / ref.path for ref in manifest.blocks]
        refs_by_path = {path.resolve(): ref for path, ref in zip(block_paths, manifest.blocks)}

        def read_block(path):
            return _read_manifest_block(manifest_path, manifest, refs_by_path[Path(path).resolve()])

        base_problem = None
        if control_method:
            base_pmp = PmpVectors(
                problem_kind="gap", objective=("0",) * len(expected_pairs),
                normalization=manifest.identity.crossing_vector,
            )
            base_problem = _build_problem_from_manifest(
                manifest_path, _dataclass_replace(manifest, pmp=base_pmp))
        return rebuild_moment_problem_from_blocks(spec, meta, manifest.pmp.problem_kind, block_paths,
                                                  base_problem=base_problem, _block_reader=read_block)

    with mp.workdps(spec.dps):
        # Validate at the stored precision, since mp.mpf() rounds to the active context.
        spin_list, coordinate, c, h_ext, prefactor_exponent = (
            _validate_gap_spec_common(spec)
        )
        delta_min_by_ell = _resolved_delta_min_by_ell(spec, spin_list)
        used_regularize = _resolved_regularize(c, spec.regularize)
        if control_method and "used_regularize" in manifest.provenance:
            used_regularize = manifest.provenance["used_regularize"]
        resolved_method = control_method or _resolved_method(spec.npoles)
        from .point_constraint import _crossing_context_for_spec

        crossing_context = _crossing_context_for_spec(spec, resolved_method)
        resolved_positive_pole_npoles = _resolved_positive_pole_npoles(
            c, resolved_method, _normalize_npoles(spec.npoles), spec.positive_pole_npoles
        )
        F_pairs = expected_pairs

        fixed_refs_by_ell = {
            ref.ell: ref
            for ref in manifest.blocks
            if ref.kind == ("chiral" if chiral else "fixed_spin")
        }
        tower_refs = [ref for ref in manifest.blocks if ref.kind == "large_spin"]
        point_refs = [ref for ref in manifest.blocks if ref.kind == "point"]
        fixed_channels: list[IdenticalScalarSpinChannel] = []
        for ell in spin_list:
            if ell not in fixed_refs_by_ell:
                raise ValueError(
                    f"manifest is missing a block file for ell = {ell}"
                )
            ref = fixed_refs_by_ell[ell]
            serialized, doc = _load_block_table(manifest_path, manifest, ref, spec=spec, c=c, h_ext=h_ext,
                                                coordinate="z" if chiral else coordinate,
                                                expected_basis=block_basis,
                                                verify_keep_all=control_method not in {
                                                    "global_control", "numeric_virasoro_control",
                                                    "chiral_global_control", "chiral_virasoro_control",
                                                })
            if chiral:
                from .chiral_from_blocks import _relabel_chiral_as_zzb
                serialized = _relabel_chiral_as_zzb(serialized)
            # an edited target_digits is honored by the files' own verified digits, or refused
            _check_block_target(doc, manifest_path.parent / ref.path, spec)
            fixed_channels.append(IdenticalScalarSpinChannel(
                ell=ell, delta_min=delta_min_by_ell[ell], serialized=serialized, prefactored=serialized,
                crossing_context=crossing_context,
            ))
        # an edited gap: a spin whose edge is no longer its file's anchor is built fresh at the new edge (numeric
        # samples at its own nodes), as an in-memory build re-interpolates it
        moved = [
            channel.ell for channel in fixed_channels
            if literal_number(channel.serialized.shift) != channel.delta_min
        ]
        if moved and chiral:
            from .controls.chiral_control import build_chiral_channel, build_chiral_symbolic_channel

            for index, channel in enumerate(fixed_channels):
                if channel.ell not in moved:
                    continue
                kwargs = dict(h_gap=channel.delta_min, h_ext=h_ext, lam=spec.lam, c=c,
                              q_order=spec.q_order, chiral_norm=spec.chiral_norm)
                if control_method in {"zbar_half_chiral_from_cli", "chiral_virasoro_symbolic_control"}:
                    fixed_channels[index] = build_chiral_symbolic_channel(
                        **kwargs, npoles=_normalize_npoles(spec.npoles),
                        n_sample_points=spec.n_sample_points, pole_selection=spec.pole_selection,
                        regularize=spec.regularize, shared_sample_points=spec.shared_sample_points)
                else:
                    fixed_channels[index] = build_chiral_channel(
                        **kwargs, mode="global" if control_method == "chiral_global_control" else "virasoro",
                        n_sample_points=len(channel.serialized.sample_points_x),
                        sample_x=channel.serialized.sample_points_x)
        elif moved and control_method:
            from .controls.global_control import build_global_scalar_spin_channel, build_numeric_virasoro_scalar_spin_channel

            for index, channel in enumerate(fixed_channels):
                if channel.ell not in moved:
                    continue
                kwargs = dict(coordinate=coordinate, ell=channel.ell, delta_min=channel.delta_min,
                              h_ext=h_ext, lam=spec.lam, chiral_norm=spec.chiral_norm)
                if control_method == "global_control":
                    fixed_channels[index] = build_global_scalar_spin_channel(
                        **kwargs, n_sample_points=len(channel.serialized.pole_list_x))
                else:
                    fixed_channels[index] = build_numeric_virasoro_scalar_spin_channel(
                        **kwargs, c=c, q_order=spec.q_order,
                        sample_x=channel.serialized.sample_points_x)
        elif moved:
            rebuilt = _spin_channels_at_edges(
                spec,
                _file_backed_exact_tables(spec, moved, c=c, h_ext=h_ext, coordinate=coordinate,
                                          used_regularize=used_regularize, resolved_method=resolved_method),
                moved, delta_min_by_ell, coordinate=coordinate, resolved_method=resolved_method,
            )
            fixed_channels = [rebuilt.get(channel.ell, channel) for channel in fixed_channels]

        # Gap-independent large-spin tower channels: rebuilt from their
        # variable="L" block files in stored order (matching the in-memory
        # builder, which appends the tower after the fixed-spin channels). Each
        # tower channel is labeled by its Lmax floor (delta_min on the ref).
        tower_channels: list[IdenticalScalarSpinChannel] = []
        stored_twists = tuple(getattr(spec, "large_spin_twists", ()) or ())
        if len(tower_refs) != len(stored_twists):
            raise ValueError(f"manifest {manifest_path}: {len(tower_refs)} tower block files for the spec's "
                             f"{len(stored_twists)} large_spin_twists (an edited spec? rebuild the problem)")
        for tower_index, ref in enumerate(tower_refs):
            serialized, doc = _load_block_table(manifest_path, manifest, ref, spec=spec, c=c, h_ext=h_ext,
                                                coordinate=coordinate)
            file_twist = _tower_block_twist(doc, ref.path, lmin=spec.large_spin_lmin)
            if file_twist != literal_number(stored_twists[tower_index]):
                raise ValueError(f"tower block {ref.path} has twist {doc['twist']}, but the spec's twist "
                                 f"{tower_index} is {stored_twists[tower_index]} (an edited spec? rebuild the problem)")
            prefactored = serialized
            tower_channels.append(
                IdenticalScalarSpinChannel(
                    ell=ref.ell,
                    delta_min=_parse_serialized_number(ref.delta_min),
                    serialized=serialized,
                    prefactored=prefactored,
                    # Retain the twist so the dimension weight is Delta = L + twist.
                    twist=(stored_twists[tower_index] if tower_index < len(stored_twists) else None),
                )
            )

        # Gap-independent point-constraint channels: degree-0 blocks frozen at
        # delta_point (= point_value on the ref). The block file already holds
        # the constant table, so the reload just re-wraps it -- no recompute.
        point_channels: list[IdenticalScalarSpinChannel] = []
        from .point_constraint import _is_point_channel

        for ref in point_refs:
            serialized, _doc = _load_block_table(
                manifest_path, manifest, ref, spec=spec, c=c, h_ext=h_ext,
                coordinate="z" if chiral else coordinate, expected_basis=block_basis,
            )
            if chiral:
                from .chiral_from_blocks import _relabel_chiral_as_zzb
                serialized = _relabel_chiral_as_zzb(serialized)
            prefactored = serialized
            channel = IdenticalScalarSpinChannel(
                ell=ref.ell,
                delta_min=_legacy_label(
                    ref.point_value
                    if ref.point_value is not None else ref.delta_min,
                    _legacy_label_dps(manifest),
                ),
                serialized=serialized,
                prefactored=prefactored,
                point_context=crossing_context,
            )
            if not _is_point_channel(channel):
                raise ValueError(f"manifest point block {ref.path!r} must be pole-free and degree zero")
            _validate_identical_scalar_operator(channel.ell, channel.delta_min, name="manifest point constraint")
            point_channels.append(channel)

        # every spec point constraint has its block (appended positivity channels may add more)
        missing = [(ell, delta) for ell, delta in (spec.point_constraints or ())
                   if not any(channel.ell == ell and literal_number(channel.delta_min) == literal_number(delta)
                              for channel in point_channels)]
        if missing:
            raise ValueError(f"manifest {manifest_path}: no block file for the spec's point constraints {missing} "
                             "(an edited spec? rebuild the problem)")
        spin_channels = fixed_channels + tower_channels + point_channels
        identity_vector = tuple(
            _parse_serialized_number(v) for v in manifest.identity.crossing_vector
        )
        # schema 1.1 stores the SDP objective/normalization explicitly; schema 1.0
        # (no "pmp") is always the gap convention (objective 0, normalization V_id).
        problem_kind = "gap"
        normalization_operator = None
        if manifest.pmp is not None:
            problem_kind = manifest.pmp.problem_kind
            objective = [_parse_serialized_number(v) for v in manifest.pmp.objective]
            normalization = [
                _parse_serialized_number(v) for v in manifest.pmp.normalization
            ]
            if manifest.pmp.normalization_operator is not None:
                normalization_operator = {
                    "spin_star": _manifest_integer(
                        manifest.pmp.normalization_operator["spin_star"], "normalization operator spin_star"
                    ),
                    "delta_star": literal_number(
                        manifest.pmp.normalization_operator["delta_star"]
                    ),
                }
                if "direction" in manifest.pmp.normalization_operator:
                    normalization_operator["direction"] = manifest.pmp.normalization_operator["direction"]
                if "ope_squared" in manifest.pmp.normalization_operator:
                    normalization_operator["ope_squared"] = literal_number(
                        manifest.pmp.normalization_operator["ope_squared"]
                    )
        else:
            objective = [mp.mpf(0)] * len(F_pairs)
            normalization = list(identity_vector)
        prefactored_blocks = [channel.prefactored for channel in spin_channels]
        pmp_build = build_prefactored_pmp_doc(
            prefactored_blocks,
            F_pairs,
            coordinate=coordinate,
            objective=objective,
            normalization=normalization,
            positive_pole_npoles=resolved_positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            explicit_sample_points=spec.explicit_sample_points,
        )
        # the SDP data at the nodes: the files' numeric block values (never the file polynomial re-evaluated)
        sample_values_by_block = _clean_sample_values_by_block(
            spin_channels, pmp_build.doc, F_pairs, coordinate=coordinate,
            lowerings=pmp_build.positive_pole_lowerings,
            sampling_blocks=pmp_build.positivity_blocks,
            sampler_for=None if control_method else _fixed_spin_sampler_for(
                spec, spin_list, {}, c=c, h_ext=h_ext, coordinate=coordinate,
                used_regularize=used_regularize, resolved_method=resolved_method,
            ),
        )

        rebuilt_spec = IdenticalScalarGapSpec(
            c=c,
            h_ext=spec.h_ext,
            delta_gap=delta_min_by_ell[0],
            coordinate=coordinate,
            delta_min_by_spin=_canonical_non_scalar_delta_min_overrides(
                delta_min_by_ell
            ),
            lam=spec.lam,
            spin_list=spin_list,
            q_order=spec.q_order,
            dps=spec.dps,
            target_digits=spec.target_digits,
            max_dps=spec.max_dps,
            npoles=spec.npoles,
            n_sample_points=spec.n_sample_points,
            pole_selection=spec.pole_selection,
            positive_pole_npoles=resolved_positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            regularize=spec.regularize,
            prefactor_exponent=prefactor_exponent,
            chiral_norm=spec.chiral_norm,
            explicit_sample_points=spec.explicit_sample_points,
            num_threads=spec.num_threads,
            shared_sample_points=spec.shared_sample_points,
            large_spin_lmin=spec.large_spin_lmin,
            large_spin_twists=spec.large_spin_twists,
            large_spin_pole_selection=spec.large_spin_pole_selection,
            large_spin_npoles=spec.large_spin_npoles,
            sample_via=spec.sample_via,
            point_constraints=spec.point_constraints,
        )

        return IdenticalScalarGapProblem(
            spec=rebuilt_spec,
            resolved_method=resolved_method,
            used_regularize=used_regularize,
            prefactor_exponent=prefactor_exponent,
            F_pairs=F_pairs,
            identity_vector=identity_vector,
            spin_channels=tuple(spin_channels),
            pmp_build=pmp_build,
            large_spin_channels=tuple(tower_channels),
            point_channels=tuple(point_channels),
            sample_values_by_block=sample_values_by_block,
            problem_kind=problem_kind,
            objective_vector=tuple(objective),
            normalization_vector=tuple(normalization),
            normalization_operator=normalization_operator,
            pmp_metadata=manifest.provenance.get("pmp_metadata"),
            scalar_G_spin0=_manifest_scalar_G_spin0(manifest),
        )
