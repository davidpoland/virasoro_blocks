"""Discard generated SDP matrices while retaining exact certification samples.

Cleanup belongs after the entire task (including certification) has finished,
never between probes: interrupted and failed tasks retain their solve inputs.
"""
from __future__ import annotations

from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

SAMPLES_NAME = "certification_samples.json.gz"
RECORD_NAME = "input_cleanup.json"
SDP_SOURCE_NAME = "vb_sdp_source.json"
_OWNED_FILE = re.compile(r"block_(data|info)_(0|[1-9][0-9]*)\.json")
_BASIS_METADATA = ("control.json", "normalization.json")
_SOLVER_RESULT_FILE = re.compile(r"(?:out|y|x_[0-9]+|[XY]_matrix_[0-9]+)\.txt")


def record_sdp_source(sdp_dir, output_dir, *, state):
    """Bind an output to its basis; return the journal for later state updates.

    Starting a ``preparing`` or ``running`` attempt removes previous solution and objective files.
    SDPB only overwrites the solutions selected by ``--writeSolution``, so an omitted output must not
    survive from an earlier solve.
    The incomplete record is published before cleanup; a cleanup failure must prevent solver launch.
    """
    from .journal import Journal

    source, output = Path(sdp_dir).resolve(), Path(output_dir).resolve()
    if (output / SDP_SOURCE_NAME).is_symlink():
        raise ValueError(f"refusing symlinked SDP source record: {output / SDP_SOURCE_NAME}")
    if any((source / name).is_symlink() for name in _BASIS_METADATA):
        raise ValueError(f"refusing symlinked SDP metadata: {source}")
    metadata = {name: _sha256(source / name) for name in _BASIS_METADATA
                if (source / name).is_file()}
    try:
        label = os.path.relpath(source, output)
    except ValueError:  # Different Windows drives cannot have a relative path.
        label = str(source)
    journal = Journal(output / SDP_SOURCE_NAME)
    journal.update(
        schema_version=1, state=state,
        sdp_directory=label, metadata_sha256=metadata,
    )
    if state in {"preparing", "running"}:
        for path in output.iterdir():
            if _SOLVER_RESULT_FILE.fullmatch(path.name):
                # Unlink a solution symlink itself, never its target. Unexpected directories raise and leave
                # the attempt incomplete. Checkpoints, diagnostic logs, and unrelated files are retained.
                path.unlink()
    return journal


def recorded_sdp_state(output_dir):
    """The recorded preparation or solver state, or None for older outputs.

    States are ``preparing``, ``prepared``, ``running``, ``completed``, and ``failed``.
    Unlike :func:`recorded_sdp_source` this does not verify the input basis; it
    tells a caller whether the solve finished before it reads any output.
    """
    path = Path(output_dir) / SDP_SOURCE_NAME
    if path.is_symlink():
        raise ValueError(f"refusing symlinked SDP source record: {path}")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("state"), str):
        raise ValueError(f"invalid SDP source record: {path}")
    return data["state"]


def recorded_sdp_source(output_dir):
    """Return a verified input-directory reference, or None for older outputs."""
    output = Path(output_dir)
    path = output / SDP_SOURCE_NAME
    if path.is_symlink():
        raise ValueError(f"refusing symlinked SDP source record: {path}")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or not isinstance(data.get("sdp_directory"), str)
            or not data["sdp_directory"]
            or not isinstance(data.get("metadata_sha256"), dict)):
        raise ValueError(f"invalid SDP source record: {path}")
    if data.get("state") != "completed":
        raise ValueError(f"solver output has no completed attempt: {path}")
    source = Path(data["sdp_directory"])
    source = source if source.is_absolute() else output / source
    present = {name for name in _BASIS_METADATA if (source / name).exists()}
    if not present or present != set(data["metadata_sha256"]):
        raise ValueError(f"recorded SDP metadata is missing or changed: {source}")
    for name, expected in data["metadata_sha256"].items():
        if name not in _BASIS_METADATA or not isinstance(expected, str):
            raise ValueError(f"invalid SDP source record: {path}")
        metadata_path = source / name
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError(f"recorded SDP metadata is missing or changed: {metadata_path}")
        _verify_metadata_hash(str(metadata_path.resolve()), _signature(metadata_path), expected,
                              origin="solver launch")
    return source


def keep_solver_inputs(value=""):
    """Parse the environment/profile flag without guessing on invalid values."""
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    raise ValueError("VB_KEEP_SOLVER_INPUTS must be 0/1 or false/true (no/yes, off/on)")


def _signature(path):
    info = path.stat(follow_symlinks=False)
    return info.st_ino, info.st_size, info.st_mtime_ns


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _verify_metadata_hash(path, signature, expected_hash, *, origin="cleanup"):
    if _sha256(Path(path)) != expected_hash:
        raise ValueError(f"SDP metadata changed since {origin}: {path}")


def _read_cleanup_record(directory):
    """Check the saved sample basis before pruning, reconstruction or reuse."""
    record_path = directory / RECORD_NAME
    if record_path.is_symlink():
        raise ValueError(f"refusing symlinked cleanup record: {record_path}")
    record = json.loads(record_path.read_text())
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or not isinstance(record.get("source_sha256"), dict)):
        raise ValueError(f"incompatible input-cleanup record: {record_path}")
    for name in _BASIS_METADATA:
        # Existing schema-1 bundles did not record metadata hashes. Keep
        # those valid records readable; new records bind the reduced basis.
        if name in record["source_sha256"]:
            expected = record["source_sha256"][name]
            path = directory / name
            if not isinstance(expected, str) or path.is_symlink() or not path.is_file():
                raise ValueError(f"SDP metadata changed since cleanup: {path}")
            _verify_metadata_hash(str(path.resolve()), _signature(path), expected)
    return record


@lru_cache(maxsize=2)
def _read_samples(path, signature, expected_hash):
    path = Path(path)
    if _sha256(path) != expected_hash:
        raise ValueError(f"certification sample checksum mismatch: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        doc = json.load(stream)
    if doc.get("schema_version") != 1 or not isinstance(doc.get("blocks"), dict):
        raise ValueError(f"invalid certification samples: {path}")
    return doc["blocks"]


def load_compacted_sdp_block(sdp_dir, block_index):
    """Read unchanged c/B strings, or None when there is no compacted input."""
    directory = Path(sdp_dir)
    record_path = directory / RECORD_NAME
    if not record_path.exists():
        return None
    record = _read_cleanup_record(directory)
    samples = directory / SAMPLES_NAME
    blocks = _read_samples(str(samples.resolve()), _signature(samples), record["samples_sha256"])
    return blocks.get(str(block_index))


def cleanup_sdp_inputs(sdp_dir):
    """Remove only this writer's matrices/metadata, after a verified compact copy.

    Keeps normalization, objectives, control, PMP documents, block tables,
    solver outputs and checkpoints. Unknown backends and symlinks are refused.
    The small c/B bundle retains the exact reduced-basis certification data;
    bilinear bases and block_info can be regenerated without block recursion.
    """
    import fcntl

    directory = Path(sdp_dir)
    if directory.is_symlink():
        raise ValueError(f"refusing symlinked SDP directory: {directory}")
    control_path = directory / "control.json"
    if control_path.is_symlink():
        raise ValueError(f"refusing symlinked SDP metadata: {directory}")
    # Lock the existing control file: no extra persistent file per solve, and
    # two completed-task workers cannot publish competing recovery records.
    # NFS implements exclusive flock as a write lock, requiring a writable
    # descriptor even though we leave the control document unchanged.
    with control_path.open("r+b") as control_stream:
        try:
            fcntl.flock(control_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"removed_files": 0, "removed_bytes": 0, "retained_bytes": 0}
        try:
            return _cleanup_sdp_inputs_locked(directory)
        finally:
            fcntl.flock(control_stream.fileno(), fcntl.LOCK_UN)


def _cleanup_sdp_inputs_locked(directory):
    control_path = directory / "control.json"
    normalization_path = directory / "normalization.json"
    if control_path.is_symlink() or normalization_path.is_symlink():
        raise ValueError(f"refusing symlinked SDP metadata: {directory}")
    control = json.loads(control_path.read_text())
    if control.get("command") != "virasoro_bootstrap.sdp_writer":
        return {"removed_files": 0, "removed_bytes": 0, "retained_bytes": 0}
    num_blocks = control["num_blocks"]
    if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks < 0:
        raise ValueError(f"invalid block count: {control_path}")
    owned = [p for p in sorted(directory.iterdir()) if _OWNED_FILE.fullmatch(p.name)]
    if not owned:
        return {"removed_files": 0, "removed_bytes": 0, "retained_bytes": 0}
    if any(p.is_symlink() or not p.is_file() for p in owned):
        raise ValueError(f"refusing nonregular SDP input: {directory}")
    signatures = {p.name: _signature(p) for p in owned}
    source_metadata = {p.name: _signature(p) for p in (control_path, normalization_path)}
    pmp_path = directory.parent / "pmp.json"
    pmp_signature = _signature(pmp_path) if pmp_path.is_file() and not pmp_path.is_symlink() else None
    pmp_hash = _sha256(pmp_path) if pmp_signature is not None else None
    record_path = directory / RECORD_NAME
    samples_path = directory / SAMPLES_NAME
    if record_path.is_symlink() or samples_path.is_symlink():
        raise ValueError(f"refusing symlinked cleanup record: {directory}")
    # Resume pruning after a kill without replacing the already verified bundle.
    if record_path.exists():
        record = _read_cleanup_record(directory)
        if record.get("schema_version") != 1 or record.get("num_blocks") != num_blocks:
            raise ValueError(f"incompatible input-cleanup record: {record_path}")
        if "pmp_sha256" in record and record["pmp_sha256"] != pmp_hash:
            raise ValueError(f"retained PMP changed since cleanup: {pmp_path}")
        blocks = _read_samples(str(samples_path.resolve()), _signature(samples_path), record["samples_sha256"])
        expected = record["source_sha256"]
        if any(p.name not in expected or _sha256(p) != expected[p.name] for p in owned):
            raise ValueError(f"SDP inputs changed since cleanup: {directory}")
    else:
        blocks = {}
        expected = {}
        data_files = {int(_OWNED_FILE.fullmatch(p.name)[2]): p for p in owned
                      if _OWNED_FILE.fullmatch(p.name)[1] == "data"}
        if set(data_files) != set(range(num_blocks)):
            raise ValueError(f"incomplete SDP inputs: {directory}")
        normalization = json.loads(normalization_path.read_text())["normalization"]
        for index, path in data_files.items():
            payload = path.read_bytes()
            data = json.loads(payload)
            expected[path.name] = hashlib.sha256(payload).hexdigest()
            c, rows = data["c"], data["B"]
            if (not isinstance(c, list) or not c or not isinstance(rows, list)
                    or len(c) != len(rows) or any(not isinstance(row, list)
                    or len(row) != len(normalization) - 1 for row in rows)):
                raise ValueError(f"invalid certification samples: {path}")
            blocks[str(index)] = {"c": c, "B": rows}
        expected.update({p.name: _sha256(p) for p in owned if p.name not in expected})
        expected.update({name: _sha256(directory / name) for name in source_metadata})
        with tempfile.NamedTemporaryFile(prefix=".samples-", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
                # Encode each block in C rather than issuing one gzip write per
                # decimal coefficient. Large solves have millions of them.
                stream.write('{"schema_version":1,"blocks":{')
                for index, (name, values) in enumerate(blocks.items()):
                    if index:
                        stream.write(",")
                    stream.write(json.dumps(name) + ":" + json.dumps(values, separators=(",", ":")))
                stream.write("}}")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            samples_hash = _sha256(temporary)
            if _read_samples(str(temporary), _signature(temporary), samples_hash) != blocks:
                raise ValueError(f"certification sample verification failed: {directory}")
            # Verify the entire source set before deleting any file.
            if any(_signature(directory / name) != sig for name, sig in signatures.items()):
                raise ValueError(f"SDP inputs changed during cleanup: {directory}")
            if any(_signature(directory / name) != sig for name, sig in source_metadata.items()):
                raise ValueError(f"SDP metadata changed during cleanup: {directory}")
            if pmp_signature is not None and _signature(pmp_path) != pmp_signature:
                raise ValueError(f"retained PMP changed during cleanup: {pmp_path}")
            temporary.replace(samples_path)
            record = {"schema_version": 1, "samples_sha256": samples_hash,
                      "source_sha256": expected, "num_blocks": num_blocks}
            if pmp_hash is not None:
                record["pmp_sha256"] = pmp_hash
            with tempfile.NamedTemporaryFile(mode="w", prefix=".cleanup-", dir=directory, delete=False) as stream:
                json.dump(record, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
                record_temporary = Path(stream.name)
            record_temporary.replace(record_path)
        finally:
            temporary.unlink(missing_ok=True)
    if set(blocks) != {str(i) for i in range(num_blocks)}:
        raise ValueError(f"incomplete compacted certification samples: {directory}")
    if any(_signature(directory / name) != sig for name, sig in source_metadata.items()):
        raise ValueError(f"SDP metadata changed before removal: {directory}")
    if pmp_signature is not None and _signature(pmp_path) != pmp_signature:
        raise ValueError(f"retained PMP changed before removal: {pmp_path}")
    removed_bytes = 0
    for path in owned:
        if _signature(path) != signatures[path.name]:
            raise ValueError(f"SDP input changed before removal: {path}")
        removed_bytes += signatures[path.name][1]
        path.unlink()
    return {"removed_files": len(owned), "removed_bytes": removed_bytes,
            "retained_bytes": samples_path.stat().st_size + record_path.stat().st_size}


def _has_finished_solver_output(sdp_dir):
    directory = Path(sdp_dir)
    solve = directory.parent
    pmp_path = solve / "pmp.json"
    output_path = solve / "sdpb_out" / "out.txt"
    return (pmp_path.is_file() and not pmp_path.is_symlink() and output_path.is_file()
            and output_path.stat().st_mtime_ns >= (directory / "control.json").stat().st_mtime_ns)


def cleanup_task_solver_inputs(task_dir):
    """Clean completed task SDPs, without walking blocks or checkpoint trees."""
    totals = {"removed_files": 0, "removed_bytes": 0, "retained_bytes": 0}
    for base, dirs, files in os.walk(Path(task_dir).resolve(), followlinks=False):
        dirs[:] = [name for name in dirs if name not in {"blocks", "sdpb_out"}
                   and not name.endswith(".ck") and not (Path(base) / name).is_symlink()]
        if "control.json" in files and "normalization.json" in files:
            # A preparation-only task requested these inputs as its output.
            # Completed solves retain their PMP as the regeneration source.
            if not _has_finished_solver_output(base):
                dirs.clear()
                continue
            stats = cleanup_sdp_inputs(base)
            for name, value in stats.items():
                totals[name] += value
            dirs.clear()
    return totals


def regenerate_sdp_inputs(solve_dir, *, eval_dps=None, output_digits=None):
    """Regenerate bilinear bases from retained PMP nodes, preserving exact c/B.

    No block recursion, interpolation, or solver call is required. Precision
    defaults to the retained normalization metadata (or the older writer log),
    rather than the ambient dps.
    """
    import fcntl

    solve = Path(solve_dir)
    if (solve / RECORD_NAME).is_file():
        sdp, solve = solve, solve.parent
    else:
        candidates = [path for path in solve.iterdir() if path.is_dir()
                      and not path.is_symlink() and (path / RECORD_NAME).is_file()]
        if len(candidates) != 1:
            raise ValueError("expected one cleaned SDP directory; pass its path explicitly")
        sdp = candidates[0]
    if sdp.is_symlink():
        raise ValueError(f"refusing symlinked SDP directory: {sdp}")
    with (sdp / "control.json").open("r+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"SDP cleanup or restoration already in progress: {sdp}") from exc
        try:
            return _regenerate_sdp_inputs_locked(solve, sdp, eval_dps=eval_dps,
                                                 output_digits=output_digits)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _regenerate_sdp_inputs_locked(solve, sdp, *, eval_dps=None, output_digits=None):
    import mpmath as mp
    from ..sdp_writer import _normalization_basis, write_sdp_directory
    from virasoro_blocks.requests import _integer_field

    record = _read_cleanup_record(sdp)
    pmp_path = solve / "pmp.json"
    if record.get("pmp_sha256") != _sha256(pmp_path):
        raise ValueError(f"retained PMP changed or has no recorded checksum: {pmp_path}")
    log = (solve / "pmp2sdp.log").read_text() if (solve / "pmp2sdp.log").exists() else ""
    original_match = re.search(r"eval_dps=(\d+)", log)
    norm_data = json.loads((sdp / "normalization.json").read_text())
    original_dps = norm_data.get("eval_dps")
    if original_dps is None and original_match is not None:
        original_dps = int(original_match[1])
    if original_dps is not None:
        original_dps = _integer_field("normalization eval_dps", original_dps)
        if original_dps < 1:
            raise ValueError("normalization eval_dps must be positive")
    if eval_dps is None:
        if original_dps is None:
            raise ValueError("original eval_dps is unknown; supply it explicitly")
        eval_dps = original_dps
    eval_dps = _integer_field("eval_dps", eval_dps)
    if eval_dps < 1:
        raise ValueError("eval_dps must be >= 1")
    if output_digits is None:
        # A reused run can retain a conversion log from another build. Its
        # printed-digit setting belongs to this basis only if precision agrees.
        matches_original = original_match is not None and int(original_match[1]) == original_dps
        match = re.search(r"output_digits=(\d+)", log) if matches_original else None
        output_digits = int(match[1]) if match else eval_dps
    output_digits = _integer_field("output_digits", output_digits)
    if output_digits < 1:
        raise ValueError("output_digits must be >= 1")
    doc = json.loads(pmp_path.read_text())
    nm = doc["normalization"]
    if norm_data["normalization"] != nm:
        raise ValueError("retained SDP and PMP normalizations disagree")
    data = {i: load_compacted_sdp_block(sdp, i)
            for i in range(len(doc["PositiveMatrixWithPrefactorArray"]))}
    if any(value is None for value in data.values()):
        raise ValueError("compacted certification samples are incomplete")
    # Recover sample values in the full PMP basis. Preserve the originally
    # rounded reduced-basis c/B strings exactly when publishing the inputs.
    samples = {}
    norm, pivot = _normalization_basis(norm_data, eval_dps=original_dps if original_dps is not None else eval_dps)
    with mp.workdps(eval_dps + 20):
        others = [i for i in range(len(norm)) if i != pivot]
        for i, block in enumerate(doc["PositiveMatrixWithPrefactorArray"]):
            values = []
            if len(data[i]["c"]) != len(block["sampleScalings"]):
                raise ValueError("retained PMP and certification sample counts disagree")
            for c, row, scaling in zip(data[i]["c"], data[i]["B"], block["sampleScalings"]):
                c, scaling = mp.mpf(c), mp.mpf(scaling)
                vector = [mp.mpf(0)] * len(norm)
                vector[pivot] = c * norm[pivot] / scaling
                for index, b in zip(others, row):
                    vector[index] = (c * norm[index] - mp.mpf(b)) / scaling
                values.append(vector)
            samples[i] = values
    with tempfile.TemporaryDirectory(prefix=".restore-sdp-", dir=solve) as temporary:
        stage = Path(temporary) / "sdp"
        write_sdp_directory(doc, stage, eval_dps=eval_dps,
                            output_digits=output_digits, sample_values_by_block=samples)
        for i, values in data.items():
            path = stage / f"block_data_{i}.json"
            block = json.loads(path.read_text())
            block.update(values)
            path.write_text(json.dumps(block))
        for path in stage.iterdir():
            # The original normalization/objective/control remain authoritative.
            if _OWNED_FILE.fullmatch(path.name):
                path.replace(sdp / path.name)
    (sdp / RECORD_NAME).unlink()
    (sdp / SAMPLES_NAME).unlink()
    return sdp
