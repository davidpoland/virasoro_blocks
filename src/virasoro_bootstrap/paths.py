"""Repository-local paths for reproducible research artifacts.

The library code should not know about the repository layout.  Research
drivers use this module to keep generated state under one ignored
``artifacts/`` root instead of scattering it across ``runs/``, ``results/``,
``cert_checks/``, ``plots/``, and ``test_output/``.

Environment variables:

``VB_ARTIFACT_ROOT``
    Root for new generated state.  Defaults to ``artifacts``.
``VB_EXPERIMENT_ROOT``
    Override for dated experiment output.  ``VB_TEST_OUTPUT_ROOT`` remains a
    legacy alias for reading older local and cluster workflows.
``VB_RUNS_ROOT`` / ``VB_RESULTS_ROOT`` / ``VB_CERTIFICATES_ROOT``
    Optional explicit overrides for the corresponding artifact categories.
``VB_RESEARCH_ROOT`` / ``VB_SCRATCH_ROOT``
    Optional roots for source-controlled research inputs and disposable local
    scratch work.
``VB_DEFAULT_PARAMS``
    Optional path of a ``default_params.json`` that replaces the copy shipped in
    the package (the production profiles and selector defaults).  It is read at
    import and must exist.  Without it only the shipped copy is read.

The public path contract is deliberately small.  Callers should use
``dated_output_dir`` or one of the category helpers rather than spelling out
repository directories themselves.
"""
from __future__ import annotations

import os
from importlib import resources
from functools import lru_cache
from datetime import date
from pathlib import Path


DEFAULT_ARTIFACT_ROOT = "artifacts"
DEFAULT_EXPERIMENTS_DIR = "experiments"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_CERTIFICATES_DIR = "certificates"
DEFAULT_FIGURES_DIR = "figures"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_RESEARCH_ROOT = "research"
DEFAULT_SCRATCH_ROOT = ".local/scratch"

# Compatibility names used by older scripts.  They now point at the experiment
# root rather than reintroducing a top-level ``test_output`` directory.
DEFAULT_TEST_OUTPUT_DIR = DEFAULT_EXPERIMENTS_DIR
DEFAULT_ROOT = f"{DEFAULT_ARTIFACT_ROOT}/{DEFAULT_EXPERIMENTS_DIR}"
LEGACY_TEST_OUTPUT_ROOT = "test_output"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


@lru_cache(maxsize=1)
def _source_checkout_root() -> Path | None:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        source_module = parent / "src" / "virasoro_bootstrap" / "paths.py"
        if (parent / "pyproject.toml").is_file() and source_module.resolve() == module_path:
            return parent
    return None


def repository_root() -> Path:
    """Return the checkout root when this package is running from source.

    Both the project marker and this module's source location identify a
    checkout. An installed wheel may sit inside an unrelated project's virtual
    environment; that enclosing project must not become its checkout root.
    Outside a checkout, use the current process directory for explicitly
    requested research paths. Only source discovery is cached.
    """
    return _source_checkout_root() or Path.cwd()


repository_root.cache_clear = _source_checkout_root.cache_clear
repository_root.cache_info = _source_checkout_root.cache_info


def artifact_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Return the root for generated repository state."""
    if root is not None:
        return Path(root).expanduser()
    return _env_path("VB_ARTIFACT_ROOT") or Path(DEFAULT_ARTIFACT_ROOT)


def artifact_dir(
    category: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return an artifact category directory without creating it."""
    category = str(category).strip("/\\")
    if not category:
        raise ValueError("artifact category must not be empty")
    return artifact_root(root) / category


def experiment_root(
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the dated experiment-output subtree.

    An explicit ``root`` is already the complete experiment root.
    ``VB_TEST_OUTPUT_ROOT`` remains a compatibility fallback; new callers
    should omit ``root`` and use ``artifacts/experiments``.
    """
    if root is not None:
        return Path(root).expanduser()
    return (
        _env_path("VB_EXPERIMENT_ROOT")
        or _env_path("VB_TEST_OUTPUT_ROOT")
        or artifact_dir(DEFAULT_EXPERIMENTS_DIR)
    )


def test_output_root(
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Compatibility alias for :func:`experiment_root`."""
    return experiment_root(root=root)


def repository_output_roots(
    base: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Return new and legacy experiment roots for discovery readers.

    The new root is first.  The legacy root is retained only for reading old
    dated results; writers should use :func:`dated_output_dir` instead.
    ``base`` is useful to scripts that discover artifacts relative to a
    checkout rather than the process working directory.
    """
    base_path = Path.cwd() if base is None else Path(base)
    configured = experiment_root()
    current = configured if configured.is_absolute() else base_path / configured
    legacy = base_path / LEGACY_TEST_OUTPUT_ROOT
    roots = []
    for candidate in (current, legacy):
        candidate = candidate.resolve() if candidate.is_absolute() else candidate
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def runs_root(
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the local run-manifest root."""
    if root is not None:
        return Path(root).expanduser()
    return _env_path("VB_RUNS_ROOT") or artifact_dir(DEFAULT_RUNS_DIR)


def results_root(
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the local aggregate-results root."""
    if root is not None:
        return Path(root).expanduser()
    return _env_path("VB_RESULTS_ROOT") or artifact_dir(DEFAULT_RESULTS_DIR)


def certificates_root(
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the local certificate-output root."""
    if root is not None:
        return Path(root).expanduser()
    return _env_path("VB_CERTIFICATES_ROOT") or artifact_dir(
        DEFAULT_CERTIFICATES_DIR
    )


def figures_root(
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the generated-figure root."""
    return artifact_dir(DEFAULT_FIGURES_DIR, root=root)


def research_root(
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the source-controlled research root."""
    if root is not None:
        return Path(root).expanduser()
    return _env_path("VB_RESEARCH_ROOT") or Path(DEFAULT_RESEARCH_ROOT)


def scratch_root(
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the disposable local scratch root."""
    if root is not None:
        return Path(root).expanduser()
    return _env_path("VB_SCRATCH_ROOT") or Path(DEFAULT_SCRATCH_ROOT)


def research_path(
    *parts: str | os.PathLike[str],
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a path inside the source-controlled research tree.

    Research inputs are anchored to the checkout when the configured root is
    relative.  This differs intentionally from artifact paths, whose relative
    form follows the process working directory so callers can redirect output
    to a temporary directory without knowing the repository location.
    """
    base = research_root(root=root)
    if not base.is_absolute():
        base = repository_root() / base
    return base.joinpath(*(Path(part) for part in parts))


_LEGACY_RESEARCH_DIRS = {
    "scripts": "scripts",
    "diagnostics": "diagnostics",
    "sweeps": "sweeps",
    "configs": "configs",
    "clusters": "cluster",
}


def resolve_research_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a research input while accepting its former top-level path.

    Research inputs live below their dedicated workspace directory. The old
    top-level names are intentionally still accepted for one migration cycle
    because sweep files, cluster handoffs, and shell history outlive a
    source-tree move. An existing literal path always wins, so a caller's own
    ``clusters/`` or ``sweeps/`` directory is never replaced by the research
    tree's file of the same name; an old compatibility symlink to the
    canonical file is reported under its canonical path.  Absolute paths and
    paths outside one of the known legacy roots are left untouched.
    """
    candidate = Path(path).expanduser()
    if not candidate.parts:
        return candidate

    if candidate.is_absolute():
        parts = candidate.parts
        try:
            research_index = parts.index("research")
        except ValueError:
            return candidate
        if research_index + 1 >= len(parts):
            return candidate
        canonical_dir = parts[research_index + 1]
        legacy_dir = next(
            (
                old
                for old, canonical in _LEGACY_RESEARCH_DIRS.items()
                if canonical == canonical_dir
            ),
            None,
        )
        if legacy_dir is not None:
            legacy = Path(*parts[:research_index], legacy_dir, *parts[research_index + 2 :])
            if legacy.exists() and not candidate.exists():
                return legacy
        return candidate

    canonical_dir = _LEGACY_RESEARCH_DIRS.get(candidate.parts[0])
    if canonical_dir is not None:
        canonical = research_path(canonical_dir, *candidate.parts[1:])
        if candidate.exists():
            # A real file at the literal path (say, the caller's own
            # ``clusters/`` directory) is what was named. Only a compatibility
            # link to the canonical file is reported under the canonical name.
            same = canonical.exists() and candidate.resolve() == canonical.resolve()
            return canonical if same else candidate
        if canonical.exists():
            return canonical
        return candidate

    if len(candidate.parts) >= 2 and candidate.parts[0] == "research":
        if candidate.exists():
            return candidate
        canonical_dir = candidate.parts[1]
        legacy_dir = next(
            (
                old
                for old, canonical in _LEGACY_RESEARCH_DIRS.items()
                if canonical == canonical_dir
            ),
            None,
        )
        if legacy_dir is not None:
            legacy = Path(legacy_dir, *candidate.parts[2:])
            if legacy.exists():
                return legacy
    return candidate


def canonical_research_relative_path(path: str | os.PathLike[str]) -> Path:
    """Return the canonical repository-relative spelling of a research path."""
    candidate = Path(path)
    if not candidate.parts:
        return candidate
    canonical_dir = _LEGACY_RESEARCH_DIRS.get(candidate.parts[0])
    if canonical_dir is not None:
        return Path("research", canonical_dir, *candidate.parts[1:])
    return candidate


def scratch_path(
    *parts: str | os.PathLike[str],
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a path inside the disposable local scratch tree."""
    return scratch_root(root=root).joinpath(*(Path(part) for part in parts))


def package_config_path(name: str = "default_params.json") -> Path:
    """Return a config shipped inside :mod:`virasoro_bootstrap`.

    ``default_params.json`` is package data rather than a research input: both
    of its readers are library modules (``defaults`` for the cross-Lambda
    values, ``hpc.tasks`` for the per-Lambda profiles), and resolving it
    against the repository made the package unimportable from an installed
    wheel -- ``repository_root()`` falls back to the process directory when
    there is no checkout marker, so the config was looked for under whatever
    directory the caller happened to be in.
    """
    return resources.files(__package__).joinpath(name)


#: Environment variable naming a ``default_params.json`` that replaces the shipped copy.
DEFAULT_PARAMS_ENV = "VB_DEFAULT_PARAMS"


def research_config_path(
    name: str = "default_params.json",
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the config the library reads: the shipped copy unless an override is named explicitly.

    The copy shipped in the package is the only implicit source.  Nothing is
    looked for in the current directory or under a research root, so an
    installed package cannot pick up a stray ``configs/default_params.json``
    from wherever it is run.  Two explicit overrides remain:

    - ``$VB_DEFAULT_PARAMS`` names a replacement ``default_params.json``
      (read at import by :mod:`virasoro_bootstrap.defaults` and
      :mod:`virasoro_bootstrap.hpc.tasks`); a path that does not exist is an
      error, not a silent fallback;
    - ``root`` reads ``<root>/configs/<name>`` when that file exists.
    """
    if root is not None:
        override = research_path("configs", name, root=root)
        return override if override.exists() else package_config_path(name)
    if name == "default_params.json":
        named = _env_path(DEFAULT_PARAMS_ENV)
        if named is not None:
            if not named.is_file():
                raise FileNotFoundError(
                    f"${DEFAULT_PARAMS_ENV} names {named}, which is not a file; unset it to use the "
                    "default_params.json shipped in virasoro_bootstrap")
            return named
    return package_config_path(name)


def dated_output_dir(
    name: str,
    *,
    root: str | os.PathLike[str] | None = None,
    on: date | None = None,
) -> Path:
    """Return ``<experiment-root>/<YYYY-MM-DD>/<name>``.

    The directory is not created here.  ``root`` is retained as an explicit
    complete-output-root override for compatibility with existing scripts.
    """
    name = str(name).strip("/\\")
    if not name:
        raise ValueError("output name must not be empty")
    return experiment_root(root=root) / (on or date.today()).isoformat() / name


def dated_artifact_dir(
    category: str,
    name: str,
    *,
    root: str | os.PathLike[str] | None = None,
    on: date | None = None,
) -> Path:
    """Return ``<artifact-root>/<category>/<YYYY-MM-DD>/<name>``."""
    name = str(name).strip("/\\")
    if not name:
        raise ValueError("artifact name must not be empty")
    return artifact_dir(category, root=root) / (on or date.today()).isoformat() / name


def result_file(
    name: str,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the default aggregate-result path for a run or analysis."""
    return results_root(root=root) / f"{name}-results.json"


def resolve_existing_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a path while accepting the old top-level artifact names.

    Existing paths always win.  If an old path is absent, known legacy roots
    are mapped to their new category so readers can follow a migrated output
    without embedding migration logic in every script.
    """
    candidate = Path(path).expanduser()
    if candidate.exists() or candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if not parts:
        return candidate
    mapping = {
        "test_output": DEFAULT_EXPERIMENTS_DIR,
        "runs": DEFAULT_RUNS_DIR,
        "results": DEFAULT_RESULTS_DIR,
        "cert_checks": DEFAULT_CERTIFICATES_DIR,
        "plots": DEFAULT_FIGURES_DIR,
    }
    category = mapping.get(parts[0])
    if category is None:
        return candidate
    return artifact_dir(category).joinpath(*parts[1:])


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_CERTIFICATES_DIR",
    "DEFAULT_EXPERIMENTS_DIR",
    "DEFAULT_FIGURES_DIR",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_RESEARCH_ROOT",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_ROOT",
    "DEFAULT_RUNS_DIR",
    "DEFAULT_SCRATCH_ROOT",
    "DEFAULT_TEST_OUTPUT_DIR",
    "LEGACY_TEST_OUTPUT_ROOT",
    "artifact_dir",
    "artifact_root",
    "certificates_root",
    "dated_artifact_dir",
    "dated_output_dir",
    "experiment_root",
    "figures_root",
    "research_root",
    "research_config_path",
    "DEFAULT_PARAMS_ENV",
    "research_path",
    "repository_root",
    "canonical_research_relative_path",
    "resolve_research_path",
    "repository_output_roots",
    "resolve_existing_path",
    "result_file",
    "results_root",
    "runs_root",
    "scratch_root",
    "scratch_path",
    "test_output_root",
]
