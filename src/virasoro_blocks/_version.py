"""The installed distribution's version (one source: pyproject.toml)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("virasoro_blocks")
except PackageNotFoundError:      # run from a source tree without an install
    __version__ = "unknown"
