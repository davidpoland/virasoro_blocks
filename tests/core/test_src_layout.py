"""Packaging smoke checks for the source-layout migration."""

from __future__ import annotations

import importlib
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


_PACKAGES = ("virasoro_blocks", "virasoro_bootstrap")


def test_source_packages_are_loaded_from_src():
    source_root = (REPOSITORY_ROOT / "src").resolve()
    for package_name in _PACKAGES:
        module = importlib.import_module(package_name)
        package_file = Path(module.__file__).resolve()
        assert source_root in package_file.parents


def test_public_api_facades_import():
    blocks_api = importlib.import_module("virasoro_blocks.api")
    bootstrap_api = importlib.import_module("virasoro_bootstrap.api")
    assert "F_block" in blocks_api.__all__
    assert "BootstrapManifest" in bootstrap_api.__all__


def test_setuptools_discovers_only_src_packages():
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    setuptools = config["tool"]["setuptools"]
    assert setuptools["package-dir"][""] == "src"
    assert setuptools["packages"]["find"]["where"] == ["src"]
