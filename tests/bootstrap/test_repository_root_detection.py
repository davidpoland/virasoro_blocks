"""Installed packages must not mistake an enclosing project for their checkout."""

import pytest

from virasoro_bootstrap import paths


@pytest.fixture(autouse=True)
def clear_root_cache():
    paths.repository_root.cache_clear()
    yield
    paths.repository_root.cache_clear()


def test_installed_package_ignores_unrelated_ancestor_project(tmp_path, monkeypatch):
    project = tmp_path / 'other-project'
    module = project / '.venv/lib/python3.12/site-packages/virasoro_bootstrap/paths.py'
    module.parent.mkdir(parents=True)
    module.write_text('')
    (project / 'pyproject.toml').write_text('[project]\nname="other"\n')
    working = tmp_path / 'work'
    working.mkdir()
    monkeypatch.setattr(paths, '__file__', str(module))
    monkeypatch.delenv('VB_RESEARCH_ROOT', raising=False)
    monkeypatch.chdir(working)
    assert paths.repository_root() == working
    unrelated_config = project / 'research/configs/default_params.json'
    unrelated_config.parent.mkdir(parents=True)
    unrelated_config.write_text('{}')
    assert paths.research_config_path() == paths.package_config_path()


def test_source_checkout_is_found_from_unrelated_cwd(tmp_path, monkeypatch):
    project = tmp_path / 'checkout'
    module = project / 'src/virasoro_bootstrap/paths.py'
    module.parent.mkdir(parents=True)
    module.write_text('')
    (project / 'pyproject.toml').write_text('[project]\nname="virasoro_blocks"\n')
    monkeypatch.setattr(paths, '__file__', str(module))
    monkeypatch.chdir(tmp_path)
    assert paths.repository_root() == project


def test_installed_fallback_follows_current_working_directory(tmp_path, monkeypatch):
    module = tmp_path / 'installed/virasoro_bootstrap/paths.py'
    module.parent.mkdir(parents=True)
    module.write_text('')
    monkeypatch.setattr(paths, '__file__', str(module))
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir(); second.mkdir()
    monkeypatch.chdir(first)
    assert paths.repository_root() == first
    monkeypatch.chdir(second)
    assert paths.repository_root() == second


def test_default_params_are_never_read_from_the_working_directory(tmp_path, monkeypatch):
    """A stray research/configs/default_params.json under the cwd (or $VB_RESEARCH_ROOT) used to override the
    shipped presets silently, and a malformed one made ``import virasoro_bootstrap.hpc.tasks`` raise."""
    stray = tmp_path / 'research/configs/default_params.json'
    stray.parent.mkdir(parents=True)
    stray.write_text('not json')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('VB_RESEARCH_ROOT', str(tmp_path / 'research'))
    monkeypatch.delenv('VB_DEFAULT_PARAMS', raising=False)
    assert paths.research_config_path() == paths.package_config_path()


def test_default_params_override_is_explicit(tmp_path, monkeypatch):
    import json

    from virasoro_bootstrap import defaults

    override = tmp_path / 'my_params.json'
    override.write_text(json.dumps({'defaults': {}, 'lambda3': {}}))
    monkeypatch.setenv(paths.DEFAULT_PARAMS_ENV, str(override))
    assert paths.research_config_path() == override
    override.write_text('{')
    monkeypatch.setattr(defaults, 'DEFAULT_PARAMS_PATH', paths.research_config_path())
    with pytest.raises(ValueError, match='my_params.json: configuration is not valid JSON'):
        defaults.load_default_params()
    monkeypatch.setenv(paths.DEFAULT_PARAMS_ENV, str(tmp_path / 'missing.json'))
    with pytest.raises(FileNotFoundError, match='VB_DEFAULT_PARAMS'):
        paths.research_config_path()
