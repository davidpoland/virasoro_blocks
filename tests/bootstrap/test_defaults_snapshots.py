"""Default profiles and import-time selectors must use coherent configuration snapshots."""
import importlib.util
import json
from pathlib import Path

import pytest

from virasoro_bootstrap import defaults, paths


def test_profile_merges_one_snapshot_but_later_calls_can_reload(monkeypatch):
    snapshots = [dict(defaults={'common':'first'}, lambda7={'order':20}),
                 dict(defaults={'common':'second'}, lambda7={'order':40})]
    calls = []
    def load():
        calls.append(None)
        return snapshots[min(len(calls)-1,1)]
    monkeypatch.setattr(defaults, 'load_default_params', load)
    assert defaults.profile(7) == {'common':'first', 'order':20}
    assert len(calls) == 1
    assert defaults.profile(7) == {'common':'second', 'order':40}


@pytest.mark.parametrize('document', [[]])
def test_config_root_must_be_an_object(tmp_path, monkeypatch, document):
    config = tmp_path / 'defaults.json'
    config.write_text(json.dumps(document))
    monkeypatch.setattr(defaults, 'DEFAULT_PARAMS_PATH', config)
    with pytest.raises(ValueError, match='object'):
        defaults.load_default_params()


def test_import_time_selectors_share_one_config_read(monkeypatch):
    class ChangingConfig:
        reads = 0
        def read_text(self):
            self.reads += 1
            value = 'first' if self.reads == 1 else 'second'
            return json.dumps({'defaults': dict.fromkeys(
                ('pole_selection','large_spin_pole_selection','global_pole_selection'), value)})
    config = ChangingConfig()
    monkeypatch.setattr(paths, 'research_config_path', lambda: config)
    spec = importlib.util.spec_from_file_location('virasoro_bootstrap._defaults_snapshot_test', Path(defaults.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.DEFAULT_POLE_SELECTION, module.DEFAULT_LARGE_SPIN_POLE_SELECTION,
            module.DEFAULT_GLOBAL_POLE_SELECTION) == ('first',) * 3
    assert config.reads == 1
