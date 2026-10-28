"""The import-time preset catalog must derive from a single config snapshot."""
import importlib.util
import sys
from pathlib import Path

from virasoro_bootstrap import defaults
from virasoro_bootstrap.hpc import tasks


def test_preset_catalogue_uses_one_snapshot(monkeypatch):
    data = defaults.load_default_params()
    first = {'defaults':data['defaults'], 'lambda7':dict(data['lambda7']),
             'lambda11':dict(data['lambda11'])}
    later = {'defaults':data['defaults'], 'lambda7':dict(data['lambda7'],dps=999),
             'lambda11':dict(data['lambda11'],dps=999)}
    reads = []
    def changing_config():
        reads.append(None)
        return first if len(reads) == 1 else later
    monkeypatch.setattr(defaults, 'load_default_params', changing_config)
    name = 'virasoro_bootstrap.hpc._preset_snapshot_test'
    spec = importlib.util.spec_from_file_location(name, Path(tasks.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)
    assert module.LAMBDA_PRESETS[7]['dps'] == first['lambda7']['dps']
    assert module.LAMBDA_PRESETS[11]['dps'] == first['lambda11']['dps']
    assert len(reads) == 1
