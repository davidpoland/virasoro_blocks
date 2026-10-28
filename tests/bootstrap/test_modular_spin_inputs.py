"""Modular spin labels and gap overrides must use genuine integer inputs."""
import numpy as np
import pytest
from virasoro_bootstrap.modular_crossing import _validate_spins, _modular_delta_min_overrides


@pytest.mark.parametrize('spin', [True, .5])
def test_invalid_spin_labels_are_rejected(spin):
    with pytest.raises(ValueError, match='integer'):
        _validate_spins((spin, 2))


@pytest.mark.parametrize('spin', [np.bool_(True)])
def test_invalid_override_spin_keys_are_rejected(spin):
    with pytest.raises(ValueError, match='integer'):
        _modular_delta_min_overrides({spin: 3}, (0, 1, 2))


@pytest.mark.parametrize('spin', [1.0, np.int64(1)])
def test_integral_spin_labels_and_overrides_remain_supported(spin):
    assert _validate_spins((spin, 0, 2)) == (0, 1, 2)
    assert _modular_delta_min_overrides({spin: 3}, (0, 1, 2)) == {1: 3}
