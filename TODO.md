# TODO

## Bugfixes

- [x] **Fix Code Duplication in Activation Probing**: The logic for `get_mean_activations` is duplicated across `cli.py`, `gui.py`, and `core/adaptive.py`. This should be refactored into a single reusable function in `core/abliteration.py` or `core/utils.py`.
- [x] **Centralize Target Modules Definition**: The list of `target_modules` is defined in `core/abliteration.py` and duplicated in `core/adaptive.py`. This should be defined in one place (e.g., `core/abliteration.py`) and imported elsewhere.
- [x] **Improve Numerical Stability in Norm Calculation**: In `core/abliteration.py` and `core/adaptive.py`, `1e-9` is added to the norm to avoid division by zero. This can be unstable for very small vectors. It should be changed to `max(norm, 1e-9)`.
- [x] **Fix `scripts/sweep_layers_weights.py`**: This script imports `get_mean_activations` from the old location (`core.cli` or `cli`) and uses a hardcoded list of target modules. It needs to be updated to import from `core.abliteration` and use `DEFAULT_TARGET_MODULES`.

## Enhancements

- [x] **Robust Model Structure Handling**: `ActivationProbeWrapper` assumes specific model attributes (`model.layers` or `model.model.layers`). It should be made more robust to handle different model architectures or provide clearer error messages.
- [x] **Refactor `gui.py` to use shared logic**: After refactoring `get_mean_activations`, update `gui.py` to use the shared function. (Partially done with `find_probe_indices`, but `get_mean_activations_from_dataset` still exists).
- [ ] **Add Unit Tests for Adaptive Ablation**: `core/adaptive.py` logic is complex and should have dedicated unit tests.
