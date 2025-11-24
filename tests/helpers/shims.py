"""Test helper shims to mock minimal `mlx` and `mlx_lm` APIs used in unit tests.

This module provides a lightweight in-test shim so unit tests can run without
native `mlx` binaries. Import this from tests to keep test modules concise.
"""
import json
import types
import sys
import numpy as np
from pathlib import Path


class _MxCoreShim(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)

    @staticmethod
    def array(x):
        return np.array(x)

    @staticmethod
    def zeros(n):
        return np.zeros(n)

    @staticmethod
    def eval(_):
        return None

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def sum(a, axis=None):
        return np.sum(a, axis=axis)

    @staticmethod
    def allclose(a, b):
        return np.allclose(a, b)

    @staticmethod
    def abs(a):
        return np.abs(a)

    @staticmethod
    def stack(arrays, axis=0):
        return np.stack(arrays, axis=axis)

    @staticmethod
    def dequantize(w, scales, biases, group_size, bits):
        return np.array(w)

    @staticmethod
    def quantize(w, group_size, bits):
        return np.array(w), np.array(w), np.array(w)

    class linalg:
        @staticmethod
        def norm(v):
            try:
                # Return a numpy scalar so callers can call .item()
                return np.array(np.linalg.norm(np.array(v)))
            except Exception:
                return np.array(0.0)


def install_shims():
    """Register shims into sys.modules. Call at the top of tests."""
    mlx_mod = types.ModuleType('mlx')
    mlx_core_mod = _MxCoreShim('mlx.core')
    sys.modules['mlx'] = mlx_mod
    sys.modules['mlx.core'] = mlx_core_mod

    # simple save_safetensors shim that writes a JSON placeholder
    def _save_safetensors(path, tensors, metadata=None):
        out = {}
        for k, v in tensors.items():
            try:
                arr = np.array(v)
                out[k] = arr.tolist()
            except Exception:
                out[k] = v
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({"tensors": out, "metadata": metadata}, f)

    mlx_core_mod.save_safetensors = _save_safetensors

    # nn shim
    nn_mod = types.ModuleType('mlx.nn')

    class _ModuleShim:
        def __init__(self):
            pass

    class _QuantizedLinearShim:
        def __init__(self):
            self.group_size = 1
            self.bits = 8

    nn_mod.Module = _ModuleShim
    nn_mod.layers = types.ModuleType('mlx.nn.layers')
    nn_mod.layers.quantized = types.ModuleType('mlx.nn.layers.quantized')
    nn_mod.layers.quantized.QuantizedLinear = _QuantizedLinearShim

    sys.modules['mlx.nn'] = nn_mod
    sys.modules['mlx.nn.layers'] = nn_mod.layers
    sys.modules['mlx.nn.layers.quantized'] = nn_mod.layers.quantized

    # mlx.utils shim
    mlx_utils_mod = types.ModuleType('mlx.utils')

    def tree_unflatten(pairs):
        return dict(pairs)

    mlx_utils_mod.tree_unflatten = tree_unflatten
    sys.modules['mlx.utils'] = mlx_utils_mod

    # mlx_lm.utils shim
    mlx_lm_mod = types.ModuleType('mlx_lm')
    mlx_lm_utils = types.ModuleType('mlx_lm.utils')

    def tree_flatten(x):
        try:
            return list(x)
        except Exception:
            return []

    mlx_lm_utils.tree_flatten = tree_flatten
    mlx_lm_mod.utils = mlx_lm_utils
    sys.modules['mlx_lm'] = mlx_lm_mod
    sys.modules['mlx_lm.utils'] = mlx_lm_utils


__all__ = ["install_shims"]
