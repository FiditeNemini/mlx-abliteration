import json
from pathlib import Path
import numpy as np

import types

# Create a tiny local shim for `mlx` to avoid native binary deps during tests.
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

    class linalg:
        @staticmethod
        def norm(v):
            try:
                return float(np.linalg.norm(np.array(v)))
            except Exception:
                return 0.0


# Register 'mlx' and 'mlx.core' modules in sys.modules so imports succeed
import sys
mlx_mod = types.ModuleType('mlx')
mlx_core_mod = _MxCoreShim('mlx.core')
sys.modules['mlx'] = mlx_mod
sys.modules['mlx.core'] = mlx_core_mod
# minimal nn shim
def _save_safetensors(path, tensors, metadata=None):
    out = {}
    for k, v in tensors.items():
        try:
            arr = np.array(v)
            out[k] = arr.tolist()
        except Exception:
            out[k] = v
    with open(path, 'w') as f:
        json.dump({"tensors": out, "metadata": metadata}, f)

mlx_core_mod.save_safetensors = _save_safetensors
nn_mod = types.ModuleType('mlx.nn')

class _ModuleShim:
    def __init__(self):
        pass

class _QuantizedLinearShim:
    def __init__(self):
        # attributes used in core code: group_size, bits
        self.group_size = 1
        self.bits = 8

nn_mod.Module = _ModuleShim
nn_mod.layers = types.ModuleType('mlx.nn.layers')
nn_mod.layers.quantized = types.ModuleType('mlx.nn.layers.quantized')
nn_mod.layers.quantized.QuantizedLinear = _QuantizedLinearShim

sys.modules['mlx.nn'] = nn_mod
sys.modules['mlx.nn.layers'] = nn_mod.layers
sys.modules['mlx.nn.layers.quantized'] = nn_mod.layers.quantized

# mlx.utils shim with tree_unflatten
mlx_utils_mod = types.ModuleType('mlx.utils')
def tree_unflatten(pairs):
    # Expect pairs as list of (key, value)
    return dict(pairs)
mlx_utils_mod.tree_unflatten = tree_unflatten
sys.modules['mlx.utils'] = mlx_utils_mod

# mlx_lm.utils shim with tree_flatten
mlx_lm_mod = types.ModuleType('mlx_lm')
mlx_lm_utils = types.ModuleType('mlx_lm.utils')
def tree_flatten(x):
    # If x is already list of pairs, return as-is; otherwise try to dict->items
    try:
        return list(x)
    except Exception:
        return []
mlx_lm_utils.tree_flatten = tree_flatten
mlx_lm_mod.utils = mlx_lm_utils
sys.modules['mlx_lm'] = mlx_lm_mod
sys.modules['mlx_lm.utils'] = mlx_lm_utils

from core.abliteration import get_ablated_parameters, save_ablated_model


class DummyModel:
    def __init__(self):
        self.parameters_dict = {
            'model.layers.0.mlp.c_proj.weight': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'model.layers.0.self_attn.o_proj.weight': np.array([[0.5, 0.2], [0.1, 0.7]]),
        }

    def parameters(self):
        return list(self.parameters_dict.items())

    def update(self, params):
        for k, v in params.items():
            self.parameters_dict[k] = v


def test_get_ablated_parameters_basic():
    model = DummyModel()
    rv = np.array([1.0, 0.0])
    new_params = get_ablated_parameters(model, rv, ablation_strength=1.0)
    assert 'model.layers.0.mlp.c_proj.weight' in new_params
    assert 'model.layers.0.self_attn.o_proj.weight' in new_params


def test_save_ablated_model_roundtrip(tmp_path):
    src = tmp_path / "src_model"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "hidden_size": 2}))
    (src / "model.safetensors").write_text("")

    model = DummyModel()

    class TokenizerMock:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

    tokenizer = TokenizerMock()
    output_dir = tmp_path / "out"
    abliteration_log = {"source_model": "dummy"}

    save_ablated_model(str(output_dir), model, tokenizer, {"hidden_size": 2}, abliteration_log, source_model_path=str(src))
    assert (output_dir / "abliteration_log.json").is_file()
