from pathlib import Path
import numpy as np
import json

from tests.helpers.shims import install_shims

# install lightweight mlx shims for tests
install_shims()

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
