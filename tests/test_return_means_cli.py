import sys
from pathlib import Path
import json
import numpy as np

from tests.helpers.shims import install_shims

install_shims()

import types

import mlx_lm

from cli import run_abliteration


class DummyModel:
    def __init__(self, hidden_size=4, num_layers=1):
        self._params = {}
        # Create a minimal callable embedding, layer, and norm to satisfy
        # ActivationProbeWrapper (they must be callable and accept inputs).
        self.hidden_size = hidden_size

        class Embedding:
            def __init__(self, hidden_size):
                self.hidden_size = hidden_size

            def __call__(self, inputs):
                # inputs: numpy array of shape (batch, seq_len)
                # return embedding of shape (batch, seq_len, hidden_size)
                import numpy as _np
                b, s = inputs.shape
                # simple deterministic embedding: expand token ids into hidden dims
                out = _np.zeros((b, s, self.hidden_size), dtype=_np.float32)
                # Put token id value in first dim to vary outputs
                out[..., 0] = inputs.astype(_np.float32)
                return out

        class LayerStub:
            def __call__(self, h, mask=None, cache=None):
                # Return the hidden state unchanged. Some implementations
                # return a tuple (output,) so both are supported implicitly.
                return h

        class NormStub:
            def __call__(self, h):
                return h

        # minimal layers attribute expected by ActivationProbeWrapper
        self.layers = [LayerStub() for _ in range(num_layers)]
        # callable embed_tokens and norm placeholders
        self.embed_tokens = Embedding(hidden_size)
        self.norm = NormStub()

    def parameters(self):
        return []

    def update(self, params):
        self._params.update(params)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        # simple whitespace split -> ints
        return [len(w) for w in text.split()]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")


def make_args(tmp_path):
    class A:
        pass

    a = A()
    # create a tiny dummy model directory with config.json
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "hidden_size": 4}))
    (model_dir / "model.safetensors").write_text("")

    # minimal datasets
    gen_dir = tmp_path / "generated_datasets"
    gen_dir.mkdir()
    harmless = gen_dir / "harmless_dataset.jsonl"
    harmful = gen_dir / "harmful_dataset.jsonl"
    harmless.write_text(json.dumps({"prompt": "This is harmless."}) + "\n")
    harmful.write_text(json.dumps({"prompt": "This is harmful."}) + "\n")

    a.model = str(model_dir)
    a.harmless_dataset = str(harmless)
    a.harmful_dataset = str(harmful)
    a.output_dir = str(tmp_path / "out")
    a.cache_dir = str(tmp_path / ".cache")
    a.layers = "all"
    a.use_layer = -1
    a.ablation_strength = 0.0
    a.probe_marker = None
    a.probe_debug = False
    a.probe_debug_n = 0
    a.probe_debug_full = False
    a.probe_mode = "follow-token"
    a.probe_span = 1
    a.ablate_k = 1
    a.ablate_method = "projection"
    a.pca_sample = 8
    a.return_means = True
    return a


def test_run_abliteration_returns_means(tmp_path, monkeypatch):
    # patch mlx_lm.load to return dummy model/tokenizer
    def fake_load(p):
        return DummyModel(), DummyTokenizer()

    monkeypatch.setattr(mlx_lm, 'load', fake_load, raising=False)

    args = make_args(tmp_path)

    result = run_abliteration(args)
    assert isinstance(result, dict)
    assert 'harmful_mean_activations' in result
    assert 'harmless_mean_activations' in result
    # confirm layer keys present and are lists/arrays
    hm = result['harmful_mean_activations']
    assert isinstance(hm, dict)
    # layer 0 should exist
    assert any(int(k) == 0 for k in hm.keys())
