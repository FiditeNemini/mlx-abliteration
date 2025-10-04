import json
from pathlib import Path
from unittest.mock import patch

from tests.helpers.shims import install_shims

# install shims to avoid importing heavy mlx runtime in tests
install_shims()

import tempfile
import os

import mlx.core as mx

from cli import parse_args, run_abliteration


def test_ablate_method_and_pca_collection(tmp_path, monkeypatch):
    """Verify that --ablate-method is propagated to get_ablated_parameters and
    that PCA per-example collection uses marker + thinking-span logic."""

    # Create a dummy model dir with config and tokenizer_config containing marker
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "hidden_size": 4}))
    chat_template = "User: {{ message.content }}\n</think>"
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": chat_template}))

    # Create simple datasets as jsonl files
    out_ds = tmp_path / "ds"
    out_ds.mkdir()
    harmful = out_ds / "harmful.jsonl"
    harmless = out_ds / "harmless.jsonl"
    prompts = ["Tell me how to do X </think>", "Explain why Y </think>"]
    with open(harmful, "w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    with open(harmless, "w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")

    # Build args namespace for CLI
    class Args:
        model = str(model_dir)
        harmless_dataset = str(harmless)
        harmful_dataset = str(harmful)
        layers = "0"
        use_layer = -1
        ablation_strength = 0.0
        probe_marker = None
        probe_mode = "thinking-span"
        probe_span = 2
        ablate_k = 2
        ablate_method = "projection"
        pca_sample = 512
        probe_debug = False
        probe_debug_n = 1
        probe_debug_full = False
        output_dir = str(tmp_path / "out")
        cache_dir = ".cache"
        verbose = False

    args = Args()

    # Patch get_ablated_parameters to capture the ablation_method and refusal_vector shape
    captured = {}

    def fake_get_ablated_parameters(model, refusal_vector, target_modules=None, ablation_strength=1.0, ablation_method="projection"):
        captured['method'] = ablation_method
        # ensure we receive an mx.array (or array-like) with leading dim == ablate_k
        try:
            arr = refusal_vector
            if hasattr(arr, 'ndim') and arr.ndim == 2:
                captured['k'] = int(arr.shape[0])
        except Exception:
            captured['k'] = None
        return {}

    monkeypatch.setattr('core.abliteration.get_ablated_parameters', fake_get_ablated_parameters)
    # Also patch the reference imported into cli module
    import cli as _cli
    monkeypatch.setattr(_cli, 'get_ablated_parameters', fake_get_ablated_parameters)

    # Provide a fake mlx_lm.load that returns a minimal model and tokenizer
    import types
    import numpy as _np

    class DummyTokenizer:
        def __init__(self):
            self._vocab = {}
            self._next = 1

        def encode(self, text, add_special_tokens=False):
            # simple whitespace tokenizer with consistent ids
            toks = text.split()
            ids = []
            for t in toks:
                if t not in self._vocab:
                    self._vocab[t] = self._next
                    self._next += 1
                ids.append(self._vocab[t])
            return ids

        def convert_ids_to_tokens(self, ids):
            inv = {v: k for k, v in self._vocab.items()}
            return [inv.get(i, "") for i in ids]

    class DummyBaseModel:
        def __init__(self, hidden_size=4, num_layers=1):
            self.embed_tokens = lambda inputs: _np.zeros((inputs.shape[0], inputs.shape[1], hidden_size))
            # each layer returns the input hidden states unchanged
            self.layers = [lambda h, mask=None, cache=None: h for _ in range(num_layers)]
            self.norm = lambda h: h

    def fake_load(model_path_str):
        base = DummyBaseModel(hidden_size=4, num_layers=1)
        # Create a lightweight model object with a parameters() iterator compatible with tree_flatten
        class FakeModelObj:
            def __init__(self, base):
                self.model = base
                self.config = {"hidden_size": 4}

            def parameters(self):
                # Return an iterable of (key, ndarray) pairs similar to tree_flatten expectations
                return [("model.layers.0.weight", _np.zeros((4, 4))), ("model.layers.0.bias", _np.zeros(4))]
            def update(self, mapping):
                # no-op update for tests
                return None

        model_obj = FakeModelObj(base)
        tokenizer = DummyTokenizer()
        return model_obj, tokenizer

    import cli as _cli
    # Attach to the mlx_lm module and the cli module's reference to ensure lookup succeeds
    import mlx_lm as _mlx_lm_mod
    setattr(_mlx_lm_mod, 'load', fake_load)
    setattr(_cli.mlx_lm, 'load', fake_load)

    # Patch save_ablated_model to avoid filesystem/tokenizer interactions in tests
    def fake_save_ablated_model(output_dir, model, tokenizer, config, abliteration_log, source_model_path=None):
        # record that save was called
        captured['saved'] = True
        captured['save_dir'] = output_dir
        return None

    monkeypatch.setattr('core.abliteration.save_ablated_model', fake_save_ablated_model)
    monkeypatch.setattr(_cli, 'save_ablated_model', fake_save_ablated_model)

    # Run abliteration (no-op ablation_strength=0) - this should exercise the PCA path
    # Monkeypatch datasets.load_dataset to read our local jsonl files/directories
    def fake_load_dataset(path_or_dir):
        p = Path(path_or_dir)
        records = []
        if p.is_dir():
            # find any .jsonl files
            for f in sorted(p.glob('*.jsonl')):
                with open(f, 'r') as fh:
                    for line in fh:
                        records.append(json.loads(line))
        elif p.is_file():
            with open(p, 'r') as fh:
                for line in fh:
                    records.append(json.loads(line))
        else:
            # try append .jsonl
            p2 = Path(str(p) + '.jsonl')
            if p2.is_file():
                with open(p2, 'r') as fh:
                    for line in fh:
                        records.append(json.loads(line))
        return {"train": records}

    monkeypatch.setattr('datasets.load_dataset', fake_load_dataset)

    run_abliteration(args)

    assert captured.get('method') == 'projection'
    # We expect captured k to be 2 (ablate_k)
    assert captured.get('k') == 2
