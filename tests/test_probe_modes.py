from tests.helpers.shims import install_shims

# install lightweight mlx shims for tests
install_shims()

import numpy as np
from pathlib import Path

from core.abliteration import ActivationProbeWrapper


class DummyTokenizer:
    def __init__(self, token_map):
        # token_map: dict[str, list[int]]
        self._map = token_map

    def encode(self, text, add_special_tokens=False):
        # naive match on provided keys (in insertion order). Return the mapped token ids.
        for k, v in self._map.items():
            if k in text:
                return v
        # default fallback token ids
        return [1, 2, 3]

    def convert_ids_to_tokens(self, ids):
        return [f"tok{int(i)}" for i in ids]


class DummyWrapper:
    def __init__(self, hidden_size=4):
        self.hidden_size = hidden_size

    def __call__(self, inputs, mask=None, layers_to_probe=None):
        # inputs shape: (1, L)
        L = inputs.shape[1]
        # create fake activations per layer: shape (1, L, hidden_size)
        captured = {}
        for l in (layers_to_probe or [0]):
            arr = np.arange(L * self.hidden_size, dtype=float).reshape(1, L, self.hidden_size)
            captured[l] = arr
        return None, captured


def test_probe_modes_marker_at_end_and_middle(tmp_path):
    # Make two keys so we can simulate a marker followed by more tokens vs marker at end
    tokenizer = DummyTokenizer({"</think> more text": [10, 20], "</think>": [10]})
    wrapper = DummyWrapper(hidden_size=3)
    # prepare fake dataset with two examples: one with marker in middle, one with marker at end
    ds = [
        {"prompt": "This is a test </think> more text"},
        {"prompt": "Another example that ends with marker </think>"},
    ]

    from cli import get_mean_activations

    import numpy as np
    from numpy.testing import assert_allclose

    # follow-token: first example -> use token index 1 (value [3,4,5]), second -> marker at end -> fallback to marker token index 0 ([0,1,2])
    means_follow, _ = get_mean_activations(ds, wrapper, tokenizer, [0], {"hidden_size": 3, "max_position_embeddings": 64}, "desc", probe_marker="</think>", probe_mode="follow-token")
    expected_follow = np.array([ (3.0 + 0.0) / 2.0, (4.0 + 1.0) / 2.0, (5.0 + 2.0) / 2.0 ])
    assert 0 in means_follow
    assert_allclose(np.array(means_follow[0]), expected_follow)

    # marker-token: both examples should use the marker token index 0 -> vectors [0,1,2]
    means_marker, _ = get_mean_activations(ds, wrapper, tokenizer, [0], {"hidden_size": 3, "max_position_embeddings": 64}, "desc", probe_marker="</think>", probe_mode="marker-token")
    expected_marker = np.array([0.0, 1.0, 2.0])
    assert 0 in means_marker
    assert_allclose(np.array(means_marker[0]), expected_marker)

    # last-token: both examples pick last token (first example index 1 -> [3,4,5], second example index 0 -> [0,1,2]) => same as follow-token here
    means_last, _ = get_mean_activations(ds, wrapper, tokenizer, [0], {"hidden_size": 3, "max_position_embeddings": 64}, "desc", probe_marker="</think>", probe_mode="last-token")
    assert 0 in means_last
    assert_allclose(np.array(means_last[0]), expected_follow)