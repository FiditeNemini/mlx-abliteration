
import unittest
import mlx.core as mx
import mlx.nn as nn
from core.abliteration import get_ablated_parameters
from mlx_lm.utils import tree_flatten

class MockLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = mx.random.normal((output_dim, input_dim))

class MockModel(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.layers = [
            nn.Module() for _ in range(num_layers)
        ]
        for i, layer in enumerate(self.layers):
            # Simulate structure: layers.0.self_attn.o_proj
            layer.self_attn = nn.Module()
            layer.self_attn.o_proj = MockLayer(hidden_dim, hidden_dim)
            
            # Simulate structure: layers.0.mlp.down_proj
            layer.mlp = nn.Module()
            layer.mlp.down_proj = MockLayer(hidden_dim, hidden_dim)

class TestHereticAblation(unittest.TestCase):
    def test_per_layer_ablation(self):
        hidden_dim = 16
        num_layers = 3
        model = MockModel(num_layers, hidden_dim)
        
        # Create refusal vectors: one per layer
        # Layer 0: vector along axis 0
        v0 = mx.zeros((hidden_dim,))
        v0[0] = 1.0
        
        # Layer 1: vector along axis 1
        v1 = mx.zeros((hidden_dim,))
        v1[1] = 1.0
        
        # Layer 2: vector along axis 2
        v2 = mx.zeros((hidden_dim,))
        v2[2] = 1.0
        
        refusal_vectors = {0: v0, 1: v1, 2: v2}
        
        # Ablation strengths: different for each layer
        strengths = {0: 1.0, 1: 0.5, 2: 0.0}
        
        # Target modules
        target_modules = ["self_attn.o_proj", "mlp.down_proj"]
        
        # Get ablated parameters
        ablated_params_nested = get_ablated_parameters(
            model,
            refusal_vector=refusal_vectors,
            target_modules=target_modules,
            ablation_strength=strengths,
            ablation_method="projection"
        )
        
        # Flatten the result for easier access
        ablated_params = dict(tree_flatten(ablated_params_nested))
        
        # Helper to get param from dict
        def get_p(layer_idx, module_name):
            key = f"layers.{layer_idx}.{module_name}.weight"
            return ablated_params[key]
            
        w0_orig = model.layers[0].self_attn.o_proj.weight
        w0_new = get_p(0, "self_attn.o_proj")
        
        w1_orig = model.layers[1].self_attn.o_proj.weight
        w1_new = get_p(1, "self_attn.o_proj")
        
        w2_orig = model.layers[2].self_attn.o_proj.weight
        w2_new = get_p(2, "self_attn.o_proj")
        
        # Layer 0: Strength 1.0 -> Should change
        diff0 = mx.linalg.norm(w0_orig - w0_new).item()
        self.assertGreater(diff0, 0.001, "Layer 0 should be ablated")
        
        # Layer 1: Strength 0.5 -> Should change, but maybe less than if it was 1.0?
        # Hard to compare directly without exact math, but it should change.
        diff1 = mx.linalg.norm(w1_orig - w1_new).item()
        self.assertGreater(diff1, 0.001, "Layer 1 should be ablated")
        
        # Layer 2: Strength 0.0 -> Should NOT change
        diff2 = mx.linalg.norm(w2_orig - w2_new).item()
        self.assertLess(diff2, 0.0001, "Layer 2 should not be ablated (strength 0)")
        
        print(f"Diffs: L0={diff0}, L1={diff1}, L2={diff2}")

if __name__ == '__main__':
    unittest.main()
