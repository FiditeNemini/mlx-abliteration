import mlx_lm
import sys
from pathlib import Path

model_path = "/Users/willdee/AI/FiditeNemini/Qwen3-Next-80B-A3B-Instruct-mxfp4-mlx"

print(f"Loading model from {model_path}...")
try:
    model, tokenizer = mlx_lm.load(model_path)
    print("Model loaded successfully.")
    print(f"Model type: {type(model)}")
    print("Model attributes:")
    print(dir(model))
    
    if hasattr(model, "model"):
        print("\nmodel.model attributes:")
        print(dir(model.model))
        print("\nmodel.model structure:")
        print(model.model)
    else:
        print("\nModel structure:")
        print(model)

    # Check for specific attributes looked for in ActivationProbeWrapper
    print("\n--- Probe Wrapper Checks ---")
    base_model = None
    if hasattr(model, "layers"):
        base_model = model
        print("Found 'layers' on top level model")
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        base_model = model.model
        print("Found 'layers' on model.model")
    
    if base_model:
        print(f"Base model type: {type(base_model)}")
        print(f"Has embed_tokens: {hasattr(base_model, 'embed_tokens')}")
        print(f"Has wte: {hasattr(base_model, 'wte')}")
        print(f"Has norm: {hasattr(base_model, 'norm')}")
        print(f"Has ln_f: {hasattr(base_model, 'ln_f')}")
    else:
        print("Could not identify base model with layers")

    print("\n--- Parameter Keys Check ---")
    from mlx_lm.utils import tree_flatten
    params = dict(tree_flatten(model.parameters()))
    keys = sorted(list(params.keys()))
    print(f"Total parameters: {len(keys)}")
    print("Sample keys (first 20):")
    for k in keys[:20]:
        print(k)
    
    print("\nChecking for scales/biases in layer 0:")
    layer0_keys = [k for k in keys if "layers.0." in k]
    for k in layer0_keys:
        if "scales" in k or "biases" in k:
            print(f"{k}: shape={params[k].shape}")

except Exception as e:
    print(f"Error loading model: {e}")
