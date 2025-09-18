import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import tree_flatten
from mlx.utils import tree_unflatten
from safetensors import safe_open

from .utils import get_module_from_key
from mlx.nn.layers.quantized import QuantizedLinear

logger = logging.getLogger(__name__)


class ActivationProbeWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            base_model = model.model
        elif hasattr(model, "layers"):
            base_model = model
        else:
            raise AttributeError("The provided model does not have the expected structure to find '.layers'.")

        if not all(hasattr(base_model, attr) for attr in ["embed_tokens", "layers", "norm"]):
            raise AttributeError("The model's structure is missing 'embed_tokens', 'layers', or 'norm'.")

        self.embedding = base_model.embed_tokens
        self.model_layers = base_model.layers
        self.norm = base_model.norm
        self.lm_head = getattr(model, "lm_head", getattr(base_model, "lm_head", None))

        if self.lm_head is None:
            logger.warning("Could not find 'lm_head'. Probing will proceed without returning logits.")

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array],
        layers_to_probe: Optional[List[int]] = None,
    ) -> Tuple[Optional[mx.array], Dict[int, mx.array]]:
        captured_activations = {}
        h = self.embedding(inputs)

        for i, layer in enumerate(self.model_layers):
            output = layer(h, mask=mask, cache=None)
            h = output[0] if isinstance(output, tuple) else output
            if layers_to_probe is not None and i in layers_to_probe:
                captured_activations[i] = h

        h = self.norm(h)
        logits = self.lm_head(h) if self.lm_head is not None else None
        return logits, captured_activations


def calculate_refusal_direction(mean_harmful_activations: mx.array, mean_harmless_activations: mx.array) -> mx.array:
    if mean_harmful_activations is None or mean_harmless_activations is None:
        raise ValueError("Mean activation vectors cannot be None.")
    refusal_dir = mean_harmful_activations - mean_harmless_activations
    logger.info(f"Calculated refusal direction vector with norm {mx.linalg.norm(refusal_dir):.4f}")
    return refusal_dir


def get_ablated_parameters(model: nn.Module, refusal_vector: mx.array, target_modules: Optional[List[str]] = None) -> Dict:
    if target_modules is None:
        target_modules = ["self_attn.o_proj", "mlp.down_proj", "mlp.c_proj"]
    v_norm = refusal_vector / (mx.linalg.norm(refusal_vector) + 1e-9)
    v_proj = v_norm[:, None]
    v_norm_T = v_norm[None, :]
    flat_params = tree_flatten(model.parameters())
    params_dict = dict(flat_params)
    processed_keys = set()
    new_flat_params = []
    modified_count = 0

    for key, W in flat_params:
        if key in processed_keys:
            continue
        is_target = any(target in key for target in target_modules) and "weight" in key
        if not is_target:
            new_flat_params.append((key, W))
            continue
        try:
            module = get_module_from_key(model, key)
        except (AttributeError, KeyError):
            logger.warning(f"Could not find module for key: {key}. Skipping ablation.")
            new_flat_params.append((key, W))
            continue

        if isinstance(module, QuantizedLinear):
            module_key = ".".join(key.split('.')[:-1])
            scales_key, biases_key = f"{module_key}.scales", f"{module_key}.biases"
            scales, biases = params_dict.get(scales_key), params_dict.get(biases_key)
            if scales is None:
                logger.warning(f"Could not find scales for quantized weight: {key}. Skipping.")
                new_flat_params.append((key, W))
                continue
            w_float = mx.dequantize(W, scales, biases, module.group_size, module.bits)
            proj_W_on_v = v_proj @ (v_norm_T @ w_float)
            w_ablated_float = w_float - proj_W_on_v
            new_w, new_scales, new_biases = mx.quantize(w_ablated_float, module.group_size, module.bits)
            new_flat_params.extend([(key, new_w), (scales_key, new_scales)])
            if new_biases is not None and biases is not None:
                new_flat_params.append((biases_key, new_biases))
            processed_keys.update([key, scales_key, biases_key])
            modified_count += 1
        elif W.ndim == 2:
            proj_W_on_v = v_proj @ (v_norm_T @ W)
            W_ablated = W - proj_W_on_v
            new_flat_params.append((key, W_ablated))
            modified_count += 1
        else:
            new_flat_params.append((key, W))
    if modified_count > 0:
        logger.info(f"Orthogonalized {modified_count} weight matrices.")
    return tree_unflatten(new_flat_params)


def save_ablated_model(
    output_dir: str,
    model: nn.Module,
    tokenizer: Any,
    config: Dict,
    abliteration_log: Dict,
    source_model_path: Optional[str] = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving abliterated model and tokenizer to {output_path}...")

    source_path = Path(source_model_path) if source_model_path else None

    if source_path and source_path.is_dir():
        logger.info(f"Copying ancillary files from {source_path}...")
        for item in source_path.iterdir():
            if item.is_file() and item.suffix not in [".safetensors", ".bin", ".pt"]:
                shutil.copy2(item, output_path)

    metadata = {}
    if source_path:
        try:
            # Look for any .safetensors file, not just one named 'weights.safetensors'
            source_sf_files = list(source_path.glob("*.safetensors"))
            if source_sf_files:
                with safe_open(source_sf_files[0], framework="mlx") as f:
                    metadata = f.metadata()
                if metadata:
                    logger.info("Successfully extracted metadata from source safetensors file.")
        except Exception as e:
            logger.error(f"Could not read metadata from source safetensors file: {e}")

    flat_params = tree_flatten(model.parameters())
    # Use a consistent filename like 'model.safetensors' for better compatibility
    mx.save_safetensors(str(output_path / "model.safetensors"), dict(flat_params), metadata=metadata)
    logger.info("New model weights saved successfully with metadata.")

    tokenizer.save_pretrained(str(output_path))
    
    # DEFINITIVE FIX: Use the config dictionary passed directly as an argument.
    if config:
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    else:
        logger.warning("No config dictionary was provided. 'config.json' will be missing.")

    with open(output_path / "abliteration_log.json", "w") as f:
        json.dump(abliteration_log, f, indent=4)
    logger.info("Model serialization complete.")

