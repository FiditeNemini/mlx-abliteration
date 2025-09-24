"""Core components for the MLX Abliteration Toolkit.

This module provides the core functionality for the abliteration process,
including model wrapping for activation probing, refusal direction calculation,
and weight modification.

Dependencies:
- mlx
- mlx-lm
- safetensors
"""
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
    """A wrapper around an MLX model to probe and capture activations.

    This class wraps an existing MLX model (or its base model) to provide a
    forward pass that captures the hidden states from specified layers.

    Attributes:
        embedding (nn.Module): The model's token embedding layer.
        model_layers (List[nn.Module]): The list of transformer layers.
        norm (nn.Module): The final normalization layer.
        lm_head (nn.Module, optional): The language model head.
    """
    def __init__(self, model: nn.Module):
        """Initializes the ActivationProbeWrapper.

        Args:
            model (nn.Module): The MLX model to wrap.

        Raises:
            AttributeError: If the model does not have the expected structure
                (e.g., missing 'layers', 'embed_tokens', or 'norm').
        """
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
            logger.warning("Could not find 'lm_head'. Probing will proceed without returning logits.", extra={"extra_info": {"event": "missing_lm_head"}})

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array],
        layers_to_probe: Optional[List[int]] = None,
    ) -> Tuple[Optional[mx.array], Dict[int, mx.array]]:
        """Performs a forward pass and captures activations.

        Args:
            inputs (mx.array): The input token IDs.
            mask (Optional[mx.array]): The attention mask.
            layers_to_probe (Optional[List[int]]): A list of layer indices from which
                to capture activations. If None, no activations are captured.

        Returns:
            A tuple containing:
            - The model's logits (if lm_head is present).
            - A dictionary mapping layer indices to their captured activations.
        """
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
    """Calculates the refusal direction vector.

    The refusal direction is the difference between the mean activations of
    harmful and harmless prompts.

    Args:
        mean_harmful_activations (mx.array): The mean activation vector for harmful prompts.
        mean_harmless_activations (mx.array): The mean activation vector for harmless prompts.

    Returns:
        mx.array: The calculated refusal direction vector.

    Raises:
        ValueError: If either of the input activation vectors is None.
    """
    if mean_harmful_activations is None or mean_harmless_activations is None:
        raise ValueError("Mean activation vectors cannot be None.")
    refusal_dir = mean_harmful_activations - mean_harmless_activations
    norm = mx.linalg.norm(refusal_dir).item()
    logger.info(f"Calculated refusal direction vector with norm {norm:.4f}", extra={"extra_info": {"event": "refusal_direction_calculated", "actual_output": {"norm": norm}}})
    return refusal_dir


def get_ablated_parameters(model: nn.Module, refusal_vector: mx.array, target_modules: Optional[List[str]] = None, ablation_strength: float = 1.0) -> Dict:
    """
    Orthogonalizes the weights of target modules with respect to the refusal vector.

    This function iterates through the model's parameters and modifies the weights
    of specified modules to be orthogonal to the refusal vector, effectively
    "ablating" the corresponding behavior.

    Args:
        model (nn.Module): The model to modify.
        refusal_vector (mx.array): The refusal direction vector.
        target_modules (Optional[List[str]]): A list of module names to target for
            ablation. Defaults to `["self_attn.o_proj", "mlp.down_proj", "mlp.c_proj"]`.

    Returns:
        Dict: A dictionary of the updated model parameters.
    """
    if target_modules is None:
        target_modules = ["self_attn.o_proj", "mlp.down_proj", "mlp.c_proj"]

    # Normalize the refusal vector
    v_norm = refusal_vector / (mx.linalg.norm(refusal_vector) + 1e-9)

    # Pre-calculate projection matrices for efficiency
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
            logger.warning(f"Could not find module for key: {key}. Skipping ablation.", extra={"extra_info": {"event": "module_not_found", "inputs": {"key": key}}})
            new_flat_params.append((key, W))
            continue

        # Handle quantized linear layers separately
        if isinstance(module, QuantizedLinear):
            module_key = ".".join(key.split('.')[:-1])
            scales_key, biases_key = f"{module_key}.scales", f"{module_key}.biases"
            scales, biases = params_dict.get(scales_key), params_dict.get(biases_key)

            if scales is None:
                logger.warning(f"Could not find scales for quantized weight: {key}. Skipping.", extra={"extra_info": {"event": "scales_not_found", "inputs": {"key": key}}})
                new_flat_params.append((key, W))
                continue

            # Dequantize, ablate, and then re-quantize
            w_float = mx.dequantize(W, scales, biases, module.group_size, module.bits)
            proj_W_on_v = v_proj @ (v_norm_T @ w_float)
            w_ablated_float = w_float - ablation_strength * proj_W_on_v

            # Verification check
            check_norm = mx.linalg.norm(v_norm_T @ w_ablated_float).item()
            logger.info(f"Orthogonalization check for {key}: norm is {check_norm:.4e}", extra={"extra_info": {"event": "ortho_check", "inputs": {"key": key}, "actual_output": {"norm": check_norm}}})

            new_w, new_scales, new_biases = mx.quantize(w_ablated_float, module.group_size, module.bits)

            new_flat_params.extend([(key, new_w), (scales_key, new_scales)])
            if new_biases is not None and biases is not None:
                new_flat_params.append((biases_key, new_biases))
            processed_keys.update([key, scales_key, biases_key])
            modified_count += 1

        # Handle standard linear layers
        elif W.ndim == 2:
            # Project the weight matrix onto the refusal vector
            proj_W_on_v = v_proj @ (v_norm_T @ W)
            # Subtract the projection to make the new weights orthogonal to the vector
            W_ablated = W - ablation_strength * proj_W_on_v

            # Verification check
            check_norm = mx.linalg.norm(v_norm_T @ W_ablated).item()
            logger.info(f"Orthogonalization check for {key}: norm is {check_norm:.4e}", extra={"extra_info": {"event": "ortho_check", "inputs": {"key": key}, "actual_output": {"norm": check_norm}}})

            new_flat_params.append((key, W_ablated))
            modified_count += 1

        else:
            new_flat_params.append((key, W))

    if modified_count > 0:
        logger.info(f"Orthogonalized {modified_count} weight matrices.", extra={"extra_info": {"event": "weights_orthogonalized", "actual_output": {"modified_count": modified_count}}})

    return tree_unflatten(new_flat_params)


def save_ablated_model(
    output_dir: str,
    model: nn.Module,
    tokenizer: Any,
    config: Dict,
    abliteration_log: Dict,
    source_model_path: Optional[str] = None,
):
    """Saves the ablated model, tokenizer, and configuration, preserving sharding if present."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving abliterated model to {output_path}...", extra={"extra_info": {"event": "save_start", "inputs": {"output_dir": output_dir}}})

    source_path = Path(source_model_path) if source_model_path else None
    if not source_path or not source_path.is_dir():
        raise ValueError("A valid source_model_path is required to save the ablated model.")

    # Copy all non-safetensors files from the source directory
    logger.info(f"Copying ancillary files from {source_path}...", extra={"extra_info": {"event": "copy_ancillary_start"}})
    for item in source_path.iterdir():
        if not item.name.endswith(".safetensors"):
            if item.is_file():
                shutil.copy2(item, output_path / item.name)
    logger.info("Ancillary files copied.", extra={"extra_info": {"event": "copy_ancillary_end"}})

    # Get all ablated parameters from the model
    ablated_params = dict(tree_flatten(model.parameters()))

    # Check for a safetensors index file to handle sharding
    index_path = source_path / "model.safetensors.index.json"
    if index_path.is_file():
        logger.info("Sharded model detected. Saving weights into respective shards.", extra={"extra_info": {"event": "sharded_save_start"}})
        with open(index_path, "r") as f:
            index_data = json.load(f)

        raw_metadata = index_data.get("metadata", {})
        # Ensure all metadata values are strings, as required by safetensors
        metadata = {str(k): str(v) for k, v in raw_metadata.items()}
        weight_map = index_data.get("weight_map", {})

        # Invert the weight map to group tensors by filename
        shards = {}
        for tensor_name, filename in weight_map.items():
            if filename not in shards:
                shards[filename] = []
            shards[filename].append(tensor_name)

        # Save each shard
        for filename, tensor_names in shards.items():
            shard_data = {name: ablated_params[name] for name in tensor_names if name in ablated_params}
            if shard_data:
                logger.info(f"Saving {len(shard_data)} tensors to shard: {filename}", extra={"extra_info": {"event": "save_shard", "inputs": {"filename": filename, "tensor_count": len(shard_data)}}})
                mx.save_safetensors(str(output_path / filename), shard_data, metadata=metadata)
            else:
                logger.warning(f"No tensors found for shard '{filename}'. It will be empty.", extra={"extra_info": {"event": "empty_shard", "inputs": {"filename": filename}}})

        # Ensure the index file is also copied
        shutil.copy2(index_path, output_path / index_path.name)

    else:
        # Handle non-sharded models (single safetensors file)
        logger.info("Single-file model detected. Saving all weights to model.safetensors.", extra={"extra_info": {"event": "single_file_save_start"}})
        source_sf_files = list(source_path.glob("*.safetensors"))
        metadata = {}
        if source_sf_files:
            try:
                with safe_open(source_sf_files[0], framework="mlx") as f:
                    raw_metadata = f.metadata()
                    # Ensure all metadata values are strings
                    metadata = {str(k): str(v) for k, v in raw_metadata.items()} if raw_metadata else {}
            except Exception as e:
                logger.error(f"Could not read metadata from source safetensors file: {e}", extra={"extra_info": {"event": "metadata_error", "error_message": str(e)}})

        mx.save_safetensors(str(output_path / "model.safetensors"), ablated_params, metadata=metadata)

    # Save tokenizer, config, and abliteration log
    tokenizer.save_pretrained(str(output_path))
    if config:
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    with open(output_path / "abliteration_log.json", "w") as f:
        json.dump(abliteration_log, f, indent=4)

    logger.info("Model serialization complete.", extra={"extra_info": {"event": "save_end"}})
