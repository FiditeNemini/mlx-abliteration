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

        Returns a tuple of (logits or None, captured_activations dict).
        """
        captured_activations: Dict[int, mx.array] = {}
        h = self.embedding(inputs)

        class DummyCache:
            def __init__(self):
                self.offset = 0
                self._store = {0: None, 1: None}

            def __getitem__(self, idx):
                return self._store.get(idx, None)

            def __setitem__(self, idx, val):
                self._store[idx] = val

            def update_and_fetch(self, keys, values):
                return keys, values

        cache = DummyCache()

        for i, layer in enumerate(self.model_layers):
            output = layer(h, mask=mask, cache=cache)
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


def get_ablated_parameters(model: nn.Module, refusal_vector: mx.array, target_modules: Optional[List[str]] = None, ablation_strength: float = 1.0, ablation_method: str = "projection") -> Dict:
    """
    Orthogonalizes the weights of target modules with respect to the refusal vector.

    This function iterates through the model's parameters and modifies the weights
    of specified modules to be orthogonal to the refusal vector, effectively
    "ablating" the corresponding behavior.

            w_ablated_float = w_float - ablation_strength * proj_W_on_v
        model (nn.Module): The model to modify.
        refusal_vector (mx.array): The refusal direction vector.
        target_modules (Optional[List[str]]): A list of module names to target for
            ablation. Defaults to `["self_attn.o_proj", "mlp.down_proj", "mlp.c_proj"]`.

    Args:
        ablation_method (str): Either 'sequential' to subtract projections for each component
            sequentially (legacy behavior), or 'projection' to build a projection matrix
            P = sum_i v_i v_i^T and remove P @ W in one step. Defaults to 'projection'.

    Returns:
        Dict: A dictionary of the updated model parameters.
    """
    if target_modules is None:
        # Include common naming variants used across model families (e.g.,
        # 'switch_mlp.down_proj' or 'mlp.up_proj') so default ablation
        # targets match more models out of the box.
        target_modules = [
            "self_attn.o_proj",
            "mlp.down_proj",
            "mlp.c_proj",
            "mlp.up_proj",
            "mlp.switch_mlp.down_proj",
            "mlp.switch_mlp.up_proj",
        ]

    # Support single-vector (shape [H]) or multiple components (shape [K, H])
    # Ensure refusal_vector is an array with leading dimension = K (number of components)
    rv = refusal_vector
    if rv.ndim == 1:
        rv = rv[None, :]

    # Normalize each component
    v_norms = []
    for i in range(rv.shape[0]):
        v = rv[i]
        v_norm = v / (mx.linalg.norm(v) + 1e-9)
        v_norms.append(v_norm)

    # Pre-calculate projection column vectors for each component
    v_projs = [v[None, :].T for v in v_norms]  # each is [H,1]
    v_norm_Ts = [v[None, :] for v in v_norms]  # each is [1,H]

    # If using projection method, build P = sum_i v_i v_i^T
    P = None
    if ablation_method == "projection":
        try:
            # build projection matrix as sum outer products of unit vectors
            P = sum((v[:, None] @ v[None, :]) for v in v_norms)
        except Exception:
            P = None

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
            if ablation_method == "projection" and P is not None:
                # apply projection removal in one step: subtract P @ w_float for each column
                proj = P @ w_float
                w_ablated_float = w_float - ablation_strength * proj
            else:
                # Sequentially remove projections onto each component
                w_ablated_float = w_float
                for v_proj_i, v_norm_T_i in zip(v_projs, v_norm_Ts):
                    proj_W_on_v = v_proj_i @ (v_norm_T_i @ w_ablated_float)
                    w_ablated_float = w_ablated_float - ablation_strength * proj_W_on_v

            # Verification check: compute max residual projection norm across components
            try:
                norms = [float(mx.linalg.norm(v_norm_T_i @ w_ablated_float).item()) for v_norm_T_i in v_norm_Ts]
                check_norm = max(norms) if norms else 0.0
            except Exception:
                check_norm = None
            logger.info(f"Orthogonalization check for {key}: norm is {check_norm}", extra={"extra_info": {"event": "ortho_check", "inputs": {"key": key}, "actual_output": {"norm": check_norm}}})

            new_w, new_scales, new_biases = mx.quantize(w_ablated_float, module.group_size, module.bits)

            new_flat_params.extend([(key, new_w), (scales_key, new_scales)])
            if new_biases is not None and biases is not None:
                new_flat_params.append((biases_key, new_biases))
            processed_keys.update([key, scales_key, biases_key])
            modified_count += 1

        # Handle standard linear layers
        elif W.ndim == 2:
            # Project the weight matrix onto the refusal vector
            # Sequentially remove projections onto each component
            if ablation_method == "projection" and P is not None:
                proj = P @ W
                W_ablated = W - ablation_strength * proj
            else:
                W_ablated = W
                for v_proj_i, v_norm_T_i in zip(v_projs, v_norm_Ts):
                    proj_W_on_v = v_proj_i @ (v_norm_T_i @ W_ablated)
                    W_ablated = W_ablated - ablation_strength * proj_W_on_v

            # Verification check: compute max residual projection norm across components
            try:
                norms = [float(mx.linalg.norm(v_norm_T_i @ W_ablated).item()) for v_norm_T_i in v_norm_Ts]
                check_norm = max(norms) if norms else 0.0
            except Exception:
                check_norm = None
            logger.info(f"Orthogonalization check for {key}: norm is {check_norm}", extra={"extra_info": {"event": "ortho_check", "inputs": {"key": key}, "actual_output": {"norm": check_norm}}})

            new_flat_params.append((key, W_ablated))
            modified_count += 1

        else:
            new_flat_params.append((key, W))

    if modified_count > 0:
        logger.info(f"Orthogonalized {modified_count} weight matrices.", extra={"extra_info": {"event": "weights_orthogonalized", "actual_output": {"modified_count": modified_count}}})

        # Emit per-tensor max-abs-diff diagnostics so callers (and logs) can
        # verify that ablation produced non-zero changes for each modified
        # parameter. This is useful when the later `model.update(...)`/save
        # path may not persist changes for unexpected reasons.
        try:
            for k, new_v in new_flat_params:
                # Only report for keys that existed in the original params dict
                orig_v = params_dict.get(k)
                if orig_v is None:
                    continue
                try:
                    # compute max abs diff using mlx.linalg if available
                    diff = mx.array(orig_v) - mx.array(new_v)
                    max_abs = float(mx.linalg.norm(diff).item())
                except Exception:
                    # fall back to a conservative numeric estimate
                    try:
                        import numpy as _np

                        o = _np.array(orig_v)
                        n = _np.array(new_v)
                        max_abs = float(_np.linalg.norm(o - n))
                    except Exception:
                        max_abs = None

                logger.info(f"Post-ablation tensor diff for {k}: {max_abs}", extra={"extra_info": {"event": "post_ablation_diff", "inputs": {"key": k}, "actual_output": {"max_abs_diff": max_abs}}})
        except Exception:
            # Never fail the ablation because of logging instrumentation
            logger.debug("Failed to emit per-tensor post-ablation diffs", exc_info=True)

    return tree_unflatten(new_flat_params)


def save_ablated_model(
    output_dir: str,
    model: nn.Module,
    tokenizer: Any,
    config: Dict,
    abliteration_log: Dict,
    source_model_path: Optional[str] = None,
    dump_dequant: bool = False,
):
    """Saves the ablated model, tokenizer, and configuration, preserving sharding if present."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving abliterated model to {output_path}...", extra={"extra_info": {"event": "save_start", "inputs": {"output_dir": output_dir}}})

    source_path = Path(source_model_path) if source_model_path else None
    if not source_path or not source_path.is_dir():
        raise ValueError("A valid source_model_path is required to save the ablated model.")

    # Copy all non-safetensors files from the source directory first
    logger.info(f"Copying ancillary files from {source_path}...", extra={"extra_info": {"event": "copy_ancillary_start"}})
    for item in source_path.iterdir():
        if not item.name.endswith(".safetensors"):
            if item.is_file():
                shutil.copy2(item, output_path / item.name)
    logger.info("Ancillary files copied.", extra={"extra_info": {"event": "copy_ancillary_end"}})

    # Get all ablated parameters from the model
    ablated_params = dict(tree_flatten(model.parameters()))

    # Optional: dump dequantized floats for ablated tensors to aid offline
    # inspection and avoid the common pitfall of comparing packed uint/int
    # shards directly. Dumps are written to <output_dir>/dequant_dumps/*.npy
    if dump_dequant:
        try:
            import numpy as _np

            dump_dir = output_path / "dequant_dumps"
            dump_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Dumping dequantized tensors to {dump_dir}", extra={"extra_info": {"event": "dequant_dump_start", "inputs": {"dump_dir": str(dump_dir)}}})

            for key, val in ablated_params.items():
                # Only dump weights and their dequantized float equivalents
                if not key.endswith("weight"):
                    continue

                tensor_to_save = None
                try:
                    # Try to get module to determine quantization attributes
                    try:
                        module = get_module_from_key(model, key)
                    except Exception:
                        module = None

                    # If module is a QuantizedLinear or there are scales present,
                    # attempt to dequantize using mlx runtime.
                    if isinstance(module, QuantizedLinear):
                        module_key = ".".join(key.split('.')[:-1])
                        scales_key = f"{module_key}.scales"
                        biases_key = f"{module_key}.biases"
                        scales = ablated_params.get(scales_key)
                        biases = ablated_params.get(biases_key)

                        if scales is not None:
                            try:
                                w_float = mx.dequantize(val, scales, biases, module.group_size, module.bits)
                                # convert to numpy if possible
                                try:
                                    tensor_to_save = _np.array(w_float)
                                except Exception:
                                    try:
                                        tensor_to_save = _np.array(w_float.tolist())
                                    except Exception:
                                        tensor_to_save = None
                            except Exception as e:
                                logger.debug(f"Failed to dequantize {key}: {e}", exc_info=True)
                                tensor_to_save = None
                    else:
                        # Non-quantized: try to coerce to numpy
                        try:
                            tensor_to_save = _np.array(val)
                        except Exception:
                            try:
                                tensor_to_save = _np.array(val.tolist())
                            except Exception:
                                tensor_to_save = None

                    if tensor_to_save is not None:
                        # sanitize filename
                        fname = key.replace('.', '_') + '.npy'
                        outp = dump_dir / fname
                        try:
                            _np.save(str(outp), tensor_to_save)
                            logger.info(f"Wrote dequantized tensor dump: {outp.name}", extra={"extra_info": {"event": "dequant_dump_write", "inputs": {"tensor": key, "file": str(outp.name)}}})
                        except Exception:
                            logger.debug(f"Failed to save dequantized dump for {key}", exc_info=True)
                except Exception:
                    # Protect dequant dump loop from any unexpected error per-tensor
                    tensor_to_save = None

        except Exception:
            logger.debug("Dequant dump instrumentation failed; continuing without dumps", exc_info=True)

    index_path = source_path / "model.safetensors.index.json"
    if index_path.is_file():
        logger.info("Sharded model detected. Saving weights into respective shards.", extra={"extra_info": {"event": "sharded_save_start"}})
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})

        # Use the weight_map as the source of truth for which tensors to save.
        # This prevents any extraneous tensors created during ablation from being saved.
        shards_to_save = {}
        for name, filename in weight_map.items():
            if name not in ablated_params:
                logger.warning(f"Tensor '{name}' from source weight_map not found in the ablated model's parameters. It will be missing from the output.", extra={"extra_info": {"event": "tensor_missing_from_ablated", "inputs": {"tensor_name": name}}})
                continue

            if filename not in shards_to_save:
                shards_to_save[filename] = {}
            shards_to_save[filename][name] = ablated_params[name]

        # Log any parameters that were ablated but will be discarded
        ablated_but_not_saved = set(ablated_params.keys()) - set(weight_map.keys())
        if ablated_but_not_saved:
            logger.warning(f"The following {len(ablated_but_not_saved)} tensor(s) were generated during ablation but are not in the source model's weight map and will be discarded: {', '.join(ablated_but_not_saved)}", extra={"extra_info": {"event": "discarding_extra_tensors", "inputs": {"tensors": list(ablated_but_not_saved)}}})

        # Save each shard with its original metadata
        for filename, shard_data in shards_to_save.items():
            source_shard_path = source_path / filename
            metadata = {}
            if source_shard_path.is_file():
                try:
                    with safe_open(source_shard_path, framework="mlx") as f:
                        raw_metadata = f.metadata()
                        if raw_metadata:
                            metadata = {str(k): str(v) for k, v in raw_metadata.items()}
                except Exception as e:
                    logger.error(f"Could not read metadata from source shard {filename}: {e}", extra={"extra_info": {"event": "shard_metadata_error", "inputs": {"filename": filename}, "error_message": str(e)}})

            logger.info(f"Saving {len(shard_data)} tensors to shard: {filename}", extra={"extra_info": {"event": "save_shard", "inputs": {"filename": filename, "tensor_count": len(shard_data)}}})
            mx.save_safetensors(str(output_path / filename), shard_data, metadata=metadata)

        # Ensure the index file is also copied (it might have been missed if not in the initial loop)
        if not (output_path / index_path.name).exists():
            shutil.copy2(index_path, output_path / index_path.name)

    else:
        # Handle non-sharded models
        logger.info("Single-file model detected. Saving all weights to model.safetensors.", extra={"extra_info": {"event": "single_file_save_start"}})
        source_sf_files = list(source_path.glob("*.safetensors"))
        metadata = {}
        if source_sf_files:
            try:
                with safe_open(source_sf_files[0], framework="mlx") as f:
                    raw_metadata = f.metadata()
                    if raw_metadata:
                        metadata = {str(k): str(v) for k, v in raw_metadata.items()}
            except Exception as e:
                logger.error(f"Could not read metadata from source safetensors file: {e}", extra={"extra_info": {"event": "metadata_error", "error_message": str(e)}})

        mx.save_safetensors(str(output_path / "model.safetensors"), ablated_params, metadata=metadata)

    # Save tokenizer and abliteration log
    tokenizer.save_pretrained(str(output_path))
    with open(output_path / "abliteration_log.json", "w") as f:
        json.dump(abliteration_log, f, indent=4)

    logger.info("Model serialization complete.", extra={"extra_info": {"event": "save_end"}})

    # Emit a directory listing and quick stats so callers can confirm which
    # files were actually written by this process. This helps debug cases
    # where the log claims success but the expected index/shards are missing
    # on disk (e.g., due to working-dir mismatches or external cleanup).
    try:
        files = []
        for p in sorted(output_path.iterdir()):
            try:
                st = p.stat()
                files.append({"name": p.name, "size": st.st_size})
            except Exception:
                files.append({"name": p.name, "size": None})

        index_present = (output_path / "model.safetensors.index.json").is_file()
        logger.info(
            "Output directory listing",
            extra={
                "extra_info": {
                    "event": "save_dir_listing",
                    "actual_output": {"file_count": len(files), "index_present": index_present, "files": files},
                }
            },
        )
    except Exception:
        logger.debug("Failed to emit save directory listing", exc_info=True)
