"""Gradio-based user interface for the MLX Abliteration Toolkit.

This script launches a Gradio web UI that allows users to interactively perform
the abliteration process on a language model.

Workflow:
1.  Set up structured logging for the GUI component.
2.  Define the `run_abliteration_stream` function, which executes the core
    abliteration logic and yields progress updates to the UI.
3.  Define the `create_ui` function, which builds the Gradio interface,
    including input fields, buttons, and output displays.
4.  Launch the Gradio application.

Dependencies:
- gradio
- torch
- mlx
- mlx-lm
- datasets
- huggingface-hub
"""
import gradio as gr
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Generator, Tuple, Any, Optional

# Add project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Prepend the vendor directory to the Python path to load our patched model
sys.path.insert(0, str(Path(__file__).parent / "core" / "vendor"))

from core.asset_resolver import resolve_asset
from core.abliteration import (
    ActivationProbeWrapper,
    calculate_refusal_direction,
    get_ablated_parameters,
    save_ablated_model,
)
from core.logging_config import setup_structured_logging
from core.utils import extract_eot_from_chat_template
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import tree_flatten
from datasets import load_dataset

def get_mean_activations_from_dataset(
    dataset,
    wrapper: ActivationProbeWrapper,
    tokenizer: Any,
    layers_to_probe: List[int],
    config: Dict,
    desc: str,
    progress: gr.Progress,
    probe_marker: Optional[str] = None,
    probe_mode: str = "follow-token",
    probe_span: int = 1,
    probe_debug: bool = False,
    probe_debug_n: int = 3,
    probe_debug_full: bool = False,
) -> Tuple[Dict[int, mx.array], list]:
    """Computes mean activations for a given dataset.

    probe_marker: Optional[str] = None,
    probe_mode: str = "follow-token",
    prompt and uses the activation of the token immediately preceding it.
    Otherwise, it defaults to using the activation of the last token.

    Args:
        dataset: The dataset to process.
        wrapper (ActivationProbeWrapper): The model wrapper for probing.
        tokenizer (Any): The tokenizer.
        layers_to_probe (List[int]): A list of layer indices to probe.
        config (Dict): The model's configuration dictionary.
        desc (str): A description for the progress bar.
        progress (gr.Progress): A Gradio Progress object to update the UI.
        probe_marker (Optional[str]): A string marker to find for probing.

    Returns:
        Dict[int, mx.array]: A dictionary mapping layer indices to mean activations.
    """
    hidden_size = config["hidden_size"]
    mean_activations = {layer: mx.zeros(hidden_size) for layer in layers_to_probe}
    counts = {layer: 0 for layer in layers_to_probe}
    max_seq_len = config.get("max_position_embeddings", 4096)

    # Gracefully handle empty or whitespace-only strings from the UI
    if probe_marker and probe_marker.strip():
        marker_tokens = mx.array(tokenizer.encode(probe_marker, add_special_tokens=False))
    else:
        marker_tokens = None

    # Track whether the marker was ever found to avoid noisy per-item warnings
    marker_found_any = False
    sample_not_found_examples: list[str] = []
    probe_debug_lines: list[str] = []

    for item in progress.tqdm(dataset, desc=desc):
        prompt = item.get("prompt") or item.get("text")
        
        # Handle chat-formatted datasets with "messages" key
        if not prompt and "messages" in item:
            # Extract user message content from messages list
            messages = item["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        prompt = msg.get("content")
                        break
        
        if not prompt:
            continue
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        _, captured = wrapper(tokens[None], mask=None, layers_to_probe=layers_to_probe)

        probe_idx = -1  # Default to the last token
        probe_idx_list = None
        if marker_tokens is not None and getattr(marker_tokens, 'size', 0) > 0:
            token_list = tokens.tolist()
            marker_list = marker_tokens.tolist()
            found = False
            # Search for the last occurrence of the marker by searching backwards
            for i in range(len(token_list) - len(marker_list), -1, -1):
                if token_list[i:i + len(marker_list)] == marker_list:
                    # Decide index according to probe_mode
                    if probe_mode == "follow-token":
                        potential_idx = i + len(marker_list)
                        if potential_idx < len(token_list):
                            probe_idx = potential_idx
                        else:
                            # fallback to marker token if marker at end
                            probe_idx = i + len(marker_list) - 1
                    elif probe_mode == "marker-token":
                        probe_idx = i + len(marker_list) - 1
                    elif probe_mode == "thinking-span":
                        # Average activations across a span of tokens after the marker
                        start = i + len(marker_list)
                        if start < len(token_list):
                            end = min(len(token_list), start + probe_span)
                            probe_idx_list = list(range(start, end))
                        else:
                            # marker at end: fallback to marker token
                            probe_idx = i + len(marker_list) - 1
                    elif probe_mode == "last-token":
                        probe_idx = len(token_list) - 1
                    found = True
                    break

            if found:
                marker_found_any = True
                if probe_debug and len(probe_debug_lines) < probe_debug_n:
                    truncated = (prompt[:200] + '...') if len(prompt) > 200 else prompt
                    if probe_idx_list is not None:
                        probe_debug_lines.append(f"probe_idx_list={probe_idx_list}, prompt={truncated}")
                    else:
                        probe_debug_lines.append(f"probe_idx={probe_idx}, prompt={truncated}")
            else:
                # store a small sample for diagnostics (prompt, tokens)
                if len(sample_not_found_examples) < probe_debug_n:
                    try:
                        sample_not_found_examples.append((prompt, token_list))
                    except Exception:
                        pass

        for layer_idx, act in captured.items():
            # decide indices to use (single index or list).
            if probe_idx_list is not None:
                # filter out-of-bounds indices
                valid_idxs = [idx for idx in probe_idx_list if 0 <= idx < act.shape[1]]
                if valid_idxs:
                    probe_act = act[0, valid_idxs, :].mean(axis=0)
                else:
                    probe_act = act[0, -1, :]
            else:
                # ensure probe_idx is within bounds; fallback to last token
                use_idx = probe_idx if (0 <= probe_idx < act.shape[1]) else act.shape[1] - 1
                probe_act = act[0, use_idx, :]
            counts[layer_idx] += 1
            delta = probe_act - mean_activations[layer_idx]
            mean_activations[layer_idx] += delta / counts[layer_idx]
        mx.eval(list(mean_activations.values()))
    if marker_tokens is not None and getattr(marker_tokens, 'size', 0) > 0 and not marker_found_any:
        try:
            marker_list = marker_tokens.tolist()
        except Exception:
            marker_list = None

        message_lines = [f"Probe marker {repr(probe_marker)} not found in any items. Using last token for all examples."]
        message_lines.append(f"Marker token ids: {marker_list}")
        if sample_not_found_examples:
            message_lines.append("Sample prompts (truncated) and token ids where marker was not found:")
            for i, (s_prompt, s_tokens) in enumerate(sample_not_found_examples):
                truncated = (s_prompt[:200] + '...') if len(s_prompt) > 200 else s_prompt
                message_lines.append(f"  [{i+1}] prompt: {truncated}")
                message_lines.append(f"       tokens (len={len(s_tokens)}): {s_tokens[:40]}{'...' if len(s_tokens)>40 else ''}")

        # Show a single UI-visible warning
        gr.Warning("\n".join(message_lines))
        logging.warning("Probe marker not found diagnostic", extra={"extra_info": {"component": "gui", "event": "probe_marker_not_found_diag", "marker": probe_marker, "marker_tokens": marker_list, "sample_count": len(sample_not_found_examples)}})

    # If a probe marker was requested but never found in any example, warn once
    if marker_tokens is not None and getattr(marker_tokens, 'size', 0) > 0 and not marker_found_any:
        try:
            marker_list = marker_tokens.tolist()
        except Exception:
            marker_list = None

        message_lines = [f"Probe marker {repr(probe_marker)} not found in any items. Using last token for all examples."]
        message_lines.append(f"Marker token ids: {marker_list}")
        if sample_not_found_examples:
            message_lines.append("Sample prompts (truncated) and token ids where marker was not found:")
            for i, (s_prompt, s_tokens) in enumerate(sample_not_found_examples):
                truncated = (s_prompt[:200] + '...') if len(s_prompt) > 200 else s_prompt
                message_lines.append(f"  [{i+1}] prompt: {truncated}")
                message_lines.append(f"       tokens (len={len(s_tokens)}): {s_tokens[:40]}{'...' if len(s_tokens)>40 else ''}")

        # Show a single UI-visible warning
        gr.Warning("\n".join(message_lines))
        logging.warning("Probe marker not found diagnostic", extra={"extra_info": {"component": "gui", "event": "probe_marker_not_found_diag", "marker": probe_marker, "marker_tokens": marker_list, "sample_count": len(sample_not_found_examples)}})

    return mean_activations, probe_debug_lines

def _load_maybe_local_json(path_str: str):
    """
    Helper function to load datasets that may be local JSONL files or Hub IDs.
    
    If path_str points to a local .json or .jsonl file, use the datasets
    `json` loader with `data_files` so the file is accepted. Otherwise
    attempt to load it as a normal dataset identifier.
    """
    p = Path(path_str)
    # If it's a local file and looks like JSON/JSONL, use the json loader
    if p.is_file() and p.suffix in (".json", ".jsonl"):
        try:
            ds = load_dataset("json", data_files=str(p))
        except TypeError:
            ds = load_dataset(str(p))
    else:
        try:
            ds = load_dataset(path_str)
        except Exception:
            # Fallback: try treating it as a json file path
            try:
                ds = load_dataset("json", data_files=str(p))
            except TypeError:
                ds = load_dataset(str(p))
    
    # Common case: datasets return a dict with a 'train' split
    if isinstance(ds, dict) and "train" in ds:
        return ds["train"]
    return ds

def run_abliteration_stream(
    model_id: str,
    harmless_id: str,
    harmful_id: str,
    output_dir: str,
    layers_str: str,
    use_layer_idx: int,
    ablation_strength: float,
    probe_marker: str,
    probe_mode: str = "follow-token",
    probe_span: int = 1,
    probe_debug: bool = False,
    probe_debug_n: int = 3,
    probe_debug_full: bool = False,
    ablate_k: int = 1,
    ablate_method: str = "projection",
    refusal_dir_method: str = "difference",
    pca_sample: int = 512,
    cache_dir: str = ".cache",
    verbose: bool = False,
    dump_dequant: bool = False,
    progress=gr.Progress(),
) -> Generator[Tuple[str, None] | Tuple[str, str], None, None]:
    """
    Runs the abliteration process and yields logs for the Gradio UI.

    This generator function performs the end-to-end abliteration workflow,
    yielding status updates that are displayed in the Gradio log component.

    Args:
        model_id (str): The path or Hugging Face Hub ID of the model.
        harmless_id (str): The path or Hub ID of the harmless dataset.
        harmful_id (str): The path or Hub ID of the harmful dataset.
        output_dir (str): The name of the directory to save the abliterated model.
        layers_str (str): A string specifying the layers to probe (e.g., "all" or "15,16").
        use_layer_idx (int): The index of the layer to use for the refusal vector.
        progress (gr.Progress): A Gradio Progress object to update the UI.

    Yields:
        probe_mode: str = "follow-token",
        A tuple containing the updated log history and an optional output file path.

    Raises:
        gr.Error: If a user-facing error occurs (e.g., invalid inputs, file not found).
    """
    log_history = ""
    def log_and_yield(message: str, extra_info: Dict = None) -> str:
        """Helper to format, log, and yield a message."""
        nonlocal log_history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        log_history += full_message + "\n"

        log_extra = {"component": "gui"}
        if extra_info:
            log_extra.update(extra_info)
        logging.info(message, extra={"extra_info": log_extra})

        return log_history

    try:
        # configure logging verbosity for this run
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Verbose logging enabled for GUI run")
        yield log_and_yield("Validating Inputs", {"event": "validation_start"}), None
        if not all([model_id, harmless_id, harmful_id, output_dir]):
            raise gr.Error("All 'Required Inputs' fields must be filled.")
        yield log_and_yield("Inputs validated", {"event": "validation_end"}), None

        yield log_and_yield("Resolving Assets", {"event": "asset_resolution_start", "inputs": {"model_id": model_id, "harmless_id": harmless_id, "harmful_id": harmful_id}}), None
        output_path = Path.cwd() / "outputs" / Path(output_dir).name
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = resolve_asset(model_id, "models", cache_dir)
        harmless_ds_path = str(resolve_asset(harmless_id, "datasets", cache_dir))
        harmful_ds_path = str(resolve_asset(harmful_id, "datasets", cache_dir))
        yield log_and_yield(f"Assets resolved. Output will be saved to: {output_path.resolve()}", {"event": "asset_resolution_end", "actual_output": {"output_path": str(output_path.resolve())}}), None

        yield log_and_yield("Loading Model and Tokenizer", {"event": "loading_start"}), None
        model, tokenizer = mlx_lm.load(str(model_path))
        yield log_and_yield("Model and tokenizer loaded successfully.", {"event": "loading_end"}), None

        # Determine the probe marker with fallback logic
        final_probe_marker = probe_marker
        if not final_probe_marker or not final_probe_marker.strip():
            yield log_and_yield("No probe marker provided by user. Checking tokenizer config...", {"event": "probe_marker_fallback_start"}), None
            tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
            if tokenizer_config_path.is_file():
                with open(tokenizer_config_path, "r") as f:
                    tokenizer_config = json.load(f)
                chat_template = tokenizer_config.get("chat_template")
                if chat_template:
                    found_marker = extract_eot_from_chat_template(chat_template)
                    if found_marker:
                        final_probe_marker = found_marker
                        yield log_and_yield(f"Found probe marker '{found_marker}' in chat_template.", {"event": "probe_marker_found_in_config", "actual_output": {"marker": found_marker}}), None

        if not final_probe_marker or not final_probe_marker.strip():
            yield log_and_yield("No probe marker found. Defaulting to last token.", {"event": "probe_marker_fallback_end"}), None
            final_probe_marker = None


        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Could not find 'config.json' in the model directory: {model_path}")
        with open(config_path, "r") as f:
            model_config = json.load(f)

        num_layers = model_config["num_hidden_layers"]
        yield log_and_yield(f"Model '{model_id}' loaded with {num_layers} layers.", {"event": "model_info", "actual_output": {"num_layers": num_layers}}), None

        harmless_dataset = _load_maybe_local_json(harmless_ds_path)
        harmful_dataset = _load_maybe_local_json(harmful_ds_path)

        yield log_and_yield("Probing Activations", {"event": "probing_start"}), None
        layers_to_probe = list(range(num_layers)) if layers_str.lower() == 'all' else [int(x.strip()) for x in layers_str.split(",")]
        wrapper = ActivationProbeWrapper(model)

        harmful_mean_activations, harmful_debug = get_mean_activations_from_dataset(
            harmful_dataset,
            wrapper,
            tokenizer,
            layers_to_probe,
            model_config,
            "Probing harmful prompts",
            progress,
            final_probe_marker,
            probe_mode=probe_mode,
            probe_span=probe_span,
            probe_debug=probe_debug,
            probe_debug_n=probe_debug_n,
            probe_debug_full=probe_debug_full,
        )
        harmless_mean_activations, harmless_debug = get_mean_activations_from_dataset(
            harmless_dataset,
            wrapper,
            tokenizer,
            layers_to_probe,
            model_config,
            "Probing harmless prompts",
            progress,
            final_probe_marker,
            probe_mode=probe_mode,
            probe_span=probe_span,
            probe_debug=probe_debug,
            probe_debug_n=probe_debug_n,
            probe_debug_full=probe_debug_full,
        )

        # If probe_debug was requested, print the collected per-example probe indices
        if probe_debug:
            debug_lines = []
            debug_lines.extend([f"HARMFUL: {line}" for line in harmful_debug])
            debug_lines.extend([f"HARMLESS: {line}" for line in harmless_debug])
            for debug_line in debug_lines:
                yield log_and_yield(debug_line, {"event": "probe_debug_line"}), None

        yield log_and_yield("Activation probing complete.", {"event": "probing_end"}), None

        yield log_and_yield("Computing Refusal Vector", {"event": "vector_computation_start"}), None
        actual_use_layer = use_layer_idx if use_layer_idx >= 0 else num_layers + use_layer_idx

        # If ablate_k > 1, compute PCA components similarly to the CLI path
        if ablate_k and ablate_k > 1:
            import numpy as _np

            def collect_per_example_means(dataset, max_samples: int = 512):
                res = []
                collected = 0
                if final_probe_marker and final_probe_marker.strip():
                    marker_tokens = mx.array(tokenizer.encode(final_probe_marker, add_special_tokens=False))
                    try:
                        marker_list = marker_tokens.tolist()
                    except Exception:
                        marker_list = None
                else:
                    marker_tokens = None
                    marker_list = None

                for item in dataset:
                    if collected >= max_samples:
                        break
                    prompt = item.get("prompt") or item.get("text")
                    if not prompt:
                        continue
                    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
                    if len(tokens) > model_config.get("max_position_embeddings", 4096):
                        tokens = tokens[: model_config.get("max_position_embeddings", 4096)]

                    _, cap = wrapper(tokens[None], mask=None, layers_to_probe=[actual_use_layer])
                    arr = cap.get(actual_use_layer)
                    if arr is None:
                        continue

                    probe_idx = -1
                    probe_idx_list = None
                    if marker_list:
                        token_list = tokens.tolist()
                        for i in range(len(token_list) - len(marker_list), -1, -1):
                            if token_list[i:i + len(marker_list)] == marker_list:
                                if probe_mode == "follow-token":
                                    potential_idx = i + len(marker_list)
                                    probe_idx = potential_idx if potential_idx < len(token_list) else i + len(marker_list) - 1
                                elif probe_mode == "marker-token":
                                    probe_idx = i + len(marker_list) - 1
                                elif probe_mode == "thinking-span":
                                    start = i + len(marker_list)
                                    if start < len(token_list):
                                        end = min(len(token_list), start + probe_span)
                                        probe_idx_list = list(range(start, end))
                                    else:
                                        probe_idx = i + len(marker_list) - 1
                                elif probe_mode == "last-token":
                                    probe_idx = len(token_list) - 1
                                break

                    if probe_idx_list is not None:
                        valid_idxs = [idx for idx in probe_idx_list if 0 <= idx < arr.shape[1]]
                        if valid_idxs:
                            vec = arr[0, valid_idxs, :].mean(axis=0)
                        else:
                            vec = arr[0, -1, :]
                    else:
                        use_idx = probe_idx if (0 <= probe_idx < arr.shape[1]) else arr.shape[1] - 1
                        vec = arr[0, use_idx, :]

                    res.append(_np.asarray(vec))
                    collected += 1

                if not res:
                    raise RuntimeError("Could not collect per-example activations for PCA")
                return _np.stack(res, axis=0)

            harm_mat = collect_per_example_means(harmful_dataset, max_samples=pca_sample)
            harm_mat_mean = harm_mat.mean(axis=0)
            harm_centered = harm_mat - harm_mat_mean
            harm_u, harm_s, harm_vt = _np.linalg.svd(harm_centered, full_matrices=False)

            harm_components = harm_vt[: ablate_k]

            harmless_mat = collect_per_example_means(harmless_dataset, max_samples=pca_sample)
            harmless_mat_mean = harmless_mat.mean(axis=0)
            harmless_centered = harmless_mat - harmless_mat_mean
            harmless_u, harmless_s, harmless_vt = _np.linalg.svd(harmless_centered, full_matrices=False)

            pc_vecs = _np.array(harm_components)
            import mlx.core as _mx

            refusal_vector = _mx.array(pc_vecs)
        else:
            refusal_vector = calculate_refusal_direction(
                harmful_mean_activations[actual_use_layer], 
                harmless_mean_activations[actual_use_layer],
                method=refusal_dir_method
            )
        yield log_and_yield(f"Refusal vector computed from layer {actual_use_layer}.", {"event": "vector_computation_end", "inputs": {"use_layer": actual_use_layer}, "actual_output": {"refusal_vector_norm": float(mx.linalg.norm(refusal_vector).item())}}), None

        yield log_and_yield("Orthogonalizing Weights & Updating Model", {"event": "orthogonalization_start"}), None
        # Snapshot 'before' parameters so we can detect whether model.update
        # actually applied changes in-memory.
        try:
            try:
                before = dict(tree_flatten(model.parameters()))
            except Exception:
                before = {}
        except Exception:
            before = {}

        ablated_params = get_ablated_parameters(model, refusal_vector, ablation_strength=ablation_strength)
        model.update(ablated_params)

        # Now compute diffs between the before snapshot and the current
        # model.parameters() after the update.
        try:
            try:
                after = dict(tree_flatten(model.parameters()))
            except Exception:
                after = {}

            # If we can compute a diff, log per-key norms for keys reported in 'ablated_params'
            for k in (ablated_params.keys() if isinstance(ablated_params, dict) else []):
                orig_v = before.get(k)
                new_v = after.get(k)
                if orig_v is None:
                    logging.info(f"Post-update check: original parameter not found for key {k}", extra={"extra_info": {"event": "post_update_missing_orig", "inputs": {"key": k}}})
                    continue
                if new_v is None:
                    logging.info(f"Post-update check: updated parameter not found for key {k}", extra={"extra_info": {"event": "post_update_missing_new", "inputs": {"key": k}}})
                    continue

                try:
                    diff = mx.array(orig_v) - mx.array(new_v)
                    max_abs = float(mx.linalg.norm(diff).item())
                except Exception:
                    import numpy as _np

                    try:
                        max_abs = float(_np.linalg.norm(_np.array(orig_v) - _np.array(new_v)))
                    except Exception:
                        max_abs = None

                logging.info(f"Post-update in-memory diff for {k}: {max_abs}", extra={"extra_info": {"event": "post_update_diff", "inputs": {"key": k}, "actual_output": {"max_abs_diff": max_abs}}})
        except Exception:
            logging.debug("Could not compute post-update in-memory diffs", exc_info=True)

        mx.eval(model.parameters())
        yield log_and_yield("Model weights have been updated.", {"event": "orthogonalization_end"}), None

        yield log_and_yield("Saving Abliterated Model", {"event": "saving_start"}), None
        abliteration_log = {
            "source_model": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        save_ablated_model(
            str(output_path), model, tokenizer, model_config, abliteration_log, source_model_path=str(model_path), dump_dequant=bool(dump_dequant)
        )

        # Determine the correct output path to display (directory for sharded, file for single)
        output_index_path = output_path / "model.safetensors.index.json"
        if output_index_path.is_file():
            # For sharded models, return the path to the index file itself
            final_output_path = str(output_index_path.resolve())
        else:
            # For single-file models, return the path to the safetensors file
            final_output_path = str((output_path / "model.safetensors").resolve())

        yield log_and_yield("✅ Abliteration process completed successfully.", {"event": "main_success", "actual_output": {"output_path": final_output_path}}), final_output_path

    except Exception as e:
        logging.error("An error occurred during abliteration", extra={"extra_info": {"component": "gui", "event": "main_error", "error_message": str(e)}}, exc_info=True)
        yield log_and_yield(f"❌ An error occurred: {e}", {"event": "main_error", "error_message": str(e)}), None
        raise gr.Error(str(e))

def create_ui() -> gr.Blocks:
    """
    Creates the Gradio user interface for the Abliteration Toolkit.

    Returns:
        gr.Blocks: The Gradio interface object.
    """
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="MLX Abliteration Toolkit") as demo:
        gr.Markdown("# ✂️ MLX Abliteration Toolkit")
        gr.Markdown("An interactive tool to perform mechanistic interpretability-driven model surgery on MLX models.")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Required Inputs"):
                        model_input = gr.Textbox(label="Base Model Path or Hub ID", placeholder="e.g., mlx-community/Phi-3-mini-4k-instruct-4bit-mlx")
                        harmless_ds_input = gr.Textbox(label="Harmless Dataset Path or Hub ID", value="mlabonne/harmless_alpaca")
                        harmful_ds_input = gr.Textbox(label="Harmful Dataset Path or Hub ID", value="mlabonne/harmful_behaviors")
                        output_dir_input = gr.Textbox(label="Output Directory Name", placeholder="e.g., ablated-phi-3-mini")
                    with gr.TabItem("Advanced Parameters"):
                        layers_input = gr.Textbox(label="Layers to Probe", value="all", info="A comma-separated list of layer indices or 'all'.")
                        use_layer_slider = gr.Slider(minimum=-48, maximum=47, step=1, value=-1, label="Use Refusal Vector from Layer", info="The layer index for the refusal vector. Negative values count from the end.")
                        strength_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=0.75, label="Ablation Strength", info="Strength of ablation. Recommended: 0.5-1.5. Start with 0.75. Higher values may cause instability.")
                        probe_marker_input = gr.Code(label="Probe Marker", language=None, lines=1)
                        probe_mode_input = gr.Dropdown(label="Probe Mode", choices=["follow-token", "marker-token", "last-token", "thinking-span"], value="follow-token", info="How to select the probe token when a marker is found.")
                        probe_span_input = gr.Number(label="Probe Span", value=1, precision=0, info="Number of tokens to average after the probe marker when using 'thinking-span'. Increase to 2-4 if the transition tokenizes into multiple tokens.")
                        probe_debug_checkbox = gr.Checkbox(label="Probe Debug", value=False, info="Emit per-example probe index debug lines into the log.")
                        probe_debug_n_input = gr.Number(label="Probe Debug N", value=3, precision=0)
                        probe_debug_full_checkbox = gr.Checkbox(label="Probe Debug Full Tokens", value=False, info="When debug enabled, show token strings if available.")
                        gr.Markdown("---")
                        ablate_k_input = gr.Number(label="Ablate k (top components)", value=1, precision=0, info="Number of top PCA components to ablate. 1 = single vector.")
                        ablate_method_input = gr.Dropdown(label="Ablation Method", choices=["projection", "sequential"], value="projection", info="Method used to ablate components.")
                        refusal_dir_method_input = gr.Dropdown(label="Refusal Direction Method", choices=["difference", "projected"], value="difference", info="Method to calculate refusal direction: 'difference' = simple harmful-harmless; 'projected' = remove harmless component.")
                        pca_sample_input = gr.Number(label="PCA Sample", value=512, precision=0, info="Max per-example samples to collect for PCA when ablate-k > 1")
                        cache_dir_input = gr.Textbox(label="Cache Directory", value=".cache", info="Directory used for downloads and caching.")
                        verbose_checkbox = gr.Checkbox(label="Verbose Logging", value=False, info="Enable verbose (debug) logging for the GUI process.")
                        dump_dequant_checkbox = gr.Checkbox(label="Dump Dequantized .npy", value=False, info="Write dequantized numpy dumps for ablated tensors into the output directory (debug only).")
                    with gr.TabItem("Dry-run Report"):
                        report_file = gr.File(label="Dry-run Suggestions JSON (outputs/diag_out/dry_run_suggestions.json)", interactive=True)
                        report_table = gr.Dataframe(headers=["layer", "diff_norm"], datatype=["number", "number"], interactive=False, label="Per-layer diff norms")
                        recommended_list = gr.Textbox(label="Recommended Layers (top-10)")
                        apply_button = gr.Button("Apply recommended layers")

                        def load_report_to_table(_):
                            try:
                                p = Path.cwd() / "outputs" / "diag_out" / "dry_run_suggestions.json"
                                if not p.is_file():
                                    return None, "No suggestions report found. Run a dry-run first."
                                data = json.loads(p.read_text())
                                layer_stats = data.get("layer_stats", [])
                                rows = []
                                for s in layer_stats:
                                    rows.append([s.get("layer"), s.get("diff_norm")])
                                rec = data.get("recommended_layers_top10", [])
                                return rows, ",".join([str(x) for x in rec])
                            except Exception as e:
                                return None, f"Error reading report: {e}"

                        def apply_recommended_layers(recommended_str):
                            if not recommended_str:
                                return None, None
                            # recommended_str expected as comma-separated or single-value list
                            parts = [p.strip() for p in recommended_str.split(",") if p.strip()]
                            if not parts:
                                return None, None
                            # Set Layers to probe as the comma-separated recommended list
                            layers_value = ",".join(parts)
                            # Set use_layer to first recommended layer
                            try:
                                use_layer_val = int(parts[0])
                            except Exception:
                                use_layer_val = -1
                            return layers_value, use_layer_val

                        report_file.change(load_report_to_table, inputs=[report_file], outputs=[report_table, recommended_list])
                        apply_button.click(apply_recommended_layers, inputs=[recommended_list], outputs=[layers_input, use_layer_slider])
                start_button = gr.Button("Start Abliteration", variant="primary", scale=1)
            with gr.Column(scale=3):
                log_output = gr.Textbox(label="Process Log", lines=20, interactive=False, autoscroll=True)
                output_file_display = gr.File(label="Abliterated Model Path", interactive=False)
        inputs = [
            model_input,
            harmless_ds_input,
            harmful_ds_input,
            output_dir_input,
            layers_input,
            use_layer_slider,
            strength_slider,
            probe_marker_input,
            probe_mode_input,
            probe_span_input,
            probe_debug_checkbox,
            probe_debug_n_input,
            probe_debug_full_checkbox,
            ablate_k_input,
            ablate_method_input,
            refusal_dir_method_input,
            pca_sample_input,
            cache_dir_input,
            verbose_checkbox,
            dump_dequant_checkbox,
        ]
        for inp in inputs:
            inp.change(lambda: (None, None), outputs=[log_output, output_file_display])
        start_button.click(fn=run_abliteration_stream, inputs=inputs, outputs=[log_output, output_file_display])
    return demo

if __name__ == "__main__":
    setup_structured_logging("abliteration-toolkit-gui", logging.INFO)
    ui = create_ui()
    ui.launch(share=False)
