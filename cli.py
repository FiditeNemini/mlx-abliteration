"""Command-line interface for the MLX Abliteration Toolkit.

This script provides a CLI for running the abliteration process, which involves
identifying and neutralizing specific behaviors in a language model.

Workflow:
1.  Parse command-line arguments.
2.  Set up structured logging.
3.  Resolve model and dataset assets (downloading from Hugging Face Hub if necessary).
4.  Load the model and datasets.
5.  Probe the model's activations on harmless and harmful prompts.
6.  Calculate the refusal direction vector.
7.  Orthogonalize the model's weights to ablate the refusal behavior.
8.  Save the abliterated model.

Dependencies:
- torch
- mlx
- mlx-lm
- datasets
- tqdm
- huggingface-hub
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import mlx.core as mx
from tqdm import tqdm
import mlx_lm

# Add project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.asset_resolver import resolve_asset
from core.abliteration import (
    ActivationProbeWrapper,
    calculate_refusal_direction,
    get_ablated_parameters,
    save_ablated_model,
)
from core.logging_config import setup_structured_logging
from core.utils import extract_eot_from_chat_template, tokenizer_marker_diff, normalize_marker

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="MLX Abliteration Toolkit CLI")
    model_group = parser.add_argument_group("Model and Dataset Inputs")
    model_group.add_argument("-m", "--model", type=str, required=True, help="Path or Hub ID of the MLX model.")
    model_group.add_argument("-hd", "--harmless-dataset", type=str, default="mlabonne/harmless_alpaca", help="Harmless prompts dataset.")
    model_group.add_argument("-ad", "--harmful-dataset", type=str, default="mlabonne/harmful_behaviors", help="Harmful prompts dataset.")
    abl_group = parser.add_argument_group("Abliteration Parameters")
    abl_group.add_argument("-l", "--layers", type=str, default="all", help="Layers to probe: 'all' or comma-separated list (e.g., '15,16').")
    abl_group.add_argument("-u", "--use-layer", type=int, default=-1, help="Layer index for the refusal vector. Default: -1 (last layer).")
    abl_group.add_argument("-s", "--ablation-strength", type=float, default=1.0, help="Strength of the ablation effect.")
    abl_group.add_argument("--probe-marker", type=str, default=None, help="String marker for precise activation probing (e.g., '</thinking>').")
    abl_group.add_argument("--probe-mode", type=str, default="follow-token", choices=["follow-token", "marker-token", "last-token", "thinking-span"], help="How to select the probe token when a marker is found: 'follow-token' (token after marker), 'marker-token' (the marker token itself), 'last-token' (always use last token), or 'thinking-span' (average a small span after the marker).")
    abl_group.add_argument("--probe-span", type=int, default=1, help="Number of tokens to average after the probe marker when using 'thinking-span' probe mode. Defaults to 1 (single token).")
    abl_group.add_argument("--ablate-k", type=int, default=1, help="Number of principal components to ablate (1 = single vector).")
    abl_group.add_argument("--ablate-method", type=str, default="projection", choices=["projection", "sequential"], help="Method used to ablate components: 'projection' builds a projection matrix and removes the subspace in one step; 'sequential' subtracts projections component-by-component.")
    abl_group.add_argument("--pca-sample", type=int, default=512, help="Maximum number of per-example activations to collect for PCA when --ablate-k > 1.")
    abl_group.add_argument("--probe-debug", action="store_true", help="Enable probe debug output (dump tokenized prompts for first N examples).")
    abl_group.add_argument("--probe-debug-n", type=int, default=3, help="Number of sample prompts to dump when --probe-debug is set.")
    abl_group.add_argument("--probe-debug-full", action="store_true", help="When used with --probe-debug, also show token strings (if tokenizer supports convert_ids_to_tokens).")
    abl_group.add_argument("--strip-marker-newline", action="store_true", help="If set, strip a single trailing newline from extracted probe markers before tokenization.")
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save the abliterated model.")
    output_group.add_argument("--cache-dir", type=str, default=".cache", help="Cache directory for downloads.")
    output_group.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    output_group.add_argument("--dump-dequant", action="store_true", help="Write dequantized .npy dumps for ablated tensors into the output directory (debug).")
    return parser.parse_args()

def parse_layers(layers_str: str, num_model_layers: int) -> List[int]:
    """Parses the --layers argument string into a list of layer indices.

    Args:
        layers_str (str): The string from the --layers argument.
        num_model_layers (int): The total number of layers in the model.

    Returns:
        List[int]: A list of layer indices to probe.

    Raises:
        ValueError: If the layers string is not in a valid format.
    """
    if layers_str.lower() == "all":
        return list(range(num_model_layers))
    try:
        return [int(x.strip()) for x in layers_str.split(",")]
    except ValueError as e:
        raise ValueError(f"Invalid format for --layers: {e}") from e

def get_mean_activations(
    dataset,
    wrapper: ActivationProbeWrapper,
    tokenizer: Any,
    layers_to_probe: List[int],
    config: Dict,
    desc: str,
    probe_marker: Optional[str] = None,
    probe_debug: bool = False,
    probe_debug_n: int = 3,
    probe_debug_full: bool = False,
    probe_mode: str = "follow-token",
    probe_span: int = 1,
) -> Dict[int, mx.array]:
    """Computes mean activations for a given dataset using Welford's algorithm.

    If a `probe_marker` is provided, it finds the marker in the tokenized
    prompt and uses the activation of the token immediately following it.
    Otherwise, it defaults to using the activation of the last token.

    Args:
        dataset: The dataset to process.
        wrapper (ActivationProbeWrapper): The model wrapper for probing.
        tokenizer (Any): The tokenizer.
        layers_to_probe (List[int]): A list of layer indices to probe.
        config (Dict): The model's configuration dictionary.
        desc (str): A description for the progress bar.
        probe_marker (Optional[str]): A string marker to find for probing.

    Returns:
        Dict[int, mx.array]: A dictionary mapping layer indices to mean activations.
    """
    hidden_size = config["hidden_size"]
    mean_activations = {layer: mx.zeros(hidden_size) for layer in layers_to_probe}
    counts = {layer: 0 for layer in layers_to_probe}
    max_seq_len = config.get("max_position_embeddings", 4096)

    if probe_marker and probe_marker.strip():
        marker_tokens = mx.array(tokenizer.encode(probe_marker, add_special_tokens=False))
    else:
        marker_tokens = None

    # Track whether the marker was ever found in the dataset to avoid noisy per-item warnings
    marker_found_any = False
    # collect up to probe_debug_n sample prompts where marker wasn't found for diagnostics
    sample_not_found_examples: list[tuple[str, list]] = []
    # collect up to probe_debug_n tokenized samples to dump when probe_debug enabled
    debug_tokenized_samples: list[tuple[str, list]] = []

    for item in tqdm(dataset, desc=desc):
        prompt = item.get("prompt") or item.get("text")
        if not prompt:
            tqdm.write("Skipping empty prompt.")
            continue

        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        _, captured = wrapper(tokens[None], mask=None, layers_to_probe=layers_to_probe)

        probe_idx = -1  # Default to the last token
        probe_idx_list = None
        if marker_tokens is not None and getattr(marker_tokens, 'size', 0) > 0:
            token_list = tokens.tolist()
            try:
                marker_list = marker_tokens.tolist()
            except Exception:
                marker_list = None

            if marker_list:
                # Search for the last occurrence of the marker by searching backwards
                found_marker = False
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
                        found_marker = True
                        break

                if found_marker:
                    marker_found_any = True
                else:
                    if len(sample_not_found_examples) < probe_debug_n:
                        try:
                            sample_not_found_examples.append((prompt, token_list))
                        except Exception:
                            pass

                # If probe_debug is enabled, collect a few tokenized samples for inspection
                if probe_debug and len(debug_tokenized_samples) < probe_debug_n:
                    try:
                        token_strs = None
                        if hasattr(tokenizer, 'convert_ids_to_tokens'):
                            try:
                                token_strs = tokenizer.convert_ids_to_tokens(token_list)
                            except Exception:
                                token_strs = None
                        debug_tokenized_samples.append((prompt, token_list if token_strs is None else token_strs))
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

    # If a probe marker was requested but never found in any example, warn once
    if marker_tokens is not None and getattr(marker_tokens, 'size', 0) > 0 and not marker_found_any:
        try:
            marker_list = marker_tokens.tolist()
        except Exception:
            marker_list = None

        diag_lines = [f"Warning: Probe marker {repr(probe_marker)} not found in any items. Using last token for all examples."]
        diag_lines.append(f"Marker token ids: {marker_list}")
        if sample_not_found_examples:
            diag_lines.append("Sample prompts (truncated) and token ids where marker was not found:")
            for i, (s_prompt, s_tokens) in enumerate(sample_not_found_examples):
                truncated = (s_prompt[:200] + '...') if len(s_prompt) > 200 else s_prompt
                diag_lines.append(f"  [{i+1}] prompt: {truncated}")
                diag_lines.append(f"       tokens (len={len(s_tokens)}): {s_tokens[:40]}{'...' if len(s_tokens)>40 else ''}")

        for line in diag_lines:
            tqdm.write(line)
        logging.warning("Probe marker not found diagnostic", extra={"extra_info": {"component": "cli", "event": "probe_marker_not_found_diag", "marker": probe_marker, "marker_tokens": marker_list, "sample_count": len(sample_not_found_examples)}})

    # If probe_debug is enabled, print the debug tokenization samples
    if probe_debug and debug_tokenized_samples:
        tqdm.write("Probe debug samples (first {}):".format(len(debug_tokenized_samples)))
        for i, (s_prompt, toks) in enumerate(debug_tokenized_samples):
            truncated = (s_prompt[:200] + '...') if len(s_prompt) > 200 else s_prompt
            tqdm.write(f"  [{i+1}] prompt: {truncated}")
            # toks may be token ids or token strings depending on tokenizer support
            if probe_debug_full and isinstance(toks, (list, tuple)) and toks and isinstance(toks[0], str):
                # already token strings
                toks_display = toks
            else:
                toks_display = toks[:80] if isinstance(toks, (list, tuple)) else toks
            toks_len = len(toks) if hasattr(toks, '__len__') else 'unknown'
            tqdm.write(f"       tokens/count: {toks_display}{'...' if isinstance(toks, (list, tuple)) and len(toks)>80 else ''} (len={toks_len})")
        logging.info("Probe debug samples emitted", extra={"extra_info": {"component": "cli", "event": "probe_debug_samples", "count": len(debug_tokenized_samples)}})

    try:
        diff = tokenizer_marker_diff(tokenizer, probe_marker) if probe_marker else None
        if diff is not None:
            tqdm.write(f"Tokenization of marker {repr(probe_marker)}: ids={diff.get('ids')}, tokens={diff.get('tokens')}")
            logging.info("Marker tokenization diff", extra={"extra_info": {"component": "cli", "event": "marker_tokenization_diff", "marker": probe_marker, "marker_ids": diff.get('ids'), "marker_tokens": diff.get('tokens')}})
    except Exception:
        pass

    return mean_activations


def run_abliteration(args: argparse.Namespace):
    """Runs the main abliteration process.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    logging.info("Resolving assets", extra={"extra_info": {"component": "cli", "event": "asset_resolution_start", "inputs": {"model": args.model, "harmless_dataset": args.harmless_dataset, "harmful_dataset": args.harmful_dataset}}})
    model_path = resolve_asset(args.model, "models", args.cache_dir)
    harmless_ds_path = str(resolve_asset(args.harmless_dataset, "datasets", args.cache_dir))
    harmful_ds_path = str(resolve_asset(args.harmful_dataset, "datasets", args.cache_dir))
    logging.info("Assets resolved", extra={"extra_info": {"component": "cli", "event": "asset_resolution_end"}})

    logging.info("Loading model and datasets", extra={"extra_info": {"component": "cli", "event": "loading_start"}})
    model, tokenizer = mlx_lm.load(str(model_path))

    # Determine the probe marker with fallback logic
    final_probe_marker = args.probe_marker
    if not final_probe_marker or not final_probe_marker.strip():
        logging.info("No probe marker provided by user. Checking tokenizer config...", extra={"extra_info": {"component": "cli", "event": "probe_marker_fallback_start"}})
        tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            with open(tokenizer_config_path, "r") as f:
                tokenizer_config = json.load(f)
            chat_template = tokenizer_config.get("chat_template")
            if chat_template:
                found_marker = extract_eot_from_chat_template(chat_template)
                if found_marker:
                    final_probe_marker = found_marker
                    logging.info(f"Found probe marker '{found_marker}' in chat_template.", extra={"extra_info": {"component": "cli", "event": "probe_marker_found_in_config", "actual_output": {"marker": found_marker}}})

    if not final_probe_marker or not final_probe_marker.strip():
        logging.info("No probe marker found. Defaulting to last token.", extra={"extra_info": {"component": "cli", "event": "probe_marker_fallback_end"}})
        final_probe_marker = None

    # Optionally normalize the marker (strip a single trailing newline) to
    # accommodate tokenizers that don't include trailing newlines in marker
    # tokenization.
    try:
        final_probe_marker = normalize_marker(final_probe_marker, strip_trailing_newline=getattr(args, "strip_marker_newline", False))
    except Exception:
        # Normalization is best-effort; fall back to the raw marker if it fails.
        pass

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find 'config.json' in the model directory: {model_path}")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    from datasets import load_dataset

    def _load_maybe_local_json(path_str: str):
        p = Path(path_str)
        # Try the simplest call first (some tests provide a fake load_dataset
        # that expects a single path argument). If that fails, fall back to
        # calling the json loader with data_files (the normal HF pattern).
        try:
            ds = load_dataset(str(p))
        except TypeError:
            # test fakes may not accept kwargs; try loader+data_files signature
            ds = load_dataset("json", data_files=str(p))
        except Exception:
            # As a last resort, attempt to call with the original string
            try:
                ds = load_dataset(path_str)
            except Exception:
                ds = load_dataset("json", data_files=str(p))

        if isinstance(ds, dict) and "train" in ds:
            return ds["train"]
        return ds

    harmless_dataset = _load_maybe_local_json(harmless_ds_path)
    harmful_dataset = _load_maybe_local_json(harmful_ds_path)
    num_layers = model_config["num_hidden_layers"]
    logging.info(f"Model loaded with {num_layers} layers.", extra={"extra_info": {"component": "cli", "event": "loading_end", "actual_output": {"num_layers": num_layers}}})

    logging.info("Probing activations", extra={"extra_info": {"component": "cli", "event": "probing_start"}})
    layers_to_probe = parse_layers(args.layers, num_layers)
    wrapper = ActivationProbeWrapper(model)
    harmful_activations = get_mean_activations(
        harmful_dataset,
        wrapper,
        tokenizer,
        layers_to_probe,
        model_config,
        "Probing harmful prompts",
        final_probe_marker,
        probe_debug=args.probe_debug,
        probe_debug_n=args.probe_debug_n,
        probe_mode=args.probe_mode,
        probe_span=args.probe_span,
    )
    harmless_activations = get_mean_activations(
        harmless_dataset,
        wrapper,
        tokenizer,
        layers_to_probe,
        model_config,
        "Probing harmless prompts",
        final_probe_marker,
        probe_debug=args.probe_debug,
        probe_debug_n=args.probe_debug_n,
        probe_debug_full=args.probe_debug_full,
        probe_mode=args.probe_mode,
        probe_span=args.probe_span,
    )
    logging.info("Activation probing complete", extra={"extra_info": {"component": "cli", "event": "probing_end"}})

    logging.info("Computing refusal vector", extra={"extra_info": {"component": "cli", "event": "vector_computation_start"}})
    use_layer_idx = args.use_layer if args.use_layer >= 0 else num_layers + args.use_layer
    if use_layer_idx not in layers_to_probe:
        raise ValueError(f"Layer {use_layer_idx} was not in the list of probed layers.")

    logging.info(
        f"Using activations from layer {use_layer_idx} for refusal direction.",
        extra={
            "extra_info": {
                "component": "cli",
                "event": "vector_computation_info",
                "inputs": {"use_layer": use_layer_idx},
            }
        },
    )

    if args.ablate_k and args.ablate_k > 1:
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

                _, cap = wrapper(tokens[None], mask=None, layers_to_probe=[use_layer_idx])
                arr = cap.get(use_layer_idx)
                if arr is None:
                    continue

                probe_idx = -1
                probe_idx_list = None
                if marker_list:
                    token_list = tokens.tolist()
                    for i in range(len(token_list) - len(marker_list), -1, -1):
                        if token_list[i:i + len(marker_list)] == marker_list:
                            if args.probe_mode == "follow-token":
                                potential_idx = i + len(marker_list)
                                probe_idx = potential_idx if potential_idx < len(token_list) else i + len(marker_list) - 1
                            elif args.probe_mode == "marker-token":
                                probe_idx = i + len(marker_list) - 1
                            elif args.probe_mode == "thinking-span":
                                start = i + len(marker_list)
                                if start < len(token_list):
                                    end = min(len(token_list), start + args.probe_span)
                                    probe_idx_list = list(range(start, end))
                                else:
                                    probe_idx = i + len(marker_list) - 1
                            elif args.probe_mode == "last-token":
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

                res.append(_np.array(vec))
                collected += 1

            if not res:
                raise RuntimeError("Could not collect per-example activations for PCA")
            return _np.stack(res, axis=0)

        harm_mat = collect_per_example_means(harmful_dataset, max_samples=args.pca_sample)
        harm_mat_mean = harm_mat.mean(axis=0)
        harm_centered = harm_mat - harm_mat_mean
        harm_u, harm_s, harm_vt = _np.linalg.svd(harm_centered, full_matrices=False)

        harm_components = harm_vt[: args.ablate_k]

        harmless_mat = collect_per_example_means(harmless_dataset, max_samples=args.pca_sample)
        harmless_mat_mean = harmless_mat.mean(axis=0)
        harmless_centered = harmless_mat - harmless_mat_mean
        harmless_u, harmless_s, harmless_vt = _np.linalg.svd(harmless_centered, full_matrices=False)

        pc_vecs = _np.array(harm_components)
        import mlx.core as _mx

        refusal_vector = _mx.array(pc_vecs)
    else:
        refusal_vector = calculate_refusal_direction(
            harmful_activations[use_layer_idx],
            harmless_activations[use_layer_idx]
        )

    try:
        norm_val = mx.linalg.norm(refusal_vector)
        try:
            norm_float = float(norm_val.item())
        except Exception:
            norm_float = float(norm_val)
    except Exception:
        try:
            norm_float = float(refusal_vector)
        except Exception:
            norm_float = 0.0

    logging.info("Refusal vector computed", extra={"extra_info": {"component": "cli", "event": "vector_computation_end", "actual_output": {"refusal_vector_norm": norm_float}}})

    logging.info("Orthogonalizing weights and updating model", extra={"extra_info": {"component": "cli", "event": "orthogonalization_start"}})
    ablated_params = get_ablated_parameters(model, refusal_vector, ablation_strength=args.ablation_strength, ablation_method=args.ablate_method)
    model.update(ablated_params)
    mx.eval(model.parameters())
    logging.info("Model parameters updated", extra={"extra_info": {"component": "cli", "event": "orthogonalization_end"}})

    logging.info("Saving abliterated model", extra={"extra_info": {"component": "cli", "event": "saving_start"}})
    abliteration_log = {
        "source_model": args.model,
        "harmless_dataset": args.harmless_dataset,
        "harmful_dataset": args.harmful_dataset,
        "probed_layers": layers_to_probe,
        "ablation_vector_from_layer": use_layer_idx,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "refusal_vector_norm": norm_float
    }
    # If caller only wants the computed means (e.g., tests), return them early
    # without attempting to save model artifacts which may require tokenizer
    # methods not present on test doubles.
    if getattr(args, "return_means", False):
        try:
            def to_list(mx_arr):
                try:
                    return mx_arr.tolist()
                except Exception:
                    try:
                        return list(mx_arr)
                    except Exception:
                        return None

            harmful_means = {int(k): to_list(v) for k, v in harmful_activations.items()}
            harmless_means = {int(k): to_list(v) for k, v in harmless_activations.items()}
            return {
                "harmful_mean_activations": harmful_means,
                "harmless_mean_activations": harmless_means,
                "model_config": model_config,
                "probed_layers": layers_to_probe,
            }
        except Exception:
            logging.exception("Failed to assemble return_means payload")
            return None

    # Call save_ablated_model using positional args to remain compatible with
    # test doubles that may not accept newer keyword args.
    save_ablated_model(
        args.output_dir,
        model,
        tokenizer,
        model_config,
        abliteration_log,
        str(model_path),
    )
    logging.info("Abliterated model saved", extra={"extra_info": {"component": "cli", "event": "saving_end", "actual_output": {"output_dir": args.output_dir}}})

def main():
    """The main entry point for the CLI script."""
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_structured_logging("abliteration-toolkit-cli", log_level)

    logging.info("Starting MLX Abliteration Toolkit CLI...", extra={"extra_info": {"component": "cli", "event": "main_start", "inputs": vars(args)}})
    try:
        run_abliteration(args)
        logging.info("✅ Abliteration process completed successfully.", extra={"extra_info": {"component": "cli", "event": "main_success", "actual_output": {"output_dir": str(Path(args.output_dir).resolve())}}})
    except Exception as e:
        logging.error("❌ An error occurred during the abliteration process.", extra={"extra_info": {"component": "cli", "event": "main_error", "error_message": str(e)}}, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
