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
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import mlx.core as mx
from datasets import load_dataset
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
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save the abliterated model.")
    output_group.add_argument("--cache-dir", type=str, default=".cache", help="Cache directory for downloads.")
    output_group.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
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

def get_mean_activations(wrapper: ActivationProbeWrapper, tokenizer, dataset, layers_to_probe: List[int], desc: str) -> Dict[int, mx.array]:
    """Computes the mean of activations for specified layers iteratively.

    Args:
        wrapper (ActivationProbeWrapper): The model wrapper for probing activations.
        tokenizer: The tokenizer for the model.
        dataset: The dataset to iterate over.
        layers_to_probe (List[int]): A list of layer indices to probe.
        desc (str): A description for the tqdm progress bar.

    Returns:
        Dict[int, mx.array]: A dictionary mapping layer indices to their mean activation vectors.
    """
    mean_activations = {layer: None for layer in layers_to_probe}
    counts = {layer: 0 for layer in layers_to_probe}
    max_seq_len = getattr(wrapper, "max_seq_len", 4096)

    for item in tqdm(dataset, desc=desc):
        prompt = item.get("prompt") or item.get("text")
        if not prompt:
            tqdm.write("Skipping empty prompt.")
            continue
        
        tokens = mx.array(tokenizer.encode(prompt))[:max_seq_len]
        _, captured = wrapper(tokens[None], mask=None, layers_to_probe=layers_to_probe)

        for layer_idx, act in captured.items():
            final_token_act = act[0, -1, :]
            if mean_activations[layer_idx] is None:
                mean_activations[layer_idx] = final_token_act
            else:
                counts[layer_idx] += 1
                delta = final_token_act - mean_activations[layer_idx]
                mean_activations[layer_idx] += delta / counts[layer_idx]
        mx.eval(list(mean_activations.values()))

    return mean_activations

def run_abliteration(args: argparse.Namespace):
    """Runs the main abliteration process.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Raises:
        ValueError: If the specified layer to use for the refusal vector was not probed.
    """
    logging.info("Resolving assets", extra={"extra_info": {"component": "cli", "event": "asset_resolution_start", "inputs": {"model": args.model, "harmless_dataset": args.harmless_dataset, "harmful_dataset": args.harmful_dataset}}})
    model_path = resolve_asset(args.model, "models", args.cache_dir)
    harmless_ds_path = str(resolve_asset(args.harmless_dataset, "datasets", args.cache_dir))
    harmful_ds_path = str(resolve_asset(args.harmful_dataset, "datasets", args.cache_dir))
    logging.info("Assets resolved", extra={"extra_info": {"component": "cli", "event": "asset_resolution_end"}})

    logging.info("Loading model and datasets", extra={"extra_info": {"component": "cli", "event": "loading_start"}})
    model, tokenizer = mlx_lm.load(str(model_path))
    harmless_dataset = load_dataset(harmless_ds_path)["train"]
    harmful_dataset = load_dataset(harmful_ds_path)["train"]
    num_layers = len(model.model.layers)
    logging.info(f"Model loaded with {num_layers} layers.", extra={"extra_info": {"component": "cli", "event": "loading_end", "actual_output": {"num_layers": num_layers}}})

    logging.info("Probing activations", extra={"extra_info": {"component": "cli", "event": "probing_start"}})
    layers_to_probe = parse_layers(args.layers, num_layers)
    wrapper = ActivationProbeWrapper(model)
    harmful_activations = get_mean_activations(wrapper, tokenizer, harmful_dataset, layers_to_probe, "Probing harmful prompts")
    harmless_activations = get_mean_activations(wrapper, tokenizer, harmless_dataset, layers_to_probe, "Probing harmless prompts")
    logging.info("Activation probing complete", extra={"extra_info": {"component": "cli", "event": "probing_end"}})

    logging.info("Computing refusal vector", extra={"extra_info": {"component": "cli", "event": "vector_computation_start"}})
    use_layer_idx = args.use_layer if args.use_layer >= 0 else num_layers + args.use_layer
    if use_layer_idx not in layers_to_probe:
        raise ValueError(f"Layer {use_layer_idx} was not in the list of probed layers.")
    
    logging.info(f"Using activations from layer {use_layer_idx} for refusal direction.", extra={"extra_info": {"component": "cli", "event": "vector_computation_info", "inputs": {"use_layer": use_layer_idx}}})
    refusal_vector = calculate_refusal_direction(
        harmful_activations[use_layer_idx],
        harmless_activations[use_layer_idx]
    )
    logging.info("Refusal vector computed", extra={"extra_info": {"component": "cli", "event": "vector_computation_end", "actual_output": {"refusal_vector_norm": float(mx.linalg.norm(refusal_vector).item())}}})

    logging.info("Orthogonalizing weights and updating model", extra={"extra_info": {"component": "cli", "event": "orthogonalization_start"}})
    ablated_params = get_ablated_parameters(model, refusal_vector)
    model.update(ablated_params)
    mx.eval(model.parameters())
    logging.info("Model parameters updated", extra={"extra_info": {"component": "cli", "event": "orthogonalization_end"}})

    logging.info("Saving abliterated model", extra={"extra_info": {"component": "cli", "event": "saving_start"}})
    abliteration_log = {
        "source_model": args.model,
        "harmless_dataset": args.harmless_dataset,
        "harmful_dataset": args.harmful_dataset,,
        "probed_layers": layers_to_probe,
        "ablitation_vector_from_layer": use_layer_idx,
        "timestamp": datetime.utcnow().isoformat(),
        "refusal_vector_norm": float(mx.linalg.norm(refusal_vector).item())
    }
    # Correctly call save_ablated_model with the updated model object
    save_ablated_model(
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        abliteration_log=abliteration_log,
        source_model_path=str(model_path)
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
