import argparse
import json
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

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] - %(message)s")

def parse_args():
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
    if layers_str.lower() == "all":
        return list(range(num_model_layers))
    try:
        return [int(x.strip()) for x in layers_str.split(",")]
    except ValueError as e:
        raise ValueError(f"Invalid format for --layers: {e}")

def get_mean_activations(wrapper: ActivationProbeWrapper, tokenizer, dataset, layers_to_probe: List[int], desc: str) -> Dict[int, mx.array]:
    """Computes the mean of activations for specified layers iteratively."""
    mean_activations = {layer: None for layer in layers_to_probe}
    counts = {layer: 0 for layer in layers_to_probe}
    max_seq_len = getattr(wrapper, "max_seq_len", 4096)

    for item in tqdm(dataset, desc=desc):
        prompt = item.get("prompt") or item.get("text")
        if not prompt:
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
    logging.info("--- Step 1: Resolving Assets ---")
    model_path = resolve_asset(args.model, "models", args.cache_dir)
    harmless_ds_path = str(resolve_asset(args.harmless_dataset, "datasets", args.cache_dir))
    harmful_ds_path = str(resolve_asset(args.harmful_dataset, "datasets", args.cache_dir))

    logging.info("--- Step 2: Loading Model and Datasets ---")
    model, tokenizer = mlx_lm.load(str(model_path))
    harmless_dataset = load_dataset(harmless_ds_path)["train"]
    harmful_dataset = load_dataset(harmful_ds_path)["train"]
    num_layers = len(model.model.layers)
    logging.info(f"Model loaded with {num_layers} layers.")

    logging.info("--- Step 3: Probing Activations ---")
    layers_to_probe = parse_layers(args.layers, num_layers)
    wrapper = ActivationProbeWrapper(model)
    harmful_activations = get_mean_activations(wrapper, tokenizer, harmful_dataset, layers_to_probe, "Probing harmful prompts")
    harmless_activations = get_mean_activations(wrapper, tokenizer, harmless_dataset, layers_to_probe, "Probing harmless prompts")

    logging.info("--- Step 4: Computing Refusal Vector ---")
    use_layer_idx = args.use_layer if args.use_layer >= 0 else num_layers + args.use_layer
    if use_layer_idx not in layers_to_probe:
        raise ValueError(f"Layer {use_layer_idx} was not in the list of probed layers.")
    
    logging.info(f"Using activations from layer {use_layer_idx} for refusal direction.")
    refusal_vector = calculate_refusal_direction(
        harmful_activations[use_layer_idx],
        harmless_activations[use_layer_idx]
    )

    logging.info("--- Step 5: Orthogonalizing Weights and Updating Model ---")
    ablated_params = get_ablated_parameters(model, refusal_vector)
    model.update(ablated_params)
    mx.eval(model.parameters())
    logging.info("Model parameters updated with ablated weights.")

    logging.info("--- Step 6: Saving Abliterated Model ---")
    abliteration_log = {
        "source_model": args.model,
        "harmless_dataset": args.harmless_dataset,
        "harmful_dataset": args.harmful_dataset,
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

def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.info("Starting MLX Abliteration Toolkit CLI...")
    try:
        run_abliteration(args)
        logging.info("✅ Abliteration process completed successfully.")
        logging.info(f"   Abliterated model saved to: {Path(args.output_dir).resolve()}")
    except Exception as e:
        logging.error("❌ An error occurred during the abliteration process.", exc_info=args.verbose)
        sys.exit(1)

if __name__ == "__main__":
    main()
