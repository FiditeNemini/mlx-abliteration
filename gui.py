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
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Generator, Tuple, Any

# Add project root to the Python path
sys.path.append(str(Path(__file__).parent))

from core.asset_resolver import resolve_asset
from core.abliteration import (
    ActivationProbeWrapper,
    calculate_refusal_direction,
    get_ablated_parameters,
    save_ablated_model,
)
from core.logging_config import setup_structured_logging
import mlx.core as mx
import mlx_lm
from datasets import load_dataset

def get_mean_activations_from_dataset(
    dataset,
    wrapper: ActivationProbeWrapper,
    tokenizer: Any,
    layers_to_probe: List[int],
    config: Dict,
    desc: str,
    progress: gr.Progress,
) -> Dict[int, mx.array]:
    """Computes mean activations for a given dataset.

    Args:
        dataset: The dataset to process.
        wrapper (ActivationProbeWrapper): The model wrapper for probing.
        tokenizer (Any): The tokenizer.
        layers_to_probe (List[int]): A list of layer indices to probe.
        config (Dict): The model's configuration dictionary.
        desc (str): A description for the progress bar.
        progress (gr.Progress): A Gradio Progress object to update the UI.

    Returns:
        Dict[int, mx.array]: A dictionary mapping layer indices to mean activations.
    """
    hidden_size = config["hidden_size"]
    mean_activations = {layer: mx.zeros(hidden_size) for layer in layers_to_probe}
    counts = {layer: 0 for layer in layers_to_probe}
    max_seq_len = config.get("max_position_embeddings", 4096)

    for item in progress.tqdm(dataset, desc=desc):
        prompt = item.get("prompt") or item.get("text")
        if not prompt:
            continue
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        _, captured = wrapper(tokens[None], mask=None, layers_to_probe=layers_to_probe)
        for layer_idx, act in captured.items():
            current_act = act[0, -1, :]
            counts[layer_idx] += 1
            delta = current_act - mean_activations[layer_idx]
            mean_activations[layer_idx] += delta / counts[layer_idx]
        mx.eval(list(mean_activations.values()))
    return mean_activations

def run_abliteration_stream(
    model_id: str,
    harmless_id: str,
    harmful_id: str,
    output_dir: str,
    layers_str: str,
    use_layer_idx: int,
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
        yield log_and_yield("Validating Inputs", {"event": "validation_start"}), None
        if not all([model_id, harmless_id, harmful_id, output_dir]):
            raise gr.Error("All 'Required Inputs' fields must be filled.")
        yield log_and_yield("Inputs validated", {"event": "validation_end"}), None

        yield log_and_yield("Resolving Assets", {"event": "asset_resolution_start", "inputs": {"model_id": model_id, "harmless_id": harmless_id, "harmful_id": harmful_id}}), None
        output_path = Path.cwd() / "outputs" / Path(output_dir).name
        output_path.mkdir(parents=True, exist_ok=True)

        cache_dir = ".cache"
        model_path = resolve_asset(model_id, "models", cache_dir)
        harmless_ds_path = str(resolve_asset(harmless_id, "datasets", cache_dir))
        harmful_ds_path = str(resolve_asset(harmful_id, "datasets", cache_dir))
        yield log_and_yield(f"Assets resolved. Output will be saved to: {output_path.resolve()}", {"event": "asset_resolution_end", "actual_output": {"output_path": str(output_path.resolve())}}), None

        yield log_and_yield("Loading Model, Tokenizer, and Config", {"event": "loading_start"}), None
        model, tokenizer = mlx_lm.load(str(model_path))

        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Could not find 'config.json' in the model directory: {model_path}")
        with open(config_path, "r") as f:
            model_config = json.load(f)
        yield log_and_yield("Model, tokenizer, and config.json loaded successfully.", {"event": "loading_end"}), None

        num_layers = model_config["num_hidden_layers"]
        yield log_and_yield(f"Model '{model_id}' loaded with {num_layers} layers.", {"event": "model_info", "actual_output": {"num_layers": num_layers}}), None

        harmless_dataset = load_dataset(harmless_ds_path, split="train")
        harmful_dataset = load_dataset(harmful_ds_path, split="train")

        yield log_and_yield("Probing Activations", {"event": "probing_start"}), None
        layers_to_probe = list(range(num_layers)) if layers_str.lower() == 'all' else [int(x.strip()) for x in layers_str.split(",")]
        wrapper = ActivationProbeWrapper(model)

        harmful_mean_activations = get_mean_activations_from_dataset(harmful_dataset, wrapper, tokenizer, layers_to_probe, model_config, "Probing harmful prompts", progress)
        harmless_mean_activations = get_mean_activations_from_dataset(harmless_dataset, wrapper, tokenizer, layers_to_probe, model_config, "Probing harmless prompts", progress)

        yield log_and_yield("Activation probing complete.", {"event": "probing_end"}), None

        yield log_and_yield("Computing Refusal Vector", {"event": "vector_computation_start"}), None
        actual_use_layer = use_layer_idx if use_layer_idx >= 0 else num_layers + use_layer_idx
        refusal_vector = calculate_refusal_direction(
            harmful_mean_activations[actual_use_layer], harmless_mean_activations[actual_use_layer]
        )
        yield log_and_yield(f"Refusal vector computed from layer {actual_use_layer}.", {"event": "vector_computation_end", "inputs": {"use_layer": actual_use_layer}, "actual_output": {"refusal_vector_norm": float(mx.linalg.norm(refusal_vector).item())}}), None

        yield log_and_yield("Orthogonalizing Weights & Updating Model", {"event": "orthogonalization_start"}), None
        ablated_params = get_ablated_parameters(model, refusal_vector)
        model.update(ablated_params)
        mx.eval(model.parameters())
        yield log_and_yield("Model weights have been updated.", {"event": "orthogonalization_end"}), None

        yield log_and_yield("Saving Abliterated Model", {"event": "saving_start"}), None
        abliteration_log = {
            "source_model": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        save_ablated_model(
            str(output_path), model, tokenizer, model_config, abliteration_log, source_model_path=str(model_path)
        )

        yield log_and_yield("✅ Abliteration process completed successfully.", {"event": "main_success", "actual_output": {"output_path": str(output_path.resolve())}}), str((output_path / "model.safetensors").resolve())

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
                        use_layer_slider = gr.Slider(minimum=-36, maximum=35, step=1, value=-1, label="Use Refusal Vector from Layer", info="The layer index for the refusal vector. Negative values count from the end.")
                start_button = gr.Button("Start Abliteration", variant="primary", scale=1)
            with gr.Column(scale=3):
                log_output = gr.Textbox(label="Process Log", lines=20, interactive=False, autoscroll=True)
                output_file_display = gr.File(label="Abliterated Model Path", interactive=False)
        inputs = [model_input, harmless_ds_input, harmful_ds_input, output_dir_input, layers_input, use_layer_slider]
        for inp in inputs:
            inp.change(lambda: (None, None), outputs=[log_output, output_file_display])
        start_button.click(fn=run_abliteration_stream, inputs=inputs, outputs=[log_output, output_file_display])
    return demo

if __name__ == "__main__":
    setup_structured_logging("abliteration-toolkit-gui", logging.INFO)
    ui = create_ui()
    ui.launch(share=False)
