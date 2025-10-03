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
    sample_not_found_examples = []
    probe_debug_lines: list[str] = []

    for item in progress.tqdm(dataset, desc=desc):
        prompt = item.get("prompt") or item.get("text")
        if not prompt:
            continue
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        _, captured = wrapper(tokens[None], mask=None, layers_to_probe=layers_to_probe)

        probe_idx = -1  # Default to the last token
        if marker_tokens is not None and getattr(marker_tokens, 'size', 0) > 0:
            token_list = tokens.tolist()
            marker_list = marker_tokens.tolist()
            found = False
            # Search for the last occurrence of the marker by searching backwards
            for i in range(len(token_list) - len(marker_list), -1, -1):
                if token_list[i:i + len(marker_list)] == marker_list:
                    # The correct index is for the token *after* the marker sequence
                    potential_idx = i + len(marker_list)
                    # Ensure the index is within the sequence bounds
                    if potential_idx < len(token_list):
                        # Marker is followed by another token; capture that following token
                        probe_idx = potential_idx
                        found = True
                    else:
                        # Marker found at end of prompt; capture marker token activation instead
                        probe_idx = i + len(marker_list) - 1
                        found = True
                    break  # Found the last marker, stop searching

            if found:
                marker_found_any = True
                if probe_debug and len(probe_debug_lines) < probe_debug_n:
                    truncated = (prompt[:200] + '...') if len(prompt) > 200 else prompt
                    probe_debug_lines.append(f"probe_idx={probe_idx}, prompt={truncated}")
            else:
                if len(sample_not_found_examples) < 3:
                    try:
                        sample_not_found_examples.append((prompt, token_list))
                    except Exception:
                        pass

        for layer_idx, act in captured.items():
            # ensure probe_idx is within bounds
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

    return mean_activations, probe_debug_lines

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
    probe_debug: bool = False,
    probe_debug_n: int = 3,
    probe_debug_full: bool = False,
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

        harmless_dataset = load_dataset(harmless_ds_path, split="train")
        harmful_dataset = load_dataset(harmful_ds_path, split="train")

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
            probe_debug=probe_debug,
            probe_debug_n=probe_debug_n,
            probe_debug_full=probe_debug_full,
        )

        # If probe_debug was requested, print the collected per-example probe indices
        if probe_debug:
            debug_lines = []
            debug_lines.extend([f"HARMFUL: {l}" for l in harmful_debug])
            debug_lines.extend([f"HARMLESS: {l}" for l in harmless_debug])
            for dl in debug_lines:
                yield log_and_yield(dl, {"event": "probe_debug_line"}), None

        yield log_and_yield("Activation probing complete.", {"event": "probing_end"}), None

        yield log_and_yield("Computing Refusal Vector", {"event": "vector_computation_start"}), None
        actual_use_layer = use_layer_idx if use_layer_idx >= 0 else num_layers + use_layer_idx
        refusal_vector = calculate_refusal_direction(
            harmful_mean_activations[actual_use_layer], harmless_mean_activations[actual_use_layer]
        )
        yield log_and_yield(f"Refusal vector computed from layer {actual_use_layer}.", {"event": "vector_computation_end", "inputs": {"use_layer": actual_use_layer}, "actual_output": {"refusal_vector_norm": float(mx.linalg.norm(refusal_vector).item())}}), None

        yield log_and_yield("Orthogonalizing Weights & Updating Model", {"event": "orthogonalization_start"}), None
        ablated_params = get_ablated_parameters(model, refusal_vector, ablation_strength=ablation_strength)
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
                        use_layer_slider = gr.Slider(minimum=-36, maximum=35, step=1, value=-1, label="Use Refusal Vector from Layer", info="The layer index for the refusal vector. Negative values count from the end.")
                        strength_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Ablation Strength", info="The strength of the ablation effect. >1.0 amplifies the effect.")
                        probe_marker_input = gr.Code(label="Probe Marker", language=None, lines=1)
                        probe_mode_input = gr.Dropdown(label="Probe Mode", choices=["follow-token", "marker-token", "last-token"], value="follow-token", info="How to select the probe token when a marker is found.")
                        probe_debug_checkbox = gr.Checkbox(label="Probe Debug", value=False, info="Emit per-example probe index debug lines into the log.")
                        probe_debug_n_input = gr.Number(label="Probe Debug N", value=3, precision=0)
                        probe_debug_full_checkbox = gr.Checkbox(label="Probe Debug Full Tokens", value=False, info="When debug enabled, show token strings if available.")
                start_button = gr.Button("Start Abliteration", variant="primary", scale=1)
            with gr.Column(scale=3):
                log_output = gr.Textbox(label="Process Log", lines=20, interactive=False, autoscroll=True)
                output_file_display = gr.File(label="Abliterated Model Path", interactive=False)
        inputs = [model_input, harmless_ds_input, harmful_ds_input, output_dir_input, layers_input, use_layer_slider, strength_slider, probe_marker_input, probe_mode_input, probe_debug_checkbox, probe_debug_n_input, probe_debug_full_checkbox]
        for inp in inputs:
            inp.change(lambda: (None, None), outputs=[log_output, output_file_display])
        start_button.click(fn=run_abliteration_stream, inputs=inputs, outputs=[log_output, output_file_display])
    return demo

if __name__ == "__main__":
    setup_structured_logging("abliteration-toolkit-gui", logging.INFO)
    ui = create_ui()
    ui.launch(share=False)
