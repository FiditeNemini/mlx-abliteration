import gradio as gr
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Generator, Tuple, Any

# Add project root to the Python path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from core.asset_resolver import resolve_asset
from core.abliteration import (
    ActivationProbeWrapper,
    calculate_refusal_direction,
    get_ablated_parameters,
    save_ablated_model,
)
import mlx.core as mx
import mlx_lm
from datasets import load_dataset

def run_abliteration_stream(
    model_id: str,
    harmless_id: str,
    harmful_id: str,
    output_dir: str,
    layers_str: str,
    use_layer_idx: int,
    progress=gr.Progress(),
) -> Generator[Tuple[str, None] | Tuple[str, str], None, None]:
    log_history = ""
    def log_and_yield(message: str) -> str:
        nonlocal log_history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        log_history += full_message + "\n"
        print(full_message)
        return log_history

    try:
        yield log_and_yield("--- Step 0: Validating Inputs ---"), None
        if not all([model_id, harmless_id, harmful_id, output_dir]):
            raise gr.Error("All 'Required Inputs' fields must be filled.")

        yield log_and_yield("--- Step 1: Resolving Assets ---"), None
        output_path = Path.cwd() / "outputs" / Path(output_dir).name
        output_path.mkdir(parents=True, exist_ok=True)
        
        cache_dir = ".cache"
        model_path = resolve_asset(model_id, "models", cache_dir)
        harmless_ds_path = str(resolve_asset(harmless_id, "datasets", cache_dir))
        harmful_ds_path = str(resolve_asset(harmful_id, "datasets", cache_dir))
        yield log_and_yield(f"Assets resolved. Output will be saved to: {output_path.resolve()}"), None

        yield log_and_yield("--- Step 2: Loading Model, Tokenizer, and Config ---"), None
        model, tokenizer = mlx_lm.load(str(model_path))
        
        # DEFINITIVE FIX: Load config.json directly from the filesystem.
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Could not find 'config.json' in the model directory: {model_path}")
        with open(config_path, "r") as f:
            model_config = json.load(f)
        yield log_and_yield("Model, tokenizer, and config.json loaded successfully."), None

        num_layers = model_config["num_hidden_layers"]
        yield log_and_yield(f"Model '{model_id}' loaded with {num_layers} layers."), None
        
        harmless_dataset = load_dataset(harmless_ds_path, split="train")
        harmful_dataset = load_dataset(harmful_ds_path, split="train")

        yield log_and_yield("--- Step 3: Probing Activations ---"), None
        layers_to_probe = list(range(num_layers)) if layers_str.lower() == 'all' else [int(x.strip()) for x in layers_str.split(",")]
        wrapper = ActivationProbeWrapper(model)

        def get_mean_activations(dataset, desc: str, config: dict) -> Dict[int, mx.array]:
            hidden_size = config["hidden_size"]
            mean_activations = {layer: mx.zeros(hidden_size) for layer in layers_to_probe}
            counts = {layer: 0 for layer in layers_to_probe}
            max_seq_len = config.get("max_position_embeddings", 4096)

            for item in progress.tqdm(dataset, desc=desc):
                prompt = item.get("prompt") or item.get("text")
                if not prompt: continue
                tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
                if len(tokens) > max_seq_len: tokens = tokens[:max_seq_len]
                
                _, captured = wrapper(tokens[None], mask=None, layers_to_probe=layers_to_probe)
                for layer_idx, act in captured.items():
                    current_act = act[0, -1, :]
                    counts[layer_idx] += 1
                    delta = current_act - mean_activations[layer_idx]
                    mean_activations[layer_idx] += delta / counts[layer_idx]
                mx.eval(list(mean_activations.values()))
            return mean_activations

        harmful_mean_activations = get_mean_activations(harmful_dataset, "Probing harmful prompts", model_config)
        harmless_mean_activations = get_mean_activations(harmless_dataset, "Probing harmless prompts", model_config)
        yield log_and_yield("Activation probing complete."), None

        yield log_and_yield("--- Step 4: Computing Refusal Vector ---"), None
        actual_use_layer = use_layer_idx if use_layer_idx >= 0 else num_layers + use_layer_idx
        refusal_vector = calculate_refusal_direction(
            harmful_mean_activations[actual_use_layer], harmless_mean_activations[actual_use_layer]
        )
        yield log_and_yield(f"Refusal vector computed from layer {actual_use_layer}."), None

        yield log_and_yield("--- Step 5: Orthogonalizing Weights & Updating Model ---"), None
        ablated_params = get_ablated_parameters(model, refusal_vector)
        model.update(ablated_params)
        mx.eval(model.parameters())
        yield log_and_yield("Model weights have been updated."), None

        yield log_and_yield("--- Step 6: Saving Abliterated Model ---"), None
        abliteration_log = {
            "source_model": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        save_ablated_model(
            str(output_path), model, tokenizer, model_config, abliteration_log, source_model_path=str(model_path)
        )
        
        yield log_and_yield("✅ Abliteration process completed successfully."), str((output_path / "model.safetensors").resolve())

    except Exception as e:
        logger.error("An error occurred during abliteration", exc_info=True)
        yield log_and_yield(f"❌ An error occurred: {e}"), None
        raise gr.Error(str(e))

def create_ui():
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
    ui = create_ui()
    ui.launch(share=False)

