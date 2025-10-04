# MLX Abliteration Toolkit

This toolkit provides a suite of tools for performing mechanistic interpretability-driven model surgery on Large Language Models (LLMs) using the Apple MLX framework. It allows for the surgical removal of specific behaviors, such as refusal to answer certain prompts, by modifying the model's weights directly.

## Overview

Abliteration is a technique that identifies and neutralizes the "refusal direction" within a model's activation space. This toolkit implements the full abliteration pipeline:

1.  **Data Collection**: Gathers activations from a model based on "harmful" and "harmless" prompt datasets.
2.  **Direction Calculation**: Computes the refusal vector based on the difference in mean activations.
3.  **Weight Orthogonalization**: Modifies the model's weight matrices to be orthogonal to the refusal direction, effectively disabling the targeted behavior.

This toolkit provides both a command-line interface (CLI) for programmatic use and a Gradio web UI for interactive experimentation.

## Installation

It is recommended to use Conda to manage the environment.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd abliteration-toolkit
    ```

2.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate abliteration-env
    ```

Alternatively, you can use `pip` with the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## CLI Usage

The `cli_script.py` provides a powerful interface for running the abliteration process from the command line.

```bash
python cli_script.py --help
```

### Examples

**Run abliteration on a local model:**
```bash
python cli_script.py \
    --model /path/to/local/mlx-model \
    --output-dir /path/to/save/ablated-model
```

**Run abliteration on a model from the Hugging Face Hub:**
```bash
python cli_script.py \
    --model "mlx-community/phi-3-mini-4k-instruct-4bit-mlx" \
    --output-dir ./ablated-phi-3
```

## Gradio UI Usage

For an interactive experience, launch the Gradio web UI.

```bash
python gradio_ui.py
```

This will start a local web server. Open the provided URL in your browser to access the interface. The UI allows you to:
- Specify model and datasets (from local paths or Hugging Face Hub).
- Configure advanced parameters like which layers to probe.
- Monitor the process in real-time via a streaming log.
- Receive a link to the final abliterated model directory upon completion.

## 🔬 Experimental Features

### Counterfactual Probing for "Thinking" Models

For more advanced models that exhibit chain-of-thought reasoning, standard ablation can be less effective due to "signal contamination" from the thinking process. This toolkit includes an experimental feature for more precise, marker-based activation probing.

This technique is designed to isolate the model's "decision" to refuse from its preceding "thinking" process, resulting in a cleaner and more effective refusal vector.

**For a detailed explanation of the theory and usage instructions, please see the [Counterfactual Probing Documentation](./docs/counterfactual_ablation.md).**

## Troubleshooting

- **Out of Memory Errors**: The abliteration process, especially activation probing, can be memory-intensive. If you encounter OOM errors, try using a smaller model or running on a machine with more RAM.
- **Hugging Face Authentication**: To access private models or datasets, you may need to log in to the Hugging Face Hub. Use the `huggingface-cli login` command in your terminal.
- **Dependency Issues**: Ensure you are using the specified Python version and have installed all packages from the `environment.yml` or `requirements.txt` file.

## Generating Datasets for Ablation

This toolkit includes a flexible script, `generate_dataset.py`, to help you create paired datasets for counterfactual ablation. The script uses a YAML file (`templates.yaml`) to define the structure and content of the generated prompts.

### The `templates.yaml` File Structure

The `templates.yaml` file is the heart of the dataset generator. It has two main sections: `concepts` and `templates`.

#### 1. The `concepts` Section

The `concepts` section defines the pairs of ideas you want the model to distinguish between. Each concept consists of a harmful behavior and its harmless counterpart.

**Structure:**

Each item under `concepts` is an object with three keys:
*   `name`: A descriptive name for the concept (e.g., "Illegal Activities").
*   `harmful_value`: A string describing the harmful or undesirable action.
*   `harmless_value`: A string describing the safe or neutral alternative.

**Example:**

```yaml
concepts:
  - name: "Unsafe Behavior"
    harmful_value: "shoplift a candy bar"
    harmless_value: "buy a candy bar"

  - name: "Misinformation"
    harmful_value: "spread a false rumor that 'Product X' is dangerously defective"
    harmless_value: "state the fact that 'Product X' has passed all safety inspections"
## MLX Abliteration Toolkit — README

This repository implements an end-to-end pipeline for mechanistic, activation-space "abliteration" of behaviors in MLX models. The project provides:

- A command-line interface: `cli.py`
- An interactive Gradio UI: `gui.py`
- A dataset generator for paired harmful/harmless prompts: `generate_dataset.py` + `templates.yaml`
- Core library code under `core/` (probing, vector computation, weight orthogonalization, saving)
- Tests and helper shims under `tests/` and `helpers/`

This README documents how to use every top-level file and the important core modules, with concrete examples.

## Quick start

1. Clone the repository and enter it:

```bash
git clone <repo-url>
cd mlx-abliteration
```

2. Create an environment (Conda recommended) and install dependencies:

```bash
conda env create -f environment.yml
conda activate abliteration-env
# or with pip
pip install -r requirements.txt
```

3. Run a quick no-op abliteration (uses your local model directory and saves outputs):

```bash
# Create a tiny dummy model directory before running (see Developer tips below)
python cli.py -m ./dummy_model -o ./out_dummy --ablation-strength 0
```

## Files & how to use them

Below are the important files with usage examples and tips.

### `cli.py` — Command-line pipeline

Purpose: run the full abliteration workflow (resolve assets, probe activations, compute refusal vector, orthogonalize weights, save ablated model).

How to run:

```bash
python cli.py -m <model-path-or-hub-id> -o <output-dir> [options]
```

Key flags (most common):

- `-m / --model`: local path or Hugging Face Hub ID of the MLX model (required)
- `-o / --output-dir`: directory where the ablated model will be written (required)
- `-hd / --harmless-dataset`, `-ad / --harmful-dataset`: dataset local path or Hub id (defaults present)
- `-l / --layers`: which layers to probe ("all" or comma-separated indices)
- `-u / --use-layer`: layer index to compute the refusal vector from (negative values allowed, default -1)
- `-s / --ablation-strength`: multiplier for ablation effect (float, default 1.0)
- `--probe-marker`: optional string marker to locate the probe token (e.g. `</thinking>`). If omitted, the code attempts to extract a marker from `tokenizer_config.json` or falls back to the last token.
- `--probe-mode`: `follow-token|marker-token|last-token` — how to choose which token to probe when a marker is found.
- `--probe-mode`: `follow-token|marker-token|last-token|thinking-span` — how to choose which token(s) to probe when a marker is found. Use `thinking-span` to average a small window of tokens following a marker (see experimental features below).
- `--ablate-method`: `projection|sequential` — how to remove the identified components from model weights. `projection` (default) builds a projection matrix from the top-k components and removes that subspace in one step; `sequential` subtracts projections component-by-component (legacy behavior). Use `projection` for multi-component ablation (recommended).

#### New probe mode: `thinking-span`

`thinking-span` averages activations across a short contiguous span of tokens immediately following an end-of-thought marker (for example `</think>`). Use this when a model's internal reasoning or the transition token is split across multiple tokenizer tokens. Recommended defaults:

- Probe Marker: `</think>` (auto-detected when present in `tokenizer_config.json`)
- Probe Mode: `thinking-span`
- Probe Span: `1` (increase to 2-4 if the post-marker content tokenizes into multiple tokens you wish to average)

`thinking-span` is conservative and is best used when `--probe-debug` shows marker tokens followed by multi-token transitions.
- `--probe-debug`: emit a few tokenization/probe diagnostics (useful for troubleshooting marker/tokenization mismatches)

Example (full):

```bash
python cli.py \
    -m mlx-community/phi-3-mini-4k-instruct-4bit-mlx \
    -o ./outputs/ablated-phi3 \
    --harmless-dataset ./generated_datasets/harmless_dataset.jsonl \
    --harmful-dataset ./generated_datasets/harmful_dataset.jsonl \
    --layers all \
    --use-layer -1 \
    --ablation-strength 1.0 \
    --probe-marker '</thinking>' \
    --probe-mode follow-token
```

Notes:

- `cli.py` uses `core/asset_resolver.resolve_asset` to accept either a local path or a Hugging Face Hub id. Use `--cache-dir` to control where downloads are stored.
- If `--ablation-strength 0` you can exercise the full pipeline (probing + save) without changing weights.

### `gui.py` — Gradio interface

Purpose: interactive UI for the same pipeline with streaming logs and an output file link.

How to run:

```bash
python gui.py
```

The UI fields map directly to CLI options (model path/Hub ID, harmless/harmful datasets, layers, probe marker, ablation strength, etc.). The UI saves outputs under `./outputs/<output_dir>` by default and returns the path to the saved safetensors (or index file for sharded models).

Use the "Probe Marker" field to paste marker strings or leave empty and rely on `tokenizer_config.json` detection.

### `generate_dataset.py` & `templates.yaml`

Purpose: generate paired `harmful_dataset.jsonl` and `harmless_dataset.jsonl` for probing.

Basic run (uses `templates.yaml` in repo):

```bash
python generate_dataset.py
```

Options:

- `--template-file`: path to YAML templates (default `templates.yaml`)
- `--output-dir`: directory to write `harmful_dataset.jsonl` and `harmless_dataset.jsonl`
- `--num-samples`: number of pairs to generate
- `--probe-marker`: optional string to insert into templates (supports templates that include a `{marker}` placeholder)
- `--append-marker`: append the marker to every generated prompt (when used with `--probe-marker`)

Example:

```bash
python generate_dataset.py --num-samples 200 --output-dir ./generated_datasets --probe-marker '</thinking>' --append-marker
```

`templates.yaml` structure (summary): contains `concepts` (with `name`, `harmful_value`, `harmless_value`) and `templates` (each with `id` and `prompt` containing `{behavior}` — optionally `{marker}`). See the shipped `templates.yaml` for examples.

### `core/abliteration.py` — core algorithm

Main functions & classes:

- `ActivationProbeWrapper(model)`: lightweight wrapper that runs a forward pass and captures hidden states per-layer. Use its call signature to get logits and a dict of captured activations.
- `calculate_refusal_direction(harmful_mean, harmless_mean)`: returns the difference vector used as the refusal direction.
- `get_ablated_parameters(model, refusal_vector, target_modules=None, ablation_strength=1.0)`: returns an updated parameter tree where specified weight matrices have been orthogonalized to the refusal vector. It handles both standard 2D weights and quantized linear layers (`QuantizedLinear`) by dequantize → orthogonalize → re-quantize.
- `save_ablated_model(output_dir, model, tokenizer, config, abliteration_log, source_model_path)`: serializes ablated weights, preserves sharding (reads `model.safetensors.index.json` if present), copies ancillary files, and writes an `abliteration_log.json`.

Developer notes:

- `get_ablated_parameters` accepts `target_modules` (defaults to `['self_attn.o_proj','mlp.down_proj','mlp.c_proj']`) and a float `ablation_strength`.
- The code writes careful diagnostic logs and performs orthogonality checks for each modified weight.

### `core/asset_resolver.py`

Single helper `resolve_asset(path_or_id, asset_type, local_cache_dir)`:

- If `path_or_id` exists locally, it returns the resolved path.
- Otherwise it calls `huggingface_hub.snapshot_download` (repo_type derived from `asset_type`, e.g., `models` → `model`).

Use `--cache-dir` in CLI or set `.cache` for GUI to control local cache location.

### `core/utils.py`

Useful helpers:

- `get_module_from_key(model, key)`: map a flattened parameter key (e.g. `model.layers.0.mlp.down_proj.weight`) to the owning module object.
- `extract_eot_from_chat_template(template_str)`: heuristic to extract a marker (e.g. `</think>`) from a Jinja-style chat template string stored in `tokenizer_config.json`.
- `tokenizer_marker_diff(tokenizer, marker)`: small diagnostic that returns token ids and token strings (if supported) for a `marker` string.

### `core/logging_config.py`

This repo uses structured JSON logging for CLI and GUI. The CLI calls `setup_structured_logging(name, level)` at startup. Logs are intended to be written to a default location (see `logging_config.py`) — use these logs when diagnosing failures or reviewing pipeline telemetry.

### `scripts/` and `tests/`

- `scripts/probe_capture.py`, `scripts/probe_diagnostics.py`, `scripts/run_cli_diag.py` are helper scripts used during development to capture/inspect probing behavior and reproduce CLI runs.
- `tests/` contains unit tests and small integration tests. Run tests with `pytest`.

### `scripts/run_cli_diag.py` — safe diagnostic dry-run

This small helper runs the CLI in a diagnostic (dry-run) mode and writes JSON/CSV suggestions for which layers look most discriminative. Important: pass the model path or Hub id on the command line rather than editing the script to avoid leaking sensitive local paths.

Examples:

```bash
# Run against a local model directory (recommended to avoid embedding paths in the repo)
python scripts/run_cli_diag.py -m /path/to/local/model -o ./outputs/diag_out

# Run against a Hugging Face Hub id
python scripts/run_cli_diag.py -m "org/model-id" -o ./outputs/diag_out --probe-marker "</think>"
```

The script will call `cli.run_abliteration` with `return_means=True`, compute per-layer diff norms, and write `dry_run_suggestions.json` and `dry_run_layer_stats.csv` into the output directory.

Run tests (recommended):

```bash
pytest -q
```

Run a single test for quick verification:

```bash
pytest tests/test_abliteration_dummy.py::test_get_ablated_parameters -q
```

## Developer & debugging tips

- Quick no-op smoke test: create a `dummy_model/` with a minimal `config.json`, optional `tokenizer_config.json`, and an (empty) `model.safetensors` file. Then run `python cli.py -m ./dummy_model -o ./out_dummy --ablation-strength 0` to go through the flows without changing weights.

- If using probe markers, verify tokenization using `--probe-debug` (CLI) or `Probe Debug` (GUI). If a marker is not found, the tool will fallback to the last token and emit a diagnostic showing a few sample prompts.

- For models on Hugging Face Hub, call `huggingface-cli login` if they are private.

- If you modify `core/abliteration.py` or probing code, run the small unit tests first before trying a large model run.

## Common failure modes

- Missing `config.json` in model directory: the CLI/GUI will raise FileNotFoundError. Ensure `config.json` exists next to `model.safetensors` or the index file.
- Probe marker never found: the code falls back to last-token probing and prints a concise diagnostic (use `--probe-debug` to inspect tokenization).
- OOM during probing: reduce dataset size, probe fewer layers, or run against a smaller model.

## Where outputs and logs go

- Abliterated models are written to the `--output-dir` you provide (CLI) or `outputs/<name>` (GUI). For sharded models the tool preserves shard filenames and writes `model.safetensors.index.json`.
- A JSON `abliteration_log.json` is written into the output directory describing inputs and the vector norm.
- Structured logs are also written by the CLI/GUI (see `core/logging_config.py` for the configured path).

## Contributing / Next steps

- Add new `target_modules` patterns by updating the default list in `get_ablated_parameters` and add unit tests in `tests/` to cover newly-targeted parameter names.
- Add support for additional quantized layer types by following the `QuantizedLinear` branch in `get_ablated_parameters`.

If you'd like, I can also:

- Add a short `README-DEVELOPER.md` with the dummy-model creation commands and a minimal test harness.
- Add example `Makefile` or `tasks.json` entries to automate common flows (run CLI, run GUI, generate dataset, run tests).

## Minimal example commands (copyable)

```bash
# Create a dummy model for quick smoke test
mkdir -p dummy_model
cat > dummy_model/config.json <<'JSON'
{"num_hidden_layers": 1, "hidden_size": 8}
JSON
touch dummy_model/model.safetensors

# Generate data (100 samples)
python generate_dataset.py --num-samples 100 --output-dir generated_datasets

# Run CLI as a no-op (ablation_strength=0)
python cli.py -m ./dummy_model -o ./out_dummy --harmless-dataset ./generated_datasets/harmless_dataset.jsonl --harmful-dataset ./generated_datasets/harmful_dataset.jsonl --ablation-strength 0

# Launch the Gradio GUI
python gui.py
```

## Recommended ablation command (thinking-span + projection)

For models that use an explicit thinking or internal monologue marker (e.g. `</think>`), a good starting CLI is to probe a small span after the marker and remove the top component via projection:

```bash
python cli.py \
    -m /path/to/your/model \
    -o ./outputs/ablated-model \
    --harmless-dataset ./generated_datasets/harmless_dataset.jsonl \
    --harmful-dataset ./generated_datasets/harmful_dataset.jsonl \
    --probe-mode thinking-span \
    --probe-span 3 \
    --ablate-k 1 \
    --ablate-method projection \
    --ablation-strength 1.0
```

Start with `--ablation-strength 0` to run a no-op save first (verifies probing, vector computation, and serialization):

```bash
python cli.py -m /path/to/your/model -o ./out_dummy --ablation-strength 0 --probe-mode thinking-span --probe-span 3 --ablate-method projection
```

Dry-run tests
---------------
There is an optional pytest that will run a dry-run of the CLI against a real, small local model. It is skipped by default; to enable it set the `REAL_MODEL_DIR` environment variable to the model directory and run pytest. Example:

```bash
# set to a local small model dir (must contain config.json and tokenizer_config.json)
export REAL_MODEL_DIR=/path/to/small-model
pytest -q tests/test_dry_run_real_model.py
```

The test will run the full probing and PCA/ablation code paths but will not overwrite your model (the test patches the save routine to avoid destructive writes). Use this to validate the full pipeline on a target model before running irreversible ablations.

---

If anything in this README is unclear or you want expanded examples for a particular model type or deployment scenario, tell me which model you plan to target and I will add a tailored example command set.
