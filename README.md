# MLX Abliteration Toolkit

A comprehensive toolkit for performing mechanistic interpretability-driven model surgery on Large Language Models (LLMs) using the Apple MLX framework. It allows surgical removal of specific behaviors (such as refusal patterns) by modifying model weights directly.

[![CI](https://github.com/FiditeNemini/mlx-abliteration/actions/workflows/ci.yml/badge.svg)](https://github.com/FiditeNemini/mlx-abliteration/actions/workflows/ci.yml)

## Overview

Abliteration identifies and neutralizes the "refusal direction" within a model's activation space. The toolkit implements a full pipeline:

1.  **Data Collection**: Gathers activations from harmful and harmless prompt datasets
2.  **Direction Calculation**: Computes the refusal vector (or multiple principal components) from mean activation differences
3.  **Weight Orthogonalization**: Modifies weight matrices to be orthogonal to the refusal direction(s)
4.  **Adaptive Search** (optional): Automatically finds optimal ablation strength

The toolkit provides:
- **CLI** (`cli.py`) for automated workflows
- **Gradio UI** (`gui.py`) for interactive experimentation
- **Dataset Generator** (`generate_dataset.py`) for creating custom probe datasets
- **Diagnostic Scripts** for layer analysis and evaluation

## Installation

Conda is recommended for environment management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FiditeNemini/mlx-abliteration.git
    cd mlx-abliteration
    ```

2.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate abliteration-env
    ```

    **Or use pip:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

```bash
# Generate datasets
python generate_dataset.py --num-samples 100 --output-dir generated_datasets

# Run abliteration (basic)
python cli.py \
    -m mlx-community/Phi-3-mini-4k-instruct-4bit-mlx \
    -o ./outputs/ablated-phi3 \
    --harmless-dataset ./generated_datasets/harmless_dataset.jsonl \
    --harmful-dataset ./generated_datasets/harmful_dataset.jsonl

# Launch interactive UI
python gui.py
```

## CLI Usage

The `cli.py` script provides a comprehensive command-line interface.

### Basic Usage

```bash
python cli.py -m <model-path-or-hub-id> -o <output-dir> [options]
```

### Key Options

**Model & Data:**
- `-m, --model`: Local path or Hugging Face Hub ID (required)
- `-o, --output-dir`: Output directory (required)
- `-hd, --harmless-dataset`: Harmless dataset path/Hub ID
- `-ad, --harmful-dataset`: Harmful dataset path/Hub ID

**Probing:**
- `-l, --layers`: Layers to probe (`all` or comma-separated, e.g., `15,16,17`)
- `-u, --use-layer`: Layer for refusal vector (default: `-1` = last layer)
- `--probe-marker`: Token marker for precise probing (e.g., `</thinking>`)
- `--probe-mode`: Token selection mode:
  - `follow-token`: Token after marker (default)
  - `marker-token`: The marker token itself
  - `last-token`: Always use last token
  - `thinking-span`: Average tokens after marker
- `--probe-span`: Number of tokens to average in `thinking-span` mode (default: 1)
- `--probe-debug`: Enable debug output for tokenization

**Ablation:**
- `-s, --ablation-strength`: Ablation multiplier (default: 1.0)
- `--ablate-k`: Number of top PCA components to ablate (default: 1)
- `--ablate-method`: Method for ablation:
  - `projection`: Build projection matrix (recommended for multi-component)
  - `sequential`: Subtract components one-by-one (legacy)

**Adaptive Search:**
- `--adaptive`: Enable automatic strength search
- `--adaptive-initial`: Starting strength (default: 0.5)
- `--adaptive-max`: Maximum strength (default: 8.0)
- `--adaptive-growth`: Growth factor (default: 1.5)
- `--adaptive-target-ratio`: Target alignment reduction (default: 0.2 = 80% reduction)

**Evaluation & Debug:**
- `--eval-after`: Run post-ablation refusal evaluation
- `--eval-prompts`: Path to evaluation prompts (JSONL)
- `--dump-dequant`: Write dequantized .npy dumps for debugging
- `--cache-dir`: Cache directory for downloads (default: `.cache`)
- `-v, --verbose`: Enable verbose logging

### Examples

**Basic abliteration with Hugging Face model:**
```bash
python cli.py \
    -m mlx-community/Phi-3-mini-4k-instruct-4bit-mlx \
    -o ./outputs/ablated-phi3
```

**Advanced: thinking-span probing with adaptive search:**
```bash
python cli.py \
    -m /path/to/local/model \
    -o ./outputs/adaptive-ablated \
    --probe-mode thinking-span \
    --probe-span 3 \
    --adaptive \
    --adaptive-initial 0.5 \
    --adaptive-max 8.0 \
    --eval-after
```

**Multi-component ablation with projection method:**
```bash
python cli.py \
    -m mlx-community/some-model \
    -o ./outputs/multi-component \
    --ablate-k 3 \
    --ablate-method projection \
    --ablation-strength 1.5
```

**Debug tokenization and probing:**
```bash
python cli.py \
    -m /path/to/model \
    -o ./outputs/debug-run \
    --probe-marker '</think>' \
    --probe-debug \
    --probe-debug-n 5 \
    --probe-debug-full
```

## Gradio UI

Launch an interactive web interface:

```bash
python gui.py
```

The UI provides:
- Input fields for model and datasets (local paths or Hugging Face Hub IDs)
- Advanced parameter configuration (layers, probing, ablation settings)
- Real-time process monitoring with streaming logs
- Dry-run report viewer with automatic layer recommendations
- Output path display for the ablated model

All CLI options are available in the UI, organized in tabs for ease of use.

## Generating Datasets

The `generate_dataset.py` script creates paired harmful/harmless datasets from YAML templates.

### Basic Usage

```bash
python generate_dataset.py \
    --template-file templates.yaml \
    --num-samples 200 \
    --output-dir ./generated_datasets
```

### With Probe Markers

```bash
python generate_dataset.py \
    --num-samples 200 \
    --probe-marker '</thinking>' \
    --append-marker \
    --output-dir ./generated_datasets
```

### Template Structure (`templates.yaml`)

The YAML file contains two main sections:

#### 1. Concepts

Define harmful/harmless behavior pairs:

```yaml
concepts:
  - name: "Unsafe Behavior"
    harmful_value: "shoplift a candy bar"
    harmless_value: "buy a candy bar"
  
  - name: "Misinformation"
    harmful_value: "spread false rumors about Product X"
    harmless_value: "share verified facts about Product X"
```

#### 2. Templates

Define prompt templates with `{behavior}` placeholders:

```yaml
templates:
  - id: "instruction"
    prompt: "Write detailed instructions on how to {behavior}."
  
  - id: "story"
    prompt: "Tell me a story about someone who wants to {behavior}."
```

Optional `{marker}` placeholder for thinking markers:

```yaml
templates:
  - id: "think-instruction"
    prompt: "Think carefully{marker}, then explain how to {behavior}."
```

The generator creates two JSONL files:
- `harmful_dataset.jsonl`: Uses `harmful_value` from concepts
- `harmless_dataset.jsonl`: Uses `harmless_value` from concepts

## Advanced Features

### Adaptive Ablation Strength Search

The `--adaptive` flag enables automatic ablation strength selection:

```bash
python cli.py \
    -m /path/to/model \
    -o ./outputs/adaptive-ablated \
    --adaptive \
    --adaptive-initial 0.5 \
    --adaptive-max 8.0 \
    --adaptive-growth 1.5 \
    --adaptive-target-ratio 0.2
```

The adaptive search:
1. Measures baseline alignment metric (harmful vs harmless activation difference)
2. Tries increasing ablation strengths (multiplicative growth)
3. Stops when alignment is reduced by target percentage (e.g., 80% for ratio=0.2)
4. Saves the recommended strength in `abliteration_log.json`

### Multi-Component Ablation (PCA)

Remove multiple principal components instead of just the mean difference:

```bash
python cli.py \
    -m /path/to/model \
    -o ./outputs/multi-component \
    --ablate-k 3 \
    --ablate-method projection \
    --pca-sample 512
```

This computes top-k PCA components from per-example activations and removes their combined subspace.

### Thinking-Span Probing

For models with explicit thinking markers (e.g., `</think>`, `</thinking>`):

```bash
python cli.py \
    -m /path/to/model \
    -o ./outputs/thinking-ablated \
    --probe-marker '</think>' \
    --probe-mode thinking-span \
    --probe-span 3
```

This averages activations across a small window after the marker, useful when the transition tokenizes into multiple tokens.

### Diagnostic Scripts

**Layer analysis:**
```bash
python scripts/run_cli_diag.py \
    -m /path/to/model \
    -o ./outputs/diag_out \
    --probe-marker '</think>'
```

Outputs `dry_run_suggestions.json` with per-layer discrimination metrics.

**Multi-layer sweep:**
```bash
PYTHONPATH=. python scripts/sweep_topk_multilayer.py \
    --model-dir /path/to/model \
    --topk 3 \
    --output-dir ./outputs/sweep_results
```

Tests combined ablation across top-k discriminative layers.

**Evaluation:**
```bash
python scripts/eval_ablated_model.py \
    --model-dir ./outputs/ablated-model \
    --prompts ./eval_prompts.jsonl
```

### Selective Dequantized Dumps (Debugging)

For debugging quantized models, enable selective dequantized dumps:

```bash
python cli.py \
    -m /path/to/model \
    -o ./outputs/ablated-with-dumps \
    --dump-dequant
```

This writes `.npy` files to `<output_dir>/dequant_dumps/` for only the tensors that changed during ablation, making it easy to inspect actual float-level differences.

## Troubleshooting

**Out of Memory Errors:**
- Use a smaller model for testing
- Reduce dataset size or number of probed layers
- Lower `--pca-sample` value for multi-component ablation

**Probe Marker Not Found:**
- Use `--probe-debug` to inspect tokenization
- Verify marker exists in your prompts
- Check if marker is split across multiple tokens
- Consider using `--probe-mode thinking-span` with appropriate `--probe-span`

**Hugging Face Authentication:**
```bash
huggingface-cli login
```

**Missing Dependencies:**
Ensure all packages from `environment.yml` or `requirements.txt` are installed.

**Unexpected Results:**
- Start with `--ablation-strength 0` to test pipeline without modification
- Use `--eval-after` to evaluate refusal behavior
- Try `--adaptive` to automatically find optimal strength
- Check `abliteration_log.json` for diagnostic information

## Output Files

After running abliteration, the output directory contains:

- `model.safetensors` or `model.safetensors.index.json` (sharded models)
- `config.json`: Model configuration
- `tokenizer.json`, `tokenizer_config.json`: Tokenizer files
- `abliteration_log.json`: Pipeline metadata and settings
- `dequant_dumps/` (if `--dump-dequant` used): Debug dumps
- `post_ablation_evaluation.json` (if `--eval-after` used): Evaluation results

Structured logs are written to `~/.mlx-llm/abliteration-toolkit-cli/log.jsonl` (CLI) or the GUI equivalent.

## Testing

Run the test suite:

```bash
pytest -q
```

Run specific tests:
```bash
pytest tests/test_abliteration_dummy.py -v
```

Dry-run test with a real model:
```bash
export REAL_MODEL_DIR=/path/to/small-model
pytest tests/test_dry_run_real_model.py -v
```

## Project Structure

```
mlx-abliteration/
├── cli.py                    # Main CLI interface
├── gui.py                    # Gradio web UI
├── generate_dataset.py       # Dataset generator
├── templates.yaml            # Dataset templates
├── core/                     # Core library
│   ├── abliteration.py      # Probing and orthogonalization
│   ├── adaptive.py          # Adaptive strength search
│   ├── asset_resolver.py    # HF Hub/local asset resolution
│   ├── utils.py             # Helper functions
│   └── logging_config.py    # Structured logging
├── scripts/                  # Diagnostic and analysis tools
│   ├── run_cli_diag.py      # Layer analysis
│   ├── eval_ablated_model.py
│   ├── sweep_topk_multilayer.py
│   └── ...
└── tests/                    # Unit and integration tests
```

## Contributing

Contributions are welcome! Key areas:

- Additional ablation targets (update `target_modules` in `core/abliteration.py`)
- Support for new quantized layer types
- Improved evaluation metrics
- Additional diagnostic scripts

See `README-DEVELOPER.md` for detailed development guidelines.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{mlx_abliteration_toolkit,
  title = {MLX Abliteration Toolkit},
  author = {FiditeNemini},
  year = {2024},
  url = {https://github.com/FiditeNemini/mlx-abliteration}
}
```

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This toolkit builds on concepts from mechanistic interpretability research and the MLX framework by Apple. Special thanks to the open-source AI community for their contributions to understanding and improving language models.
