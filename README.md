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

## Troubleshooting

- **Out of Memory Errors**: The abliteration process, especially activation probing, can be memory-intensive. If you encounter OOM errors, try using a smaller model or running on a machine with more RAM.
- **Hugging Face Authentication**: To access private models or datasets, you may need to log in to the Hugging Face Hub. Use the `huggingface-cli login` command in your terminal.
- **Dependency Issues**: Ensure you are using the specified Python version and have installed all packages from the `environment.yml` or `requirements.txt` file.
