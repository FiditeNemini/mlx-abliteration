# MLX Abliteration Toolkit — Developer Guide

This document is for contributors and maintainers. It explains the repository layout, developer workflows, debugging and diagnostic techniques, testing guidance, and how to extend core behavior safely.

Keep this file lightweight and actionable: copy-paste the shell commands into your terminal (zsh) to reproduce the steps.

## Repository layout (high level)

- `cli.py` — Main command-line entry point for the end-to-end pipeline.
- `gui.py` — Gradio web UI wrapping the same pipeline with streaming logs and file outputs.
- `generate_dataset.py` — Dataset generator that reads `templates.yaml` and writes paired `harmful/harmless` jsonl files.
- `core/` — Core implementation:
  - `abliteration.py` — ActivationProbeWrapper, refusal vector computation, ablation (weight orthogonalization), and saving logic.
  - `asset_resolver.py` — Local path vs Hugging Face Hub resolution and caching.
  - `utils.py` — Small helpers: module/key mapping, marker extraction, tokenizer diagnostics.
  - `logging_config.py` — Structured logging setup used by CLI and GUI.
- `scripts/` — Development helper scripts (probe capture, diagnostics, repeatable CLI runs).
- `tests/` — Unit and small integration tests + `helpers/shims.py` (test shims/mocks).
- `generated_datasets/` and `outputs/` — example artifacts and outputs.

## Recommended developer environment

Use the provided environment spec files.

```bash
# Create the conda env (preferred)
conda env create -f environment.yml
conda activate abliteration-env

# Or install with pip into a venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use an IDE (VS Code), open the workspace file `mlx-abliteration.code-workspace` to pick up recommended settings.

## Quick developer smoke tests

1) Create a dummy model (fast, safe):

```bash
mkdir -p dummy_model
cat > dummy_model/config.json <<'JSON'
{"num_hidden_layers": 1, "hidden_size": 8}
JSON
# optional: provide a tokenizer_config.json to test marker extraction
cat > dummy_model/tokenizer_config.json <<'JSON'
{"chat_template": "{{ message.content }} </thinking> {{ message.author }}"}
JSON
touch dummy_model/model.safetensors
```

2) Generate a small dataset to probe against:

```bash
python generate_dataset.py --num-samples 10 --output-dir generated_datasets
```

3) Run a no-op abliteration (ablation_strength=0) to exercise end-to-end I/O and save flow:

```bash
python cli.py -m ./dummy_model -o ./out_dummy --harmless-dataset ./generated_datasets/harmless_dataset.jsonl --harmful-dataset ./generated_datasets/harmful_dataset.jsonl --ablation-strength 0
```

If this completes and `./out_dummy` contains `model.safetensors` (or `model.safetensors.index.json`), the save path and copying logic worked.

## Running tests

Unit tests are in `tests/` and use pytest. Run the full suite:

```bash
pytest -q
```

Run a single test to iterate faster:

```bash
pytest tests/test_abliteration_dummy.py -q
```

If you add or change logic in `core/abliteration.py`, add tests under `tests/` mirroring the existing patterns. Use `helpers/shims.py` to mock `mlx` runtime where possible so tests remain fast.

## Diagnostics & debugging

1) Structured logs


2) Probe marker / tokenization issues


Example CLI debug run:

```bash
python cli.py -m ./dummy_model -o ./out_dummy --harmless-dataset ./generated_datasets/harmless_dataset.jsonl --harmful-dataset ./generated_datasets/harmful_dataset.jsonl --probe-marker '</thinking>' --probe-debug --probe-debug-n 5
```
# If your marker includes a trailing newline that your tokenizer omits, strip it:
python cli.py -m ./dummy_model -o ./out_dummy --harmless-dataset ./generated_datasets/harmless_dataset.jsonl --harmful-dataset ./generated_datasets/harmful_dataset.jsonl --ablation-strength 0 --probe-marker '</thinking>' --strip-marker-newline

3) Inspecting captured activations

- The `ActivationProbeWrapper` returns a `captured_activations` dict keyed by layer index. To inspect shapes and values, add a small snippet in a local debug script (or use the existing `scripts/probe_capture.py`) that loads the model via `mlx_lm.load(...)`, runs the wrapper, and prints shapes.

4) Reproducing GUI runs from CLI

- The GUI delegates to the same core functions. If a GUI run fails, reproduce it with `cli.py` using the same inputs (model id, datasets, probe marker, layers) to get structured logs and stack traces in the terminal.

## Core code pointers (where to look & what to edit)

- `core/abliteration.py`
  - ActivationProbeWrapper: responsible for safe forwards and capturing hidden states. If you need to change how activations are captured (e.g., capture pre-attention or pre-MLP), edit here.
  - calculate_refusal_direction: simple difference of mean vectors — keep it deterministic and documented if you change normalization.
  - get_ablated_parameters: central piece — it enumerates flattened parameters, matches `target_modules` patterns, handles quantized layers (dequantize/quantize), and returns a new parameter tree. When adding new target patterns or supporting new quantized types, update here and add tests.
  - save_ablated_model: handles sharded vs single-file safetensors, copies ancillary files, writes metadata and `abliteration_log.json`.

- `core/asset_resolver.py` — modify only if you need alternative retrieval logic (S3, private storage). It currently uses `huggingface_hub.snapshot_download` when a local path isn't found.

- `core/utils.py` — utility functions used across the codebase. If you refactor parameter naming schemes, update `get_module_from_key` accordingly.

## Extending behavior safely

1) Adding new ablation targets

- Update `target_modules` default in `get_ablated_parameters` or pass `target_modules` explicitly from CLI/GUI wiring.
- Add a unit test in `tests/` that constructs a small dummy parameter map (see `tests/test_up_proj_target.py` for the pattern) and asserts the ablated param appears and is modified.

2) Supporting new quantized layer types

- Follow the `QuantizedLinear` branch in `get_ablated_parameters`: dequantize → orthogonalize → quantize. If a new quantized class exists in `mlx.nn.layers`, add a branch and include tests using `helpers/shims.py` to mock quantize/dequantize behaviors.

3) Changing probe selection logic

- The probe token selection logic is implemented both in `cli.get_mean_activations` and `gui.get_mean_activations_from_dataset`. If you change it, update both places and run `tests/test_probe_modes.py`.

## Useful scripts in `scripts/`

- `scripts/probe_capture.py` — capture a batch of activations for offline inspection. Useful to generate a small pickle/npz with activations to iterate quickly without re-loading heavy models.
- `scripts/probe_diagnostics.py` — utilities to inspect marker tokenization mismatches and print compact diagnostics.
- `scripts/run_cli_diag.py` — repeatable, parameterized wrapper to run `cli.py` with canned options; helpful for CI-style smoke runs.

If you'd like, I can add short wrappers that produce a `./debug/activations.npz` file for use with NumPy-based analysis.

## Quality gates & CI suggestions

- Run `pytest -q` as part of pre-merge checks. Add a fast smoke test that runs the no-op (`--ablation-strength 0`) on a tiny dummy model to ensure serialization logic remains intact.
- Add linting (flake8/ruff) in CI. The repo already includes `setup.cfg` so wire the same settings into CI.

## Common troubleshooting checklist

1. If the CLI fails with `FileNotFoundError: config.json` — check your `model` asset path or the resolved HF snapshot directory; ensure `config.json` is in the model root.
2. If probe marker is not found — enable `--probe-debug` and inspect token ids and a few sample prompts.
3. If saving fails for sharded models — confirm `model.safetensors.index.json` is present in the source; the save logic relies on the source weight map to determine shard contents.
4. If tests fail after a refactor — run the failing test with `-q -k <testname>` and run a single-file debug.

## Suggested follow-ups I can implement

- Add `README-DEVELOPER.md` examples as runnable `Makefile` targets.
- Add a small `dev_tools/` script to produce a canonical `dummy_model/` automatically.
- Provide a ready-to-run `tasks.json` for VS Code to run tests, CLI smoke, and launch the GUI.

Tell me which of these you'd like next and I will add it.

---

Last updated: 2025-10-04
