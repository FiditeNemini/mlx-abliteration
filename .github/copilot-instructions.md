## Quick orientation

This repository is the "MLX Abliteration Toolkit" — a small toolset for
mechanistic model surgery on MLX-based LLMs. Key entry points:

- `cli.py` — main CLI to run the full pipeline (resolve assets, probe activations,
  compute refusal direction, orthogonalize weights, save ablated model).
- `gui.py` — Gradio UI wrapper (used for interactive experiments).
- `generate_dataset.py` and `templates.yaml` — generate paired harmful/harmless datasets.
- `core/` — implementation: `abliteration.py`, `asset_resolver.py`, `utils.py`,
  `logging_config.py` (structured JSON logging). Tests live in `tests/`.

When making changes, prefer editing small, well-scoped modules in `core/` and
adjusting the CLI wiring in `cli.py` rather than large rewrites.

## Big-picture architecture (what to know fast)

- Data flow: prompts -> tokenizer -> ActivationProbeWrapper -> mean activations
  (Welford) -> refusal vector (harmful - harmless) -> weight orthogonalization
  -> save ablated model. Follow this flow across `cli.py` -> `core/abliteration.py` ->
  `core/utils.py` for concrete function names.
- Model I/O: models are MLX artifacts (safetensors or sharded safetensors). The
  code preserves sharding during save by reading `model.safetensors.index.json`.
- Asset resolution: `core/asset_resolver.resolve_asset` accepts local paths or
  Hugging Face Hub IDs (it uses `huggingface_hub.snapshot_download`).
- Probing: `ActivationProbeWrapper` captures hidden states per-layer during a
  forward pass. The probe can use a token marker (probe marker) extracted from
  `tokenizer_config.json` via `core/utils.extract_eot_from_chat_template` or the
  CLI `--probe-marker` flag.

## Developer workflows & commands

- Create environment (recommended):

  - Use Conda: `conda env create -f environment.yml` and `conda activate abliteration-env`.
  - Or: `pip install -r requirements.txt` (verify Python version in `environment.yml`).

- Run CLI (happy path):

  - Example: `python cli.py -m <model-path-or-hub-id> -o ./outdir`
  - Common flags: `--harmless-dataset`, `--harmful-dataset`, `--layers`, `--use-layer`, `--probe-marker`, `--strip-marker-newline`, `--ablation-strength`, `--cache-dir`.

- Generate datasets: `python generate_dataset.py --template-file templates.yaml --num-samples 100`

- Logging: structured JSON logs are written to `~/.mlx-llm/abliteration-toolkit-cli/log.jsonl` by default
  (see `core/logging_config.py`). Use this file to inspect pipeline events and telemetry.

- Tests: run the unit tests in `tests/` with your test runner (e.g., `pytest tests/`).

## Project-specific conventions & patterns

- Structured logging: logging calls often pass `extra={"extra_info": {...}}`. Keep
  new log statements compatible with `JsonFormatter` (avoid non-serializable objects).
- Parameter keys: the toolkit flattens parameter trees (see `mlx_lm.utils.tree_flatten`
  usages). When adding new parameters, ensure they appear in the model's parameter map
  expected by `save_ablated_model` and `get_ablated_parameters`.
- Probe marker extraction: `extract_eot_from_chat_template` uses a conservative
  regex to find an end-of-thought marker in `tokenizer_config.json`'s `chat_template`.
  If you add new chat-template formats, update that function.
  - Note: markers sometimes include a trailing newline that some tokenizers omit when tokenizing; use `--strip-marker-newline` (CLI) or the GUI checkbox to remove a single trailing newline from markers before tokenization when needed.
- Quantized layers: code special-cases `QuantizedLinear` (dequantize -> ablate -> re-quantize).
  If adding support for other quantized layer types, mirror this pattern.

## Integration points & external dependencies

- Hugging Face Hub: `core/asset_resolver` uses `huggingface_hub.snapshot_download`.
  Ensure callers provide valid repo IDs or local paths. Network errors are converted
  into `FileNotFoundError` for the caller.
- MLX runtime APIs: the code imports `mlx.core`, `mlx.nn` and helper functions from
  `mlx_lm`. When changing model code, run quick smoke tests to ensure `ActivationProbeWrapper`
  still finds `embed_tokens`, `layers`, `norm`, and optionally `lm_head`.
- safetensors: used for reading/writing binary model shards. Keep metadata copying
  behavior in `save_ablated_model` when modifying serialization.

## Files to reference when coding

- `cli.py` — end-to-end pipeline wiring and CLI flags (default values & fallback logic for probe marker).
- `core/abliteration.py` — ActivationProbeWrapper, calculate_refusal_direction,
  get_ablated_parameters, save_ablated_model (the heart of model surgery logic).
- `core/asset_resolver.py` — hub/local path resolution and caching behavior.
- `core/utils.py` — helper utilities used in several places (module lookup, marker extraction).
- `core/logging_config.py` — JSON log format and default file location.
- `generate_dataset.py` and `templates.yaml` — dataset generation pattern and examples.

## Concrete examples for AI edits

- Probe/token-index changes: the probe index logic appears in two places and must be changed together:
  - `ActivationProbeWrapper.__call__` in `core/abliteration.py` (where activations are captured per-layer)
  - `get_mean_activations` in `cli.py` and `get_mean_activations_from_dataset` in `gui.py` (where `probe_idx` is chosen and Welford averaging is applied)
  Example edit pattern: change how `probe_idx` is computed (marker vs last token), then run the CLI and GUI smoke flows to ensure both behave the same.

- Adding an ablation target: update the `target_modules` default in `get_ablated_parameters` (file: `core/abliteration.py`). If you add patterns like `"mlp.up_proj"`, ensure unit tests cover both standard 2D weight tensors and quantized paths (`QuantizedLinear`).

## Gradio UI (exact wiring and how to run)

- Entry point: `gui.py`. Launch the UI with:

  ```bash
  python gui.py
  ```

- The UI uses `gr.Blocks` and wires `start_button.click` to `run_abliteration_stream`.
  - Inputs: model, harmless/harmful dataset IDs, output name, layers, layer index, ablation strength, and `probe_marker`.
  - Note: the `probe_marker` UI component is a `gr.Code` field with `language=None` and expects a short one-line marker string (or empty to fallback to last token / tokenized chat_template detection).
  - Progress: `run_abliteration_stream` yields log strings and returns the final path to the abliterated model (index file for sharded models or `model.safetensors` for single-file models).


## Safety and testing notes

- The repo manipulates model weights. Add unit tests that run on small dummy
  models (mock `mlx` modules) to verify `get_ablated_parameters` and
  `save_ablated_model` behavior without large downloads.
- When changing serialization, run a round-trip smoke test: load source model,
  apply ablation with `--ablation-strength 0` (no-op), save, then verify the
  output directory still contains the expected `model.safetensors` or shards and `model.safetensors.index.json` if present.

---

If anything above is unclear or a section is missing examples you need (tests, CLI workflows, or MLX runtime details), tell me what you'd like expanded and I'll iterate.

## Example: safe patch + test for adding an ablation target

Below is a minimal, safe example showing how to add `"mlp.up_proj"` to the default
`target_modules` list in `get_ablated_parameters`, plus an associated unit test.

- Code edit (update `core/abliteration.py` near the top of `get_ablated_parameters`):

  - Replace the default list:

    - `target_modules = ["self_attn.o_proj", "mlp.down_proj", "mlp.c_proj"]`

    + `target_modules = ["self_attn.o_proj", "mlp.down_proj", "mlp.c_proj", "mlp.up_proj"]`

  - Why this is safe: we're only expanding the list of string patterns used to
    match parameter keys. No new tensor ops are introduced and quantized-paths
    are handled by the existing `QuantizedLinear` branch.

- Minimal unit test (add to `tests/`):

  - Create a tiny test that constructs a dummy model parameter map containing a
    key like `model.layers.0.mlp.up_proj.weight` and asserts that
    `get_ablated_parameters` returns a modified parameter for that key.

  - Example test steps:
    1. Create dummy model with parameter `'model.layers.0.mlp.up_proj.weight'` (small 2x2 numpy array).
    2. Call `get_ablated_parameters(model, refusal_vector, ablation_strength=1.0)`.
    3. Assert the returned mapping contains `'model.layers.0.mlp.up_proj.weight'`.

  - This follows the pattern used in `tests/test_abliteration_dummy.py` (mocking `mlx`), so it runs quickly as a unit test.

## Dev checklist — full smoke test (Conda + CLI)

Use this checklist when you want to run the full pipeline locally against a small model or a local dummy model directory.

1. Create the environment (recommended):

   ```bash
   conda env create -f environment.yml
   conda activate abliteration-env
   ```

   Or with pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare a tiny local dummy model directory for a no-op smoke test:

   - Create a directory `./dummy_model/` containing at minimum:
     - `config.json` with a small config (e.g., `{"num_hidden_layers": 1, "hidden_size": 8}`)
     - `tokenizer_config.json` (optional) with a `chat_template` if you want to test `probe_marker` extraction
     - a placeholder `model.safetensors` file (can be empty for a smoke run that uses `--ablation-strength 0`)

   Example:

   ```bash
   mkdir -p dummy_model
   cat > dummy_model/config.json <<'JSON'
   {"num_hidden_layers": 1, "hidden_size": 8}
   JSON
   touch dummy_model/model.safetensors
   ```

3. Run a no-op abliteration (ablation strength 0) which still does probing + save

   ```bash
   python cli.py -m ./dummy_model -o ./out_dummy --harmless-dataset ./generated_datasets/harmless_dataset.jsonl --harmful-dataset ./generated_datasets/harmful_dataset.jsonl --ablation-strength 0
   ```

   - Expected: pipeline runs through probing and computes a refusal vector, then saves output to `./out_dummy`. If `model.safetensors.index.json` exists in the source, saved output will include the index file.

4. Verify output

   - Check `~/.mlx-llm/abliteration-toolkit-cli/log.jsonl` for structured logs and the `./out_dummy` directory for `model.safetensors` or `model.safetensors.index.json` and `abliteration_log.json`.

5. If you change probing logic that relies on a `probe_marker`, test both:

   - Provide `--probe-marker '</thinking>'` on the CLI.
   - Or add a `tokenizer_config.json` with `chat_template` that the code's `extract_eot_from_chat_template` can parse.

If you'd like, I can (A) add the small example test file for `mlp.up_proj` now, or (B) convert the test shims into a reusable helper under `tests/helpers/` and update tests to import it. Which do you prefer?
