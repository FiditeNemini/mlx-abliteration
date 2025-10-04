import os
import json
from pathlib import Path
import pytest

from tests.helpers.shims import install_shims

# install shims for environments that lack native mlx
install_shims()

from cli import run_abliteration
import importlib

# Skip if REAL_MODEL_DIR not set or if the runtime loader is unavailable
def _mlx_loader_available() -> bool:
    try:
        mlx_lm = importlib.import_module('mlx_lm')
        return hasattr(mlx_lm, 'load')
    except Exception:
        return False


@pytest.mark.skipif(not os.environ.get("REAL_MODEL_DIR") or not _mlx_loader_available(), reason="Set REAL_MODEL_DIR and ensure mlx_lm.load is available to run this test")
def test_dry_run_real_model(monkeypatch, tmp_path):
    """Dry-run the abliteration pipeline on a real small model directory.

    This test is disabled by default and only runs when REAL_MODEL_DIR is set.
    It sets --ablation-strength to 0 to avoid changing weights and patches the
    serializer to avoid accidental writes.
    """
    model_dir = Path(os.environ["REAL_MODEL_DIR"]).expanduser()
    assert model_dir.is_dir(), f"REAL_MODEL_DIR not found: {model_dir}"

    # Prepare args-like object
    class Args:
        model = str(model_dir)
        harmless_dataset = str(Path(__file__).resolve().parents[1] / "generated_datasets" / "harmless_dataset.jsonl")
        harmful_dataset = str(Path(__file__).resolve().parents[1] / "generated_datasets" / "harmful_dataset.jsonl")
        layers = "all"
        use_layer = -1
        ablation_strength = 0.0
        probe_marker = None
        probe_mode = "thinking-span"
        probe_span = 3
        ablate_k = 1
        ablate_method = "projection"
        pca_sample = 64
        probe_debug = False
        probe_debug_n = 2
        probe_debug_full = False
        output_dir = str(tmp_path / "out_real_dry")
        cache_dir = ".cache"
        verbose = False

    args = Args()

    # Patch save_ablated_model to prevent writes
    def fake_save_ablated_model(output_dir, model, tokenizer, config, abliteration_log, source_model_path=None):
        # just ensure the function is callable and record the call
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "dry_run_log.json", "w") as fh:
            json.dump({"called": True, "source": source_model_path}, fh)

    monkeypatch.setattr('core.abliteration.save_ablated_model', fake_save_ablated_model)
    monkeypatch.setattr('cli.save_ablated_model', fake_save_ablated_model)

    # Run the pipeline (should complete without exception)
    run_abliteration(args)

    # Ensure dry run log written
    assert (Path(args.output_dir) / "dry_run_log.json").is_file()
