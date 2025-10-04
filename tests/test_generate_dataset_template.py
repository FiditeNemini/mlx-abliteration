import json
from pathlib import Path

from tests.helpers.shims import install_shims

# install shims to avoid importing heavy mlx runtime in tests
install_shims()

import tempfile
import os

from generate_dataset import parse_args, generate_datasets


def test_generate_with_model_chat_template(tmp_path, monkeypatch):
    # Prepare a tiny templates.yaml in the repo root (use existing one)
    repo_root = Path(__file__).resolve().parents[1]

    # Create a dummy model directory with tokenizer_config.json containing a chat_template
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir()
    chat_template = "System: HELLO\n{{ message.content }}\n</think>"
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": chat_template}))

    # Build args namespace
    class Args:
        template_file = str(repo_root / "templates.yaml")
        output_dir = str(tmp_path / "out")
        num_samples = 2
        probe_marker = None
        append_marker = False
        model = str(model_dir)

    args = Args()

    # Run generation
    generate_datasets(args)

    # Verify files exist and prompts end with the marker from chat_template
    harmful = Path(args.output_dir) / "harmful_dataset.jsonl"
    harmless = Path(args.output_dir) / "harmless_dataset.jsonl"
    assert harmful.is_file()
    assert harmless.is_file()

    def read_first_prompt(p):
        with open(p, "r") as fh:
            line = fh.readline()
            return json.loads(line)["prompt"]

    h_prompt = read_first_prompt(harmful)
    hh_prompt = read_first_prompt(harmless)

    assert h_prompt.strip().endswith("</think>")
    assert hh_prompt.strip().endswith("</think>")
