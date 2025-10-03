#!/usr/bin/env python3
import os
# suppress huggingface/tokenizers parallelism warning in forked processes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
"""Probe marker diagnostics helper.

Usage:
  python scripts/probe_diagnostics.py --model /path/to/model [--probe-marker '</thinking>'] [--probe-debug-full]

This script loads the tokenizer for a local MLX model and inspects the
generated_datasets (harmless/harmful) in the repo to report whether the
probe marker is found and how it's tokenized.
"""
import argparse
import json
import traceback
from pathlib import Path
from typing import Optional

from datasets import load_dataset

import mlx_lm
from core.utils import extract_eot_from_chat_template, tokenizer_marker_diff


def find_marker_from_config(model_path: Path) -> Optional[str]:
    cfg = model_path / "tokenizer_config.json"
    if not cfg.is_file():
        return None
    try:
        with open(cfg, "r") as f:
            data = json.load(f)
        chat_template = data.get("chat_template")
        if chat_template:
            return extract_eot_from_chat_template(chat_template)
    except Exception:
        return None
    return None


def analyze_dataset(dataset_path: Path, tokenizer, marker: Optional[str], sample_n: int = 3, name: str = "dataset", append_marker: bool = False):
    print(f"\nAnalyzing {name}: {dataset_path}")
    ds = load_dataset("json", data_files=str(dataset_path))["train"]
    marker_ids = None
    if marker:
        try:
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
            print(f"Marker literal: {repr(marker)} -> ids: {marker_ids}")
        except Exception as e:
            print(f"Failed to encode marker: {e}")

    found_count = 0
    not_found_samples = []
    inspected = 0
    for item in ds:
        if inspected >= 200:
            break
        prompt = item.get("prompt") or item.get("text")
        if not prompt:
            continue
        inspected += 1
        try:
            prompt_to_encode = prompt + marker if (append_marker and marker) else prompt
            toks = tokenizer.encode(prompt_to_encode, add_special_tokens=False)
            token_list = list(toks)
        except Exception:
            token_list = None

        found = False
        if marker_ids and token_list:
            # search for last occurrence and treat marker-at-end as a valid match
            m = list(marker_ids)
            for i in range(len(token_list) - len(m), -1, -1):
                if token_list[i:i+len(m)] == m:
                    # if marker is followed by another token, good; if marker is at end, still treat as found
                    found = True
                    break

        if found:
            found_count += 1
        else:
            if len(not_found_samples) < sample_n:
                not_found_samples.append((prompt, token_list))

    print(f"Inspected up to {inspected} items. Marker found in {found_count} examples.")
    if not_found_samples:
        print(f"Showing up to {len(not_found_samples)} samples where marker wasn't found:")
        for i, (p, toks) in enumerate(not_found_samples):
            print(f"  [{i+1}] prompt (truncated): {p[:200]}{'...' if len(p)>200 else ''}")
            if toks is None:
                print("       tokenization: <failed>")
            else:
                # show token ids and token strings if available
                ids_preview = toks[:80]
                print(f"       token ids (len={len(toks)}): {ids_preview}{'...' if len(toks)>80 else ''}")
                if hasattr(tokenizer, 'convert_ids_to_tokens'):
                    try:
                        toks_str = tokenizer.convert_ids_to_tokens(ids_preview)
                        print(f"       token strings: {toks_str}")
                    except Exception:
                        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to local model folder")
    p.add_argument("--probe-marker", default=None, help="Optional explicit probe marker")
    p.add_argument("--probe-debug-full", action="store_true", help="Show token strings when possible")
    p.add_argument("--sample-n", type=int, default=3, help="Samples to show per dataset where marker not found")
    p.add_argument("--append-marker", action="store_true", help="Append the probe marker to each prompt before tokenization (simulate marker present in model output)")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return

    print(f"Loading tokenizer (and model stub) from: {model_path}")
    try:
        model, tokenizer = mlx_lm.load(str(model_path))
    except Exception:
        print("mlx_lm.load failed; attempting to only load tokenizer via mlx_lm.load if available")
        traceback.print_exc()
        try:
            # try alternative attribute
            _, tokenizer = mlx_lm.load(str(model_path))
        except Exception:
            print("Unable to load tokenizer via mlx_lm.load; aborting.")
            return

    # Determine probe marker
    final_marker = args.probe_marker
    if not final_marker:
        cfg_marker = find_marker_from_config(model_path)
        if cfg_marker:
            final_marker = cfg_marker
            print(f"Found probe marker in tokenizer_config.json chat_template: {repr(final_marker)}")
        else:
            print("No probe marker provided and none found in tokenizer_config.json; will default to last token for probing.")

    try:
        diff = tokenizer_marker_diff(tokenizer, final_marker) if final_marker else None
        if diff is not None:
            print(f"Marker tokenization diff: ids={diff.get('ids')}, tokens={diff.get('tokens')}")
    except Exception:
        pass

    repo_dir = Path(__file__).resolve().parents[1]
    harmless = repo_dir / "generated_datasets" / "harmless_dataset.jsonl"
    harmful = repo_dir / "generated_datasets" / "harmful_dataset.jsonl"

    analyze_dataset(harmless, tokenizer, final_marker, sample_n=args.sample_n, name="harmless", append_marker=args.append_marker)
    analyze_dataset(harmful, tokenizer, final_marker, sample_n=args.sample_n, name="harmful", append_marker=args.append_marker)


if __name__ == "__main__":
    main()
