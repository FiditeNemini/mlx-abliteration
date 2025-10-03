#!/usr/bin/env python3
import os
# suppress huggingface/tokenizers parallelism warning in forked processes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
"""Capture activations for a few dataset prompts using ActivationProbeWrapper.

Usage:
  PYTHONPATH=/path/to/repo python scripts/probe_capture.py --model /path/to/model --marker '</think>' --n 3

This script avoids CLI wiring and resolve_asset; it directly loads the tokenizer and model
via mlx_lm.load, loads local JSONL datasets via datasets.load_dataset(..., data_files=...),
and calls ActivationProbeWrapper to capture activations for inspection.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx_lm
from datasets import load_dataset

from core.abliteration import ActivationProbeWrapper


def find_probe_idx_from_tokens(token_list, marker_ids):
    if not marker_ids:
        return -1
    m = list(marker_ids)
    for i in range(len(token_list) - len(m), -1, -1):
        if token_list[i:i+len(m)] == m:
            # if marker followed by token, use following token; if at end, use marker token
            potential_idx = i + len(m)
            if potential_idx < len(token_list):
                return potential_idx
            else:
                return i + len(m) - 1
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--marker", default=None)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print("Model path does not exist:", model_path)
        return

    print("Loading model and tokenizer from:", model_path)
    try:
        model, tokenizer = mlx_lm.load(str(model_path))
        print("Loaded model and tokenizer.")
    except Exception:
        print("mlx_lm.load failed; attempting to load tokenizer only (if available)")
        try:
            _, tokenizer = mlx_lm.load(str(model_path))
            model = None
            print("Loaded tokenizer only.")
        except Exception:
            print("Unable to load model or tokenizer via mlx_lm.load; aborting without traceback.")
            return

    marker_ids = None
    if args.marker:
        try:
            marker_ids = tokenizer.encode(args.marker, add_special_tokens=False)
            print(f"Marker {repr(args.marker)} -> ids: {marker_ids}")
        except Exception as e:
            print("Failed to encode marker:", e)

    # load model config from config.json (some MLX model objects don't expose a .config attr)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find config.json in model path: {model_path}")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # load small sample from generated datasets
    repo_dir = Path(__file__).resolve().parents[1]
    harmless_path = repo_dir / "generated_datasets" / "harmless_dataset.jsonl"
    ds = load_dataset("json", data_files=str(harmless_path))["train"]
    wrapper = ActivationProbeWrapper(model)
    num_layers = model_config.get("num_hidden_layers")
    if num_layers is None:
        # fallback to probing a single layer
        layers = [0]
    else:
        layers = list(range(num_layers))
    print("Probing first", args.n, "examples from harmless dataset")
    for idx, item in enumerate(ds):
        if idx >= args.n:
            break
        prompt = item.get("prompt") or item.get("text")
        print(f"\n[{idx+1}] prompt: {prompt[:200]}{'...' if len(prompt)>200 else ''}")
        toks = tokenizer.encode(prompt, add_special_tokens=False)
        token_list = list(toks)
        print(" token ids (len=", len(token_list), ") sample:", token_list[:40])
        probe_idx = find_probe_idx_from_tokens(token_list, marker_ids)
        print(" computed probe_idx:", probe_idx)

        tokens_arr = mx.array(token_list)
        _, captured = wrapper(tokens_arr[None], mask=None, layers_to_probe=layers)
        for layer_idx in sorted(captured.keys()):
            act = captured[layer_idx]
            # handle negative index fallback
            use_idx = probe_idx if (probe_idx >= 0 and probe_idx < act.shape[1]) else act.shape[1] - 1
            vec = act[0, use_idx, :]
            try:
                norm = float(mx.linalg.norm(vec).item())
            except Exception:
                # fallback if mx.linalg.norm not available
                norm = float((vec * vec).sum().sqrt().item()) if hasattr(vec, 'sqrt') else float((vec * vec).sum().item())
            print(f"  layer {layer_idx}: probe_index={use_idx}, norm={norm:.6f}, vec_sample={vec.tolist()[:6]}")


if __name__ == '__main__':
    main()
