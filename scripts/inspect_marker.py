#!/usr/bin/env python3
"""Inspect a model's chat_template and extract/tokenize the probe marker.

Usage: python scripts/inspect_marker.py /path/to/model_dir

This script prints the chat_template, runs the repo helper to extract the EOT marker,
and (if available) loads the tokenizer via `mlx_lm.load` to show token ids/strings for the
marker which helps pick probe modes.
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_marker.py /path/to/model_dir")
        return 2

    model_dir = Path(sys.argv[1])
    if not model_dir.exists():
        print("Model path not found:", model_dir)
        return 2

    tc = model_dir / "tokenizer_config.json"
    if not tc.exists():
        print("tokenizer_config.json not found at", tc)
        return 2

    with open(tc, "r") as f:
        cfg = json.load(f)

    chat_template = cfg.get("chat_template")
    print("chat_template:")
    print(chat_template)

    # import core.utils from repo
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from core.utils import extract_eot_from_chat_template

        marker = extract_eot_from_chat_template(chat_template) if chat_template else None
        print("\nextracted marker:", repr(marker))
    except Exception as e:
        marker = None
        print("Could not import core.utils.extract_eot_from_chat_template:", e)

    # Try to load tokenizer via mlx_lm if available
    if marker is not None:
        try:
            import mlx_lm

            model, tokenizer = mlx_lm.load(str(model_dir))
            try:
                ids = tokenizer.encode(marker, add_special_tokens=False)
                print("\nmarker token ids:", ids)
            except Exception as e:
                print("Failed to encode marker:", e)

            try:
                toks = tokenizer.convert_ids_to_tokens(ids)
                print("marker token strings:", toks)
            except Exception:
                print("tokenizer does not support convert_ids_to_tokens or conversion failed")
        except Exception as e:
            print("Could not load tokenizer via mlx_lm.load:", e)
            print("Run this script in an environment where mlx_lm is available to get tokenization output.")
    else:
        print("No marker extracted; nothing to tokenize.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
