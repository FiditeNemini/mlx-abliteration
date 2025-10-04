#!/usr/bin/env python3
"""Automatic probe diagnostics for a model.

This script runs locally. It:
 - Loads the model & tokenizer via mlx_lm
 - Extracts candidate markers (from chat_template.jinja or defaults)
 - Tokenizes markers and prints token ids/strings
 - Samples prompts from harmful_dataset.jsonl and for each prompt computes
   activation norms at token positions around the marker for a given layer.

Usage: python scripts/auto_probe_diagnose.py /path/to/model_dir [--samples N] [--layer IDX] [--probe-span S]

Run in the repository root with your conda env active so mlx_lm and mlx are importable.
"""
import argparse
import json
from pathlib import Path
import sys
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--samples', type=int, default=8, help='Number of prompts to sample from harmful dataset')
    parser.add_argument('--layer', type=int, default=-1, help='Layer index to probe (negative indexes count from end)')
    parser.add_argument('--probe-span', type=int, default=4, help='Span length after marker to inspect')
    parser.add_argument('--dataset', type=str, default='generated_datasets/harmful_dataset.jsonl', help='Path to harmful dataset jsonl')
    parser.add_argument('--marker', type=str, default=None, help='Optional explicit marker to use (overrides template extraction)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    try:
        import mlx.core as mx
        import mlx_lm
        from core.abliteration import ActivationProbeWrapper
        from core.utils import extract_eot_from_chat_template
    except Exception as e:
        print('Error importing MLX/cli helpers. Run this in the project env where mlx_lm is available.')
        print(e)
        return 2

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print('Model dir not found:', model_dir)
        return 2

    # Try to get chat template
    ct_path = model_dir / 'chat_template.jinja'
    chat_template = None
    if ct_path.is_file():
        chat_template = ct_path.read_text()
        print('Loaded chat_template.jinja (truncated):')
        print(chat_template[:1200])
        print('---')

    extracted = None
    if chat_template:
        try:
            extracted = extract_eot_from_chat_template(chat_template)
        except Exception:
            extracted = None

    candidate_markers = []
    if args.marker:
        candidate_markers.append(args.marker)
    if extracted:
        candidate_markers.append(extracted)
    # common candidates
    for c in ['</think>', '<|im_start|>assistant\n', '<|im_start|>assistant']:
        if c not in candidate_markers:
            candidate_markers.append(c)

    print('Candidate markers:', candidate_markers)

    print('\nLoading model/tokenizer via mlx_lm.load (may be slow)...')
    model, tokenizer = mlx_lm.load(str(model_dir))
    print('Model/tokenizer loaded.')

    # Tokenize candidates
    for m in candidate_markers:
        try:
            ids = tokenizer.encode(m, add_special_tokens=False)
            try:
                toks = tokenizer.convert_ids_to_tokens(ids)
            except Exception:
                toks = None
            print('\nMarker repr:', repr(m))
            print(' ids:', ids)
            print(' toks:', toks)
        except Exception as e:
            print('Failed to tokenize marker', repr(m), e)

    # Load sample prompts from dataset
    ds_path = Path(args.dataset)
    if not ds_path.is_file():
        print('Dataset not found at', ds_path)
        return 2

    prompts = []
    with open(ds_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.samples:
                break
            try:
                obj = json.loads(line)
                prompts.append(obj.get('prompt') or obj.get('text'))
            except Exception:
                continue

    if not prompts:
        print('No prompts loaded from dataset')
        return 2

    wrapper = ActivationProbeWrapper(model)

    # Determine numeric layer index (support negative)
    config_path = model_dir / 'config.json'
    layer_idx = args.layer
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text())
            n_layers = cfg.get('num_hidden_layers')
            if layer_idx < 0:
                layer_idx = n_layers + layer_idx
        except Exception:
            pass

    print('\nProbing {} prompts at layer {} with probe_span={}'.format(len(prompts), layer_idx, args.probe_span))

    for pi, prompt in enumerate(prompts):
        print('\n------ Prompt #{} ------'.format(pi + 1))
        short = (prompt[:400] + '...') if len(prompt) > 400 else prompt
        print(short)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # convert to mx.array
        tarr = mx.array(token_ids)
        try:
            _, captured = wrapper(tarr[None], mask=None, layers_to_probe=[layer_idx])
        except Exception as e:
            print('Wrapper forward error:', e)
            continue

        act = captured.get(layer_idx)
        if act is None:
            print('No activations captured for layer', layer_idx)
            continue

        token_list = tarr.tolist()
        # For each candidate marker, search last occurrence and compute norms over span
        for m in candidate_markers:
            try:
                m_ids = tokenizer.encode(m, add_special_tokens=False)
            except Exception:
                continue
            if not m_ids:
                continue
            # search last occurrence
            found_idx = -1
            for i in range(len(token_list) - len(m_ids), -1, -1):
                if token_list[i:i + len(m_ids)] == m_ids:
                    found_idx = i
                    break
            if found_idx == -1:
                # not found
                continue
            start = found_idx + len(m_ids)
            end = min(len(token_list), start + args.probe_span)
            indices = list(range(start, end)) if start < len(token_list) else [found_idx + len(m_ids) - 1]
            norms = []
            for idx in indices:
                if idx < 0 or idx >= act.shape[1]:
                    norms.append(float('nan'))
                else:
                    vec = act[0, idx, :]
                    try:
                        val = float(mx.linalg.norm(vec).item())
                    except Exception:
                        # fallback if mx returns numpy-like
                        try:
                            import numpy as _np
                            val = float(_np.linalg.norm(vec))
                        except Exception:
                            val = float('nan')
                    norms.append(val)
            print('\nMarker:', repr(m))
            print(' token_idx_range:', indices)
            print(' norms:', ['{:.4f}'.format(n) if not math.isnan(n) else 'nan' for n in norms])
            # recommend best index (max norm)
            try:
                valid = [(i, n) for i, n in zip(indices, norms) if not math.isnan(n)]
                if valid:
                    best = max(valid, key=lambda x: x[1])
                    print(' recommended_probe_index:', best[0], 'norm={:.4f}'.format(best[1]))
            except Exception:
                pass

    print('\nDiagnostics complete. Use --probe-mode thinking-span and probe-span around the recommended index or use marker-token if the marker itself has the highest norm.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
