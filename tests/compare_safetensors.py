#!/usr/bin/env python3
# tools/compare_safetensors.py
# Usage: python tools/compare_safetensors.py /path/to/source_model_dir /path/to/output_model_dir

import sys, json
from pathlib import Path
import numpy as np
from safetensors import safe_open

def load_index(path):
    p = Path(path)
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        raise FileNotFoundError(f"Index not found: {idx}")
    return json.loads(idx.read_text())

def load_tensor(shard_dir, shard_name, tensor_name):
    p = Path(shard_dir) / shard_name
    if not p.exists():
        return None
    # Use numpy backend to avoid importing mlx core in this diagnostic script
    with safe_open(str(p), framework="np") as f:
        if tensor_name not in f.keys():
            return None
        return f.get_tensor(tensor_name)

def main(src_dir, out_dir):
    src = Path(src_dir)
    out = Path(out_dir)
    sidx = load_index(src)
    oidx = load_index(out)
    smap = sidx.get("weight_map", {})
    omap = oidx.get("weight_map", {})

    targets = ['self_attn.o_proj.weight','mlp.down_proj.weight','mlp.c_proj.weight','mlp.up_proj.weight']
    pairs = {}
    for name, fname in smap.items():
        for t in targets:
            if t in name:
                pairs.setdefault(name, {})['src'] = fname
    for name, fname in omap.items():
        for t in targets:
            if t in name:
                pairs.setdefault(name, {})['out'] = fname

    if not pairs:
        print("No target tensors found to compare")
        return

    for tensor_name, files in pairs.items():
        src_file = files.get('src')
        out_file = files.get('out')
        if not src_file:
            print("SRC MISSING", tensor_name)
            continue
        if not out_file:
            print("OUT MISSING", tensor_name)
            continue
        src_arr = load_tensor(src, src_file, tensor_name)
        out_arr = load_tensor(out, out_file, tensor_name)
        if src_arr is None:
            print("SRC SHARD MISSING", tensor_name, src_file)
            continue
        if out_arr is None:
            print("OUT SHARD MISSING", tensor_name, out_file)
            continue
        try:
            # Coerce to float64 for a stable numeric comparison. Some layers
            # may be stored in low-precision integer or packed formats; converting
            # to float avoids overflow/underflow artifacts when diffing.
            srcf = np.array(src_arr, dtype=np.float64)
            outf = np.array(out_arr, dtype=np.float64)
            diff = float(np.max(np.abs(srcf - outf)))
        except Exception as e:
            print("ERR diff", tensor_name, e)
            continue
        print(f"{tensor_name}: max_abs_diff={diff:e}, src_file={src_file}, out_file={out_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/compare_safetensors.py /path/to/source_model_dir /path/to/output_model_dir")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])