#!/usr/bin/env python3
# tools/dequant_compare.py
# Usage:
# python tools/dequant_compare.py /path/to/source_model_dir /path/to/output_model_dir

import sys, json
from pathlib import Path
import numpy as np
from safetensors import safe_open
import torch

# These imports require your runtime mlx & mlx_lm to be importable
import mlx.core as mx
import mlx_lm

# Ensure repo root is on sys.path so 'core' imports resolve when running this
# script from the repository root.
from pathlib import Path as _P
_root = str((_P(__file__).resolve().parents[1]))
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.utils import get_module_from_key

def load_index(path):
    p = Path(path)
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        raise FileNotFoundError(f"Index not found: {idx}")
    return json.loads(idx.read_text())

def load_tensor_pt(shard_dir, shard_name, tensor_name):
    p = Path(shard_dir) / shard_name
    if not p.exists():
        return None
    # Use PyTorch backend so bfloat16 and other dtypes are handled
    with safe_open(str(p), framework="pt") as f:
        if tensor_name not in f.keys():
            return None
        t = f.get_tensor(tensor_name)
        # ensure torch tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return t

def dequantize_with_module(packed_w, scales, biases, module):
    # module.group_size and module.bits used by the MLX dequantize helper
    return mx.dequantize(packed_w, scales, biases, getattr(module, "group_size", 1), getattr(module, "bits", 8))

def main(src_dir, out_dir, model_dir):
    src = Path(src_dir)
    out = Path(out_dir)
    sidx = load_index(src)
    oidx = load_index(out)
    smap = sidx.get("weight_map", {})
    omap = oidx.get("weight_map", {})

    # candidate target patterns that get_ablated_parameters targets by default
    target_patterns = ["self_attn.o_proj", "mlp.down_proj", "mlp.c_proj", "mlp.up_proj", "mlp.switch_mlp.down_proj", "mlp.switch_mlp.up_proj"]
    candidates = []
    for name in smap.keys():
        if any(tp in name for tp in target_patterns):
            candidates.append(name)

    if not candidates:
        print("No candidate target tensors found in source weight map.")
        return

    # Load the model object to inspect modules (and quantization attributes)
    model, tokenizer = mlx_lm.load(str(model_dir))
    for tensor_name in sorted(candidates):
        src_shard = smap.get(tensor_name)
        out_shard = omap.get(tensor_name)
        if not src_shard or not out_shard:
            print("SKIP MISSING SHARD:", tensor_name)
            continue

        # Try to find module to get group_size/bits
        try:
            module = get_module_from_key(model, tensor_name)
        except Exception as e:
            module = None

        packed_src = load_tensor_pt(src, src_shard, tensor_name)
        packed_out = load_tensor_pt(out, out_shard, tensor_name)
        if packed_src is None or packed_out is None:
            print("SHARD/TENSOR MISSING:", tensor_name, src_shard, out_shard)
            continue

        # Try load scales/biases for dequantization
        scales_name = tensor_name.replace(".weight", ".scales")
        biases_name = tensor_name.replace(".weight", ".biases")
        scales_src = load_tensor_pt(src, src_shard, scales_name)
        biases_src = load_tensor_pt(src, src_shard, biases_name)

        scales_out = load_tensor_pt(out, out_shard, scales_name)
        biases_out = load_tensor_pt(out, out_shard, biases_name)

        if module is not None:
            try:
                # Convert torch tensors (possibly bfloat16) to numpy and then to mx.array
                def to_mx_array(t):
                    if t is None:
                        return None
                    if isinstance(t, torch.Tensor):
                        # move to CPU and convert to float32 for scales/biases if needed
                        if t.dtype == torch.bfloat16:
                            t = t.to(torch.float32)
                        return mx.array(t.cpu().numpy())
                    else:
                        return mx.array(np.array(t))

                psrc = to_mx_array(packed_src)
                pout = to_mx_array(packed_out)
                ssrc = to_mx_array(scales_src)
                bsrc = to_mx_array(biases_src)
                sout = to_mx_array(scales_out)
                bout = to_mx_array(biases_out)

                fs = dequantize_with_module(psrc, ssrc, bsrc, module)
                fo = dequantize_with_module(pout, sout, bout, module)
                diff = float(np.max(np.abs(np.array(fs) - np.array(fo))))
                print(f"{tensor_name}: dequantized max_abs_diff={diff:e} (using module attrs group_size={getattr(module,'group_size',None)} bits={getattr(module,'bits',None)})")
            except Exception as e:
                print(f"{tensor_name}: dequantize failed: {e}")
        else:
            print(f"{tensor_name}: no module info (cannot reliably dequantize).")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python tools/dequant_compare.py /path/to/source_model_dir /path/to/output_model_dir /path/to/model_dir_for_loading')
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])