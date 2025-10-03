#!/usr/bin/env python3
import os
# suppress huggingface/tokenizers parallelism warning in forked processes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
"""Run the CLI's run_abliteration with controlled args and print full traceback on error.

This bypasses the normal argparse-driven entrypoint so we can capture exceptions
and logs when running in the diagnostic environment.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli

def make_args():
    class A:
        pass
    a = A()
    # configure the same args used previously
    a.model = "/Users/willdee/AI/FiditeNemini/Qwen3-Next-80B-A3B-Thinking-8bit"
    a.harmless_dataset = "./generated_datasets/harmless_dataset.jsonl"
    a.harmful_dataset = "./generated_datasets/harmful_dataset.jsonl"
    a.output_dir = "./outputs/diag_out"
    a.cache_dir = ".cache"
    a.layers = "all"
    a.use_layer = -1
    a.ablation_strength = 0.0
    a.probe_marker = "</think>"
    a.probe_debug = True
    a.probe_debug_n = 3
    a.probe_debug_full = True
    a.verbose = True
    return a

def main():
    args = make_args()
    try:
        cli.run_abliteration(args)
        print("run_abliteration completed")
    except Exception:
        print("Exception in run_abliteration:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
