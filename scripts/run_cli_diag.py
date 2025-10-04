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
import argparse


def make_args(model: str, output_dir: str = "./outputs/diag_out", probe_marker: str = "</think>"):
    class A:
        pass

    a = A()
    # configure the same args used previously but take sensitive values from CLI
    a.model = model
    a.harmless_dataset = "./generated_datasets/harmless_dataset.jsonl"
    a.harmful_dataset = "./generated_datasets/harmful_dataset.jsonl"
    a.output_dir = output_dir
    a.cache_dir = ".cache"
    a.layers = "all"
    a.use_layer = -1
    a.ablation_strength = 0.0
    a.probe_marker = probe_marker
    a.probe_debug = True
    a.probe_debug_n = 3
    a.probe_debug_full = True
    # new CLI options (defaults)
    a.probe_mode = "follow-token"
    a.probe_span = 1
    a.ablate_k = 1
    a.ablate_method = "projection"
    a.pca_sample = 512
    return a

def main():
    # parse required model path from command-line to avoid hard-coded sensitive paths
    parser = argparse.ArgumentParser(description="Run CLI diagnostic dry-run for abliteration.")
    parser.add_argument("-m", "--model", required=True, help="Path or hub id to the source model (required).")
    parser.add_argument("-o", "--output-dir", default="./outputs/diag_out", help="Output directory for suggestions (default: ./outputs/diag_out)")
    parser.add_argument("--probe-marker", default="</think>", help="Optional probe marker token to use (default: </think>)")
    parsed = parser.parse_args()

    args = make_args(model=parsed.model, output_dir=parsed.output_dir, probe_marker=parsed.probe_marker)
    try:
        # Request the CLI to return mean activations so we don't re-run probes here
        setattr(args, "return_means", True)
        result = cli.run_abliteration(args)
        print("run_abliteration completed")
        if result and isinstance(result, dict) and "harmful_mean_activations" in result:
            import json
            import numpy as _np
            # Use returned means to compute per-layer diffs
            harmful_means = result["harmful_mean_activations"]
            harmless_means = result["harmless_mean_activations"]
            model_config = result.get("model_config", {})
            num_layers = model_config.get("num_hidden_layers") or max(int(k) for k in harmful_means.keys()) + 1
            layer_stats = []
            for layer_str in sorted(harmful_means.keys(), key=lambda x: int(x)):
                layer = int(layer_str)
                hm = harmful_means.get(layer_str) or harmful_means.get(str(layer))
                lim = harmless_means.get(layer_str) or harmless_means.get(str(layer))
                if hm is None or lim is None:
                    diff_norm = None
                else:
                    try:
                        hm_arr = _np.array(hm)
                        li_arr = _np.array(lim)
                        diff_norm = float(_np.linalg.norm(hm_arr - li_arr))
                    except Exception:
                        diff_norm = None
                layer_stats.append({"layer": layer, "diff_norm": diff_norm})

            # recommend top-10 layers
            valid = [s for s in layer_stats if s["diff_norm"] is not None]
            top10 = sorted(valid, key=lambda x: x["diff_norm"], reverse=True)[:10]

            suggestions = {
                "model": args.model,
                "layers": args.layers,
                "use_layer": args.use_layer,
                "probe_marker": args.probe_marker,
                "probe_mode": "thinking-span" if args.probe_marker else "follow-token",
                "probe_span": getattr(args, "probe_span", 1),
                "ablation_strength": args.ablation_strength,
                "ablate_k": args.ablate_k,
                "ablate_method": args.ablate_method,
                "layer_stats": layer_stats,
                "recommended_layers_top10": [s["layer"] for s in top10],
            }

            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "dry_run_suggestions.json"
            with open(report_path, "w") as f:
                json.dump(suggestions, f, indent=2)

            # Also write a human-readable CSV summary
            try:
                import csv
                csv_path = out_dir / "dry_run_layer_stats.csv"
                with open(csv_path, "w", newline="") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["layer", "diff_norm"])
                    for s in layer_stats:
                        writer.writerow([s["layer"], s["diff_norm"] if s["diff_norm"] is not None else ""])
                print(f"Wrote dry-run suggestions to: {report_path}")
                print(f"Wrote CSV summary to: {csv_path}")
                print("Top recommended layers (top-10):", [s["layer"] for s in top10])
            except Exception as e:
                print("Wrote JSON suggestions but failed to write CSV summary:", e)
        else:
            print("CLI did not return mean activations; falling back to internal recompute path")
            # Fallback: original recompute logic (keep previous behavior) — for brevity call script again without return_means
            setattr(args, "return_means", False)
            cli.run_abliteration(args)
    except Exception:
        print("Exception in run_abliteration:")
        traceback.print_exc()
    except Exception:
        print("Exception in run_abliteration:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
