"""Small registry for model-family adapters located in this package.

This module discovers submodules that provide a `Model` symbol and exposes
helpers to list available families and return minimal adapter metadata.

The registry is intentionally small and read-only: it does not modify any
model code. It is used by higher-level code to pick per-family heuristics
(recommended probe marker, probe-mode, etc.).
"""
from __future__ import annotations

from typing import Dict, List
from pathlib import Path

# Package path on disk (this module lives in core/vendor/mlx_lm/models)
HERE = Path(__file__).resolve().parent

# Per-family heuristic metadata (add more families here as needed)
FAMILY_METADATA: Dict[str, Dict] = {
    # qwen variants
    "qwen": {},
    "qwen2": {},
    "qwen2_moe": {},
    "qwen2_vl": {},
    "qwen3": {},
    "qwen3_moe": {},
    "qwen3_next": {"recommended_probe_marker": "</think>", "probe_mode": "thinking-span", "probe_span": 1},

    # LLaMA family
    "llama": {},
    "llama4": {},
    "llama4_text": {},

    # Mistral / Mixtral
    "mistral3": {},
    "mixtral": {},

    # Gemma family
    "gemma": {},
    "gemma2": {},
    "gemma3": {},
    "gemma3_text": {},
    "gemma3n": {},

    # Phi / Phi3
    "phi": {},
    "phi3": {},
    "phi3small": {},

    # Falcon / Starcoder / Code models
    "falcon_h1": {},
    "starcoder2": {},

    # GPT family
    "gpt2": {},
    "gpt_neox": {},
    "gpt_bigcode": {},

    # Other families seen upstream
    "baichuan_m1": {},
    "cohere": {},
    "internlm2": {},
    "internlm3": {},
    "stablelm": {},
    "openelm": {},
    "plamo": {},
    "mamba": {},
    "mamba2": {},
    "smollm3": {},
    "exaone": {},
    "exaone4": {},
    "deepseek": {},
    "deepseek_v2": {},
    "deepseek_v3": {},
    "granite": {},

    # Keep this list extensible; heuristics may be added per-family as we learn defaults
}


def discover_families() -> List[str]:
    """Return candidate family module basenames under this package.

    This function does not import modules to avoid triggering any heavy
    or native initializations (e.g., `mlx.core`). It simply lists .py
    files and treats non-utility modules as families.
    """
    families: List[str] = []
    for p in HERE.iterdir():
        if p.suffix != ".py":
            continue
        name = p.stem
        if name.startswith("_"):
            continue
        # skip utility modules
        if name in {"base", "cache", "gated_delta", "rope_utils", "switch_layers", "registry", "__init__"}:
            continue
        families.append(name)
    return sorted(families)


def get_family_metadata(family_name: str) -> Dict:
    """Return metadata for a known family; empty dict if unknown."""
    return FAMILY_METADATA.get(family_name, {})


def get_available_families() -> List[str]:
    """Public alias: list of candidate family names (module basenames)."""
    return discover_families()


def get_adapter_for_family(family_name: str) -> Dict:
    """Return a minimal adapter metadata dict for a family without importing it.

    The returned dict contains:
      - `module_name`: the importable module path as a string (e.g. core.vendor.mlx_lm.models.qwen3_next)
      - `family`: family_name
      - any known probe heuristics from FAMILY_METADATA
    """
    mod_path = f"core.vendor.mlx_lm.models.{family_name}"
    meta = get_family_metadata(family_name).copy()
    meta.setdefault("module_name", mod_path)
    meta.setdefault("family", family_name)
    return meta


__all__ = ["get_available_families", "get_adapter_for_family", "get_family_metadata"]
