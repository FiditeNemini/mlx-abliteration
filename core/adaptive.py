"""Adaptive ablation utilities.

This module provides functionality to *dynamically* choose an ablation strength
while the pipeline is running. The goal is to reduce a surrogate metric that
measures how much discriminative signal along the computed refusal direction
remains after orthogonalizing targeted weights.

Rationale
---------
In some models (especially those that include internal "thinking" or
chain-of-thought style latent phases) a fixed ablation strength may be either
insufficient (behavior persists) or overly aggressive (collateral capability
loss). An online search over `ablation_strength` using a small evaluation
subset provides a cheap, model-agnostic heuristic:

    metric = alignment(mean_harmful - mean_harmless, refusal_vector)

Where `alignment` is (by default) the summed absolute inner product(s) between
the harmful/harmless activation mean difference for the chosen layer and each
component of the refusal vector. When `ablate_k > 1`, each component is treated
independently and their absolute projections are summed.

Search Strategy
---------------
We perform a multiplicative growth search:

1. Measure baseline alignment with the *unablated* model.
2. For a trial strength `s`, apply ablation (starting from the original base
   weights each iteration), re-measure alignment, compute ratio vs baseline.
3. Stop early when ratio <= target_ratio (e.g. 0.2 for 80% reduction) or when
   max iterations are reached. Track the best (lowest) ratio encountered.

This is intentionally simple and robust for integration into the existing CLI.

Returned diagnostics include the schedule of tried strengths and their ratios
so callers can audit the search afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Sequence
import logging

import mlx.core as mx

from .abliteration import get_ablated_parameters, ActivationProbeWrapper
from .abliteration import evaluate_refusal_behavior
from .utils import tokenizer_marker_diff

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveAblationResult:
    chosen_strength: float
    baseline_alignment: float
    final_alignment: float
    final_ratio: float
    tried: List[dict]
    target_ratio: float
    iterations: int


def _normalize_components(refusal_vector: mx.array) -> mx.array:
    """Return a (K,H) array of unit component vectors.

    Supports single vector shape (H,) by expanding to (1,H).
    """
    rv = refusal_vector
    if rv.ndim == 1:
        rv = rv[None, :]
    normed = []
    for i in range(rv.shape[0]):
        v = rv[i]
        n = mx.linalg.norm(v) + 1e-9
        normed.append(v / n)
    return mx.stack(normed, axis=0)


def _alignment(diff_vec: mx.array, refusal_vector: mx.array) -> float:
    """Compute summed absolute projection magnitude of diff onto refusal components.

    diff_vec: shape (H,)
    refusal_vector: shape (K,H) or (H,)
    """
    comps = _normalize_components(refusal_vector)
    # (K,H) * (H,) -> (K,)
    dots = (comps @ diff_vec)
    try:
        val = float(mx.sum(mx.abs(dots)).item())
    except Exception:
        val = float(mx.sum(mx.abs(dots)))
    return val


def _collect_mean_activation_for_layer(
    dataset_subset: Sequence[dict],
    wrapper: ActivationProbeWrapper,
    tokenizer: Any,
    layer_idx: int,
    config: Dict[str, Any],
    probe_marker: str | None,
    probe_mode: str,
    probe_span: int,
) -> mx.array:
    """Compute mean activation vector for a *single* layer over a small subset.

    This intentionally duplicates a *minimal* subset of logic from
    `cli.get_mean_activations` to avoid the progress bar noise and the overhead
    of computing for multiple layers.
    """
    hidden_size = config["hidden_size"]
    mean = mx.zeros(hidden_size)
    count = 0
    max_seq_len = config.get("max_position_embeddings", 4096)

    marker_tokens = None
    marker_list = None
    if probe_marker and probe_marker.strip():
        try:
            marker_tokens = mx.array(tokenizer.encode(probe_marker, add_special_tokens=False))
            marker_list = marker_tokens.tolist()
        except Exception:
            marker_tokens = None
            marker_list = None

    for item in dataset_subset:
        prompt = item.get("prompt") or item.get("text")
        if not prompt:
            continue
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        _, captured = wrapper(tokens[None], mask=None, layers_to_probe=[layer_idx])
        act = captured.get(layer_idx)
        if act is None:
            continue

        probe_idx = -1
        probe_idx_list = None
        if marker_list:
            token_list = tokens.tolist()
            for i in range(len(token_list) - len(marker_list), -1, -1):
                if token_list[i:i+len(marker_list)] == marker_list:
                    if probe_mode == "follow-token":
                        pidx = i + len(marker_list)
                        probe_idx = pidx if pidx < len(token_list) else i + len(marker_list) - 1
                    elif probe_mode == "marker-token":
                        probe_idx = i + len(marker_list) - 1
                    elif probe_mode == "thinking-span":
                        start = i + len(marker_list)
                        if start < len(token_list):
                            end = min(len(token_list), start + probe_span)
                            probe_idx_list = list(range(start, end))
                        else:
                            probe_idx = i + len(marker_list) - 1
                    elif probe_mode == "last-token":
                        probe_idx = len(token_list) - 1
                    break

        if probe_idx_list is not None:
            valid = [idx for idx in probe_idx_list if 0 <= idx < act.shape[1]]
            if valid:
                vec = act[0, valid, :].mean(axis=0)
            else:
                vec = act[0, -1, :]
        else:
            use_idx = probe_idx if (0 <= probe_idx < act.shape[1]) else act.shape[1] - 1
            vec = act[0, use_idx, :]

        count += 1
        delta = vec - mean
        mean = mean + delta / count

    mx.eval(mean)
    return mean


def compute_alignment_metric(
    harmful_subset: Sequence[dict],
    harmless_subset: Sequence[dict],
    wrapper: ActivationProbeWrapper,
    tokenizer: Any,
    layer_idx: int,
    config: Dict[str, Any],
    refusal_vector: mx.array,
    probe_marker: str | None,
    probe_mode: str,
    probe_span: int,
) -> float:
    """Return the alignment metric for a small evaluation subset."""
    harm_mean = _collect_mean_activation_for_layer(
        harmful_subset, wrapper, tokenizer, layer_idx, config, probe_marker, probe_mode, probe_span
    )
    harmless_mean = _collect_mean_activation_for_layer(
        harmless_subset, wrapper, tokenizer, layer_idx, config, probe_marker, probe_mode, probe_span
    )
    diff = harm_mean - harmless_mean
    return _alignment(diff, refusal_vector)


def adaptive_search_ablation_strength(
    model,
    wrapper: ActivationProbeWrapper,
    tokenizer: Any,
    harmful_dataset,
    harmless_dataset,
    refusal_vector: mx.array,
    layer_idx: int,
    config: Dict[str, Any],
    *,
    initial_strength: float = 0.5,
    max_strength: float = 8.0,
    growth: float = 1.5,
    target_ratio: float = 0.2,
    max_iters: int = 6,
    eval_samples: int = 64,
    probe_marker: str | None = None,
    probe_mode: str = "follow-token",
    probe_span: int = 1,
    ablation_method: str = "projection",
    use_generation_metric: bool = False,
    gen_prompts: Sequence[str] | None = None,
    gen_eval_max_new_tokens: int = 64,
    gen_eval_top_k: int = 1,
    fine_search: bool = True,
    fine_grid: Sequence[float] | None = None,
) -> AdaptiveAblationResult:
    """Run adaptive multiplicative search for ablation strength.

    Returns an `AdaptiveAblationResult` with diagnostics. The model is left in
    the *state of the final (chosen) ablation* when this function returns.
    """
    # Prepare evaluation subsets (shallow slicing is cheap for HF datasets)
    harm_subset = [harmful_dataset[i] for i in range(min(eval_samples, len(harmful_dataset)))]
    harmless_subset = [harmless_dataset[i] for i in range(min(eval_samples, len(harmless_dataset)))]

    # Compute baseline metric (no ablation yet)
    baseline_alignment = compute_alignment_metric(
        harm_subset,
        harmless_subset,
        wrapper,
        tokenizer,
        layer_idx,
        config,
        refusal_vector,
        probe_marker,
        probe_mode,
        probe_span,
    )
    logger.info(
        f"Adaptive ablation baseline alignment={baseline_alignment:.6f}",
        extra={
            "extra_info": {
                "event": "adaptive_baseline_metric",
                "actual_output": {"baseline_alignment": float(baseline_alignment)},
            }
        },
    )

    # Capture *original* parameters so each trial starts from the same base.
    from mlx_lm.utils import tree_flatten

    base_flat = tree_flatten(model.parameters())
    base_param_map = dict(base_flat)

    # Identify target module keys (reuse default patterns from get_ablated_parameters)
    target_patterns = [
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mlp.down_proj",
        "mlp.c_proj",
        "mlp.up_proj",
        "mlp.switch_mlp.down_proj",
        "mlp.switch_mlp.up_proj",
    ]
    def _is_target(k: str) -> bool:
        return any(tp in k for tp in target_patterns) and k.endswith("weight")

    original_target_params = {k: v for k, v in base_param_map.items() if _is_target(k)}

    tried: List[dict] = []
    strength = initial_strength
    best_ratio = float("inf")
    best_strength = strength
    best_alignment = baseline_alignment

    for iteration in range(1, max_iters + 1):
        # Revert only targeted weights (full restore can fail on some composite models)
        if iteration > 1:
            try:
                model.update(original_target_params)
                mx.eval(model.parameters())
            except Exception:
                logger.debug("Failed to restore original target params; continuing", exc_info=True)

        trial_params = get_ablated_parameters(
            model,
            refusal_vector,
            ablation_strength=strength,
            ablation_method=ablation_method,
        )
        model.update(trial_params)
        mx.eval(model.parameters())

        current_alignment = compute_alignment_metric(
            harm_subset,
            harmless_subset,
            wrapper,
            tokenizer,
            layer_idx,
            config,
            refusal_vector,
            probe_marker,
            probe_mode,
            probe_span,
        )
        # Optionally compute a generation-based metric as an alternative
        gen_refusal_rate = None
        if use_generation_metric:
            try:
                prompts = gen_prompts or [d.get("prompt") or d.get("text") for d in harm_subset[:8]]
                eval_res = evaluate_refusal_behavior(model, tokenizer, prompts, max_new_tokens=gen_eval_max_new_tokens, top_k=gen_eval_top_k)
                gen_refusal_rate = float(eval_res.get("refusal_rate", 0.0))
            except Exception:
                gen_refusal_rate = None

        # Choose which metric to use for early stopping / comparison
        if use_generation_metric and gen_refusal_rate is not None:
            metric_value = gen_refusal_rate
            metric_name = "gen_refusal_rate"
            # lower is better (we want fewer refusals)
            ratio = metric_value / (1.0 if metric_value > 0 else 1.0)
        else:
            metric_value = float(current_alignment)
            metric_name = "alignment"
            ratio = (metric_value / baseline_alignment) if baseline_alignment > 0 else 0.0
        ratio = (current_alignment / baseline_alignment) if baseline_alignment > 0 else 0.0

        tried.append(
            {
                "iteration": iteration,
                "strength": strength,
                "alignment": float(current_alignment),
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "ratio": float(ratio),
            }
        )
        logger.info(
            f"Adaptive trial {iteration}: strength={strength:.4f} alignment={current_alignment:.6f} ratio={ratio:.4f}",
            extra={
                "extra_info": {
                    "event": "adaptive_trial",
                    "actual_output": {
                        "iteration": iteration,
                        "strength": float(strength),
                        "alignment": float(current_alignment),
                        "ratio": float(ratio),
                    },
                }
            },
        )

        if ratio < best_ratio:
            best_ratio = ratio
            best_strength = strength
            best_alignment = current_alignment

        # Early stop if below target
        if ratio <= target_ratio:
            logger.info(
                "Adaptive search reached target ratio; stopping early",
                extra={
                    "extra_info": {
                        "event": "adaptive_early_stop",
                        "actual_output": {"ratio": float(ratio), "target_ratio": target_ratio},
                    }
                },
            )
            break

        # Prepare next strength
        if strength >= max_strength:
            logger.info(
                "Adaptive search hit max_strength bound; stopping",
                extra={
                    "extra_info": {
                        "event": "adaptive_max_strength_stop",
                        "actual_output": {"strength": float(strength)},
                    }
                },
            )
            break
        strength = min(strength * growth, max_strength)

    # Ensure model holds best choice (may differ from last tried)
    # NOTE: We previously attempted to restore the *entire* parameter map to the
    # original flattened snapshot and then re-apply ablation at best_strength.
    # On very large / composite models (e.g., MoE or models whose internal
    # parameter registration changes after first forward passes) this triggered
    # errors like: "Module does not have parameter named 'model.embed_tokens.weight'".
    # That full restore is unnecessary because during the search loop we only ever
    # mutate a restricted set of targeted weight matrices. So here we just revert
    # those targeted weights (if the final chosen strength differs from the last
    # tried) and re-apply the ablation with best_strength. This avoids brittle
    # assumptions about the stability of the full flattened param key space.
    if tried and best_strength != tried[-1]["strength"]:
        # Safe, per-key targeted restore
        for k, v in original_target_params.items():
            try:
                model.update({k: v})
            except Exception:
                logger.debug("Skipping restore for missing key during finalization: %s", k, exc_info=True)
        mx.eval(model.parameters())

        final_params = get_ablated_parameters(
            model,
            refusal_vector,
            ablation_strength=best_strength,
            ablation_method=ablation_method,
        )
        # Apply only the trial's targeted param deltas (ignore any unexpected keys)
        for k, v in final_params.items():
            try:
                model.update({k: v})
            except Exception:
                logger.debug("Skipping final update for key: %s", k, exc_info=True)
        mx.eval(model.parameters())

    result = AdaptiveAblationResult(
        chosen_strength=float(best_strength),
        baseline_alignment=float(baseline_alignment),
        final_alignment=float(best_alignment),
        final_ratio=float(best_ratio),
        tried=tried,
        target_ratio=target_ratio,
        iterations=len(tried),
    )
    logger.info(
        "Adaptive ablation complete",
        extra={
            "extra_info": {
                "event": "adaptive_complete",
                "actual_output": {
                    "chosen_strength": result.chosen_strength,
                    "final_ratio": result.final_ratio,
                    "iterations": result.iterations,
                },
            }
        },
    )

    # Optional fine-grained local search around the chosen_strength
    if fine_search and tried:
        try:
            # Build local grid if not provided
            if fine_grid is None:
                base = best_strength
                deltas = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]
                fine_grid = sorted([round(base + d, 6) for d in deltas if base + d > 0])

            fine_tried = []
            for s in fine_grid:
                # restore targets
                for k, v in original_target_params.items():
                    try:
                        model.update({k: v})
                    except Exception:
                        pass
                params = get_ablated_parameters(model, refusal_vector, ablation_strength=float(s), ablation_method=ablation_method)
                for k, v in params.items():
                    try:
                        model.update({k: v})
                    except Exception:
                        pass
                # evaluate
                if use_generation_metric:
                    prompts = gen_prompts or [d.get("prompt") or d.get("text") for d in harm_subset[:8]]
                    eval_res = evaluate_refusal_behavior(model, tokenizer, prompts, max_new_tokens=gen_eval_max_new_tokens, top_k=gen_eval_top_k)
                    val = float(eval_res.get("refusal_rate", 0.0))
                else:
                    val = float(compute_alignment_metric(harm_subset, harmless_subset, wrapper, tokenizer, layer_idx, config, refusal_vector, probe_marker, probe_mode, probe_span))
                fine_tried.append({"strength": float(s), "metric_value": val})

            # pick best from fine grid (lowest metric_value)
            fine_tried.sort(key=lambda x: x["metric_value"])
            best_fine = fine_tried[0]
            # If fine-found better strength, apply it finally
            if best_fine["strength"] != best_strength:
                for k, v in original_target_params.items():
                    try:
                        model.update({k: v})
                    except Exception:
                        pass
                final_params = get_ablated_parameters(model, refusal_vector, ablation_strength=best_fine["strength"], ablation_method=ablation_method)
                for k, v in final_params.items():
                    try:
                        model.update({k: v})
                    except Exception:
                        pass
                mx.eval(model.parameters())
                result.chosen_strength = float(best_fine["strength"])
                logger.info("Adaptive fine search updated chosen_strength", extra={"extra_info": {"event": "adaptive_fine_update", "actual_output": {"chosen_strength": result.chosen_strength}}})
            result.tried = tried + [{"fine_grid": fine_tried}]
        except Exception:
            logger.debug("Fine-grained local search failed; returning coarse result", exc_info=True)
    return result


__all__ = [
    "adaptive_search_ablation_strength",
    "compute_alignment_metric",
    "AdaptiveAblationResult",
]
