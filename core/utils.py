"""Utility functions for the MLX Abliteration Toolkit.

This module contains miscellaneous helper functions used by other parts of
the application.

Dependencies:
- mlx
"""
import functools
from mlx.nn import Module

def get_module_from_key(model: Module, key: str) -> Module:
    """Retrieves a submodule from a model using a dot-separated key.

    For example, if the key is "model.layers.0.self_attn.o_proj.weight",
    this function will return the `o_proj` module.

    Args:
        model (Module): The container model.
        key (str): The dot-separated key to the parameter.

    Returns:
        Module: The submodule that owns the parameter.
    """
    # The key to a parameter is its path in the module tree.
    # We want the module that contains the parameter, so we split the
    # key and ignore the last part (the parameter name itself).
    module_keys = key.split('.')[:-1]

    # A common pattern is that the parameter keys start with 'model.'
    # but the model object itself is the root.
    if module_keys and module_keys[0] == 'model':
        module_keys = module_keys[1:]

    def _get_sub(mod, k):
        if not hasattr(mod, k):
            # If the key is a number, it's likely an index in a list of modules
            if k.isdigit():
                return mod[int(k)]
        return getattr(mod, k)

    return functools.reduce(_get_sub, module_keys, model)

def extract_eot_from_chat_template(template_str: str) -> str | None:
    """
    Extracts the end-of-thought marker from a Jinja2 chat template string.

    This function looks for a pattern like `</think>` or `</thought>` that
    typically follows a `message.content` block in a chat template.

    Args:
        template_str (str): The Jinja2 chat template.

    Returns:
        Optional[str]: The extracted marker (e.g., '</think>') or None if not found.
    """
    import re
    # This regex looks for a closing XML-like tag immediately following the 'message.content' part
    # It captures the content of the tag (e.g., </think>)
    match = re.search(r"\{\{.*?message\.content.*?\}\}(.*?)\{\{", template_str, re.DOTALL)
    if match:
        # The marker is the captured group, stripped of whitespace
        marker = match.group(1).strip()
        if marker:
            return marker
    return None


def tokenizer_marker_diff(tokenizer: object, marker: str) -> dict:
    """Return a small diagnostic dict describing how `tokenizer` encodes `marker`.

    The returned dict has keys:
      - marker: the literal marker string
      - ids: list of token ids produced by tokenizer.encode(marker, add_special_tokens=False)
      - tokens: list of token strings if the tokenizer exposes `convert_ids_to_tokens`, otherwise None

    This helper is intentionally minimal and safe to call in debug paths.
    """
    result = {"marker": marker, "ids": None, "tokens": None}
    if marker is None:
        return result
    try:
        ids = tokenizer.encode(marker, add_special_tokens=False)  # type: ignore
        result["ids"] = list(ids) if hasattr(ids, '__iter__') else [ids]  # type: ignore
    except Exception:
        result["ids"] = None
    try:
        if hasattr(tokenizer, "convert_ids_to_tokens") and result["ids"] is not None:
            toks = tokenizer.convert_ids_to_tokens(result["ids"])  # type: ignore
            result["tokens"] = list(toks)  # type: ignore
    except Exception:
        result["tokens"] = None
    return result


def find_probe_indices(
    tokens: list[int],
    marker_tokens: list[int] | None,
    probe_mode: str = "follow-token",
    probe_span: int = 1
) -> int | list[int]:
    """
    Finds the token index/indices to probe based on a marker and mode.

    Args:
        tokens: The list of token IDs for the prompt.
        marker_tokens: The list of token IDs for the marker. If None, defaults to last token.
        probe_mode: Strategy to select tokens ("follow-token", "marker-token", "last-token", "thinking-span").
        probe_span: Number of tokens to include for "thinking-span".

    Returns:
        tuple[int | list[int], bool]: A tuple containing:
            - The index or list of indices to probe. Returns -1 (last token) if marker not found or mode is "last-token".
            - A boolean indicating if the marker was found (always False for "last-token").
    """
    if probe_mode == "last-token" or not marker_tokens:
        return len(tokens) - 1, False

    # Search for the last occurrence of the marker by searching backwards
    marker_len = len(marker_tokens)
    for i in range(len(tokens) - marker_len, -1, -1):
        if tokens[i : i + marker_len] == marker_tokens:
            if probe_mode == "follow-token":
                potential_idx = i + marker_len
                # If marker is at the very end, fallback to the last token of the marker
                return (potential_idx if potential_idx < len(tokens) else i + marker_len - 1), True
            elif probe_mode == "marker-token":
                return i + marker_len - 1, True
            elif probe_mode == "thinking-span":
                start = i + marker_len
                if start < len(tokens):
                    end = min(len(tokens), start + probe_span)
                    return list(range(start, end)), True
                else:
                    # Fallback to marker token
                    return i + marker_len - 1, True
            break
    
    # Marker not found, fallback to last token
    return len(tokens) - 1, False
