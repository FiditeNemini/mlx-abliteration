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
    # This regex finds the part immediately following the `message.content` block
    # and captures everything up to the next `{{` or the end of the string. This
    # lets the function extract markers that are XML-like (e.g. </think>) or
    # special tokens such as '<|im_start|>assistant\n' which may include newline
    # characters. The use of (?=\{\{|\Z) ensures we stop before the next Jinja
    # placeholder or at end-of-template.
    match = re.search(r"\{\{.*?message\.content.*?\}\}(.*?)(?=\{\{|\Z)", template_str, re.DOTALL)
    if not match:
        return None

    raw = match.group(1)
    # If the captured string is only whitespace, treat it as no marker found.
    if raw is None:
        return None
    if raw.strip() == "":
        return None

    # Preserve the captured text (including newlines) since some markers rely on
    # trailing newline characters for correct tokenization. Trim only terminal
    # spaces but keep newlines intact at the end of the marker.
    # Strategy: rstrip spaces but not newlines.
    marker = raw.rstrip(' ')  # remove trailing spaces, preserve newlines
    return marker


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
        ids = tokenizer.encode(marker, add_special_tokens=False)
        result["ids"] = list(ids) if hasattr(ids, '__iter__') else [ids]
    except Exception:
        result["ids"] = None
    try:
        if hasattr(tokenizer, "convert_ids_to_tokens") and result["ids"] is not None:
            toks = tokenizer.convert_ids_to_tokens(result["ids"])
            result["tokens"] = list(toks)
    except Exception:
        result["tokens"] = None
    return result


def normalize_marker(marker: str, strip_trailing_newline: bool = False) -> str:
    """Normalize a marker string for use in probing.

    Args:
        marker: the raw extracted marker string (may include newlines and spaces)
        strip_trailing_newline: if True, remove a single trailing newline character
            from the marker. This is useful for tokenizers that don't include a
            trailing newline in the tokenization of markers.

    Returns:
        The normalized marker string.
    """
    if marker is None:
        return marker
    if strip_trailing_newline:
        # Remove exactly one trailing newline if present, and also remove
        # trailing carriage return for windows-style endings.
        if marker.endswith("\r\n"):
            return marker[:-2]
        if marker.endswith("\n") or marker.endswith("\r"):
            return marker[:-1]
    # Default: trim only trailing spaces (preserve newlines unless stripped)
    return marker.rstrip(' ')
