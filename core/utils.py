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
