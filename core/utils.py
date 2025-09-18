# Copyright © 2023-2024 Apple Inc.

import functools
from mlx.nn import Module

def get_module_from_key(model: Module, key: str) -> Module:
    """
    Retrieves a submodule from a model using a dot-separated key.

    For example, if the key is "model.layers.0.self_attn.o_proj.weight",
    this function will return the `o_proj` module.

    Args:
        model: The container model.
        key: The dot-separated key to the parameter.

    Returns:
        The submodule that owns the parameter.
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
