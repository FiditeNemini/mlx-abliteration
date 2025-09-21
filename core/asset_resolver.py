"""Resolves assets for the MLX Abliteration Toolkit.

This module provides a unified function to handle assets, which can be either
local file paths or identifiers on the Hugging Face Hub. It supports
downloading and caching of models and datasets.

Dependencies:
- huggingface-hub
"""
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

logger = logging.getLogger(__name__)

def resolve_asset(path_or_id: str, asset_type: str, local_cache_dir: str) -> Path:
    """Resolves an asset, downloading it from the Hub if necessary.

    This function checks if the given `path_or_id` exists as a local path.
    If not, it assumes it is a Hugging Face Hub repository ID and attempts
    to download it using `snapshot_download`.

    Args:
        path_or_id (str): The local path or Hugging Face Hub repository ID.
        asset_type (str): The type of asset ("models" or "datasets"), used to
            organize the cache directory.
        local_cache_dir (str): The root directory for caching downloaded assets.

    Returns:
        Path: The resolved local filesystem path to the asset.

    Raises:
        ValueError: If `asset_type` is not a valid string for a directory name.
        FileNotFoundError: If the asset cannot be found locally or on the Hub.
    """
    local_path = Path(path_or_id)
    extra_info = {"component": "asset_resolver", "inputs": {"path_or_id": path_or_id, "asset_type": asset_type, "local_cache_dir": local_cache_dir}}

    if local_path.exists():
        logger.info(f"Found local asset at: {local_path.resolve()}", extra={"extra_info": {**extra_info, "event": "local_asset_found", "actual_output": {"resolved_path": str(local_path.resolve())}}})
        return local_path.resolve()

    logger.info(f"'{path_or_id}' not found locally. Assuming it's a Hugging Face Hub ID and attempting download.", extra={"extra_info": {**extra_info, "event": "hub_download_start"}})

    if not asset_type or not asset_type.isalnum():
         raise ValueError(f"Invalid asset_type '{asset_type}'. Must be a simple string like 'models' or 'datasets'.")

    repo_type_map = {
        "models": "model",
        "datasets": "dataset",
    }
    repo_type = repo_type_map.get(asset_type, None)

    target_dir = Path(local_cache_dir) / asset_type / path_or_id.replace("/", "--")

    try:
        downloaded_path = snapshot_download(
            repo_id=path_or_id,
            repo_type=repo_type,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            etag_timeout=30,
        )
        logger.info(f"Successfully downloaded '{path_or_id}' to '{downloaded_path}'", extra={"extra_info": {**extra_info, "event": "hub_download_success", "actual_output": {"downloaded_path": str(downloaded_path)}}})
        return Path(downloaded_path)
    except HfHubHTTPError as e:
        logger.error(f"HTTP Error: Could not download '{path_or_id}'. Check the Hub ID and your connection. Details: {e}", extra={"extra_info": {**extra_info, "event": "hub_download_http_error", "error_message": str(e)}}, exc_info=True)
        raise FileNotFoundError(f"Asset '{path_or_id}' not found on Hugging Face Hub.") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while trying to download '{path_or_id}': {e}", extra={"extra_info": {**extra_info, "event": "hub_download_unexpected_error", "error_message": str(e)}}, exc_info=True)
        raise
