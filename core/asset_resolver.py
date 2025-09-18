import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

# It's good practice to configure a logger for the module
# The actual configuration (level, handler) will be done in the entrypoint scripts
logger = logging.getLogger(__name__)

def resolve_asset(path_or_id: str, asset_type: str, local_cache_dir: str) -> Path:
    """
    Resolves an asset string, which can be a local path or a Hugging Face Hub ID.

    If the path is a local path, it resolves it. If it's a Hub ID, it downloads
    the asset from the Hugging Face Hub and returns the local path to the cached asset.
    This provides a single, reliable interface for all asset loading.

    Args:
        path_or_id (str): The local path or Hugging Face Hub repository ID of the asset.
        asset_type (str): The type of asset (e.g., "models", "datasets"). This is used
                          to create a subdirectory within the cache.
        local_cache_dir (str): The root directory for caching downloaded assets.

    Returns:
        Path: The resolved local filesystem path to the asset.

    Raises:
        FileNotFoundError: If the remote asset does not exist on the Hub or the local path is invalid.
        ValueError: If the asset_type is not a valid string for a directory name.
    """
    local_path = Path(path_or_id)

    if local_path.exists():
        logger.info(f"Found local asset at: {local_path.resolve()}")
        return local_path.resolve()

    logger.info(f"'{path_or_id}' not found locally. Assuming it's a Hugging Face Hub ID and attempting download.")

    if not asset_type or not asset_type.isalnum():
         raise ValueError(f"Invalid asset_type '{asset_type}'. Must be a simple string like 'models' or 'datasets'.")

    # Determine repo_type for snapshot_download. This helps huggingface_hub search efficiently.
    # We infer it from our internal `asset_type` string.
    repo_type_map = {
        "models": "model",
        "datasets": "dataset",
    }
    # Use .get() to avoid a KeyError. If not found, repo_type will be None, which is acceptable for snapshot_download.
    repo_type = repo_type_map.get(asset_type, None)

    # Construct the target directory for the download, e.g., ./.cache/models/user--repo-name
    # This keeps the cache organized and prevents collisions.
    # Using replace is a robust way to handle repo IDs with slashes.
    target_dir = Path(local_cache_dir) / asset_type / path_or_id.replace("/", "--")

    try:
        # Use local_dir to have a clean, predictable cache structure as per the spec.
        # This downloads the repo into the specified folder.
        downloaded_path = snapshot_download(
            repo_id=path_or_id,
            repo_type=repo_type,
            local_dir=target_dir,
            local_dir_use_symlinks=False,  # Creates a copy, more robust for redistribution
            resume_download=True,          # Resumes interrupted downloads
            etag_timeout=30,               # Increase timeout for stability on slower connections
        )
        logger.info(f"Successfully downloaded '{path_or_id}' to '{downloaded_path}'")
        return Path(downloaded_path)
    except HfHubHTTPError as e:
        # Re-raise as a more user-friendly FileNotFoundError for easier catching upstream.
        logger.error(f"HTTP Error: Could not download '{path_or_id}'. Check the Hub ID and your connection. Details: {e}")
        raise FileNotFoundError(f"Asset '{path_or_id}' not found on Hugging Face Hub.") from e
    except Exception as e:
        # Catch other potential errors during download (e.g., network issues, permissions)
        logger.error(f"An unexpected error occurred while trying to download '{path_or_id}': {e}")
        raise
