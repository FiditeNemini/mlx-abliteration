"""Structured JSON logging configuration for the MLX Abliteration Toolkit.

This module provides a custom JSON formatter and a setup function to configure
a logger that writes structured JSON logs to a file.

Dependencies:
- None
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

class JsonFormatter(logging.Formatter):
    """A logging formatter that outputs log records as JSON strings.

    This formatter includes standard log information as well as any extra
    data passed to the logger.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record into a JSON string.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The JSON-formatted log string.
        """
        log_object = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "event": record.getMessage(),
        }
        if hasattr(record, "extra_info"):
            log_object.update(record.extra_info)
        return json.dumps(log_object)

def setup_structured_logging(app_name: str, level: int = logging.INFO):
    """Configures a structured JSON logger for the application.

    This function sets up a logger that writes to a file in the user's
    home directory (`~/.mlx-llm/{app_name}/log.jsonl`).

    Args:
        app_name (str): The name of the application, used to create the log directory.
        level (int): The logging level to set for the logger.
    """
    log_dir = Path(os.path.expanduser("~")) / ".mlx-llm" / app_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.jsonl"

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a file handler
    handler = logging.FileHandler(log_file)
    handler.setFormatter(JsonFormatter())

    # Add the handler to the root logger
    logger.addHandler(handler)
