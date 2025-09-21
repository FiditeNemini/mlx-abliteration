import sys
from pathlib import Path
from unittest.mock import patch
import pytest

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli import parse_args

def test_parse_args():
    """Tests the command-line argument parsing."""
    test_args = [
        "cli.py",
        "--model", "test-model",
        "--output-dir", "test-output",
        "--layers", "1,2,3",
        "--use-layer", "2",
        "--verbose"
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.model == "test-model"
        assert args.output_dir == "test-output"
        assert args.layers == "1,2,3"
        assert args.use_layer == 2
        assert args.verbose is True
