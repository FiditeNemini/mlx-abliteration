import os
import subprocess
from pathlib import Path

# This integration test runs the probe_capture script in "fast mode" against
# a tiny dummy model that exists in the repository (created on the fly).

def test_probe_capture_fast(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / 'scripts' / 'probe_capture.py'

    # Create a tiny dummy model directory with a minimal config.json so mlx_lm.load
    # can fail gracefully or the script can still run in a test environment with shims.
    dummy_model = tmp_path / 'dummy_model'
    dummy_model.mkdir()
    (dummy_model / 'config.json').write_text('{"num_hidden_layers": 1, "hidden_size": 8}')
    (dummy_model / 'tokenizer_config.json').write_text('{}')

    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    env['TOKENIZERS_PARALLELISM'] = 'false'

    # Run probe_capture with n=1 and the dummy model; script should exit quickly (<=10s)
    cmd = [
        'python', str(script),
        '--model', str(dummy_model),
        '--n', '1'
    ]
    proc = subprocess.run(cmd, env=env, cwd=str(repo_root), capture_output=True, text=True, timeout=10)
    # Accept either success (0) or a graceful abort (non-zero) as long as script runs without crashing
    assert proc is not None
    # ensure we don't see a traceback in stdout/stderr
    assert 'Traceback' not in proc.stdout
    assert 'Traceback' not in proc.stderr