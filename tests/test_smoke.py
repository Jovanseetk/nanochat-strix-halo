"""Exercise eager training and cached inference without datasets or a tokenizer."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_eager_training_and_cached_inference(dtype):
    env = dict(os.environ, NANOCHAT_COMPILE="0", NANOCHAT_DTYPE=dtype)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.smoke_test", "--device-type", "cpu"],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Smoke test passed" in result.stdout


def test_required_rocm_cannot_silently_fall_back_to_cpu():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.smoke_test", "--device-type", "cpu", "--require-rocm"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "ROCm GPU is required" in result.stderr
