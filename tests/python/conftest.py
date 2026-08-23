"""Shared fixtures for the M4 Python test suite.

Run with pytest from the repo root; this file puts the `python/` package on
sys.path so `import infer_train` resolves without installation.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

GGUF = ROOT / "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"
REFERENCE_DIR = pathlib.Path(__file__).resolve().parent / "reference_outputs"
REFERENCE_GENERATE = REFERENCE_DIR / "reference_generate.txt"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: end-to-end 1.5B generation tests (slow)"
    )
    config.addinivalue_line("markers", "slow: long-running model tests")


@pytest.fixture(scope="session")
def model():
    """The 1.5B Qwen2 model loaded through the Python wrapper (once)."""
    from infer_train.model import load_model

    if not GGUF.exists():
        pytest.skip(f"{GGUF.name} not present - skipping engine e2e tests")
    m = load_model(str(GGUF))
    yield m
    m.free()


@pytest.fixture(scope="session")
def prompt():
    return "<｜User｜>What is 1+1?<｜Assistant｜><think>\n"
