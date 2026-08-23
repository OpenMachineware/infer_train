"""M7: inference-time fine-tuning API (Model.finetune) on the 1.5B model."""

import sys
import time
from pathlib import Path

import pytest

from infer_train import load_model

MODEL = Path(__file__).resolve().parent.parent.parent / \
    "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"


@pytest.mark.slow
def test_finetune_loop():
    model = load_model(str(MODEL))
    try:
        assert model.info("vocab") == 151936
        start = time.time()
        losses = model.finetune(
            "What is 1+1?", "1+1 equals 2.", lr=1e-5
        )
        elapsed = time.time() - start
        # a handful of teacher-forced steps ran and returned finite losses
        assert len(losses) >= 4, losses
        assert all(l >= 0 for l in losses), losses
        # the loss should not explode; the first steps are the noisiest
        assert losses[-1] < 50, losses
        print("finetune losses:", losses[:4], "...", losses[-1])
        print(f"finetune took {elapsed:.1f}s")
        # the adapted model still generates (regression check)
        out = model.generate("1+1=", max_tokens=8, seed=7)
        assert isinstance(out, str) and len(out) > 0
    finally:
        model.free()
