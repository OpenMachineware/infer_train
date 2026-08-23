"""M7: HTTP server smoke test (OpenAI-compatible endpoints)."""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = ROOT / "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"


def _wait_ready(port, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError("server did not become ready")


@pytest.mark.slow
def test_server_endpoints():
    port = 18099
    env = dict(os.environ)
    env["INFERTRAIN_MODEL"] = str(MODEL)
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "infer_train.server", "--port", str(port)],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_ready(port)
        # /v1/models
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
            models = json.load(r)
        assert models["object"] == "list"
        assert len(models["data"]) >= 1

        # /v1/completions
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps({"prompt": "1+1=", "max_tokens": 6, "temperature": 0.6, "seed": 7}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            comp = json.load(r)
        assert comp["object"] == "text_completion"
        assert len(comp["choices"][0]["text"]) > 0

        # /v1/chat/completions
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            chat = json.load(r)
        assert chat["choices"][0]["message"]["content"]

        # /v1/finetune + status
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/finetune",
            data=json.dumps({"input": "1+1=", "target": "1+1 equals 2", "lr": 1e-5}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            job = json.load(r)
        assert job["status"] == "queued"
        for _ in range(60):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/finetune/status", timeout=30) as r:
                status = json.load(r)
            done = [j for j in status["data"] if j["id"] == job["id"]]
            if done and done[0]["status"] in ("done", "failed"):
                assert done[0]["status"] == "done"
                assert done[0]["loss"] is not None
                break
            time.sleep(2)
        else:
            pytest.fail("finetune job did not finish")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
