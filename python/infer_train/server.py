"""M7: the infer_train HTTP server (OpenAI/Claude-compatible surface).

Run::

    INFERTRAIN_MODEL=model.gguf python -m infer_train.server \
        [--host 127.0.0.1] [--port 8080] [--ctx-size 512]

Endpoints (OpenAI-compatible):
  * GET  /v1/models                - available models
  * POST /v1/chat/completions      - chat completion (messages array)
  * POST /v1/completions           - text completion (+ `stream: true` SSE)
  * POST /v1/finetune              - queue an inference-time fine-tune job
  * GET  /v1/finetune/status       - job progress / losses
  * GET  /health                   - liveness probe

Private conventions:
  * API key auth is enabled when the env var ``INFERTRAIN_API_KEY`` is set
    (``Authorization: Bearer <key>`` or ``x-api-key: <key>``).
  * Streaming uses the engine's token-at-a-time logits export
    (`infer_train_forward_logits`) with a small Python sampler.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import model as _model_mod
from .binding import _lib

_FINETUNE_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()


def _api_key_ok(headers) -> bool:
    key = os.environ.get("INFERTRAIN_API_KEY", "")
    if not key:
        return True
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):] == key
    return headers.get("x-api-key", "") == key


def _sample_token(logits, temp, top_k, top_p, rng):
    if temp <= 0:
        return int(max(range(len(logits)), key=lambda i: logits[i]))
    vals = logits
    if top_k > 0:
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        keep = set(order[:top_k])
        vals = [v if i in keep else float("-inf") for i, v in enumerate(vals)]
    mx = max(vals)
    exps = [math.exp((v - mx) / temp) for v in vals]
    total = sum(exps)
    if top_p < 1.0:
        pairs = sorted(enumerate(exps), key=lambda p: -p[1])
        acc = 0.0
        keep = set()
        for i, e in pairs:
            keep.add(i)
            acc += e / total
            if acc >= top_p:
                break
        exps = [e if i in keep else 0.0 for i, e in enumerate(exps)]
        total = sum(exps)
    r = rng.random() * total
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if r < acc:
            return i
    return max(range(len(exps)), key=lambda i: exps[i])


class _Handler(BaseHTTPRequestHandler):
    server_version = "infer_train/0.1.0-M7"

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message, status=400):
        self._send_json({"error": {"message": message, "type": "invalid_request_error"}}, status)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def _model(self):
        return self.server.model

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # -- endpoints -----------------------------------------------------------

    def do_GET(self):
        if not _api_key_ok(self.headers):
            return self._send_error("invalid API key", 401)
        if self.path == "/health":
            return self._send_json({"status": "ok", "model": self._model()._path})
        if self.path == "/v1/models":
            return self._send_json({
                "object": "list",
                "data": [{
                    "id": os.path.basename(self._model()._path),
                    "object": "model",
                    "created": 0,
                    "owned_by": "infer_train",
                }],
            })
        if self.path == "/v1/finetune/status":
            with _JOBS_LOCK:
                jobs = [dict(j) for j in _FINETUNE_JOBS.values()]
            return self._send_json({"object": "list", "data": jobs})
        return self._send_error("not found", 404)

    def do_POST(self):
        if not _api_key_ok(self.headers):
            return self._send_error("invalid API key", 401)
        try:
            body = self._read_body()
        except Exception:
            return self._send_error("invalid JSON body", 400)
        if self.path == "/v1/completions":
            return self._completions(body)
        if self.path == "/v1/chat/completions":
            return self._chat_completions(body)
        if self.path == "/v1/finetune":
            return self._finetune(body)
        return self._send_error("not found", 404)

    # -- completions ----------------------------------------------------------

    def _prompt_of(self, body):
        if "prompt" in body:
            return body["prompt"]
        messages = body.get("messages", [])
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(content + "\n")
            else:
                parts.append(content + "\n")
        return "\n".join(parts).strip()

    def _completions(self, body):
        prompt = self._prompt_of(body)
        if not prompt:
            return self._send_error("prompt is required", 400)
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens", 32))
        temperature = float(body.get("temperature", 0.6))
        top_p = float(body.get("top_p", 0.95))
        top_k = int(body.get("top_k", 40))
        seed = body.get("seed")
        if stream:
            return self._completions_sse(prompt, max_tokens, temperature, top_p, top_k, seed)
        model = self._model()
        model.reset_cache()
        out = model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=None if seed is None else int(seed),
        )
        return self._send_json({
            "id": "cmpl-%d" % int(time.time() * 1000),
            "object": "text_completion",
            "created": int(time.time()),
            "model": os.path.basename(model._path),
            "choices": [{"text": out, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(out), "total_tokens": len(out)},
        })

    def _completions_sse(self, prompt, max_tokens, temperature, top_p, top_k, seed):
        model = self._model()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        rng = random.Random(seed)
        try:
            model.reset_cache()
            tokens = _model_mod._encode(model, prompt)
            pos = 0
            logits = None
            for tok in tokens:
                ptr = _lib.infer_train_forward_logits(model._ptr, tok, pos)
                pos += 1
                logits = self._read_logits(ptr)
            eos = model.info("eos") or 2
            cmpl_id = "cmpl-%d" % int(time.time() * 1000)
            finish = "length"
            for _ in range(max_tokens):
                next_token = _sample_token(logits, temperature, top_k, top_p, rng)
                if next_token == eos:
                    finish = "stop"
                    break
                ptr = _lib.infer_train_forward_logits(model._ptr, next_token, pos)
                pos += 1
                logits = self._read_logits(ptr)
                piece = self._decode_token(model, next_token)
                data = {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": os.path.basename(model._path),
                    "choices": [{"text": piece, "index": 0, "finish_reason": None}],
                }
                self.wfile.write(("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
            done = {
                "id": cmpl_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": os.path.basename(model._path),
                "choices": [{"text": "", "index": 0, "finish_reason": finish}],
            }
            self.wfile.write(("data: " + json.dumps(done) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as exc:
            try:
                self.wfile.write(("data: " + json.dumps({"error": str(exc)}) + "\n\n").encode())
                self.wfile.flush()
            except Exception:
                pass

    def _read_logits(self, ptr):
        if not ptr:
            raise RuntimeError("engine returned NULL logits")
        try:
            vocab = self._model().info("vocab") or 0
            import ctypes

            return list(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))[:vocab])
        finally:
            _lib.infer_train_free_buffer(ptr)

    def _decode_token(self, model, token):
        import ctypes

        n = ctypes.c_int64(1)
        tok_arr = (ctypes.c_int64 * 1)(token)
        ptr = _lib.infer_train_decode(model._ptr, tok_arr, 1)
        if not ptr:
            return ""
        try:
            return ctypes.string_at(ptr).decode("utf-8", "replace")
        finally:
            _lib.infer_train_free_string(ptr)

    def _chat_completions(self, body):
        prompt = self._prompt_of(body)
        if not prompt:
            return self._send_error("messages are required", 400)
        model = self._model()
        model.reset_cache()
        out = model.generate(
            prompt,
            max_tokens=int(body.get("max_tokens", 64)),
            temperature=float(body.get("temperature", 0.6)),
            top_p=float(body.get("top_p", 0.95)),
            top_k=int(body.get("top_k", 40)),
            seed=None if body.get("seed") is None else int(body["seed"]),
        )
        return self._send_json({
            "id": "chatcmpl-%d" % int(time.time() * 1000),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(model._path),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    # -- fine-tuning -----------------------------------------------------------

    def _finetune(self, body):
        input_text = body.get("input") or body.get("input_text") or body.get("prompt")
        target_text = body.get("target") or body.get("target_text")
        if not input_text or not target_text:
            return self._send_error("finetune needs 'input' and 'target' fields", 400)
        lr = float(body.get("lr", 1e-5))
        job_id = "ft-%d" % int(time.time() * 1000)
        with _JOBS_LOCK:
            _FINETUNE_JOBS[job_id] = {
                "id": job_id,
                "status": "queued",
                "steps": 0,
                "loss": None,
                "losses": [],
            }

        def run():
            model = self._model()
            try:
                with _JOBS_LOCK:
                    _FINETUNE_JOBS[job_id]["status"] = "running"
                losses = model.finetune(input_text, target_text, lr=lr)
                with _JOBS_LOCK:
                    _FINETUNE_JOBS[job_id]["status"] = "done"
                    _FINETUNE_JOBS[job_id]["losses"] = [float(l) for l in losses]
                    _FINETUNE_JOBS[job_id]["loss"] = float(losses[-1]) if losses else None
                    _FINETUNE_JOBS[job_id]["steps"] = len(losses)
            except Exception as exc:
                with _JOBS_LOCK:
                    _FINETUNE_JOBS[job_id]["status"] = "failed"
                    _FINETUNE_JOBS[job_id]["error"] = str(exc)

        threading.Thread(target=run, daemon=True).start()
        return self._send_json({"id": job_id, "status": "queued"}, 202)


def make_server(model_path, host="127.0.0.1", port=8080, ctx_size=512):
    model = _model_mod.load_model(model_path)
    # NOTE: ctx_size is engine-side (KV cache); accepted for API parity.
    _ = ctx_size
    server = ThreadingHTTPServer((host, port), _Handler)
    server.model = model  # type: ignore[attr-defined]
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="infer_train HTTP server (M7)")
    parser.add_argument("-m", "--model", default=os.environ.get("INFERTRAIN_MODEL", ""))
    parser.add_argument("--host", default=os.environ.get("INFERTRAIN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("INFERTRAIN_PORT", "8080")))
    parser.add_argument("-c", "--ctx-size", type=int, default=512)
    args = parser.parse_args(argv)
    if not args.model:
        raise SystemExit("error: set INFERTRAIN_MODEL or pass -m MODEL")
    server = make_server(args.model, args.host, args.port, args.ctx_size)
    print(f"infer_train server on http://{args.host}:{args.port} (model: {args.model})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
