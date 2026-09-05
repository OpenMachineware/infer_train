"""High-level model wrapper over the engine's C API (M4 task 1-2).

Usage::

    from infer_train.model import load_model

    model = load_model("DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf")
    print(model.info("hidden"))
    out = model.generate("<|User|>What is 1+1?<|Assistant|><think>\\n",
                         max_tokens=120, seed=7)
    print(out)

The wrapper is a thin, ergonomic layer over ``binding``; all engine calls
are UTF-8 byte strings, and the generated text comes back as ``str``.
"""

from __future__ import annotations

import ctypes
from typing import Optional

from .binding import EngineError, _lib

__all__ = ["Model", "load_model"]


class Model:
    """A loaded GGUF model + tokenizer held by the engine (opaque handle)."""

    __slots__ = ("_ptr", "_path", "_freed")

    def __init__(self, ptr: int, path: str):
        if not ptr:
            raise EngineError(
                f"infer_train_load_model('{path}') failed (NULL). "
                "Check that the .gguf exists (architecture and tokenizer "
                "metadata are read from the GGUF itself); generation also "
                "needs the KV cache to fit prompt + max_tokens."
            )
        self._ptr = ptr
        self._path = path
        self._freed = False

    # -- queries -------------------------------------------------------------

    def info(self, key: str) -> Optional[int]:
        """Query a config value; returns None for unknown keys.

        Keys: n_layers, hidden, ffn, n_heads, n_kv_heads, head_dim, vocab,
        bos, eos, ctx_len.
        """
        value = _lib.infer_train_model_info(
            self._ptr, key.encode("utf-8")
        )
        return None if value < 0 else int(value)

    @property
    def config(self) -> dict:
        keys = (
            "n_layers",
            "hidden",
            "ffn",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "vocab",
            "bos",
            "eos",
            "ctx_len",
        )
        return {k: self.info(k) for k in keys if self.info(k) is not None}

    def reset_cache(self):
        """Clear the KV cache for a fresh conversation."""
        _lib.infer_train_reset_cache(self._ptr)

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """Run the autoregressive loop and return the completion string.

        The prompt is encoded with the Qwen2 BOS token (same as the
        ``infer_train`` CLI).  Pass ``seed`` for reproducible sampling.
        """
        if not prompt:
            raise ValueError("prompt must be a non-empty string")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        out_ptr = _lib.infer_train_generate(
            self._ptr,
            prompt.encode("utf-8"),
            max_tokens,
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
            top_k,
            -1 if seed is None else int(seed),
            1 if verbose else 0,
        )
        if not out_ptr:
            raise EngineError(
                "infer_train_generate failed (NULL). The usual causes: "
                "prompt + max_tokens exceeds the KV cache capacity "
                "(see model.info('ctx_len')), or a sampling parameter "
                "is invalid."
            )
        try:
            return ctypes.string_at(out_ptr).decode("utf-8")
        finally:
            _lib.infer_train_free_string(out_ptr)

    # -- M7: inference-time fine-tuning ---------------------------------------

    def finetune(
        self,
        input_text: str,
        target_text: str,
        lr: float = 1e-5,
    ) -> list:
        """Adapt the output head toward ``target_text`` given ``input_text``.

        Teacher-forced LoRA-style adaptation (only the LM head updates);
        returns the per-step losses.  The model keeps serving while the
        adapter trains - every update is re-synced into the live weights.
        """
        if not input_text or not target_text:
            raise ValueError("input_text and target_text must be non-empty")
        prompt = _encode(self, input_text)
        targets = _encode(self, target_text)
        eos = self.info("eos") or 2
        targets.append(eos)
        session = FinetuneSession(self, lr)
        losses = []
        try:
            # advance the model state over the prompt (forward-only)
            for i, token in enumerate(prompt):
                session.step(token, i, None)
            # teacher forcing over the targets
            pos = len(prompt)
            for i, token in enumerate(targets):
                target = targets[i + 1] if i + 1 < len(targets) else eos
                loss = session.step(token, pos, target)
                pos += 1
                losses.append(loss)
        finally:
            session.free()
        return losses

    # -- lifecycle -----------------------------------------------------------

    def free(self):
        """Release the engine-side model (best-effort; frees buffers)."""
        if not self._freed:
            _lib.infer_train_free_model(self._ptr)
            self._freed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()
        return False

    def __del__(self):  # pragma: no cover - GC safety net
        try:
            if getattr(self, "_freed", True) is False:
                self.free()
        except Exception:
            pass


def load_model(path: str) -> Model:
    """Load a GGUF model + tokenizer into the engine and wrap it."""
    ptr = _lib.infer_train_load_model(path.encode("utf-8"))
    return Model(int(ptr or 0), path)


class FinetuneSession:
    """M7: inference-time fine-tuning on a live model.

    The engine keeps an fp32 adapter copy of the *output head* (LoRA-style:
    every other parameter stays frozen).  Each ``step`` computes one
    forward through the transformer stack and one AdamW update toward the
    target token; the fp16 head inside the model is re-synced after every
    update, so concurrent inference requests immediately see the adapted
    weights.
    """

    __slots__ = ("_model", "_ft", "_freed")

    def __init__(self, model: "Model", lr: float = 1e-5):
        if model._freed:
            raise EngineError("model already freed")
        self._model = model
        self._freed = False
        ptr = _lib.infer_train_finetune_create(model._ptr, ctypes.c_float(lr))
        if not ptr:
            raise EngineError("infer_train_finetune_create failed (NULL)")
        self._ft = ptr

    def step(self, token: int, position: int, target: Optional[int]) -> float:
        """One forward step (and one update when ``target`` is given).

        ``target=None`` only advances the model state (prompt processing).
        """
        return float(
            _lib.infer_train_finetune_step(
                self._ft,
                self._model._ptr,
                int(token),
                int(position),
                -1 if target is None else int(target),
                ctypes.c_float(1e-5),
            )
        )

    def free(self):
        if not self._freed:
            _lib.infer_train_finetune_free(self._ft)
            self._freed = True

    def __del__(self):
        try:
            if getattr(self, "_freed", True) is False:
                self.free()
        except Exception:
            pass


def _encode(model: "Model", text: str) -> list:
    n = ctypes.c_int64(0)
    ptr = _lib.infer_train_encode(
        model._ptr, text.encode("utf-8"), ctypes.byref(n)
    )
    if not ptr:
        raise EngineError("infer_train_encode failed")
    try:
        return list(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))[: n.value])
    finally:
        _lib.infer_train_free_buffer(ptr)
