"""infer_train: a high-performance LLM inference engine for Apple Silicon.

M4 (PyTorch ecosystem): this package exposes

* :func:`load_model` / :class:`~infer_train.model.Model` - load a GGUF
  (Q5_K_M/Q6_K/fp16) model and generate text through the Mojo engine's C
  API, and
* :func:`optimize` - compile a ``torch.nn.Module`` with
  ``torch.compile(model, backend="infer_train")``, translating the captured
  FX graph onto the engine's operators (``matmul`` / ``rms_norm`` /
  ``softmax`` / ``add`` / ``swiglu`` / ``embedding`` / ``lm_head`` and the
  M5 fused kernels), including ``torch.cond`` / ``torch.while_loop``
  control flow (static conditions resolve at compile time).

Usage::

    import torch
    from infer_train import optimize

    model = MyTransformer(...).eval()
    compiled = optimize(model)
    out = compiled(inputs)          # same call signature as model

    from infer_train import load_model
    m = load_model("DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf")
    print(m.generate("<|User|>What is 1+1?<|Assistant|><think>\\n",
                     max_tokens=120, seed=7))
"""

from __future__ import annotations

__version__ = "0.1.0-M5"

from .binding import EngineError, load_library  # noqa: F401  (library checks)
from .model import Model, load_model  # noqa: F401

_BACKEND_NAME = "infer_train"


def _backend_compiler(gm, example_inputs):
    from .backend import infer_train_backend

    return infer_train_backend(gm, example_inputs)


def optimize(model, *, strict: bool = False, verbose: bool = True):
    """Compile ``model`` with ``torch.compile(backend="infer_train")``.

    Equivalent to ``torch.compile(model, backend="infer_train")`` but also
    prints the translation summary by default.  ``strict=True`` raises on
    any FX op without a native engine implementation instead of falling
    back to torch.
    """
    import torch

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "torch.compile requires PyTorch 2.0+; the installed version "
            f"({torch.__version__}) does not provide it."
        )
    _register()

    if strict:
        # strict mode: wrap the backend with the flag
        def _strict_backend(gm, example_inputs):
            from .backend import infer_train_backend

            return infer_train_backend(
                gm, example_inputs, strict=True, verbose=verbose
            )

        return torch.compile(model, backend=_strict_backend)
    return torch.compile(model, backend=_BACKEND_NAME)


def _register():
    """Register the "infer_train" backend with torch._dynamo (idempotent)."""
    try:
        import torch._dynamo as dynamo
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for the torch.compile backend; install it "
            "first (`pip install torch`)."
        ) from e
    if _BACKEND_NAME not in dynamo.list_backends(exclude_tags=("debug",)):
        dynamo.register_backend(
            _backend_compiler, name=_BACKEND_NAME, tags=["infer_train", "m4"]
        )


__all__ = [
    "__version__",
    "EngineError",
    "Model",
    "load_model",
    "load_library",
    "optimize",
]

# Auto-register the "infer_train" torch.compile backend when torch is
# importable (best-effort: the model API works without torch).
try:
    _register()
except Exception:  # pragma: no cover - torch missing or unusable
    pass
