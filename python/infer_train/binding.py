"""ctypes bindings for the infer_train engine shared library.

This module is the only place that talks to ``libinfer_train`` directly.  It
is deliberately torch-free: it speaks in raw bytes/pointers, and
``backend.py`` adapts torch tensors onto it.

Library discovery order:
    1. ``INFER_TRAIN_LIB`` environment variable (explicit path);
    2. ``infer_train/_lib/libinfer_train.{dylib,so,dll}`` (built by setup.py);
    3. ``<repo>/python/infer_train/_lib/...`` when running from a checkout.

Dtype codes (C ABI convention, see infer_train_bindings.mojo):
    0 = f32, 1 = f16, 2 = i32.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

__all__ = [
    "EngineError",
    "DTYPE_F32",
    "DTYPE_F16",
    "DTYPE_I32",
    "ELEM_SIZE",
    "load_library",
    "Tensor",
    "Snapshot",
    "run_op",
    "run_backward",
    "adamw_step",
    "ABI_VERSION",
]


class EngineError(RuntimeError):
    """Raised when the engine returns NULL / -1 on a call."""


# -- dtype codes -------------------------------------------------------------

DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_I32 = 2

ELEM_SIZE = {DTYPE_F32: 4, DTYPE_F16: 2, DTYPE_I32: 4}

ABI_VERSION = 1

_LIB_NAME = {
    "darwin": "libinfer_train.dylib",
    "linux": "libinfer_train.so",
    "win32": "libinfer_train.dll",
}.get(sys.platform, "libinfer_train.so")


def _candidate_paths():
    env = os.environ.get("INFER_TRAIN_LIB")
    if env:
        yield Path(env)
    here = Path(__file__).resolve().parent
    yield here / "_lib" / _LIB_NAME
    # running from a source checkout: <repo>/python/infer_train/_lib
    repo = here.parent.parent
    yield repo / "python" / "infer_train" / "_lib" / _LIB_NAME
    yield repo / "build" / _LIB_NAME


def load_library():
    """Load libinfer_train and raise a helpful EngineError when not found."""
    for path in _candidate_paths():
        if path.is_file():
            # RTLD_GLOBAL: the engine's @export'd thread-pool workers must
            # be visible to libinfer_train_tp.dylib's dlsym(RTLD_DEFAULT).
            return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    tried = "\n  ".join(str(p) for p in _candidate_paths())
    raise EngineError(
        "libinfer_train shared library not found. Build it with\n"
        "  python setup.py build_mojo\n"
        "or set INFER_TRAIN_LIB to its full path. Tried:\n  " + tried
    )


_lib = load_library()


def _setup_prototypes():
    _lib.infer_train_version.restype = ctypes.c_int64

    # model-level
    _lib.infer_train_load_model.argtypes = [ctypes.c_char_p]
    _lib.infer_train_load_model.restype = ctypes.c_void_p
    _lib.infer_train_generate.argtypes = [
        ctypes.c_void_p,  # model
        ctypes.c_char_p,  # prompt (UTF-8, NUL-terminated)
        ctypes.c_int64,  # max_tokens
        ctypes.c_float,  # temperature
        ctypes.c_float,  # top_p
        ctypes.c_int64,  # top_k
        ctypes.c_int64,  # seed (-1 = None)
        ctypes.c_int32,  # verbose
    ]
    _lib.infer_train_generate.restype = ctypes.c_void_p
    _lib.infer_train_model_info.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    _lib.infer_train_model_info.restype = ctypes.c_int64
    _lib.infer_train_reset_cache.argtypes = [ctypes.c_void_p]
    _lib.infer_train_reset_cache.restype = None
    _lib.infer_train_free_model.argtypes = [ctypes.c_void_p]
    _lib.infer_train_free_model.restype = None
    _lib.infer_train_free_string.argtypes = [ctypes.c_void_p]
    _lib.infer_train_free_string.restype = None

    # tensor-level
    _lib.infer_train_tensor_create.argtypes = [
        ctypes.c_int32,  # dtype code
        ctypes.c_int64,  # rank
        ctypes.POINTER(ctypes.c_int64),  # shape
        ctypes.c_void_p,  # data (copied in)
    ]
    _lib.infer_train_tensor_create.restype = ctypes.c_void_p
    _lib.infer_train_tensor_dtype.argtypes = [ctypes.c_void_p]
    _lib.infer_train_tensor_dtype.restype = ctypes.c_int32
    _lib.infer_train_tensor_rank.argtypes = [ctypes.c_void_p]
    _lib.infer_train_tensor_rank.restype = ctypes.c_int64
    _lib.infer_train_tensor_numel.argtypes = [ctypes.c_void_p]
    _lib.infer_train_tensor_numel.restype = ctypes.c_int64
    _lib.infer_train_tensor_shape.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
    ]
    _lib.infer_train_tensor_shape.restype = ctypes.c_int64
    _lib.infer_train_tensor_copy_out.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _lib.infer_train_tensor_copy_out.restype = ctypes.c_int64
    _lib.infer_train_tensor_free.argtypes = [ctypes.c_void_p]
    _lib.infer_train_tensor_free.restype = None

    # op-level (single-shot execution; see infer_train_bindings.mojo)
    _lib.infer_train_run_op.argtypes = [
        ctypes.c_char_p,  # op name
        ctypes.POINTER(ctypes.c_int64),  # input handle addresses
        ctypes.c_int64,  # n inputs
        ctypes.POINTER(ctypes.c_int64),  # out: dtype code
        ctypes.POINTER(ctypes.c_int64),  # out: rank
        ctypes.POINTER(ctypes.c_int64),  # out: shape (8 slots)
        ctypes.POINTER(ctypes.c_int64),  # out: numel
        ctypes.POINTER(ctypes.c_int64),  # out: nbytes
    ]
    _lib.infer_train_run_op.restype = ctypes.c_void_p
    _lib.infer_train_free_buffer.argtypes = [ctypes.c_void_p]
    _lib.infer_train_free_buffer.restype = None

    # M7: inference-time fine-tuning + tokenizer access
    _lib.infer_train_finetune_create.argtypes = [
        ctypes.c_void_p, ctypes.c_float,
    ]
    _lib.infer_train_finetune_create.restype = ctypes.c_void_p
    _lib.infer_train_finetune_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_float,
    ]
    _lib.infer_train_finetune_step.restype = ctypes.c_float
    _lib.infer_train_finetune_free.argtypes = [ctypes.c_void_p]
    _lib.infer_train_finetune_free.restype = None
    _lib.infer_train_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64),
    ]
    _lib.infer_train_encode.restype = ctypes.c_void_p
    _lib.infer_train_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    ]
    _lib.infer_train_decode.restype = ctypes.c_void_p
    _lib.infer_train_forward_logits.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
    ]
    _lib.infer_train_forward_logits.restype = ctypes.c_void_p

    # op-level backward (M6)
    _lib.infer_train_run_backward.argtypes = [
        ctypes.c_char_p,  # op name
        ctypes.POINTER(ctypes.c_int64),  # input handle addresses
        ctypes.c_int64,  # n inputs
        ctypes.POINTER(ctypes.c_int64),  # grad-output handle addresses
        ctypes.c_int64,  # n grad outputs
        ctypes.POINTER(ctypes.c_int64),  # out: n grads
        ctypes.POINTER(ctypes.c_int64),  # out: buffer addresses (8)
        ctypes.POINTER(ctypes.c_int64),  # out: dtypes (8)
        ctypes.POINTER(ctypes.c_int64),  # out: ranks (8)
        ctypes.POINTER(ctypes.c_int64),  # out: shapes (8*8)
        ctypes.POINTER(ctypes.c_int64),  # out: numels (8)
        ctypes.POINTER(ctypes.c_int64),  # out: nbytes (8)
    ]
    _lib.infer_train_run_backward.restype = ctypes.c_int32

    # stateless AdamW step (M6)
    _lib.infer_train_adamw_step.argtypes = [
        ctypes.c_void_p,  # param handle
        ctypes.c_void_p,  # grad handle
        ctypes.c_void_p,  # m handle (fp32)
        ctypes.c_void_p,  # v handle (fp32)
        ctypes.POINTER(ctypes.c_int64),  # step counter (in/out)
        ctypes.c_float,  # lr
        ctypes.c_float,  # beta1
        ctypes.c_float,  # beta2
        ctypes.c_float,  # eps
        ctypes.c_float,  # weight decay
    ]
    _lib.infer_train_adamw_step.restype = ctypes.c_int32


_setup_prototypes()

_version_ok = _lib.infer_train_version()
if _version_ok != ABI_VERSION:
    raise EngineError(
        f"libinfer_train ABI version mismatch: expected {ABI_VERSION}, "
        f"got {_version_ok}. Rebuild with `python setup.py build_mojo`."
    )


# -- tensor handle -----------------------------------------------------------


class Tensor:
    """Handle to an engine-side tensor; owns its buffer until ``free()``.

    Create with :meth:`from_bytes` / :meth:`from_buffer` (data is copied
    into engine memory), or obtain one from :meth:`Graph.run`.
    """

    __slots__ = ("_ptr", "_dtype", "_rank", "_numel", "_shape", "_freed")

    def __init__(self, ptr: int):
        if not ptr:
            raise EngineError("infer_train_tensor_create returned NULL")
        self._ptr = ptr
        self._dtype = int(_lib.infer_train_tensor_dtype(ptr))
        self._rank = int(_lib.infer_train_tensor_rank(ptr))
        self._numel = int(_lib.infer_train_tensor_numel(ptr))
        shape_buf = (ctypes.c_int64 * max(self._rank, 1))()
        _lib.infer_train_tensor_shape(ptr, shape_buf)
        self._shape = tuple(int(shape_buf[i]) for i in range(self._rank))
        self._freed = False

    @classmethod
    def from_bytes(cls, dtype_code: int, shape, data: bytes) -> "Tensor":
        shape = tuple(int(d) for d in shape)
        rank = len(shape)
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= ELEM_SIZE[dtype_code]
        if len(data) != nbytes:
            raise EngineError(
                f"data length {len(data)} != numel*elem_size {nbytes}"
            )
        shape_arr = (ctypes.c_int64 * rank)(*shape)
        ptr = _lib.infer_train_tensor_create(
            dtype_code, rank, shape_arr, data
        )
        return cls(ptr)

    @classmethod
    def from_buffer(cls, dtype_code: int, shape, data_ptr: int) -> "Tensor":
        """Copy ``numel(shape) * elem_size`` bytes from ``data_ptr`` in."""
        shape = tuple(int(d) for d in shape)
        rank = len(shape)
        shape_arr = (ctypes.c_int64 * rank)(*shape)
        ptr = _lib.infer_train_tensor_create(
            dtype_code, rank, shape_arr, data_ptr
        )
        return cls(ptr)

    @property
    def dtype(self) -> int:
        return self._dtype

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def numel(self) -> int:
        return self._numel

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def nbytes(self) -> int:
        return self._numel * ELEM_SIZE[self._dtype]

    @property
    def ptr(self) -> int:
        return self._ptr

    def copy_out_into(self, dst_ptr: int) -> int:
        """Copy the payload into caller memory at ``dst_ptr``."""
        n = _lib.infer_train_tensor_copy_out(self._ptr, dst_ptr)
        if n < 0:
            raise EngineError("infer_train_tensor_copy_out failed")
        return int(n)

    def to_bytes(self) -> bytes:
        buf = ctypes.create_string_buffer(self.nbytes)
        self.copy_out_into(ctypes.addressof(buf))
        return buf.raw

    def free(self):
        if not self._freed:
            _lib.infer_train_tensor_free(self._ptr)
            self._freed = True

    def __del__(self):  # pragma: no cover - GC safety net
        try:
            if getattr(self, "_freed", True) is False:
                self.free()
        except Exception:
            pass


# -- convenience -------------------------------------------------------------


class Snapshot:
    """A tensor payload copied out of the engine (graph-independent)."""

    __slots__ = ("dtype", "shape", "data")

    def __init__(self, dtype: int, shape: tuple, data: bytes):
        self.dtype = dtype
        self.shape = shape
        self.data = data


def run_backward(op_name: str, tensors, grads) -> list[Snapshot | None]:
    """Run one engine op's backward; returns one gradient per forward
    input (``None`` marks the no-gradient sentinel, e.g. integer ids)."""
    n = len(tensors)
    m = len(grads)
    handle_arr = (ctypes.c_int64 * n)(*(t.ptr for t in tensors))
    grad_arr = (ctypes.c_int64 * m)(*(g.ptr for g in grads))
    out_n = ctypes.c_int64()
    addrs = (ctypes.c_int64 * 8)()
    dtype = (ctypes.c_int64 * 8)()
    rank = (ctypes.c_int64 * 8)()
    shape = (ctypes.c_int64 * 64)()
    numel = (ctypes.c_int64 * 8)()
    nbytes = (ctypes.c_int64 * 8)()
    rc = _lib.infer_train_run_backward(
        op_name.encode("utf-8"),
        handle_arr,
        n,
        grad_arr,
        m,
        ctypes.byref(out_n),
        addrs,
        dtype,
        rank,
        shape,
        numel,
        nbytes,
    )
    if rc != 0:
        raise EngineError(
            f"infer_train_run_backward('{op_name}') failed (invalid op, "
            f"inputs or grad-output shapes)"
        )
    snaps = []
    try:
        for i in range(int(out_n.value)):
            if numel[i] == 0:
                snaps.append(None)
                continue
            data = ctypes.string_at(addrs[i], int(nbytes[i]))
            shp = tuple(
                int(shape[i * 8 + j]) for j in range(int(rank[i]))
            )
            snaps.append(Snapshot(int(dtype[i]), shp, data))
    finally:
        for i in range(int(out_n.value)):
            if addrs[i]:
                _lib.infer_train_free_buffer(addrs[i])
    return snaps


def adamw_step(
    param: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    step: int,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    wd: float,
) -> int:
    """One engine-side AdamW update; mutates param/m/v, returns the new
    step count."""
    sp = ctypes.c_int64(step)
    rc = _lib.infer_train_adamw_step(
        param.ptr,
        grad.ptr,
        m.ptr,
        v.ptr,
        ctypes.byref(sp),
        ctypes.c_float(lr),
        ctypes.c_float(b1),
        ctypes.c_float(b2),
        ctypes.c_float(eps),
        ctypes.c_float(wd),
    )
    if rc != 0:
        raise EngineError("infer_train_adamw_step failed")
    return int(sp.value)


def run_op(op_name: str, tensors) -> list[Snapshot]:
    """Run one engine op on handles (single-shot C call).

    The engine returns the kernel's raw output buffer plus its metadata;
    the payload is copied out into :class:`Snapshot` objects and the buffer
    is released before returning.
    """
    n = len(tensors)
    handle_arr = (ctypes.c_int64 * n)(*(t.ptr for t in tensors))
    d = ctypes.c_int64()
    r = ctypes.c_int64()
    shape = (ctypes.c_int64 * 8)()
    numel = ctypes.c_int64()
    nbytes = ctypes.c_int64()
    ptr = _lib.infer_train_run_op(
        op_name.encode("utf-8"),
        handle_arr,
        n,
        ctypes.byref(d),
        ctypes.byref(r),
        shape,
        ctypes.byref(numel),
        ctypes.byref(nbytes),
    )
    if not ptr:
        raise EngineError(
            f"infer_train_run_op('{op_name}') failed with "
            f"{n} input(s) (NULL). The op is either not on the M4 "
            f"whitelist or the input shapes/dtypes violate its contract; "
            f"see the UnsupportedOpError raised by the backend for the "
            f"FX-node context."
        )
    try:
        data = ctypes.string_at(ptr, int(nbytes.value))
    finally:
        _lib.infer_train_free_buffer(ptr)
    out_shape = tuple(int(shape[i]) for i in range(int(r.value)))
    return [Snapshot(int(d.value), out_shape, data)]
