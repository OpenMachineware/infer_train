# Adding a New Operator to InferTrain

This guide walks through discovering, implementing, registering, and testing a new
operator in the InferTrain engine. It assumes the project conventions:

- Mojo **1.0.0** — only `def` (no `fn`), `mut` in place of `inout`, `^` for moves,
  `Optional[...]` in place of `Option`.
- No runtime mutable globals; no trait-existential dynamic dispatch (tagged structs instead).
- Every operator is registered through `OpRegistry.register(...)` / `register_default_ops()`
  with an `OpInfo` entry carrying `forward`, `forward_with_saved`, and `backward` kernels.

Read these first, they define the contract:

- `src/core/ops/base/op_interface.mojo` — `AnyTensor`, `OpInfo`, `to_any` / `from_any`.
- `src/core/ops/base/op_registry.mojo` — per-dtype dispatch + `register_default_ops()`.
- `src/core/ops/base/op_autograd.mojo` — the type-erased `forward_with_saved` / `backward`
  functions that `OpInfo` stores.

---

## 1. Diagnosing an unsupported operator

There are **two** independent layers that can reject an op, and they fail very differently.

### 1.1 The `unimplemented()` abort path (kernel layer)

`unimplemented()` lives in `src/core/utils.mojo` and terminates the process via
`abort()` — Mojo 1.0 has no built-in `unimplemented`, so this is modeled with a trap
instruction:

```mojo
def unimplemented(message: String = "not implemented") -> None:
    _ = message
    abort()
```

Every CPU kernel guards its contract with `unimplemented(...)` before touching memory,
e.g. `src/core/ops/cpu/add_cpu.mojo`:

```mojo
def _add_cpu_kernel[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    if a.shape() != b.shape():
        unimplemented("add_cpu: shape mismatch")
    ...
```

So if a **registered** op is called with a bad shape/dtype, you get a hard abort, not a
graceful error. That is exactly why the C-API adds a whitelist (next section).

### 1.2 The binding whitelist (`_check_op_inputs` / `_run_op_native`)

`src/bindings/infer_train_bindings.mojo` exposes `infer_train_run_op`, which validates
**before** dispatching so a bad Python call never reaches an aborting kernel:

```mojo
@export
def infer_train_run_op(op_ptr, inputs, n_inputs, ...) abi("C") -> Optional[Pointer[UInt8, ...]]:
    var op = _cstr_to_string(op_ptr)
    var input_list = _collect_inputs(inputs, Int(n_inputs))
    if not _check_op_inputs(op, input_list):
        return None                       # NULL -> the Python layer raises
    var result = _run_op_native(op, input_list)
    if not result:
        return None
    ...
```

- `_check_op_inputs(op, inputs) -> Bool` (line 698) is a hand-written if/else over every
  whitelisted op name, checking arity, rank, dtype, and shape compatibility. Any op name
  that has no branch falls through to `return False`.
- `_run_op_native(op, inputs) -> Optional[AnyTensor]` (line 919) then calls the concrete
  kernel directly (it deliberately avoids the registry's `List.append` result-building,
  which is miscompiled in Mojo 1.0 shared libraries).

**Diagnosis flow for "op not supported":**

1. An unknown op name → `_check_op_inputs` returns `False` → `infer_train_run_op` returns
   `NULL` → Python raises with the full context (op name, shapes, node).
2. A known op with a bad shape → same NULL path (no abort).
3. A known op, valid shapes, but an unsupported dtype → the CPU dispatch function hits its
   `unimplemented("...: unsupported dtype")` fallthrough (e.g. `matmul_dispatch_cpu` only
   accepts `float32` / `float16`).

When adding an op to the C API surface you must extend **both** `_check_op_inputs` and
`_run_op_native` (and, if it is trainable, `_check_grad_shapes` + `_run_backward_native`).

---

## 2. Reference implementations and validation

InferTrain's numeric behavior was validated against two independent references:

- **llama.cpp** — the quantized formats (`ggml-quants.c`) and the model forward; ground-truth
  logits were dumped to `reference_logits_5.npy` from `llama-cpp-python`.
- **PyTorch** (via `gguf-py` / HF transformers) — the dequantized f32 reference semantics.

`tools/ref_forward.py` is the cross-check harness: it dequantizes every weight from the GGUF
with vectorized NumPy and re-implements the Qwen2 forward (rmsnorm, RoPE, causal attention,
SwiGLU FFN) in pure NumPy, then compares per-step top-5 logits against the llama.cpp dump:

```python
ref = _np.load("reference_logits_5.npy")
for i, tok in enumerate(toks):
    logits = forward(tok, i, K, V)[0]
    rrow = ref[i+1]
    print(f"step {i}: top5 {_np.argsort(logits)[::-1][:5].tolist()} "
          f"ref {_np.argsort(rrow)[::-1][:5].tolist()} "
          f"corr {_np.corrcoef(logits, rrow)[0,1]:.5f}")
```

For a **new** operator, follow the same recipe: write a small NumPy reference in
`tools/` (or inline in a test), run both on the same random inputs, and compare element-wise.
Two numeric rules from the codebase carry over to any new kernel:

- fp16 inputs are **widened per element and accumulated in f32**, then cast back on store
  (see the M3 note in `src/core/ops/cpu/matmul_cpu.mojo`).
- `eps` for RMSNorm defaults to `1e-5` in the kernels; the NumPy reference in
  `ref_forward.py` uses `1e-6` — match the reference you are validating against.

---

## 3. Registration steps (the checklist)

Adding a trainable elementwise/rank-2 operator means touching **five** places. Order matters
because later steps import earlier ones.

### 3.1 Write the CPU kernel — `src/core/ops/cpu/<name>_cpu.mojo`

Follow the existing shape: a private `_<name>_cpu_kernel[dtype]` (runtime shapes), a
`<name>_cpu[dtype, ...]` (comptime shapes) and a `<name>_cpu_dynamic[dtype]` entry point the
registry calls. Use the `simd_utils` widths `W_F16 = 8` / `W_F32 = 4`.

### 3.2 Add the dispatch function — `src/core/ops/base/op_registry.mojo`

Write `_<name>_typed_cpu[dtype]` (an `if dtype == ...` ladder) plus
`<name>_dispatch_cpu` / `<name>_dispatch_gpu`. The GPU dispatch usually delegates to CPU
until a backend lands.

### 3.3 Add the type-erased autograd — `src/core/ops/base/op_autograd.mojo`

Write `<name>_fws_cpu` / `<name>_bwd_cpu` (and `<name>_fws_gpu` / `<name>_bwd_gpu` that
delegate to CPU). These are what `OpInfo.forward_with_saved` / `OpInfo.backward` point to.

### 3.4 Register the `OpInfo` entries — `register_default_ops()`

One `self.register(...)` per device (CPU priority `0`, `Device.MetalGPU` priority `10`).

### 3.5 (Optional) Whitelist for the C API — `src/bindings/infer_train_bindings.mojo`

Extend `_check_op_inputs` and `_run_op_native` so `infer_train_run_op` accepts the op.

---

## 4. Worked example: adding `gelu` end-to-end

GELU (tanh approximation): `gelu(x) = 0.5 * x * (1 + tanh(a * (x + b * x^3)))` with
`a = sqrt(2/pi)` and `b = 0.044715`. Its derivative is:

```
gelu'(x) = 0.5*(1 + t) + 0.5*x*(1 - t^2)*a*(1 + 3*b*x^2),   t = tanh(a*(x + b*x^3))
```

### 4.1 The CPU kernel — `src/core/ops/cpu/gelu_cpu.mojo`

```mojo
from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.math import tanh
from std.utils.static_tuple import StaticTuple


def _gelu_scalar(x: Float32) -> Float32:
    # a = sqrt(2/pi), b = 0.044715 (tanh approximation)
    var x2 = x * x
    var inner = Float32(0.7978845608028654) * (
        x + Float32(0.044715) * x2 * x
    )
    return Float32(0.5) * x * (Float32(1.0) + tanh(inner))


def _gelu_deriv(x: Float32) -> Float32:
    var x2 = x * x
    var inner = Float32(0.7978845608028654) * (
        x + Float32(0.044715) * x2 * x
    )
    var t = tanh(inner)
    return (
        Float32(0.5) * (Float32(1.0) + t)
        + Float32(0.5) * x * (Float32(1.0) - t * t)
        * Float32(0.7978845608028654)
        * (Float32(1.0) + Float32(3.0) * Float32(0.044715) * x2)
    )


def _gelu_cpu_kernel[dtype: DType](
    x: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    var out = tensor_zeros[dtype, 2](x.shape())
    for i in range(x.numel()):
        out.set(i, Scalar[dtype](_gelu_scalar(Float32(x.get(i)))))
    return out


def gelu_cpu_dynamic[dtype: DType](
    x: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Runtime-shaped GELU."""
    return _gelu_cpu_kernel[dtype](x)


def gelu_cpu_forward_with_saved[dtype: DType](
    x: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = gelu_cpu_dynamic[dtype](x)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(x)
    saved.append(out)
    return (out, saved^)


def gelu_cpu_backward[dtype: DType](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    """grad_x = grad_out * gelu'(x); `saved` = [x, out]."""
    var x = saved[0]
    var grad_x = tensor_zeros[dtype, 2](grad_out.shape())
    for i in range(grad_out.numel()):
        grad_x.set(
            i,
            Scalar[dtype](
                Float32(grad_out.get(i)) * _gelu_deriv(Float32(x.get(i)))
            ),
        )
    var result = List[Tensor[dtype, 2]]()
    result.append(grad_x)
    return result^
```

### 4.2 Dispatch — add to `src/core/ops/base/op_registry.mojo`

```mojo
def _gelu_typed_cpu[dtype: DType](
    inputs: List[AnyTensor]
) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var out = gelu_cpu_dynamic[dtype](x)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def gelu_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _gelu_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _gelu_typed_cpu[DType.float16](inputs)
    unimplemented("gelu_cpu: unsupported dtype")
    return List[AnyTensor]()


def gelu_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    return gelu_dispatch_cpu(inputs)
```

### 4.3 Autograd — add to `src/core/ops/base/op_autograd.mojo`

Mirror the `softmax` entries (`softmax_fws_cpu` is a 1-input, rank-2 op exactly like GELU):

```mojo
def gelu_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var out = gelu_cpu_dynamic[DType.float32](x)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](x))
        saved.append(to_any[DType.float32, 2](out))
        return (outputs^, saved^)
    # ...float16 branch identical, then:
    unimplemented("gelu_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def gelu_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var grads = gelu_cpu_backward[DType.float32](
            g, _rebuild2[DType.float32](saved)
        )
        results.append(to_any[DType.float32, 2](grads[0]))
        return results^
    # ...float16 branch
    unimplemented("gelu_bwd_cpu: unsupported dtype")
    return results^


def gelu_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return gelu_fws_cpu(inputs)


def gelu_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return gelu_bwd_cpu(grad_outputs, saved)
```

### 4.4 Register — in `register_default_ops()`

```mojo
self.register(
    "gelu",
    OpInfo("gelu", gelu_dispatch_cpu, gelu_fws_cpu, gelu_bwd_cpu, Device.CPU, 0),
)
self.register(
    "gelu",
    OpInfo("gelu", gelu_dispatch_gpu, gelu_fws_gpu, gelu_bwd_gpu, Device.MetalGPU, 10),
)
```

### 4.5 C-API whitelist — `src/bindings/infer_train_bindings.mojo`

In `_check_op_inputs` add a branch before the final `return False`:

```mojo
if op == "gelu":
    if len(inputs) != 1 or inputs[0].rank != 2:
        return False
    return inputs[0].dtype == DType.float32 or inputs[0].dtype == DType.float16
```

and in `_run_op_native` dispatch to `gelu_cpu_dynamic` for both dtypes (as `rms_norm` does).

---

## 5. Testing requirements

Add a unit test to `tests/test_ops.mojo` (the M3 numeric suite). It uses the project's
`check_f32` / `check_f16` helpers: tolerance `1e-4` for f32 and `1e-2` for f16.

```mojo
from src.core.ops.cpu.gelu_cpu import gelu_cpu_dynamic
from std.math import tanh

def test_gelu():
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 5))
    var vals = List[Float32]()
    for v in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        vals.append(Float32(v))
    for i in range(5):
        x.set(i, Scalar[DType.float32](vals[i]))
    var out = gelu_cpu_dynamic[DType.float32](x)
    for i in range(5):
        var v = vals[i]
        var inner = Float32(0.7978845608028654) * (
            v + Float32(0.044715) * v * v * v
        )
        var expect = Float32(0.5) * v * (Float32(1.0) + tanh(inner))
        check_f32(Float32(out.get(i)), expect, "gelu[" + String(i) + "]")
```

Wire it into `main()` (`test_gelu()` next to `test_swiglu()`) and build with the Makefile
target (`make test-m3`, which builds `tests/test_ops.mojo` with `-I .` and the thread-pool
`-Xlinker` flag), or directly:

```bash
pixi run mojo build -I . tests/test_ops.mojo \
  -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib -o tests/test_ops
./tests/test_ops
```

For a trainable op also add a backward gradient check to `tests/test_backward.mojo` (finite
difference against the forward) so `make test-m6-mojo` covers the `_bwd_cpu` path.
