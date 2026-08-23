# Adding a New Hardware Platform (CUDA / ROCm / …)

This guide describes the backend contract a new accelerator must satisfy, the minimal kernel
set to implement, the memory model, the C-helper pattern for driver APIs, and how to test
numerical parity against the CPU kernels.

Read these first:

- `src/core/device.mojo` — the `Device` tag and runtime accelerator detection.
- `src/core/tensor.mojo` — `Tensor[dtype, rank]` storage and device transfer.
- `src/core/ops/base/op_interface.mojo` + `op_registry.mojo` — per-device dispatch.
- `src/core/ops/gpu/*.mojo` — the current stub pattern (delegates to CPU).
- `src/core/memory.mojo` + `src/core/thread_pool.mojo` + `tools/thread_pool.c` — the
  C-helper / `external_call` pattern.

Conventions: Mojo 1.0 (only `def`, `mut self`, `^` moves), no trait-existential dispatch —
backends are selected by a tagged `Device` and a `priority` on each `OpInfo`.

---

## 1. The interface contract

### 1.1 Device tags (`device.mojo`)

`Device` is a register-passable struct carrying an `Int8` tag (Mojo 1.0 removed `enum`):

```mojo
struct Device(Copyable, Equatable, Movable, ImplicitlyCopyable):
    var _tag: Int8

    comptime CPU = Device(Int8(0))
    comptime MetalGPU = Device(Int8(1))
    comptime CUDAGPU = Device(Int8(2))
    comptime AMDGPU = Device(Int8(3))

    def is_cpu(self) -> Bool:
        return self._tag == 0

    def is_gpu(self) -> Bool:
        return self._tag != 0
```

`get_default_device()` prefers Metal, then falls back to CPU. Adding a platform means either
using the existing `CUDAGPU` / `AMDGPU` tags or adding a new `comptime` tag + a `name()`
branch, and extending `has_*_gpu()` / `get_default_device()` detection as needed.

### 1.2 Tensor storage (`tensor.mojo`)

`Tensor[dtype, rank]` holds its buffer as `Pointer[Scalar[dtype], MutUntrackedOrigin]`, an
`_device: Device` tag, and a comptime-rank `StaticTuple` shape. Storage is **untracked** —
the `MemoryPool` owns the bulk buffer and is reset wholesale between requests. The two
transfer entry points already exist and are where a real device copy plugs in:

```mojo
def to_device(self, device: Device) -> Tensor[Self.dtype, Self.rank]:
    # M1: copies in host memory and re-tags; a real backend replaces the body
    var result = Tensor[Self.dtype, Self.rank](self._shape, device)
    for i in range(self._numel):
        result._data.unsafe_store(i, self._data.unsafe_load[width=1](offset=i))
    return result

def copy_to_host[dtype: DType, rank: Int](
    src: Tensor[dtype, rank]
) -> Tensor[dtype, rank]:
    return src.to_device(Device.CPU)
```

### 1.3 Per-device dispatch (`op_registry.mojo`)

`OpRegistry` stores a `List[OpInfo]` **per op name, one per device**. `OpRegistry.get`
selects the implementation: an explicit `preferred_device` wins, then the default device,
then the first (CPU) entry. Each `OpInfo` carries a `priority` — CPU is `0`, accelerators
register at `10`:

```mojo
self.register(
    "matmul",
    OpInfo("matmul", matmul_dispatch_cpu, matmul_fws_cpu, matmul_bwd_cpu, Device.CPU, 0),
)
self.register(
    "matmul",
    OpInfo("matmul", matmul_dispatch_gpu, matmul_fws_gpu, matmul_bwd_gpu, Device.MetalGPU, 10),
)
```

`OpInfo`'s three kernels are type-erased function pointers:

```mojo
struct OpInfo(Copyable, Movable, ImplicitlyCopyable):
    var name: String
    var forward: def(List[AnyTensor]) thin -> List[AnyTensor]
    var forward_with_saved: def(List[AnyTensor]) thin -> Tuple[List[AnyTensor], List[AnyTensor]]
    var backward: def(List[AnyTensor], List[AnyTensor]) thin -> List[AnyTensor]
    var device: Device
    var priority: Int
```

The dispatch functions rebuild typed tensors with `from_any[dtype, rank]` and repack with
`to_any[dtype, rank]` (both zero-copy pointer bitcasts).

### 1.4 The current `gpu/` stubs

Every `src/core/ops/gpu/*_gpu.mojo` keeps the dispatch structure in place but **delegates to
the CPU kernel** until a real backend lands. The contract is: only the body of the `*_gpu`
function changes. Example — `src/core/ops/gpu/add_gpu.mojo`:

```mojo
def add_gpu_dynamic[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    _ = _gpu_available[dtype]()
    return add_cpu_dynamic[dtype](a, b)   # replace with a device launch
```

---

## 2. Minimal port steps

### 2.1 Implement the core kernels for the new device

The transformer hot path (`src/core/transformer.mojo`) calls these CPU entry points; each
needs a device counterpart in `src/core/ops/gpu/`:

| Op | CPU kernel (reference) | Notes |
|----|------------------------|-------|
| matmul | `matmul_weight_cpu[_threaded]` in `matmul_cpu.mojo` | weight-major `y = x @ w^T`, `w` is `[out, in]` |
| rms_norm | `rms_norm_weight_cpu` in `rms_norm_cpu.mojo` | weighted, `eps = 1e-5` |
| softmax | `softmax_cpu_dynamic` in `softmax_cpu.mojo` | stable max-subtract along last axis |
| add | `add_cpu_dynamic` / `add_row_cpu` in `add_cpu.mojo` | elementwise + rank-1 bias broadcast |
| swiglu | `swiglu_cpu_dynamic` in `swiglu_cpu.mojo` | `out = silu(gate) * up` |
| embedding | `embedding_cpu_dynamic` in `embedding_cpu.mojo` | `[vocab, hidden]` table, row = token |
| rope | `rope_cpu_dynamic` in `rope_cpu.mojo` | NeoX pairing, `[n_heads, T, head_dim]` |
| mha | `mha_forward` in `attention/mha.mojo` | causal attention, KV-cache mutation |

Keep the numerics rules: fp16 inputs widen per element and accumulate in f32, then cast back
on store (`matmul_cpu.mojo` M3 note); RMSNorm `eps` defaults to `1e-5`.

### 2.2 Register device-specific `OpInfo` entries

For each op, add a `<op>_dispatch_<device>` function and a second `self.register(...)` entry
with the new `Device` tag (priority `10`). Until the real kernels land, the device dispatch
may delegate to CPU — the registry entry keeps the dispatch point in place, exactly as
`mha_dispatch_gpu` does today:

```mojo
def mha_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    # GPU attention falls back to the CPU kernel until the backend lands
    return mha_dispatch_cpu(inputs)
```

### 2.3 Wire memory

`MemoryPool` (`src/core/memory.mojo`) already carries a `Device` and allocates one bulk
block, bumping an offset for aligned sub-ranges; `allocate(size, alignment=64)` and
`reset()` are the API. On M1 the GPU pool reuses the host allocation path because pure
Mojo 1.0 has no host-side GPU allocator. For a real backend, `MemoryPool.__init__` is where
a device-memory allocator (e.g. `cudaMalloc` via the C-helper pattern) replaces
`unsafe_alloc`, and `Tensor.to_device` / `copy_to_host` are where `cudaMemcpy`-style
transfers plug in.

---

## 3. The C-helper pattern for driver APIs

Mojo 1.0's stdlib has no threading and the toolchain cannot link C object files directly, so
driver-facing code is implemented in C and reached through `external_call`. The reference is
`tools/thread_pool.c` (built to `python/infer_train/_lib/libinfer_train_tp.dylib` via the
Makefile `tp` target) plus the mmap helpers the memory layer already calls:

```mojo
# src/core/memory.mojo
var ptr = external_call[
    "it_mmap",
    Pointer[UInt8, MutUntrackedOrigin],
    Pointer[UInt8, MutUntrackedOrigin],
    Pointer[Int64, MutUntrackedOrigin],
](_cstr(file_path), size_slot)
```

The C side exports plain symbols (`it_mmap`, `it_munmap`, `tp_run`, `tp_num_pcores`). The
loader `_load_tp_library()` `dlopen`s the dylib with `RTLD_NOW | RTLD_GLOBAL` so the C side's
`dlsym(RTLD_DEFAULT, ...)` finds Mojo `@export abi("C")` workers. A CUDA/ROCm port follows
the same shape: add `it_cuda_*` / `it_hip_*` symbols to a small C/CUDA shim library, compile
it to a dylib, link with `-Xlinker`, and call it through `external_call`. A worker is a Mojo
function with signature

```mojo
@export
def it_worker(ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64) abi("C"):
    # pure pointer math over ctx; no allocation, no shared writes
```

Keep the proven convention from `matmul_cpu.mojo`: pass context as an **Int64 array**
(`[ptr, ptr, out_ptr, M, K, N, dtype_code, ...]`) rather than a struct — Mojo 1.0
miscompiles field stores onto `unsafe_alloc` slots in shared-library builds, and rebuild the
typed pointers inside the worker with `Pointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=...)`.

---

## 4. Testing: numerical parity vs. the CPU kernels

The acceptance bar is parity against the CPU kernels, which are themselves validated against
llama.cpp / PyTorch (see `tools/ref_forward.py` and `reference_logits_5.npy`). Reuse the
project tolerances:

- **f32**: max abs diff `<= 1e-4` (`check_f32` in `tests/test_ops.mojo`).
- **f16**: max abs diff `<= 1e-2` (`check_f16`; fp16 has ~3 decimal digits).
- **Optimizer verify**: `verify_dags(..., tolerance=Float32(1e-4))` by default.

Per-kernel parity test pattern (mirror `tests/test_ops.mojo`): build the same inputs, run the
CPU kernel and the new device kernel, compare element-wise with the appropriate tolerance:

```mojo
var ref = matmul_weight_cpu[DType.float16](x, w)
var got = matmul_weight_cuda[DType.float16](x, w)   # new backend
for i in range(ref.numel()):
    check_f16(Float32(got.get(i)), Float32(ref.get(i)), "matmul_cuda[" + String(i) + "]")
```

For f16 matmul, allow a slightly looser tolerance than `1e-2` when the device kernel uses a
different reduction order (the CPU kernel already documents that f16 dot products inject
rounding noise vs. f32 accumulation); when in doubt, accumulate in f32 and only cast on
store, matching the CPU kernel. Add each new device kernel to the relevant `tests/*.mojo`
suite and build with `-Xlinker` for the shim library:

```bash
pixi run mojo build -I . tests/test_ops.mojo \
  -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib -o tests/test_ops
./tests/test_ops
```

Finally, run the model-level `test_e2e` / `test_forward` targets to confirm end-to-end
logits still match the reference dump after the new backend is selected by `Device`.
