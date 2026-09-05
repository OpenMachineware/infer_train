# bindings/infer_train_bindings.mojo
#
# M4: C ABI exported from the engine, built with
#
#     pixi run mojo build -I . src/bindings/infer_train_bindings.mojo \
#         --emit shared-lib -o python/infer_train/_lib/libinfer_train.dylib
#
# Every exported function uses `@export` + `abi("C")`; Mojo 1.0 rules that
# shaped the design:
#   * `fn` is gone -> `def`;
#   * fixed-width ABI types are spelled Int32/Int64/Float32/UInt8 (they are
#     auto-imported builtins; `from std.builtin import Int64` does NOT work);
#   * `Pointer` is non-nullable: nullable handles cross the boundary as
#     `Optional[Pointer[...]]` (None <-> NULL, verified against ctypes);
#   * module-level mutable globals are not supported, so the API carries no
#     global error state: failures return NULL / -1 and the Python layer
#     raises with the full context it already has (op name, shapes, node);
#   * heap structs use `unsafe_alloc` + `p[0] = value` (the deprecated
#     `__getitem__` form is the only one available for non-scalar pointers);
#   * `destroy_pointee()` + `unsafe_free()` release a heap struct properly.
#
# Exported surface
# ----------------
#   -- lifecycle --
#   infer_train_version() -> Int64                     (ABI version, 1)
#
#   -- model-level (M4 task 1) --
#   infer_train_load_model(path) -> Model* | NULL      (UTF-8 C string)
#   infer_train_generate(model, prompt, max_tokens,
#                        temperature, top_p, top_k, seed, verbose)
#                                                      -> char* | NULL
#                     (seed < 0 means None; free with infer_train_free_string)
#   infer_train_model_info(model, key) -> Int64   (config queries, -1 unknown)
#   infer_train_reset_cache(model)                     (new conversation)
#   infer_train_free_model(model)
#   infer_train_free_string(s)
#
#   -- tensor-level (for the torch.compile backend) --
#   infer_train_tensor_create(dtype_code, rank, shape, data) -> Tensor* | NULL
#       dtype codes: 0 = f32, 1 = f16, 2 = i32; data is copied in.
#   infer_train_tensor_dtype / _rank / _numel / _shape / _copy_out
#   infer_train_tensor_free(t)
#
#   -- op-level (single-shot; M4 executes each translated FX node as one
#      call instead of a persistent engine-side graph) --
#   infer_train_run_op(op, inputs, n, out_dtype, out_rank, out_shape,
#                      out_numel, out_nbytes) -> raw output buffer | NULL
#   infer_train_free_buffer(p)
#
# The run path validates every op against a whitelist BEFORE dispatching:
# the M1 kernels report bad shapes through `unimplemented()` which calls
# `abort()` and would take the host process down.  A run never aborts.
#
# Memory model: every `Tensor*` handle owns exactly one buffer (either a copy
# of the caller's data or a kernel output).  `infer_train_run_op` returns the
# kernel's output buffer directly; the Python side copies it out immediately
# and releases it with `infer_train_free_buffer`.

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.collections import Span
from std.utils.static_tuple import StaticTuple

# NOTE: this file is a *compilation entry* (built directly with
# `mojo build -I . ...`), so it uses absolute `src.` imports - Mojo entry
# files cannot use `..`-relative imports ("cannot import relative to a
# top-level package").
from src.core.tensor import Tensor
from src.core.device import Device
from src.core.ops.base.op_interface import (
    AnyTensor,
    ANYTENSOR_MAX_RANK,
    from_any,
    to_any,
)
from src.core.ops.cpu.add_cpu import add_cpu_dynamic, add_row_cpu
from src.core.ops.cpu.embedding_cpu import embedding_cpu_dynamic
from src.core.ops.cpu.rms_norm_cpu import rms_norm_cpu_dynamic
from src.core.ops.cpu.softmax_cpu import softmax_cpu_dynamic
from src.core.ops.cpu.swiglu_cpu import swiglu_cpu_dynamic
from src.core.ops.cpu.matmul_cpu import (
    matmul_cpu_dynamic,
    matmul_weight_cpu,
    matmul_weight_cpu_threaded,
)
from src.core.ops.fused.matmul_add import (
    fused_matmul_add_bias,
    fused_matmul_add,
)
from src.core.ops.fused.matmul_rms_norm import fused_matmul_rms_norm
from src.core.ops.fused.swiglu_matmul import fused_swiglu_matmul
from src.runtime.inference import Model, load_model, generate
from src.core.train_optimizer import adamw_step_raw
from src.core.ops.cpu.rms_norm_cpu import rms_norm_weight_cpu
from src.core.ops.loss.cross_entropy import cross_entropy_forward
from src.core.ops.base.op_autograd import (
    mha_seq_fws_cpu,
    mha_seq_bwd_cpu,
    rope_fws_cpu,
    rope_bwd_cpu,
    rms_norm_weight_fws_cpu,
    rms_norm_weight_bwd_cpu,
    matmul_fws_cpu,
    matmul_bwd_cpu,
    lm_head_fws_cpu,
    lm_head_bwd_cpu,
    add_fws_cpu,
    add_bwd_cpu,
    add_bias_fws_cpu,
    add_bias_bwd_cpu,
    rms_norm_fws_cpu,
    rms_norm_bwd_cpu,
    softmax_fws_cpu,
    softmax_bwd_cpu,
    swiglu_fws_cpu,
    swiglu_bwd_cpu,
    embedding_fws_cpu,
    embedding_bwd_cpu,
    swiglu_ffn_fws_cpu,
    swiglu_ffn_bwd_cpu,
    cross_entropy_fws_cpu,
    cross_entropy_bwd_cpu,
)


# -- dtype codes (C ABI convention) ------------------------------------------

comptime DCODE_F32 = 0
comptime DCODE_F16 = 1
comptime DCODE_I32 = 2

comptime ABIVERSION = 1


# -- small helpers -----------------------------------------------------------
#


def _cstr_to_string(p: Pointer[UInt8, MutUntrackedOrigin]) -> String:
    """Copy a NUL-terminated UTF-8 C string into a Mojo String."""
    var len = 0
    while p.unsafe_load(offset=len) != 0:
        len += 1
    var buf = unsafe_alloc[UInt8](len + 1)
    for i in range(len):
        buf.unsafe_offset(i).unsafe_store(val=p.unsafe_load(offset=i))
    buf.unsafe_offset(len).unsafe_store(val=UInt8(0))
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=len)
    return String(unsafe_from_utf8=span)


def _string_to_cstr(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    """Copy a Mojo String into a fresh NUL-terminated UTF-8 C string.

    The caller must release the result with `infer_train_free_string`.
    """
    var bytes = s.as_bytes()
    var n = len(bytes)
    var buf = unsafe_alloc[UInt8](n + 1)
    for i in range(n):
        buf.unsafe_offset(i).unsafe_store(val=bytes[i])
    buf.unsafe_offset(n).unsafe_store(val=UInt8(0))
    return buf


def _dtype_from_code(code: Int32) -> DType:
    if code == DCODE_F16:
        return DType.float16
    if code == DCODE_I32:
        return DType.int32
    return DType.float32


def _code_from_dtype(dtype: DType) -> Int32:
    if dtype == DType.float16:
        return DCODE_F16
    if dtype == DType.int32:
        return DCODE_I32
    return DCODE_F32


def _elem_size_of(dtype: DType) -> Int:
    if dtype == DType.float16:
        return 2
    return 4


# -- lifecycle ---------------------------------------------------------------
#


@export
def it_mw_worker(ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64) abi("C"):
    # The context is an Int64 array [x, w, out, M, K, N, dtype]; the
    # typed pointers are rebuilt with `unsafe_from_address` (plain loads -
    # safe on pool threads).  Mojo-side *struct* contexts are avoided:
    # field stores onto `unsafe_alloc` slots are miscompiled in Mojo 1.0
    # shared libraries.
    var hdr = ctx.unsafe_bitcast[Int64]()
    var x_addr = Int(hdr.unsafe_load(offset=0))
    var w_addr = Int(hdr.unsafe_load(offset=1))
    var out_addr = Int(hdr.unsafe_load(offset=2))
    var M = Int(hdr.unsafe_load(offset=3))
    var K = Int(hdr.unsafe_load(offset=4))
    var N = Int(hdr.unsafe_load(offset=5))
    var dtype_code = Int(hdr.unsafe_load(offset=6))
    var j = Int(idx)
    if dtype_code == 1:
        var xp = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
            unsafe_from_address=x_addr
        )
        var wp = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
            unsafe_from_address=w_addr
        )
        var op = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
            unsafe_from_address=out_addr
        )
        var k_main = (K // 8) * 8
        for i in range(M):
            var acc = SIMD[DType.float32, 8](0)
            var k = 0
            while k < k_main:
                var xv = xp.unsafe_load[width=8](offset=i * K + k).cast[
                    DType.float32
                ]()
                var wv = wp.unsafe_load[width=8](offset=j * K + k).cast[
                    DType.float32
                ]()
                acc = acc + xv * wv
                k += 8
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(xp.unsafe_load(offset=i * K + k)) * Float32(
                    wp.unsafe_load(offset=j * K + k)
                )
                k += 1
            op.unsafe_store(i * N + j, Scalar[DType.float16](total))
    else:
        var xp32 = Pointer[Scalar[DType.float32], MutUntrackedOrigin](
            unsafe_from_address=x_addr
        )
        var wp32 = Pointer[Scalar[DType.float32], MutUntrackedOrigin](
            unsafe_from_address=w_addr
        )
        var op32 = Pointer[Scalar[DType.float32], MutUntrackedOrigin](
            unsafe_from_address=out_addr
        )
        var k_main32 = (K // 4) * 4
        for i in range(M):
            var acc = SIMD[DType.float32, 4](0)
            var k = 0
            while k < k_main32:
                var xv = xp32.unsafe_load[width=4](offset=i * K + k)
                var wv = wp32.unsafe_load[width=4](offset=j * K + k)
                acc = acc + xv * wv
                k += 4
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(xp32.unsafe_load(offset=i * K + k)) * Float32(
                    wp32.unsafe_load(offset=j * K + k)
                )
                k += 1
            op32.unsafe_store(i * N + j, Scalar[DType.float32](total))


@export
def it_mw_multi_worker(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64
) abi("C"):
    # batched matmul worker: ctx = [M, K, dtype, n_pairs, n1, n2, n3,
    # (x, w, out) x n_pairs].  Tasks are distributed across the per-pair
    # column ranges: pair p owns columns [offset_p, offset_p + n_p).
    var hdr = ctx.unsafe_bitcast[Int64]()
    var M = Int(hdr.unsafe_load(offset=0))
    var K = Int(hdr.unsafe_load(offset=1))
    var dtype_code = Int(hdr.unsafe_load(offset=2))
    var n_pairs = Int(hdr.unsafe_load(offset=3))
    var n1 = Int(hdr.unsafe_load(offset=4))
    var n2 = Int(hdr.unsafe_load(offset=5))
    var n3 = Int(hdr.unsafe_load(offset=6))
    var task = Int(idx)
    var pair = 0
    var j = task
    var np = n1
    if task >= n1:
        pair = 1
        j = task - n1
        np = n2
        if task >= n1 + n2:
            pair = 2
            j = task - n1 - n2
            np = n3
    if pair >= n_pairs:
        return
    var base = 7 + pair * 3
    var x_addr = Int(hdr.unsafe_load(offset=base))
    var w_addr = Int(hdr.unsafe_load(offset=base + 1))
    var out_addr = Int(hdr.unsafe_load(offset=base + 2))
    if dtype_code == 1:
        var xp = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
            unsafe_from_address=x_addr
        )
        var wp = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
            unsafe_from_address=w_addr
        )
        var op = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
            unsafe_from_address=out_addr
        )
        var k_main = (K // 8) * 8
        for i in range(M):
            var acc = SIMD[DType.float32, 8](0)
            var k = 0
            while k < k_main:
                var xv = xp.unsafe_load[width=8](offset=i * K + k).cast[
                    DType.float32
                ]()
                var wv = wp.unsafe_load[width=8](offset=j * K + k).cast[
                    DType.float32
                ]()
                acc = acc + xv * wv
                k += 8
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(xp.unsafe_load(offset=i * K + k)) * Float32(
                    wp.unsafe_load(offset=j * K + k)
                )
                k += 1
            op.unsafe_store(i * np + j, Scalar[DType.float16](total))
    else:
        var xp32 = Pointer[Scalar[DType.float32], MutUntrackedOrigin](
            unsafe_from_address=x_addr
        )
        var wp32 = Pointer[Scalar[DType.float32], MutUntrackedOrigin](
            unsafe_from_address=w_addr
        )
        var op32 = Pointer[Scalar[DType.float32], MutUntrackedOrigin](
            unsafe_from_address=out_addr
        )
        var k_main32 = (K // 4) * 4
        for i in range(M):
            var acc = SIMD[DType.float32, 4](0)
            var k = 0
            while k < k_main32:
                var xv = xp32.unsafe_load[width=4](offset=i * K + k)
                var wv = wp32.unsafe_load[width=4](offset=j * K + k)
                acc = acc + xv * wv
                k += 4
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(xp32.unsafe_load(offset=i * K + k)) * Float32(
                    wp32.unsafe_load(offset=j * K + k)
                )
                k += 1
            op32.unsafe_store(i * np + j, Scalar[DType.float32](total))


@export
def infer_train_version() abi("C") -> Int64:
    return ABIVERSION


# -- model-level API (M4 task 1) --------------------------------------------
#


@export
def infer_train_load_model(
    path: Pointer[UInt8, MutUntrackedOrigin],
) abi("C") -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    try:
        var path_s = _cstr_to_string(path)
        var m = unsafe_alloc[Model](1)
        m[unsafe_offset=0] = load_model(path_s)
        return Optional(m.unsafe_bitcast[UInt8]())
    except:
        return None


@export
def infer_train_generate(
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    prompt: Pointer[UInt8, MutUntrackedOrigin],
    max_tokens: Int64,
    temperature: Float32,
    top_p: Float32,
    top_k: Int64,
    seed: Int64,
    verbose: Int32,
) abi("C") -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    try:
        if not m:
            return None
        var model = m.value().unsafe_bitcast[Model]()
        var prompt_s = _cstr_to_string(prompt)
        var opt_seed: Optional[Int] = None
        if seed >= 0:
            opt_seed = Optional(Int(seed))
        var out = generate(
            model[unsafe_offset=0],
            prompt_s,
            Int(max_tokens),
            temperature,
            top_p,
            Int(top_k),
            verbose != 0,
            opt_seed,
        )
        return Optional(_string_to_cstr(out))
    except:
        return None


@export
def infer_train_model_info(
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    key: Pointer[UInt8, MutUntrackedOrigin],
) abi("C") -> Int64:
    if not m:
        return -1
    var model = m.value().unsafe_bitcast[Model]()
    var k = _cstr_to_string(key)
    var cfg = model[unsafe_offset=0].transformer.config
    if k == "n_layers":
        return Int64(cfg.n_layers)
    if k == "hidden":
        return Int64(cfg.hidden)
    if k == "ffn":
        return Int64(cfg.ffn)
    if k == "n_heads":
        return Int64(cfg.n_heads)
    if k == "n_kv_heads":
        return Int64(cfg.n_kv_heads)
    if k == "head_dim":
        return Int64(cfg.head_dim)
    if k == "vocab":
        return Int64(cfg.vocab)
    if k == "bos":
        return Int64(cfg.bos_id)
    if k == "eos":
        return Int64(cfg.eos_id)
    if k == "ctx_len":
        return Int64(model[unsafe_offset=0].transformer.cache.capacity())
    return -1


@export
def infer_train_reset_cache(
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
) abi("C"):
    if not m:
        return
    var model = m.value().unsafe_bitcast[Model]()
    model[unsafe_offset=0].transformer.reset_cache()


@export
def infer_train_free_string(s: Pointer[UInt8, MutUntrackedOrigin]) abi("C"):
    s.unsafe_free()


def _free_buffer[dtype: DType, rank: Int](t: Tensor[dtype, rank]):
    t.data().unsafe_bitcast[UInt8]().unsafe_free()


def _free_model_contents(mut model: Model):
    """Best-effort release of the buffers M1-M3 leaves untracked.

    Tensor storage is `Pointer[..., MutUntrackedOrigin]` (never freed by any
    destructor), so we walk the dequantized weights, the KV cache and the
    GGUF file buffer explicitly.  The tokenizer/registry internals are owned
    by the struct fields themselves and are released by
    `unsafe_deinit_pointee()`.
    """
    _free_buffer[DType.float16, 2](model.transformer.params.token_embd)
    _free_buffer[DType.float16, 1](model.transformer.params.output_norm_w)
    _free_buffer[DType.float16, 2](model.transformer.params.output_w)
    for i in range(len(model.transformer.params.layers)):
        var lw = model.transformer.params.layers[i]
        _free_buffer[DType.float16, 1](lw.attn_norm_w)
        _free_buffer[DType.float16, 2](lw.q_w)
        _free_buffer[DType.float16, 2](lw.k_w)
        _free_buffer[DType.float16, 2](lw.v_w)
        _free_buffer[DType.float16, 2](lw.o_w)
        _free_buffer[DType.float16, 1](lw.q_b)
        _free_buffer[DType.float16, 1](lw.k_b)
        _free_buffer[DType.float16, 1](lw.v_b)
        _free_buffer[DType.float16, 1](lw.ffn_norm_w)
        _free_buffer[DType.float16, 2](lw.gate_w)
        _free_buffer[DType.float16, 2](lw.up_w)
        _free_buffer[DType.float16, 2](lw.down_w)
    for i in range(model.transformer.cache.num_layers()):
        _free_buffer[DType.float16, 3](model.transformer.cache.layers[i].k)
        _free_buffer[DType.float16, 3](model.transformer.cache.layers[i].v)
    model.transformer.ctx.munmap_all()


@export
def infer_train_free_model(
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
) abi("C"):
    if not m:
        return
    var model = m.value().unsafe_bitcast[Model]()
    _free_model_contents(model[unsafe_offset=0])
    model.unsafe_deinit_pointee()
    # M4 note: `model.unsafe_free()` is deliberately NOT called.  In Mojo
    # 1.0 shared-library builds, freeing a heap struct whose fields own
    # heap containers (List/Dict) corrupts or hangs the process allocator
    # (reproduced with a minimal Dict-owning struct).  The deinit above
    # releases everything the struct owns; only the struct shell (~a few
    # hundred bytes) leaks until process exit.  M5 revisits with the
    # memory-planning milestone.


# -- tensor-level API --------------------------------------------------------
#


struct CTensor(Movable):
    """Handle wrapper: a type-erased tensor plus the buffer we must free."""

    var any: AnyTensor
    var owned: Pointer[UInt8, MutUntrackedOrigin]

    def __init__(
        out self,
        var any: AnyTensor,
        owned: Pointer[UInt8, MutUntrackedOrigin],
    ):
        self.any = any
        self.owned = owned


@export
def infer_train_tensor_create(
    dtype_code: Int32,
    rank: Int64,
    shape: Pointer[Int64, MutUntrackedOrigin],
    data: Pointer[UInt8, MutUntrackedOrigin],
) abi("C") -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    if rank < 1 or rank > ANYTENSOR_MAX_RANK:
        return None
    if (
        dtype_code != DCODE_F32
        and dtype_code != DCODE_F16
        and dtype_code != DCODE_I32
    ):
        return None
    var dtype = _dtype_from_code(dtype_code)
    var static_shape = StaticTuple[Int, ANYTENSOR_MAX_RANK](fill=0)
    var numel = 1
    for i in range(rank):
        var d = Int(shape.unsafe_load(offset=i))
        if d < 0:
            return None
        static_shape[i] = d
        numel *= d
    var nbytes = numel * _elem_size_of(dtype)
    if nbytes < 1:
        nbytes = 1
    var buf = unsafe_alloc[UInt8](nbytes, alignment=64)
    for i in range(nbytes):
        buf.unsafe_offset(i).unsafe_store(val=data.unsafe_load(offset=i))
    var any = AnyTensor(dtype, Int(rank), static_shape, numel, Device.CPU, buf)
    var t = unsafe_alloc[CTensor](1)
    t[unsafe_offset=0] = CTensor(any, buf)
    return Optional(t.unsafe_bitcast[UInt8]())


@export
def infer_train_tensor_dtype(
    t: Optional[Pointer[UInt8, MutUntrackedOrigin]],
) abi("C") -> Int32:
    if not t:
        return -1
    var ct = t.value().unsafe_bitcast[CTensor]()
    return _code_from_dtype(ct[unsafe_offset=0].any.dtype)


@export
def infer_train_tensor_rank(
    t: Optional[Pointer[UInt8, MutUntrackedOrigin]],
) abi("C") -> Int64:
    if not t:
        return -1
    var ct = t.value().unsafe_bitcast[CTensor]()
    return Int64(ct[unsafe_offset=0].any.rank)


@export
def infer_train_tensor_numel(
    t: Optional[Pointer[UInt8, MutUntrackedOrigin]],
) abi("C") -> Int64:
    if not t:
        return -1
    var ct = t.value().unsafe_bitcast[CTensor]()
    return Int64(ct[unsafe_offset=0].any.numel)


@export
def infer_train_tensor_shape(
    t: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    out_shape: Pointer[Int64, MutUntrackedOrigin],
) abi("C") -> Int64:
    """
    Write the dims into `out_shape` (caller provides `rank` slots);
    returns rank.
    """
    if not t:
        return -1
    var ct = t.value().unsafe_bitcast[CTensor]()
    var rank = ct[unsafe_offset=0].any.rank
    for i in range(rank):
        out_shape.unsafe_offset(i).unsafe_store(
            val=Int64(ct[unsafe_offset=0].any.shape[i])
        )
    return Int64(rank)


@export
def infer_train_tensor_copy_out(
    t: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    dst: Pointer[UInt8, MutUntrackedOrigin],
) abi("C") -> Int64:
    """Copy the raw payload (numel * elem_size bytes) into `dst`."""
    if not t:
        return -1
    var ct = t.value().unsafe_bitcast[CTensor]()
    var nbytes = ct[unsafe_offset=0].any.numel * _elem_size_of(
        ct[unsafe_offset=0].any.dtype
    )
    for i in range(nbytes):
        dst.unsafe_offset(i).unsafe_store(
            val=ct[unsafe_offset=0].any.data.unsafe_load(offset=i)
        )
    return Int64(nbytes)


@export
def infer_train_tensor_free(
    t: Optional[Pointer[UInt8, MutUntrackedOrigin]],
) abi("C"):
    if not t:
        return
    var ct = t.value().unsafe_bitcast[CTensor]()
    ct[unsafe_offset=0].owned.unsafe_free()
    ct.unsafe_free()


# -- op-level API (single-shot execution) ------------------------------------
#
# M4 executes each translated FX node as ONE exported call: the op runs on
# a freshly built input list and the raw output buffer is handed back to the
# caller (who frees it with `infer_train_free_buffer`).  There is no
# persistent engine-side graph object.
#
# WHY: Mojo 1.0's shared-library codegen miscompiles heap structs that own
# containers (List/Dict) in two independent ways - (1) move-assigning onto
# a raw `unsafe_alloc` slot tears the uninitialized memory down, and (2)
# `List[AnyTensor]` growth/appends write past the list buffer.  Both corrupt
# the process heap across exported-call boundaries (reproduced with minimal
# structs; the same code runs cleanly in an executable, which is why the M3
# test suite never saw it).  The single-shot design keeps every container on
# the stack, pre-reserves the one AnyTensor list it needs, and returns only
# plain buffers - none of the miscompiled paths are reachable.  The
# Interpreter/OpRegistry graph execution path stays in M1-M3 untouched and
# returns in M5 (full-graph batching) once the toolchain bug is fixed or the
# interpreter's list handling is rewritten.


def _collect_inputs(
    inputs: Pointer[Int64, MutUntrackedOrigin], n_inputs: Int
) -> List[AnyTensor]:
    """Collect the external input AnyTensors.

    The C side passes handle *addresses* as Int64s: Mojo 1.0's
    `Pointer[Pointer[T]].__getitem__` lowers to a slot address instead of a
    loaded element in shared-library builds (verified against ctypes), so we
    keep pointer-to-pointer indexing out of the ABI surface and rebuild the
    handle pointers from raw addresses here.  The list is pre-reserved:
    growing appends are one of the miscompiled paths (see the module
    section header).
    """
    var input_list = List[AnyTensor]()
    input_list.reserve(8)
    for i in range(n_inputs):
        var addr = inputs.unsafe_load(offset=i)
        var hp = Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(addr)
        )
        var ct = hp.unsafe_bitcast[CTensor]()
        input_list.append(ct[unsafe_offset=0].any)
    return input_list^


def _check_op_inputs(op: String, inputs: List[AnyTensor]) -> Bool:
    """Validate an op's inputs BEFORE dispatch.

    The M1 kernels report contract violations through `unimplemented()`,
    which calls `abort()`; this check keeps a bad C call from taking the
    host process down.  Returns False on any contract violation.
    """
    if op == "matmul" or op == "lm_head":
        if len(inputs) != 2:
            return False
        var a = inputs[0]
        var b = inputs[1]
        if a.rank != 2 or b.rank != 2 or a.dtype != b.dtype:
            return False
        if a.dtype != DType.float32 and a.dtype != DType.float16:
            return False
        if op == "matmul":
            return a.shape[1] == b.shape[0]
        # lm_head: y = W @ x with W stored [out, in]
        return a.shape[1] == b.shape[1]
    if op == "add":
        if len(inputs) != 2:
            return False
        var a = inputs[0]
        var b = inputs[1]
        if (
            a.rank != 2
            or b.rank != 2
            or a.dtype != b.dtype
            or a.shape != b.shape
        ):
            return False
        return a.dtype == DType.float32 or a.dtype == DType.float16
    if op == "add_bias":
        if len(inputs) != 2:
            return False
        var x = inputs[0]
        var bias = inputs[1]
        if x.rank != 2 or bias.rank != 1 or x.dtype != bias.dtype:
            return False
        if x.dtype != DType.float32 and x.dtype != DType.float16:
            return False
        return x.shape[1] == bias.shape[0]
    if op == "rms_norm" or op == "softmax":
        if len(inputs) != 1 or inputs[0].rank != 2:
            return False
        return (
            inputs[0].dtype == DType.float32
            or inputs[0].dtype == DType.float16
        )
    if op == "swiglu":
        if len(inputs) != 2:
            return False
        var gate = inputs[0]
        var up = inputs[1]
        if (
            gate.rank != 2
            or up.rank != 2
            or gate.dtype != up.dtype
            or gate.shape != up.shape
        ):
            return False
        return gate.dtype == DType.float32 or gate.dtype == DType.float16
    if op == "swiglu_ffn":
        if len(inputs) != 4:
            return False
        var x = inputs[0]
        var gate_w = inputs[1]
        var up_w = inputs[2]
        var down_w = inputs[3]
        for t in inputs:
            if t.rank != 2:
                return False
            if t.dtype != x.dtype:
                return False
        if x.dtype != DType.float32 and x.dtype != DType.float16:
            return False
        # x[M, H], gate_w[F, H], up_w[F, H], down_w[H, F]
        if x.shape[1] != gate_w.shape[1] or x.shape[1] != up_w.shape[1]:
            return False
        if gate_w.shape[0] != up_w.shape[0]:
            return False
        return x.shape[1] == down_w.shape[0]
    if op == "embedding":
        if len(inputs) != 2:
            return False
        var tokens = inputs[0]
        var table = inputs[1]
        if tokens.rank != 1 or tokens.dtype != DType.int32:
            return False
        if table.rank != 2:
            return False
        return table.dtype == DType.float32 or table.dtype == DType.float16
    if op == "fused_matmul_add_bias":
        if len(inputs) != 3:
            return False
        var x = inputs[0]
        var w = inputs[1]
        var bias = inputs[2]
        if x.rank != 2 or w.rank != 2 or bias.rank != 1:
            return False
        if x.dtype != w.dtype or x.dtype != bias.dtype:
            return False
        if x.dtype != DType.float32 and x.dtype != DType.float16:
            return False
        return x.shape[1] == w.shape[1] and bias.shape[0] == w.shape[0]
    if op == "fused_matmul_add":
        if len(inputs) != 3:
            return False
        var x2 = inputs[0]
        var w2 = inputs[1]
        var b2 = inputs[2]
        if x2.rank != 2 or w2.rank != 2 or b2.rank != 2:
            return False
        if x2.dtype != w2.dtype or x2.dtype != b2.dtype:
            return False
        if x2.dtype != DType.float32 and x2.dtype != DType.float16:
            return False
        if x2.shape[1] != w2.shape[1]:
            return False
        return b2.shape[0] == x2.shape[0] and b2.shape[1] == w2.shape[0]
    if op == "fused_matmul_rms_norm":
        if len(inputs) != 2:
            return False
        var x3 = inputs[0]
        var w3 = inputs[1]
        if x3.rank != 2 or w3.rank != 2 or x3.dtype != w3.dtype:
            return False
        if x3.dtype != DType.float32 and x3.dtype != DType.float16:
            return False
        return x3.shape[1] == w3.shape[1]
    if op == "fused_swiglu_matmul":
        if len(inputs) != 3:
            return False
        var g = inputs[0]
        var u = inputs[1]
        var w4 = inputs[2]
        if g.rank != 2 or u.rank != 2 or w4.rank != 2:
            return False
        if g.dtype != u.dtype or g.dtype != w4.dtype:
            return False
        if g.dtype != DType.float32 and g.dtype != DType.float16:
            return False
        if g.shape != u.shape:
            return False
        return g.shape[1] == w4.shape[1]
    if op == "rms_norm_weight":
        if len(inputs) != 2:
            return False
        var xw = inputs[0]
        var ww = inputs[1]
        if xw.rank != 2 or ww.rank != 1 or xw.dtype != ww.dtype:
            return False
        if xw.dtype != DType.float32 and xw.dtype != DType.float16:
            return False
        return xw.shape[1] == ww.shape[0]
    if op == "cross_entropy":
        if len(inputs) != 2:
            return False
        var logits = inputs[0]
        var targets = inputs[1]
        if logits.rank != 2 or targets.rank != 1:
            return False
        if targets.dtype != DType.int32:
            return False
        if logits.dtype != DType.float32 and logits.dtype != DType.float16:
            return False
        return logits.shape[0] == targets.shape[0]
    if op == "mha_seq":
        if len(inputs) != 10:
            return False
        var xs = inputs[0]
        if xs.dtype != DType.float32 and xs.dtype != DType.float16:
            return False
        if xs.rank != 2:
            return False
        var cfg = inputs[8]
        var pos = inputs[9]
        if cfg.rank != 1 or cfg.dtype != DType.int32 or cfg.shape[0] != 3:
            return False
        if pos.rank != 1 or pos.dtype != DType.float32 or pos.shape[0] != 2:
            return False
        for k in range(1, 8):
            if inputs[k].rank != (2 if k <= 4 else 1):
                return False
            if inputs[k].dtype != xs.dtype:
                return False
        # projection width checks
        if xs.shape[1] != inputs[1].shape[1]:
            return False
        if inputs[1].shape[0] != inputs[5].shape[0]:
            return False
        if inputs[2].shape[0] != inputs[6].shape[0]:
            return False
        if inputs[3].shape[0] != inputs[7].shape[0]:
            return False
        if inputs[4].shape[0] != xs.shape[1]:
            return False
        if inputs[4].shape[1] != inputs[1].shape[0]:
            return False
        return True
    if op == "rope":
        if len(inputs) != 2:
            return False
        var xr = inputs[0]
        var pr = inputs[1]
        if xr.rank != 3 or pr.rank != 1 or pr.shape[0] != 1:
            return False
        if pr.dtype != DType.float32:
            return False
        return xr.dtype == DType.float32 or xr.dtype == DType.float16
    # mha and anything else stay off the C API whitelist.
    return False


def _run_op_native(op: String, inputs: List[AnyTensor]) -> Optional[AnyTensor]:
    """Run one whitelisted op by calling the kernels directly.

    The M1 registry dispatch functions build their result list with
    `List[AnyTensor].append`, which is miscompiled in Mojo 1.0 shared
    libraries (see the module section header).  All whitelisted M4 ops
    return exactly one tensor, so this local dispatch calls the same
    kernels the OpInfo entries point to without building AnyTensor lists.
    `_check_op_inputs` has already validated the contract; nothing here
    can reach `unimplemented()`.
    """
    if op == "matmul":
        var a = inputs[0]
        var b = inputs[1]
        if a.dtype == DType.float32:
            var x = from_any[DType.float32, 2](a)
            var y = from_any[DType.float32, 2](b)
            return Optional(
                to_any[DType.float32, 2](
                    matmul_cpu_dynamic[DType.float32](x, y)
                )
            )
        var x16 = from_any[DType.float16, 2](a)
        var y16 = from_any[DType.float16, 2](b)
        return Optional(
            to_any[DType.float16, 2](
                matmul_cpu_dynamic[DType.float16](x16, y16)
            )
        )
    if op == "lm_head":
        var a = inputs[0]
        var w = inputs[1]
        if a.dtype == DType.float32:
            var x = from_any[DType.float32, 2](a)
            var w32 = from_any[DType.float32, 2](w)
            return Optional(
                to_any[DType.float32, 2](
                    matmul_weight_cpu_threaded[DType.float32](x, w32)
                )
            )
        var x16 = from_any[DType.float16, 2](a)
        var w16 = from_any[DType.float16, 2](w)
        return Optional(
            to_any[DType.float16, 2](
                matmul_weight_cpu_threaded[DType.float16](x16, w16)
            )
        )
    if op == "add":
        var a = inputs[0]
        var b = inputs[1]
        if a.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    add_cpu_dynamic[DType.float32](
                        from_any[DType.float32, 2](a),
                        from_any[DType.float32, 2](b),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                add_cpu_dynamic[DType.float16](
                    from_any[DType.float16, 2](a),
                    from_any[DType.float16, 2](b),
                )
            )
        )
    if op == "add_bias":
        var x_any = inputs[0]
        var bias = inputs[1]
        if x_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    add_row_cpu[DType.float32](
                        from_any[DType.float32, 2](x_any),
                        from_any[DType.float32, 1](bias),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                add_row_cpu[DType.float16](
                    from_any[DType.float16, 2](x_any),
                    from_any[DType.float16, 1](bias),
                )
            )
        )
    if op == "rms_norm":
        var x_any = inputs[0]
        if x_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    rms_norm_cpu_dynamic[DType.float32](
                        from_any[DType.float32, 2](x_any)
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                rms_norm_cpu_dynamic[DType.float16](
                    from_any[DType.float16, 2](x_any)
                )
            )
        )
    if op == "softmax":
        var x_any = inputs[0]
        if x_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    softmax_cpu_dynamic[DType.float32](
                        from_any[DType.float32, 2](x_any)
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                softmax_cpu_dynamic[DType.float16](
                    from_any[DType.float16, 2](x_any)
                )
            )
        )
    if op == "swiglu":
        var gate = inputs[0]
        var up = inputs[1]
        if gate.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    swiglu_cpu_dynamic[DType.float32](
                        from_any[DType.float32, 2](gate),
                        from_any[DType.float32, 2](up),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                swiglu_cpu_dynamic[DType.float16](
                    from_any[DType.float16, 2](gate),
                    from_any[DType.float16, 2](up),
                )
            )
        )
    if op == "swiglu_ffn":
        var x_any = inputs[0]
        var gate_w = inputs[1]
        var up_w = inputs[2]
        var down_w = inputs[3]
        if x_any.dtype == DType.float32:
            var x = from_any[DType.float32, 2](x_any)
            var gw = from_any[DType.float32, 2](gate_w)
            var uw = from_any[DType.float32, 2](up_w)
            var dw = from_any[DType.float32, 2](down_w)
            var g = matmul_weight_cpu[DType.float32](x, gw)
            var u = matmul_weight_cpu[DType.float32](x, uw)
            var h = swiglu_cpu_dynamic[DType.float32](g, u)
            return Optional(
                to_any[DType.float32, 2](
                    matmul_weight_cpu[DType.float32](h, dw)
                )
            )
        var x = from_any[DType.float16, 2](x_any)
        var gw = from_any[DType.float16, 2](gate_w)
        var uw = from_any[DType.float16, 2](up_w)
        var dw = from_any[DType.float16, 2](down_w)
        var g = matmul_weight_cpu[DType.float16](x, gw)
        var u = matmul_weight_cpu[DType.float16](x, uw)
        var h = swiglu_cpu_dynamic[DType.float16](g, u)
        return Optional(
            to_any[DType.float16, 2](matmul_weight_cpu[DType.float16](h, dw))
        )
    if op == "embedding":
        var tokens = inputs[0]
        var table = inputs[1]
        if table.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    embedding_cpu_dynamic[DType.float32](
                        from_any[DType.int32, 1](tokens),
                        from_any[DType.float32, 2](table),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                embedding_cpu_dynamic[DType.float16](
                    from_any[DType.int32, 1](tokens),
                    from_any[DType.float16, 2](table),
                )
            )
        )
    if op == "fused_matmul_add_bias":
        var x_any = inputs[0]
        var w_any = inputs[1]
        var bias_any = inputs[2]
        if x_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    fused_matmul_add_bias[DType.float32](
                        from_any[DType.float32, 2](x_any),
                        from_any[DType.float32, 2](w_any),
                        from_any[DType.float32, 1](bias_any),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                fused_matmul_add_bias[DType.float16](
                    from_any[DType.float16, 2](x_any),
                    from_any[DType.float16, 2](w_any),
                    from_any[DType.float16, 1](bias_any),
                )
            )
        )
    if op == "fused_matmul_add":
        var x2_any = inputs[0]
        var w2_any = inputs[1]
        var b2_any = inputs[2]
        if x2_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    fused_matmul_add[DType.float32](
                        from_any[DType.float32, 2](x2_any),
                        from_any[DType.float32, 2](w2_any),
                        from_any[DType.float32, 2](b2_any),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                fused_matmul_add[DType.float16](
                    from_any[DType.float16, 2](x2_any),
                    from_any[DType.float16, 2](w2_any),
                    from_any[DType.float16, 2](b2_any),
                )
            )
        )
    if op == "fused_matmul_rms_norm":
        var x3_any = inputs[0]
        var w3_any = inputs[1]
        if x3_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    fused_matmul_rms_norm[DType.float32](
                        from_any[DType.float32, 2](x3_any),
                        from_any[DType.float32, 2](w3_any),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                fused_matmul_rms_norm[DType.float16](
                    from_any[DType.float16, 2](x3_any),
                    from_any[DType.float16, 2](w3_any),
                )
            )
        )
    if op == "fused_swiglu_matmul":
        var g_any = inputs[0]
        var u_any = inputs[1]
        var w4_any = inputs[2]
        if g_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    fused_swiglu_matmul[DType.float32](
                        from_any[DType.float32, 2](g_any),
                        from_any[DType.float32, 2](u_any),
                        from_any[DType.float32, 2](w4_any),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                fused_swiglu_matmul[DType.float16](
                    from_any[DType.float16, 2](g_any),
                    from_any[DType.float16, 2](u_any),
                    from_any[DType.float16, 2](w4_any),
                )
            )
        )
    if op == "rms_norm_weight":
        var xw_any = inputs[0]
        var ww_any = inputs[1]
        if xw_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 2](
                    rms_norm_weight_cpu[DType.float32](
                        from_any[DType.float32, 2](xw_any),
                        from_any[DType.float32, 1](ww_any),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 2](
                rms_norm_weight_cpu[DType.float16](
                    from_any[DType.float16, 2](xw_any),
                    from_any[DType.float16, 1](ww_any),
                )
            )
        )
    if op == "cross_entropy":
        var logits_any = inputs[0]
        var targets_any = inputs[1]
        if logits_any.dtype == DType.float32:
            return Optional(
                to_any[DType.float32, 1](
                    cross_entropy_forward[DType.float32](
                        from_any[DType.float32, 2](logits_any),
                        from_any[DType.int32, 1](targets_any),
                    )
                )
            )
        return Optional(
            to_any[DType.float16, 1](
                cross_entropy_forward[DType.float16](
                    from_any[DType.float16, 2](logits_any),
                    from_any[DType.int32, 1](targets_any),
                )
            )
        )
    if op == "mha_seq":
        var fws = mha_seq_fws_cpu(inputs)
        if len(fws[0]) == 0:
            return None
        return Optional(fws[0][0])
    if op == "rope":
        var fws = rope_fws_cpu(inputs)
        if len(fws[0]) == 0:
            return None
        return Optional(fws[0][0])
    return None


@export
def infer_train_run_op(
    op_ptr: Pointer[UInt8, MutUntrackedOrigin],
    inputs: Pointer[Int64, MutUntrackedOrigin],
    n_inputs: Int64,
    out_dtype: Pointer[Int64, MutUntrackedOrigin],
    out_rank: Pointer[Int64, MutUntrackedOrigin],
    out_shape: Pointer[Int64, MutUntrackedOrigin],
    out_numel: Pointer[Int64, MutUntrackedOrigin],
    out_nbytes: Pointer[Int64, MutUntrackedOrigin],
) abi("C") -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    """Run one op; returns the raw output buffer (free via
    `infer_train_free_buffer`) and fills the out-params with its metadata.
    Returns NULL on any validation error (the Python layer raises with
    full context).
    """
    var op = _cstr_to_string(op_ptr)
    var input_list = _collect_inputs(inputs, Int(n_inputs))
    if not _check_op_inputs(op, input_list):
        return None
    var result = _run_op_native(op, input_list)
    if not result:
        return None
    var any = result.value()
    out_dtype.unsafe_offset(0).unsafe_store(
        val=Int64(_code_from_dtype(any.dtype))
    )
    out_rank.unsafe_offset(0).unsafe_store(val=Int64(any.rank))
    for i in range(any.rank):
        out_shape.unsafe_offset(i).unsafe_store(val=Int64(any.shape[i]))
    out_numel.unsafe_offset(0).unsafe_store(val=Int64(any.numel))
    out_nbytes.unsafe_offset(0).unsafe_store(
        val=Int64(any.numel * _elem_size_of(any.dtype))
    )
    return Optional(any.data)


# -- M6: op-level backward (single-shot) -------------------------------------
#
# infer_train_run_backward(op, inputs, grad_outputs, ...) computes the op's
# backward: it replays forward_with_saved internally (so the caller passes
# the same inputs as the forward) and returns one gradient per input
# (zero-numel entries are "no gradient" sentinels).


def _check_grad_shapes(
    op: String, inputs: List[AnyTensor], grad_outs: List[AnyTensor]
) -> Bool:
    if len(grad_outs) < 1:
        return False
    var g = grad_outs[0]
    var x = inputs[0]
    if op != "embedding":
        # embedding's first input is the integer token ids; the gradient
        # always lives in the table's floating dtype
        if g.dtype != x.dtype:
            return False
    if op == "matmul":
        var b = inputs[1]
        return (
            g.rank == 2
            and g.shape[0] == x.shape[0]
            and g.shape[1] == b.shape[1]
        )
    if op == "lm_head":
        var w = inputs[1]
        return (
            g.rank == 2
            and g.shape[0] == x.shape[0]
            and g.shape[1] == w.shape[0]
        )
    if op == "add" or op == "rms_norm" or op == "softmax":
        return g.rank == 2 and g.shape == x.shape
    if op == "rms_norm_weight":
        return g.rank == 2 and g.shape == x.shape
    if op == "mha_seq":
        return g.rank == 2 and g.shape == x.shape
    if op == "rope":
        return g.rank == 3 and g.shape == x.shape
    if op == "add_bias":
        return g.rank == 2 and g.shape == x.shape
    if op == "swiglu":
        return g.rank == 2 and g.shape == x.shape
    if op == "swiglu_ffn":
        return (
            g.rank == 2
            and g.shape[0] == x.shape[0]
            and g.shape[1] == x.shape[1]
        )
    if op == "embedding":
        var table = inputs[1]
        return (
            g.rank == 2
            and g.dtype == table.dtype
            and g.shape[0] == x.shape[0]
            and g.shape[1] == table.shape[1]
        )
    if op == "cross_entropy":
        return g.rank == 1 and g.numel == 1
    return False


def _run_backward_native(
    op: String, inputs: List[AnyTensor], grad_outs: List[AnyTensor]
) -> List[AnyTensor]:
    """One whitelisted op's backward (single-shot, pre-reserved lists)."""
    if op == "matmul":
        var fws = matmul_fws_cpu(inputs)
        return matmul_bwd_cpu(grad_outs, fws[1])
    if op == "lm_head":
        var fws = lm_head_fws_cpu(inputs)
        return lm_head_bwd_cpu(grad_outs, fws[1])
    if op == "add":
        var fws = add_fws_cpu(inputs)
        return add_bwd_cpu(grad_outs, fws[1])
    if op == "add_bias":
        var fws = add_bias_fws_cpu(inputs)
        return add_bias_bwd_cpu(grad_outs, fws[1])
    if op == "rms_norm":
        var fws = rms_norm_fws_cpu(inputs)
        return rms_norm_bwd_cpu(grad_outs, fws[1])
    if op == "softmax":
        var fws = softmax_fws_cpu(inputs)
        return softmax_bwd_cpu(grad_outs, fws[1])
    if op == "swiglu":
        var fws = swiglu_fws_cpu(inputs)
        return swiglu_bwd_cpu(grad_outs, fws[1])
    if op == "embedding":
        var fws = embedding_fws_cpu(inputs)
        return embedding_bwd_cpu(grad_outs, fws[1])
    if op == "swiglu_ffn":
        var fws = swiglu_ffn_fws_cpu(inputs)
        return swiglu_ffn_bwd_cpu(grad_outs, fws[1])
    if op == "cross_entropy":
        var fws = cross_entropy_fws_cpu(inputs)
        return cross_entropy_bwd_cpu(grad_outs, fws[1])
    if op == "rms_norm_weight":
        var fws = rms_norm_weight_fws_cpu(inputs)
        return rms_norm_weight_bwd_cpu(grad_outs, fws[1])
    if op == "mha_seq":
        var fws = mha_seq_fws_cpu(inputs)
        return mha_seq_bwd_cpu(grad_outs, fws[1])
    if op == "rope":
        var fws = rope_fws_cpu(inputs)
        return rope_bwd_cpu(grad_outs, fws[1])
    var empty = List[AnyTensor]()
    empty.reserve(8)
    return empty^


@export
def infer_train_run_backward(
    op_ptr: Pointer[UInt8, MutUntrackedOrigin],
    inputs: Pointer[Int64, MutUntrackedOrigin],
    n_inputs: Int64,
    grads: Pointer[Int64, MutUntrackedOrigin],
    n_grads: Int64,
    out_n: Pointer[Int64, MutUntrackedOrigin],
    out_addrs: Pointer[Int64, MutUntrackedOrigin],
    out_dtype: Pointer[Int64, MutUntrackedOrigin],
    out_rank: Pointer[Int64, MutUntrackedOrigin],
    out_shape: Pointer[Int64, MutUntrackedOrigin],
    out_numel: Pointer[Int64, MutUntrackedOrigin],
    out_nbytes: Pointer[Int64, MutUntrackedOrigin],
) abi("C") -> Int32:
    """Backward of one whitelisted op; writes up to 8 gradients.

    out_addrs[i] is the raw gradient buffer (free with
    infer_train_free_buffer; zero for the no-gradient sentinel), the
    parallel arrays carry its metadata.  Returns 0 on success, -1 on any
    validation error (never aborts).
    """
    var op = _cstr_to_string(op_ptr)
    var input_list = _collect_inputs(inputs, Int(n_inputs))
    var grad_list = _collect_inputs(grads, Int(n_grads))
    if not _check_op_inputs(op, input_list):
        return -1
    if not _check_grad_shapes(op, input_list, grad_list):
        return -1
    var result = _run_backward_native(op, input_list, grad_list)
    var n = len(result)
    if n > 8:
        n = 8
    out_n.unsafe_offset(0).unsafe_store(val=Int64(n))
    for i in range(n):
        var any = result[i]
        if any.numel == 0:
            out_addrs.unsafe_offset(i).unsafe_store(val=Int64(0))
            out_dtype.unsafe_offset(i).unsafe_store(val=Int64(0))
            out_rank.unsafe_offset(i).unsafe_store(val=Int64(0))
            out_numel.unsafe_offset(i).unsafe_store(val=Int64(0))
            out_nbytes.unsafe_offset(i).unsafe_store(val=Int64(0))
            continue
        out_addrs.unsafe_offset(i).unsafe_store(val=Int64(Int(any.data)))
        out_dtype.unsafe_offset(i).unsafe_store(
            val=Int64(_code_from_dtype(any.dtype))
        )
        out_rank.unsafe_offset(i).unsafe_store(val=Int64(any.rank))
        for j in range(any.rank):
            out_shape.unsafe_offset(i * 8 + j).unsafe_store(
                val=Int64(any.shape[j])
            )
        out_numel.unsafe_offset(i).unsafe_store(val=Int64(any.numel))
        out_nbytes.unsafe_offset(i).unsafe_store(
            val=Int64(any.numel * _elem_size_of(any.dtype))
        )
    return 0


# -- M6: stateless AdamW step (for the PyTorch comparison tests) -------------


@export
def infer_train_adamw_step(
    param_ptr: Pointer[UInt8, MutUntrackedOrigin],
    grad_ptr: Pointer[UInt8, MutUntrackedOrigin],
    m_ptr: Pointer[UInt8, MutUntrackedOrigin],
    v_ptr: Pointer[UInt8, MutUntrackedOrigin],
    step_ptr: Pointer[Int64, MutUntrackedOrigin],
    lr: Float32,
    b1: Float32,
    b2: Float32,
    eps: Float32,
    wd: Float32,
) abi("C") -> Int32:
    """One AdamW update over engine-side buffers (param/grad/m/v handles).

    m/v are fp32 accumulators; param/grad share one dtype (f32 or f16).
    The step counter is read from / written back to *step_ptr.  Returns 0
    on success, -1 on validation error.
    """
    var ct_p = param_ptr.unsafe_bitcast[CTensor]()
    var ct_g = grad_ptr.unsafe_bitcast[CTensor]()
    var ct_m = m_ptr.unsafe_bitcast[CTensor]()
    var ct_v = v_ptr.unsafe_bitcast[CTensor]()
    var p = ct_p[unsafe_offset=0].any
    var g = ct_g[unsafe_offset=0].any
    var m = ct_m[unsafe_offset=0].any
    var v = ct_v[unsafe_offset=0].any
    if (
        p.dtype != g.dtype
        or p.numel != g.numel
        or m.numel != p.numel
        or v.numel != p.numel
        or m.dtype != DType.float32
        or v.dtype != DType.float32
    ):
        return -1
    var t = Int(step_ptr.unsafe_load(offset=0))
    if p.dtype == DType.float32:
        adamw_step_raw[DType.float32](p, g, m, v, t, lr, b1, b2, eps, wd)
    elif p.dtype == DType.float16:
        adamw_step_raw[DType.float16](p, g, m, v, t, lr, b1, b2, eps, wd)
    else:
        return -1
    step_ptr.unsafe_offset(0).unsafe_store(val=Int64(t))
    return 0


@export
def infer_train_free_buffer(p: Pointer[UInt8, MutUntrackedOrigin]) abi("C"):
    p.unsafe_free()


# -- M7: inference-time fine-tuning (LoRA-style output adapter) --------------
#
# `infer_train_finetune_*` adapts the model *while it serves*: a persistent
# fp32 copy of the output head (vocab x hidden) + AdamW moments.  Each step
# computes the final hidden state, the adapter logits, and one update toward
# the target token.  `target < 0` = forward-only (advance cache/state).
# After each update the fp16 head in the model is re-synced, so subsequent
# inference requests see the adapted weights immediately.


struct FTState(Movable):
    var w: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var m: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var v: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var vocab: Int
    var hidden: Int
    var t: Int
    var lr: Float32

    def __init__(
        out self,
        w: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
        m: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
        v: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
        vocab: Int,
        hidden: Int,
        t: Int,
        lr: Float32,
    ):
        self.w = w
        self.m = m
        self.v = v
        self.vocab = vocab
        self.hidden = hidden
        self.t = t
        self.lr = lr


@export
def infer_train_finetune_create(
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    lr: Float32,
) abi("C") -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    if not m:
        return None
    var model = m.value().unsafe_bitcast[Model]()
    var vocab = model[unsafe_offset=0].transformer.config.vocab
    var hidden = model[unsafe_offset=0].transformer.config.hidden
    var ft = unsafe_alloc[FTState](1)
    var w_p = unsafe_alloc[Scalar[DType.float32]](vocab * hidden)
    var m_p = unsafe_alloc[Scalar[DType.float32]](vocab * hidden)
    var v_p = unsafe_alloc[Scalar[DType.float32]](vocab * hidden)
    ft[unsafe_offset=0] = FTState(w_p, m_p, v_p, vocab, hidden, 0, lr)
    # seed the adapter from the model's current head (fp16 -> fp32);
    # M11: head_fp16() dequantizes the Q4-resident head on demand
    var head = model[unsafe_offset=0].transformer.head_fp16()
    for i in range(vocab * hidden):
        ft[unsafe_offset=0].w.unsafe_store(
            i, Scalar[DType.float32](Float32(head.get(i)))
        )
        ft[unsafe_offset=0].m.unsafe_store(i, Scalar[DType.float32](0))
        ft[unsafe_offset=0].v.unsafe_store(i, Scalar[DType.float32](0))
    return Optional(ft.unsafe_bitcast[UInt8]())


@export
def infer_train_finetune_step(
    ft_p: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    token: Int64,
    position: Int64,
    target: Int64,
    lr: Float32,
) abi("C") -> Float32:
    try:
        if not ft_p or not m:
            return Float32(0)
        var ft = ft_p.value().unsafe_bitcast[FTState]()
        var model = m.value().unsafe_bitcast[Model]()
        var vocab = ft[unsafe_offset=0].vocab
        var hidden = ft[unsafe_offset=0].hidden
        var h = model[unsafe_offset=0].transformer.forward_hidden(
            Int(token), Int(position)
        )
        if target < 0:
            return Float32(0)  # forward-only: cache/SSM state advanced
        var target_i = Int(target)
        if target_i < 0 or target_i >= vocab:
            return Float32(0)
        # logits + stable softmax over the fp32 adapter head
        var mx = Float32(-3.0e38)
        var lsum = Float32(0)
        var probs = unsafe_alloc[Scalar[DType.float32]](vocab)
        for i in range(vocab):
            var acc = Float32(0)
            for j in range(hidden):
                acc += Float32(
                    ft[unsafe_offset=0].w.unsafe_load(offset=i * hidden + j)
                ) * Float32(h.get(j))
            probs.unsafe_store(i, Scalar[DType.float32](acc))
            if acc > mx:
                mx = acc
        for i in range(vocab):
            var e = exp_f32(Float32(probs.unsafe_load(offset=i)) - mx)
            probs.unsafe_store(i, Scalar[DType.float32](e))
            lsum += e
        var inv = Float32(1.0) / lsum
        var loss = Float32(0)
        for i in range(vocab):
            var p = Float32(probs.unsafe_load(offset=i)) * inv
            if p < Float32(1e-9):
                p = Float32(1e-9)  # clamp: avoid log(0) -> inf
            probs.unsafe_store(i, Scalar[DType.float32](p))
            if i == target_i:
                loss = -log_f32(p)
        # AdamW update: dW[i, j] = (p_i - 1[i==target]) * h[j]
        ft[unsafe_offset=0].t += 1
        var b1 = Float32(0.9)
        var b2 = Float32(0.999)
        var t = Float32(ft[unsafe_offset=0].t)
        var bc1 = Float32(1.0) - pow_f32(b1, t)
        var bc2 = Float32(1.0) - pow_f32(b2, t)
        if bc1 < Float32(1e-12):
            bc1 = Float32(1e-12)
        if bc2 < Float32(1e-12):
            bc2 = Float32(1e-12)
        var eps = Float32(1e-8)
        for i in range(vocab):
            var label = Float32(1.0) if i == target_i else Float32(0)
            var base = i * hidden
            for j in range(hidden):
                var g = (
                    Float32(probs.unsafe_load(offset=i)) - label
                ) * Float32(h.get(j))
                var m_v = (
                    b1
                    * Float32(
                        ft[unsafe_offset=0].m.unsafe_load(offset=base + j)
                    )
                    + (Float32(1.0) - b1) * g
                )
                var v_v = (
                    b2
                    * Float32(
                        ft[unsafe_offset=0].v.unsafe_load(offset=base + j)
                    )
                    + (Float32(1.0) - b2) * g * g
                )
                ft[unsafe_offset=0].m.unsafe_store(
                    base + j, Scalar[DType.float32](m_v)
                )
                ft[unsafe_offset=0].v.unsafe_store(
                    base + j, Scalar[DType.float32](v_v)
                )
                var update = (m_v / bc1) / (sqrt_f32(v_v / bc2) + eps)
                ft[unsafe_offset=0].w.unsafe_store(
                    base + j,
                    Scalar[DType.float32](
                        Float32(
                            ft[unsafe_offset=0].w.unsafe_load(offset=base + j)
                        )
                        - lr * update
                    ),
                )
        # sync the fp16 head back into the model (inference sees it now);
        # M11: set_head_fp16() installs it in both load modes
        var head = model[unsafe_offset=0].transformer.head_fp16()
        for i in range(vocab * hidden):
            head.set(
                i,
                Scalar[DType.float16](
                    Float32(ft[unsafe_offset=0].w.unsafe_load(offset=i))
                ),
            )
        model[unsafe_offset=0].transformer.set_head_fp16(head)
        probs.unsafe_free()
        return loss
    except:
        return Float32(0)


@export
def infer_train_finetune_free(
    ft_p: Optional[Pointer[UInt8, MutUntrackedOrigin]],
) abi("C"):
    if not ft_p:
        return
    var ft = ft_p.value().unsafe_bitcast[FTState]()
    ft[unsafe_offset=0].w.unsafe_free()
    ft[unsafe_offset=0].m.unsafe_free()
    ft[unsafe_offset=0].v.unsafe_free()
    # shell leaks until process exit (same policy as infer_train_free_model)


@export
def infer_train_encode(
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    text: Pointer[UInt8, MutUntrackedOrigin],
    out_n: Pointer[Int64, MutUntrackedOrigin],
) abi("C") -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    if not m:
        return None
    var model = m.value().unsafe_bitcast[Model]()
    var tokens = model[unsafe_offset=0].tokenizer.encode_with_bos(
        _cstr_to_string(text)
    )
    var buf = unsafe_alloc[Scalar[DType.int32]](len(tokens))
    for i in range(len(tokens)):
        buf.unsafe_store(i, Scalar[DType.int32](tokens[i]))
    out_n.unsafe_store(val=Int64(len(tokens)))
    return Optional(buf.unsafe_bitcast[UInt8]())


@export
def infer_train_decode(
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    tokens: Pointer[Int64, MutUntrackedOrigin],
    n: Int64,
) abi("C") -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    if not m:
        return None
    var model = m.value().unsafe_bitcast[Model]()
    var list = List[Int]()
    for i in range(Int(n)):
        list.append(Int(tokens.unsafe_load(offset=i)))
    return Optional(
        _string_to_cstr(model[unsafe_offset=0].tokenizer.decode(list))
    )


def exp_f32(x: Float32) -> Float32:
    from std.math import exp

    return exp(x)


def log_f32(x: Float32) -> Float32:
    from std.math import log

    return log(x)


def pow_f32(b: Float32, e: Float32) -> Float32:
    from std.math import pow

    return pow(b, e)


def sqrt_f32(x: Float32) -> Float32:
    from std.math import sqrt

    return sqrt(x)


@export
def infer_train_forward_logits(
    m: Optional[Pointer[UInt8, MutUntrackedOrigin]],
    token: Int64,
    position: Int64,
) abi("C") -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
    """M7: one forward step's raw f32 logits buffer (vocab floats).

    Used by the HTTP server's streaming path (token-at-a-time sampling).
    Free with infer_train_free_buffer.
    """
    try:
        if not m:
            return None
        var model = m.value().unsafe_bitcast[Model]()
        var logits = model[unsafe_offset=0].transformer.forward(
            Int(token), Int(position)
        )
        var vocab = model[unsafe_offset=0].transformer.config.vocab
        var buf = unsafe_alloc[Scalar[DType.float32]](vocab)
        for i in range(vocab):
            buf.unsafe_store(i, logits.get(i))
        return Optional(buf.unsafe_bitcast[UInt8]())
    except:
        return None
