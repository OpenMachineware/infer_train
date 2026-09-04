# core/ops/cpu/matmul_cpu.mojo
#
# CPU matrix multiplication (P0, M3 fix).
#
# The kernel is written once per element dtype and adapts to ARM/x86/RISC-V
# purely through the compiler `--target-triple`; the SIMD width is a
# comptime value chosen from the element dtype so the inner accumulation
# vectorizes without any runtime dispatch.
#
# M3 note: dot products always accumulate in f32 (fp16 inputs are widened
# per element).  f16 accumulation over K ~ 1536 terms injects ~4% rounding
# noise per dot product, which flattens the logits after a few transformer
# layers (llama.cpp accumulates in f32 for the same reason).  Results are
# cast back to the element dtype on store.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ...thread_pool import parallel_run
from ..quantized.dequantize import dequantize_blocks
from ..quantized.quant_types import (
    QuantType,
    block_bytes,
    block_elems,
    group_size as quant_group_size,
)
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import sqrt
from ..fused.matmul_rms_norm import fused_matmul_rms_norm

comptime W_F16 = 8  # 8 x f16 = 128-bit NEON/SSE vector
comptime W_F32 = 4  # 4 x f32 = 128-bit NEON/SSE vector


def _matmul_kernel_f16(
    a: Tensor[DType.float16, 2], b: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b.shape()[1]
    if K != b.shape()[0]:
        unimplemented("matmul_cpu: K mismatch between a and b")
    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    var n_main = (N // W_F16) * W_F16

    for i in range(M):
        var j = 0
        while j < n_main:
            var acc = SIMD[DType.float32, W_F16](0)
            for k in range(K):
                var a_ik = Float32(a.get(i * K + k))
                var b_row = b.data().unsafe_load[width=W_F16](
                    offset=k * N + j
                ).cast[DType.float32]()
                acc = acc + a_ik * b_row
            out.data().unsafe_store(
                i * N + j, acc.cast[DType.float16]()
            )
            j += W_F16
        while j < N:
            var acc = Float32(0)
            for k in range(K):
                acc += Float32(a.get(i * K + k)) * Float32(b.get(k * N + j))
            out.set(i * N + j, Scalar[DType.float16](acc))
            j += 1
    return out


def _matmul_kernel_f32(
    a: Tensor[DType.float32, 2], b: Tensor[DType.float32, 2]
) -> Tensor[DType.float32, 2]:
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b.shape()[1]
    if K != b.shape()[0]:
        unimplemented("matmul_cpu: K mismatch between a and b")
    var out = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, N))
    var n_main = (N // W_F32) * W_F32

    for i in range(M):
        var j = 0
        while j < n_main:
            var acc = SIMD[DType.float32, W_F32](0)
            for k in range(K):
                var a_ik = Float32(a.get(i * K + k))
                var b_row = b.data().unsafe_load[width=W_F32](
                    offset=k * N + j
                )
                acc = acc + a_ik * b_row
            out.data().unsafe_store(i * N + j, acc)
            j += W_F32
        while j < N:
            var acc = Float32(0)
            for k in range(K):
                acc += Float32(a.get(i * K + k)) * Float32(b.get(k * N + j))
            out.set(i * N + j, Scalar[DType.float32](acc))
            j += 1
    return out


def _matmul_cpu_kernel[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Dispatch to the concrete per-dtype kernel.

    The views are pointer bitcasts (zero copy): Mojo 1.0's `comptime if`
    does not narrow `Tensor[dtype, 2]` to `Tensor[DType.float16, 2]` inside
    the branch, so the concrete kernel is reached through a re-typed view
    over the same storage.
    """
    comptime if dtype == DType.float16:
        var a16 = Tensor[DType.float16, 2](
            a.shape(),
            a.data().unsafe_bitcast[Scalar[DType.float16]](),
            a.device(),
        )
        var b16 = Tensor[DType.float16, 2](
            b.shape(),
            b.data().unsafe_bitcast[Scalar[DType.float16]](),
            b.device(),
        )
        var out = _matmul_kernel_f16(a16, b16)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    elif dtype == DType.float32:
        var a32 = Tensor[DType.float32, 2](
            a.shape(),
            a.data().unsafe_bitcast[Scalar[DType.float32]](),
            a.device(),
        )
        var b32 = Tensor[DType.float32, 2](
            b.shape(),
            b.data().unsafe_bitcast[Scalar[DType.float32]](),
            b.device(),
        )
        var out = _matmul_kernel_f32(a32, b32)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    else:
        unimplemented("matmul_cpu: unsupported dtype")
        return tensor_zeros[dtype, 2](StaticTuple[Int, 2](0, 0))


def matmul_cpu[dtype: DType, M: Int, N: Int, K: Int](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Comptime-shaped matmul: M/N/K are compile-time constants.

    This is the static-shape entry point used when shapes are known at
    compile time (small layers, tests).  It forwards to the runtime-shaped
    kernel below after asserting the expected shapes.
    """
    if (
        a.shape()[0] != M
        or a.shape()[1] != K
        or b.shape()[0] != K
        or b.shape()[1] != N
    ):
        unimplemented("matmul_cpu: static shape mismatch")
    return _matmul_cpu_kernel[dtype](a, b)


def matmul_cpu_dynamic[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Runtime-shaped matmul; M/N/K are read from the tensors."""
    return _matmul_cpu_kernel[dtype](a, b)


def _matmul_weight_kernel_f16(
    x: Tensor[DType.float16, 2], w: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    """y[i, j] = sum_k x[i, k] * w[j, k]; w is stored [out, in] (GGUF layout)."""
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("matmul_weight: K mismatch")
    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    var k_main = (K // W_F16) * W_F16
    for i in range(M):
        for j in range(N):
            var acc = SIMD[DType.float32, W_F16](0)
            var k = 0
            while k < k_main:
                var xv = x.data().unsafe_load[width=W_F16](
                    offset=i * K + k
                ).cast[DType.float32]()
                var wv = w.data().unsafe_load[width=W_F16](
                    offset=j * K + k
                ).cast[DType.float32]()
                acc = acc + xv * wv
                k += W_F16
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(x.get(i * K + k)) * Float32(
                    w.get(j * K + k)
                )
                k += 1
            out.set(i * N + j, Scalar[DType.float16](total))
    return out


def _matmul_weight_kernel_f32(
    x: Tensor[DType.float32, 2], w: Tensor[DType.float32, 2]
) -> Tensor[DType.float32, 2]:
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("matmul_weight: K mismatch")
    var out = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, N))
    var k_main = (K // W_F32) * W_F32
    for i in range(M):
        for j in range(N):
            var acc = SIMD[DType.float32, W_F32](0)
            var k = 0
            while k < k_main:
                var xv = x.data().unsafe_load[width=W_F32](offset=i * K + k)
                var wv = w.data().unsafe_load[width=W_F32](offset=j * K + k)
                acc = acc + xv * wv
                k += W_F32
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(x.get(i * K + k)) * Float32(
                    w.get(j * K + k)
                )
                k += 1
            out.set(i * N + j, Scalar[DType.float32](total))
    return out


def matmul_weight_cpu[dtype: DType](
    x: Tensor[dtype, 2], w: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Weight-major linear: y = W @ x where W is stored [out, in].

    This is the layout GGUF files actually carry (dims are ggml-ordered,
    innermost first), so transformer projections call this kernel directly
    with the dequantized weights - no transposes anywhere.
    """
    comptime if dtype == DType.float16:
        var x16 = Tensor[DType.float16, 2](
            x.shape(),
            x.data().unsafe_bitcast[Scalar[DType.float16]](),
            x.device(),
        )
        var w16 = Tensor[DType.float16, 2](
            w.shape(),
            w.data().unsafe_bitcast[Scalar[DType.float16]](),
            w.device(),
        )
        var out = _matmul_weight_kernel_f16(x16, w16)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    elif dtype == DType.float32:
        var x32 = Tensor[DType.float32, 2](
            x.shape(),
            x.data().unsafe_bitcast[Scalar[DType.float32]](),
            x.device(),
        )
        var w32 = Tensor[DType.float32, 2](
            w.shape(),
            w.data().unsafe_bitcast[Scalar[DType.float32]](),
            w.device(),
        )
        var out = _matmul_weight_kernel_f32(x32, w32)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    else:
        unimplemented("matmul_weight: unsupported dtype")
        return tensor_zeros[dtype, 2](StaticTuple[Int, 2](0, 0))


# -- M5: multithreaded weight-major matmul ----------------------------------
#
# The output columns are independent: column j is the dot product of every
# input row with w[j, :].  `matmul_weight_cpu_threaded` splits the N
# columns across the C thread pool (tools/thread_pool.c); each worker is
# the exported `it_mw_worker` below - a pure pointer-math kernel with a
# disjoint output range.  The context block is [x, w, out, M, K, N, dtype]
# as an Int64 array [x, w, out, M, K, N, dtype]; the worker rebuilds the
# typed pointers with `unsafe_from_address` (plain loads).


def matmul_weight_cpu_threaded[dtype: DType](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    nthreads: Int = 0,
) -> Tensor[dtype, 2]:
    """Multithreaded weight-major matmul (see the module section header).

    Falls back to the single-threaded kernel below the parallelization
    threshold or when the thread pool is unavailable.
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("matmul_weight_threaded: K mismatch")
    # threading pays off only for wide projections: the pool wake-up
    # latency otherwise eats the parallel speedup (M5 measurement)
    if N < 4096 or nthreads == 1:
        return matmul_weight_cpu[dtype](x, w)

    var out = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    comptime dtype_code = 1 if dtype == DType.float16 else 0
    # Int64-array context (element stores are the proven dylib-safe
    # pattern; struct contexts are miscompiled - see the bindings notes)
    var hdr = unsafe_alloc[Int64](8)
    hdr.unsafe_offset(0).unsafe_store(val=Int64(Int(x.data())))
    hdr.unsafe_offset(1).unsafe_store(val=Int64(Int(w.data())))
    hdr.unsafe_offset(2).unsafe_store(val=Int64(Int(out.data())))
    hdr.unsafe_offset(3).unsafe_store(val=Int64(M))
    hdr.unsafe_offset(4).unsafe_store(val=Int64(K))
    hdr.unsafe_offset(5).unsafe_store(val=Int64(N))
    hdr.unsafe_offset(6).unsafe_store(val=Int64(dtype_code))

    var raw = hdr.unsafe_bitcast[UInt8]()
    var rc = parallel_run(String("it_mw_worker"), raw, N, nthreads)
    if rc != 0:
        # pool unavailable: recompute single-threaded (out is overwritten)
        return matmul_weight_cpu[dtype](x, w)
    return out


def _matmul_multi_threaded[dtype: DType](
    x1: Tensor[dtype, 2],
    w1: Tensor[dtype, 2],
    x2: Tensor[dtype, 2],
    w2: Tensor[dtype, 2],
    x3: Tensor[dtype, 2],
    w3: Tensor[dtype, 2],
    n_pairs: Int,
    nthreads: Int,
) -> Tuple[Tensor[dtype, 2], Tensor[dtype, 2], Tensor[dtype, 2]]:
    """One pool submission for up to 3 weight-major matmuls.

    The task index space is n_pairs * N; each task picks its pair by
    idx // N.  Amortizes the pool wake-up latency over the whole layer's
    projections (QKV / gate+up) instead of paying it per matrix.
    """
    var M = x1.shape()[0]
    var K = x1.shape()[1]
    var N = w1.shape()[0]
    var out1 = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    var out2 = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    var out3 = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    var hdr = unsafe_alloc[Int64](17)
    hdr.unsafe_offset(0).unsafe_store(val=Int64(M))
    hdr.unsafe_offset(1).unsafe_store(val=Int64(K))
    hdr.unsafe_offset(2).unsafe_store(val=Int64(N))
    comptime dtype_code = 1 if dtype == DType.float16 else 0
    hdr.unsafe_offset(3).unsafe_store(val=Int64(dtype_code))
    hdr.unsafe_offset(4).unsafe_store(val=Int64(n_pairs))
    hdr.unsafe_offset(5).unsafe_store(val=Int64(Int(x1.data())))
    hdr.unsafe_offset(6).unsafe_store(val=Int64(Int(w1.data())))
    hdr.unsafe_offset(7).unsafe_store(val=Int64(Int(out1.data())))
    hdr.unsafe_offset(8).unsafe_store(val=Int64(Int(x2.data())))
    hdr.unsafe_offset(9).unsafe_store(val=Int64(Int(w2.data())))
    hdr.unsafe_offset(10).unsafe_store(val=Int64(Int(out2.data())))
    hdr.unsafe_offset(11).unsafe_store(val=Int64(Int(x3.data())))
    hdr.unsafe_offset(12).unsafe_store(val=Int64(Int(w3.data())))
    hdr.unsafe_offset(13).unsafe_store(val=Int64(Int(out3.data())))

    print(
        "[multi] ctx: x=", Int(x1.data()), " w=", Int(w1.data()),
        " out1=", Int(out1.data()), " out2=", Int(out2.data()),
        " out3=", Int(out3.data()),
    )
    var raw = hdr.unsafe_bitcast[UInt8]()
    var rc = parallel_run(
        String("it_mw_multi_worker"), raw, n_pairs * N, nthreads
    )
    if rc != 0:
        out1 = matmul_weight_cpu[dtype](x1, w1)
        if n_pairs >= 2:
            out2 = matmul_weight_cpu[dtype](x2, w2)
        if n_pairs >= 3:
            out3 = matmul_weight_cpu[dtype](x3, w3)
    return (out1, out2, out3)


def matmul_weight_3_threaded[dtype: DType](
    x: Tensor[dtype, 2],
    w1: Tensor[dtype, 2],
    w2: Tensor[dtype, 2],
    w3: Tensor[dtype, 2],
    mut o1: Tensor[dtype, 2],
    mut o2: Tensor[dtype, 2],
    mut o3: Tensor[dtype, 2],
    nthreads: Int = 0,
):
    """QKV-style: three projections of the same input in one submission.

    Outputs are written into the caller's preallocated tensors (tuple
    returns across dylib function boundaries are unreliable in Mojo 1.0
    shared libraries).  Each pair may have its own output width (Q is
    hidden wide; K/V are kv-hidden wide under GQA).
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var n1 = w1.shape()[0]
    var n2 = w2.shape()[0]
    var n3 = w3.shape()[0]
    if o1.shape()[0] != M or o1.shape()[1] != n1:
        unimplemented("matmul_weight_3_threaded: bad out1 shape")
    if o2.shape()[0] != M or o2.shape()[1] != n2:
        unimplemented("matmul_weight_3_threaded: bad out2 shape")
    if o3.shape()[0] != M or o3.shape()[1] != n3:
        unimplemented("matmul_weight_3_threaded: bad out3 shape")
    if n1 + n2 + n3 < 4096 or nthreads == 1:
        o1 = matmul_weight_cpu[dtype](x, w1)
        o2 = matmul_weight_cpu[dtype](x, w2)
        o3 = matmul_weight_cpu[dtype](x, w3)
        return
    var hdr = unsafe_alloc[Int64](16)
    hdr.unsafe_offset(0).unsafe_store(val=Int64(M))
    hdr.unsafe_offset(1).unsafe_store(val=Int64(K))
    comptime dtype_code = 1 if dtype == DType.float16 else 0
    hdr.unsafe_offset(2).unsafe_store(val=Int64(dtype_code))
    hdr.unsafe_offset(3).unsafe_store(val=Int64(3))
    hdr.unsafe_offset(4).unsafe_store(val=Int64(n1))
    hdr.unsafe_offset(5).unsafe_store(val=Int64(n2))
    hdr.unsafe_offset(6).unsafe_store(val=Int64(n3))
    hdr.unsafe_offset(7).unsafe_store(val=Int64(Int(x.data())))
    hdr.unsafe_offset(8).unsafe_store(val=Int64(Int(w1.data())))
    hdr.unsafe_offset(9).unsafe_store(val=Int64(Int(o1.data())))
    hdr.unsafe_offset(10).unsafe_store(val=Int64(Int(x.data())))
    hdr.unsafe_offset(11).unsafe_store(val=Int64(Int(w2.data())))
    hdr.unsafe_offset(12).unsafe_store(val=Int64(Int(o2.data())))
    hdr.unsafe_offset(13).unsafe_store(val=Int64(Int(x.data())))
    hdr.unsafe_offset(14).unsafe_store(val=Int64(Int(w3.data())))
    hdr.unsafe_offset(15).unsafe_store(val=Int64(Int(o3.data())))

    var raw = hdr.unsafe_bitcast[UInt8]()
    var rc = parallel_run(
        String("it_mw_multi_worker"), raw, n1 + n2 + n3, nthreads
    )
    if rc != 0:
        o1 = matmul_weight_cpu[dtype](x, w1)
        o2 = matmul_weight_cpu[dtype](x, w2)
        o3 = matmul_weight_cpu[dtype](x, w3)


def matmul_weight_2_threaded[dtype: DType](
    x: Tensor[dtype, 2],
    w1: Tensor[dtype, 2],
    w2: Tensor[dtype, 2],
    mut o1: Tensor[dtype, 2],
    mut o2: Tensor[dtype, 2],
    nthreads: Int = 0,
):
    """gate+up style: two projections of the same input in one submission."""
    var M = x.shape()[0]
    var K = x.shape()[1]
    var n1 = w1.shape()[0]
    var n2 = w2.shape()[0]
    if o1.shape()[0] != M or o1.shape()[1] != n1:
        unimplemented("matmul_weight_2_threaded: bad out1 shape")
    if o2.shape()[0] != M or o2.shape()[1] != n2:
        unimplemented("matmul_weight_2_threaded: bad out2 shape")
    if n1 + n2 < 4096 or nthreads == 1:
        o1 = matmul_weight_cpu[dtype](x, w1)
        o2 = matmul_weight_cpu[dtype](x, w2)
        return
    var hdr = unsafe_alloc[Int64](16)
    hdr.unsafe_offset(0).unsafe_store(val=Int64(M))
    hdr.unsafe_offset(1).unsafe_store(val=Int64(K))
    comptime dtype_code = 1 if dtype == DType.float16 else 0
    hdr.unsafe_offset(2).unsafe_store(val=Int64(dtype_code))
    hdr.unsafe_offset(3).unsafe_store(val=Int64(2))
    hdr.unsafe_offset(4).unsafe_store(val=Int64(n1))
    hdr.unsafe_offset(5).unsafe_store(val=Int64(n2))
    hdr.unsafe_offset(6).unsafe_store(val=Int64(0))
    hdr.unsafe_offset(7).unsafe_store(val=Int64(Int(x.data())))
    hdr.unsafe_offset(8).unsafe_store(val=Int64(Int(w1.data())))
    hdr.unsafe_offset(9).unsafe_store(val=Int64(Int(o1.data())))
    hdr.unsafe_offset(10).unsafe_store(val=Int64(Int(x.data())))
    hdr.unsafe_offset(11).unsafe_store(val=Int64(Int(w2.data())))
    hdr.unsafe_offset(12).unsafe_store(val=Int64(Int(o2.data())))

    var raw = hdr.unsafe_bitcast[UInt8]()
    var rc = parallel_run(
        String("it_mw_multi_worker"), raw, n1 + n2, nthreads
    )
    if rc != 0:
        o1 = matmul_weight_cpu[dtype](x, w1)
        o2 = matmul_weight_cpu[dtype](x, w2)


def matmul_cpu_forward_with_saved[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    """Forward pass that also returns the inputs for the backward pass.

    The saved tensors are *views* (they share the input buffers) so no large
    intermediate payload is duplicated.
    """
    var out = matmul_cpu_dynamic[dtype](a, b)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(a)
    saved.append(b)
    return (out, saved^)


def matmul_cpu_backward[dtype: DType](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    """Backward for `matmul_cpu_dynamic`: out = a @ b (a[M,K], b[K,N]).

    grad_a[i,k] = sum_j grad_out[i,j] * b[k,j]
    grad_b[k,j] = sum_i a[i,k] * grad_out[i,j]

    `saved` = [a, b] (views captured by `matmul_cpu_forward_with_saved`).
    """
    var a = saved[0]
    var b = saved[1]
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b.shape()[1]
    var grad_a = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, K))
    var grad_b = tensor_zeros[dtype, 2](StaticTuple[Int, 2](K, N))

    comptime W = 8 if dtype == DType.float16 else 4
    var n_main = (N // W) * W

    # grad_a[i, k] = dot(grad_out[i, :], b[k, :]) - contiguous over N
    for i in range(M):
        for k in range(K):
            var acc = SIMD[DType.float32, W](0)
            var j = 0
            while j < n_main:
                var gv = grad_out.data().unsafe_load[width=W](
                    offset=i * N + j
                ).cast[DType.float32]()
                var bv = b.data().unsafe_load[width=W](
                    offset=k * N + j
                ).cast[DType.float32]()
                acc = acc + gv * bv
                j += W
            var total = Float32(acc.reduce_add())
            while j < N:
                total += Float32(grad_out.get(i * N + j)) * Float32(
                    b.get(k * N + j)
                )
                j += 1
            grad_a.set(i * K + k, Scalar[dtype](total))

    # grad_b[k, j] = sum_i a[i, k] * grad_out[i, j] (strided over i)
    for k in range(K):
        for j in range(N):
            var acc = Float32(0)
            for i in range(M):
                acc += Float32(a.get(i * K + k)) * Float32(
                    grad_out.get(i * N + j)
                )
            grad_b.set(k * N + j, Scalar[dtype](acc))

    var result = List[Tensor[dtype, 2]]()
    result.append(grad_a)
    result.append(grad_b)
    return result^


def matmul_weight_cpu_forward_with_saved[dtype: DType](
    x: Tensor[dtype, 2], w: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    """Forward for the weight-major linear (y = W @ x, W stored [out, in])
    plus the inputs needed by the backward pass."""
    var out = matmul_weight_cpu[dtype](x, w)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(x)
    saved.append(w)
    return (out, saved^)


def matmul_weight_cpu_backward[dtype: DType](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    """Backward for `matmul_weight_cpu`: y = x @ w^T (x[M,K], w[N,K]).

    grad_x[i,k] = sum_j grad_out[i,j] * w[j,k]   (grad_out @ w)
    grad_w[j,k] = sum_i grad_out[i,j] * x[i,k]   (grad_out^T @ x)

    `saved` = [x, w].
    """
    var x = saved[0]
    var w = saved[1]
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    var grad_x = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, K))
    var grad_w = tensor_zeros[dtype, 2](StaticTuple[Int, 2](N, K))

    comptime W = 8 if dtype == DType.float16 else 4
    var k_main = (K // W) * W

    # grad_x[i, :] = sum_j grad_out[i, j] * w[j, :]
    for i in range(M):
        for j in range(N):
            var g = Float32(grad_out.get(i * N + j))
            var k = 0
            while k < k_main:
                var xv = grad_x.data().unsafe_load[width=W](
                    offset=i * K + k
                ).cast[DType.float32]()
                var wv = w.data().unsafe_load[width=W](
                    offset=j * K + k
                ).cast[DType.float32]()
                grad_x.data().unsafe_store(
                    i * K + k,
                    (xv + SIMD[DType.float32, W](g) * wv).cast[dtype](),
                )
                k += W
            while k < K:
                grad_x.set(
                    i * K + k,
                    Scalar[dtype](
                        Float32(grad_x.get(i * K + k))
                        + g * Float32(w.get(j * K + k))
                    ),
                )
                k += 1

    # grad_w[j, :] = sum_i grad_out[i, j] * x[i, :]
    for i in range(M):
        for j in range(N):
            var g = Float32(grad_out.get(i * N + j))
            var k = 0
            while k < k_main:
                var wv = grad_w.data().unsafe_load[width=W](
                    offset=j * K + k
                ).cast[DType.float32]()
                var xv = x.data().unsafe_load[width=W](
                    offset=i * K + k
                ).cast[DType.float32]()
                grad_w.data().unsafe_store(
                    j * K + k,
                    (wv + SIMD[DType.float32, W](g) * xv).cast[dtype](),
                )
                k += W
            while k < K:
                grad_w.set(
                    j * K + k,
                    Scalar[dtype](
                        Float32(grad_w.get(j * K + k))
                        + g * Float32(x.get(i * K + k))
                    ),
                )
                k += 1

    var result = List[Tensor[dtype, 2]]()
    result.append(grad_x)
    result.append(grad_w)
    return result^


# -- M7: generic quantized matmul (comptime-specialized) ---------------------
#
# `matmul_quantized_cpu` is the comptime-parameterized entry point for
# GGUF block-format weights.  `quant_type` selects the dequantizer at
# compile time (the `comptime if` inside `dequantize_blocks` keeps only
# the chosen branch, so each instantiation is a dedicated kernel).
#
# Q4-resident (the M11 change): the weight is NEVER dequantized into a
# dense tensor.  The kernel walks the K dimension in block units and, for
# every block, calls `dequantize_blocks` to decode exactly that one block
# into a small scratch buffer, folds the block into the SIMD accumulator,
# and discards it before the next block - the dequantized data does not
# leave the kernel scope.  Peak extra memory is one block (256 x f16 =
# 512 B for the K formats), so a quantized weight's resident footprint
# stays its on-disk (Q4) size instead of doubling into fp16.
#
# Numerics: the per-block decode is the same bit-exact `ggml-quants.c`
# kernels used by the whole-tensor path (validated by
# `tests/test_dequant_m7.mojo`), and the SIMD accumulation order matches
# `matmul_weight_cpu` (f32 accumulation, see the module header), so for
# M == 1 the result is bit-identical to dequantize-then-matmul.  For
# M > 1 each input row runs the same single-row kernel (the block is
# re-decoded per row); the engine's decode path is single-token, so this
# is the common case.
#
# For the GGUF block formats the scales (and mins) are packed inside every
# quantized block - that is the llama.cpp `ggml-quants.c` layout - so the
# `scale` / `zero_point` arguments are ignored for the current formats;
# they are part of the generic signature for future formats that keep
# their scales in a separate tensor.
#
# `group_size` is the sub-block (scale group) size; pass 0 to accept the
# format's built-in layout, or the canonical size from
# `quant_group_size(quant_type)` to assert it.


def _matmul_quantized_row_kernel[
    dtype: DType,
    quant_type: QuantType,
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    b: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin],
    N: Int,
    nb: Int,
):
    """Fused per-block dequant + dot product for one input row.

    `x` is the row [K]; `b` is the quantized weight [N, nb blocks]; `dst`
    receives the N dot products.  For each output row j the K dimension is
    walked block by block: one block is dequantized into the scratch
    buffer, folded into the f32 SIMD accumulator, and discarded.  The
    scratch (one block, e.g. 256 x f16 = 512 B) is the kernel's only
    extra allocation - the dequantized values never leave this scope.
    """
    comptime be = block_elems(quant_type)
    comptime bb = block_bytes(quant_type)
    comptime W = 8 if dtype == DType.float16 else 4
    comptime be_chunks = be // W
    var scratch = unsafe_alloc[Scalar[dtype]](be)
    for j in range(N):
        var acc = SIMD[DType.float32, W](0)
        var row = b.unsafe_offset(j * nb * bb)
        var k = 0
        for blk in range(nb):
            dequantize_blocks[dtype, quant_type](row, blk * bb, scratch, 1)
            var l = 0
            while l < be_chunks:
                var xv = x.unsafe_load[width=W](
                    offset=k + l * W
                ).cast[DType.float32]()
                var wv = scratch.unsafe_load[width=W](
                    offset=l * W
                ).cast[DType.float32]()
                acc = acc + xv * wv
                l += 1
            k += be
        dst.unsafe_store(j, acc.reduce_add().cast[dtype]())
    scratch.unsafe_free()


def matmul_quantized_cpu[
    dtype: DType,
    quant_type: QuantType,
    group_size: Int,
](
    a: Tensor[dtype, 2],
    b_quant: Tensor[DType.uint8, 2],
    scale: Tensor[dtype, 1],
    zero_point: Optional[Tensor[dtype, 1]] = None,
) -> Tensor[dtype, 2]:
    """Quantized weight-major matmul: `y[i, j] = sum_k a[i, k] * b[j, k]`.

    `a` is the activation [M, K] (FP16/FP32); `b_quant` is the quantized
    weight [N, *] in the raw block layout of `quant_type` (row-major
    super-blocks, N rows of K elements each); the result is [M, N] in
    `dtype`.  The dequantization is bit-exact with llama.cpp's
    `ggml-quants.c` and happens per block INSIDE the matmul kernel (see
    the module section header): no dense dequantized copy of the weight
    is ever materialized.
    """
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b_quant.shape()[0]
    var be = block_elems(quant_type)
    var bb = block_bytes(quant_type)
    if be == 0 or K % be != 0:
        unimplemented("matmul_quantized_cpu: K not a multiple of block size")
    comptime W = 8 if dtype == DType.float16 else 4
    if K % W != 0:
        unimplemented("matmul_quantized_cpu: K not a multiple of SIMD width")
    if b_quant.numel() != N * (K // be) * bb:
        unimplemented("matmul_quantized_cpu: b_quant byte size mismatch")
    if group_size != 0 and group_size != quant_group_size(quant_type):
        unimplemented("matmul_quantized_cpu: group_size mismatch for format")
    _ = scale
    _ = zero_point

    var out = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    var a_ptr = a.data()
    var b_ptr = b_quant.data()
    var out_ptr = out.data()
    var nb = K // be
    for i in range(M):
        _matmul_quantized_row_kernel[dtype, quant_type](
            a_ptr.unsafe_offset(i * K),
            b_ptr,
            out_ptr.unsafe_offset(i * N),
            N,
            nb,
        )
    return out


# -- M8: JIT shape specialization for the fused matmul + RMSNorm op ----------
#
# `fused_matmul_rms_norm` (ops/fused) is the runtime-shaped generic kernel.
# For the hot model shapes we additionally compile
# `_fused_matmul_rms_norm_specialized[M, N, K, W, UNROLL, TILE]`:
#
#   * M/N/K fix every loop bound at compile time (the comptime-JIT model
#     of M5: "compiling" a new shape is instantiating this kernel);
#   * W is the SIMD lane width chosen by the SIMD autotuner (task 1);
#   * UNROLL widens the effective k-loop vector to W*UNROLL lanes;
#   * TILE > 0 blocks the k loop into TILE-element tiles (a register
#     pressure knob); TILE = 0 keeps the single-accumulator structure.
#
# The runtime entry is `JitCache.run_fused_jit` (jit/jit_cache.mojo): it
# builds the shape signature from the runtime (M, N, K), asks the cache
# for the compiled kernel (`get_or_compile`), and runs it.  Shapes without
# a compiled specialization fall back to the generic kernel and are
# recorded (Mojo 1.0 has no runtime codegen).  The entry lives in
# jit_cache.mojo because Mojo 1.0 rejects circular imports and jit_cache
# already depends on this module.


def _fused_matmul_rms_norm_specialized[
    M: Int, N: Int, K: Int, W: Int, UNROLL: Int = 1, TILE: Int = 0
](
    x: Tensor[DType.float16, 2],
    w: Tensor[DType.float16, 2],
    eps: Float32,
) -> Tensor[DType.float16, 2]:
    """y = rms_norm(x @ w^T, eps) with comptime M/N/K/W/UNROLL/TILE.

    f16 inputs, f32 accumulation (the M3 numerics rule), result cast back
    to f16 on store - identical numerics to the generic fused kernel,
    with the loop structure fixed at compile time.
    """
    comptime VW = W * UNROLL
    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    comptime k_vec = (K // VW) * VW
    comptime n_vec = (N // W) * W
    for i in range(M):
        # pass 1: y[i, :] = x[i, :] @ w^T (dot products, f32 accumulation)
        for j in range(N):
            comptime if TILE > 0:
                # k loop blocked into TILE-element tiles (comptime bound)
                comptime k_tile = (TILE // VW) * VW if TILE >= VW else VW
                var total = Float32(0)
                var k = 0
                while k < k_vec:
                    var acc = SIMD[DType.float32, VW](0)
                    var k_end = k + k_tile
                    if k_end > k_vec:
                        k_end = k_vec
                    while k < k_end:
                        var xv = x.data().unsafe_load[width=VW](
                            offset=i * K + k
                        ).cast[DType.float32]()
                        var wv = w.data().unsafe_load[width=VW](
                            offset=j * K + k
                        ).cast[DType.float32]()
                        acc = acc + xv * wv
                        k += VW
                    total += Float32(acc.reduce_add())
                while k < K:
                    total += Float32(x.get(i * K + k)) * Float32(
                        w.get(j * K + k)
                    )
                    k += 1
                out.set(i * N + j, Scalar[DType.float16](total))
            else:
                var acc = SIMD[DType.float32, VW](0)
                var k = 0
                while k < k_vec:
                    var xv = x.data().unsafe_load[width=VW](
                        offset=i * K + k
                    ).cast[DType.float32]()
                    var wv = w.data().unsafe_load[width=VW](
                        offset=j * K + k
                    ).cast[DType.float32]()
                    acc = acc + xv * wv
                    k += VW
                var total = Float32(acc.reduce_add())
                while k < K:
                    total += Float32(x.get(i * K + k)) * Float32(
                        w.get(j * K + k)
                    )
                    k += 1
                out.set(i * N + j, Scalar[DType.float16](total))
        # pass 2: RMSNorm the row in place (SIMD width W)
        var ss = Float32(0)
        var j = 0
        while j < n_vec:
            var v = out.data().unsafe_load[width=W](
                offset=i * N + j
            ).cast[DType.float32]()
            ss += Float32((v * v).reduce_add())
            j += W
        while j < N:
            var v = Float32(out.get(i * N + j))
            ss += v * v
            j += 1
        var inv = Float32(1.0) / sqrt(ss / Float32(N) + eps)
        j = 0
        while j < n_vec:
            var v = out.data().unsafe_load[width=W](
                offset=i * N + j
            ).cast[DType.float32]()
            out.data().unsafe_store(
                i * N + j, (v * SIMD[DType.float32, W](inv)).cast[DType.float16]()
            )
            j += W
        while j < N:
            var v = Float32(out.get(i * N + j))
            out.set(i * N + j, Scalar[DType.float16](v * inv))
    return out


def fused_matmul_rms_norm_key(m: Int, n: Int, k: Int) -> String:
    """The shape signature of a fused matmul+rmsnorm (M, N, K)."""
    return (
        "fused/" + String(m) + String("/") + String(n) + String("/") + String(k)
    )


struct CompiledFusedKernel(Copyable):
    """A compiled fused kernel: the executable result of one shape
    specialization.

    `run` dispatches to the comptime-specialized instantiation for
    (m, n, k) at the SIMD bit width the kernel was compiled with; a shape
    without a compiled specialization runs the generic kernel (the M5
    comptime-JIT fallback).
    """
    var key: String
    var m: Int
    var n: Int
    var k: Int
    var width: Int  # SIMD bit width (64/128/256)

    def __init__(out self, key: String, m: Int, n: Int, k: Int, width: Int):
        self.key = key
        self.m = m
        self.n = n
        self.k = k
        self.width = width

    def __init__(out self, *, copy: CompiledFusedKernel):
        self.key = copy.key
        self.m = copy.m
        self.n = copy.n
        self.k = copy.k
        self.width = copy.width

    def run(
        self,
        x: Tensor[DType.float16, 2],
        w: Tensor[DType.float16, 2],
        eps: Float32,
    ) -> Tensor[DType.float16, 2]:
        # The specialized shape table (comptime instantiations).  The two
        # 1.5B-model FFN shapes (hidden 1536, ffn 8960) plus a small test
        # shape that exercises the UNROLL/TILE knobs.
        if self.m == 1 and self.n == 8960 and self.k == 1536:
            if self.width == 256:
                return _fused_matmul_rms_norm_specialized[1, 8960, 1536, 16](
                    x, w, eps
                )
            elif self.width == 64:
                return _fused_matmul_rms_norm_specialized[1, 8960, 1536, 4](
                    x, w, eps
                )
            return _fused_matmul_rms_norm_specialized[1, 8960, 1536, 8](
                x, w, eps
            )
        if self.m == 1 and self.n == 1536 and self.k == 8960:
            if self.width == 256:
                return _fused_matmul_rms_norm_specialized[1, 1536, 8960, 16](
                    x, w, eps
                )
            elif self.width == 64:
                return _fused_matmul_rms_norm_specialized[1, 1536, 8960, 4](
                    x, w, eps
                )
            return _fused_matmul_rms_norm_specialized[1, 1536, 8960, 8](
                x, w, eps
            )
        if self.m == 2 and self.n == 256 and self.k == 128:
            if self.width == 256:
                return _fused_matmul_rms_norm_specialized[2, 256, 128, 16, 2, 64](
                    x, w, eps
                )
            elif self.width == 64:
                return _fused_matmul_rms_norm_specialized[2, 256, 128, 4, 2, 32](
                    x, w, eps
                )
            return _fused_matmul_rms_norm_specialized[2, 256, 128, 8, 2, 64](
                x, w, eps
            )
        # shape without a compiled specialization: the generic kernel
        return fused_matmul_rms_norm[DType.float16](x, w, eps)


def compile_fused_kernel(m: Int, n: Int, k: Int, width: Int) -> CompiledFusedKernel:
    """The compile function for fused matmul+rmsnorm specializations.

    Passed to `JitCache.get_or_compile` as a thin function value - the
    Mojo 1.0 form of the task's `compile_fn: Function` parameter (a
    top-level function; thin functions cannot capture state, so the shape
    is passed explicitly).
    """
    return CompiledFusedKernel(
        fused_matmul_rms_norm_key(m, n, k), m, n, k, width
    )
