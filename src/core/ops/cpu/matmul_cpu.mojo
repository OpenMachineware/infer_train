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
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc

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
