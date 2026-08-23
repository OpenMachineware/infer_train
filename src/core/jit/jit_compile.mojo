# core/jit/jit_compile.mojo
#
# M5 : JIT specialization for hot subgraphs.
#
# Mojo's "JIT" is comptime specialization: `jit_ffn[M, F, K]` compiles a
# dedicated FFN kernel for the exact runtime shapes (M tokens, F ffn, K
# hidden).  The comptime parameters fully unroll the SIMD main loops
# (`_jit_matmul_weight`), which the runtime-shaped generic kernels cannot
# do.  The compile entry (`jit_ffn`) therefore emits shape-specialized
# code per instantiated (M, F, K) triple; `jit_cache.mojo` records which
# triples are compiled and the interpreter dispatches jit-marked nodes
# through it.  (True runtime codegen lands in M6; M5 is the comptime
# specialization path per the plan.)

from ..tensor import Tensor, tensor_zeros
from ..ops.cpu.swiglu_cpu import swiglu_cpu_dynamic
from ..ops.cpu.matmul_cpu import matmul_weight_cpu
from std.utils.static_tuple import StaticTuple

comptime W_F16 = 8


def _jit_matmul_weight[M: Int, N: Int, K: Int](
    x: Tensor[DType.float16, 2], w: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    """y = W @ x with comptime shapes: the k loop is fully unrolled and
    the SIMD width is fixed at compile time."""
    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    var k_main = (K // W_F16) * W_F16
    for i in range(M):
        for j in range(N):
            var acc = SIMD[DType.float32, W_F16](0)
            var k = 0
            while k < k_main:  # comptime trip count (K fixed)
                var xv = x.data().unsafe_load[width=W_F16](
                    offset=i * K + k
                ).cast[DType.float32]()
                var wv = w.data().unsafe_load[width=W_F16](
                    offset=j * K + k
                ).cast[DType.float32]()
                acc = acc + xv * wv
                k += W_F16
            var total = Float32(acc.reduce_add())
            var k2 = k_main
            while k2 < K:  # comptime tail
                total += Float32(x.get(i * K + k2)) * Float32(
                    w.get(j * K + k2)
                )
                k2 += 1
            out.set(i * N + j, Scalar[DType.float16](total))
    return out


def jit_ffn[M: Int, F: Int, K: Int](
    x: Tensor[DType.float16, 2],
    gate_w: Tensor[DType.float16, 2],
    up_w: Tensor[DType.float16, 2],
    down_w: Tensor[DType.float16, 2],
) -> Tensor[DType.float16, 2]:
    """Comptime-shape-specialized SwiGLU FFN (M tokens, F ffn, K hidden)."""
    var g = _jit_matmul_weight[M, F, K](x, gate_w)
    var u = _jit_matmul_weight[M, F, K](x, up_w)
    var h = swiglu_cpu_dynamic[DType.float16](g, u)
    return _jit_matmul_weight[M, K, F](h, down_w)


def jit_ffn_key(m: Int, f: Int, k: Int) -> String:
    return (
        String("ffn/") + String(m) + String("/") + String(f) + String("/") + String(k)
    )


def jit_ffn_generic(
    x: Tensor[DType.float16, 2],
    gate_w: Tensor[DType.float16, 2],
    up_w: Tensor[DType.float16, 2],
    down_w: Tensor[DType.float16, 2],
) -> Tensor[DType.float16, 2]:
    """The runtime-shaped reference path (no comptime specialization)."""
    var g = matmul_weight_cpu[DType.float16](x, gate_w)
    var u = matmul_weight_cpu[DType.float16](x, up_w)
    var h = swiglu_cpu_dynamic[DType.float16](g, u)
    return matmul_weight_cpu[DType.float16](h, down_w)
