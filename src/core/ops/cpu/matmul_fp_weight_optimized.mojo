# Optimized FP16 weight-major matmul following llama.cpp's llamafile_sgemm approach
#
# Key optimizations:
# 1. Unroll K loop (8 elements at a time)
# 2. Use FP16 SIMD operations (compiler will generate fmla if beneficial)
# 3. Use multiple accumulators for better instruction-level parallelism

from src.core.tensor import tensor_zeros, Tensor
from src.core.utils import unimplemented
from std.utils import StaticTuple
from std.memory import Pointer
from std.origin import MutAnyOrigin, MutUntrackedOrigin


def matmul_weight_f16_optimized(
    x: Tensor[DType.float16, 2], w: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    """Optimized FP16 weight-major matmul with FP16 accumulation.

    Layout: w [N, K], x [M, K] -> y [M, N]
    Formula: y[i,j] = dot(x[i,:], w[j,:])

    Optimizations:
    - K loop unrolling (8 elements at a time)
    - Use FP16 accumulation (faster than FP32 cast)
    - Multiple accumulators for ILP
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("matmul_weight_f16_optimized: K mismatch")

    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    var k_main = (K // 8) * 8

    for i in range(M):
        for j in range(N):
            # Use FP16 accumulation - compiler will use fmla if beneficial
            var acc = SIMD[DType.float16, 8](0)

            var k = 0
            while k < k_main:
                # Load 8 elements from x[i, k:k+8]
                var xv = x.data().unsafe_load[width=8](offset=i * K + k)
                # Load 8 elements from w[j, k:k+8]
                var wv = w.data().unsafe_load[width=8](offset=j * K + k)
                # Multiply-add in FP16
                acc = acc + xv * wv
                k += 8

            # Horizontal sum
            var total = Float32(0)
            for lane in range(8):
                total += Float32(acc[lane])

            # Scalar tail
            while k < K:
                total += Float32(x.get(i * K + k)) * Float32(w.get(j * K + k))
                k += 1

            out.set(i * N + j, Scalar[DType.float16](total))

    return out
