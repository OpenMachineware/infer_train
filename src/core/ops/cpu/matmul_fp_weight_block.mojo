# Block GEMM for FP16 weight-major matmul
#
# Strategy (matching llamafile_sgemm):
# 1. Compute RN output elements at once
# 2. Load x vector once, reuse for RN weight rows
# 3. Reduces memory access to x by RN×

from src.core.tensor import tensor_zeros, Tensor
from src.core.utils import unimplemented
from src.core.cpu_features import CpuFlags, detect_cpu_flags
from std.utils import StaticTuple


# Block sizes
comptime RN = 8   # number of outputs to compute at once


def matmul_weight_f16_block(
    x: Tensor[DType.float16, 2], w: Tensor[DType.float16, 2], flags: CpuFlags
) -> Tensor[DType.float16, 2]:
    """Block GEMM for FP16 weight-major matmul with CPU feature dispatch.

    Layout: w [N, K], x [M, K] -> y [M, N]

    Key optimization from llama.cpp llamafile_sgemm:
    - Load x[i, k:k+8] once
    - Update RN=8 output accumulators
    - Reduces memory access to x by 8×

    Dispatch:
    - NEON available: Block GEMM (46-62 GFLOPS on M1)
    - No NEON: Scalar fallback (slow but correct)
    """
    if not flags.has_neon():
        # Fallback to scalar for non-NEON CPUs
        return _matmul_weight_f16_scalar(x, w)

    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("matmul_weight_f16_block: K mismatch")

    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))

    var k_main = (K // 8) * 8
    var n_main = (N // RN) * RN

    for i in range(M):
        # Process RN outputs at once
        var jj = 0
        while jj < n_main:
            # Accumulators for RN outputs
            var acc0 = SIMD[DType.float16, 8](0)
            var acc1 = SIMD[DType.float16, 8](0)
            var acc2 = SIMD[DType.float16, 8](0)
            var acc3 = SIMD[DType.float16, 8](0)
            var acc4 = SIMD[DType.float16, 8](0)
            var acc5 = SIMD[DType.float16, 8](0)
            var acc6 = SIMD[DType.float16, 8](0)
            var acc7 = SIMD[DType.float16, 8](0)

            var k = 0
            while k < k_main:
                # Load x[i, k:k+8] ONCE, reuse 8 times
                var xv = x.data().unsafe_load[width=8](offset=i * K + k)

                # Load 8 weight vectors
                var wv0 = w.data().unsafe_load[width=8](offset=(jj + 0) * K + k)
                var wv1 = w.data().unsafe_load[width=8](offset=(jj + 1) * K + k)
                var wv2 = w.data().unsafe_load[width=8](offset=(jj + 2) * K + k)
                var wv3 = w.data().unsafe_load[width=8](offset=(jj + 3) * K + k)
                var wv4 = w.data().unsafe_load[width=8](offset=(jj + 4) * K + k)
                var wv5 = w.data().unsafe_load[width=8](offset=(jj + 5) * K + k)
                var wv6 = w.data().unsafe_load[width=8](offset=(jj + 6) * K + k)
                var wv7 = w.data().unsafe_load[width=8](offset=(jj + 7) * K + k)

                # Update all 8 accumulators with the same xv
                acc0 = acc0 + xv * wv0
                acc1 = acc1 + xv * wv1
                acc2 = acc2 + xv * wv2
                acc3 = acc3 + xv * wv3
                acc4 = acc4 + xv * wv4
                acc5 = acc5 + xv * wv5
                acc6 = acc6 + xv * wv6
                acc7 = acc7 + xv * wv7

                k += 8

            # Horizontal sum and store
            out.set(i * N + jj + 0, Scalar[DType.float16](_hsum_f16(acc0)))
            out.set(i * N + jj + 1, Scalar[DType.float16](_hsum_f16(acc1)))
            out.set(i * N + jj + 2, Scalar[DType.float16](_hsum_f16(acc2)))
            out.set(i * N + jj + 3, Scalar[DType.float16](_hsum_f16(acc3)))
            out.set(i * N + jj + 4, Scalar[DType.float16](_hsum_f16(acc4)))
            out.set(i * N + jj + 5, Scalar[DType.float16](_hsum_f16(acc5)))
            out.set(i * N + jj + 6, Scalar[DType.float16](_hsum_f16(acc6)))
            out.set(i * N + jj + 7, Scalar[DType.float16](_hsum_f16(acc7)))

            jj += RN

        # Tail: scalar
        while jj < N:
            var acc = Float32(0)
            for k_tail in range(K):
                acc += Float32(x.get(i * K + k_tail)) * Float32(w.get(jj * K + k_tail))
            out.set(i * N + jj, Scalar[DType.float16](acc))
            jj += 1

    return out


def _matmul_weight_f16_scalar(
    x: Tensor[DType.float16, 2], w: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    """Scalar fallback for non-NEON CPUs.

    Simple implementation without SIMD - correct but slow.
    Used as fallback when NEON is not available.
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("_matmul_weight_f16_scalar: K mismatch")

    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))

    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                acc += Float32(x.get(i * K + k)) * Float32(w.get(j * K + k))
            out.set(i * N + j, Scalar[DType.float16](acc))

    return out


@always_inline
def _hsum_f16(v: SIMD[DType.float16, 8]) -> Float32:
    """Horizontal sum of FP16 vector."""
    var total = Float32(0)
    for lane in range(8):
        total += Float32(v[lane])
    return total
