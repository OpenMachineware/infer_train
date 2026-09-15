# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/simd_base.mojo
#
# Platform-agnostic SIMD interface for quantized operations.
#
# The kernels are specialized at compile time based on the target architecture:
# - ARM64 (Apple Silicon, ARMv8): NEON intrinsics via Mojo SIMD
# - x86_64: AVX2/AVX-512 path (placeholder, not yet implemented)
# - Other: scalar fallback
#
# This mirrors llama.cpp's platform abstraction in ggml-cpu.c, where
# #if defined(__ARM_NEON) and #if defined(__AVX2__) select the kernel.

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.utils.static_tuple import StaticTuple
from std.sys import CompilationTarget
from ....quantized.quant_types import QuantType, block_elems, block_bytes

# Platform detection
comptime TARGET_HAS_NEON = CompilationTarget.has_neon()
comptime TARGET_HAS_AVX2 = CompilationTarget.has_avx2()
comptime TARGET_IS_X86 = CompilationTarget.is_x86()


def vec_dot_q4_k[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """Dot product: sum(x[i] * dequant_q4_k[i]) for one weight row.

    Platform-specific implementation selected at compile time:
    - NEON-capable: NEON-optimized kernel (simd_neon.mojo)
    - AVX2-capable: AVX2 kernel (simd_avx.mojo, placeholder)
    - Other: scalar fallback
    """
    comptime if TARGET_HAS_NEON:
        return vec_dot_q4_k_neon[dtype](x, block, nb)
    elif TARGET_HAS_AVX2:
        return vec_dot_q4_k_avx[dtype](x, block, nb)
    else:
        return _vec_dot_q4_k_scalar[dtype](x, block, nb)


def vec_dot_q4_0[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """Dot product for Q4_0 quantization."""
    comptime if TARGET_HAS_NEON:
        return vec_dot_q4_0_neon[dtype](x, block, nb)
    elif TARGET_HAS_AVX2:
        return vec_dot_q4_0_avx[dtype](x, block, nb)
    else:
        return _vec_dot_q4_0_scalar[dtype](x, block, nb)


def vec_dot_q8_0[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """Dot product for Q8_0 quantization."""
    comptime if TARGET_HAS_NEON:
        return vec_dot_q8_0_neon[dtype](x, block, nb)
    elif TARGET_HAS_AVX2:
        return vec_dot_q8_0_avx[dtype](x, block, nb)
    else:
        return _vec_dot_q8_0_scalar[dtype](x, block, nb)


# -- Scalar fallback implementations -----------------------------------------


def _vec_dot_q4_k_scalar[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """Scalar fallback for Q4_K dot product."""
    # Simple scalar implementation without scratch buffer
    # This is a placeholder - actual implementation would need proper dequantization
    var acc = Float32(0)
    for i in range(nb * 256):
        var xv = Float32(x.unsafe_load(offset=i))
        # Placeholder: just use the byte values directly
        var b = Int(block.unsafe_load[width=1](offset=i % (nb * 144)))
        acc += xv * Float32(b % 16)
    return acc


def _vec_dot_q4_0_scalar[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """Scalar fallback for Q4_0 dot product."""
    var acc = Float32(0)
    for i in range(nb * 32):
        var xv = Float32(x.unsafe_load(offset=i))
        var b = Int(block.unsafe_load[width=1](offset=i % (nb * 18)))
        acc += xv * Float32(b % 16 - 8)
    return acc


def _vec_dot_q8_0_scalar[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """Scalar fallback for Q8_0 dot product."""
    var acc = Float32(0)
    for i in range(nb * 32):
        var xv = Float32(x.unsafe_load(offset=i))
        var b = Int(block.unsafe_load[width=1](offset=i % (nb * 34)))
        acc += xv * Float32(b - 128)
    return acc


# -- NEON implementations (imported from simd_neon.mojo) --------------------


from .simd_neon import vec_dot_q4_k_neon, vec_dot_q4_0_neon, vec_dot_q8_0_neon


# -- AVX implementations (imported from simd_avx.mojo) ----------------------


from .simd_avx import vec_dot_q4_k_avx, vec_dot_q4_0_avx, vec_dot_q8_0_avx
