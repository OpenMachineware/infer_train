# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/simd_base.mojo
#
# SIMD kernel dispatch for quantized operations.
#
# Kernels receive CpuFlags as parameter - zero runtime overhead after
# initial detection. Call pattern:
#   vec_dot_q4_k_q8_k(w, x, model.cpu_flags)

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import CompilationTarget
from ....cpu_features import CpuFlags, FEATURE_NEON, FEATURE_MMLA, FEATURE_AVX2
from ...quantized.quant_types import QuantType, block_elems, block_bytes

# Compile-time platform detection (for fallback)
comptime TARGET_HAS_NEON = CompilationTarget.has_neon()
comptime TARGET_HAS_AVX2 = CompilationTarget.has_avx2()


# ============================================================================
# Qn_K × Q8_K vec_dot kernels
# ============================================================================


def vec_dot_qk_q8k(
    quant_type: QuantType,
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
    flags: CpuFlags,
) -> Float32:
    """Unified vec_dot dispatch for all Qn_K formats.

    Simpler than having if-else chains in every matmul function.
    """
    if quant_type == QuantType.Q4_K_M:
        return vec_dot_q4_k_q8_k(w_block, q8_data, flags)
    elif quant_type == QuantType.Q5_K:
        return vec_dot_q5_k_q8_k(w_block, q8_data, flags)
    elif quant_type == QuantType.Q6_K:
        return vec_dot_q6_k_q8_k(w_block, q8_data, flags)
    elif quant_type == QuantType.Q2_K:
        return vec_dot_q2_k_q8_k(w_block, q8_data, flags)
    elif quant_type == QuantType.Q3_K:
        return vec_dot_q3_k_q8_k(w_block, q8_data, flags)
    else:
        return Float32(0)


def vec_dot_q4_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
    flags: CpuFlags,
) -> Float32:
    """Q4_K × Q8_K dot product with CPU dispatch."""
    # Note: MMLA (i8mm) is only available on M2+, not on M1.
    # Mojo doesn't have compile-time MMLA detection, so we use NEON/SDOT.
    if flags.has_neon():
        return vec_dot_q4_k_q8_k_neon(w_block, q8_data)
    elif flags.has_avx2():
        return _vec_dot_q4_k_q8_k_avx(w_block, q8_data)
    else:
        return _vec_dot_q4_k_q8_k_scalar(w_block, q8_data)


def vec_dot_q5_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
    flags: CpuFlags,
) -> Float32:
    """Q5_K × Q8_K dot product with CPU dispatch."""
    if flags.has_neon():
        return vec_dot_q5_k_q8_k_neon(w_block, q8_data)
    elif flags.has_avx2():
        return _vec_dot_q5_k_q8_k_avx(w_block, q8_data)
    else:
        return _vec_dot_q5_k_q8_k_scalar(w_block, q8_data)


def vec_dot_q6_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
    flags: CpuFlags,
) -> Float32:
    """Q6_K × Q8_K dot product with CPU dispatch."""
    if flags.has_neon():
        return vec_dot_q6_k_q8_k_neon(w_block, q8_data)
    elif flags.has_avx2():
        return _vec_dot_q6_k_q8_k_avx(w_block, q8_data)
    else:
        return _vec_dot_q6_k_q8_k_scalar(w_block, q8_data)


def vec_dot_q2_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
    flags: CpuFlags,
) -> Float32:
    """Q2_K × Q8_K dot product with CPU dispatch."""
    if flags.has_neon():
        return vec_dot_q2_k_q8_k_neon(w_block, q8_data)
    elif flags.has_avx2():
        return _vec_dot_q2_k_q8_k_avx(w_block, q8_data)
    else:
        return _vec_dot_q2_k_q8_k_scalar(w_block, q8_data)


def vec_dot_q3_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
    flags: CpuFlags,
) -> Float32:
    """Q3_K × Q8_K dot product with CPU dispatch."""
    if flags.has_neon():
        return vec_dot_q3_k_q8_k_neon(w_block, q8_data)
    elif flags.has_avx2():
        return _vec_dot_q3_k_q8_k_avx(w_block, q8_data)
    else:
        return _vec_dot_q3_k_q8_k_scalar(w_block, q8_data)


# ============================================================================
# Scalar fallbacks
# ============================================================================


def _vec_dot_q4_k_q8_k_scalar(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q5_k_q8_k_scalar(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q6_k_q8_k_scalar(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q2_k_q8_k_scalar(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q3_k_q8_k_scalar(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q4_k_q8_k_avx(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q5_k_q8_k_avx(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q6_k_q8_k_avx(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q2_k_q8_k_avx(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


def _vec_dot_q3_k_q8_k_avx(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    return Float32(0)


# ============================================================================
# Import SIMD implementations
# ============================================================================


from .simd_neon import (
    vec_dot_q3_k_q8_k as vec_dot_q3_k_q8_k_neon,
    vec_dot_q4_k_q8_k_nrc2,
)
from .q2k_q8k_dot import vec_dot_q2_k_q8_k as vec_dot_q2_k_q8_k_neon
from .q4k_q8k_dot import vec_dot_q4_k_q8_k as vec_dot_q4_k_q8_k_neon
from .q5k_q8k_dot import vec_dot_q5_k_q8_k as vec_dot_q5_k_q8_k_neon
from .q6k_q8k_dot import vec_dot_q6_k_q8_k as vec_dot_q6_k_q8_k_neon
