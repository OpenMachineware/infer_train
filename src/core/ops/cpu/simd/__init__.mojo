# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/__init__.mojo
#
# SIMD kernel exports for quantized operations.

from .simd_base import (
    vec_dot_qk_q8k,  # Unified dispatch
    vec_dot_q4_k_q8_k,
    vec_dot_q5_k_q8_k,
    vec_dot_q6_k_q8_k,
    vec_dot_q2_k_q8_k,
    vec_dot_q3_k_q8_k,
    vec_dot_q4_k_q8_k_nrc2,
)
