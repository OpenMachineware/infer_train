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
    # IQ series
    vec_dot_iq3s_q8k,
    vec_dot_iq3xxs_q8k,
    vec_dot_iq2s_q8k,
    vec_dot_iq2xxs_q8k,
    vec_dot_iq2xs_q8k,
    vec_dot_iq1s_q8k,
    vec_dot_iq1m_q8k,
    vec_dot_iq4xs_q8k,
    vec_dot_iq4nl_q80,
)
