# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/__init__.mojo

from .simd_base import vec_dot_q4_k, vec_dot_q4_0, vec_dot_q8_0
from .simd_base import TARGET_HAS_NEON, TARGET_HAS_AVX2, TARGET_IS_X86
