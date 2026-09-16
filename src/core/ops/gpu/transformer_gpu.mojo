# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/transformer_gpu.mojo
#
# GPU-enabled transformer operations.
#
# Status: Phase 1 - Infrastructure only. Actual GPU acceleration requires:
#   - Phase 2: Tiled GEMM for FP16/FP32
#   - Phase 3: Quantized GPU matmul (dequant on device)
#   - Phase 4: GPU attention
#
# Current behavior: All ops fall back to CPU because:
#   1. Weights are quantized (Q4_K, Q6_K, etc.) - GPU matmul only supports FP16/FP32
#   2. Attention has no GPU kernel
#   3. RMS norm with weight is CPU-only
#
# This file exists to enable future GPU path and provide the infrastructure.

from ...device import Device, get_default_device, has_metal_gpu
from ...tensor import Tensor
from ..cpu.matmul_cpu import matmul_weight_cpu
from ..cpu.add_cpu import add_cpu_dynamic, add_row_cpu
from ..cpu.swiglu_cpu import swiglu_cpu_dynamic
from ..attention.mha import mha_forward_v2, mha_forward_batch, MHAOptions
from ..attention.kv_cache import KVCacheLayer
from .gpu_runtime import gpu_available
from .matmul_gpu import matmul_gpu_dynamic
from .rms_norm_gpu import rms_norm_gpu_dynamic
from .add_gpu import add_gpu_dynamic
from .swiglu_gpu import swiglu_gpu_dynamic
from .fused_gpu import (
    fused_matmul_add_bias_gpu,
    fused_matmul_add_gpu,
    fused_swiglu_matmul_gpu,
)


def can_use_gpu[dtype: DType]() -> Bool:
    """Check if GPU is available for the given dtype."""
    return gpu_available[dtype]()


def should_use_gpu() -> Bool:
    """Check if GPU should be used (device preference + availability)."""
    var dev = get_default_device()
    return dev.is_gpu()


# -- GPU RMS Norm with Weight -------------------------------------------------
#
# The GPU rms_norm_gpu_dynamic doesn't take a weight parameter.
# For now, we use the CPU version. A GPU kernel could be added later.


def rms_norm_weight_gpu[dtype: DType](
    x: Tensor[dtype, 2], w: Tensor[dtype, 1], eps: Float32
) -> Tensor[dtype, 2]:
    """RMSNorm with weight: out = x / sqrt(mean(x^2) + eps) * w.

    Currently uses CPU because the GPU kernel doesn't support weight.
    """
    # TODO: Implement GPU version with weight
    # For now, fall back to CPU
    from ..cpu.rms_norm_cpu import rms_norm_weight_cpu
    return rms_norm_weight_cpu[dtype](x, w, eps)


# -- GPU Matmul with Quantized Weights ---------------------------------------
#
# The quantized weight projections require on-device dequantization,
# which is not yet implemented. Fall back to CPU for quantized weights.


def matmul_quant_gpu[dtype: DType](
    x: Tensor[dtype, 2],
    w_quant: Tensor[DType.uint8, 2],
    scale: Tensor[dtype, 1],
    quant_type: Int,
) -> Tensor[dtype, 2]:
    """Matmul with quantized weights on GPU.

    Not yet implemented - requires on-device dequantization.
    Falls back to CPU.
    """
    # TODO: Implement quantized GPU matmul
    # For now, fall back to CPU
    from ..cpu.matmul_cpu import matmul_weight_cpu_threaded
    _ = quant_type  # unused for now
    return matmul_weight_cpu_threaded[dtype](x, w_quant)


# -- GPU Attention -----------------------------------------------------------
#
# GPU attention requires:
#   1. KV cache in GPU memory
#   2. Flash attention kernel
#   3. RoPE on GPU
#
# Not yet implemented. Falls back to CPU.


def mha_forward_gpu(
    x: Tensor[DType.float16, 2],
    wq: Tensor[DType.float16, 2],
    wk: Tensor[DType.float16, 2],
    wv: Tensor[DType.float16, 2],
    wo: Tensor[DType.float16, 2],
    bq: Tensor[DType.float16, 1],
    bk: Tensor[DType.float16, 1],
    bv: Tensor[DType.float16, 1],
    cache_layer: KVCacheLayer,
    start_pos: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    rope_theta: Float32,
    opts: MHAOptions,
) -> Tensor[DType.float16, 2]:
    """Multi-head attention on GPU.

    Not yet implemented - falls back to CPU.
    """
    # TODO: Implement GPU attention
    # For now, use CPU
    return mha_forward_v2(
        x, wq, wk, wv, wo, bq, bk, bv,
        cache_layer, start_pos,
        n_heads, n_kv_heads, head_dim, rope_theta, opts,
        Tensor[DType.float16, 1](),  # dummy_scale
    )
