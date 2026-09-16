# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/layer_gpu.mojo
#
# Single transformer layer on GPU with zero-copy activation chaining.

from max.gpu.host import DeviceContext, DeviceBuffer
from src.core.ops.gpu.matmul_fp16_gpu import matmul_fp16_gpu_pipeline
from src.core.ops.gpu.attention_decode_gpu import attention_decode_gpu_pipeline
from src.core.ops.gpu.rope_gpu import rope_gpu_pipeline
from src.core.ops.gpu.add_gpu import add_gpu_pipeline
from src.core.ops.gpu.swiglu_gpu import swiglu_gpu_pipeline
from src.core.ops.gpu.rms_norm_gpu import rms_norm_weight_gpu_pipeline
from src.core.ops.gpu.kv_cache_gpu import KVCacheLayerGPU
from std.math import sqrt


# ============================================================================
# Complete Layer Forward
# ============================================================================


def layer_forward_gpu(
    ctx: DeviceContext,
    x_buf: DeviceBuffer[DType.float16],  # [1, hidden]
    layer_idx: Int,
    position: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    hidden: Int,
    ffn_dim: Int,
    theta: Float32,
    # Weights (all on GPU)
    attn_norm_w_buf: DeviceBuffer[DType.float16],
    q_w_buf: DeviceBuffer[DType.float16],
    k_w_buf: DeviceBuffer[DType.float16],
    v_w_buf: DeviceBuffer[DType.float16],
    o_w_buf: DeviceBuffer[DType.float16],
    ffn_norm_w_buf: DeviceBuffer[DType.float16],
    gate_w_buf: DeviceBuffer[DType.float16],
    up_w_buf: DeviceBuffer[DType.float16],
    down_w_buf: DeviceBuffer[DType.float16],
    # KV cache (on GPU)
    mut kv_cache: KVCacheLayerGPU,
) raises -> DeviceBuffer[DType.float16]:
    """Complete transformer layer on GPU.

    Pipeline:
    1. RMS Norm (attention)
    2. QKV Projection
    3. RoPE
    4. KV Cache Update
    5. Attention
    6. Output Projection
    7. Residual Add
    8. RMS Norm (FFN)
    9. Gate/Up Projection
    10. SwiGLU
    11. Down Projection
    12. Residual Add

    All operations stay on GPU, no CPU transfers.
    """
    # Make x mutable for residual updates
    var x = x_buf

    # 1. RMS Norm (attention)
    var normed = rms_norm_weight_gpu_pipeline(ctx, x, attn_norm_w_buf, 1, hidden)

    # 2. QKV Projection
    var q_buf = matmul_fp16_gpu_pipeline(normed, q_w_buf, ctx, 1, hidden, n_heads * head_dim)
    var k_buf = matmul_fp16_gpu_pipeline(normed, k_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)
    var v_buf = matmul_fp16_gpu_pipeline(normed, v_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)

    # 3. RoPE
    q_buf = rope_gpu_pipeline(ctx, q_buf, n_heads, 1, head_dim, position, theta)
    k_buf = rope_gpu_pipeline(ctx, k_buf, n_kv_heads, 1, head_dim, position, theta)

    # 4. Update KV cache with new k, v
    kv_cache.update(ctx, k_buf, v_buf, position)

    # 5. Attention
    var kv_len = kv_cache.filled_len()
    var attn_out = attention_decode_gpu_pipeline(
        ctx, q_buf, kv_cache.get_k_buffer(), kv_cache.get_v_buffer(),
        n_heads, head_dim, kv_len
    )

    # 6. Output Projection
    attn_out = matmul_fp16_gpu_pipeline(attn_out, o_w_buf, ctx, 1, n_heads * head_dim, hidden)

    # 7. Residual Add
    x = add_gpu_pipeline(ctx, x, attn_out, hidden)

    # 8. RMS Norm (FFN)
    var normed2 = rms_norm_weight_gpu_pipeline(ctx, x, ffn_norm_w_buf, 1, hidden)

    # 9. Gate/Up Projection
    var gate = matmul_fp16_gpu_pipeline(normed2, gate_w_buf, ctx, 1, hidden, ffn_dim)
    var up = matmul_fp16_gpu_pipeline(normed2, up_w_buf, ctx, 1, hidden, ffn_dim)

    # 10. SwiGLU
    var hidden_ffn = swiglu_gpu_pipeline(ctx, gate, up, ffn_dim)

    # 11. Down Projection
    var ffn_out = matmul_fp16_gpu_pipeline(hidden_ffn, down_w_buf, ctx, 1, ffn_dim, hidden)

    # 12. Residual Add
    x = add_gpu_pipeline(ctx, x, ffn_out, hidden)

    return x
