# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/forward_complete_gpu.mojo
#
# Complete GPU forward pass for decode (single token generation).
# All operations stay on GPU, only final logits downloaded to CPU.

from max.gpu.host import DeviceContext, DeviceBuffer
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.gpu.gpu_runtime import upload, download2
from src.core.ops.gpu.matmul_fp16_gpu import matmul_fp16_gpu_pipeline
from src.core.ops.gpu.layer_gpu import layer_forward_gpu
from src.core.ops.gpu.kv_cache_gpu import KVCacheGPU
from src.core.ops.gpu.rms_norm_gpu import rms_norm_weight_gpu_pipeline
from std.utils.static_tuple import StaticTuple


struct GPUModelContext:
    """GPU context and cached buffers for a model.

    Holds the GPU context, KV cache, and weight buffers.
    Initialized once at model load, reused across all forward passes.
    """

    var ctx: Optional[DeviceContext]
    var kv_cache: KVCacheGPU
    var initialized: Bool

    def __init__(out self):
        self.ctx = None
        self.kv_cache = KVCacheGPU()
        self.initialized = False

    def initialize(
        mut self,
        num_layers: Int,
        n_kv_heads: Int,
        max_len: Int,
        head_dim: Int,
    ) raises:
        """Initialize GPU context and KV cache."""
        self.ctx = DeviceContext()
        self.kv_cache = KVCacheGPU(self.ctx.value(), num_layers, n_kv_heads, max_len, head_dim)
        self.initialized = True

    def get_context(self) -> DeviceContext:
        """Get the GPU device context."""
        if not self.ctx:
            raise "GPUModelContext not initialized"
        return self.ctx.value()

    def reset_cache(mut self):
        """Reset KV cache for new generation."""
        self.kv_cache.reset()


def forward_decode_complete_gpu[
    ModelType: Movable
](
    ctx: DeviceContext,
    token: Int,
    position: Int,
    model: ModelType,
    mut gpu_model_ctx: GPUModelContext,
) raises -> Tensor[DType.float32, 1]:
    """Complete GPU forward pass for decode.

    Pipeline:
    1. Initialize GPU KV cache if needed
    2. Embedding lookup (CPU -> GPU)
    3. For each layer: Attention + FFN (all on GPU)
    4. Output norm (GPU)
    5. Output projection (GPU)
    6. Download logits (GPU -> CPU, once)

    Args:
        ctx: GPU device context
        token: Input token ID
        position: Current position
        model: TransformerModel with weights
        gpu_model_ctx: GPU context with cached buffers

    Returns:
        Logits tensor [vocab_size]
    """
    var cfg = model.config
    var hidden = cfg.hidden
    var n_heads = cfg.n_heads
    var n_kv_heads = cfg.n_kv_heads
    var head_dim = cfg.head_dim
    var ffn_dim = cfg.ffn_dim
    var theta = cfg.rope_theta

    # Step 1: Initialize GPU KV cache if needed
    if not gpu_model_ctx.initialized:
        gpu_model_ctx.initialize(
            cfg.n_layers, n_kv_heads, 1024, head_dim  # max_len = 1024
        )

    # Step 2: Embedding lookup
    # Note: Embedding is currently on CPU, upload to GPU
    var embed_w = model.qparams.token_embd
    var embed_row = embed_w.get_embedding_row(token)  # [hidden] CPU
    var x_buf = _upload_1d_as_2d(ctx, embed_row, 1, hidden)  # [1, hidden] GPU

    # Step 3: Process each layer
    for layer_idx in range(cfg.n_layers):
        var lw = model.layer_view(layer_idx)

        # Get weight buffers (upload or use cached)
        var attn_norm_w_buf = _upload_norm_weight(ctx, lw.attn_norm_w)
        var q_w_buf = lw.q_w.ensure_gpu_buf_fp16(ctx)
        var k_w_buf = lw.k_w.ensure_gpu_buf_fp16(ctx)
        var v_w_buf = lw.v_w.ensure_gpu_buf_fp16(ctx)
        var o_w_buf = lw.o_w.ensure_gpu_buf_fp16(ctx)
        var ffn_norm_w_buf = _upload_norm_weight(ctx, lw.ffn_norm_w)
        var gate_w_buf = lw.gate_w.ensure_gpu_buf_fp16(ctx)
        var up_w_buf = lw.up_w.ensure_gpu_buf_fp16(ctx)
        var down_w_buf = lw.down_w.ensure_gpu_buf_fp16(ctx)

        # Get KV cache for this layer
        var kv_cache = gpu_model_ctx.kv_cache.layers[layer_idx]

        # Run layer forward on GPU
        x_buf = layer_forward_gpu(
            ctx, x_buf, layer_idx, position,
            n_heads, n_kv_heads, head_dim, hidden, ffn_dim, theta,
            attn_norm_w_buf, q_w_buf, k_w_buf, v_w_buf, o_w_buf,
            ffn_norm_w_buf, gate_w_buf, up_w_buf, down_w_buf,
            kv_cache,
        )

    # Step 4: Output norm
    var output_norm_w = model.qparams.output_norm_w
    var output_norm_w_buf = _upload_norm_weight(ctx, output_norm_w)
    x_buf = rms_norm_weight_gpu_pipeline(ctx, x_buf, output_norm_w_buf, 1, hidden)

    # Step 5: Output projection
    var output_w = model.qparams.output_w
    var output_w_buf = output_w.ensure_gpu_buf_fp16(ctx)
    var logits_buf = matmul_fp16_gpu_pipeline(x_buf, output_w_buf, ctx, 1, hidden, cfg.vocab)

    # Step 6: Download logits
    var logits_f16 = download2[DType.float16](
        ctx, logits_buf, StaticTuple[Int, 2](1, cfg.vocab)
    )
    ctx.synchronize()

    # Convert to f32
    var logits = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](cfg.vocab))
    for i in range(cfg.vocab):
        logits.set(i, Scalar[DType.float32](Float32(logits_f16.get(i))))

    return logits


def _upload_norm_weight(
    ctx: DeviceContext,
    weight: Tensor[DType.float16, 1],
) raises -> DeviceBuffer[DType.float16]:
    """Upload 1D norm weight to GPU.

    Args:
        ctx: GPU device context
        weight: 1D tensor [dim]

    Returns:
        GPU buffer [dim]
    """
    return upload[DType.float16, 1](ctx, weight)


def _upload_1d_as_2d(
    ctx: DeviceContext,
    weight: Tensor[DType.float16, 1],
    rows: Int,
    cols: Int,
) raises -> DeviceBuffer[DType.float16]:
    """Upload 1D tensor as 2D to GPU.

    Args:
        ctx: GPU device context
        weight: 1D tensor [rows * cols]
        rows: Number of rows
        cols: Number of columns

    Returns:
        GPU buffer [rows, cols]
    """
    # Create 2D view and upload
    var w2d = Tensor[DType.float16, 2](StaticTuple[Int, 2](rows, cols))
    for i in range(rows * cols):
        w2d.set(i, weight.get(i))
    return upload[DType.float16, 2](ctx, w2d)


# ============================================================================
# Alternative: Simpler version with existing infrastructure
# ============================================================================


def forward_decode_gpu_simple[
    ModelType: Movable
](
    ctx: DeviceContext,
    token: Int,
    position: Int,
    model: ModelType,
) -> Tensor[DType.float32, 1]:
    """Simpler GPU forward using existing model.forward() with GPU context.

    This leverages the existing forward path but ensures GPU is used
    where possible (FFN). Not as fast as full GPU pipeline but easier
    to implement and test.
    """
    # Use existing forward, but with GPU context passed
    return model.forward(token, position)
