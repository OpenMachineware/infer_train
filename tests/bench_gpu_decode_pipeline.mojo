# Benchmark GPU decode forward pipeline
#
# Simulates a complete decode step with GPU pipeline:
# - Embedding lookup (simulated)
# - All layers: Norm → QKV → RoPE → Attention → FFN
# - All operations on GPU with zero-copy
#
# Measures end-to-end latency and identifies bottlenecks.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.gpu.gpu_runtime import get_gpu_context, upload, download2
from src.core.ops.gpu.matmul_fp16_gpu import matmul_fp16_gpu_pipeline
from src.core.ops.gpu.attention_decode_gpu import attention_decode_gpu_pipeline
from src.core.ops.gpu.rope_gpu import rope_gpu_pipeline
from src.core.ops.gpu.rms_norm_gpu import rms_norm_weight_gpu_pipeline
from src.core.ops.gpu.add_gpu import add_gpu_pipeline
from src.core.ops.gpu.swiglu_gpu import swiglu_gpu_pipeline
from src.core.ops.gpu.kv_cache_gpu import KVCacheLayerGPU
from max.gpu.host import DeviceContext, DeviceBuffer
from std.utils.static_tuple import StaticTuple
from std.collections.optional import Optional
from src.core.thread_pool import now_ns


def _create_weight_tensor(rows: Int, cols: Int) -> Tensor[DType.float16, 2]:
    """Create a weight tensor with small values."""
    var t = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](rows, cols))
    for i in range(rows):
        for j in range(cols):
            var val = Float32(0.01) * Float32((i + j) % 100)
            t.set(i * cols + j, Scalar[DType.float16](val))
    return t


def _create_1d_tensor(dim: Int) -> Tensor[DType.float16, 1]:
    """Create a 1D tensor."""
    var t = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](dim))
    for i in range(dim):
        var val = Float32(0.01) * Float32(i % 100)
        t.set(i, Scalar[DType.float16](val))
    return t


def benchmark_single_layer(
    ctx: DeviceContext,
    x_buf: DeviceBuffer[DType.float16],
    position: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    hidden: Int,
    ffn_dim: Int,
    theta: Float32,
    # Weights (pre-uploaded)
    attn_norm_w_buf: DeviceBuffer[DType.float16],
    q_w_buf: DeviceBuffer[DType.float16],
    k_w_buf: DeviceBuffer[DType.float16],
    v_w_buf: DeviceBuffer[DType.float16],
    o_w_buf: DeviceBuffer[DType.float16],
    ffn_norm_w_buf: DeviceBuffer[DType.float16],
    gate_w_buf: DeviceBuffer[DType.float16],
    up_w_buf: DeviceBuffer[DType.float16],
    down_w_buf: DeviceBuffer[DType.float16],
    # KV cache
    mut kv_cache: KVCacheLayerGPU,
) raises -> DeviceBuffer[DType.float16]:
    """Run one transformer layer on GPU."""

    # 1. RMS Norm (attention)
    var normed = rms_norm_weight_gpu_pipeline(ctx, x_buf, attn_norm_w_buf, 1, hidden)

    # 2. QKV Projection
    var q_buf = matmul_fp16_gpu_pipeline(normed, q_w_buf, ctx, 1, hidden, n_heads * head_dim)
    var k_buf = matmul_fp16_gpu_pipeline(normed, k_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)
    var v_buf = matmul_fp16_gpu_pipeline(normed, v_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)

    # 3. RoPE
    q_buf = rope_gpu_pipeline(ctx, q_buf, n_heads, 1, head_dim, position, theta)
    k_buf = rope_gpu_pipeline(ctx, k_buf, n_kv_heads, 1, head_dim, position, theta)

    # 4. KV Cache Update
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
    var x = add_gpu_pipeline(ctx, x_buf, attn_out, hidden)

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


def main() raises:
    # Model configuration (Qwen2-7B style)
    var n_layers = 32
    var n_heads = 32
    var n_kv_heads = 32
    var head_dim = 128
    var hidden = n_heads * head_dim  # 4096
    var ffn_dim = 18944
    var theta = Float32(1000000.0)
    var max_seq_len = 1024

    print("=" * 70)
    print("GPU Decode Pipeline Benchmark")
    print("=" * 70)
    print("Model config:")
    print("  Layers:", n_layers)
    print("  Hidden:", hidden)
    print("  Heads:", n_heads)
    print("  Head dim:", head_dim)
    print("  FFN dim:", ffn_dim)
    print()

    # Get GPU context
    print("Initializing GPU context...")
    var ctx = get_gpu_context()

    # Create KV caches for all layers
    print("Creating GPU KV caches...")
    var kv_caches = List[KVCacheLayerGPU]()
    for layer in range(n_layers):
        kv_caches.append(KVCacheLayerGPU(ctx, n_kv_heads, max_seq_len, head_dim))

    # Create and upload weights (simulate model loading)
    print("Creating and uploading weights...")
    var start = now_ns()

    # For simplicity, use the same weights for all layers
    var attn_norm_w = _create_1d_tensor(hidden)
    var q_w = _create_weight_tensor(n_heads * head_dim, hidden)
    var k_w = _create_weight_tensor(n_kv_heads * head_dim, hidden)
    var v_w = _create_weight_tensor(n_kv_heads * head_dim, hidden)
    var o_w = _create_weight_tensor(hidden, n_heads * head_dim)
    var ffn_norm_w = _create_1d_tensor(hidden)
    var gate_w = _create_weight_tensor(ffn_dim, hidden)
    var up_w = _create_weight_tensor(ffn_dim, hidden)
    var down_w = _create_weight_tensor(hidden, ffn_dim)

    # Upload to GPU
    var attn_norm_w_buf = upload[DType.float16, 1](ctx, attn_norm_w)
    var q_w_buf = upload[DType.float16, 2](ctx, q_w)
    var k_w_buf = upload[DType.float16, 2](ctx, k_w)
    var v_w_buf = upload[DType.float16, 2](ctx, v_w)
    var o_w_buf = upload[DType.float16, 2](ctx, o_w)
    var ffn_norm_w_buf = upload[DType.float16, 1](ctx, ffn_norm_w)
    var gate_w_buf = upload[DType.float16, 2](ctx, gate_w)
    var up_w_buf = upload[DType.float16, 2](ctx, up_w)
    var down_w_buf = upload[DType.float16, 2](ctx, down_w)

    var upload_time = now_ns() - start
    print("  Weight upload time:", Float64(upload_time) / 1e6, "ms")
    print()

    # Create input (simulated embedding)
    print("Creating input...")
    var x_cpu = _create_1d_tensor(hidden)
    var x_buf = upload[DType.float16, 1](ctx, x_cpu)
    # Reshape to 2D [1, hidden]
    # For simplicity, we'll create a proper 2D tensor
    var x_2d = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x_2d.set(i, x_cpu.get(i))
    x_buf = upload[DType.float16, 2](ctx, x_2d)

    # Warmup
    print("Warming up (1 iteration)...")
    ctx.synchronize()
    start = now_ns()
    _ = benchmark_single_layer(
        ctx, x_buf, 0, n_heads, n_kv_heads, head_dim, hidden, ffn_dim, theta,
        attn_norm_w_buf, q_w_buf, k_w_buf, v_w_buf, o_w_buf,
        ffn_norm_w_buf, gate_w_buf, up_w_buf, down_w_buf,
        kv_caches[0],
    )
    ctx.synchronize()
    var warmup_time = now_ns() - start
    print("  Warmup time:", Float64(warmup_time) / 1e6, "ms")
    print()

    # Benchmark full forward pass (all layers)
    print("Benchmarking full forward pass (32 layers)...")
    var iterations = 10
    var total_time = Float64(0)
    var final_x = x_buf

    for iter in range(iterations):
        # Reset KV caches
        for layer in range(n_layers):
            kv_caches[layer].reset()

        ctx.synchronize()
        start = now_ns()

        # Run all layers
        var x = x_buf
        for layer in range(n_layers):
            x = benchmark_single_layer(
                ctx, x, iter, n_heads, n_kv_heads, head_dim, hidden, ffn_dim, theta,
                attn_norm_w_buf, q_w_buf, k_w_buf, v_w_buf, o_w_buf,
                ffn_norm_w_buf, gate_w_buf, up_w_buf, down_w_buf,
                kv_caches[layer],
            )

        final_x = x

        ctx.synchronize()
        var elapsed = now_ns() - start
        total_time += Float64(elapsed)

        if iter == 0:
            print("  Iteration 1:", Float64(elapsed) / 1e6, "ms")

    var avg_time = total_time / Float64(iterations)
    print("  Average time:", avg_time / 1e6, "ms")
    print("  Time per layer:", avg_time / 1e6 / Float64(n_layers), "ms")
    print()

    # Download result to verify
    print("Downloading result...")
    var result = download2[DType.float16](ctx, final_x, StaticTuple[Int, 2](1, hidden))
    ctx.synchronize()
    print("  Output shape: (", result.shape()[0], ", ", result.shape()[1], ")")
    print("  Output [0]:", Float32(result.get(0)))
    print()

    # Performance analysis
    print("=" * 70)
    print("Performance Summary:")
    print("=" * 70)
    print("  Full forward (32 layers):", avg_time / 1e6, "ms")
    print("  Per layer:", avg_time / 1e6 / Float64(n_layers), "ms")
    print()
    print("Operations per layer:")
    print("  - 2x RMS Norm")
    print("  - 5x Matmul (Q, K, V, O, down)")
    print("  - 2x Matmul (gate, up)")
    print("  - 2x RoPE")
    print("  - 1x Attention (Q@K^T, softmax, @V)")
    print("  - 1x SwiGLU")
    print("  - 2x Add (residual)")
    print("  - 1x KV cache update")
    print()
    print("Target: <30ms for full forward (vs ~108ms CPU)")
    print("Current:", avg_time / 1e6, "ms")
    if avg_time / 1e6 < 30.0:
        print("  ✓ TARGET ACHIEVED!")
    else:
        print("  Gap:", avg_time / 1e6 / 30.0, "x slower than target")
    print()
    print("Note: This is the INITIAL implementation.")
    print("Performance optimization comes later (kernel fusion, shared memory, etc.)")
    print("=" * 70)
