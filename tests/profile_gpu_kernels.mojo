# Profile GPU decode pipeline
#
# Measures time for each individual kernel to identify bottlenecks.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.gpu.gpu_runtime import get_gpu_context, upload, download2
from src.core.ops.gpu.matmul_fp16_gpu import matmul_fp16_gpu_pipeline
from src.core.ops.gpu.attention_decode_gpu import attention_decode_gpu_pipeline
from src.core.ops.gpu.rope_gpu import rope_gpu_pipeline
from src.core.ops.gpu.rms_norm_gpu import rms_norm_weight_gpu_pipeline
from src.core.ops.gpu.add_gpu import add_gpu_pipeline
from src.core.ops.gpu.swiglu_gpu import swiglu_gpu_pipeline
from src.core.ops.gpu.kv_cache_gpu import KVCacheLayerGPU
from max.gpu.host import DeviceContext
from std.utils.static_tuple import StaticTuple
from src.core.thread_pool import now_ns


def _create_test_tensor(rows: Int, cols: Int) -> Tensor[DType.float16, 2]:
    """Create a test tensor with small values."""
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


def main() raises:
    # Model configuration (Qwen2-7B style)
    var n_heads = 32
    var n_kv_heads = 32
    var head_dim = 128
    var hidden = n_heads * head_dim  # 4096
    var ffn_dim = 18944
    var theta = Float32(1000000.0)
    var max_seq_len = 128

    print("=" * 70)
    print("GPU Kernel Profiling")
    print("=" * 70)
    print("Model config:")
    print("  Hidden:", hidden)
    print("  Heads:", n_heads)
    print("  Head dim:", head_dim)
    print("  FFN dim:", ffn_dim)
    print()

    # Get GPU context
    print("Initializing GPU context...")
    var ctx = get_gpu_context()

    # Create test data
    print("Creating test data...")
    var x_cpu = _create_test_tensor(1, hidden)
    var norm_w_cpu = _create_1d_tensor(hidden)
    var q_w_cpu = _create_test_tensor(n_heads * head_dim, hidden)
    var k_w_cpu = _create_test_tensor(n_kv_heads * head_dim, hidden)
    var v_w_cpu = _create_test_tensor(n_kv_heads * head_dim, hidden)
    var o_w_cpu = _create_test_tensor(hidden, n_heads * head_dim)
    var gate_w_cpu = _create_test_tensor(ffn_dim, hidden)
    var up_w_cpu = _create_test_tensor(ffn_dim, hidden)
    var down_w_cpu = _create_test_tensor(hidden, ffn_dim)

    # Upload to GPU
    print("Uploading to GPU...")
    var x_buf = upload[DType.float16, 2](ctx, x_cpu)
    var norm_w_buf = upload[DType.float16, 1](ctx, norm_w_cpu)
    var q_w_buf = upload[DType.float16, 2](ctx, q_w_cpu)
    var k_w_buf = upload[DType.float16, 2](ctx, k_w_cpu)
    var v_w_buf = upload[DType.float16, 2](ctx, v_w_cpu)
    var o_w_buf = upload[DType.float16, 2](ctx, o_w_cpu)
    var gate_w_buf = upload[DType.float16, 2](ctx, gate_w_cpu)
    var up_w_buf = upload[DType.float16, 2](ctx, up_w_cpu)
    var down_w_buf = upload[DType.float16, 2](ctx, down_w_cpu)

    # Create KV cache
    var kv_cache = KVCacheLayerGPU(ctx, n_kv_heads, max_seq_len, head_dim)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = rms_norm_weight_gpu_pipeline(ctx, x_buf, norm_w_buf, 1, hidden)
        _ = matmul_fp16_gpu_pipeline(x_buf, q_w_buf, ctx, 1, hidden, n_heads * head_dim)
    ctx.synchronize()

    print()
    print("Profiling individual kernels...")
    print("-" * 70)

    var iterations = 10

    # Profile RMSNorm
    var start = now_ns()
    for _ in range(iterations):
        _ = rms_norm_weight_gpu_pipeline(ctx, x_buf, norm_w_buf, 1, hidden)
    ctx.synchronize()
    var rms_norm_time = Float64(now_ns() - start) / Float64(iterations)
    print("  RMSNorm:              ", rms_norm_time / 1e6, "ms")

    # Profile Matmul (Q projection)
    var normed = rms_norm_weight_gpu_pipeline(ctx, x_buf, norm_w_buf, 1, hidden)
    start = now_ns()
    for _ in range(iterations):
        _ = matmul_fp16_gpu_pipeline(normed, q_w_buf, ctx, 1, hidden, n_heads * head_dim)
    ctx.synchronize()
    var matmul_q_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Matmul (Q):           ", matmul_q_time / 1e6, "ms")

    # Profile Matmul (K projection)
    start = now_ns()
    for _ in range(iterations):
        _ = matmul_fp16_gpu_pipeline(normed, k_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)
    ctx.synchronize()
    var matmul_k_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Matmul (K):           ", matmul_k_time / 1e6, "ms")

    # Profile Matmul (V projection)
    start = now_ns()
    for _ in range(iterations):
        _ = matmul_fp16_gpu_pipeline(normed, v_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)
    ctx.synchronize()
    var matmul_v_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Matmul (V):           ", matmul_v_time / 1e6, "ms")

    # Profile RoPE
    var q_buf = matmul_fp16_gpu_pipeline(normed, q_w_buf, ctx, 1, hidden, n_heads * head_dim)
    var k_buf = matmul_fp16_gpu_pipeline(normed, k_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)
    start = now_ns()
    for _ in range(iterations):
        _ = rope_gpu_pipeline(ctx, q_buf, n_heads, 1, head_dim, 0, theta)
    ctx.synchronize()
    var rope_time = Float64(now_ns() - start) / Float64(iterations)
    print("  RoPE:                 ", rope_time / 1e6, "ms")

    # Profile KV Cache Update
    var v_buf = matmul_fp16_gpu_pipeline(normed, v_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)
    kv_cache.reset()
    start = now_ns()
    for _ in range(iterations):
        kv_cache.update(ctx, k_buf, v_buf, 0)
    ctx.synchronize()
    var kv_cache_time = Float64(now_ns() - start) / Float64(iterations)
    print("  KV Cache Update:      ", kv_cache_time / 1e6, "ms")

    # Profile Attention
    kv_cache.update(ctx, k_buf, v_buf, 0)
    start = now_ns()
    for _ in range(iterations):
        _ = attention_decode_gpu_pipeline(
            ctx, q_buf, kv_cache.get_k_buffer(), kv_cache.get_v_buffer(),
            n_heads, head_dim, 1
        )
    ctx.synchronize()
    var attention_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Attention:            ", attention_time / 1e6, "ms")

    # Profile Matmul (O projection)
    var attn_out = attention_decode_gpu_pipeline(
        ctx, q_buf, kv_cache.get_k_buffer(), kv_cache.get_v_buffer(),
        n_heads, head_dim, 1
    )
    start = now_ns()
    for _ in range(iterations):
        _ = matmul_fp16_gpu_pipeline(attn_out, o_w_buf, ctx, 1, n_heads * head_dim, hidden)
    ctx.synchronize()
    var matmul_o_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Matmul (O):           ", matmul_o_time / 1e6, "ms")

    # Profile Add (residual)
    start = now_ns()
    for _ in range(iterations):
        _ = add_gpu_pipeline(ctx, x_buf, attn_out, hidden)
    ctx.synchronize()
    var add_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Add (residual):       ", add_time / 1e6, "ms")

    # Profile Matmul (gate)
    start = now_ns()
    for _ in range(iterations):
        _ = matmul_fp16_gpu_pipeline(x_buf, gate_w_buf, ctx, 1, hidden, ffn_dim)
    ctx.synchronize()
    var matmul_gate_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Matmul (gate):        ", matmul_gate_time / 1e6, "ms")

    # Profile Matmul (up)
    start = now_ns()
    for _ in range(iterations):
        _ = matmul_fp16_gpu_pipeline(x_buf, up_w_buf, ctx, 1, hidden, ffn_dim)
    ctx.synchronize()
    var matmul_up_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Matmul (up):          ", matmul_up_time / 1e6, "ms")

    # Profile SwiGLU
    var gate_buf = matmul_fp16_gpu_pipeline(x_buf, gate_w_buf, ctx, 1, hidden, ffn_dim)
    var up_buf = matmul_fp16_gpu_pipeline(x_buf, up_w_buf, ctx, 1, hidden, ffn_dim)
    start = now_ns()
    for _ in range(iterations):
        _ = swiglu_gpu_pipeline(ctx, gate_buf, up_buf, ffn_dim)
    ctx.synchronize()
    var swiglu_time = Float64(now_ns() - start) / Float64(iterations)
    print("  SwiGLU:               ", swiglu_time / 1e6, "ms")

    # Profile Matmul (down)
    var hidden_ffn = swiglu_gpu_pipeline(ctx, gate_buf, up_buf, ffn_dim)
    start = now_ns()
    for _ in range(iterations):
        _ = matmul_fp16_gpu_pipeline(hidden_ffn, down_w_buf, ctx, 1, ffn_dim, hidden)
    ctx.synchronize()
    var matmul_down_time = Float64(now_ns() - start) / Float64(iterations)
    print("  Matmul (down):        ", matmul_down_time / 1e6, "ms")

    print()
    print("-" * 70)
    print("Summary:")
    var total_time = (
        rms_norm_time * 2 +  # 2x RMSNorm
        matmul_q_time + matmul_k_time + matmul_v_time + matmul_o_time +
        matmul_gate_time + matmul_up_time + matmul_down_time +
        rope_time * 2 +  # 2x RoPE
        kv_cache_time +
        attention_time +
        add_time * 2 +  # 2x Add
        swiglu_time
    )
    print("  Total per layer:      ", total_time / 1e6, "ms")
    print("  Total for 32 layers:  ", total_time * 32 / 1e6, "ms")
    print()
    print("Breakdown:")
    var matmul_total = (
        matmul_q_time + matmul_k_time + matmul_v_time + matmul_o_time +
        matmul_gate_time + matmul_up_time + matmul_down_time
    )
    print("  Matmul total:         ", matmul_total / 1e6, "ms (", matmul_total / total_time * 100, "%)")
    var attention_total = attention_time + rope_time * 2 + kv_cache_time
    print("  Attention total:      ", attention_total / 1e6, "ms (", attention_total / total_time * 100, "%)")
    var other_total = rms_norm_time * 2 + add_time * 2 + swiglu_time
    print("  Other (norm/add/etc): ", other_total / 1e6, "ms (", other_total / total_time * 100, "%)")
    print("=" * 70)
