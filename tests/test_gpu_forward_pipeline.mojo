# Test GPU forward pipeline
#
# This test verifies that all GPU pipeline kernels work correctly
# in a complete forward pass.

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
from std.collections.optional import Optional


def _create_test_tensor(rows: Int, cols: Int) -> Tensor[DType.float16, 2]:
    """Create a test tensor with small random values."""
    var t = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](rows, cols))
    for i in range(rows):
        for j in range(cols):
            # Use deterministic pattern instead of random
            var val = Float32(0.1) * Float32((i * cols + j) % 10)
            t.set(i * cols + j, Scalar[DType.float16](val))
    return t


def _create_test_tensor_1d(dim: Int) -> Tensor[DType.float16, 1]:
    """Create a 1D test tensor."""
    var t = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](dim))
    for i in range(dim):
        var val = Float32(0.1) * Float32(i % 10)
        t.set(i, Scalar[DType.float16](val))
    return t


def test_gpu_pipeline():
    """Test complete GPU pipeline with synthetic data."""
    print("Testing GPU pipeline kernels...")

    try:
        var ctx = get_gpu_context()
        var n_heads = 8
        var n_kv_heads = 8
        var head_dim = 64
        var hidden = n_heads * head_dim  # 512
        var ffn_dim = 2048
        var max_len = 128

        # Create test inputs
        print("  Creating test tensors...")
        var x_cpu = _create_test_tensor(1, hidden)
        var norm_w_cpu = _create_test_tensor_1d(hidden)

        # Upload to GPU
        print("  1. Uploading tensors to GPU...")
        var x_buf = upload[DType.float16, 2](ctx, x_cpu)
        var norm_w_buf = upload[DType.float16, 1](ctx, norm_w_cpu)

        # Test RMS Norm
        print("  2. Testing RMS Norm...")
        var normed = rms_norm_weight_gpu_pipeline(ctx, x_buf, norm_w_buf, 1, hidden)

        # Test Matmul (simulate Q projection)
        print("  3. Testing Matmul (Q projection)...")
        var q_w_cpu = _create_test_tensor(n_heads * head_dim, hidden)
        var q_w_buf = upload[DType.float16, 2](ctx, q_w_cpu)
        var q_buf = matmul_fp16_gpu_pipeline(normed, q_w_buf, ctx, 1, hidden, n_heads * head_dim)

        # Test RoPE
        print("  4. Testing RoPE...")
        q_buf = rope_gpu_pipeline(ctx, q_buf, n_heads, 1, head_dim, 0)

        # Test KV Cache
        print("  5. Testing KV Cache...")
        var kv_cache = KVCacheLayerGPU(ctx, n_kv_heads, max_len, head_dim)

        # Create K and V for position 0
        var k_w_cpu = _create_test_tensor(n_kv_heads * head_dim, hidden)
        var v_w_cpu = _create_test_tensor(n_kv_heads * head_dim, hidden)
        var k_w_buf = upload[DType.float16, 2](ctx, k_w_cpu)
        var v_w_buf = upload[DType.float16, 2](ctx, v_w_cpu)
        var k_buf = matmul_fp16_gpu_pipeline(normed, k_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)
        var v_buf = matmul_fp16_gpu_pipeline(normed, v_w_buf, ctx, 1, hidden, n_kv_heads * head_dim)

        # Update KV cache
        kv_cache.update(ctx, k_buf, v_buf, 0)
        print("     KV cache updated at position 0, filled_len = ", kv_cache.filled_len())

        # Test Attention
        print("  6. Testing Attention...")
        var attn_out = attention_decode_gpu_pipeline(
            ctx, q_buf, kv_cache.get_k_buffer(), kv_cache.get_v_buffer(),
            n_heads, head_dim, 1
        )

        # Test Add (residual)
        print("  7. Testing Add (residual)...")
        var x_after_attn = add_gpu_pipeline(ctx, x_buf, attn_out, hidden)

        # Test FFN (Gate + Up projections)
        print("  8. Testing FFN...")
        var gate_w_cpu = _create_test_tensor(ffn_dim, hidden)
        var up_w_cpu = _create_test_tensor(ffn_dim, hidden)
        var gate_w_buf = upload[DType.float16, 2](ctx, gate_w_cpu)
        var up_w_buf = upload[DType.float16, 2](ctx, up_w_cpu)
        var gate_buf = matmul_fp16_gpu_pipeline(x_after_attn, gate_w_buf, ctx, 1, hidden, ffn_dim)
        var up_buf = matmul_fp16_gpu_pipeline(x_after_attn, up_w_buf, ctx, 1, hidden, ffn_dim)

        # Test SwiGLU
        print("  9. Testing SwiGLU...")
        var hidden_ffn = swiglu_gpu_pipeline(ctx, gate_buf, up_buf, ffn_dim)

        # Test Down projection
        print("  10. Testing Down projection...")
        var down_w_cpu = _create_test_tensor(hidden, ffn_dim)
        var down_w_buf = upload[DType.float16, 2](ctx, down_w_cpu)
        var ffn_out = matmul_fp16_gpu_pipeline(hidden_ffn, down_w_buf, ctx, 1, ffn_dim, hidden)

        # Final residual add
        print("  11. Testing final residual add...")
        var x_out = add_gpu_pipeline(ctx, x_after_attn, ffn_out, hidden)

        # Download result
        print("  12. Downloading result...")
        var result = download2[DType.float16](ctx, x_out, StaticTuple[Int, 2](1, hidden))
        ctx.synchronize()

        print("✓ GPU pipeline test passed!")
        var sh = result.shape()
        print("  Output shape: (", sh[0], ", ", sh[1], ")")
        print("  Output [0]:", Float32(result.get(0)))

    except:
        print("✗ GPU pipeline test failed with exception")


def main():
    test_gpu_pipeline()


from src.core.ops.gpu.gpu_runtime import upload, download2
from std.collections.optional import Optional
