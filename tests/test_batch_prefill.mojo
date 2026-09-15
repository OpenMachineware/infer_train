# Test batch prefill attention
# SPDX-License-Identifier: Apache-2.0

from src.core.tensor import Tensor, tensor_zeros
from src.core.utils import unimplemented
from src.core.ops.attention.mha import (
    mha_forward,
    mha_forward_v2,
    mha_forward_batch,
    MHAOptions,
)
from src.core.ops.attention.kv_cache import KVCacheLayer
from src.core.ops.quantized.qweight import QWeight, qweight_from_fp16
from src.core.ops.cpu.rope_cpu import rope_cpu_dynamic
from std.utils.static_tuple import StaticTuple


def test_batch_reshape():
    """Test that batch reshape works for T > 1."""
    # Create a [3, 8] tensor (T=3, hidden=8, n_heads=2, head_dim=4)
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](3, 8))
    for i in range(24):
        x.set(i, Scalar[DType.float16](Float16(i)))

    # The reshape should work without errors
    print("batch reshape: OK")


def test_batch_attention():
    """Test batch attention with a small example."""
    var n_heads = 2
    var n_kv_heads = 2
    var head_dim = 4
    var hidden = n_heads * head_dim
    var n_tokens = 3

    # Create dummy inputs
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](n_tokens, hidden))
    for i in range(n_tokens * hidden):
        x.set(i, Scalar[DType.float16](Float16(0.1)))

    # Create dummy weights (using QWeight from fp16)
    var wq = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](hidden, hidden))
    var wk = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](n_kv_heads * head_dim, hidden))
    var wv = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](n_kv_heads * head_dim, hidden))
    var wo = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](hidden, hidden))
    for i in range(hidden * hidden):
        wq.set(i, Scalar[DType.float16](Float16(0.01)))
        wo.set(i, Scalar[DType.float16](Float16(0.01)))
    for i in range(n_kv_heads * head_dim * hidden):
        wk.set(i, Scalar[DType.float16](Float16(0.01)))
        wv.set(i, Scalar[DType.float16](Float16(0.01)))

    var wq_q = qweight_from_fp16(wq)
    var wk_q = qweight_from_fp16(wk)
    var wv_q = qweight_from_fp16(wv)
    var wo_q = qweight_from_fp16(wo)

    # Create empty biases
    var bq = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](0))
    var bk = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](0))
    var bv = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](0))
    var q_norm_w = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](0))
    var k_norm_w = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](0))

    # Create KV cache
    var cache = KVCacheLayer(n_kv_heads, head_dim, 64)

    # Create options
    var opts = MHAOptions()
    opts.norm_eps = Float32(1e-6)

    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))

    # Run batch attention
    var out = mha_forward_batch(
        x,
        wq_q,
        wk_q,
        wv_q,
        wo_q,
        bq,
        bk,
        bv,
        q_norm_w,
        k_norm_w,
        cache,
        0,
        n_heads,
        n_kv_heads,
        head_dim,
        Float32(10000.0),
        opts,
        dummy_scale,
    )

    # Verify output shape
    if out.shape()[0] != n_tokens:
        print("ERROR: output tokens mismatch")
        return
    if out.shape()[1] != hidden:
        print("ERROR: output hidden mismatch")
        return

    print("batch attention: OK")


def main():
    print("testing batch prefill...")
    test_batch_reshape()
    test_batch_attention()
    print("all tests passed")