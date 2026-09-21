# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/flash_attention_cpu.mojo
#
# Flash Attention CPU implementation with online softmax.
# Based on llama.cpp's ggml_compute_forward_flash_attn_ext_f16_one_chunk.
#
# Key algorithm (https://arxiv.org/pdf/2112.05682.pdf):
# - Online softmax: maintain running max (M) and running sum (S)
# - Rescale V accumulation when new max found: V *= exp(old_max - new_max)
# - Incremental V update: V += v * exp(score - current_max)
# - O(1) memory (no score storage)
#
# Supports:
# - Decode mode (M=1): single query token
# - Prefill mode (M>1): batch query tokens (each attends to all previous positions)

from ...tensor import Tensor, tensor_zeros
from std.math import exp, sqrt
from ..attention.kv_cache import KVCacheLayer
from ...cpu_features import detect_cpu_flags
from std.utils import StaticTuple

# NEON width for Float32
comptime SIMD_W = 8


def flash_attention_decode(
    q: Tensor[DType.float16, 1],
    cache: KVCacheLayer,
    kv_head: Int,
    start_pos: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 1]:
    """Flash Attention for decode (single query token).

    Args:
        q: Query vector [head_dim]
        cache: KV cache
        kv_head: KV head index (for GQA)
        start_pos: Current position (attend to [first, start_pos])
        head_dim: Head dimension
        scale: Attention scale (1/sqrt(head_dim))

    Returns:
        Output vector [head_dim]
    """
    var max_len = cache.max_len
    var first = cache.first_position()
    if first < 0:
        first = 0
    var seq = start_pos + 1
    var n_kv = seq - first

    # Online softmax state
    var M = Float32(-3.0e38)  # Running max
    var S = Float32(0.0)      # Running sum

    # Output accumulator (FP32 for numerical stability)
    var O = Tensor[DType.float32, 1](StaticTuple[Int, 1](head_dim))

    # Cache access helpers
    var quant = cache.is_quantized()
    var dense = cache.page_size == 0
    var k_ptr = cache.k.data()
    var v_ptr = cache.v.data()
    var k_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var v_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var q_ptr = q.data()

    # Process each KV position
    for ic in range(first, seq):
        # Compute Q·K score
        var score = Float32(0)
        var k_base = (kv_head * max_len + ic) * head_dim

        if quant:
            cache.get_k_row(kv_head, ic, k_row)
            # SIMD dot product
            var acc_vec = SIMD[DType.float32, SIMD_W](0)
            var d = 0
            while d + SIMD_W <= head_dim:
                var qv = q_ptr.unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
                var kv = k_row.data().unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
                acc_vec = acc_vec + qv * kv
                d += SIMD_W
            score = acc_vec.reduce_add()
            # Handle remainder
            while d < head_dim:
                score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(k_row.get(d))
                d += 1
        else:
            if dense:
                # SIMD dot product for dense cache
                var acc_vec = SIMD[DType.float32, SIMD_W](0)
                var d = 0
                while d + SIMD_W <= head_dim:
                    var qv = q_ptr.unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
                    var kv = k_ptr.unsafe_load[width=SIMD_W](offset=k_base + d).cast[DType.float32]()
                    acc_vec = acc_vec + qv * kv
                    d += SIMD_W
                score = acc_vec.reduce_add()
                while d < head_dim:
                    score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(
                        k_ptr.unsafe_load(offset=k_base + d)
                    )
                    d += 1
            else:
                # Paged cache (slower path)
                for d in range(head_dim):
                    score += Float32(q_ptr.unsafe_load(offset=d)) * cache.get_k(kv_head, ic, d)

        score *= scale

        # Online softmax update
        # ref: llama.cpp ops.cpp:8765-8800
        var Mold = M
        var ms = Float32(1.0)  # Rescale factor for existing O
        var vs = Float32(1.0)  # Weight for new V row

        if score > M:
            # New max found - rescale existing O
            M = score
            ms = exp(Mold - M)
            # Rescale O: O *= exp(old_max - new_max)
            for d in range(head_dim):
                O.set(d, Scalar[DType.float32](Float32(O.get(d)) * ms))
        else:
            # No new max - compute weight for new V
            vs = exp(score - M)

        # Accumulate V: O += v * exp(score - current_max)
        if quant:
            cache.get_v_row(kv_head, ic, v_row)
            for d in range(head_dim):
                O.set(d, Scalar[DType.float32](
                    Float32(O.get(d)) + vs * Float32(v_row.get(d))
                ))
        else:
            if dense:
                for d in range(head_dim):
                    O.set(d, Scalar[DType.float32](
                        Float32(O.get(d)) + vs * Float32(
                            v_ptr.unsafe_load(offset=(kv_head * max_len + ic) * head_dim + d)
                        )
                    ))
            else:
                for d in range(head_dim):
                    O.set(d, Scalar[DType.float32](
                        Float32(O.get(d)) + vs * cache.get_v(kv_head, ic, d)
                    ))

        # Update running sum: S = S*ms + vs
        S = S * ms + vs

    # Final normalization: O /= S
    var S_inv = Float32(0.0)
    if S != 0.0:
        S_inv = Float32(1.0) / S
    var out = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    for d in range(head_dim):
        out.set(d, Scalar[DType.float16](Float32(O.get(d)) * S_inv))

    return out


def flash_attention_prefill(
    q: Tensor[DType.float16, 2],  # [T, head_dim] - all query tokens for one head
    cache: KVCacheLayer,
    kv_head: Int,
    start_pos: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 2]:
    """Flash Attention for prefill (batch query tokens).

    Each query token at position t attends to all positions [first, start_pos + t].

    Args:
        q: Query vectors [T, head_dim] for one head
        cache: KV cache
        kv_head: KV head index (for GQA)
        start_pos: Starting position (tokens are at [start_pos, start_pos + T))
        head_dim: Head dimension
        scale: Attention scale (1/sqrt(head_dim))

    Returns:
        Output vectors [T, head_dim]
    """
    var T = q.shape()[0]
    var max_len = cache.max_len
    var first = cache.first_position()
    if first < 0:
        first = 0

    var quant = cache.is_quantized()
    var dense = cache.page_size == 0
    var k_ptr = cache.k.data()
    var v_ptr = cache.v.data()
    var k_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var v_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))

    var out = Tensor[DType.float16, 2](StaticTuple[Int, 2](T, head_dim))

    # Process each query token
    for t in range(T):
        var pos = start_pos + t
        var seq = pos + 1
        var n_kv = seq - first

        # Online softmax state for this query
        var M = Float32(-3.0e38)
        var S = Float32(0.0)
        var O = Tensor[DType.float32, 1](StaticTuple[Int, 1](head_dim))
        var q_ptr = q.data().unsafe_offset(t * head_dim)

        # Attend to all KV positions up to this query position
        for ic in range(first, seq):
            # Compute Q·K score
            var score = Float32(0)
            var k_base = (kv_head * max_len + ic) * head_dim

            if quant:
                cache.get_k_row(kv_head, ic, k_row)
                var acc_vec = SIMD[DType.float32, SIMD_W](0)
                var d = 0
                while d + SIMD_W <= head_dim:
                    var qv = q_ptr.unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
                    var kv = k_row.data().unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
                    acc_vec = acc_vec + qv * kv
                    d += SIMD_W
                score = acc_vec.reduce_add()
                while d < head_dim:
                    score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(k_row.get(d))
                    d += 1
            else:
                if dense:
                    var acc_vec = SIMD[DType.float32, SIMD_W](0)
                    var d = 0
                    while d + SIMD_W <= head_dim:
                        var qv = q_ptr.unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
                        var kv = k_ptr.unsafe_load[width=SIMD_W](offset=k_base + d).cast[DType.float32]()
                        acc_vec = acc_vec + qv * kv
                        d += SIMD_W
                    score = acc_vec.reduce_add()
                    while d < head_dim:
                        score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(
                            k_ptr.unsafe_load(offset=k_base + d)
                        )
                        d += 1
                else:
                    for d in range(head_dim):
                        score += Float32(q_ptr.unsafe_load(offset=d)) * cache.get_k(kv_head, ic, d)

            score *= scale

            # Online softmax update
            var Mold = M
            var ms = Float32(1.0)
            var vs = Float32(1.0)

            if score > M:
                M = score
                ms = exp(Mold - M)
                for d in range(head_dim):
                    O.set(d, Scalar[DType.float32](Float32(O.get(d)) * ms))
            else:
                vs = exp(score - M)

            # Accumulate V
            if quant:
                cache.get_v_row(kv_head, ic, v_row)
                for d in range(head_dim):
                    O.set(d, Scalar[DType.float32](
                        Float32(O.get(d)) + vs * Float32(v_row.get(d))
                    ))
            else:
                if dense:
                    for d in range(head_dim):
                        O.set(d, Scalar[DType.float32](
                            Float32(O.get(d)) + vs * Float32(
                                v_ptr.unsafe_load(offset=(kv_head * max_len + ic) * head_dim + d)
                            )
                        ))
                else:
                    for d in range(head_dim):
                        O.set(d, Scalar[DType.float32](
                            Float32(O.get(d)) + vs * cache.get_v(kv_head, ic, d)
                        ))

            S = S * ms + vs

        # Normalize and store
        var S_inv = Float32(0.0)
        if S != 0.0:
            S_inv = Float32(1.0) / S
        for d in range(head_dim):
            out.set(t * head_dim + d, Scalar[DType.float16](Float32(O.get(d)) * S_inv))

    return out
