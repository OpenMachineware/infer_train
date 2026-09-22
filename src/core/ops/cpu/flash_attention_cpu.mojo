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
# Chunking (llama.cpp ops.cpp:9261-9291):
# - KV sequence is split into chunks for better cache locality
# - Each chunk produces partial results: [M, S, O]
# - Chunks are merged using online softmax reduction
#
# Supports:
# - Decode mode (M=1): single query token
# - Prefill mode (M>1): batch query tokens (each attends to all previous positions)

from ...tensor import Tensor, tensor_zeros
from std.math import exp, sqrt
from ..attention.kv_cache import KVCacheLayer
from ...cpu_features import detect_cpu_flags
from std.utils import StaticTuple

# Chunk size for KV processing (tunable, llama.cpp uses ~256 for typical scenarios)
comptime KV_CHUNK_SIZE = 256

# NEON width for Float32
comptime SIMD_W = 8


@always_inline
def _dot_product_qk_neon(
    q_ptr: UnsafePointer[Scalar[DType.float16]],
    k_ptr: UnsafePointer[Scalar[DType.float16]],
    head_dim: Int,
) -> Float32:
    """NEON SIMD dot product for Q·K^T."""
    var acc_vec = SIMD[DType.float32, SIMD_W](0)
    var d = 0
    while d + SIMD_W <= head_dim:
        var qv = q_ptr.unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
        var kv = k_ptr.unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
        acc_vec = acc_vec + qv * kv
        d += SIMD_W
    var acc = acc_vec.reduce_add()
    while d < head_dim:
        acc += Float32(q_ptr.unsafe_load(offset=d)) * Float32(k_ptr.unsafe_load(offset=d))
        d += 1
    return acc


@always_inline
def _dot_product_qk_scalar(
    q_ptr: UnsafePointer[Scalar[DType.float16]],
    k_ptr: UnsafePointer[Scalar[DType.float16]],
    head_dim: Int,
) -> Float32:
    """Scalar dot product fallback."""
    var acc = Float32(0)
    for d in range(head_dim):
        acc += Float32(q_ptr.unsafe_load(offset=d)) * Float32(k_ptr.unsafe_load(offset=d))
    return acc


def flash_attention_decode(
    q: Tensor[DType.float16, 1],
    cache: KVCacheLayer,
    kv_head: Int,
    start_pos: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 1]:
    """Flash Attention for decode (single query token).

    Uses chunked KV processing for better cache locality on long sequences.
    Dynamic dispatch: uses NEON SIMD if available, scalar fallback otherwise.
    """
    var has_neon = detect_cpu_flags().has_neon()
    if has_neon:
        return _flash_attention_decode_neon_chunked(q, cache, kv_head, start_pos, head_dim, scale)
    else:
        return _flash_attention_decode_scalar_chunked(q, cache, kv_head, start_pos, head_dim, scale)


# ============================================================================
# Chunked KV processing with online softmax reduction
# ============================================================================

@always_inline
def _merge_online_softmax(
    M1: Float32, S1: Float32, O1: Tensor[DType.float32, 1],
    M2: Float32, S2: Float32, O2: Tensor[DType.float32, 1],
    head_dim: Int,
) -> Tuple[Float32, Float32]:
    """Merge two online softmax states (llama.cpp style).

    Args:
        M1, S1, O1: First partial result (O1 is unnormalized)
        M2, S2, O2: Second partial result (O2 is unnormalized)

    Returns:
        Updated (M, S) and O1 is modified in-place

    Algorithm (llama.cpp ops.cpp:9189-9199):
        M_new = max(M1, M2)
        scale1 = exp(M1 - M_new)  # rescale O1 (NOT multiply by S1!)
        scale2 = exp(M2 - M_new)  # rescale O2
        O = O1 * scale1 + O2 * scale2
        S = S1 * scale1 + S2 * scale2
    """
    var M_new = M1 if M1 > M2 else M2

    # Rescale and merge O (unnormalized)
    var scale1 = exp(M1 - M_new)
    var scale2 = exp(M2 - M_new)

    for d in range(head_dim):
        O1.set(d, O1.get(d) * scale1 + O2.get(d) * scale2)

    var S_new = S1 * scale1 + S2 * scale2

    return (M_new, S_new)


def _flash_attention_decode_neon_chunked(
    q: Tensor[DType.float16, 1],
    cache: KVCacheLayer,
    kv_head: Int,
    start_pos: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 1]:
    """Flash Attention NEON SIMD with chunked KV processing.

    Splits KV sequence into chunks for better cache locality.
    Each chunk produces partial [M, S, O] which are merged at the end.
    """
    var max_len = cache.max_len
    var first = cache.first_position()
    if first < 0:
        first = 0
    var seq = start_pos + 1
    var n_kv = seq - first

    # If KV is small, process directly (no chunking overhead)
    if n_kv <= KV_CHUNK_SIZE:
        return _flash_attention_decode_neon(q, cache, kv_head, start_pos, head_dim, scale)

    # Chunked processing
    var quant = cache.is_quantized()
    var dense = cache.page_size == 0
    var k_ptr = cache.k.data()
    var v_ptr = cache.v.data()
    var k_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var v_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var q_ptr = q.data()

    # Global online softmax state
    var M_global = Float32(-3.0e38)
    var S_global = Float32(0.0)
    var O_global = Tensor[DType.float32, 1](StaticTuple[Int, 1](head_dim))

    # Chunk partial state
    var M_chunk = Float32(-3.0e38)
    var S_chunk = Float32(0.0)
    var O_chunk = Tensor[DType.float32, 1](StaticTuple[Int, 1](head_dim))

    # Process chunks
    var ic0 = first
    while ic0 < seq:
        var ic1 = ic0 + KV_CHUNK_SIZE
        if ic1 > seq:
            ic1 = seq

        # Reset chunk state
        M_chunk = Float32(-3.0e38)
        S_chunk = Float32(0.0)
        for d in range(head_dim):
            O_chunk.set(d, 0.0)

        # Process one chunk
        for ic in range(ic0, ic1):
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
                    # Paged: use row buffer for SIMD (not element-wise get_k)
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

            score *= scale

            # Online softmax update (chunk-local)
            var Mold = M_chunk
            var ms = Float32(1.0)
            var vs = Float32(1.0)

            if score > M_chunk:
                M_chunk = score
                ms = exp(Mold - M_chunk)

            vs = exp(score - M_chunk)

            # Update S and O
            S_chunk = S_chunk * ms + vs

            # Get V row
            if quant:
                cache.get_v_row(kv_head, ic, v_row)
                for d in range(head_dim):
                    O_chunk.set(d, O_chunk.get(d) * ms + vs * Float32(v_row.get(d)))
            else:
                if dense:
                    var d = 0
                    while d + SIMD_W <= head_dim:
                        var ov = O_chunk.data().unsafe_load[width=SIMD_W](offset=d)
                        var vv = v_ptr.unsafe_load[width=SIMD_W](offset=(kv_head * max_len + ic) * head_dim + d).cast[DType.float32]()
                        ov = ov * ms + vv * vs
                        O_chunk.data().unsafe_store(d, ov)
                        d += SIMD_W
                    while d < head_dim:
                        var ov = O_chunk.get(d)
                        var vv = Float32(v_ptr.unsafe_load(offset=(kv_head * max_len + ic) * head_dim + d))
                        O_chunk.set(d, ov * ms + vv * vs)
                        d += 1
                else:
                    # Paged: use row buffer for SIMD
                    cache.get_v_row(kv_head, ic, v_row)
                    var d = 0
                    while d + SIMD_W <= head_dim:
                        var ov = O_chunk.data().unsafe_load[width=SIMD_W](offset=d)
                        var vv = v_row.data().unsafe_load[width=SIMD_W](offset=d).cast[DType.float32]()
                        ov = ov * ms + vv * vs
                        O_chunk.data().unsafe_store(d, ov)
                        d += SIMD_W
                    while d < head_dim:
                        O_chunk.set(d, O_chunk.get(d) * ms + vs * Float32(v_row.get(d)))
                        d += 1

        # Merge chunk partial into global state
        (M_global, S_global) = _merge_online_softmax(M_global, S_global, O_global, M_chunk, S_chunk, O_chunk, head_dim)

        ic0 = ic1

    # Final normalization
    var S_inv = Float32(1.0) / S_global
    var out = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    for d in range(head_dim):
        out.set(d, Scalar[DType.float16](O_global.get(d) * S_inv))

    return out


def _flash_attention_decode_neon(
    q: Tensor[DType.float16, 1],
    cache: KVCacheLayer,
    kv_head: Int,
    start_pos: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 1]:
    """Flash Attention NEON SIMD implementation.

    Args:
        q: Query vector [head_dim]
        cache: KV cache
        kv_head: KV head index (for GQA)
        start_pos: Current position
        head_dim: Head dimension
        scale: Attention scale

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
                # Paged: use row buffer
                cache.get_v_row(kv_head, ic, v_row)
                for d in range(head_dim):
                    O.set(d, Scalar[DType.float32](
                        Float32(O.get(d)) + vs * Float32(v_row.get(d))
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


def _flash_attention_decode_scalar_chunked(
    q: Tensor[DType.float16, 1],
    cache: KVCacheLayer,
    kv_head: Int,
    start_pos: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 1]:
    """Scalar fallback with chunked KV processing."""
    var max_len = cache.max_len
    var first = cache.first_position()
    if first < 0:
        first = 0
    var seq = start_pos + 1
    var n_kv = seq - first

    # If KV is small, process directly
    if n_kv <= KV_CHUNK_SIZE:
        return _flash_attention_decode_scalar(q, cache, kv_head, start_pos, head_dim, scale)

    # Chunked processing
    var quant = cache.is_quantized()
    var dense = cache.page_size == 0
    var k_ptr = cache.k.data()
    var v_ptr = cache.v.data()
    var k_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var v_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var q_ptr = q.data()

    # Global online softmax state
    var M_global = Float32(-3.0e38)
    var S_global = Float32(0.0)
    var O_global = Tensor[DType.float32, 1](StaticTuple[Int, 1](head_dim))

    # Chunk partial state
    var M_chunk = Float32(-3.0e38)
    var S_chunk = Float32(0.0)
    var O_chunk = Tensor[DType.float32, 1](StaticTuple[Int, 1](head_dim))

    # Process chunks
    var ic0 = first
    while ic0 < seq:
        var ic1 = ic0 + KV_CHUNK_SIZE
        if ic1 > seq:
            ic1 = seq

        # Reset chunk state
        M_chunk = Float32(-3.0e38)
        S_chunk = Float32(0.0)
        for d in range(head_dim):
            O_chunk.set(d, 0.0)

        # Process one chunk
        for ic in range(ic0, ic1):
            # Compute Q·K score
            var score = Float32(0)

            if quant:
                cache.get_k_row(kv_head, ic, k_row)
                for d in range(head_dim):
                    score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(k_row.get(d))
            else:
                if dense:
                    for d in range(head_dim):
                        score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(
                            k_ptr.unsafe_load(offset=(kv_head * max_len + ic) * head_dim + d)
                        )
                else:
                    # Paged: use row buffer
                    cache.get_k_row(kv_head, ic, k_row)
                    for d in range(head_dim):
                        score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(k_row.get(d))

            score *= scale

            # Online softmax update (chunk-local)
            var Mold = M_chunk
            var ms = Float32(1.0)
            var vs = Float32(1.0)

            if score > M_chunk:
                M_chunk = score
                ms = exp(Mold - M_chunk)

            vs = exp(score - M_chunk)
            S_chunk = S_chunk * ms + vs

            # Get V row
            if quant:
                cache.get_v_row(kv_head, ic, v_row)
                for d in range(head_dim):
                    O_chunk.set(d, O_chunk.get(d) * ms + vs * Float32(v_row.get(d)))
            else:
                if dense:
                    for d in range(head_dim):
                        var ov = O_chunk.get(d)
                        var vv = Float32(v_ptr.unsafe_load(offset=(kv_head * max_len + ic) * head_dim + d))
                        O_chunk.set(d, ov * ms + vv * vs)
                else:
                    # Paged: use row buffer
                    cache.get_v_row(kv_head, ic, v_row)
                    for d in range(head_dim):
                        O_chunk.set(d, O_chunk.get(d) * ms + vs * Float32(v_row.get(d)))

        # Merge chunk partial into global state
        (M_global, S_global) = _merge_online_softmax(M_global, S_global, O_global, M_chunk, S_chunk, O_chunk, head_dim)

        ic0 = ic1

    # Final normalization
    var S_inv = Float32(1.0) / S_global
    var out = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    for d in range(head_dim):
        out.set(d, Scalar[DType.float16](O_global.get(d) * S_inv))

    return out


def _flash_attention_decode_scalar(
    q: Tensor[DType.float16, 1],
    cache: KVCacheLayer,
    kv_head: Int,
    start_pos: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 1]:
    """Flash Attention scalar fallback (no SIMD)."""
    var max_len = cache.max_len
    var first = cache.first_position()
    if first < 0:
        first = 0
    var seq = start_pos + 1

    var M = Float32(-3.0e38)
    var S = Float32(0.0)
    var O = Tensor[DType.float32, 1](StaticTuple[Int, 1](head_dim))

    var quant = cache.is_quantized()
    var dense = cache.page_size == 0
    var k_ptr = cache.k.data()
    var v_ptr = cache.v.data()
    var k_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var v_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var q_ptr = q.data()

    for ic in range(first, seq):
        var score = Float32(0)
        var k_base = (kv_head * max_len + ic) * head_dim

        if quant:
            cache.get_k_row(kv_head, ic, k_row)
            for d in range(head_dim):
                score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(k_row.get(d))
        else:
            if dense:
                for d in range(head_dim):
                    score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(
                        k_ptr.unsafe_load(offset=k_base + d)
                    )
            else:
                # Paged: use row buffer
                cache.get_k_row(kv_head, ic, k_row)
                for d in range(head_dim):
                    score += Float32(q_ptr.unsafe_load(offset=d)) * Float32(k_row.get(d))

        score *= scale

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
                # Paged: use row buffer
                cache.get_v_row(kv_head, ic, v_row)
                for d in range(head_dim):
                    O.set(d, Scalar[DType.float32](
                        Float32(O.get(d)) + vs * Float32(v_row.get(d))
                    ))

        S = S * ms + vs

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
                    # Paged: use row buffer for SIMD
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
                    # Paged: use row buffer
                    cache.get_v_row(kv_head, ic, v_row)
                    for d in range(head_dim):
                        O.set(d, Scalar[DType.float32](
                            Float32(O.get(d)) + vs * Float32(v_row.get(d))
                        ))

            S = S * ms + vs

        # Normalize and store
        var S_inv = Float32(0.0)
        if S != 0.0:
            S_inv = Float32(1.0) / S
        for d in range(head_dim):
            out.set(t * head_dim + d, Scalar[DType.float16](Float32(O.get(d)) * S_inv))

    return out
