# core/ops/attention/mha.mojo
#
# Multi-head attention with GQA and a preallocated KV cache (M3).
#
# The kernel is assembled from the M1/M3 base kernels exactly as the spec
# asks: matmul (QKV projection + output projection), add_row (biases),
# rope (positional info on Q and K), softmax (inline, f32 for stability),
# and the cache-backed score/output accumulation.
#
# Layout notes (Qwen2 / GGUF):
#   * weights are GGUF layout, so q = x @ wq + bq needs no transposes;
#   * Q has n_heads heads of head_dim, K/V have n_kv_heads (GQA: query head
#     h reads KV head h * n_kv_heads // n_heads);
#   * the current token's position is start_pos; scores only attend to
#     positions <= start_pos (causal) - the cache is only filled that far;
#   * scale = 1/sqrt(head_dim) (HF Qwen2: no learned attn scale).
#
# This is the single-token decode path (T == 1); the prefill feeds one
# token at a time, which is trivially correct and fast enough for M3.
# Flash attention / batching land in M5.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ..cpu.matmul_cpu import (
    matmul_weight_cpu,
    matmul_weight_cpu_backward,
    matmul_weight_cpu_threaded,
    matmul_weight_3_threaded,
)
from ..quantized.qweight import QWeight, qweight_from_fp16
from ..cpu.add_cpu import add_row_cpu
from ..cpu.rope_cpu import (
    rope_cpu_dynamic,
    rope_cpu_backward,
    rope_cpu_rot,
)
from .kv_cache import KVCacheLayer
from std.utils.static_tuple import StaticTuple
from std.math import exp, sqrt


def _qkv_reshape[dtype: DType](
    x: Tensor[dtype, 2], n_heads: Int, head_dim: Int
) -> Tensor[dtype, 3]:
    """View [T, n_heads*head_dim] as [n_heads, T, head_dim] (T must be 1)."""
    var n_tokens = x.shape()[0]
    if n_tokens != 1:
        unimplemented("mha: single-token decode only (T==1)")
    return Tensor[dtype, 3](
        StaticTuple[Int, 3](n_heads, n_tokens, head_dim),
        x.data(),
        x.device(),
    )


def _flat_view[dtype: DType](
    x: Tensor[dtype, 3], hidden: Int
) -> Tensor[dtype, 2]:
    """View [n_heads, 1, head_dim] as [1, hidden]."""
    if x.shape()[1] != 1 or x.numel() != hidden:
        unimplemented("mha: bad reshape to flat")
    return Tensor[dtype, 2](
        StaticTuple[Int, 2](1, hidden), x.data(), x.device()
    )


struct MHAOptions(Copyable, Movable, ImplicitlyCopyable):
    """Variant switches for the generalized single-token MHA (M7).

    * `q_norm`/`k_norm`: per-head RMSNorm on Q/K with the passed weight
      vectors (hunyuan-dense + qwen35 full-attention layers);
    * `norm_before_rope`: qwen35 normalizes *before* RoPE, hunyuan-dense
      *after* RoPE;
    * `gate`: multiply the attention output by sigmoid(gate) before the
      output projection (qwen35 fused Q+gate projection);
    * `n_rot`: number of rotated dims (0 = all pairs, qwen35 MRoPE uses
      64 of 256).
    """

    var q_norm: Bool
    var k_norm: Bool
    var norm_before_rope: Bool
    var gate: Bool
    var n_rot: Int
    var norm_eps: Float32

    def __init__(out self):
        self.q_norm = False
        self.k_norm = False
        self.norm_before_rope = False
        self.gate = False
        self.n_rot = 0
        self.norm_eps = Float32(1e-6)


def mha_forward(
    x: Tensor[DType.float16, 2],
    wq: Tensor[DType.float16, 2],
    wk: Tensor[DType.float16, 2],
    wv: Tensor[DType.float16, 2],
    wo: Tensor[DType.float16, 2],
    bq: Tensor[DType.float16, 1],
    bk: Tensor[DType.float16, 1],
    bv: Tensor[DType.float16, 1],
    mut cache: KVCacheLayer,
    start_pos: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    rope_theta: Float32,
) -> Tensor[DType.float16, 2]:
    """M3/M7: the classic Qwen2 path (no Q/K norms, full RoPE, no gate).

    M11: fp16-only (the cached decode path is fp16 by design); the fp16
    weights are wrapped as `QWeight`s and projected through the same
    Q4-resident path as the quantized models.
    """
    var opts = MHAOptions()
    var ds = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    return mha_forward_v2(
        x, qweight_from_fp16(wq), qweight_from_fp16(wk),
        qweight_from_fp16(wv), qweight_from_fp16(wo), bq, bk, bv,
        Tensor[DType.float16, 1](StaticTuple[Int, 1](0)),
        Tensor[DType.float16, 1](StaticTuple[Int, 1](0)), cache, start_pos, n_heads,
        n_kv_heads, head_dim, rope_theta, opts, ds,
    )


def mha_forward_v2(
    x: Tensor[DType.float16, 2],
    wq: QWeight,
    wk: QWeight,
    wv: QWeight,
    wo: QWeight,
    bq: Tensor[DType.float16, 1],
    bk: Tensor[DType.float16, 1],
    bv: Tensor[DType.float16, 1],
    q_norm_w: Tensor[DType.float16, 1],
    k_norm_w: Tensor[DType.float16, 1],
    mut cache: KVCacheLayer,
    start_pos: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    rope_theta: Float32,
    opts: MHAOptions,
    dummy_scale: Tensor[DType.float16, 1],
) -> Tensor[DType.float16, 2]:
    """Generalized single-token MHA (M7), Q4-resident (M11).

    Qwen2 (defaults) -> identical to `mha_forward`.  hunyuan-dense and
    qwen35 full-attention layers pass the q/k norm weights and variant
    switches.  K/V are stored into the cache *after* RoPE/normalization.

    The Q/K/V/O projections are `QWeight`s: quantized weights are
    projected through the fused per-block-dequant matmul (`QWeight.proj`
    -> `matmul_quantized_cpu`, the dequantized values never leave the
    kernel scope), materialized fp16 weights through the threaded
    weight-major kernel.  Everything else (bias, RoPE, per-head Q/K norm,
    gate, KV cache, softmax) is unchanged from the M7 fp16 path.
    """
    var q_flat = wq.proj(x, dummy_scale)
    var k_flat = wk.proj(x, dummy_scale)
    var v_flat = wv.proj(x, dummy_scale)
    if bq.numel() > 0:
        q_flat = add_row_cpu[DType.float16](q_flat, bq)
    if bk.numel() > 0:
        k_flat = add_row_cpu[DType.float16](k_flat, bk)
    if bv.numel() > 0:
        v_flat = add_row_cpu[DType.float16](v_flat, bv)
    # qwen35 fuses Q+gate into wq ([q0,g0,q1,g1,...] per head): gather the
    # query part explicitly instead of the plain reshape.
    var q3 = tensor_zeros[DType.float16, 3](
        StaticTuple[Int, 3](n_heads, 1, head_dim)
    )
    if opts.gate:
        for h in range(n_heads):
            for d in range(head_dim):
                q3.set(h * head_dim + d, q_flat.get(h * 2 * head_dim + d))
    else:
        q3 = _qkv_reshape[DType.float16](q_flat, n_heads, head_dim)
    var k3 = _qkv_reshape[DType.float16](k_flat, n_kv_heads, head_dim)
    var v3 = _qkv_reshape[DType.float16](v_flat, n_kv_heads, head_dim)

    if opts.q_norm and opts.norm_before_rope:
        q3 = rms_norm_heads[DType.float16](q3, q_norm_w, opts.norm_eps)
    if opts.k_norm and opts.norm_before_rope:
        k3 = rms_norm_heads[DType.float16](k3, k_norm_w, opts.norm_eps)

    var q_rot = q3
    var k_rot = k3
    if opts.n_rot > 0 and opts.n_rot < head_dim:
        q_rot = rope_cpu_rot[DType.float16](q3, start_pos, rope_theta, opts.n_rot)
        k_rot = rope_cpu_rot[DType.float16](k3, start_pos, rope_theta, opts.n_rot)
    else:
        q_rot = rope_cpu_dynamic[DType.float16](q3, start_pos, rope_theta)
        k_rot = rope_cpu_dynamic[DType.float16](k3, start_pos, rope_theta)

    if opts.q_norm and not opts.norm_before_rope:
        q_rot = rms_norm_heads[DType.float16](q_rot, q_norm_w, opts.norm_eps)
    if opts.k_norm and not opts.norm_before_rope:
        k_rot = rms_norm_heads[DType.float16](k_rot, k_norm_w, opts.norm_eps)

    # store K/V into the cache (dense or paged; M7 1.4)
    var max_len = cache.max_len
    if start_pos < 0 or start_pos >= max_len:
        unimplemented("mha: position beyond KV cache capacity")
    for h in range(n_kv_heads):
        for d in range(head_dim):
            cache.set_kv(
                h, start_pos, d, Float32(k_rot.get(h * head_dim + d)),
                Float32(v3.get(h * head_dim + d)),
            )
    if start_pos + 1 > cache.filled:
        cache.filled = start_pos + 1
    var seq = start_pos + 1
    # sliding window (M7): skip positions older than the window
    var first = cache.first_position()
    if first < 0:
        first = 0
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var out = tensor_zeros[DType.float16, 3](
        StaticTuple[Int, 3](n_heads, 1, head_dim)
    )

    # M7 perf: dense caches are read through hoisted raw pointers; paged
    # caches go through the (slower but allocation-free) accessors.
    var k_ptr = cache.k.data()
    var v_ptr = cache.v.data()
    var dense = cache.page_size == 0
    for h in range(n_heads):
        var kv_head = h * n_kv_heads // n_heads
        var scores = List[Float32]()
        for t in range(first, seq):
            var acc = Float32(0)
            var k_base = (kv_head * max_len + t) * head_dim
            for d in range(head_dim):
                var qv = Float32(q_rot.get(h * head_dim + d))
                var kv = Float32(0)
                if dense:
                    kv = Float32(
                        k_ptr.unsafe_load[width=1](offset=k_base + d)
                    )
                else:
                    kv = cache.get_k(kv_head, t, d)
                acc += qv * kv
            scores.append(acc * scale)
        var n_scores = seq - first
        var mx = Float32(-3.0e38)
        for i in range(n_scores):
            if scores[i] > mx:
                mx = scores[i]
        var total = Float32(0)
        for i in range(n_scores):
            var e = exp(scores[i] - mx)
            scores[i] = e
            total += e
        var inv = Float32(1.0) / total
        for i in range(n_scores):
            scores[i] = scores[i] * inv
        for d in range(head_dim):
            var acc = Float32(0)
            for i in range(n_scores):
                var vv = Float32(0)
                if dense:
                    vv = Float32(
                        v_ptr.unsafe_load[width=1](
                            offset=(kv_head * max_len + first + i) * head_dim
                            + d
                        )
                    )
                else:
                    vv = cache.get_v(kv_head, first + i, d)
                acc += scores[i] * vv
            out.set(h * head_dim + d, Scalar[DType.float16](acc))

    if opts.gate:
        # qwen35: fused Q+gate projection - the gate lives in the second
        # half of `q_flat`, interleaved per head: [q0, g0, q1, g1, ...]
        for h in range(n_heads):
            for d in range(head_dim):
                var g = Float32(
                    q_flat.get(h * 2 * head_dim + head_dim + d)
                )
                var s = Float32(1.0) / (Float32(1.0) + exp(-g))
                out.set(
                    h * head_dim + d,
                    Scalar[DType.float16](Float32(out.get(h * head_dim + d)) * s),
                )
    var out_flat = _flat_view[DType.float16](out, n_heads * head_dim)
    return wo.proj(out_flat, dummy_scale)


def rms_norm_heads[dtype: DType](
    x: Tensor[dtype, 3],
    weight: Tensor[dtype, 1],
    eps: Float32,
) -> Tensor[dtype, 3]:
    """Per-head RMSNorm over the last dim (shared weight across heads).

    out[h, t, d] = x[h, t, d] / sqrt(mean(x[h, t, :]^2) + eps) * w[d]
    (hunyuan-dense / qwen35 attn_q_norm & attn_k_norm).
    """
    var n_heads = x.shape()[0]
    var n_tokens = x.shape()[1]
    var head_dim = x.shape()[2]
    if weight.shape()[0] != head_dim:
        unimplemented("rms_norm_heads: weight length != head_dim")
    var out = tensor_zeros[dtype, 3](x.shape())
    for h in range(n_heads):
        for t in range(n_tokens):
            var ss = Float32(0)
            var base = (h * n_tokens + t) * head_dim
            for d in range(head_dim):
                var v = Float32(x.get(base + d))
                ss += v * v
            var inv = Float32(1.0) / sqrt(ss / Float32(head_dim) + eps)
            for d in range(head_dim):
                out.set(
                    base + d,
                    Scalar[dtype](
                        Float32(x.get(base + d)) * inv * Float32(weight.get(d))
                    ),
                )
    return out


def multi_head_attention[dtype: DType, num_heads: Int, head_dim: Int](
    query: Tensor[dtype, 3], key: Tensor[dtype, 3], value: Tensor[dtype, 3]
) -> Tensor[dtype, 3]:
    """Bare stateless MHA over [n_heads, T, head_dim] tensors (no cache).

    Full causal attention over the whole sequence; used by tests and kept
    for API parity with the M1 signature.
    """
    var n_tokens = query.shape()[1]
    if key.shape()[1] != n_tokens or value.shape()[1] != n_tokens:
        unimplemented("multi_head_attention: sequence mismatch")
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var out = tensor_zeros[dtype, 3](query.shape())
    for h in range(num_heads):
        for t in range(n_tokens):
            var scores = List[Float32]()
            for t2 in range(t + 1):  # causal
                var acc = Float32(0)
                for d in range(head_dim):
                    acc += Float32(query.get((h * n_tokens + t) * head_dim + d)) * Float32(
                        key.get((h * n_tokens + t2) * head_dim + d)
                    )
                scores.append(acc * scale)
            var mx = Float32(-3.0e38)
            for i in range(len(scores)):
                if scores[i] > mx:
                    mx = scores[i]
            var total = Float32(0)
            for i in range(len(scores)):
                var e = exp(scores[i] - mx)
                scores[i] = e
                total += e
            for i in range(len(scores)):
                scores[i] = scores[i] / total
            for d in range(head_dim):
                var acc = Float32(0)
                for t2 in range(t + 1):
                    acc += scores[t2] * Float32(
                        value.get((h * n_tokens + t2) * head_dim + d)
                    )
                out.set(
                    (h * n_tokens + t) * head_dim + d, Scalar[dtype](acc)
                )
    return out


def mha_forward_with_saved(
    x: Tensor[DType.float16, 2],
    wq: Tensor[DType.float16, 2],
    wk: Tensor[DType.float16, 2],
    wv: Tensor[DType.float16, 2],
    wo: Tensor[DType.float16, 2],
    bq: Tensor[DType.float16, 1],
    bk: Tensor[DType.float16, 1],
    bv: Tensor[DType.float16, 1],
    mut cache: KVCacheLayer,
    start_pos: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    rope_theta: Float32,
) -> Tuple[Tensor[DType.float16, 2], List[Tensor[DType.float16, 2]]]:
    var out = mha_forward(
        x,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        cache,
        start_pos,
        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta,
    )
    var saved = List[Tensor[DType.float16, 2]]()
    saved.append(x)
    return (out, saved^)


def mha_backward[dtype: DType](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    """Typed stub - the cached MHA backward lives behind the registry's
    erased dispatcher (see `mha_backward_erased` below); this typed entry
    only exists for API parity."""
    _ = grad_out
    _ = saved
    unimplemented("mha_backward: use the registry erased dispatcher")
    return List[Tensor[dtype, 2]]()


# -- M6: stateless full-sequence MHA (forward + backward) --------------------
#
# `mha_seq` is the training-friendly attention op: causal attention over a
# whole [M, hidden] sequence (no KV cache).  It composes the same kernels as
# the cached decode path (QKV projection + bias, RoPE, scaled scores,
# softmax, value mixing, output projection) so its backward is the exact
# Q/K/V + softmax + matmul chain the M6 spec asks for.


def _mha_seq_forward_typed[dtype: DType](
    x: Tensor[dtype, 2],
    wq: Tensor[dtype, 2],
    wk: Tensor[dtype, 2],
    wv: Tensor[dtype, 2],
    wo: Tensor[dtype, 2],
    bq: Tensor[dtype, 1],
    bk: Tensor[dtype, 1],
    bv: Tensor[dtype, 1],
    start_pos: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    rope_theta: Float32,
) -> Tuple[
    Tensor[dtype, 2],
    Tensor[dtype, 3],
    Tensor[dtype, 3],
    Tensor[dtype, 3],
    Tensor[dtype, 3],
    Tensor[dtype, 2],
]:
    """Forward plus the intermediates the backward pass needs.

    Returns (out, q_rot, k_rot, v3, p, o_flat) where p is [n_heads, M, M]
    of attention probabilities (zero-padded above the causal diagonal).
    """
    var M = x.shape()[0]
    var hidden = x.shape()[1]
    var kv_hidden = wk.shape()[0]
    var q_flat = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, hidden))
    var k_flat = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, kv_hidden))
    var v_flat = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, kv_hidden))
    matmul_weight_3_threaded[dtype](x, wq, wk, wv, q_flat, k_flat, v_flat)
    if bq.numel() > 0:
        q_flat = add_row_cpu[dtype](q_flat, bq)
    if bk.numel() > 0:
        k_flat = add_row_cpu[dtype](k_flat, bk)
    if bv.numel() > 0:
        v_flat = add_row_cpu[dtype](v_flat, bv)
    # NOTE: the flat layout is [M, n_heads*head_dim] with the head index
    # fastest, so a plain [n_heads, M, head_dim] *view* would scramble the
    # heads for M > 1.  Gather explicitly.
    var q3 = tensor_zeros[dtype, 3](
        StaticTuple[Int, 3](n_heads, M, head_dim)
    )
    var k3 = tensor_zeros[dtype, 3](
        StaticTuple[Int, 3](n_kv_heads, M, head_dim)
    )
    var v3 = tensor_zeros[dtype, 3](
        StaticTuple[Int, 3](n_kv_heads, M, head_dim)
    )
    for t in range(M):
        for h in range(n_heads):
            for d in range(head_dim):
                q3.set(
                    (h * M + t) * head_dim + d,
                    q_flat.get(t * hidden + h * head_dim + d),
                )
        for h in range(n_kv_heads):
            for d in range(head_dim):
                k3.set(
                    (h * M + t) * head_dim + d,
                    k_flat.get(t * kv_hidden + h * head_dim + d),
                )
                v3.set(
                    (h * M + t) * head_dim + d,
                    v_flat.get(t * kv_hidden + h * head_dim + d),
                )
    var q_rot = rope_cpu_dynamic[dtype](q3, start_pos, rope_theta)
    var k_rot = rope_cpu_dynamic[dtype](k3, start_pos, rope_theta)

    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var p = tensor_zeros[dtype, 3](
        StaticTuple[Int, 3](n_heads, M, M)
    )
    var o = tensor_zeros[dtype, 3](
        StaticTuple[Int, 3](n_heads, M, head_dim)
    )
    for h in range(n_heads):
        var kv = h * n_kv_heads // n_heads
        for t in range(M):
            var scores = List[Float32]()
            for t2 in range(t + 1):
                var acc = Float32(0)
                for d in range(head_dim):
                    acc += Float32(
                        q_rot.get((h * M + t) * head_dim + d)
                    ) * Float32(k_rot.get((kv * M + t2) * head_dim + d))
                scores.append(acc * scale)
            var mx = Float32(-3.0e38)
            for i in range(len(scores)):
                if scores[i] > mx:
                    mx = scores[i]
            var total = Float32(0)
            for i in range(len(scores)):
                var e = exp(scores[i] - mx)
                scores[i] = e
                total += e
            var inv = Float32(1.0) / total
            for i in range(len(scores)):
                scores[i] = scores[i] * inv
            for t2 in range(t + 1):
                p.set(
                    (h * M + t) * M + t2, Scalar[dtype](scores[t2])
                )
            for d in range(head_dim):
                var acc = Float32(0)
                for t2 in range(t + 1):
                    acc += scores[t2] * Float32(
                        v3.get((kv * M + t2) * head_dim + d)
                    )
                o.set((h * M + t) * head_dim + d, Scalar[dtype](acc))
    var o_flat = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, hidden))
    for t in range(M):
        for h in range(n_heads):
            for d in range(head_dim):
                o_flat.set(
                    t * hidden + h * head_dim + d,
                    o.get((h * M + t) * head_dim + d),
                )
    var out = matmul_weight_cpu_threaded[dtype](o_flat, wo)
    return (out, q_rot, k_rot, v3, p, o_flat)


def _mha_seq_backward_typed[dtype: DType](
    grad_out: Tensor[dtype, 2],
    x: Tensor[dtype, 2],
    wq: Tensor[dtype, 2],
    wk: Tensor[dtype, 2],
    wv: Tensor[dtype, 2],
    wo: Tensor[dtype, 2],
    bq: Tensor[dtype, 1],
    bk: Tensor[dtype, 1],
    bv: Tensor[dtype, 1],
    q_rot: Tensor[dtype, 3],
    k_rot: Tensor[dtype, 3],
    v3: Tensor[dtype, 3],
    p: Tensor[dtype, 3],
    o_flat: Tensor[dtype, 2],
    start_pos: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    rope_theta: Float32,
) -> Tuple[
    Tensor[dtype, 2],
    Tensor[dtype, 2],
    Tensor[dtype, 2],
    Tensor[dtype, 2],
    Tensor[dtype, 2],
    Tensor[dtype, 1],
    Tensor[dtype, 1],
    Tensor[dtype, 1],
]:
    """Full backward of the stateless causal MHA."""
    var M = x.shape()[0]
    var hidden = x.shape()[1]
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    # 1. output projection: grad_o_flat = grad_out @ wo; grad_wo = grad_out^T @ o
    var mw_saved = List[Tensor[dtype, 2]]()
    mw_saved.append(o_flat)
    mw_saved.append(wo)
    var mw_grads = matmul_weight_cpu_backward[dtype](grad_out, mw_saved)
    var grad_o_flat = mw_grads[0]
    var grad_wo = mw_grads[1]

    var grad_o = tensor_zeros[dtype, 3](
        StaticTuple[Int, 3](n_heads, M, head_dim)
    )
    for t in range(M):
        for h in range(n_heads):
            for d in range(head_dim):
                grad_o.set(
                    (h * M + t) * head_dim + d,
                    grad_o_flat.get(t * hidden + h * head_dim + d),
                )
    var grad_qrot = tensor_zeros[dtype, 3](q_rot.shape())
    var grad_krot = tensor_zeros[dtype, 3](k_rot.shape())
    var grad_v = tensor_zeros[dtype, 3](v3.shape())

    for h in range(n_heads):
        var kv = h * n_kv_heads // n_heads
        for t in range(M):
            # g_p[t2] = <grad_o[h,t,:], v[kv,t2,:]>
            var dot = Float32(0)
            for t2 in range(t + 1):
                var acc = Float32(0)
                for d in range(head_dim):
                    acc += Float32(
                        grad_o.get((h * M + t) * head_dim + d)
                    ) * Float32(v3.get((kv * M + t2) * head_dim + d))
                dot += Float32(p.get((h * M + t) * M + t2)) * acc
            for t2 in range(t + 1):
                var acc = Float32(0)
                for d in range(head_dim):
                    acc += Float32(
                        grad_o.get((h * M + t) * head_dim + d)
                    ) * Float32(v3.get((kv * M + t2) * head_dim + d))
                var pt = Float32(p.get((h * M + t) * M + t2))
                var gs = pt * (acc - dot)
                # grad_v
                for d in range(head_dim):
                    grad_v.set(
                        (kv * M + t2) * head_dim + d,
                        Scalar[dtype](
                            Float32(
                                grad_v.get((kv * M + t2) * head_dim + d)
                            )
                            + pt
                            * Float32(
                                grad_o.get((h * M + t) * head_dim + d)
                            )
                        ),
                    )
                # grad_qrot / grad_krot
                for d in range(head_dim):
                    grad_qrot.set(
                        (h * M + t) * head_dim + d,
                        Scalar[dtype](
                            Float32(
                                grad_qrot.get((h * M + t) * head_dim + d)
                            )
                            + gs
                            * scale
                            * Float32(
                                k_rot.get((kv * M + t2) * head_dim + d)
                            )
                        ),
                    )
                    grad_krot.set(
                        (kv * M + t2) * head_dim + d,
                        Scalar[dtype](
                            Float32(
                                grad_krot.get((kv * M + t2) * head_dim + d)
                            )
                            + gs
                            * scale
                            * Float32(
                                q_rot.get((h * M + t) * head_dim + d)
                            )
                        ),
                    )

    # 2. RoPE backward (inverse rotation)
    var grad_q = rope_cpu_backward[dtype, 0, 0](
        grad_qrot, q_rot, start_pos, rope_theta
    )
    var grad_k = rope_cpu_backward[dtype, 0, 0](
        grad_krot, k_rot, start_pos, rope_theta
    )

    # 3. flatten + bias gradients (explicit scatter back to the flat layout)
    var kv_hidden = wk.shape()[0]
    var grad_q_flat = tensor_zeros[dtype, 2](
        StaticTuple[Int, 2](M, hidden)
    )
    var grad_k_flat = tensor_zeros[dtype, 2](
        StaticTuple[Int, 2](M, kv_hidden)
    )
    var grad_v_flat = tensor_zeros[dtype, 2](
        StaticTuple[Int, 2](M, kv_hidden)
    )
    for t in range(M):
        for h in range(n_heads):
            for d in range(head_dim):
                grad_q_flat.set(
                    t * hidden + h * head_dim + d,
                    grad_q.get((h * M + t) * head_dim + d),
                )
        for h in range(n_kv_heads):
            for d in range(head_dim):
                grad_k_flat.set(
                    t * kv_hidden + h * head_dim + d,
                    grad_k.get((h * M + t) * head_dim + d),
                )
                grad_v_flat.set(
                    t * kv_hidden + h * head_dim + d,
                    grad_v.get((h * M + t) * head_dim + d),
                )
    var grad_bq = tensor_zeros[dtype, 1](bq.shape())
    var grad_bk = tensor_zeros[dtype, 1](bk.shape())
    var grad_bv = tensor_zeros[dtype, 1](bv.shape())
    for i in range(M):
        for j in range(hidden):
            grad_bq.set(
                j,
                Scalar[dtype](
                    Float32(grad_bq.get(j)) + Float32(grad_q_flat.get(i * hidden + j))
                ),
            )
        for j in range(kv_hidden):
            grad_bk.set(
                j,
                Scalar[dtype](
                    Float32(grad_bk.get(j)) + Float32(grad_k_flat.get(i * kv_hidden + j))
                ),
            )
            grad_bv.set(
                j,
                Scalar[dtype](
                    Float32(grad_bv.get(j)) + Float32(grad_v_flat.get(i * kv_hidden + j))
                ),
            )

    # 4. projection gradients: grad_x accumulates the three QKV terms
    var grad_x = tensor_zeros[dtype, 2](x.shape())
    var mwq_saved = List[Tensor[dtype, 2]]()
    mwq_saved.append(x)
    mwq_saved.append(wq)
    var mwq_grads = matmul_weight_cpu_backward[dtype](grad_q_flat, mwq_saved)
    grad_x = _add_t2[dtype](grad_x, mwq_grads[0])
    var grad_wq = mwq_grads[1]

    var mwk_saved = List[Tensor[dtype, 2]]()
    mwk_saved.append(x)
    mwk_saved.append(wk)
    var mwk_grads = matmul_weight_cpu_backward[dtype](grad_k_flat, mwk_saved)
    grad_x = _add_t2[dtype](grad_x, mwk_grads[0])
    var grad_wk = mwk_grads[1]

    var mwv_saved = List[Tensor[dtype, 2]]()
    mwv_saved.append(x)
    mwv_saved.append(wv)
    var mwv_grads = matmul_weight_cpu_backward[dtype](grad_v_flat, mwv_saved)
    grad_x = _add_t2[dtype](grad_x, mwv_grads[0])
    var grad_wv = mwv_grads[1]

    return (
        grad_x,
        grad_wq,
        grad_wk,
        grad_wv,
        grad_wo,
        grad_bq,
        grad_bk,
        grad_bv,
    )


def _add_t2[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Element-wise tensor add used by the MHA backward accumulation."""
    var out = tensor_zeros[dtype, 2](a.shape())
    for i in range(a.numel()):
        out.set(i, Scalar[dtype](Float32(a.get(i)) + Float32(b.get(i))))
    return out
