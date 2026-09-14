# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/llama2_optimized.mojo
#
# High-performance CPU kernels following llama2.mojo optimization patterns:
# - 256-bit SIMD width (16 x f32 / 8 x f32)
# - batch_matmul for fused QKV projections
# - parallelize + DeviceContext for native Mojo parallelism
# - Adaptive thread count to minimize coroutine overhead
#
# This replaces the legacy C-thread-pool-based matmul implementations.

from std.algorithm import vectorize
from std.math import sqrt, exp
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.sys import simd_width_of, num_performance_cores
from max.algorithm import parallelize
from max.gpu.host import DeviceContext

# 256-bit SIMD width (4 * simd_width_of[Float32] on M1 = 64 bytes = 16 floats)
comptime SIMD_NELTS = 4 * simd_width_of[DType.float32]()
alias BufferPtr = Pointer[Float32, MutUntrackedOrigin]
alias BufferPtrF16 = Pointer[Scalar[DType.float16], MutUntrackedOrigin]


# ============================================================================
# Core vectorized kernels
# ============================================================================


@always_inline
def vec_rmsnorm(
    o: BufferPtr,
    x: BufferPtr,
    weight: BufferPtr,
    size: Int,
):
    """Vectorized RMSNorm with 256-bit SIMD.

    out[j] = weight[j] * x[j] / sqrt(mean(x^2) + eps)
    """
    var ss: Float32 = 0.0

    # Vectorized sum of squares
    for j in range(0, size, SIMD_NELTS):
        var val = x.unsafe_load[width=SIMD_NELTS](j) ** 2
        ss += val.reduce_add()

    ss = ss / Float32(size) + 1e-5
    ss = 1.0 / sqrt(ss)

    # Vectorized normalize and scale
    for j in range(0, size, SIMD_NELTS):
        var val = weight.unsafe_load[width=SIMD_NELTS](j) * ss * x.unsafe_load[width=SIMD_NELTS](j)
        o.unsafe_store[width=SIMD_NELTS](j, val)


@always_inline
def vec_softmax(x: BufferPtr, start: Int, end: Int):
    """Vectorized stable softmax over x[start:end]."""
    var max_val: Float32 = -1e9

    # Find max value
    for ii in range(start, end, SIMD_NELTS):
        var val = x.unsafe_load[width=SIMD_NELTS](ii).reduce_max()
        if val > max_val:
            max_val = val

    # Compute exp and sum
    var ssum: Float32 = 0.0
    for ii in range(start, end, SIMD_NELTS):
        var vals = x.unsafe_load[width=SIMD_NELTS](ii)
        var exp_vals = exp(vals - max_val)
        x.unsafe_store[width=SIMD_NELTS](ii, exp_vals)
        ssum += exp_vals.reduce_add()

    # Normalize
    for ii in range(start, end, SIMD_NELTS):
        x.unsafe_store[width=SIMD_NELTS](
            ii, x.unsafe_load[width=SIMD_NELTS](ii) / ssum
        )


@always_inline
def vec_add(dest: BufferPtr, src: BufferPtr, size: Int):
    """Vectorized elementwise add."""
    for i in range(0, size, SIMD_NELTS):
        var a = dest.unsafe_load[width=SIMD_NELTS](i)
        var b = src.unsafe_load[width=SIMD_NELTS](i)
        dest.unsafe_store[width=SIMD_NELTS](i, a + b)


# ============================================================================
# Batch matmul - fused QKV projections
# ============================================================================


@always_inline
def batch_matmul[
    n: Int  # Number of outputs (1, 2, or 3)
](
    C: StaticTuple[BufferPtr, n],
    A: BufferPtr,  # Input row [K]
    B: StaticTuple[BufferPtr, n],  # Weight rows [K] x n
    rows: Int,  # Number of output rows (usually = n)
    cols: Int,  # K dimension
    workers: Int,
    ctx: Optional[DeviceContext] = None,
):
    """Compute n matmuls in parallel: C[i] = A @ B[i]^T for i in 0..n-1.

    This is the key optimization from llama2.mojo: instead of computing Q, K, V
    separately with three memory passes over the input, we compute all three
    in a single pass, reusing the loaded input vector for each weight row.

    For QKV projections: A is [hidden], B is [3, hidden], C is [3]
    For gate/up projections: A is [hidden], B is [2, hidden], C is [2]
    """
    @parameter
    def compute_row(i: Int):
        # Stack allocation for n accumulators (each SIMD_NELTS wide)
        var tmp_ptr = stack_allocation[n * SIMD_NELTS, Float32]()

        # Zero accumulators
        comptime for k in range(n):
            tmp_ptr.unsafe_store[width=SIMD_NELTS](k * SIMD_NELTS, SIMD[DType.float32, SIMD_NELTS](0))

        # Vectorized dot product with all n weight rows simultaneously
        def dot[_nelts: Int](j: Int) {imm A, imm B, mut tmp_ptr}:
            var a = A.unsafe_load[width=_nelts](j)

            comptime for k in range(n):
                var val = a * B[k].unsafe_load[width=_nelts](j)
                var curr = tmp_ptr.unsafe_load[width=_nelts](k * SIMD_NELTS)
                tmp_ptr.unsafe_store[width=_nelts](k * SIMD_NELTS, curr + val)

        vectorize[SIMD_NELTS](cols, dot)

        # Store results
        comptime for k in range(n):
            C[k].unsafe_store(i, tmp_ptr.unsafe_load[width=SIMD_NELTS](k * SIMD_NELTS).reduce_add())

    # Adaptive worker count: each thread should process at least 32 rows
    # to minimize coroutine overhead (llama2.mojo insight)
    var adapted_workers = min(workers, max(1, rows // 32))
    parallelize[compute_row](rows, adapted_workers, ctx)


@always_inline
def matmul_single(
    C: BufferPtr,
    A: BufferPtr,
    B: BufferPtr,
    rows: Int,
    cols: Int,
    workers: Int,
    ctx: Optional[DeviceContext] = None,
):
    """Single matmul: C = A @ B^T where A is [cols], B is [rows, cols], C is [rows]."""
    batch_matmul[1](
        StaticTuple[BufferPtr, 1](C),
        A,
        StaticTuple[BufferPtr, 1](B),
        rows,
        cols,
        workers,
        ctx,
    )


# ============================================================================
# Optimized attention kernels
# ============================================================================


@always_inline
def attention_qk(
    q: BufferPtr,  # [n_heads, head_dim]
    k_cache: BufferPtr,  # [seq_len, kv_dim]
    att: BufferPtr,  # [n_heads, seq_len]
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    kv_dim: Int,
    seq_len: Int,
    sqrt_head_dim: Float32,
):
    """Compute Q @ K^T / sqrt(head_dim) for all heads.

    This is the hottest loop in inference - fully vectorized.
    """
    var kv_mul = n_heads // n_kv_heads

    for h in range(n_heads):
        var q_offset = h * head_dim
        var att_offset = h * seq_len
        var kv_head = h // kv_mul

        for t in range(seq_len):
            var k_offset = t * kv_dim + kv_head * head_dim
            var score: Float32 = 0.0

            for i in range(0, head_dim, SIMD_NELTS):
                score += (
                    q.unsafe_load[width=SIMD_NELTS](q_offset + i)
                    * k_cache.unsafe_load[width=SIMD_NELTS](k_offset + i)
                ).reduce_add()

            att.unsafe_store[width=1](att_offset + t, score / sqrt_head_dim)


@always_inline
def attention_weighted_v(
    att: BufferPtr,  # [n_heads, seq_len] - softmaxed scores
    v_cache: BufferPtr,  # [seq_len, kv_dim]
    xb: BufferPtr,  # [n_heads, head_dim] - output
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    kv_dim: Int,
    seq_len: Int,
):
    """Compute softmax(att) @ V for all heads.

    Fully vectorized accumulation of weighted value vectors.
    """
    var kv_mul = n_heads // n_kv_heads

    for h in range(n_heads):
        var att_offset = h * seq_len
        var xb_offset = h * head_dim
        var kv_head = h // kv_mul

        # Zero output
        for d in range(0, head_dim, SIMD_NELTS):
            xb.unsafe_store[width=SIMD_NELTS](xb_offset + d, SIMD[DType.float32, SIMD_NELTS](0))

        # Accumulate weighted values
        for t in range(seq_len):
            var a = att.unsafe_load[width=1](att_offset + t)
            var v_offset = t * kv_dim + kv_head * head_dim

            for d in range(0, head_dim, SIMD_NELTS):
                var xbi = xb.unsafe_load[width=SIMD_NELTS](xb_offset + d)
                    + a * v_cache.unsafe_load[width=SIMD_NELTS](v_offset + d)
                xb.unsafe_store[width=SIMD_NELTS](xb_offset + d, xbi)


# ============================================================================
# RoPE rotation
# ============================================================================


@always_inline
def rope_rotation(
    q: BufferPtr,
    k: BufferPtr,
    freq_cis_real: BufferPtr,
    freq_cis_imag: BufferPtr,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
):
    """Apply RoPE rotation to Q and K."""
    for i in range(n_heads):
        for j in range(0, head_dim, 2):
            var fcr = freq_cis_real.unsafe_load[width=1](j // 2)
            var fci = freq_cis_imag.unsafe_load[width=1](j // 2)

            # Q rotation
            var q_idx = i * head_dim + j
            var q0 = q.unsafe_load[width=1](q_idx)
            var q1 = q.unsafe_load[width=1](q_idx + 1)
            q.unsafe_store[width=1](q_idx, q0 * fcr - q1 * fci)
            q.unsafe_store[width=1](q_idx + 1, q0 * fci + q1 * fcr)

            # K rotation (only for kv heads)
            if i < n_kv_heads:
                var k_idx = i * head_dim + j
                var k0 = k.unsafe_load[width=1](k_idx)
                var k1 = k.unsafe_load[width=1](k_idx + 1)
                k.unsafe_store[width=1](k_idx, k0 * fcr - k1 * fci)
                k.unsafe_store[width=1](k_idx + 1, k0 * fci + k1 * fcr)


# ============================================================================
# Optimized transformer layer
# ============================================================================


struct OptimizedLayerState(Movable):
    """Preallocated buffers for a transformer layer."""
    var x: BufferPtr  # activation [hidden]
    var xb: BufferPtr  # residual branch [hidden]
    var xb2: BufferPtr  # second residual [hidden]
    var hb: BufferPtr  # FFN hidden [ffn_dim]
    var hb2: BufferPtr  # FFN second hidden [ffn_dim]
    var q: BufferPtr  # query [hidden]
    var att: BufferPtr  # attention scores [n_heads, seq_len]
    var key_cache: BufferPtr  # KV cache K
    var value_cache: BufferPtr  # KV cache V

    var hidden: Int
    var ffn_dim: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var seq_len: Int
    var kv_dim: Int
    var allocated: Bool

    def __init__(out self):
        self.x = BufferPtr()
        self.xb = BufferPtr()
        self.xb2 = BufferPtr()
        self.hb = BufferPtr()
        self.hb2 = BufferPtr()
        self.q = BufferPtr()
        self.att = BufferPtr()
        self.key_cache = BufferPtr()
        self.value_cache = BufferPtr()
        self.hidden = 0
        self.ffn_dim = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.head_dim = 0
        self.seq_len = 0
        self.kv_dim = 0
        self.allocated = False

    def alloc(
        mut self,
        hidden: Int,
        ffn_dim: Int,
        n_heads: Int,
        n_kv_heads: Int,
        head_dim: Int,
        seq_len: Int,
        n_layers: Int,
    ):
        self.hidden = hidden
        self.ffn_dim = ffn_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.kv_dim = (n_kv_heads * hidden) // n_heads

        self.x = unsafe_alloc[Float32](hidden)
        self.xb = unsafe_alloc[Float32](hidden)
        self.xb2 = unsafe_alloc[Float32](hidden)
        self.hb = unsafe_alloc[Float32](ffn_dim)
        self.hb2 = unsafe_alloc[Float32](ffn_dim)
        self.q = unsafe_alloc[Float32](hidden)
        self.att = unsafe_alloc[Float32](n_heads * seq_len)
        # KV cache shared across layers
        self.key_cache = unsafe_alloc[Float32](n_layers * seq_len * self.kv_dim)
        self.value_cache = unsafe_alloc[Float32](n_layers * seq_len * self.kv_dim)
        self.allocated = True


struct OptimizedTransformer(Movable):
    """Optimized transformer using llama2.mojo patterns."""
    var workers: Int
    var ctx: DeviceContext

    def __init__(out self, workers: Int = 0):
        if workers == 0:
            # Optimal: 4 threads for M1 Max (llama2.mojo finding)
            self.workers = min(num_performance_cores(), 4)
        else:
            self.workers = workers
        self.ctx = DeviceContext(api="cpu")

    @always_inline
    def transformer_layer(
        self,
        token: Int,
        pos: Int,
        state: OptimizedLayerState,
        weights: object,  # TransformerWeights struct
        layer: Int,
    ):
        """One transformer layer with all optimizations.

        This combines:
        1. batch_matmul for QKV projections
        2. Vectorized attention computation
        3. Fused RoPE rotation
        4. Vectorized FFN
        """
        var hidden = state.hidden
        var head_dim = state.head_dim
        var kv_dim = state.kv_dim
        var kv_mul = state.n_heads // state.n_kv_heads
        var sqrt_head_dim = sqrt(Float32(head_dim))

        # Embedding lookup (handled externally in the caller)

        # Attention RMSNorm
        vec_rmsnorm(state.xb, state.x, weights.rms_att_weight[layer], hidden)

        # KV cache offset for this layer and position
        var loff = layer * state.seq_len * kv_dim
        var k_ptr = state.key_cache.unsafe_offset(loff + pos * kv_dim)
        var v_ptr = state.value_cache.unsafe_offset(loff + pos * kv_dim)

        # QKV projections - batched for memory efficiency
        if kv_dim == hidden:
            # Full QKV (no GQA compression)
            batch_matmul[3](
                StaticTuple[BufferPtr, 3](
                    state.q, k_ptr, v_ptr
                ),
                state.xb,
                StaticTuple[BufferPtr, 3](
                    weights.wq[layer], weights.wk[layer], weights.wv[layer]
                ),
                hidden,
                hidden,
                self.workers,
                self.ctx,
            )
        else:
            # GQA: K/V are smaller
            matmul_single(state.q, state.xb, weights.wq[layer], hidden, hidden, self.workers, self.ctx)
            batch_matmul[2](
                StaticTuple[BufferPtr, 2](k_ptr, v_ptr),
                state.xb,
                StaticTuple[BufferPtr, 2](weights.wk[layer], weights.wv[layer]),
                kv_dim,
                hidden,
                self.workers,
                self.ctx,
            )

        # RoPE rotation
        rope_rotation(
            state.q, k_ptr,
            weights.freq_cis_real[pos], weights.freq_cis_imag[pos],
            state.n_heads, state.n_kv_heads, head_dim
        )

        # Zero attention output
        for i in range(0, hidden, SIMD_NELTS):
            state.xb.unsafe_store[width=SIMD_NELTS](i, SIMD[DType.float32, SIMD_NELTS](0))

        # Multi-head attention
        attention_qk(
            state.q, state.key_cache.unsafe_offset(loff), state.att,
            state.n_heads, state.n_kv_heads, head_dim, kv_dim, pos + 1, sqrt_head_dim
        )

        # Softmax per head
        for h in range(state.n_heads):
            var att_offset = h * state.seq_len
            vec_softmax(state.att, att_offset, att_offset + pos + 1)

        # Weighted value aggregation
        attention_weighted_v(
            state.att, state.value_cache.unsafe_offset(loff), state.xb,
            state.n_heads, state.n_kv_heads, head_dim, kv_dim, pos + 1
        )

        # Output projection
        matmul_single(state.xb2, state.xb, weights.wo[layer], hidden, hidden, self.workers, self.ctx)

        # Residual connection
        vec_add(state.x, state.xb2, hidden)

        # FFN RMSNorm
        vec_rmsnorm(state.xb, state.x, weights.rms_ffn_weight[layer], hidden)

        # FFN projections (gate + up)
        batch_matmul[2](
            StaticTuple[BufferPtr, 2](state.hb, state.hb2),
            state.xb,
            StaticTuple[BufferPtr, 2](weights.w1[layer], weights.w3[layer]),
            state.ffn_dim,
            hidden,
            self.workers,
            self.ctx,
        )

        # SiLU activation: hb = hb * sigmoid(hb)
        for i in range(0, state.ffn_dim, SIMD_NELTS):
            var initial_hb = state.hb.unsafe_load[width=SIMD_NELTS](i)
            var sigmoid_val = 1.0 / (1.0 + exp(-initial_hb))
            var hbi = initial_hb * sigmoid_val * state.hb2.unsafe_load[width=SIMD_NELTS](i)
            state.hb.unsafe_store[width=SIMD_NELTS](i, hbi)

        # FFN down projection
        matmul_single(state.xb, state.hb, weights.w2[layer], hidden, state.ffn_dim, self.workers, self.ctx)

        # Residual connection
        vec_add(state.x, state.xb, hidden)