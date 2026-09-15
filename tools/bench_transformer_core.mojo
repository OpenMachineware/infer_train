# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_transformer_core.mojo
#
# Pure compute benchmark - transformer layers only (no tokenizer, no sampling).
# Measures the core compute throughput for stories15M model.

from std.algorithm import vectorize
from std.math import sqrt, exp
from std.memory import Pointer, unsafe_memcpy, unsafe_memset_zero, stack_allocation
from std.memory.alloc import unsafe_alloc
from std.sys import argv, size_of, num_performance_cores, simd_width_of
from std.time import perf_counter_ns
from std.utils import StaticTuple
from max.algorithm import parallelize
from max.gpu.host import DeviceContext
from std.collections import List

comptime NUM_CONFIG_INT = 7
comptime SIMD_NELTS = 4 * simd_width_of[DType.float32]()

comptime BufferPtr = Pointer[Float32, MutUntrackedOrigin]


struct Matrix(Movable):
    var data: BufferPtr
    var allocated: Int
    var dims: List[Int]

    def __init__(out self, *dims: Int):
        self.dims = List[Int]()
        for i in range(len(dims)):
            self.dims.append(dims[i])
        var s = 1
        for i in range(len(self.dims)):
            s *= self.dims[i]
        self.allocated = 1
        self.data = unsafe_alloc[Float32](s)

    def __init__(out self, ptr: BufferPtr, var dims: List[Int]):
        self.data = ptr
        self.allocated = 0
        self.dims = dims^

    @always_inline
    def size(self) -> Int:
        var s = 1
        for i in range(len(self.dims)):
            s *= self.dims[i]
        return s

    @always_inline
    def zero(mut self):
        unsafe_memset_zero(self.data, self.size())

    @always_inline
    def slice(self, idx: Int) -> BufferPtr:
        if len(self.dims) > 2:
            var stride = self.dims[1] * self.dims[2]
            return self.data.unsafe_offset(idx * stride)
        elif len(self.dims) > 1:
            return self.data.unsafe_offset(idx * self.dims[1])
        else:
            return self.data.unsafe_offset(idx)

    @always_inline
    def slice(self, idx1: Int, idx2: Int) -> BufferPtr:
        var cols = self.dims[len(self.dims)-1]
        var rows = self.dims[len(self.dims)-2]
        var offset = idx1 * rows * cols + idx2 * cols
        return self.data.unsafe_offset(offset)


struct Config:
    var dim: Int
    var kv_dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var kv_mul: Int
    var vocab_size: Int
    var seq_len: Int
    var head_size: Int
    var shared_weights: Bool

    def __init__(out self, filename: String) raises:
        var f = open(filename, "r")
        var bytes_of_config_params = NUM_CONFIG_INT * size_of[DType.int32]()
        var config_data_raw = f.read_bytes(bytes_of_config_params)
        f.close()
        var int32_ptr = config_data_raw.unsafe_ptr().unsafe_bitcast[Int32]()
        self.dim = Int(int32_ptr.unsafe_offset(0)[])
        self.hidden_dim = Int(int32_ptr.unsafe_offset(1)[])
        self.n_layers = Int(int32_ptr.unsafe_offset(2)[])
        self.n_heads = Int(int32_ptr.unsafe_offset(3)[])
        self.n_kv_heads = Int(int32_ptr.unsafe_offset(4)[])
        self.vocab_size = Int(int32_ptr.unsafe_offset(5)[])
        self.seq_len = Int(int32_ptr.unsafe_offset(6)[])
        self.head_size = self.dim // self.n_heads
        self.kv_dim = (self.n_kv_heads * self.dim) // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv_heads
        self.shared_weights = self.vocab_size > 0
        if not self.shared_weights:
            self.vocab_size = -self.vocab_size


struct TransformerWeights:
    var token_embedding_table: Matrix
    var rms_att_weight: Matrix
    var wq: Matrix
    var wk: Matrix
    var wv: Matrix
    var wo: Matrix
    var rms_ffn_weight: Matrix
    var w1: Matrix
    var w3: Matrix
    var w2: Matrix
    var rms_final_weight: Matrix
    var freq_cis_real: Matrix
    var freq_cis_imag: Matrix
    var wcls: Matrix

    def __init__(out self, file_name: String, config: Config) raises:
        var f = open(file_name, "r")
        _ = f.read_bytes(NUM_CONFIG_INT * size_of[DType.int32]())

        @parameter
        def read_weights(*dims: Int) raises -> Matrix:
            var dim_list = List[Int]()
            var num_elements = 1
            for i in range(len(dims)):
                dim_list.append(dims[i])
                num_elements *= dims[i]
            var tmp = f.read_bytes(num_elements * size_of[Float32]())
            var data = tmp.unsafe_take_allocation().unsafe_leak().unsafe_bitcast[Float32]()
            return Matrix(data, dim_list^)

        self.token_embedding_table = read_weights(config.vocab_size, config.dim)
        self.rms_att_weight = read_weights(config.n_layers, config.dim)
        self.wq = read_weights(config.n_layers, config.dim, config.dim)
        self.wk = read_weights(config.n_layers, config.kv_dim, config.dim)
        self.wv = read_weights(config.n_layers, config.kv_dim, config.dim)
        self.wo = read_weights(config.n_layers, config.dim, config.dim)
        self.rms_ffn_weight = read_weights(config.n_layers, config.dim)
        self.w1 = read_weights(config.n_layers, config.hidden_dim, config.dim)
        self.w2 = read_weights(config.n_layers, config.dim, config.hidden_dim)
        self.w3 = read_weights(config.n_layers, config.hidden_dim, config.dim)
        self.rms_final_weight = read_weights(config.dim)
        self.freq_cis_real = read_weights(config.seq_len, config.head_size // 2)
        self.freq_cis_imag = read_weights(config.seq_len, config.head_size // 2)

        if config.shared_weights:
            var dims = self.token_embedding_table.dims.copy()
            self.wcls = Matrix(self.token_embedding_table.data, dims^)
            self.wcls.allocated = 0
        else:
            self.wcls = read_weights(config.vocab_size, config.dim)

        f.close()


struct RunState:
    var x: Matrix
    var xb: Matrix
    var xb2: Matrix
    var hb: Matrix
    var hb2: Matrix
    var q: Matrix
    var att: Matrix
    var key_cache: Matrix
    var value_cache: Matrix

    def __init__(out self, dim: Int, hidden_dim: Int, n_heads: Int, seq_len: Int, n_layers: Int, kv_dim: Int):
        self.x = Matrix(dim)
        self.xb = Matrix(dim)
        self.xb2 = Matrix(dim)
        self.hb = Matrix(hidden_dim)
        self.hb2 = Matrix(hidden_dim)
        self.q = Matrix(dim)
        self.att = Matrix(n_heads, seq_len)
        self.key_cache = Matrix(n_layers, seq_len, kv_dim)
        self.value_cache = Matrix(n_layers, seq_len, kv_dim)


struct Transformer:
    var workers: Int
    var ctx: DeviceContext

    def __init__(out self, workers: Int) raises:
        self.workers = workers
        self.ctx = DeviceContext(api="cpu")

    @always_inline
    def transformer(
        self,
        token: Int,
        pos: Int,
        config: Config,
        mut state: RunState,
        weights: TransformerWeights,
    ):
        var dim = config.dim
        var hidden_dim = config.hidden_dim
        var head_size = config.head_size
        var kv_dim = config.kv_dim
        var kv_mul = config.kv_mul
        var sqrt_head_size = sqrt(Float32(head_size))

        # Embedding
        var content_row = weights.token_embedding_table.slice(token)
        unsafe_memcpy(dest=state.x.data, src=content_row, count=dim)

        var freq_cis_real_row = weights.freq_cis_real.slice(pos)
        var freq_cis_imag_row = weights.freq_cis_imag.slice(pos)

        for l in range(config.n_layers):
            # RMSNorm
            var ss: Float32 = 0.0
            for j in range(0, dim, SIMD_NELTS):
                var val = state.x.data.unsafe_load[width=SIMD_NELTS](j) ** 2
                ss += val.reduce_add()
            ss = ss / Float32(dim) + 1e-5
            var inv = 1.0 / sqrt(ss)
            for j in range(0, dim, SIMD_NELTS):
                var w = weights.rms_att_weight.slice(l).unsafe_load[width=SIMD_NELTS](j)
                var v = state.x.data.unsafe_load[width=SIMD_NELTS](j)
                state.xb.data.unsafe_store[width=SIMD_NELTS](j, w * inv * v)

            var loff = l * config.seq_len * kv_dim
            var k_ptr = state.key_cache.slice(l, pos)
            var v_ptr = state.value_cache.slice(l, pos)

            # QKV projections
            if kv_dim == dim:
                # Full QKV - batched
                for out_idx in range(dim):
                    var acc_q: Float32 = 0.0
                    var acc_k: Float32 = 0.0
                    var acc_v: Float32 = 0.0
                    for j in range(0, dim, SIMD_NELTS):
                        var xj = state.xb.data.unsafe_load[width=SIMD_NELTS](j)
                        var wq = weights.wq.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                        var wk = weights.wk.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                        var wv = weights.wv.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                        acc_q += (xj * wq).reduce_add()
                        acc_k += (xj * wk).reduce_add()
                        acc_v += (xj * wv).reduce_add()
                    state.q.data.unsafe_store[width=1](out_idx, acc_q)
                    k_ptr.unsafe_store[width=1](out_idx, acc_k)
                    v_ptr.unsafe_store[width=1](out_idx, acc_v)
            else:
                # GQA - separate
                for out_idx in range(dim):
                    var acc: Float32 = 0.0
                    for j in range(0, dim, SIMD_NELTS):
                        var xj = state.xb.data.unsafe_load[width=SIMD_NELTS](j)
                        var w = weights.wq.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                        acc += (xj * w).reduce_add()
                    state.q.data.unsafe_store[width=1](out_idx, acc)
                for out_idx in range(kv_dim):
                    var acc_k: Float32 = 0.0
                    var acc_v: Float32 = 0.0
                    for j in range(0, dim, SIMD_NELTS):
                        var xj = state.xb.data.unsafe_load[width=SIMD_NELTS](j)
                        var wk = weights.wk.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                        var wv = weights.wv.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                        acc_k += (xj * wk).reduce_add()
                        acc_v += (xj * wv).reduce_add()
                    k_ptr.unsafe_store[width=1](out_idx, acc_k)
                    v_ptr.unsafe_store[width=1](out_idx, acc_v)

            # RoPE
            for i in range(config.n_heads):
                for j in range(0, head_size, 2):
                    var fcr = freq_cis_real_row.unsafe_offset(j // 2)[]
                    var fci = freq_cis_imag_row.unsafe_offset(j // 2)[]
                    var q_idx = i * head_size + j
                    var q0 = state.q.data.unsafe_offset(q_idx)[]
                    var q1 = state.q.data.unsafe_offset(q_idx + 1)[]
                    state.q.data.unsafe_offset(q_idx)[] = q0 * fcr - q1 * fci
                    state.q.data.unsafe_offset(q_idx + 1)[] = q0 * fci + q1 * fcr
                    if i < config.n_kv_heads:
                        var k_idx = i * head_size + j
                        var k0 = k_ptr.unsafe_offset(k_idx)[]
                        var k1 = k_ptr.unsafe_offset(k_idx + 1)[]
                        k_ptr.unsafe_offset(k_idx)[] = k0 * fcr - k1 * fci
                        k_ptr.unsafe_offset(k_idx + 1)[] = k0 * fci + k1 * fcr

            # Zero attention output
            for i in range(0, dim, SIMD_NELTS):
                state.xb.data.unsafe_store[width=SIMD_NELTS](i, SIMD[DType.float32, SIMD_NELTS](0))

            # Multi-head attention
            for h in range(config.n_heads):
                var q_offset = h * head_size
                var att_offset = h * config.seq_len

                for t in range(pos + 1):
                    var k_offset = loff + t * kv_dim + (h // kv_mul) * head_size
                    var score: Float32 = 0.0
                    for i in range(0, head_size, SIMD_NELTS):
                        score += (
                            state.q.data.unsafe_load[width=SIMD_NELTS](q_offset + i)
                            * state.key_cache.data.unsafe_load[width=SIMD_NELTS](k_offset + i)
                        ).reduce_add()
                    state.att.data.unsafe_offset(att_offset + t)[] = score / sqrt_head_size

                # Softmax
                var max_val: Float32 = -1e9
                for ii in range(att_offset, att_offset + pos + 1, SIMD_NELTS):
                    var v = state.att.data.unsafe_load[width=SIMD_NELTS](ii).reduce_max()
                    if v > max_val:
                        max_val = v
                var ssum: Float32 = 0.0
                for ii in range(att_offset, att_offset + pos + 1, SIMD_NELTS):
                    var vals = state.att.data.unsafe_load[width=SIMD_NELTS](ii)
                    var exp_vals = exp(vals - max_val)
                    state.att.data.unsafe_store[width=SIMD_NELTS](ii, exp_vals)
                    ssum += exp_vals.reduce_add()
                for ii in range(att_offset, att_offset + pos + 1, SIMD_NELTS):
                    state.att.data.unsafe_store[width=SIMD_NELTS](
                        ii, state.att.data.unsafe_load[width=SIMD_NELTS](ii) / ssum
                    )

                # Weighted value
                var xb_offset = h * head_size
                for t in range(pos + 1):
                    var v_offset = loff + t * kv_dim + (h // kv_mul) * head_size
                    var a = state.att.data.unsafe_offset(att_offset + t)[]
                    for i in range(0, head_size, SIMD_NELTS):
                        var xbi = state.xb.data.unsafe_offset(xb_offset + i).unsafe_load[width=SIMD_NELTS](0)
                            + a * state.value_cache.data.unsafe_offset(v_offset + i).unsafe_load[width=SIMD_NELTS](0)
                        state.xb.data.unsafe_offset(xb_offset + i).unsafe_store[width=SIMD_NELTS](0, xbi)

            # Output projection
            for out_idx in range(dim):
                var acc: Float32 = 0.0
                for j in range(0, dim, SIMD_NELTS):
                    var xj = state.xb.data.unsafe_load[width=SIMD_NELTS](j)
                    var w = weights.wo.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                    acc += (xj * w).reduce_add()
                state.xb2.data.unsafe_store[width=1](out_idx, acc)

            # Residual
            for i in range(0, dim, SIMD_NELTS):
                var a = state.x.data.unsafe_load[width=SIMD_NELTS](i)
                var b = state.xb2.data.unsafe_load[width=SIMD_NELTS](i)
                state.x.data.unsafe_store[width=SIMD_NELTS](i, a + b)

            # FFN RMSNorm
            var ss_ffn: Float32 = 0.0
            for j in range(0, dim, SIMD_NELTS):
                var val = state.x.data.unsafe_load[width=SIMD_NELTS](j) ** 2
                ss_ffn += val.reduce_add()
            ss_ffn = ss_ffn / Float32(dim) + 1e-5
            var inv_ffn = 1.0 / sqrt(ss_ffn)
            for j in range(0, dim, SIMD_NELTS):
                var w = weights.rms_ffn_weight.slice(l).unsafe_load[width=SIMD_NELTS](j)
                var v = state.x.data.unsafe_load[width=SIMD_NELTS](j)
                state.xb.data.unsafe_store[width=SIMD_NELTS](j, w * inv_ffn * v)

            # FFN gate + up
            for out_idx in range(hidden_dim):
                var acc_g: Float32 = 0.0
                var acc_u: Float32 = 0.0
                for j in range(0, dim, SIMD_NELTS):
                    var xj = state.xb.data.unsafe_load[width=SIMD_NELTS](j)
                    var wg = weights.w1.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                    var wu = weights.w3.slice(l).unsafe_offset(out_idx * dim).unsafe_load[width=SIMD_NELTS](j)
                    acc_g += (xj * wg).reduce_add()
                    acc_u += (xj * wu).reduce_add()
                # SiLU: gate * sigmoid(gate)
                var g = acc_g
                var silu = g / (1.0 + exp(-g))
                state.hb.data.unsafe_store[width=1](out_idx, silu * acc_u)

            # FFN down
            for out_idx in range(dim):
                var acc: Float32 = 0.0
                for j in range(0, hidden_dim, SIMD_NELTS):
                    var h = state.hb.data.unsafe_load[width=SIMD_NELTS](j)
                    var w = weights.w2.slice(l).unsafe_offset(out_idx * hidden_dim).unsafe_load[width=SIMD_NELTS](j)
                    acc += (h * w).reduce_add()
                state.xb.data.unsafe_store[width=1](out_idx, acc)

            # Residual
            for i in range(0, dim, SIMD_NELTS):
                var a = state.x.data.unsafe_load[width=SIMD_NELTS](i)
                var b = state.xb.data.unsafe_load[width=SIMD_NELTS](i)
                state.x.data.unsafe_store[width=SIMD_NELTS](i, a + b)


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: mojo bench_transformer_core.mojo <checkpoint> [-j workers] [-n steps]")
        return

    var checkpoint = args[1]
    var steps = 256
    var workers = min(num_performance_cores(), 4)

    for i in range(2, len(args), 2):
        if args[i] == "-j":
            workers = atol(args[i + 1])
        if args[i] == "-n":
            steps = atol(args[i + 1])

    var transformer = Transformer(workers)
    print("workers:", transformer.workers, "SIMD:", SIMD_NELTS)

    var config = Config(checkpoint)
    print("config: layers=", config.n_layers, " dim=", config.dim, " hidden=", config.hidden_dim)
    var weights = TransformerWeights(checkpoint, config)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    var state = RunState(config.dim, config.hidden_dim, config.n_heads, config.seq_len, config.n_layers, config.kv_dim)

    var token = 1
    var pos = 0

    # Warmup
    for _ in range(3):
        transformer.transformer(token, pos, config, state, weights)
    state.key_cache.zero()
    state.value_cache.zero()

    # Benchmark
    var start = perf_counter_ns()
    while pos < steps:
        transformer.transformer(token, pos, config, state, weights)
        pos += 1
        if pos == 1:
            start = perf_counter_ns()
    var end = perf_counter_ns()

    var elapsed_ms = (end - start) // 1_000_000
    var tps = Float64(pos - 1) * 1000.0 / Float64(elapsed_ms)
    print("achieved tok/s:", tps)
