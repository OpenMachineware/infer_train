# core/transformer.mojo
#
# Decoder-only transformer runtime (M3 -> M7).
#
# M3 scope: Qwen2 (architecture `qwen2`), typed single-token forward.
# M7 adds two more architectures, selected from the GGUF metadata:
#
#   ARCH_QWEN2    - qwen2/qwen2.5/qwen3 family (the M3 path)
#   ARCH_HUNYUAN  - hunyuan-dense (Hy-MT2): per-head Q/K RMSNorm applied
#                   *after* RoPE, Llama3-style ffn_norm, tied embeddings
#   ARCH_QWEN35   - qwen35 (Qwen3.8-27B hybrid): Gated DeltaNet linear
#                   attention layers every 3 of 4 layers (causal conv1d +
#                   SiLU + per-head L2 norm + decayed delta-rule
#                   recurrence + gated output norm) interleaved with
#                   full-attention layers (fused Q+gate, QK-norm, MRoPE
#                   over n_rot dims, sigmoid gate); the MTP block is not
#                   used (llama.cpp's default single-model decode samples
#                   from the main lm_head too).
#
# The forward calls the registered kernel implementations directly (the
# same entry points the OpRegistry dispatches to), which keeps the compute
# graph typed and comptime-specializable; `build_graph` still produces the
# M2-style op skeleton for the interpreter.

from .gguf_loader import (
    GGUFContext,
    GGUFTensor,
    find_tensor,
    get_meta_uint,
    get_meta_float,
    get_meta_str,
)
from .graph import Graph, AttrValue
from .tensor import Tensor, tensor_zeros
from .ops.quantized.dequantize import dequantize_into
from .ops.cpu.embedding_cpu import embedding_cpu_dynamic
from .ops.cpu.matmul_cpu import (
    matmul_cpu_dynamic,
    matmul_weight_cpu,
    matmul_weight_cpu_threaded,
    matmul_weight_2_threaded,
    matmul_weight_3_threaded,
)
from .ops.cpu.add_cpu import add_cpu_dynamic, add_row_cpu
from .ops.cpu.swiglu_cpu import swiglu_cpu_dynamic
from .ops.attention.mha import (
    mha_forward,
    mha_forward_v2,
    MHAOptions,
    rms_norm_heads,
)
from .ops.attention.kv_cache import KVCache, KVCacheLayer
from std.utils.static_tuple import StaticTuple
from std.math import sqrt, exp, log
from std.memory.unsafe import bitcast

comptime DEFAULT_KV_CACHE_LEN = 1024

# -- architecture tags --------------------------------------------------------

comptime ARCH_QWEN2 = Int8(0)
comptime ARCH_HUNYUAN = Int8(1)
comptime ARCH_QWEN35 = Int8(2)


def arch_name(arch: Int8) -> String:
    if arch == ARCH_HUNYUAN:
        return String("hunyuan-dense")
    if arch == ARCH_QWEN35:
        return String("qwen35")
    return String("qwen2")


struct TransformerConfig(Copyable, Movable, ImplicitlyCopyable):
    var arch: Int8
    var n_layers: Int
    var hidden: Int
    var ffn: Int
    var n_heads: Int
    var n_kv_heads: Int
    var vocab: Int
    var head_dim: Int
    var rope_theta: Float32
    var norm_eps: Float32
    var bos_id: Int
    var eos_id: Int
    # qwen35 extras
    var n_rot: Int  # MRoPE rotated dims (0 = whole head)
    var rope_sections: StaticTuple[Int, 4]
    var ssm_d_conv: Int
    var ssm_d_state: Int
    var ssm_dt_rank: Int
    var ssm_n_group: Int
    var ssm_d_inner: Int
    var full_attn_interval: Int
    var n_nextn: Int

    def __init__(out self):
        self.arch = ARCH_QWEN2
        self.n_layers = 0
        self.hidden = 0
        self.ffn = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.vocab = 0
        self.head_dim = 0
        self.rope_theta = Float32(10000.0)
        self.norm_eps = Float32(1e-6)
        self.bos_id = 1
        self.eos_id = 2
        self.n_rot = 0
        self.rope_sections = StaticTuple[Int, 4](fill=0)
        self.ssm_d_conv = 0
        self.ssm_d_state = 0
        self.ssm_dt_rank = 0
        self.ssm_n_group = 0
        self.ssm_d_inner = 0
        self.full_attn_interval = 4
        self.n_nextn = 0

    def is_recurrent(self, layer: Int) -> Bool:
        if self.arch != ARCH_QWEN35:
            return False
        return (layer + 1) % self.full_attn_interval != 0


struct LayerWeights(Copyable, Movable, ImplicitlyCopyable):
    var attn_norm_w: Tensor[DType.float16, 1]
    var q_w: Tensor[DType.float16, 2]
    var k_w: Tensor[DType.float16, 2]
    var v_w: Tensor[DType.float16, 2]
    var o_w: Tensor[DType.float16, 2]
    var q_b: Tensor[DType.float16, 1]
    var k_b: Tensor[DType.float16, 1]
    var v_b: Tensor[DType.float16, 1]
    var ffn_norm_w: Tensor[DType.float16, 1]
    var post_attn_norm_w: Tensor[DType.float16, 1]
    var gate_w: Tensor[DType.float16, 2]
    var up_w: Tensor[DType.float16, 2]
    var down_w: Tensor[DType.float16, 2]
    # per-head Q/K norms (hunyuan-dense, qwen35 full attention)
    var attn_q_norm: Tensor[DType.float16, 1]
    var attn_k_norm: Tensor[DType.float16, 1]
    # qwen35 recurrent-layer tensors (attn_qkv doubles as wqkv; attn_gate
    # is the z output-gate projection)
    var attn_gate: Tensor[DType.float16, 2]
    var ssm_conv1d: Tensor[DType.float16, 2]  # [channels, kernel]
    var ssm_dt: Tensor[DType.float16, 1]
    var ssm_a: Tensor[DType.float16, 1]
    var ssm_beta: Tensor[DType.float16, 2]
    var ssm_alpha: Tensor[DType.float16, 2]
    var ssm_norm: Tensor[DType.float16, 1]
    var ssm_out: Tensor[DType.float16, 2]

    def __init__(out self):
        self.attn_norm_w = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.q_w = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.k_w = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.v_w = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.o_w = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.q_b = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.k_b = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.v_b = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.ffn_norm_w = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.post_attn_norm_w = Tensor[DType.float16, 1](
            StaticTuple[Int, 1](0)
        )
        self.gate_w = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.up_w = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.down_w = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.attn_q_norm = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.attn_k_norm = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.attn_gate = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.ssm_conv1d = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.ssm_dt = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.ssm_a = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.ssm_beta = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.ssm_alpha = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.ssm_norm = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.ssm_out = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))


struct TransformerWeights(Movable):
    var token_embd: Tensor[DType.float16, 2]
    var output_norm_w: Tensor[DType.float16, 1]
    var output_w: Tensor[DType.float16, 2]
    var layers: List[LayerWeights]

    def __init__(out self):
        self.token_embd = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.output_norm_w = Tensor[DType.float16, 1](StaticTuple[Int, 1](0))
        self.output_w = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.layers = List[LayerWeights]()


struct SSMLayerState(Movable):
    """One recurrent (Gated DeltaNet) layer's persistent state.

    `conv_state` keeps the last (kernel-1) pre-activation timesteps per
    conv channel [channels, kernel-1] (fp16); `state` is the per-value-head
    [S x S] recurrence matrix (fp32, [n_v_heads, S, S]).
    """

    var conv_state: Tensor[DType.float16, 2]
    var state: Tensor[DType.float32, 3]
    var enabled: Bool

    def __init__(out self):
        self.conv_state = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.state = Tensor[DType.float32, 3](StaticTuple[Int, 3](0, 0, 0))
        self.enabled = False

    def setup(mut self, channels: Int, kernel: Int, n_heads: Int, s: Int):
        self.conv_state = tensor_zeros[DType.float16, 2](
            StaticTuple[Int, 2](channels, kernel - 1)
        )
        self.state = tensor_zeros[DType.float32, 3](
            StaticTuple[Int, 3](n_heads, s, s)
        )
        self.enabled = True

    def reset(mut self):
        for i in range(self.conv_state.numel()):
            self.conv_state.set(i, Scalar[DType.float16](Float32(0)))
        for i in range(self.state.numel()):
            self.state.set(i, Scalar[DType.float32](Float32(0)))


struct TransformerModel(Movable):
    var config: TransformerConfig
    var ctx: GGUFContext
    var weights: Dict[String, GGUFTensor]  # M2 name->info table (kept)
    var params: TransformerWeights  # dequantized fp16 weights
    var cache: KVCache
    var ssm_states: List[SSMLayerState]  # per recurrent layer (qwen35)

    def __init__(
        out self,
        config: TransformerConfig,
        var ctx: GGUFContext,
        kv_cache_len: Int = DEFAULT_KV_CACHE_LEN,
    ):
        self.config = config
        self.ctx = ctx^
        self.weights = Dict[String, GGUFTensor]()
        self.params = TransformerWeights()
        self.cache = KVCache()
        self.ssm_states = List[SSMLayerState]()
        self.load_weights()
        self.cache = KVCache(config.n_layers, config.n_kv_heads, kv_cache_len, config.head_dim)
        # qwen35: recurrent layers keep no KV cache and carry SSM state.
        if config.arch == ARCH_QWEN35:
            for l in range(config.n_layers):
                if config.is_recurrent(l):
                    self.cache.layers[l] = KVCacheLayer(
                        config.n_kv_heads, 0, config.head_dim
                    )
                    var st = SSMLayerState()
                    st.setup(
                        2 * config.ssm_n_group * config.ssm_d_state
                        + config.ssm_d_inner,
                        config.ssm_d_conv,
                        config.ssm_dt_rank,
                        config.ssm_d_state,
                    )
                    self.ssm_states.append(st^)
                else:
                    var st = SSMLayerState()
                    self.ssm_states.append(st^)

    # -- weight loading -----------------------------------------------------

    def load_weights(mut self):
        var cfg = self.config
        var params = TransformerWeights()
        params.token_embd = dequantize_weight(
            self.ctx, find_tensor(self.ctx, "token_embd.weight").value()
        )
        params.output_norm_w = dequantize_vector(
            self.ctx, find_tensor(self.ctx, "output_norm.weight").value()
        )
        var output_t = find_tensor(self.ctx, "output.weight")
        if output_t:
            params.output_w = dequantize_weight(
                self.ctx, output_t.value()
            )
        else:
            params.output_w = params.token_embd  # tied embeddings
        params.layers = List[LayerWeights]()
        for i in range(cfg.n_layers):
            var lw = LayerWeights()
            var base = "blk." + String(i)
            lw.attn_norm_w = dequantize_vector(
                self.ctx,
                find_tensor(self.ctx, base + ".attn_norm.weight").value(),
            )
            lw.ffn_norm_w = dequantize_vector_opt(
                self.ctx,
                find_tensor(self.ctx, base + ".ffn_norm.weight"),
            )
            var post_norm = find_tensor(
                self.ctx, base + ".post_attention_norm.weight"
            )
            if post_norm:
                lw.post_attn_norm_w = dequantize_vector(
                    self.ctx, post_norm.value()
                )
            lw.gate_w = dequantize_weight(
                self.ctx,
                find_tensor(self.ctx, base + ".ffn_gate.weight").value(),
            )
            lw.up_w = dequantize_weight(
                self.ctx,
                find_tensor(self.ctx, base + ".ffn_up.weight").value(),
            )
            lw.down_w = dequantize_weight(
                self.ctx,
                find_tensor(self.ctx, base + ".ffn_down.weight").value(),
            )
            if cfg.arch == ARCH_QWEN35 and cfg.is_recurrent(i):
                lw.q_w = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".attn_qkv.weight").value(),
                )
                lw.attn_gate = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".attn_gate.weight").value(),
                )
                lw.ssm_conv1d = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".ssm_conv1d.weight").value(),
                )
                lw.ssm_dt = dequantize_vector(
                    self.ctx,
                    find_tensor(self.ctx, base + ".ssm_dt.bias").value(),
                )
                lw.ssm_a = dequantize_vector(
                    self.ctx,
                    find_tensor(self.ctx, base + ".ssm_a").value(),
                )
                lw.ssm_beta = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".ssm_beta.weight").value(),
                )
                lw.ssm_alpha = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".ssm_alpha.weight").value(),
                )
                lw.ssm_norm = dequantize_vector(
                    self.ctx,
                    find_tensor(self.ctx, base + ".ssm_norm.weight").value(),
                )
                lw.ssm_out = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".ssm_out.weight").value(),
                )
            else:
                lw.q_w = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".attn_q.weight").value(),
                )
                lw.k_w = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".attn_k.weight").value(),
                )
                lw.v_w = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".attn_v.weight").value(),
                )
                lw.o_w = dequantize_weight(
                    self.ctx,
                    find_tensor(self.ctx, base + ".attn_output.weight").value(),
                )
                lw.q_b = dequantize_vector_opt(
                    self.ctx, find_tensor(self.ctx, base + ".attn_q.bias")
                )
                lw.k_b = dequantize_vector_opt(
                    self.ctx, find_tensor(self.ctx, base + ".attn_k.bias")
                )
                lw.v_b = dequantize_vector_opt(
                    self.ctx, find_tensor(self.ctx, base + ".attn_v.bias")
                )
                var q_norm = find_tensor(self.ctx, base + ".attn_q_norm.weight")
                if q_norm:
                    lw.attn_q_norm = dequantize_vector(
                        self.ctx, q_norm.value()
                    )
                var k_norm = find_tensor(self.ctx, base + ".attn_k_norm.weight")
                if k_norm:
                    lw.attn_k_norm = dequantize_vector(
                        self.ctx, k_norm.value()
                    )
            params.layers.append(lw^)
        self.params = params^

    # -- forward ------------------------------------------------------------

    def reset_cache(mut self):
        self.cache.reset()
        for i in range(len(self.ssm_states)):
            if self.ssm_states[i].enabled:
                self.ssm_states[i].reset()

    def forward(mut self, token: Int, position: Int) raises -> Tensor[DType.float32, 1]:
        """One autoregressive step: `token` at absolute `position`.

        Returns the f32 logits [vocab] for sampling the next token.
        """
        var cfg = self.config
        var x = self.forward_hidden(token, position)
        x = rms_norm_weight[DType.float16](
            x, self.params.output_norm_w, cfg.norm_eps
        )
        var logits16 = matmul_weight_cpu_threaded[DType.float16](
            x, self.params.output_w
        )

        var logits = tensor_zeros[DType.float32, 1](
            StaticTuple[Int, 1](cfg.vocab)
        )
        for i in range(cfg.vocab):
            logits.set(i, Scalar[DType.float32](Float32(logits16.get(i))))
        return logits

    def forward_hidden(
        mut self, token: Int, position: Int
    ) raises -> Tensor[DType.float16, 2]:
        """M7: the transformer stack only (pre-LM-head hidden state).

        Used by the inference-time fine-tuning API (finetune adapters hook
        into the final hidden state).
        """
        var cfg = self.config
        var toks = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](1))
        toks.set(0, Scalar[DType.int32](token))
        var x = embedding_cpu_dynamic[DType.float16](
            toks, self.params.token_embd
        )

        for layer in range(cfg.n_layers):
            if cfg.is_recurrent(layer):
                x = self._layer_forward_ssm(layer, x)
            elif cfg.arch == ARCH_HUNYUAN:
                x = self._layer_forward_hunyuan(layer, x, position)
            elif cfg.arch == ARCH_QWEN35:
                x = self._layer_forward_q35_attn(layer, x, position)
            else:
                x = self._layer_forward(layer, x, position)
        return x

    def _layer_forward(
        mut self, layer: Int, x: Tensor[DType.float16, 2], position: Int
    ) -> Tensor[DType.float16, 2]:
        """Qwen2 layer (the M3 path, unchanged)."""
        var cfg = self.config
        var lw = self.params.layers[layer]
        var normed = rms_norm_weight[DType.float16](
            x, lw.attn_norm_w, cfg.norm_eps
        )
        var attn = mha_forward[DType.float16](
            normed, lw.q_w, lw.k_w, lw.v_w, lw.o_w, lw.q_b, lw.k_b, lw.v_b,
            self.cache.layers[layer], position, cfg.n_heads, cfg.n_kv_heads,
            cfg.head_dim, cfg.rope_theta,
        )
        var resid = add_cpu_dynamic[DType.float16](x, attn)
        var normed2 = rms_norm_weight[DType.float16](
            resid, lw.ffn_norm_w, cfg.norm_eps
        )
        return _ffn_swiglu(normed2, lw, resid, cfg.ffn)

    def _layer_forward_hunyuan(
        mut self, layer: Int, x: Tensor[DType.float16, 2], position: Int
    ) -> Tensor[DType.float16, 2]:
        """hunyuan-dense layer: RoPE *before* per-head Q/K RMSNorm."""
        var cfg = self.config
        var lw = self.params.layers[layer]
        var normed = rms_norm_weight[DType.float16](
            x, lw.attn_norm_w, cfg.norm_eps
        )
        var opts = MHAOptions()
        opts.q_norm = True
        opts.k_norm = True
        opts.norm_before_rope = False
        opts.norm_eps = cfg.norm_eps
        var attn = mha_forward_v2[DType.float16](
            normed, lw.q_w, lw.k_w, lw.v_w, lw.o_w, lw.q_b, lw.k_b, lw.v_b,
            lw.attn_q_norm, lw.attn_k_norm, self.cache.layers[layer],
            position, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
            cfg.rope_theta, opts,
        )
        var resid = add_cpu_dynamic[DType.float16](x, attn)
        var normed2 = rms_norm_weight[DType.float16](
            resid, lw.ffn_norm_w, cfg.norm_eps
        )
        return _ffn_swiglu(normed2, lw, resid, cfg.ffn)

    def _layer_forward_q35_attn(
        mut self, layer: Int, x: Tensor[DType.float16, 2], position: Int
    ) -> Tensor[DType.float16, 2]:
        """qwen35 full-attention layer (QK-norm + MRoPE + sigmoid gate)."""
        var cfg = self.config
        var lw = self.params.layers[layer]
        var normed = rms_norm_weight[DType.float16](
            x, lw.attn_norm_w, cfg.norm_eps
        )
        var opts = MHAOptions()
        opts.q_norm = True
        opts.k_norm = True
        opts.norm_before_rope = True
        opts.gate = True
        opts.n_rot = cfg.n_rot
        opts.norm_eps = cfg.norm_eps
        var attn = mha_forward_v2[DType.float16](
            normed, lw.q_w, lw.k_w, lw.v_w, lw.o_w, lw.q_b, lw.k_b, lw.v_b,
            lw.attn_q_norm, lw.attn_k_norm, self.cache.layers[layer],
            position, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
            cfg.rope_theta, opts,
        )
        var resid = add_cpu_dynamic[DType.float16](x, attn)
        var normed2 = rms_norm_weight[DType.float16](
            resid, lw.post_attn_norm_w, cfg.norm_eps
        )
        return _ffn_swiglu(normed2, lw, resid, cfg.ffn)

    def _layer_forward_ssm(
        mut self, layer: Int, x: Tensor[DType.float16, 2]
    ) raises -> Tensor[DType.float16, 2]:
        """qwen35 Gated DeltaNet layer (single token).

        attn_norm -> wqkv/z/alpha/beta projections -> causal conv1d (SiLU)
        -> q/k L2 norm -> decayed delta-rule recurrence per value head ->
        RMSNorm(o, ssm_norm) * silu(z) -> ssm_out -> residual ->
        post_attention_norm -> SwiGLU FFN -> residual.
        """
        var cfg = self.config
        var lw = self.params.layers[layer]
        var n_k = cfg.ssm_n_group
        var n_v = cfg.ssm_dt_rank
        var s = cfg.ssm_d_state
        var n_kv = n_k * s  # 2048 = 16 heads * 128
        var n_vv = cfg.ssm_d_inner  # 6144 = 48 heads * 128
        var gated = tensor_zeros[DType.float16, 2](
            StaticTuple[Int, 2](1, n_vv)
        )
        var normed = rms_norm_weight[DType.float16](
            x, lw.attn_norm_w, cfg.norm_eps
        )
        # z gate (attn_gate) + wqkv in one threaded pass
        var qkv = tensor_zeros[DType.float16, 2](
            StaticTuple[Int, 2](1, 2 * n_kv + n_vv)
        )
        var z = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, n_vv))
        matmul_weight_2_threaded[DType.float16](
            normed, lw.q_w, lw.attn_gate, qkv, z
        )
        var alpha = matmul_weight_cpu_threaded[DType.float16](
            normed, lw.ssm_alpha
        )
        var beta = matmul_weight_cpu_threaded[DType.float16](
            normed, lw.ssm_beta
        )
        alpha = add_row_cpu[DType.float16](alpha, lw.ssm_dt)

        # causal depthwise conv (kernel taps over [state..., current])
        var conv_out = tensor_zeros[DType.float16, 2](
            StaticTuple[Int, 2](1, 2 * n_kv + n_vv)
        )
        var channels = 2 * n_kv + n_vv
        var kernel = lw.ssm_conv1d.shape()[1]
        var wq = lw.ssm_conv1d
        for c in range(channels):
            var acc = Float32(0)
            for tap in range(kernel - 1):
                acc += Float32(wq.get(c * kernel + tap)) * Float32(
                    self.ssm_states[layer].conv_state.get(
                        c * (kernel - 1) + tap
                    )
                )
            acc += Float32(wq.get(c * kernel + kernel - 1)) * Float32(
                qkv.get(c)
            )
            # shift the state ring left and store the new input
            for tap in range(kernel - 2):
                self.ssm_states[layer].conv_state.set(
                    c * (kernel - 1) + tap,
                    Scalar[DType.float16](
                        Float32(
                            self.ssm_states[layer].conv_state.get(
                                c * (kernel - 1) + tap + 1
                            )
                        )
                    ),
                )
            self.ssm_states[layer].conv_state.set(
                c * (kernel - 1) + kernel - 2,
                Scalar[DType.float16](Float32(qkv.get(c))),
            )
            var y = _silu_f32(acc)
            conv_out.set(c, Scalar[DType.float16](y))

        # recurrence: q/k L2-normalized (per head), q scaled by 1/sqrt(S)
        # M7 perf: hoist the raw fp32 state pointer + the fp16 conv pointers
        # out of the per-element loops (List indexing costs ~10x here).
        var k_sqrt = Float32(1.0) / sqrt(Float32(s))
        var q_scale = k_sqrt
        var state_p = self.ssm_states[layer].state.data()
        var conv_p = conv_out.data()
        var qkv_p = conv_out.data()
        for h in range(n_v):
            var kh = h % n_k
            # gamma_h = exp(softplus(alpha_h) * a_h)   (a_h = ssm_a[h])
            var a_h = Float32(lw.ssm_a.get(h))
            var g = _softplus_f32(Float32(alpha.get(h))) * a_h
            var gamma = exp(g)
            var beta_h = _sigmoid_f32(Float32(beta.get(h)))
            # L2 norms of q/k heads
            var q_ss = Float32(0)
            var k_ss = Float32(0)
            for i in range(s):
                var qv = Float32(conv_out.get(kh * s + i))
                var kv = Float32(conv_out.get(n_kv + kh * s + i))
                q_ss += qv * qv
                k_ss += kv * kv
            var q_inv = Float32(1.0) / sqrt(q_ss + cfg.norm_eps)
            var k_inv = Float32(1.0) / sqrt(k_ss + cfg.norm_eps)
            # S <- gamma * S ;  d = beta*(v - S^T k) ; S += k d^T ; o = S^T q
            var o = List[Float32]()
            var state_base = (h * s) * s
            for i in range(s):
                # decay + delta residual (pointer-based inner loops)
                var sk_dot = Float32(0)
                for j in range(s):
                    var sv = Float32(
                        state_p.unsafe_load[width=1](
                            offset=state_base + j * s + i
                        )
                    )
                    var kv = Float32(
                        conv_p.unsafe_load[width=1](offset=n_kv + kh * s + j)
                    ) * k_inv
                    state_p.unsafe_store(
                        state_base + j * s + i,
                        Scalar[DType.float32](sv * gamma),
                    )
                    sk_dot += sv * kv
                var vv = Float32(
                    conv_p.unsafe_load[width=1](offset=2 * n_kv + h * s + i)
                )
                var d = beta_h * (vv - sk_dot)
                for j in range(s):
                    var kv = Float32(
                        conv_p.unsafe_load[width=1](offset=n_kv + kh * s + j)
                    )
                    state_p.unsafe_store(
                        state_base + j * s + i,
                        Scalar[DType.float32](
                            Float32(
                                state_p.unsafe_load[width=1](
                                    offset=state_base + j * s + i
                                )
                            )
                            + (kv * k_inv) * d
                        ),
                    )
                # output: o_i = S^T q  (using the post-update state)
                var acc = Float32(0)
                for j in range(s):
                    acc += Float32(
                        state_p.unsafe_load[width=1](
                            offset=state_base + j * s + i
                        )
                    ) * Float32(
                        qkv_p.unsafe_load[width=1](offset=kh * s + j)
                    )
                o.append(acc * q_inv * q_scale)
            # gated output norm: RMSNorm(o, ssm_norm) * silu(z)
            var o_ss = Float32(0)
            for i in range(s):
                o_ss += o[i] * o[i]
            var o_inv = Float32(1.0) / sqrt(o_ss / Float32(s) + cfg.norm_eps)
            for i in range(s):
                var z_act = _silu_f32(Float32(z.get(h * s + i)))
                var wv = Float32(lw.ssm_norm.get(i))
                gated.set(
                    h * s + i,
                    Scalar[DType.float16](o[i] * o_inv * wv * z_act),
                )


        var attn_out = matmul_weight_cpu_threaded[DType.float16](
            gated, lw.ssm_out
        )
        var resid = add_cpu_dynamic[DType.float16](x, attn_out)
        var normed2 = rms_norm_weight[DType.float16](
            resid, lw.post_attn_norm_w, cfg.norm_eps
        )
        return _ffn_swiglu(normed2, lw, resid, cfg.ffn)


# -- shared helpers ----------------------------------------------------------


def _ffn_swiglu(
    normed: Tensor[DType.float16, 2],
    lw: LayerWeights,
    resid: Tensor[DType.float16, 2],
    ffn: Int,
) -> Tensor[DType.float16, 2]:
    var g = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, ffn))
    var u = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, ffn))
    matmul_weight_2_threaded[DType.float16](normed, lw.gate_w, lw.up_w, g, u)
    var h = swiglu_cpu_dynamic[DType.float16](g, u)
    var d = matmul_weight_cpu_threaded[DType.float16](h, lw.down_w)
    return add_cpu_dynamic[DType.float16](resid, d)


def _sigmoid_f32(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))


def _silu_f32(x: Float32) -> Float32:
    return x * _sigmoid_f32(x)


def _softplus_f32(x: Float32) -> Float32:
    # log(1 + exp(x)), stabilized
    if x > Float32(20.0):
        return x
    if x < Float32(-20.0):
        return Float32(0)
    return log(Float32(1.0) + exp(x))


def rms_norm_weight[dtype: DType](
    x: Tensor[dtype, 2], weight: Tensor[dtype, 1], eps: Float32
) -> Tensor[dtype, 2]:
    """RMSNorm with a learned weight: out = x / sqrt(mean(x^2) + eps) * w.

    (Qwen2 applies the norm *weight* multiplicatively after normalizing;
    this is separate from the unweighted M1 `rms_norm_cpu` kernel.)
    M5: SIMD main loops with a scalar tail.
    """
    var rows = x.shape()[0]
    var cols = x.shape()[1]
    if weight.shape()[0] != cols:
        return tensor_zeros[dtype, 2](x.shape())
    var out = tensor_zeros[dtype, 2](x.shape())
    comptime W = 8 if dtype == DType.float16 else 4
    var cols_main = (cols // W) * W
    for i in range(rows):
        var base = i * cols
        var acc = SIMD[DType.float32, W](0)
        var j = 0
        while j < cols_main:
            var v = x.data().unsafe_load[width=W](
                offset=base + j
            ).cast[DType.float32]()
            acc = acc + v * v
            j += W
        var ss = Float32(acc.reduce_add())
        while j < cols:
            var v = Float32(x.get(base + j))
            ss += v * v
            j += 1
        var inv = Float32(1.0) / sqrt(ss / Float32(cols) + eps)
        j = 0
        while j < cols_main:
            var v = x.data().unsafe_load[width=W](
                offset=base + j
            ).cast[DType.float32]()
            var wv = weight.data().unsafe_load[width=W](
                offset=j
            ).cast[DType.float32]()
            out.data().unsafe_store(
                base + j,
                (v * SIMD[DType.float32, W](inv) * wv).cast[dtype](),
            )
            j += W
        while j < cols:
            var v = Float32(x.get(base + j)) * inv * Float32(weight.get(j))
            out.set(base + j, Scalar[dtype](v))
            j += 1
    return out


def tensor_numel(tensor: GGUFTensor) -> Int:
    var total = 1
    for i in range(tensor.n_dims):
        total *= tensor.dims[i]
    return total


def dequantize_weight(
    ctx: GGUFContext, tensor: GGUFTensor
) -> Tensor[DType.float16, 2]:
    """Dequantize a GGUF weight into a rank-2 fp16 tensor.

    GGUF dims are ggml-ordered (innermost/fastest first), so the matrix the
    model actually computes with is `data` viewed as (dims[1], dims[0]) -
    i.e. W[j, i] = data[j*dims[0] + i], shape [out, in].  We materialize
    exactly that shape so `matmul_weight` (y = W @ x) reads contiguous rows.
    Rank-1 tensors (norms, biases) are materialized as `[d0, 1]`.
    """
    var numel = tensor_numel(tensor)
    var shape = StaticTuple[Int, 2](0, 0)
    if tensor.n_dims >= 2:
        shape = StaticTuple[Int, 2](tensor.dims[1], tensor.dims[0])
    else:
        shape = StaticTuple[Int, 2](tensor.dims[0], 1)
    var out = tensor_zeros[DType.float16, 2](shape)
    dequantize_into(
        tensor.ggml_type,
        ctx.data,
        ctx.data_offset + tensor.offset,
        out,
        numel,
    )
    return out


def dequantize_vector(
    ctx: GGUFContext, tensor: GGUFTensor
) -> Tensor[DType.float16, 1]:
    """Dequantize a rank-1 GGUF tensor into a rank-1 fp16 tensor."""
    var numel = tensor_numel(tensor)
    var tmp = tensor_zeros[DType.float16, 2](
        StaticTuple[Int, 2](1, numel)
    )
    dequantize_into(
        tensor.ggml_type,
        ctx.data,
        ctx.data_offset + tensor.offset,
        tmp,
        numel,
    )
    return tmp.reshape[1](StaticTuple[Int, 1](numel))


def dequantize_vector_opt(
    ctx: GGUFContext, tensor: Optional[GGUFTensor]
) -> Tensor[DType.float16, 1]:
    if tensor:
        return dequantize_vector(ctx, tensor.value())
    return Tensor[DType.float16, 1](StaticTuple[Int, 1](0))


def _arch_from_gguf(ctx: GGUFContext) -> Int8:
    var arch = get_meta_str(ctx, "general.architecture", String("qwen2"))
    if arch == "hunyuan-dense":
        return ARCH_HUNYUAN
    if arch == "qwen35":
        return ARCH_QWEN35
    return ARCH_QWEN2


def load_config(ctx: GGUFContext) -> TransformerConfig:
    """Extract the architecture config from GGUF metadata."""
    var config = TransformerConfig()
    config.arch = _arch_from_gguf(ctx)
    var prefix = "qwen2."
    if config.arch == ARCH_HUNYUAN:
        prefix = "hunyuan-dense."
    elif config.arch == ARCH_QWEN35:
        prefix = "qwen35."
    config.n_layers = get_meta_uint(ctx, prefix + "block_count", 0)
    config.hidden = get_meta_uint(ctx, prefix + "embedding_length", 0)
    config.ffn = get_meta_uint(ctx, prefix + "feed_forward_length", 0)
    config.n_heads = get_meta_uint(ctx, prefix + "attention.head_count", 0)
    config.n_kv_heads = get_meta_uint(
        ctx, prefix + "attention.head_count_kv", 0
    )
    config.rope_theta = Float32(
        get_meta_float(ctx, prefix + "rope.freq_base", 10000.0)
    )
    config.norm_eps = Float32(
        get_meta_float(
            ctx, prefix + "attention.layer_norm_rms_epsilon", 1e-6
        )
    )
    config.bos_id = get_meta_uint(ctx, "tokenizer.ggml.bos_token_id", 1)
    config.eos_id = get_meta_uint(ctx, "tokenizer.ggml.eos_token_id", 2)
    if config.arch == ARCH_QWEN35:
        config.ssm_d_conv = get_meta_uint(ctx, prefix + "ssm.conv_kernel", 4)
        config.ssm_d_state = get_meta_uint(ctx, prefix + "ssm.state_size", 0)
        config.ssm_dt_rank = get_meta_uint(
            ctx, prefix + "ssm.time_step_rank", 0
        )
        config.ssm_n_group = get_meta_uint(ctx, prefix + "ssm.group_count", 0)
        config.ssm_d_inner = get_meta_uint(ctx, prefix + "ssm.inner_size", 0)
        config.full_attn_interval = get_meta_uint(
            ctx, prefix + "full_attention_interval", 4
        )
        config.n_nextn = get_meta_uint(ctx, prefix + "nextn_predict_layers", 0)
        config.n_rot = get_meta_uint(ctx, prefix + "rope.dimension_count", 0)
        var sec = _read_meta_int_array(
            ctx, prefix + "rope.dimension_sections"
        )
        for i in range(4):
            if i < len(sec):
                config.rope_sections[i] = sec[i]
        # the MTP block is part of block_count; drop it (main layers only)
        config.n_layers = config.n_layers - config.n_nextn
    # Vocab size is the second dim of the embedding matrix.
    var embedding = find_tensor(ctx, "token_embd.weight")
    if embedding:
        config.vocab = embedding.value().dims[1]
    # head_dim: qwen35/hunyuan-dense store the per-head key length in the
    # metadata (hidden/n_heads is not integral for qwen35: 5120/24).
    var key_len = get_meta_uint(ctx, prefix + "attention.key_length", 0)
    if key_len > 0:
        config.head_dim = key_len
    elif config.n_heads > 0:
        config.head_dim = config.hidden // config.n_heads
    return config


def _read_meta_int_array(ctx: GGUFContext, key: String) -> List[Int]:
    from .gguf_loader import GGUFMetaValue, Reader

    var value = ctx.metadata.get(key, GGUFMetaValue())
    if value.kind != 5 or value.arr_len <= 0:
        return List[Int]()
    var reader = Reader(ctx.data)
    reader.offset = value.arr_offset
    var out = List[Int]()
    for i in range(value.arr_len):
        var raw = reader.read_u32()
        if value.arr_type == 5:  # INT32
            out.append(Int(bitcast[DType.int32](raw)))
        else:
            out.append(Int(raw))
    return out^


def collect_weights(ctx: GGUFContext) -> Dict[String, GGUFTensor]:
    var weights = Dict[String, GGUFTensor]()
    for tensor in ctx.tensors:
        weights[tensor.name] = tensor
    return weights^


def build_graph(model: TransformerModel) -> Graph:
    """Build the decoder graph node structure (M2 op skeleton).

    The typed forward in `TransformerModel.forward` is the execution path
    used for inference; this graph mirrors the same topology for the
    interpreter/registry layer.
    """
    var graph = Graph()
    var cfg = model.config

    var embed_attrs = Dict[String, AttrValue]()
    var embed_id = graph.add_node("embedding", List[Int](), embed_attrs)

    var current = embed_id
    for layer in range(cfg.n_layers):
        var attrs = Dict[String, AttrValue]()
        attrs["layer"] = AttrValue(layer)

        # input RMSNorm
        var norm1 = graph.add_node("rms_norm", [current], attrs)
        # multi-head attention (qkv projection + rope + attention + out)
        var attn = graph.add_node("mha", [norm1], attrs)
        var resid1 = graph.add_node("add", [current, attn], attrs)
        # FFN RMSNorm
        var norm2 = graph.add_node("rms_norm", [resid1], attrs)
        # SwiGLU feed-forward (gate/up matmuls + silu-mul + down matmul)
        var ffn = graph.add_node("swiglu_ffn", [norm2], attrs)
        var resid2 = graph.add_node("add", [resid1, ffn], attrs)
        current = resid2

    var final_attrs = Dict[String, AttrValue]()
    var final_norm = graph.add_node("rms_norm", [current], final_attrs)
    var lm_head = graph.add_node("lm_head", [final_norm], final_attrs)
    _ = lm_head
    return graph^
