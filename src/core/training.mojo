# core/training.mojo
#
# M6 Phase 5: the Mojo-layer training API.
#
# `TrainModel` owns a small decoder-style transformer whose compute graph is
# built from the registered operators (embedding -> per-layer
# [rms_norm_weight -> mha_seq -> add -> rms_norm_weight -> swiglu_ffn -> add]
# -> rms_norm_weight -> lm_head -> cross_entropy) and executed through the
# interpreter's `run_with_grad`.  Parameters are fp32 masters (AMP keeps a
# parallel fp16 shadow set for the graph run); the AdamW optimizer maintains
# the fp32 gradient buffers and per-parameter state.
#
#   train_step(tokens, targets, accumulation_steps) -> (loss, grads)
#   eval_step(tokens, targets) -> (loss, accuracy)
#
# Gradient accumulation sums `accumulation_steps` mini-batches before one
# optimizer step; with AMP the scaled gradients are unscaled right before
# the step and the GradScaler backs off on Inf/NaN.

from .tensor import Tensor, tensor_zeros
from .graph import Graph, AttrValue
from .ops.base.op_interface import AnyTensor, to_any, from_any
from .ops.base.op_registry import OpRegistry
from .ops.base.op_autograd import no_grad_any
from .train_optimizer import AdamW
from .gradient_scaler import GradScaler
from .utils import unimplemented
from std.utils.static_tuple import StaticTuple
from std.math import sqrt
from ..runtime.interpreter import Interpreter


struct TrainConfig(Copyable, ImplicitlyCopyable, Movable):
    var n_layers: Int
    var hidden: Int
    var ffn: Int
    var n_heads: Int
    var n_kv_heads: Int
    var vocab: Int
    var head_dim: Int
    var rope_theta: Float32
    var eps: Float32

    def __init__(out self):
        self.n_layers = 2
        self.hidden = 32
        self.ffn = 64
        self.n_heads = 2
        self.n_kv_heads = 2
        self.vocab = 64
        self.head_dim = 16
        self.rope_theta = Float32(10000.0)
        self.eps = Float32(1e-5)


struct LayerParams(Copyable, ImplicitlyCopyable, Movable):
    var attn_norm_w: Tensor[DType.float32, 1]
    var q_w: Tensor[DType.float32, 2]
    var k_w: Tensor[DType.float32, 2]
    var v_w: Tensor[DType.float32, 2]
    var o_w: Tensor[DType.float32, 2]
    var q_b: Tensor[DType.float32, 1]
    var k_b: Tensor[DType.float32, 1]
    var v_b: Tensor[DType.float32, 1]
    var ffn_norm_w: Tensor[DType.float32, 1]
    var gate_w: Tensor[DType.float32, 2]
    var up_w: Tensor[DType.float32, 2]
    var down_w: Tensor[DType.float32, 2]

    def __init__(out self):
        self.attn_norm_w = Tensor[DType.float32, 1](StaticTuple[Int, 1](0))
        self.q_w = Tensor[DType.float32, 2](StaticTuple[Int, 2](0, 0))
        self.k_w = Tensor[DType.float32, 2](StaticTuple[Int, 2](0, 0))
        self.v_w = Tensor[DType.float32, 2](StaticTuple[Int, 2](0, 0))
        self.o_w = Tensor[DType.float32, 2](StaticTuple[Int, 2](0, 0))
        self.q_b = Tensor[DType.float32, 1](StaticTuple[Int, 1](0))
        self.k_b = Tensor[DType.float32, 1](StaticTuple[Int, 1](0))
        self.v_b = Tensor[DType.float32, 1](StaticTuple[Int, 1](0))
        self.ffn_norm_w = Tensor[DType.float32, 1](StaticTuple[Int, 1](0))
        self.gate_w = Tensor[DType.float32, 2](StaticTuple[Int, 2](0, 0))
        self.up_w = Tensor[DType.float32, 2](StaticTuple[Int, 2](0, 0))
        self.down_w = Tensor[DType.float32, 2](StaticTuple[Int, 2](0, 0))


def _lcg_next(mut state: Int) -> Float32:
    state = (state * 1103515245 + 12345) % 2147483648
    return Float32(state) / Float32(2147483648.0)


def _fill_weight(mut t: Tensor[DType.float32, 2], mut state: Int):
    var cols = t.shape()[1]
    if cols < 1:
        cols = 1
    var bound = Float32(1.0) / sqrt(Float32(cols))
    for i in range(t.numel()):
        var u = _lcg_next(state)
        t.set(i, Scalar[DType.float32]((u * 2.0 - 1.0) * bound))


struct TrainModel(Movable):
    var cfg: TrainConfig
    var token_embd: Tensor[DType.float32, 2]
    var output_norm_w: Tensor[DType.float32, 1]
    var output_w: Tensor[DType.float32, 2]
    var layers: List[LayerParams]
    # graph plumbing
    var registry: OpRegistry
    var graph: Graph
    var interpreter: Interpreter
    var loss_node: Int
    var logits_node: Int
    # training state
    var opt: AdamW
    var scaler: GradScaler
    var amp: Bool
    var shadow: List[AnyTensor]  # fp16 copies of the params (AMP)
    var step_count: Int
    # fixed graph inputs (shared across calls)
    var cfg_tensor: Tensor[DType.int32, 1]
    var pos_theta: Tensor[DType.float32, 1]

    def __init__(out self, config: TrainConfig, seed: Int = 0):
        self.cfg = config
        var hidden = config.hidden
        var ffn = config.ffn
        var vocab = config.vocab
        var n_heads = config.n_heads
        var head_dim = config.head_dim

        self.token_embd = tensor_zeros[DType.float32, 2](
            StaticTuple[Int, 2](vocab, hidden)
        )
        self.output_norm_w = tensor_zeros[DType.float32, 1](
            StaticTuple[Int, 1](hidden)
        )
        self.output_w = tensor_zeros[DType.float32, 2](
            StaticTuple[Int, 2](vocab, hidden)
        )
        self.layers = List[LayerParams]()
        var layer_count = 0
        while layer_count < config.n_layers:
            var lp = LayerParams()
            lp.attn_norm_w = tensor_zeros[DType.float32, 1](
                StaticTuple[Int, 1](hidden)
            )
            lp.q_w = tensor_zeros[DType.float32, 2](
                StaticTuple[Int, 2](n_heads * head_dim, hidden)
            )
            lp.k_w = tensor_zeros[DType.float32, 2](
                StaticTuple[Int, 2](config.n_kv_heads * head_dim, hidden)
            )
            lp.v_w = tensor_zeros[DType.float32, 2](
                StaticTuple[Int, 2](config.n_kv_heads * head_dim, hidden)
            )
            lp.o_w = tensor_zeros[DType.float32, 2](
                StaticTuple[Int, 2](hidden, n_heads * head_dim)
            )
            lp.q_b = tensor_zeros[DType.float32, 1](
                StaticTuple[Int, 1](n_heads * head_dim)
            )
            lp.k_b = tensor_zeros[DType.float32, 1](
                StaticTuple[Int, 1](config.n_kv_heads * head_dim)
            )
            lp.v_b = tensor_zeros[DType.float32, 1](
                StaticTuple[Int, 1](config.n_kv_heads * head_dim)
            )
            lp.ffn_norm_w = tensor_zeros[DType.float32, 1](
                StaticTuple[Int, 1](hidden)
            )
            lp.gate_w = tensor_zeros[DType.float32, 2](
                StaticTuple[Int, 2](ffn, hidden)
            )
            lp.up_w = tensor_zeros[DType.float32, 2](
                StaticTuple[Int, 2](ffn, hidden)
            )
            lp.down_w = tensor_zeros[DType.float32, 2](
                StaticTuple[Int, 2](hidden, ffn)
            )
            self.layers.append(lp^)
            layer_count += 1

        self.cfg_tensor = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](3))
        self.cfg_tensor.set(0, Scalar[DType.int32](n_heads))
        self.cfg_tensor.set(1, Scalar[DType.int32](config.n_kv_heads))
        self.cfg_tensor.set(2, Scalar[DType.int32](head_dim))
        self.pos_theta = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](2))
        self.pos_theta.set(0, Scalar[DType.float32](Float32(0)))
        self.pos_theta.set(1, Scalar[DType.float32](config.rope_theta))

        self.registry = OpRegistry()
        self.registry.register_default_ops()
        self.graph = Graph()
        self.loss_node = 0
        self.logits_node = 0
        self.amp = False
        self.shadow = List[AnyTensor]()
        self.step_count = 0
        self.opt = AdamW(Float32(1e-3))
        self.scaler = GradScaler()
        self.interpreter = Interpreter(Graph(), OpRegistry())

        self._init_weights(seed)
        self._build_graph()
        self._register_params()
        # (re)attach the interpreter to the real graph; the interpreter
        # owns the graph/registry from here on, so leave fresh shells in
        # the model's fields (all graph access goes through the interpreter)
        self.interpreter = Interpreter(self.graph^, self.registry^)
        self.graph = Graph()
        self.registry = OpRegistry()

    # -- initialization ------------------------------------------------------

    def _init_weights(mut self, seed: Int):
        var state = seed
        if state < 1:
            state = 1
        _fill_weight(self.token_embd, state)
        _fill_weight(self.output_w, state)
        for l in range(self.cfg.n_layers):
            _fill_weight(self.layers[l].q_w, state)
            _fill_weight(self.layers[l].k_w, state)
            _fill_weight(self.layers[l].v_w, state)
            _fill_weight(self.layers[l].o_w, state)
            _fill_weight(self.layers[l].gate_w, state)
            _fill_weight(self.layers[l].up_w, state)
            _fill_weight(self.layers[l].down_w, state)
        for i in range(self.cfg.hidden):
            self.output_norm_w.set(i, Scalar[DType.float32](Float32(1.0)))
        for l in range(self.cfg.n_layers):
            for i in range(self.cfg.hidden):
                self.layers[l].attn_norm_w.set(
                    i, Scalar[DType.float32](Float32(1.0))
                )
                self.layers[l].ffn_norm_w.set(
                    i, Scalar[DType.float32](Float32(1.0))
                )

    # -- graph construction --------------------------------------------------

    def _build_graph(mut self):
        var graph = Graph()
        var embed_id = graph.add_node("embedding", List[Int](), _entry_attrs(2))
        # shared mha_seq config inputs
        var cfg_id = graph.add_node("identity", List[Int](), _entry_attrs(1))
        var pos_id = graph.add_node("identity", List[Int](), _entry_attrs(1))
        var current = embed_id
        var layer_idx = 0
        while layer_idx < self.cfg.n_layers:
            var wids = List[Int]()
            var wcount = 0
            while wcount < 12:
                var wnode = graph.add_node(
                    "identity", List[Int](), _entry_attrs(1)
                )
                wids.append(wnode)
                wcount += 1
            # attn_norm, q, k, v, o, qb, kb, vb, ffn_norm, gate, up, down
            var normed1 = graph.add_node(
                "rms_norm_weight", _in2(current, wids[0]), _no_attrs()
            )
            var attn_inputs = List[Int]()
            for k in range(1, 8):
                attn_inputs.append(wids[k])
            attn_inputs.insert(0, normed1)
            attn_inputs.append(cfg_id)
            attn_inputs.append(pos_id)
            var attn = graph.add_node("mha_seq", attn_inputs, _no_attrs())
            var resid1 = graph.add_node("add", _in2(current, attn), _no_attrs())
            var normed2 = graph.add_node(
                "rms_norm_weight", _in2(resid1, wids[8]), _no_attrs()
            )
            var ffn_inputs = List[Int]()
            ffn_inputs.append(normed2)
            ffn_inputs.append(wids[9])
            ffn_inputs.append(wids[10])
            ffn_inputs.append(wids[11])
            var ffn = graph.add_node("swiglu_ffn", ffn_inputs, _no_attrs())
            current = graph.add_node("add", _in2(resid1, ffn), _no_attrs())
            layer_idx += 1
        var out_norm_id = graph.add_node(
            "identity", List[Int](), _entry_attrs(1)
        )
        var out_w_id = graph.add_node("identity", List[Int](), _entry_attrs(1))
        var final_norm = graph.add_node(
            "rms_norm_weight", _in2(current, out_norm_id), _no_attrs()
        )
        var logits_id = graph.add_node(
            "lm_head", _in2(final_norm, out_w_id), _no_attrs()
        )
        var targets_id = graph.add_node(
            "identity", List[Int](), _entry_attrs(1)
        )
        var loss_id = graph.add_node(
            "cross_entropy", _in2(logits_id, targets_id), _no_attrs()
        )
        self.graph = graph^
        self.loss_node = loss_id
        self.logits_node = logits_id

    def _register_params(mut self):
        var opt = AdamW(Float32(1e-3))
        opt.add_param[DType.float32, 2](self.token_embd)
        for l in range(self.cfg.n_layers):
            opt.add_param[DType.float32, 1](self.layers[l].attn_norm_w)
            opt.add_param[DType.float32, 2](self.layers[l].q_w)
            opt.add_param[DType.float32, 2](self.layers[l].k_w)
            opt.add_param[DType.float32, 2](self.layers[l].v_w)
            opt.add_param[DType.float32, 2](self.layers[l].o_w)
            opt.add_param[DType.float32, 1](self.layers[l].q_b)
            opt.add_param[DType.float32, 1](self.layers[l].k_b)
            opt.add_param[DType.float32, 1](self.layers[l].v_b)
            opt.add_param[DType.float32, 1](self.layers[l].ffn_norm_w)
            opt.add_param[DType.float32, 2](self.layers[l].gate_w)
            opt.add_param[DType.float32, 2](self.layers[l].up_w)
            opt.add_param[DType.float32, 2](self.layers[l].down_w)
        opt.add_param[DType.float32, 1](self.output_norm_w)
        opt.add_param[DType.float32, 2](self.output_w)
        self.opt = opt^

    # -- input list assembly -------------------------------------------------

    def build_inputs(
        mut self,
        tokens: Tensor[DType.int32, 1],
        targets: Tensor[DType.int32, 1],
    ) -> List[AnyTensor]:
        """Assemble the external input list in the entry-node order."""
        var inputs = List[AnyTensor]()
        inputs.append(to_any[DType.int32, 1](tokens))
        inputs.append(to_any[DType.float32, 2](self.token_embd))
        inputs.append(to_any[DType.int32, 1](self.cfg_tensor))
        inputs.append(to_any[DType.float32, 1](self.pos_theta))
        for l in range(self.cfg.n_layers):
            inputs.append(to_any[DType.float32, 1](self.layers[l].attn_norm_w))
            inputs.append(to_any[DType.float32, 2](self.layers[l].q_w))
            inputs.append(to_any[DType.float32, 2](self.layers[l].k_w))
            inputs.append(to_any[DType.float32, 2](self.layers[l].v_w))
            inputs.append(to_any[DType.float32, 2](self.layers[l].o_w))
            inputs.append(to_any[DType.float32, 1](self.layers[l].q_b))
            inputs.append(to_any[DType.float32, 1](self.layers[l].k_b))
            inputs.append(to_any[DType.float32, 1](self.layers[l].v_b))
            inputs.append(to_any[DType.float32, 1](self.layers[l].ffn_norm_w))
            inputs.append(to_any[DType.float32, 2](self.layers[l].gate_w))
            inputs.append(to_any[DType.float32, 2](self.layers[l].up_w))
            inputs.append(to_any[DType.float32, 2](self.layers[l].down_w))
        inputs.append(to_any[DType.float32, 1](self.output_norm_w))
        inputs.append(to_any[DType.float32, 2](self.output_w))
        inputs.append(to_any[DType.int32, 1](targets))
        return inputs^

    # -- training / eval -----------------------------------------------------

    # -- training / eval -----------------------------------------------------

    def train_step(
        mut self,
        tokens: Tensor[DType.int32, 1],
        targets: Tensor[DType.int32, 1],
        accumulation_steps: Int = 1,
    ) -> Tuple[Float32, List[AnyTensor]]:
        """One forward+backward pass with optimizer bookkeeping.

        Returns (loss, parameter gradients).  The optimizer step fires every
        `accumulation_steps` calls.
        """
        var inputs = self.build_inputs(tokens, targets)
        var result = self.interpreter.run_with_grad(inputs, self.loss_node)
        var loss = Float32(0)
        if len(result[0]) > 0:
            var loss_t = result[0][0]
            if loss_t.dtype == DType.float32:
                loss = Float32(
                    loss_t.data.unsafe_bitcast[
                        Scalar[DType.float32]
                    ]().unsafe_load[width=1](offset=0)
                )
            else:
                loss = Float32(
                    loss_t.data.unsafe_bitcast[
                        Scalar[DType.float16]
                    ]().unsafe_load[width=1](offset=0)
                )
        if self.amp:
            self.scaler.scale_grads(result[1])
            var inf = self.scaler.found_inf(result[1])
            if inf:
                self.scaler.update(True)
                self.opt.zero_grad()
                self.step_count += 1
                return (loss, result[1].copy())
            self.opt.accumulate_grads(result[1])
            self.step_count += 1
            if self.step_count % accumulation_steps == 0:
                var opt_grads = List[AnyTensor]()
                for gi in range(len(self.opt.groups)):
                    for ei in range(len(self.opt.groups[gi].entries)):
                        opt_grads.append(self.opt.groups[gi].entries[ei].grad)
                self.scaler.unscale_grads(opt_grads)
                self.opt.step()
                self.scaler.update(False)
                self.opt.zero_grad()
            return (loss, result[1].copy())

        self.opt.accumulate_grads(result[1])
        self.step_count += 1
        if self.step_count % accumulation_steps == 0:
            self.opt.step()
            self.opt.zero_grad()
        return (loss, result[1].copy())

    def eval_step(
        mut self,
        tokens: Tensor[DType.int32, 1],
        targets: Tensor[DType.int32, 1],
    ) -> Tuple[Float32, Float32]:
        """Forward-only eval: returns (loss, token accuracy)."""
        var inputs = self.build_inputs(tokens, targets)
        var outputs = self.interpreter.run(inputs)
        var loss = Float32(0)
        if len(outputs) > 0:
            var loss_t = outputs[0]
            if loss_t.dtype == DType.float32:
                loss = Float32(
                    loss_t.data.unsafe_bitcast[
                        Scalar[DType.float32]
                    ]().unsafe_load[width=1](offset=0)
                )
            else:
                loss = Float32(
                    loss_t.data.unsafe_bitcast[
                        Scalar[DType.float16]
                    ]().unsafe_load[width=1](offset=0)
                )
        # accuracy: argmax of the logits vs targets
        var logits = no_grad_any()
        try:
            logits = self.interpreter.graph.tensors[self.logits_node][0]
        except:
            pass
        var rows = logits.shape[0]
        var cols = logits.shape[1]
        var correct = 0
        if logits.dtype == DType.float32:
            var data = logits.data.unsafe_bitcast[Scalar[DType.float32]]()
            for i in range(rows):
                var best = 0
                var best_v = Float32(data.unsafe_load[width=1](offset=0))
                for j in range(cols):
                    var v = Float32(
                        data.unsafe_load[width=1](offset=i * cols + j)
                    )
                    if v > best_v:
                        best_v = v
                        best = j
                if best == Int(targets.get(i)):
                    correct += 1
        else:
            var data = logits.data.unsafe_bitcast[Scalar[DType.float16]]()
            for i in range(rows):
                var best = 0
                var best_v = Float32(data.unsafe_load[width=1](offset=0))
                for j in range(cols):
                    var v = Float32(
                        data.unsafe_load[width=1](offset=i * cols + j)
                    )
                    if v > best_v:
                        best_v = v
                        best = j
                if best == Int(targets.get(i)):
                    correct += 1
        var accuracy = Float32(correct) / Float32(rows)
        return (loss, accuracy)

    def zero_grad(mut self):
        self.opt.zero_grad()

    def enable_amp(mut self, enabled: Bool = True):
        self.amp = enabled


# -- module-level helpers -----------------------------------------------------


def _entry_attrs(n_inputs: Int) -> Dict[String, AttrValue]:
    var attrs = Dict[String, AttrValue]()
    attrs["n_inputs"] = AttrValue(n_inputs)
    return attrs^


def _no_attrs() -> Dict[String, AttrValue]:
    return Dict[String, AttrValue]()


def _in2(a: Int, b: Int) -> List[Int]:
    var l = List[Int]()
    l.append(a)
    l.append(b)
    return l^


# -- M7: inference-time fine-tuning ---------------------------------
#
# `finetune_step` runs one parameter update during serving.  LoRA mode is
# expressed through the frozen flags: `set_trainable` marks which parameter
# slots update; everything else is skipped by the optimizer (the M6 graph /
# autograd path is untouched - this is built ON TOP of train_step).


struct FinetuneMode(Copyable, Equatable, ImplicitlyCopyable, Movable):
    var _tag: Int8

    def __init__(out self, tag: Int8):
        self._tag = tag

    comptime Off = FinetuneMode(Int8(0))
    comptime Lora = FinetuneMode(Int8(1))
    comptime Full = FinetuneMode(Int8(2))

    def __eq__(self, other: Self) -> Bool:
        return self._tag == other._tag

    def __ne__(self, other: Self) -> Bool:
        return self._tag != other._tag


def param_names(model: TrainModel) -> List[String]:
    """The optimizer entry order's parameter names (matches
    mmdl_storage._collect_params)."""
    var names = List[String]()
    var cfg = model.cfg
    names.append("token_embd.weight")
    for l in range(cfg.n_layers):
        var base = "blk." + String(l) + "."
        names.append(base + "attn_norm.weight")
        names.append(base + "attn_q.weight")
        names.append(base + "attn_k.weight")
        names.append(base + "attn_v.weight")
        names.append(base + "attn_output.weight")
        names.append(base + "attn_q.bias")
        names.append(base + "attn_k.bias")
        names.append(base + "attn_v.bias")
        names.append(base + "ffn_norm.weight")
        names.append(base + "ffn_gate.weight")
        names.append(base + "ffn_up.weight")
        names.append(base + "ffn_down.weight")
    names.append("output_norm.weight")
    names.append("output.weight")
    return names^


def set_trainable(mut model: TrainModel, names: List[String]):
    """LoRA-style parameter selection: only the named parameters update.

    Every other parameter is frozen (its gradient is still computed, but
    the optimizer skips it).  An empty list freezes everything; the special
    name "*" unfreezes all (full fine-tuning).
    """
    var all_names = param_names(model)
    var freeze_all = len(names) == 0
    for i in range(model.opt.num_params()):
        var is_match = False
        if not freeze_all:
            for n in names:
                if n == "*" or n == all_names[i]:
                    is_match = True
        model.opt.set_frozen(i, not is_match)


def finetune_step(
    mut model: TrainModel,
    batch: Tensor[DType.int32, 1],
    targets: Tensor[DType.int32, 1],
    optimizer: Optional[AdamW] = None,
    accum_steps: Int = 1,
) -> Tuple[Float32, List[AnyTensor]]:
    """One fine-tuning update (inference-time training).

    Runs the M6 forward+backward and one optimizer step.  `optimizer`
    overrides the step hyperparameters (its group-0 lr/weight_decay are
    applied to the model's wired optimizer); `accum_steps` accumulates
    that many mini-batches per step.  Which parameters actually change is
    controlled by `set_trainable` (LoRA mode: only the selected subset).
    """
    if optimizer:
        var lr = optimizer.value().groups[0].lr
        var wd = optimizer.value().groups[0].weight_decay
        model.opt.set_group_hyperparams(0, lr, wd)
    return train_step(model, batch, targets, accum_steps)


# -- module-level API (the spec's train_step) ---------------------------------


def train_step(
    mut model: TrainModel,
    batch: Tensor[DType.int32, 1],
    targets: Tensor[DType.int32, 1],
    accumulation_steps: Int = 1,
) -> Tuple[Float32, List[AnyTensor]]:
    """Forward -> loss -> backward -> (accumulated) parameter update.

    Thin wrapper around `TrainModel.train_step` keeping the spec's
    `train_step(model, batch, targets, ...)` calling convention.
    """
    return model.train_step(batch, targets, accumulation_steps)


def eval_step(
    mut model: TrainModel,
    batch: Tensor[DType.int32, 1],
    targets: Tensor[DType.int32, 1],
) -> Tuple[Float32, Float32]:
    """Forward-only evaluation: (loss, accuracy)."""
    return model.eval_step(batch, targets)
