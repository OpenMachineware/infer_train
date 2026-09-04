# runtime/interpreter.mojo
#
# The interpreter executes a `Graph` in topological order, dispatching each
# node to the best available device implementation from the `OpRegistry`.
#
# M5 additions:
#   * JIT integration: a node whose attrs carry `jit = 1` dispatches
#     through the `JitCache` (comptime-shape-specialized kernels) instead
#     of the generic registry path.
#   * Multithreading: the kernels the interpreter dispatches to are the
#     same pool-backed kernels the model forward uses (batched QKV /
#     gate+up / threaded lm_head), so independent-nodes parallelism comes
#     from inside the kernels (see thread_pool.mojo and the M5 report).
#
# M9 additions (CPU scheduling):
#   * The interpreter now integrates the Mojo-native work-stealing pool
#     (core/scheduler/thread_pool.mojo).  The critical part is the FFI fix:
#     when the engine runs as a shared library loaded by Python/C, no Mojo
#     main() initializes the async runtime, so every `parallelize` inside a
#     kernel would silently run inline (the "only the main thread works"
#     bug).  `run()` / `run_with_grad()` therefore call `ensure_runtime()`
#     first - idempotent and cheap - so the kernel-level pools always have a
#     live thread pool to dispatch onto.
#
# M6 additions:
#   * entry nodes consume the external inputs positionally: a node with
#     `n_inputs = k` in its attrs takes the next k external inputs,
#     otherwise it takes everything that remains (preserves the M1-M5
#     single-entry behavior).
#   * `run_with_grad` executes the forward with `forward_with_saved`, then
#     walks the graph in reverse topological order calling each op's
#     `backward`, accumulating gradients at multi-consumer tensors, and
#     returns the (filtered) gradients of the external inputs.

from ..core.graph import Graph, AttrValue
from ..core.ops.base.op_interface import AnyTensor, from_any, to_any
from ..core.ops.base.op_registry import OpRegistry
from ..core.ops.base.op_autograd import (
    accumulate_any,
    ones_like_any,
)
from ..core.tensor import Tensor
from ..core.utils import unimplemented
from ..core.jit.jit_cache import JitCache
from ..core.simd_utils import AutotuneCache
from ..core.scheduler.thread_pool import ensure_runtime, worker_count


struct Interpreter(Movable):
    var graph: Graph
    var registry: OpRegistry
    var jit_cache: JitCache
    var jit_enabled: Bool
    var simd_autotune: Bool  # M8: opt-in SIMD width autotuning
    var autotune_cache: AutotuneCache
    var autotune_done: Bool

    def __init__(
        out self,
        var graph: Graph,
        var registry: OpRegistry,
        simd_autotune: Bool = False,
    ):
        self.graph = graph^
        self.registry = registry^
        self.jit_cache = JitCache()
        self.jit_enabled = True
        self.simd_autotune = simd_autotune
        self.autotune_cache = AutotuneCache()
        self.autotune_done = False

    def _is_jit_node(self, node_id: Int) -> Bool:
        if not self.jit_enabled:
            return False
        if self.graph.nodes[node_id].op_type != "swiglu_ffn":
            return False
        var jit_attr = self.graph.nodes[node_id].attrs.get(
            "jit", AttrValue(0)
        )
        return jit_attr.int_val == 1

    def _run_jit_ffn(
        mut self, node_id: Int, inputs: List[AnyTensor]
    ) -> List[AnyTensor]:
        """Dispatch a jit-marked swiglu_ffn node through the JIT cache.

        With M8 SIMD autotuning on, the projection k-loop width comes from
        the autotune cache (benchmarked once for the input shapes); the
        default path keeps the legacy 128-bit width.
        """
        var x = from_any[DType.float16, 2](inputs[0])
        var gw = from_any[DType.float16, 2](inputs[1])
        var uw = from_any[DType.float16, 2](inputs[2])
        var dw = from_any[DType.float16, 2](inputs[3])
        var out: Tensor[DType.float16, 2]
        if self.simd_autotune:
            var width = self.autotune_cache.get(x.shape()[1])
            out = self.jit_cache.run_ffn_width(x, gw, uw, dw, width)
        else:
            out = self.jit_cache.run_ffn(x, gw, uw, dw)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float16, 2](out))
        return results^

    def _autotune_inputs(mut self, inputs: List[AnyTensor]):
        """M8: benchmark the SIMD widths for the input row lengths (once,
        on the first run when simd_autotune is enabled)."""
        for t in inputs:
            if t.rank == 2:
                var dim = t.shape[1]
                if t.dtype == DType.float16:
                    self.autotune_cache.autotune_f16(dim)
                else:
                    self.autotune_cache.autotune_f32(dim)

    def _entry_take_count(self, node_id: Int, remaining: Int) -> Int:
        """How many external inputs this entry node consumes.

        An explicit `n_inputs` attr wins; otherwise the node takes all
        remaining external inputs (the M1-M5 behavior).
        """
        var n_attr = self.graph.nodes[node_id].attrs.get(
            "n_inputs", AttrValue(-1)
        )
        if n_attr.int_val >= 0:
            return n_attr.int_val
        return remaining

    def _collect_node_inputs(
        mut self, node_id: Int, inputs: List[AnyTensor], mut cursor: Int
    ) -> List[AnyTensor]:
        """Gather the tensors one node consumes (entry nodes read the
        external input list positionally)."""
        var node_inputs = List[AnyTensor]()
        if len(self.graph.nodes[node_id].inputs) == 0:
            var take = self._entry_take_count(
                node_id, len(inputs) - cursor
            )
            for i in range(take):
                node_inputs.append(inputs[cursor + i])
            cursor += take
        else:
            for input_id in self.graph.nodes[node_id].inputs:
                try:
                    for tensor in self.graph.tensors[input_id]:
                        node_inputs.append(tensor)
                except:
                    pass
        return node_inputs^

    def pool_workers(self) -> Int:
        """The CPU pool's worker count for this run (>= 1).

        Exposed so hosts can observe how many threads the kernels'
        `parallelize` calls will dispatch onto.  Implies the runtime is
        initialized.
        """
        return worker_count()

    def run(mut self, inputs: List[AnyTensor]) -> List[AnyTensor]:
        """Execute the graph and return the last node's outputs.

        Device scheduling lives in `OpRegistry.get`: a node's `device_hint`
        wins when set, otherwise the default device is used, otherwise the
        registry falls back to the first (CPU) implementation.

        The first thing we do is `ensure_runtime()`: when the engine is a
        shared library driven by Python/C, no Mojo main() started the async
        runtime, and without it every kernel-level `parallelize` would run
        inline on the caller (the "only the main thread works" bug).
        """
        ensure_runtime()
        # M8: the autotune stage runs once, before the first dispatch,
        # benchmarking the SIMD widths for the input row lengths.
        if self.simd_autotune and not self.autotune_done:
            self._autotune_inputs(inputs)
            self.autotune_done = True
        var order = self.graph.topo_sort()
        var cursor = 0
        for node_id in order:
            var node_inputs = self._collect_node_inputs(
                node_id, inputs, cursor
            )

            if self._is_jit_node(node_id):
                var jit_outputs = self._run_jit_ffn(node_id, node_inputs)
                self.graph.tensors[node_id] = jit_outputs^
                continue

            var op = self.registry.get(
                self.graph.nodes[node_id].op_type,
                self.graph.nodes[node_id].device_hint,
            )
            var outputs = op.forward(node_inputs)
            self.graph.tensors[node_id] = outputs^

        if len(order) == 0:
            return List[AnyTensor]()
        var last = order[len(order) - 1]
        try:
            return self.graph.tensors[last].copy()
        except:
            return List[AnyTensor]()

    def _forward_with_saved(
        mut self, inputs: List[AnyTensor]
    ) -> List[Int]:
        """Forward pass that also records every node's saved tensors."""
        var order = self.graph.topo_sort()
        var cursor = 0
        for node_id in order:
            var node_inputs = self._collect_node_inputs(
                node_id, inputs, cursor
            )
            var op = self.registry.get(
                self.graph.nodes[node_id].op_type,
                self.graph.nodes[node_id].device_hint,
            )
            var fws = op.forward_with_saved(node_inputs)
            self.graph.tensors[node_id] = fws[0].copy()
            self.graph.nodes[node_id].saved = fws[1].copy()
        return order^

    def run_with_grad(
        mut self, inputs: List[AnyTensor], loss_node: Int
    ) -> Tuple[List[AnyTensor], List[AnyTensor]]:
        """Forward, then walk backward from `loss_node` to collect gradients.

        Returns (outputs of the last node, gradients of the external
        inputs that received a gradient - zero-numel "no gradient"
        sentinels are filtered out).

        Gradient accumulation: a tensor consumed by several nodes gets its
        gradient summed across all consumers before its own backward runs.
        All saved tensors are views over the forward buffers, and every
        node's saved list is cleared after its backward call.
        """
        ensure_runtime()
        var order = self._forward_with_saved(inputs)

        if loss_node < 0 or loss_node >= len(self.graph.nodes):
            unimplemented("run_with_grad: loss_node out of range")

        # seed: dLoss/dLoss = ones
        var grads = Dict[Int, List[AnyTensor]]()
        var seed = List[AnyTensor]()
        try:
            for t in self.graph.tensors[loss_node]:
                seed.append(ones_like_any(t))
        except:
            pass
        try:
            grads[loss_node] = seed^
        except:
            pass

        var input_grads = List[AnyTensor]()
        var entry_grads = Dict[Int, List[AnyTensor]]()

        # reverse topological traversal
        var i = len(order) - 1
        while i >= 0:
            var node_id = order[i]
            var node_grads_list = grads.get(node_id, List[AnyTensor]())
            if len(node_grads_list) > 0:
                var op = self.registry.get(
                    self.graph.nodes[node_id].op_type,
                    self.graph.nodes[node_id].device_hint,
                )
                var bw = op.backward(
                    node_grads_list, self.graph.nodes[node_id].saved
                )
                if len(self.graph.nodes[node_id].inputs) == 0:
                    # entry node: the returned grads belong to the
                    # external inputs; keep them per-node so the final
                    # list follows the external input order
                    try:
                        entry_grads[node_id] = bw^
                    except:
                        pass
                else:
                    var j = 0
                    for input_id in self.graph.nodes[node_id].inputs:
                        if j >= len(bw):
                            break
                        var g = bw[j]
                        j += 1
                        if g.numel == 0:
                            continue
                        var bucket = grads.get(
                            input_id, List[AnyTensor]()
                        )
                        if len(bucket) == 0:
                            bucket.append(g)
                        else:
                            var acc = bucket[0]
                            if (
                                acc.numel == g.numel
                                and acc.dtype == g.dtype
                                and acc.rank == g.rank
                            ):
                                accumulate_any(acc, g)
                            else:
                                bucket.append(g)
                        try:
                            grads[input_id] = bucket^
                        except:
                            pass
                # release the saved tensors of this node
                self.graph.nodes[node_id].saved = List[AnyTensor]()
            i -= 1

        # flatten the entry gradients in external-input order
        for entry_id in self.graph.entry:
            var egrads = entry_grads.get(entry_id, List[AnyTensor]())
            for g in egrads:
                if g.numel > 0:
                    input_grads.append(g)


        if len(order) == 0:
            return (List[AnyTensor](), input_grads^)
        var last = order[len(order) - 1]
        var outputs = List[AnyTensor]()
        try:
            outputs = self.graph.tensors[last].copy()
        except:
            pass
        return (outputs^, input_grads^)
