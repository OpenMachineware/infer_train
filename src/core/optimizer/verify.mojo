# core/optimizer/verify.mojo
#
# M5: optimization correctness verification.
#
# The verification contract: run the *original* and the *optimized* DAG
# through the same interpretive executor on identical small inputs and
# compare every output element.  Any mismatch aborts optimization with a
# clear report (pass name, node index, max diff) so a broken pass can be
# pinpointed instead of shipping silently wrong results.
#
# The executor runs the DAG in topological order through the OpRegistry
# dispatch (the same kernels the runtime uses), materializing constants
# inline and consuming external inputs by position.

from .dag_ir import (
    Dag,
    DagNode,
    OP_CONST,
    OP_INPUT,
    op_name,
)
from ..cfg import CFGGraph, TS_RET, TS_COND, TS_JUMP
from .cfg_to_dag import _cond_truth
from ..ops.base.op_interface import AnyTensor, from_any, to_any
from ..ops.base.op_registry import OpRegistry
from ..tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple


def clone_dag(dag: Dag) -> Dag:
    """Deep-copy a Dag (nodes, edges, shapes, const handles)."""
    var copy = Dag(dag.n_inputs)
    for node in dag.nodes:
        var id = copy.add_node(node.op, node.inputs, node.dtype)
        copy.nodes[id].shape = node.shape.copy()
        copy.nodes[id].const_data = node.const_data
        copy.nodes[id].name = node.name
    copy.set_outputs(dag.outputs)
    return copy^


def execute_dag(
    dag: Dag, inputs: List[AnyTensor]
) -> List[AnyTensor]:
    """Interpret the DAG; returns the output tensors (in dag.outputs order)."""
    var registry = OpRegistry()
    registry.register_default_ops()
    var results = List[AnyTensor]()
    var count = 0
    while count < len(dag.nodes):
        var dummy = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
        results.append(to_any[DType.float32, 1](dummy))
        count += 1

    var input_cursor = 0
    for i in range(len(dag.nodes)):
        if dag.nodes[i].op == OP_CONST:
            results[i] = dag.nodes[i].const_data.value()
            continue
        if dag.nodes[i].op == OP_INPUT:
            results[i] = inputs[input_cursor]
            input_cursor += 1
            continue
        var node_inputs = List[AnyTensor]()
        for input_id in dag.nodes[i].inputs:
            node_inputs.append(results[input_id])
        var op_info = registry.get(op_name(dag.nodes[i].op))
        var outputs = op_info.forward(node_inputs)
        if len(outputs) > 0:
            results[i] = outputs[0]
        else:
            results[i] = node_inputs[0]

    var graph_outputs = List[AnyTensor]()
    for out in dag.outputs:
        graph_outputs.append(results[out])
    return graph_outputs^


def _max_abs_diff(a: AnyTensor, b: AnyTensor) -> Float32:
    var max_diff = Float32(0)
    var n = a.numel
    if b.numel < n:
        n = b.numel
    if a.dtype == DType.float32:
        var pa = a.data.unsafe_bitcast[Scalar[DType.float32]]()
        var pb = b.data.unsafe_bitcast[Scalar[DType.float32]]()
        for i in range(n):
            var d = pa.unsafe_load(offset=i) - pb.unsafe_load(offset=i)
            if d < Float32(0):
                d = -d
            if d > max_diff:
                max_diff = d
    elif a.dtype == DType.float16:
        var pa = a.data.unsafe_bitcast[Scalar[DType.float16]]()
        var pb = b.data.unsafe_bitcast[Scalar[DType.float16]]()
        for i in range(n):
            var d = Float32(pa.unsafe_load(offset=i)) - Float32(
                pb.unsafe_load(offset=i)
            )
            if d < Float32(0):
                d = -d
            if d > max_diff:
                max_diff = d
    else:
        var pa = a.data.unsafe_bitcast[Scalar[DType.int32]]()
        var pb = b.data.unsafe_bitcast[Scalar[DType.int32]]()
        for i in range(n):
            var d = pa.unsafe_load(offset=i) - pb.unsafe_load(offset=i)
            var df = Float32(abs(Int(d)))
            if df > max_diff:
                max_diff = df
    return max_diff


def verify_dags(
    original: Dag,
    optimized: Dag,
    inputs: List[AnyTensor],
    tolerance: Float32 = Float32(1e-4),
) -> Bool:
    """Run both DAGs on `inputs` and compare their outputs."""
    var before = execute_dag(original, inputs)
    var after = execute_dag(optimized, inputs)
    if len(before) != len(after):
        print(
            "[verify] FAIL: output count changed: ",
            len(before),
            " -> ",
            len(after),
        )
        return False
    for i in range(len(before)):
        var diff = _max_abs_diff(before[i], after[i])
        if diff > tolerance:
            print("[verify] FAIL: output ", i, " max diff ", diff)
            return False
    return True


def verify_optimization(
    original: Dag,
    mut optimized: Dag,
    inputs: List[AnyTensor],
    pass_name: String,
    tolerance: Float32 = Float32(1e-4),
) -> Bool:
    """Verify `optimized` still matches `original`; abort-style failure.

    Returns True on success; on mismatch prints the offending pass and
    returns False so the caller can stop the pipeline.
    """
    if verify_dags(original, optimized, inputs, tolerance):
        return True
    print("[verify] optimization pass '", pass_name, "' changed semantics")
    return False


# -- CFG interpreter (verification of cfg_to_dag) ---------------------------


def _scalar_gt_zero(data: AnyTensor) -> Bool:
    if data.dtype == DType.float32:
        var p = data.data.unsafe_bitcast[Scalar[DType.float32]]()
        return p.unsafe_load(offset=0) > Scalar[DType.float32](0.0)
    if data.dtype == DType.float16:
        var p16 = data.data.unsafe_bitcast[Scalar[DType.float16]]()
        return p16.unsafe_load(offset=0) > Scalar[DType.float16](0.0)
    if data.dtype == DType.int32:
        var pi = data.data.unsafe_bitcast[Scalar[DType.int32]]()
        return pi.unsafe_load(offset=0) > Scalar[DType.int32](0)
    return False


def execute_cfg(cfg: CFGGraph, inputs: List[AnyTensor]) -> List[AnyTensor]:
    """Interpret a CFG; returns the exit block's node results."""
    var registry = OpRegistry()
    registry.register_default_ops()
    var results = Dict[Int, AnyTensor]()
    var dummy = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
    var dummy_any = to_any[DType.float32, 1](dummy)

    # positional inputs + constants: OP_INPUT nodes in dag order, consts inline
    var input_cursor = 0
    for i in range(len(cfg.dag.nodes)):
        if cfg.dag.nodes[i].op == OP_INPUT:
            results[i] = inputs[input_cursor]
            input_cursor += 1
        elif cfg.dag.nodes[i].op == OP_CONST:
            results[i] = cfg.dag.nodes[i].const_data.value()

    var pc = cfg.entry
    var steps = 0
    var last_block_last_node = -1
    var trips = Dict[Int, Int]()
    while True:
        steps += 1
        if steps > 1_000_000:
            print("[verify] execute_cfg: runaway loop")
            break
        if len(cfg.blocks[pc].nodes) > 0:
            last_block_last_node = cfg.blocks[pc].nodes[
                len(cfg.blocks[pc].nodes) - 1
            ]
        for node_id in cfg.blocks[pc].nodes:
            var op = cfg.dag.nodes[node_id].op
            if op == OP_CONST:
                results[node_id] = cfg.dag.nodes[node_id].const_data.value()
                continue
            var node_inputs = List[AnyTensor]()
            for input_id in cfg.dag.nodes[node_id].inputs:
                node_inputs.append(results.get(input_id, dummy_any))
            var outputs = registry.get(op_name(op)).forward(node_inputs)
            if len(outputs) > 0:
                results[node_id] = outputs[0]
        if cfg.blocks[pc].terminator == TS_RET:
            break
        if cfg.blocks[pc].terminator == TS_JUMP:
            var succ = -1
            for edge in cfg.edges:
                if edge.src == pc and edge.kind == 0:
                    succ = edge.dst
            pc = succ
            continue
        if cfg.blocks[pc].terminator == TS_COND:
            # static trip-count loops: honor the declared count
            if cfg.blocks[pc].static_trips >= 0:
                var done = trips.get(pc, 0)
                if done >= cfg.blocks[pc].static_trips:
                    for edge in cfg.edges:
                        if edge.src == pc and edge.kind == 2:
                            pc = edge.dst
                    continue
                trips[pc] = done + 1
                for edge in cfg.edges:
                    if edge.src == pc and edge.kind == 1:
                        pc = edge.dst
                continue
            var truth = _cond_truth(cfg, cfg.blocks[pc].cond_node)
            if truth == 0 and cfg.blocks[pc].cond_node in results:
                truth = 1 if _scalar_gt_zero(
                    results.get(cfg.blocks[pc].cond_node, dummy_any)
                ) else 2
            if truth == 0:
                break
            var taken = -1
            for edge in cfg.edges:
                if edge.src == pc and edge.kind == truth:
                    taken = edge.dst
            pc = taken
            continue
        break

    var outputs = List[AnyTensor]()
    for node_id in cfg.blocks[cfg.exit].nodes:
        outputs.append(results.get(node_id, dummy_any))
    if len(outputs) == 0:
        # diamond-shaped CFGs leave the exit block empty: the result is
        # the taken branch's last value (tracked during the walk)
        outputs.append(results.get(last_block_last_node, dummy_any))
    return outputs^
