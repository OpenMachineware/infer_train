# core/optimizer/cfg_to_dag.mojo
#
# M5 key transform: lower a CFG to a single DAG with guards.
#
# Static control flow only (M5 scope): every branch condition must be a
# constant node and every loop must carry a compile-time trip count.  The
# taken branches are inlined in execution order, loops are unrolled
# `static_trips` times (loop-carried references resolve to the previous
# iteration's copies), and each statically-resolved decision is recorded
# as a `Guard` for verification/tracing.  A data-dependent condition or an
# unknown trip count makes the transform report `success = False` - the
# graph stays a CFG and lands in M6's dynamic control flow.

from ..cfg import CFGGraph, TS_RET, TS_COND, TS_JUMP
from .dag_ir import (
    Dag,
    DagNode,
    OP_CONST,
    OP_INPUT,
)
from ..ops.base.op_interface import AnyTensor


struct Guard(Movable):
    var block: Int
    var cond_node: Int
    var taken: Int  # 1 = true edge taken, 2 = false edge taken

    def __init__(out self, block: Int, cond_node: Int, taken: Int):
        self.block = block
        self.cond_node = cond_node
        self.taken = taken


struct CfgToDagResult(Movable):
    var dag: Dag
    var guards: List[Guard]
    var success: Bool

    def __init__(out self, var dag: Dag, var guards: List[Guard], success: Bool):
        self.dag = dag^
        self.guards = guards^
        self.success = success


def _cond_truth(cfg: CFGGraph, cond_node: Int) -> Int:
    """Evaluate a static condition: 1 (true), 2 (false), 0 (not static)."""
    if cond_node < 0 or cond_node >= len(cfg.dag.nodes):
        return 0
    if cfg.dag.nodes[cond_node].op != OP_CONST:
        return 0
    if not cfg.dag.nodes[cond_node].const_data:
        return 0
    var data = cfg.dag.nodes[cond_node].const_data.value()
    if data.dtype == DType.float32:
        var p = data.data.unsafe_bitcast[Scalar[DType.float32]]()
        if p.unsafe_load(offset=0) > Scalar[DType.float32](0.0):
            return 1
        return 2
    if data.dtype == DType.float16:
        var p16 = data.data.unsafe_bitcast[Scalar[DType.float16]]()
        if p16.unsafe_load(offset=0) > Scalar[DType.float16](0.0):
            return 1
        return 2
    if data.dtype == DType.int32:
        var pi = data.data.unsafe_bitcast[Scalar[DType.int32]]()
        if pi.unsafe_load(offset=0) > Scalar[DType.int32](0):
            return 1
        return 2
    return 0


def _copy_into(
    mut out_dag: Dag,
    src_dag: Dag,
    node_id: Int,
    mut remap: Dict[Int, Int],
) -> Int:
    """Copy one src_dag node into out_dag (recursively copying inputs);
    returns the new node id."""
    if node_id in remap:
        return remap.get(node_id, -1)
    # inputs first
    for input_id in src_dag.nodes[node_id].inputs:
        _ = _copy_into(out_dag, src_dag, input_id, remap)
    var new_inputs = List[Int]()
    for input_id in src_dag.nodes[node_id].inputs:
        new_inputs.append(remap.get(input_id, -1))
    var id = out_dag.add_node(
        src_dag.nodes[node_id].op, new_inputs, src_dag.nodes[node_id].dtype
    )
    out_dag.nodes[id].shape = src_dag.nodes[node_id].shape.copy()
    out_dag.nodes[id].const_data = src_dag.nodes[node_id].const_data
    out_dag.nodes[id].name = src_dag.nodes[node_id].name
    remap[node_id] = id
    return id


def _edge_kind_of(cfg: CFGGraph, src: Int, dst: Int) -> Int:
    for edge in cfg.edges:
        if edge.src == src and edge.dst == dst:
            return edge.kind
    return -1


def _targets_of(cfg: CFGGraph, src: Int, kind: Int) -> Int:
    for edge in cfg.edges:
        if edge.src == src and edge.kind == kind:
            return edge.dst
    return -1


def _lower_cfg(
    mut cfg: CFGGraph,
    mut out_dag: Dag,
    mut guards: List[Guard],
) -> Bool:
    """Fill `out_dag`/`guards` from the static CFG; False = dynamic CFG."""
    var remap = Dict[Int, Int]()
    var last_copied = -1

    # copy entry inputs first (input nodes must be the first out_dag nodes
    # so execute_dag's positional input consumption matches)
    for i in range(len(cfg.dag.nodes)):
        if cfg.dag.nodes[i].op == OP_INPUT:
            _ = _copy_into(out_dag, cfg.dag, i, remap)

    var visited = Dict[Int, Bool]()
    var queue = List[Int]()
    queue.append(cfg.entry)

    while len(queue) > 0:
        var block_id = queue[len(queue) - 1]
        queue.pop()
        if visited.get(block_id, False):
            continue
        visited[block_id] = True

        if cfg.blocks[block_id].terminator == TS_RET:
            continue

        if cfg.blocks[block_id].terminator == TS_JUMP:
            for node_id in cfg.blocks[block_id].nodes:
                last_copied = _copy_into(out_dag, cfg.dag, node_id, remap)
            var succ = _targets_of(cfg, block_id, 0)
            if succ >= 0:
                queue.append(succ)
            continue

        if cfg.blocks[block_id].terminator == TS_COND:
            var truth = _cond_truth(cfg, cfg.blocks[block_id].cond_node)
            if truth == 0:
                return False
            guards.append(
                Guard(block_id, cfg.blocks[block_id].cond_node, truth)
            )
            var taken = _targets_of(cfg, block_id, truth)
            if taken >= 0:
                # while-loop header: unroll `static_trips` times when known
                if cfg.blocks[block_id].static_trips >= 0:
                    var body = _targets_of(cfg, block_id, 1)
                    var exit = _targets_of(cfg, block_id, 2)
                    for trip in range(cfg.blocks[block_id].static_trips):
                        var next_remap = Dict[Int, Int]()
                        # inherit loop-invariant mappings (entry/consts);
                        # body nodes are re-copied each iteration so their
                        # previous-iteration copies stay reachable
                        for key in remap.keys():
                            var is_body = False
                            for bid in cfg.blocks[body].nodes:
                                if bid == key:
                                    is_body = True
                            if not is_body:
                                next_remap[key] = remap.get(key, -1)
                        for node_id in cfg.blocks[body].nodes:
                            last_copied = _copy_into(
                                out_dag, cfg.dag, node_id, next_remap
                            )
                        remap = next_remap^
                    queue.append(exit)
                else:
                    queue.append(taken)
            continue

    if len(out_dag.outputs) == 0 and last_copied >= 0:
        var outputs = List[Int]()
        outputs.append(last_copied)
        out_dag.set_outputs(outputs^)
    return True


def cfg_to_dag(var cfg: CFGGraph) -> CfgToDagResult:
    """Lower `cfg` to a DAG (static control flow only).

    The caller's CFG is consumed; the result carries the merged DAG, the
    guards, and whether the lowering succeeded.
    """
    var dag = Dag(cfg.dag.n_inputs)
    var guards = List[Guard]()
    var ok = _lower_cfg(cfg, dag, guards)
    return CfgToDagResult(dag^, guards^, ok)
