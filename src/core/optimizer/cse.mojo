# core/optimizer/cse.mojo
#
# M5 pass: common subexpression elimination on the optimization IR.
#
# Two nodes are equivalent when their op code, dtype and input indices all
# match (constants additionally compare their payload bytes), so the later
# node's consumers are re-wired to the earlier node and the duplicate
# becomes dead.  The IR is a DAG in topological order, which makes one
# forward sweep enough.

from .dag_ir import (
    Dag,
    OP_CONST,
)

comptime KEY_CAP = 4096


struct NodeKey(Movable):
    var op: Int
    var dtype: Int
    var in0: Int
    var in1: Int
    var in2: Int
    var const_hash: Int
    var hashed: Bool  # True when const_hash participates

    def __init__(
        out self,
        op: Int,
        dtype: Int,
        in0: Int,
        in1: Int,
        in2: Int,
        const_hash: Int,
        hashed: Bool,
    ):
        self.op = op
        self.dtype = dtype
        self.in0 = in0
        self.in1 = in1
        self.in2 = in2
        self.const_hash = const_hash
        self.hashed = hashed


def _key_equal(a: NodeKey, b: NodeKey) -> Bool:
    if a.op != b.op or a.dtype != b.dtype:
        return False
    if a.in0 != b.in0 or a.in1 != b.in1 or a.in2 != b.in2:
        return False
    if a.hashed and b.hashed and a.const_hash != b.const_hash:
        return False
    if a.hashed != b.hashed:
        return False
    return True


def _const_hash(dag: Dag, node_id: Int) -> Int:
    if not dag.nodes[node_id].const_data:
        return 0
    var data = dag.nodes[node_id].const_data.value()
    var h = 1469598103
    h = h * 1099511628211 + dag.nodes[node_id].dtype
    for d in dag.nodes[node_id].shape:
        h = h * 1099511628211 + d
    # sample the payload bytes (full hash for small tensors, strided for
    # large ones - collision-free enough for CSE verification)
    var step = 1
    if data.numel > 1024:
        step = data.numel // 1024
    var p = data.data.unsafe_bitcast[UInt8]()
    var elem = 4
    if data.dtype == DType.float16:
        elem = 2
    var i = 0
    while i < data.numel:
        for b in range(elem):
            var byte = Int(p.unsafe_load(offset=i * elem + b))
            h = h * 1099511628211 + byte
        i += step
    return h


def _make_key(dag: Dag, node_id: Int) -> NodeKey:
    var in0 = -1
    var in1 = -1
    var in2 = -1
    if len(dag.nodes[node_id].inputs) > 0:
        in0 = dag.nodes[node_id].inputs[0]
    if len(dag.nodes[node_id].inputs) > 1:
        in1 = dag.nodes[node_id].inputs[1]
    if len(dag.nodes[node_id].inputs) > 2:
        in2 = dag.nodes[node_id].inputs[2]
    var chash = 0
    var hashed = False
    if dag.nodes[node_id].op == OP_CONST:
        if dag.nodes[node_id].const_data:
            chash = _const_hash(dag, node_id)
            hashed = True
    return NodeKey(
        dag.nodes[node_id].op,
        dag.nodes[node_id].dtype,
        in0,
        in1,
        in2,
        chash,
        hashed
    )


def dag_cse(mut dag: Dag) -> Int:
    """Eliminate duplicate subexpressions; returns duplicates removed."""
    var seen = Dict[Int, List[Int]]()  # op code -> list of node ids
    var removed = 0
    for i in range(len(dag.nodes)):
        var key = _make_key(dag, i)
        var bucket = seen.get(dag.nodes[i].op, List[Int]())
        var match_id = -1
        for other_id in bucket:
            var other_key = _make_key(dag, other_id)
            if _key_equal(key, other_key):
                match_id = other_id
                break
        if match_id >= 0:
            # rewire consumers and leave this node dead
            for j in range(len(dag.nodes)):
                for k in range(len(dag.nodes[j].inputs)):
                    if dag.nodes[j].inputs[k] == i:
                        dag.nodes[j].inputs[k] = match_id
            for o in range(len(dag.outputs)):
                if dag.outputs[o] == i:
                    dag.outputs[o] = match_id
            dag.nodes[i].op = OP_CONST
            dag.nodes[i].inputs = List[Int]()
            removed += 1
        else:
            bucket.append(i)
            seen[dag.nodes[i].op] = bucket^
    return removed
