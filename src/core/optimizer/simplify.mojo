# core/optimizer/simplify.mojo
#
# M5 pass: algebraic simplification of the optimization IR.
#
#   add(x, const0)      -> x       (constant-zero operand dropped)
#   add_bias(x, const0) -> x       (zero bias dropped)
#
# The rewritten node's consumers are re-wired to the surviving operand and
# the node itself becomes dead (dce removes it in the same pipeline run).

from .dag_ir import (
    Dag,
    OP_CONST,
    OP_ADD,
    OP_ADD_BIAS,
)


def _rewire_inputs(mut dag: Dag, from_id: Int, to_id: Int) -> Int:
    """Replace every `from_id` input edge with `to_id`; returns rewires."""
    var rewired = 0
    for i in range(len(dag.nodes)):
        for j in range(len(dag.nodes[i].inputs)):
            if dag.nodes[i].inputs[j] == from_id:
                dag.nodes[i].inputs[j] = to_id
                rewired += 1
    return rewired


def _is_zero_const(dag: Dag, node_id: Int) -> Bool:
    if dag.nodes[node_id].op != OP_CONST:
        return False
    var data = dag.nodes[node_id].const_data.value()
    if data.dtype == DType.float32:
        var p = data.data.unsafe_bitcast[Scalar[DType.float32]]()
        for i in range(data.numel):
            if p.unsafe_load(offset=i) != Scalar[DType.float32](0.0):
                return False
        return True
    if data.dtype == DType.float16:
        var p16 = data.data.unsafe_bitcast[Scalar[DType.float16]]()
        for i in range(data.numel):
            if p16.unsafe_load(offset=i) != Scalar[DType.float16](0.0):
                return False
        return True
    return False


def dag_simplify(mut dag: Dag) -> Int:
    """Apply the algebraic identities; returns simplifications done."""
    var simplified = 0
    for i in range(len(dag.nodes)):
        if dag.nodes[i].op == OP_ADD and len(dag.nodes[i].inputs) == 2:
            var other = -1
            if _is_zero_const(dag, dag.nodes[i].inputs[0]):
                other = dag.nodes[i].inputs[1]
            elif _is_zero_const(dag, dag.nodes[i].inputs[1]):
                other = dag.nodes[i].inputs[0]
            if other >= 0:
                _ = _rewire_inputs(dag, i, other)
                dag.nodes[i].op = OP_CONST  # placeholder: dce drops it
                dag.nodes[i].inputs = List[Int]()
                simplified += 1
        elif (
            dag.nodes[i].op == OP_ADD_BIAS
            and len(dag.nodes[i].inputs) == 2
        ):
            if _is_zero_const(dag, dag.nodes[i].inputs[1]):
                _ = _rewire_inputs(dag, i, dag.nodes[i].inputs[0])
                dag.nodes[i].op = OP_CONST  # placeholder: dce drops it
                dag.nodes[i].inputs = List[Int]()
                simplified += 1
    return simplified
