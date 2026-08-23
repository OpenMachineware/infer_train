# core/optimizer/shape_inference.mojo
#
# M5 pass: static shape inference over the optimization IR.
#
# Every non-const node gets its output shape from its inputs (rank-2
# projection rules plus the elementwise/broadcast rules).  Const/input
# nodes carry their shapes already; `set_input_shape` lets the caller
# declare input shapes before inference runs.
#
# Mojo 1.0 note: `List` is Copyable but not ImplicitlyCopyable, so node
# fields are read through `dag.nodes[i].field` directly instead of copying
# whole nodes.

from .dag_ir import (
    Dag,
    OP_MATMUL,
    OP_LM_HEAD,
    OP_ADD,
    OP_ADD_BIAS,
    OP_RMS_NORM,
    OP_SOFTMAX,
    OP_SWIGLU,
    OP_EMBEDDING,
    OP_FUSED_MATMUL_ADD_BIAS,
    OP_FUSED_MATMUL_ADD,
    OP_FUSED_MATMUL_RMS_NORM,
    OP_FUSED_SWIGLU_MATMUL,
    OP_CONST,
    OP_INPUT,
)


def set_input_shape(
    mut dag: Dag, input_id: Int, shape: List[Int], dtype: Int
):
    """Declare the shape/dtype of an OP_INPUT node."""
    dag.nodes[input_id].shape = shape.copy()
    dag.nodes[input_id].dtype = dtype


def dag_shape_inference(mut dag: Dag) -> Int:
    """Infer output shapes for every node; returns nodes inferred.

    Nodes must be topologically ordered (they are, by construction).
    """
    var inferred = 0
    for i in range(len(dag.nodes)):
        var op = dag.nodes[i].op
        if op == OP_CONST or op == OP_INPUT:
            continue
        var a_shape = dag.nodes[dag.nodes[i].inputs[0]].shape.copy()
        var a_dtype = dag.nodes[dag.nodes[i].inputs[0]].dtype
        var shape = List[Int]()
        if op == OP_MATMUL:
            var b_shape = dag.nodes[dag.nodes[i].inputs[1]].shape.copy()
            shape.append(a_shape[0])
            shape.append(b_shape[1])
        elif (
            op == OP_LM_HEAD
            or op == OP_FUSED_MATMUL_ADD_BIAS
            or op == OP_FUSED_MATMUL_RMS_NORM
        ):
            var w_shape = dag.nodes[dag.nodes[i].inputs[1]].shape.copy()
            shape.append(a_shape[0])
            shape.append(w_shape[0])
        elif (
            op == OP_FUSED_MATMUL_ADD
            or op == OP_ADD
            or op == OP_ADD_BIAS
            or op == OP_RMS_NORM
            or op == OP_SOFTMAX
            or op == OP_SWIGLU
        ):
            for d in a_shape:
                shape.append(d)
        elif op == OP_FUSED_SWIGLU_MATMUL:
            var w_shape = dag.nodes[dag.nodes[i].inputs[2]].shape.copy()
            shape.append(a_shape[0])
            shape.append(w_shape[0])
        elif op == OP_EMBEDDING:
            var t_shape = dag.nodes[dag.nodes[i].inputs[0]].shape.copy()
            var w_shape = dag.nodes[dag.nodes[i].inputs[1]].shape.copy()
            shape.append(t_shape[0])
            shape.append(w_shape[1])
        dag.nodes[i].shape = shape^
        dag.nodes[i].dtype = a_dtype
        inferred += 1
    return inferred


def shape_numel(shape: List[Int]) -> Int:
    var total = 1
    for d in shape:
        total *= d
    return total


def elem_bytes(dtype: Int) -> Int:
    if dtype == 1:
        return 2
    return 4
