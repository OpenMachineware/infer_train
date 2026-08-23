# core/optimizer/constant_fold.mojo
#
# M5 pass: compile-time evaluation of nodes whose inputs are all constants.
#
# The fold reuses the real registered kernels (via the OpRegistry dispatch)
# so the folded value is bit-for-bit what the runtime would compute.  Large
# projection folds are skipped by a size guard - folding a 1M-element
# matmul at compile time costs more than it saves.

from .dag_ir import (
    Dag,
    OP_CONST,
    OP_INPUT,
    OP_MATMUL,
    OP_LM_HEAD,
    op_name,
)
from .shape_inference import shape_numel
from ..ops.base.op_interface import AnyTensor
from ..ops.base.op_registry import OpRegistry

comptime FOLD_MAX_ELEMS = 65536


def _is_foldable(dag: Dag, node_id: Int) -> Bool:
    if dag.nodes[node_id].op == OP_CONST or dag.nodes[node_id].op == OP_INPUT:
        return False
    for input_id in dag.nodes[node_id].inputs:
        if dag.nodes[input_id].op != OP_CONST:
            return False
        var data = dag.nodes[input_id].const_data.value()
        if data.numel > FOLD_MAX_ELEMS:
            return False
    return True


def dag_constant_fold(mut dag: Dag) -> Int:
    """Fold const-only nodes into OP_CONST nodes; returns folds done."""
    var registry = OpRegistry()
    registry.register_default_ops()
    var folded = 0
    for i in range(len(dag.nodes)):
        if dag.nodes[i].op == OP_CONST:
            continue
        if not _is_foldable(dag, i):
            continue
        var inputs = List[AnyTensor]()
        for input_id in dag.nodes[i].inputs:
            inputs.append(dag.nodes[input_id].const_data.value())
        var name = op_name(dag.nodes[i].op)
        var op_info = registry.get(name)
        var outputs = op_info.forward(inputs)
        if len(outputs) != 1:
            continue
        # replace this node with the folded constant
        dag.nodes[i].op = OP_CONST
        dag.nodes[i].inputs = List[Int]()
        dag.nodes[i].const_data = Optional(outputs[0])
        var shape = List[Int]()
        for d in range(outputs[0].rank):
            shape.append(outputs[0].shape[d])
        dag.nodes[i].shape = shape^
        folded += 1
    return folded
