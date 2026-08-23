# core/optimizer/dce.mojo
#
# M5 pass: dead code elimination on the optimization IR.
#
# Everything not reachable from the graph outputs (walking input edges
# backward) is dropped; live node indices are remapped.  Nodes left dead
# by simplify/cse/fusion are collected here.

from .dag_ir import (
    Dag,
    DagNode,
    OP_CONST,
)


def dag_dce(mut dag: Dag) -> Int:
    """Drop unreachable nodes; returns the number removed."""
    var n = len(dag.nodes)
    var live = List[Bool]()
    for i in range(n):
        live.append(False)

    # worklist from the outputs
    var stack = List[Int]()
    for out in dag.outputs:
        stack.append(out)
    while len(stack) > 0:
        var current = stack[len(stack) - 1]
        stack.pop()
        if current < 0 or current >= n or live[current]:
            continue
        live[current] = True
        for input_id in dag.nodes[current].inputs:
            stack.append(input_id)

    # build the new node list with remapping
    var remap = Dict[Int, Int]()
    var new_nodes = List[DagNode]()
    for i in range(n):
        if live[i]:
            remap[i] = len(new_nodes)
            var new_inputs = List[Int]()
            for input_id in dag.nodes[i].inputs:
                new_inputs.append(remap.get(input_id, -1))
            var moved = DagNode(
                dag.nodes[i].op, new_inputs, dag.nodes[i].dtype
            )
            moved.shape = dag.nodes[i].shape.copy()
            moved.const_data = dag.nodes[i].const_data
            moved.name = dag.nodes[i].name
            new_nodes.append(moved^)
    var removed = n - len(new_nodes)

    dag.nodes = new_nodes^
    var new_outputs = List[Int]()
    for out in dag.outputs:
        new_outputs.append(remap.get(out, -1))
    dag.outputs = new_outputs^
    return removed
