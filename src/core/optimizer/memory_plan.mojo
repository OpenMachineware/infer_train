# core/optimizer/memory_plan.mojo
#
# M5 pass: memory planning via liveness analysis.
#
# Every non-constant tensor's live range is [first use, last use] (graph
# outputs live until the end).  The intervals are then greedily assigned
# to reusable buffer slots: two tensors whose lifetimes do not overlap
# share one slot, sized to the largest tenant.  The pass annotates each
# node with (live_start, live_end, slot) and reports the pool size against
# the naive allocate-everything baseline - the engine's MemoryPool can
# allocate `planned_bytes` once and reuse the slots across the graph.

from .dag_ir import Dag, OP_CONST, OP_INPUT
from .shape_inference import shape_numel, elem_bytes


struct MemoryPlan(Copyable, Movable, ImplicitlyCopyable):
    var baseline_bytes: Int
    var planned_bytes: Int
    var n_slots: Int
    var n_pooled: Int

    def __init__(out self):
        self.baseline_bytes = 0
        self.planned_bytes = 0
        self.n_slots = 0
        self.n_pooled = 0


def _tensor_bytes(dag: Dag, node_id: Int) -> Int:
    if len(dag.nodes[node_id].shape) == 0:
        return 0
    return shape_numel(dag.nodes[node_id].shape.copy()) * elem_bytes(
        dag.nodes[node_id].dtype
    )


def dag_memory_plan(mut dag: Dag) -> MemoryPlan:
    """Run liveness analysis + slot assignment; returns the plan stats."""
    var n = len(dag.nodes)
    var plan = MemoryPlan()

    var first_use = List[Int](length=n, fill=1 << 30)
    var last_use = List[Int](length=n, fill=-1)
    # graph outputs stay alive until the end
    for out in dag.outputs:
        last_use[out] = n
        if first_use[out] > n:
            first_use[out] = n

    for i in range(n):
        for input_id in dag.nodes[i].inputs:
            if i < first_use[input_id]:
                first_use[input_id] = i
            if i > last_use[input_id]:
                last_use[input_id] = i

    # constants are persistent weights (outside the activation pool);
    # inputs are caller-owned.  Only computed tensors join the pool.
    for i in range(n):
        if dag.nodes[i].op == OP_CONST or dag.nodes[i].op == OP_INPUT:
            continue
        plan.baseline_bytes += _tensor_bytes(dag, i)
        dag.nodes[i].live_start = first_use[i]
        dag.nodes[i].live_end = last_use[i]

    # greedy interval-to-slot assignment (first-fit on slot end times)
    var slot_end = List[Int]()
    var slot_size = List[Int]()
    for i in range(n):
        if dag.nodes[i].op == OP_CONST or dag.nodes[i].op == OP_INPUT:
            continue
        if len(dag.nodes[i].shape) == 0:
            continue
        var bytes = _tensor_bytes(dag, i)
        var assigned = -1
        for s in range(len(slot_end)):
            if slot_end[s] <= first_use[i] and slot_size[s] >= bytes:
                assigned = s
                break
        if assigned < 0:
            # try to fit into any free slot by growing it (largest tenant)
            for s in range(len(slot_end)):
                if slot_end[s] <= first_use[i]:
                    assigned = s
                    break
        if assigned < 0:
            assigned = len(slot_end)
            slot_end.append(-1)
            slot_size.append(0)
        if slot_size[assigned] < bytes:
            slot_size[assigned] = bytes
        if last_use[i] > slot_end[assigned]:
            slot_end[assigned] = last_use[i]
        dag.nodes[i].slot = assigned
        plan.n_pooled += 1

    for s in range(len(slot_size)):
        plan.planned_bytes += slot_size[s]
    plan.n_slots = len(slot_size)
    if plan.planned_bytes < 1:
        plan.planned_bytes = 1
    return plan
