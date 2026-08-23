# core/optimizer/dag_optimizer.mojo
#
# M5: DAG optimization infrastructure - the compact optimization IR and the
# pass pipeline driver.
#
# The optimizer works on `Dag`: a flat list of `DagNode`s with integer op
# codes and index-based edges.  This IR is small enough to serialize across
# the C ABI (the Python backend passes raw node tables), and it keeps every
# pass trivially testable in isolation.  The full passes live in their own
# modules (cse.mojo / dce.mojo / constant_fold.mojo / simplify.mojo /
# shape_inference.mojo / fusion.mojo / memory_plan.mojo); this module wires
# them into the M5 pipeline:
#
#     shape_inference -> constant_fold -> simplify -> cse -> fusion
#       -> shape_inference -> dce -> memory_plan
#
# The pipeline is idempotent-safe and each pass reports what it changed so
# the verify module (verify.mojo) can compare pre/post semantics.

from .dag_ir import (
    Dag,
    DagNode,
    OP_CONST,
    OP_INPUT,
    op_name,
)
from .shape_inference import dag_shape_inference
from .constant_fold import dag_constant_fold
from .simplify import dag_simplify
from .cse import dag_cse
from .fusion import dag_fusion
from .dce import dag_dce
from .memory_plan import dag_memory_plan, MemoryPlan


struct OptimizeStats(Copyable, Movable, ImplicitlyCopyable):
    var folded: Int
    var simplified: Int
    var cse_removed: Int
    var fused: Int
    var dce_removed: Int
    var baseline_bytes: Int
    var planned_bytes: Int

    def __init__(out self):
        self.folded = 0
        self.simplified = 0
        self.cse_removed = 0
        self.fused = 0
        self.dce_removed = 0
        self.baseline_bytes = 0
        self.planned_bytes = 0


def optimize_dag(
    mut dag: Dag,
    flags: Int = 0,
) -> OptimizeStats:
    """Run the M5 optimization pipeline over `dag` in place.

    `flags` bitmask (for per-pass toggling; 0 = all on):
        1  skip constant_fold
        2  skip simplify
        4  skip cse
        8  skip fusion
        16 skip dce
    Returns per-pass change counts plus the memory-planning numbers.
    """
    var stats = OptimizeStats()

    dag_shape_inference(dag)

    if flags & 1 == 0:
        stats.folded = dag_constant_fold(dag)
    if flags & 2 == 0:
        stats.simplified = dag_simplify(dag)
    if flags & 8 == 0:
        stats.fused = dag_fusion(dag)
    if flags & 4 == 0:
        stats.cse_removed = dag_cse(dag)

    dag_shape_inference(dag)

    if flags & 16 == 0:
        stats.dce_removed = dag_dce(dag)

    # liveness analysis + buffer reuse plan (always run: it only annotates)
    var plan = dag_memory_plan(dag)
    stats.baseline_bytes = plan.baseline_bytes
    stats.planned_bytes = plan.planned_bytes
    return stats


def summarize_dag(dag: Dag) -> String:
    """One-line node histogram for tests and diagnostics."""
    var counts = Dict[Int, Int]()
    for node in dag.nodes:
        counts[node.op] = counts.get(node.op, 0) + 1
    var s = String("DAG: ") + String(len(dag.nodes)) + String(" nodes (")
    var first = True
    for key in counts.keys():
        if not first:
            s += String(", ")
        s += op_name(key) + String(" x") + String(counts.get(key, 0))
        first = False
    s += String(")")
    return s
