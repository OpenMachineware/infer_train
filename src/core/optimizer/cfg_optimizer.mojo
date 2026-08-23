# core/optimizer/cfg_optimizer.mojo
#
# M5: CFG optimization pipeline.
#
#     cfg_inline -> cfg_dce -> cfg_cse -> cfg_to_dag -> dag pipeline
#
# The first three passes work on the block structure; cfg_to_dag lowers
# the (static) control flow into a single DAG, after which the full DAG
# pipeline (constant fold / simplify / fusion / cse / dce / memory plan)
# applies uniformly - the "cfg_to_dag" stage from the M5
# plan.  Data-dependent control flow makes the lowering fail cleanly and
# the CFG survives untouched for M6.

from ..cfg import CFGGraph
from .dag_ir import Dag
from .cfg_inline import cfg_inline
from .cfg_dce import cfg_dce
from .cfg_cse import cfg_cse
from .cfg_to_dag import cfg_to_dag, _lower_cfg, CfgToDagResult, Guard
from .dag_optimizer import optimize_dag, OptimizeStats


struct CfgOptimizeResult(Movable):
    var dag: Dag
    var guards: List[Guard]
    var stats: OptimizeStats
    var success: Bool

    def __init__(
        out self,
        var dag: Dag,
        var guards: List[Guard],
        stats: OptimizeStats,
        success: Bool,
    ):
        self.dag = dag^
        self.guards = guards^
        self.stats = stats
        self.success = success


def optimize_cfg(var cfg: CFGGraph) -> CfgOptimizeResult:
    """Run the CFG pipeline and lower to an optimized DAG."""
    _ = cfg_inline(cfg)
    _ = cfg_dce(cfg)
    _ = cfg_cse(cfg)

    var dag = Dag(cfg.dag.n_inputs)
    var guards = List[Guard]()
    var ok = _lower_cfg(cfg, dag, guards)
    if not ok:
        var empty_stats = OptimizeStats()
        return CfgOptimizeResult(dag^, guards^, empty_stats, False)

    var stats = optimize_dag(dag)
    return CfgOptimizeResult(dag^, guards^, stats, True)
