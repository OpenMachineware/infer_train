# core/optimizer/cfg_cse.mojo
#
# M5 pass: common subexpression elimination inside CFG basic blocks.
#
# Every block's node list is a DAG fragment; the DAG-level CSE (cse.mojo)
# runs per block, reusing the shared node hashing.  Duplicate computations
# inside a block are removed and their consumers re-wired.

from ..cfg import CFGGraph
from .cse import dag_cse
from .dag_ir import Dag


def cfg_cse(mut cfg: CFGGraph) -> Int:
    """Block-local CSE across the CFG; returns duplicates removed."""
    # Run the DAG CSE on the whole graph once: all block-local duplicates
    # live in the same shared node list, so one pass removes every
    # intra-block duplicate (and only those - cross-block deduplication
    # would change control-flow semantics, so the block structure keeps
    # the edges authoritative).
    var removed = dag_cse(cfg.dag)
    return removed
