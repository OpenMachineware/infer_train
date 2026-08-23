# core/cfg.mojo
#
# M5: control-flow graph (CFG) structures for static control flow.
#
# A `CFGGraph` holds basic blocks plus the `Dag` whose nodes the blocks
# reference (each block owns a list of node ids executed in order).  The
# terminator kinds cover the two M5 control-flow shapes:
#
#   TS_RET   - block ends the graph (exit block)
#   TS_COND  - if/while condition: true/false edges
#   TS_JUMP  - unconditional edge (fallthrough)
#
# Structured builders (`add_if` / `add_while`) create the classic diamond
# and loop layouts; M5 only requires *static* control flow (compile-time
# conditions and fixed trip counts), which is exactly what cfg_to_dag
# lowers.  Data-dependent conditions stay as guards for M6.

from .optimizer.dag_ir import Dag, OP_CONST
from .ops.base.op_interface import AnyTensor

comptime TS_RET = 0
comptime TS_COND = 1
comptime TS_JUMP = 2


def ts_name(kind: Int) -> String:
    if kind == TS_RET:
        return "ret"
    if kind == TS_COND:
        return "cond"
    if kind == TS_JUMP:
        return "jump"
    return "ts?"


struct CFGEdge(Copyable, Movable, ImplicitlyCopyable):
    var src: Int
    var dst: Int
    var kind: Int  # 0 fallthrough/jump, 1 true, 2 false

    def __init__(out self, src: Int, dst: Int, kind: Int = 0):
        self.src = src
        self.dst = dst
        self.kind = kind


struct CFGBlock(Movable):
    var id: Int
    var nodes: List[Int]  # indices into the graph's Dag
    var terminator: Int
    var cond_node: Int  # Dag node id when terminator == TS_COND
    var static_trips: Int  # >= 0: known trip count (while loops)

    def __init__(out self, id: Int):
        self.id = id
        self.nodes = List[Int]()
        self.terminator = TS_RET
        self.cond_node = -1
        self.static_trips = -1


struct CFGGraph(Movable):
    var dag: Dag
    var blocks: List[CFGBlock]
    var edges: List[CFGEdge]
    var entry: Int
    var exit: Int

    def __init__(out self, var dag: Dag):
        self.dag = dag^
        self.blocks = List[CFGBlock]()
        self.edges = List[CFGEdge]()
        self.entry = 0
        self.exit = 0

    def new_block(mut self) -> Int:
        var id = len(self.blocks)
        self.blocks.append(CFGBlock(id))
        return id

    def add_node_to_block(mut self, block: Int, node_id: Int):
        self.blocks[block].nodes.append(node_id)

    def set_terminator(
        mut self, block: Int, kind: Int, cond_node: Int = -1
    ):
        self.blocks[block].terminator = kind
        self.blocks[block].cond_node = cond_node

    def add_edge(mut self, src: Int, dst: Int, kind: Int = 0):
        self.edges.append(CFGEdge(src, dst, kind))

    def set_entry(mut self, block: Int):
        self.entry = block

    def set_exit(mut self, block: Int):
        self.exit = block

    def edge_targets(self, src: Int) -> List[Int]:
        """Return the dsts of all edges leaving `src` (in edge order)."""
        var targets = List[Int]()
        for edge in self.edges:
            if edge.src == src:
                targets.append(edge.dst)
        return targets^


def _edge_kind_of(cfg: CFGGraph, src: Int, dst: Int) -> Int:
    for edge in cfg.edges:
        if edge.src == src and edge.dst == dst:
            return edge.kind
    return -1


def build_if_cond_graph(
    var dag: Dag,
    cond_node: Int,
    then_nodes: List[Int],
    else_nodes: List[Int],
) -> CFGGraph:
    """Build the classic if diamond: cond -> then/else -> merge."""
    var cfg = CFGGraph(dag^)
    var entry = cfg.new_block()
    var then_block = cfg.new_block()
    var else_block = cfg.new_block()
    var exit = cfg.new_block()
    cfg.set_entry(entry)
    cfg.set_exit(exit)

    cfg.set_terminator(entry, TS_COND, cond_node)
    for n in then_nodes:
        cfg.add_node_to_block(then_block, n)
    for n in else_nodes:
        cfg.add_node_to_block(else_block, n)
    cfg.set_terminator(then_block, TS_JUMP)
    cfg.set_terminator(else_block, TS_JUMP)
    cfg.set_terminator(exit, TS_RET)

    cfg.add_edge(entry, then_block, 1)
    cfg.add_edge(entry, else_block, 2)
    cfg.add_edge(then_block, exit, 0)
    cfg.add_edge(else_block, exit, 0)
    return cfg^


def build_while_loop_graph(
    var dag: Dag,
    cond_node: Int,
    body_nodes: List[Int],
    static_trips: Int = -1,
) -> CFGGraph:
    """Build a while loop: header(cond) -> body -> back to header; exit."""
    var cfg = CFGGraph(dag^)
    var header = cfg.new_block()
    var body = cfg.new_block()
    var exit = cfg.new_block()
    cfg.set_entry(header)
    cfg.set_exit(exit)

    cfg.set_terminator(header, TS_COND, cond_node)
    cfg.blocks[header].static_trips = static_trips
    for n in body_nodes:
        cfg.add_node_to_block(body, n)
    cfg.set_terminator(body, TS_JUMP)
    cfg.set_terminator(exit, TS_RET)

    cfg.add_edge(header, body, 1)
    cfg.add_edge(header, exit, 2)
    cfg.add_edge(body, header, 0)
    return cfg^


def cfg_validate(cfg: CFGGraph) -> Bool:
    var n = len(cfg.blocks)
    if cfg.entry < 0 or cfg.entry >= n or cfg.exit < 0 or cfg.exit >= n:
        return False
    for edge in cfg.edges:
        if edge.src < 0 or edge.src >= n or edge.dst < 0 or edge.dst >= n:
            return False
        if edge.kind < 0 or edge.kind > 2:
            return False
    return True
