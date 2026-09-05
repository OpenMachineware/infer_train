# core/optimizer/cfg_dce.mojo
#
# M5 pass: dead-basic-block elimination for CFGs.
#
# Blocks unreachable from the entry block (walking the edge set forward)
# are removed; the block list and edges are rebuilt with index remapping.

from ..cfg import CFGGraph, CFGBlock, CFGEdge


def cfg_dce(mut cfg: CFGGraph) -> Int:
    """Remove unreachable blocks; returns the number removed."""
    var n = len(cfg.blocks)
    var reachable = List[Bool](length=n, fill=False)

    var stack = List[Int]()
    stack.append(cfg.entry)
    while len(stack) > 0:
        var current = stack.pop()
        if current < 0 or current >= n or reachable[current]:
            continue
        reachable[current] = True
        for edge in cfg.edges:
            if edge.src == current:
                stack.append(edge.dst)

    var remap = Dict[Int, Int]()
    var new_blocks = List[CFGBlock]()
    for i in range(n):
        if reachable[i]:
            remap[i] = len(new_blocks)
            var moved = CFGBlock(len(new_blocks))
            moved.nodes = cfg.blocks[i].nodes.copy()
            moved.terminator = cfg.blocks[i].terminator
            moved.cond_node = cfg.blocks[i].cond_node
            moved.static_trips = cfg.blocks[i].static_trips
            new_blocks.append(moved^)

    var removed = n - len(new_blocks)
    cfg.blocks = new_blocks^
    var new_edges = List[CFGEdge]()
    for edge in cfg.edges:
        if reachable[edge.src] and reachable[edge.dst]:
            new_edges.append(
                CFGEdge(
                    remap.get(edge.src, -1),
                    remap.get(edge.dst, -1),
                    edge.kind,
                )
            )
    cfg.edges = new_edges^
    cfg.entry = remap.get(cfg.entry, 0)
    cfg.exit = remap.get(cfg.exit, 0)
    return removed
