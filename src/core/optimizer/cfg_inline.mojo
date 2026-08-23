# core/optimizer/cfg_inline.mojo
#
# M5 pass: inline small basic blocks into their single predecessor.
#
# A block is inlined when it has exactly one incoming edge, is not the
# entry, its terminator is a plain jump (no branch), and its predecessor
# is also a plain-jump block (so the merged code runs on the same path).
# Its nodes are appended to the predecessor and the two edges are replaced
# by one, eliminating a block dispatch at runtime.

from ..cfg import CFGGraph, CFGEdge, TS_JUMP


def _in_degree(cfg: CFGGraph, block: Int) -> Int:
    var count = 0
    for edge in cfg.edges:
        if edge.dst == block:
            count += 1
    return count


def _pred_of(cfg: CFGGraph, block: Int) -> Int:
    for edge in cfg.edges:
        if edge.dst == block:
            return edge.src
    return -1


def _succ_of(cfg: CFGGraph, block: Int) -> Int:
    var succ = -1
    var count = 0
    for edge in cfg.edges:
        if edge.src == block:
            succ = edge.dst
            count += 1
    if count == 1:
        return succ
    return -1


def cfg_inline(mut cfg: CFGGraph, max_nodes: Int = 8) -> Int:
    """Inline small single-predecessor blocks; returns inlines done."""
    var inlined = 0
    var changed = True
    while changed:
        changed = False
        var n = len(cfg.blocks)
        for i in range(n):
            if i == cfg.entry or i == cfg.exit:
                continue
            if cfg.blocks[i].terminator != TS_JUMP:
                continue
            if len(cfg.blocks[i].nodes) > max_nodes:
                continue
            if _in_degree(cfg, i) != 1:
                continue
            var pred = _pred_of(cfg, i)
            var succ = _succ_of(cfg, i)
            if pred < 0 or succ < 0 or pred == i:
                continue
            if cfg.blocks[pred].terminator != TS_JUMP:
                continue
            # merge: append the block's nodes to the predecessor
            for node_id in cfg.blocks[i].nodes:
                cfg.blocks[pred].nodes.append(node_id)
            # edge surgery: drop (pred->i) and (i->succ), add (pred->succ)
            var new_edges = List[CFGEdge]()
            for edge in cfg.edges:
                if edge.src == pred and edge.dst == i:
                    continue
                if edge.src == i:
                    continue
                new_edges.append(edge)
            new_edges.append(CFGEdge(pred, succ, 0))
            cfg.edges = new_edges^
            # leave block i empty; cfg_dce removes it in the pipeline
            cfg.blocks[i].nodes = List[Int]()
            inlined += 1
            changed = True
            break
    return inlined
