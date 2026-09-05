# tests/test_optimizer.mojo
#
# M5: DAG optimization passes + verify.
#
# Builds a small computation graph with redundancies (CSE), dead code
# (DCE), constant expressions (constant fold), algebraic identities
# (simplify) and fusion opportunities, then checks:
#   1. the optimized DAG still produces identical outputs (verify_dags);
#   2. every pass reports the expected change count;
#   3. each pass in isolation preserves semantics (per-pass verify);
#   4. the memory plan pools non-overlapping lifetimes (>=20% smaller).
#
# Built with `-I .` so `src.` imports resolve (see the Makefile).

from src.core.cfg import CFGGraph, TS_JUMP, TS_RET
from src.core.optimizer.cfg_to_dag import Guard
from src.core.optimizer.cfg_optimizer import optimize_cfg
from src.core.optimizer.cfg_inline import cfg_inline
from src.core.optimizer.cfg_dce import cfg_dce
from src.core.optimizer.verify import execute_cfg, _max_abs_diff
from src.core.cfg import build_if_cond_graph, build_while_loop_graph
from src.core.optimizer.dag_ir import (
    Dag,
    OP_ADD,
    OP_ADD_BIAS,
    OP_LM_HEAD,
    OP_RMS_NORM,
    OP_SOFTMAX,
    OP_FUSED_MATMUL_ADD_BIAS,
)
from src.core.optimizer.dag_optimizer import optimize_dag, summarize_dag
from src.core.optimizer.verify import verify_dags, clone_dag, execute_dag
from src.core.optimizer.shape_inference import (
    set_input_shape,
    dag_shape_inference,
)
from src.core.optimizer.constant_fold import dag_constant_fold
from src.core.optimizer.simplify import dag_simplify
from src.core.optimizer.cse import dag_cse
from src.core.optimizer.fusion import dag_fusion
from src.core.optimizer.dce import dag_dce
from src.core.optimizer.memory_plan import dag_memory_plan
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.base.op_interface import AnyTensor, to_any, from_any
from std.utils.static_tuple import StaticTuple
from std.os.os import abort


def _shape(rows: Int, cols: Int) -> List[Int]:
    var s = List[Int]()
    s.append(rows)
    s.append(cols)
    return s^


def _in(a: Int, b: Int) -> List[Int]:
    var l = List[Int]()
    l.append(a)
    l.append(b)
    return l^


def _one(a: Int) -> List[Int]:
    var l = List[Int]()
    l.append(a)
    return l^


def _fill_f32(mut t: Tensor[DType.float32, 2], seed: Int, scale: Float32):
    var n = t.numel()
    for i in range(n):
        var v = Float32((i * 7 + seed * 13) % 101) / Float32(101.0) * scale
        t.set(i, Scalar[DType.float32](v))


def _make_const(rows: Int, cols: Int, seed: Int) -> AnyTensor:
    var t = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](rows, cols))
    _fill_f32(t, seed, 1)
    return to_any[DType.float32, 2](t)


def _zero_const(rows: Int, cols: Int) -> AnyTensor:
    var t = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](rows, cols))
    return to_any[DType.float32, 2](t)


def build_test_dag(mut dag: Dag, mut inputs: List[AnyTensor]):
    """Fill `dag`/`inputs` with a DAG exercising every pass."""
    var x = dag.add_input(0)
    set_input_shape(dag, x, _shape(2, 4), 0)

    var w = dag.add_const(_make_const(6, 4, 1), "W")
    var bias = dag.add_const(_make_const(6, 1, 2), "bias")
    var zero = dag.add_const(_zero_const(2, 6), "zero")
    var c1 = dag.add_const(_make_const(2, 6, 4), "c1")
    var c2 = dag.add_const(_make_const(2, 6, 5), "c2")

    var lm0 = dag.add_node(OP_LM_HEAD, _in(x, w), 0)
    var ab = dag.add_node(OP_ADD_BIAS, _in(lm0, bias), 0)
    var rn = dag.add_node(OP_RMS_NORM, _one(ab), 0)
    var lm1 = dag.add_node(OP_LM_HEAD, _in(x, w), 0)  # CSE dup
    _ = dag.add_node(OP_SOFTMAX, _one(lm1), 0)  # dead
    var z = dag.add_node(OP_ADD, _in(rn, zero), 0)  # x + 0 -> x
    var cs = dag.add_node(OP_ADD, _in(c1, c2), 0)  # const fold
    var out = dag.add_node(OP_ADD, _in(z, cs), 0)
    dag.set_outputs(_one(out))

    var input_x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    _fill_f32(input_x, 9, 1)
    inputs.append(to_any[DType.float32, 2](input_x))


def check(label: String, ok: Bool):
    if ok:
        print("[ok]   ", label)
    else:
        print("[FAIL] ", label)
        abort()


def main():
    print("== M5 optimizer tests ==")

    # -- full pipeline + verify --------------------------------------------
    var dag = Dag(1)
    var inputs = List[AnyTensor]()
    build_test_dag(dag, inputs)
    var original = clone_dag(dag)
    var stats = optimize_dag(dag)

    print("after pipeline: ", summarize_dag(dag))
    print(
        "  folded=",
        stats.folded,
        " simplified=",
        stats.simplified,
        " cse=",
        stats.cse_removed,
        " fused=",
        stats.fused,
        " dce=",
        stats.dce_removed,
    )
    check(
        "pipeline preserves outputs (verify_dags)",
        verify_dags(original, dag, inputs),
    )
    check("constant fold ran", stats.folded >= 1)
    check("simplify ran (x + 0 -> x)", stats.simplified >= 1)
    check("cse removed the duplicate lm_head", stats.cse_removed >= 1)
    check("fusion ran (lm_head + add_bias)", stats.fused >= 1)
    check("dce removed dead code", stats.dce_removed >= 1)

    var fused_present = False
    for node in dag.nodes:
        if node.op == OP_FUSED_MATMUL_ADD_BIAS:
            fused_present = True
    check("fused_matmul_add_bias node present", fused_present)

    # -- per-pass verify (each pass alone must preserve semantics) --------
    var d1 = Dag(1)
    var i1 = List[AnyTensor]()
    build_test_dag(d1, i1)
    _ = dag_constant_fold(d1)
    check("per-pass verify: constant_fold", verify_dags(original, d1, i1))

    var d2 = Dag(1)
    var i2 = List[AnyTensor]()
    build_test_dag(d2, i2)
    _ = dag_simplify(d2)
    _ = dag_dce(d2)
    check("per-pass verify: simplify", verify_dags(original, d2, i2))

    var d3 = Dag(1)
    var i3 = List[AnyTensor]()
    build_test_dag(d3, i3)
    _ = dag_cse(d3)
    _ = dag_dce(d3)
    check("per-pass verify: cse", verify_dags(original, d3, i3))

    var d4 = Dag(1)
    var i4 = List[AnyTensor]()
    build_test_dag(d4, i4)
    _ = dag_fusion(d4)
    _ = dag_dce(d4)
    check("per-pass verify: fusion", verify_dags(original, d4, i4))

    # -- memory planning: chain graph, liveness pooling >= 20% ------------
    var chain = Dag(1)
    var h = chain.add_input(0)
    set_input_shape(chain, h, _shape(2, 6), 0)
    var n_blocks = 16
    var w_const = chain.add_const(_make_const(6, 6, 7), "Wc")
    var prev = h
    for _ in range(n_blocks):
        var lm = chain.add_node(OP_LM_HEAD, _in(prev, w_const), 0)
        var rn = chain.add_node(OP_RMS_NORM, _one(lm), 0)
        prev = rn
    chain.set_outputs(_one(prev))

    _ = dag_shape_inference(chain)
    var plan = dag_memory_plan(chain)
    var reduction_pct = (
        (plan.baseline_bytes - plan.planned_bytes) * 100 // plan.baseline_bytes
    )
    print(
        "  memory plan: baseline=",
        plan.baseline_bytes,
        "B planned=",
        plan.planned_bytes,
        "B (",
        reduction_pct,
        "% smaller)",
        " slots=",
        plan.n_slots,
    )
    check("memory plan reduces peak >= 20%", reduction_pct >= 20)

    # the chain graph also verifies under the full pipeline
    var chain_copy = clone_dag(chain)
    var chain_inputs = List[AnyTensor]()
    var cx = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 6))
    _fill_f32(cx, 5, 1)
    chain_inputs.append(to_any[DType.float32, 2](cx))
    var cstats = optimize_dag(chain)
    print("  chain after pipeline: ", summarize_dag(chain))
    check(
        "chain graph verifies after pipeline",
        verify_dags(chain_copy, chain, chain_inputs),
    )
    check("chain fusion fused all blocks", cstats.fused == n_blocks)

    # ------------------------------------------------------------------
    # CFG: static control flow -> cfg_to_dag -> dag pipeline + verify
    # ------------------------------------------------------------------
    var c_dag = Dag(1)
    var cx_in = c_dag.add_input(0)
    set_input_shape(c_dag, cx_in, _shape(2, 6), 0)
    var cw1 = c_dag.add_const(_make_const(6, 6, 21), "W1")
    var cw2 = c_dag.add_const(_make_const(6, 6, 22), "W2")
    var cbias = c_dag.add_const(_make_const(6, 1, 23), "bias")
    var ccond = c_dag.add_const(_make_const(1, 1, 24), "cond")
    # make the condition a 1.0 (true)
    var ccond_t = from_any[DType.float32, 2](
        c_dag.nodes[ccond].const_data.value()
    )
    ccond_t.set(0, Scalar[DType.float32](1.0))
    var t_lm = c_dag.add_node(OP_LM_HEAD, _in(cx_in, cw1), 0)
    var t_ab = c_dag.add_node(OP_ADD_BIAS, _in(t_lm, cbias), 0)
    var e_lm = c_dag.add_node(OP_LM_HEAD, _in(cx_in, cw2), 0)
    var then_nodes = List[Int]()
    then_nodes.append(t_lm)
    then_nodes.append(t_ab)
    var else_nodes = List[Int]()
    else_nodes.append(e_lm)
    var cfg_if = build_if_cond_graph(c_dag^, ccond, then_nodes^, else_nodes^)

    var cfg_inputs = List[AnyTensor]()
    var cix = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 6))
    _fill_f32(cix, 11, 1)
    cfg_inputs.append(to_any[DType.float32, 2](cix))

    var cfg_ref = execute_cfg(cfg_if, cfg_inputs)
    var lower_res = optimize_cfg(cfg_if^)
    check("cfg_to_dag succeeds on static if", lower_res.success)
    check("static if recorded a guard", len(lower_res.guards) == 1)
    if len(lower_res.guards) == 1:
        check("guard took the true branch", lower_res.guards[0].taken == 1)
    # lowered DAG output vs CFG interpreter output
    var cfg_dag_ref = execute_dag(lower_res.dag, cfg_inputs)
    var cfg_match = True
    if len(cfg_ref) == len(cfg_dag_ref):
        for k in range(len(cfg_ref)):
            var diff = _max_abs_diff(cfg_ref[k], cfg_dag_ref[k])
            if diff > Float32(1e-4):
                print(
                    "  [cfg] output ",
                    k,
                    " diff=",
                    diff,
                    " numel ref=",
                    cfg_ref[k].numel,
                    " got=",
                    cfg_dag_ref[k].numel,
                )
                cfg_match = False
    else:
        print("  [cfg] output count ", len(cfg_ref), " vs ", len(cfg_dag_ref))
        cfg_match = False
    check("cfg_to_dag output matches CFG interpreter", cfg_match)
    check(
        "cfg pipeline fused the then-branch linear", lower_res.stats.fused >= 1
    )
    print("  lowered dag: ", summarize_dag(lower_res.dag))

    # static while: unrolls `trips` times
    var w_dag = Dag(1)
    var wh = w_dag.add_input(0)
    set_input_shape(w_dag, wh, _shape(2, 6), 0)
    var ww = w_dag.add_const(_make_const(6, 6, 31), "Ww")
    var wcond = w_dag.add_const(_make_const(1, 1, 32), "cond")
    var wcond_t = from_any[DType.float32, 2](
        w_dag.nodes[wcond].const_data.value()
    )
    wcond_t.set(0, Scalar[DType.float32](1.0))
    var body_lm = w_dag.add_node(OP_LM_HEAD, _in(wh, ww), 0)
    var body_rn = w_dag.add_node(OP_RMS_NORM, _one(body_lm), 0)
    var body_nodes = List[Int]()
    body_nodes.append(body_lm)
    body_nodes.append(body_rn)
    var cfg_while = build_while_loop_graph(w_dag^, wcond, body_nodes^, 4)

    var w_inputs = List[AnyTensor]()
    var wix = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 6))
    _fill_f32(wix, 13, 1)
    w_inputs.append(to_any[DType.float32, 2](wix))

    var w_ref = execute_cfg(cfg_while, w_inputs)
    var w_res = optimize_cfg(cfg_while^)
    check("cfg_to_dag succeeds on static while", w_res.success)
    var w_dag_ref = execute_dag(w_res.dag, w_inputs)
    var w_match = True
    if len(w_ref) == len(w_dag_ref):
        for k in range(len(w_ref)):
            if _max_abs_diff(w_ref[k], w_dag_ref[k]) > Float32(1e-4):
                w_match = False
    else:
        w_match = False
    check("static while unroll matches CFG interpreter", w_match)
    check("while unrolled all 4 iterations (fused x4)", w_res.stats.fused >= 4)
    print("  lowered while dag: ", summarize_dag(w_res.dag))

    # dynamic control flow refuses to lower (M6 scope)
    var d_dag = Dag(2)
    var dx = d_dag.add_input(0)
    var dcond = d_dag.add_input(1)
    set_input_shape(d_dag, dx, _shape(2, 6), 0)
    set_input_shape(d_dag, dcond, _shape(1, 1), 0)
    var d_lm = d_dag.add_node(OP_RMS_NORM, _one(dx), 0)
    var d_then = List[Int]()
    d_then.append(d_lm)
    var d_else = List[Int]()
    var d_cfg = build_if_cond_graph(d_dag^, dcond, d_then^, d_else^)
    var d_res = optimize_cfg(d_cfg^)
    check("data-dependent cond refuses cfg_to_dag (M6)", not d_res.success)

    # cfg_inline: a jump chain merges into its predecessor
    var i_dag = Dag(1)
    var ix = i_dag.add_input(0)
    set_input_shape(i_dag, ix, _shape(2, 6), 0)
    var i_cfg = CFGGraph(i_dag^)
    var ib0 = i_cfg.new_block()
    var ib1 = i_cfg.new_block()
    var ib2 = i_cfg.new_block()
    i_cfg.set_entry(ib0)
    i_cfg.set_exit(ib2)
    i_cfg.set_terminator(ib0, TS_JUMP)
    i_cfg.set_terminator(ib1, TS_JUMP)
    i_cfg.set_terminator(ib2, TS_RET)
    i_cfg.add_edge(ib0, ib1, 0)
    i_cfg.add_edge(ib1, ib2, 0)
    var inlined = cfg_inline(i_cfg)
    check("cfg_inline merged the jump chain", inlined == 1)

    # cfg_dce: an unreachable block is dropped
    var x_dag = Dag(1)
    var xx = x_dag.add_input(0)
    set_input_shape(x_dag, xx, _shape(2, 6), 0)
    var x_cfg = CFGGraph(x_dag^)
    var xb0 = x_cfg.new_block()
    var xb1 = x_cfg.new_block()
    var xdead = x_cfg.new_block()
    x_cfg.set_entry(xb0)
    x_cfg.set_exit(xb1)
    x_cfg.set_terminator(xb0, TS_JUMP)
    x_cfg.set_terminator(xb1, TS_RET)
    x_cfg.set_terminator(xdead, TS_RET)
    x_cfg.add_edge(xb0, xb1, 0)
    var dce_removed = cfg_dce(x_cfg)
    check("cfg_dce removed the unreachable block", dce_removed == 1)

    print("== optimizer tests passed ==")
