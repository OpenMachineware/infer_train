# core/optimizer/fusion.mojo
#
# M5 pass: pattern-matching operator fusion on the optimization IR.
#
#   lm_head(x, w) + add_bias(·, b)   -> fused_matmul_add_bias(x, w, b)
#   lm_head(x, w) + rms_norm(·)      -> fused_matmul_rms_norm(x, w)
#   swiglu(g, u) + lm_head(·, w)     -> fused_swiglu_matmul(g, u, w)
#
# Each match requires the producer to have exactly one consumer (checked
# against the whole edge set), so fusing can never duplicate work.  The
# producer node is left dead and dce removes it in the same pipeline run.

from .dag_ir import (
    Dag,
    OP_CONST,
    OP_LM_HEAD,
    OP_ADD_BIAS,
    OP_RMS_NORM,
    OP_SWIGLU,
    OP_FUSED_MATMUL_ADD_BIAS,
    OP_FUSED_MATMUL_RMS_NORM,
    OP_FUSED_SWIGLU_MATMUL,
)


def _use_count(dag: Dag, node_id: Int) -> Int:
    var count = 0
    for node in dag.nodes:
        for input_id in node.inputs:
            if input_id == node_id:
                count += 1
    return count


def dag_fusion(mut dag: Dag) -> Int:
    """Apply the fusion patterns; returns fusions done."""
    var fused = 0
    for i in range(len(dag.nodes)):
        if dag.nodes[i].op == OP_ADD_BIAS and len(dag.nodes[i].inputs) == 2:
            var producer = dag.nodes[i].inputs[0]
            var bias = dag.nodes[i].inputs[1]
            if (
                dag.nodes[producer].op == OP_LM_HEAD
                and len(dag.nodes[producer].inputs) == 2
                and _use_count(dag, producer) == 1
            ):
                dag.nodes[i].op = OP_FUSED_MATMUL_ADD_BIAS
                var inputs = List[Int]()
                inputs.append(dag.nodes[producer].inputs[0])
                inputs.append(dag.nodes[producer].inputs[1])
                inputs.append(bias)
                dag.nodes[i].inputs = inputs^
                dag.nodes[producer].op = OP_CONST
                dag.nodes[producer].inputs = List[Int]()
                fused += 1
        elif dag.nodes[i].op == OP_RMS_NORM and len(dag.nodes[i].inputs) == 1:
            var producer = dag.nodes[i].inputs[0]
            if (
                dag.nodes[producer].op == OP_LM_HEAD
                and len(dag.nodes[producer].inputs) == 2
                and _use_count(dag, producer) == 1
            ):
                dag.nodes[i].op = OP_FUSED_MATMUL_RMS_NORM
                dag.nodes[i].inputs = dag.nodes[producer].inputs.copy()
                dag.nodes[producer].op = OP_CONST
                dag.nodes[producer].inputs = List[Int]()
                fused += 1
        elif dag.nodes[i].op == OP_LM_HEAD and len(dag.nodes[i].inputs) == 2:
            var producer = dag.nodes[i].inputs[0]
            var w = dag.nodes[i].inputs[1]
            if (
                dag.nodes[producer].op == OP_SWIGLU
                and len(dag.nodes[producer].inputs) == 2
                and _use_count(dag, producer) == 1
            ):
                dag.nodes[i].op = OP_FUSED_SWIGLU_MATMUL
                var inputs = List[Int]()
                inputs.append(dag.nodes[producer].inputs[0])
                inputs.append(dag.nodes[producer].inputs[1])
                inputs.append(w)
                dag.nodes[i].inputs = inputs^
                dag.nodes[producer].op = OP_CONST
                dag.nodes[producer].inputs = List[Int]()
                fused += 1
    return fused
