# core/optimizer/dag_ir.mojo
#
# M5: the compact DAG optimization IR shared by every pass.
#
# A `Dag` is a flat list of nodes with integer op codes and index-based
# edges.  Constants hold their payload inline as `AnyTensor` (the same
# type-erased handle the OpRegistry uses), so passes can fold them with the
# real kernels and the verify executor can run the graph as-is.  The IR is
# deliberately small: every pass and the verify harness operate on this one
# structure, and it is the shape the Python-side translator mirrors.

from ..ops.base.op_interface import AnyTensor

# -- op codes ----------------------------------------------------------------

comptime OP_CONST = 0
comptime OP_INPUT = 1
comptime OP_MATMUL = 2
comptime OP_LM_HEAD = 3
comptime OP_ADD = 4
comptime OP_ADD_BIAS = 5
comptime OP_RMS_NORM = 6
comptime OP_SOFTMAX = 7
comptime OP_SWIGLU = 8
comptime OP_EMBEDDING = 9
comptime OP_FUSED_MATMUL_ADD_BIAS = 10
comptime OP_FUSED_MATMUL_ADD = 11
comptime OP_FUSED_MATMUL_RMS_NORM = 12
comptime OP_FUSED_SWIGLU_MATMUL = 13
comptime OP_COUNT = 14

# dtype codes (match the C ABI convention)
comptime DT_F32 = 0
comptime DT_F16 = 1
comptime DT_I32 = 2


def op_name(op: Int) -> String:
    """Debug name of an op code."""
    if op == OP_CONST:
        return "const"
    if op == OP_INPUT:
        return "input"
    if op == OP_MATMUL:
        return "matmul"
    if op == OP_LM_HEAD:
        return "lm_head"
    if op == OP_ADD:
        return "add"
    if op == OP_ADD_BIAS:
        return "add_bias"
    if op == OP_RMS_NORM:
        return "rms_norm"
    if op == OP_SOFTMAX:
        return "softmax"
    if op == OP_SWIGLU:
        return "swiglu"
    if op == OP_EMBEDDING:
        return "embedding"
    if op == OP_FUSED_MATMUL_ADD_BIAS:
        return "fused_matmul_add_bias"
    if op == OP_FUSED_MATMUL_ADD:
        return "fused_matmul_add"
    if op == OP_FUSED_MATMUL_RMS_NORM:
        return "fused_matmul_rms_norm"
    if op == OP_FUSED_SWIGLU_MATMUL:
        return "fused_swiglu_matmul"
    return "op?"


def op_is_fused(op: Int) -> Bool:
    return op >= OP_FUSED_MATMUL_ADD_BIAS and op < OP_COUNT


def max_inputs_of(op: Int) -> Int:
    """Maximum number of inputs an op consumes (for validation)."""
    if (
        op == OP_MATMUL
        or op == OP_LM_HEAD
        or op == OP_ADD
        or op == OP_RMS_NORM
        or op == OP_SOFTMAX
        or op == OP_FUSED_MATMUL_RMS_NORM
        or op == OP_SWIGLU
        or op == OP_EMBEDDING
    ):
        return 2
    if (
        op == OP_ADD_BIAS
        or op == OP_FUSED_MATMUL_ADD_BIAS
        or op == OP_FUSED_MATMUL_ADD
        or op == OP_FUSED_SWIGLU_MATMUL
    ):
        return 3
    if op == OP_CONST or op == OP_INPUT:
        return 0
    return 0


struct DagNode(Copyable, Movable):
    var op: Int
    var inputs: List[Int]
    var dtype: Int
    var shape: List[Int]
    var const_data: Optional[AnyTensor]
    var name: String
    # memory-planning annotations (filled by memory_plan.mojo)
    var live_start: Int
    var live_end: Int
    var slot: Int

    def __init__(
        out self,
        op: Int,
        inputs: List[Int],
        dtype: Int,
    ):
        self.op = op
        self.inputs = inputs.copy()
        self.dtype = dtype
        self.shape = List[Int]()
        self.const_data = None
        self.name = String("")
        self.live_start = -1
        self.live_end = -1
        self.slot = -1


struct Dag(Movable):
    var nodes: List[DagNode]
    var outputs: List[Int]
    var n_inputs: Int

    def __init__(out self, n_inputs: Int = 0):
        self.nodes = List[DagNode]()
        self.outputs = List[Int]()
        self.n_inputs = n_inputs

    def add_node(
        mut self,
        op: Int,
        inputs: List[Int],
        dtype: Int,
    ) -> Int:
        """Append a node; returns its index."""
        var node = DagNode(op, inputs, dtype)
        self.nodes.append(node^)
        return len(self.nodes) - 1

    def add_input(mut self, dtype: Int) -> Int:
        return self.add_node(OP_INPUT, List[Int](), dtype)

    def add_const(mut self, var data: AnyTensor, name: String) -> Int:
        var dtype = DT_F32
        if data.dtype == DType.float16:
            dtype = DT_F16
        elif data.dtype == DType.int32:
            dtype = DT_I32
        var id = self.add_node(OP_CONST, List[Int](), dtype)
        self.nodes[id].const_data = Optional(data)
        self.nodes[id].name = name
        var shape = List[Int]()
        for i in range(data.rank):
            shape.append(data.shape[i])
        self.nodes[id].shape = shape^
        return id

    def set_outputs(mut self, outputs: List[Int]):
        self.outputs = outputs.copy()

    def validate(self) -> Bool:
        """Cheap structural check (inputs in range, arity respected)."""
        var n = len(self.nodes)
        for i in range(n):
            var node = self.nodes[i]
            if node.op >= OP_COUNT:
                return False
            if len(node.inputs) > max_inputs_of(node.op):
                return False
            for input_id in node.inputs:
                if input_id < 0 or input_id >= n:
                    return False
        for out in self.outputs:
            if out < 0 or out >= n:
                return False
        return True
