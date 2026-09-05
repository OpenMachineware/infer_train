# core/graph.mojo
#
# Computation graph construction and topological ordering.

from .device import Device
from .ops.base.op_interface import AnyTensor

# NodeID is just an integer index into the graph's node list.
comptime NodeID = Int


struct AttrValue(Copyable, ImplicitlyCopyable, Movable):
    """Small tagged union for operator attributes (stand-in for `Any`)."""

    var kind: Int8  # 0 = int, 1 = float, 2 = string
    var int_val: Int
    var float_val: Float32
    var str_val: String

    def __init__(out self, value: Int):
        self.kind = 0
        self.int_val = value
        self.float_val = 0
        self.str_val = String("")

    def __init__(out self, value: Float32):
        self.kind = 1
        self.int_val = 0
        self.float_val = value
        self.str_val = String("")

    def __init__(out self, value: String):
        self.kind = 2
        self.int_val = 0
        self.float_val = 0
        self.str_val = value


struct GraphNode(Movable):
    var node_id: Int
    var op_type: String
    var inputs: List[Int]
    var outputs: List[Int]
    var attrs: Dict[String, AttrValue]
    var saved: List[AnyTensor]
    var device_hint: Optional[Device]

    def __init__(
        out self,
        node_id: Int,
        op_type: String,
        inputs: List[Int],
        attrs: Dict[String, AttrValue],
    ):
        self.node_id = node_id
        self.op_type = op_type
        self.inputs = inputs.copy()
        self.outputs = List[Int]()
        self.attrs = attrs.copy()
        self.saved = List[AnyTensor]()
        self.device_hint = None


struct Graph(Movable):
    var nodes: List[GraphNode]
    var entry: List[Int]
    var exit: List[Int]
    var tensors: Dict[Int, List[AnyTensor]]

    def __init__(out self):
        self.nodes = List[GraphNode]()
        self.entry = List[Int]()
        self.exit = List[Int]()
        self.tensors = Dict[Int, List[AnyTensor]]()

    def add_node(
        mut self,
        op_type: String,
        inputs: List[Int],
        attrs: Dict[String, AttrValue],
    ) -> Int:
        """Append a node and wire its incoming edges; returns its NodeID."""
        var node_id = len(self.nodes)
        for input_id in inputs:
            self.nodes[input_id].outputs.append(node_id)
        var node = GraphNode(node_id, op_type, inputs, attrs)
        self.nodes.append(node^)
        if len(inputs) == 0:
            self.entry.append(node_id)
        return node_id

    def topo_sort(self) -> List[Int]:
        """Kahn's algorithm; returns nodes in dependency order."""
        var n = len(self.nodes)
        var indegree = List[Int]()
        for i in range(n):
            indegree.append(len(self.nodes[i].inputs))

        var queue = List[Int]()
        for i in range(n):
            if indegree[i] == 0:
                queue.append(i)

        var order = List[Int]()
        var head = 0
        while head < len(queue):
            var u = queue[head]
            head += 1
            order.append(u)
            for output_id in self.nodes[u].outputs:
                indegree[output_id] -= 1
                if indegree[output_id] == 0:
                    queue.append(output_id)
        return order^
