from src.core.ops.base.op_registry import OpRegistry
from src.core.ops.base.op_interface import to_any, AnyTensor
from src.core.tensor import Tensor, tensor_zeros
from src.core.graph import Graph, AttrValue
from src.runtime.interpreter import Interpreter
from src.core.sampler import Sampler, greedy_sample, sample
from std.utils.static_tuple import StaticTuple

def main():
    var reg = OpRegistry()
    reg.register_default_ops()

    # Build a graph: matmul -> rms_norm
    var graph = Graph()
    var attrs0 = Dict[String, AttrValue]()
    var n0 = graph.add_node("matmul", List[Int](), attrs0)
    var attrs1 = Dict[String, AttrValue]()
    attrs1["dim"] = AttrValue(4)
    var n1 = graph.add_node("rms_norm", [n0], attrs1)

    # inputs: a (2x3), b (3x4)
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var b = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](3, 4))
    for i in range(6):
        a.set(i, Scalar[DType.float32](1.0))
    for i in range(12):
        b.set(i, Scalar[DType.float32](1.0))

    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float32, 2](a))
    inputs.append(to_any[DType.float32, 2](b))

    var interp = Interpreter(graph^, reg^)
    var outputs = interp.run(inputs)
    print("graph output numel:", outputs[0].numel)

    # sampler
    var logits = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](5))
    for i in range(5):
        logits.set(i, Scalar[DType.float32](Float32(i)))
    var s = Sampler(temperature=Float32(1.0), top_k=0, top_p=Float32(1.0))
    var generated = List[Int]()
    print("sampled:", sample[DType.float32, 5](logits, s, generated))
    print("greedy:", greedy_sample[DType.float32, 5](logits))
    print("OK")
