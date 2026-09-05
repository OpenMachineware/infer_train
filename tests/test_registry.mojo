from src.core.ops.base.op_registry import OpRegistry
from src.core.ops.base.op_interface import to_any, AnyTensor
from src.core.tensor import Tensor, tensor_zeros
from src.core.device import Device
from std.utils.static_tuple import StaticTuple


def main():
    var reg = OpRegistry()
    reg.register_default_ops()

    # matmul via registry (AnyTensor)
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var b = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](3, 2))
    for i in range(6):
        a.set(i, Scalar[DType.float32](Float32(i + 1)))
    for i in range(6):
        b.set(i, Scalar[DType.float32](1.0))

    var op = reg.get("matmul", Device.CPU)
    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float32, 2](a))
    inputs.append(to_any[DType.float32, 2](b))
    var outs = op.forward(inputs)
    print("registry matmul out:", outs[0].numel, "dtype:", outs[0].dtype)
    print("OK")
