from src.core.ops.cpu.matmul_cpu import matmul_cpu_dynamic
from src.core.ops.cpu.rms_norm_cpu import rms_norm_cpu
from src.core.ops.cpu.softmax_cpu import softmax_cpu
from src.core.tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple

def main():
    # matmul 2x3 @ 3x2 = 2x2
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var b = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](3, 2))
    for i in range(6):
        a.set(i, Scalar[DType.float32](Float32(i + 1)))
    for i in range(6):
        b.set(i, Scalar[DType.float32](1.0))
    var c = matmul_cpu_dynamic[DType.float32](a, b)
    print("matmul[0,0]:", c.get(0), "matmul[0,1]:", c.get(1))

    # rms_norm [2, 4]
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    for i in range(8):
        x.set(i, Scalar[DType.float32](1.0))
    var r = rms_norm_cpu[DType.float32, 4](x)
    print("rms[0]:", r.get(0))

    # softmax [2, 4]
    var s = softmax_cpu[DType.float32, 4](x)
    print("softmax[0]:", s.get(0))
    print("OK")
