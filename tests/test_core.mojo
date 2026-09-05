from src.core.tensor import Tensor, tensor_zeros, tensor_copy
from src.core.device import Device, get_default_device, has_metal_gpu
from src.core.memory import MemoryPool
from std.utils.static_tuple import StaticTuple


def main():
    print("metal:", has_metal_gpu())
    print("default device:", get_default_device().name())
    var t = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    t.set(4, Scalar[DType.float32](7.5))
    print("shape:", t.shape()[0], t.shape()[1], "numel:", t.numel())
    print("get4:", t.get(4))
    print("index(1,1):", t.index(1, 1))
    var v = t.reshape[1](StaticTuple[Int, 1](6))
    print("view numel:", v.numel(), "get4:", v.get(4))
    var c = tensor_copy[DType.float32, 2](t)
    print("copy get4:", c.get(4))
    var pool = MemoryPool(1024)
    _ = pool.allocate(128)
    print("pool offset:", pool.used(), "capacity:", pool.capacity())
    pool.reset()
    print("pool after reset:", pool.used())
    print("OK")
