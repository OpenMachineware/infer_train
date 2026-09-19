# Test Q4_K performance across different sizes
from src.core.ops.cpu.simd.q4k_q8k_dot import vec_dot_q4_k_q8_k
from src.core.thread_pool import now_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

comptime QK_K = 256

def test_size(nb: Int, label: String):
    var q4_block_size = 144
    var q8_block_size = 292
    var iterations = 10000
    if nb >= 64:
        iterations = 5000
    if nb >= 256:
        iterations = 2000

    var q4_mem = unsafe_alloc[UInt8](nb * q4_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8_block_size, alignment=64)
    var q4_ptr = Pointer[UInt8, MutUntrackedOrigin](q4_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    for i in range(nb):
        var bo = i * q4_block_size
        q4_ptr.unsafe_offset(bo).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))
        q4_ptr.unsafe_offset(bo + 2).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](0.0))
        for j in range(12):
            q4_ptr.unsafe_offset(bo + 4 + j).unsafe_store(UInt8(40 + (j % 8)))
        for j in range(128):
            q4_ptr.unsafe_offset(bo + 16 + j).unsafe_store(UInt8((j * 2) % 256))

    for i in range(nb):
        var bo = i * q8_block_size
        q8_ptr.unsafe_offset(bo).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))
        for j in range(256):
            q8_ptr.unsafe_offset(bo + 4 + j).unsafe_store(UInt8(64))
        for j in range(16):
            q8_ptr.unsafe_offset(bo + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(Scalar[DType.int16](1024))

    var result = Float32(0)
    for _ in range(1000):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q4_k_q8_k(q4_ptr.unsafe_offset(i * q4_block_size), q8_ptr.unsafe_offset(i * q8_block_size))
        result = sum

    var start = now_ns()
    for _ in range(iterations):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q4_k_q8_k(q4_ptr.unsafe_offset(i * q4_block_size), q8_ptr.unsafe_offset(i * q8_block_size))
        result = sum
    var end = now_ns()

    var elapsed_s = Float64(end - start) / 1_000_000_000.0
    var flops = 2.0 * Float64(nb * QK_K) * Float64(iterations)
    var gflops = flops / elapsed_s / 1_000_000_000.0

    print("Mojo     ", label, ":", gflops, "GFLOPS (nb=", nb, "result=", result, ")")

def main():
    print("=== Mojo Q4_K Multi-Size Performance ===")
    test_size(1, "nb=1")
    test_size(16, "nb=16")
    test_size(64, "nb=64")
    test_size(256, "nb=256")
