# Mojo Q5_K multi-size test
from src.core.ops.cpu.simd.q5k_q8k_dot import vec_dot_q5_k_q8_k
from src.core.thread_pool import now_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def test_size(nb: Int, label: String):
    var q5_block_size = 176
    var q8_block_size = 292
    var iterations = 100000  # Same as llama.cpp test
    if nb >= 64:
        iterations = 50000
    if nb >= 256:
        iterations = 10000

    var q5_mem = unsafe_alloc[UInt8](nb * q5_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8_block_size, alignment=64)
    var q5_ptr = Pointer[UInt8, MutUntrackedOrigin](q5_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    for i in range(nb):
        var bo = i * q5_block_size
        q5_ptr.unsafe_offset(bo).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))
        q5_ptr.unsafe_offset(bo + 2).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](0.0))
        for j in range(12):
            q5_ptr.unsafe_offset(bo + 4 + j).unsafe_store(UInt8((j + 1) | (j << 4)))
        for j in range(32):
            q5_ptr.unsafe_offset(bo + 16 + j).unsafe_store(UInt8(j % 256))
        for j in range(128):
            q5_ptr.unsafe_offset(bo + 48 + j).unsafe_store(UInt8(j % 256))

    for i in range(nb):
        var bo = i * q8_block_size
        q8_ptr.unsafe_offset(bo).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))
        for j in range(256):
            q8_ptr.unsafe_offset(bo + 4 + j).unsafe_store(UInt8(64))
        for j in range(16):
            q8_ptr.unsafe_offset(bo + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(Scalar[DType.int16](1024))

    for _ in range(10000):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q5_k_q8_k(q5_ptr.unsafe_offset(i * q5_block_size), q8_ptr.unsafe_offset(i * q8_block_size))

    var start = now_ns()
    var result = Float32(0)
    for _ in range(iterations):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q5_k_q8_k(q5_ptr.unsafe_offset(i * q5_block_size), q8_ptr.unsafe_offset(i * q8_block_size))
        result = sum
    var end = now_ns()

    var elapsed_ms = Float64(end - start) / 1_000_000.0
    var gflops = 2.0 * Float64(nb * 256) * Float64(iterations) / (elapsed_ms / 1000.0) / 1_000_000_000.0
    print("Mojo", label + ":", gflops, "GFLOPS")

def main():
    print("=== Mojo Q5_K Multi-Size Performance ===")
    test_size(1, "nb=1")
    test_size(16, "nb=16")
    test_size(64, "nb=64")
    test_size(256, "nb=256")
