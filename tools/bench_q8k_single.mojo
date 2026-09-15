# Benchmark: Q8_K + SDOT single-threaded matmul
#
# Tests the Q8_K + SDOT path without threading.

from src.core.tensor import Tensor, tensor_zeros
from src.core.thread_pool import now_ns, num_pcores
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.quantized.quant_types import QuantType
from std.memory.alloc import unsafe_alloc
from std.utils.static_tuple import StaticTuple


comptime QK_K = 256
comptime BB_Q4K = 144


def create_valid_q4_k_weights(N: Int, K: Int) -> Tensor[DType.uint8, 2]:
    """Create Q4_K weights with proper block format."""
    var nb = K // QK_K
    var total_bytes = N * nb * BB_Q4K
    var data = unsafe_alloc[UInt8](total_bytes)
    
    for row in range(N):
        for block_idx in range(nb):
            var block = data.unsafe_offset(row * nb * BB_Q4K + block_idx * BB_Q4K)
            
            block.unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
                val=Float16(1.0), offset=0
            )
            block.unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
                val=Float16(0.0), offset=1
            )
            
            for i in range(12):
                block.unsafe_offset(4 + i).unsafe_store(val=UInt8(8))
            
            for i in range(128):
                block.unsafe_offset(16 + i).unsafe_store(val=UInt8(0x88))
    
    return Tensor[DType.uint8, 2](StaticTuple[Int, 2](N, nb * BB_Q4K), data)


def create_activation(M: Int, K: Int) -> Tensor[DType.float16, 2]:
    """Create activation tensor."""
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.data().unsafe_offset(i).unsafe_store(
            val=Scalar[DType.float16](Float16(Float32((i % 100 + 1)) / 100.0))
        )
    return x


def bench_q8k_single(M: Int, N: Int, K: Int, n_warmup: Int = 3, n_iter: Int = 10):
    """Benchmark Q8_K + SDOT single-threaded matmul."""
    print("\n=== Q8_K + SDOT Single-threaded ===")
    print("  M=", M, ", N=", N, ", K=", K)
    
    var x = create_activation(M, K)
    var w_quant = create_valid_q4_k_weights(N, K)
    var scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    
    # Warmup
    for _ in range(n_warmup):
        _ = matmul_quantized_q8k[QuantType.Q4_K_M](x, w_quant, scale)
    
    # Benchmark
    var times = List[Int]()
    for _ in range(n_iter):
        var t0 = now_ns()
        var out = matmul_quantized_q8k[QuantType.Q4_K_M](x, w_quant, scale)
        var t1 = now_ns()
        times.append(t1 - t0)
        _ = out
    
    var total = 0
    for t in times:
        total += t
    var ms = Float64(total) / 1e6 / Float64(n_iter)
    var gflops = 2.0 * Float64(M) * Float64(N) * Float64(K) / (ms * 1e6)
    
    print("  Result:", ms, "ms,", gflops, "GFLOPS")


def main():
    print("=" * 60)
    bench_q8k_single(1, 1024, 1024)
    print("=" * 60)
    bench_q8k_single(1, 4096, 4096)
    print("=" * 60)
    bench_q8k_single(1, 11008, 4096)