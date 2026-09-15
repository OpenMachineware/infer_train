# Benchmark: Q8_K + SDOT threaded matmul
#
# Compares:
# 1. Single-threaded Q8_K + SDOT
# 2. Threaded Q8_K + SDOT
# 3. FP32 SIMD threaded (current best)
#
# Uses properly formatted Q4_K blocks.

from src.core.tensor import Tensor, tensor_zeros
from src.core.thread_pool import now_ns, num_pcores
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.cpu.matmul_q8k_threaded import matmul_quantized_q8k_threaded
from src.core.ops.cpu.matmul_cpu import matmul_quantized_cpu_threaded
from src.core.ops.quantized.quant_types import QuantType
from std.memory.alloc import unsafe_alloc
from std.utils.static_tuple import StaticTuple


comptime QK_K = 256
comptime BB_Q4K = 144


def create_valid_q4_k_weights(N: Int, K: Int) -> Tensor[DType.uint8, 2]:
    """Create Q4_K weights with proper block format (valid layout, test values)."""
    var nb = K // QK_K
    var total_bytes = N * nb * BB_Q4K
    var data = unsafe_alloc[UInt8](total_bytes)
    
    for row in range(N):
        for block_idx in range(nb):
            var block = data.unsafe_offset(row * nb * BB_Q4K + block_idx * BB_Q4K)
            
            # d = 1.0, dmin = 0.0 (as fp16)
            block.unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
                val=Float16(1.0), offset=0
            )
            block.unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
                val=Float16(0.0), offset=1
            )
            
            # scales = 12 bytes, providing scale = 8 for each sub-block
            for i in range(12):
                block.unsafe_offset(4 + i).unsafe_store(val=UInt8(8))
            
            # qs = 128 bytes of 4-bit values (all 8s, packed as 0x88)
            for i in range(128):
                block.unsafe_offset(16 + i).unsafe_store(val=UInt8(0x88))
    
    var shape = StaticTuple[Int, 2](N, nb * BB_Q4K)
    return Tensor[DType.uint8, 2](shape, data)


def create_activation(M: Int, K: Int) -> Tensor[DType.float16, 2]:
    """Create activation tensor with test values."""
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.data().unsafe_offset(i).unsafe_store(
            val=Scalar[DType.float16](Float16(Float32((i % 100 + 1)) / 100.0))
        )
    return x


def bench_q8k_sdot_threaded(M: Int, N: Int, K: Int, n_warmup: Int = 3, n_iter: Int = 10):
    """Benchmark Q8_K + SDOT threaded matmul."""
    print("\n=== Q8_K + SDOT Threaded Benchmark ===")
    print("  M=", M, ", N=", N, ", K=", K)
    print("  Warmup:", n_warmup, ", Iterations:", n_iter)
    
    # Create input and weights
    var x = create_activation(M, K)
    var w_quant = create_valid_q4_k_weights(N, K)
    
    # Dummy scale (not used for Q4_K)
    var scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    
    # Warmup single-threaded Q8_K
    print("\nWarming up...")
    for _ in range(n_warmup):
        _ = matmul_quantized_q8k[QuantType.Q4_K_M](x, w_quant, scale)
    
    # Benchmark single-threaded Q8_K
    var times_q8k = List[Int]()
    for _ in range(n_iter):
        var t0 = now_ns()
        var out_q8k = matmul_quantized_q8k[QuantType.Q4_K_M](x, w_quant, scale)
        var t1 = now_ns()
        times_q8k.append(t1 - t0)
        _ = out_q8k
    
    var total_q8k = 0
    for t in times_q8k:
        total_q8k += t
    var ms_q8k = Float64(total_q8k) / 1e6 / Float64(n_iter)
    var gflops_q8k = 2.0 * Float64(M) * Float64(N) * Float64(K) / (ms_q8k * 1e6)
    
    print("Q8_K + SDOT (1 thread):", ms_q8k, "ms,", gflops_q8k, "GFLOPS")
    
    # Warmup threaded Q8_K
    for _ in range(n_warmup):
        _ = matmul_quantized_q8k_threaded[QuantType.Q4_K_M](x, w_quant, scale)
    
    # Benchmark threaded Q8_K
    var times_q8k_t = List[Int]()
    for _ in range(n_iter):
        var t0 = now_ns()
        var out_q8k_t = matmul_quantized_q8k_threaded[QuantType.Q4_K_M](x, w_quant, scale)
        var t1 = now_ns()
        times_q8k_t.append(t1 - t0)
        _ = out_q8k_t
    
    var total_q8k_t = 0
    for t in times_q8k_t:
        total_q8k_t += t
    var ms_q8k_t = Float64(total_q8k_t) / 1e6 / Float64(n_iter)
    var gflops_q8k_t = 2.0 * Float64(M) * Float64(N) * Float64(K) / (ms_q8k_t * 1e6)
    var speedup_q8k = ms_q8k / ms_q8k_t
    
    print("Q8_K + SDOT (", num_pcores(), " threads):", ms_q8k_t, "ms,", gflops_q8k_t, "GFLOPS (", speedup_q8k, "x speedup)")
    
    # Benchmark FP32 SIMD threaded (current production path)
    for _ in range(n_warmup):
        _ = matmul_quantized_cpu_threaded[DType.float16, QuantType.Q4_K_M, 32](x, w_quant, scale)
    
    var times_fp32_t = List[Int]()
    for _ in range(n_iter):
        var t0 = now_ns()
        var out_fp32_t = matmul_quantized_cpu_threaded[DType.float16, QuantType.Q4_K_M, 32](x, w_quant, scale)
        var t1 = now_ns()
        times_fp32_t.append(t1 - t0)
        _ = out_fp32_t
    
    var total_fp32_t = 0
    for t in times_fp32_t:
        total_fp32_t += t
    var ms_fp32_t = Float64(total_fp32_t) / 1e6 / Float64(n_iter)
    var gflops_fp32_t = 2.0 * Float64(M) * Float64(N) * Float64(K) / (ms_fp32_t * 1e6)
    
    print("FP32 SIMD (", num_pcores(), " threads):", ms_fp32_t, "ms,", gflops_fp32_t, "GFLOPS")
    
    print("\nSummary:")
    print("  Q8_K + SDOT threaded:", gflops_q8k_t, "GFLOPS")
    print("  FP32 SIMD threaded:  ", gflops_fp32_t, "GFLOPS")
    if gflops_q8k_t > gflops_fp32_t:
        print("  Q8_K + SDOT is", gflops_q8k_t / gflops_fp32_t, "x faster!")
    else:
        print("  FP32 SIMD is", gflops_fp32_t / gflops_q8k_t, "x faster")


def main():
    # Typical layer dimensions (7B model)
    # Q projection: [4096, 4096]
    # FFN up: [11008, 4096]
    
    print("=" * 60)
    bench_q8k_sdot_threaded(1, 4096, 4096)
    print("=" * 60)
    bench_q8k_sdot_threaded(1, 11008, 4096)
    print("=" * 60)
    # Small N case (below parallelization threshold)
    bench_q8k_sdot_threaded(1, 128, 4096)