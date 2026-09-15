# Benchmark single Q4_K matmul
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.quant_types import QuantType, block_bytes, block_elems
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.cpu.matmul_q8k_threaded import matmul_quantized_q8k_threaded
from src.core.thread_pool import now_ns
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

comptime QK_K = 256


def create_q4_k_block() -> Pointer[UInt8, MutUntrackedOrigin]:
    """Create a synthetic Q4_K block for benchmarking."""
    var bb = block_bytes(QuantType.Q4_K_M)  # 144 bytes
    var block = unsafe_alloc[UInt8](bb)
    
    # d = 1.0 (fp16 = 0x3C00)
    block.unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(val=Scalar[DType.float16](1.0), offset=0)
    # dmin = 0.0
    block.unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(val=Scalar[DType.float16](0.0), offset=1)
    
    # scales: 12 bytes
    var scales = block.unsafe_offset(4)
    for i in range(12):
        scales.unsafe_store(val=Scalar[DType.uint8](32), offset=i)  # scale = 32
    
    # qs: 128 bytes, fill with 0x88 pattern
    var qs = block.unsafe_offset(16)
    for i in range(128):
        qs.unsafe_store(val=Scalar[DType.uint8](0x88), offset=i)
    
    return block


def main() raises:
    # Matrix dimensions (7B model typical sizes)
    var M = 1  # batch size (decode)
    var K = 4096  # hidden dim
    var N = 14336  # FFN output dim (larger for more accurate timing)
    
    var nb = K // QK_K  # number of blocks
    var bb = block_bytes(QuantType.Q4_K_M)
    
    print("Matrix size:", M, "x", K, "x", N)
    print("Blocks:", nb, "Block bytes:", bb)
    print("Weight tensor size:", N * nb * bb, "bytes")
    
    # Create weight tensor (N rows, each row has nb blocks of bb bytes)
    var w_quant = tensor_zeros[DType.uint8, 2](StaticTuple[Int, 2](N, nb * bb))
    
    # Fill with synthetic Q4_K blocks
    var block_template = create_q4_k_block()
    for n in range(N):
        for b in range(nb):
            for i in range(bb):
                w_quant.data().unsafe_offset(n * nb * bb + b * bb + i).unsafe_store(
                    val=block_template.unsafe_load[width=1](offset=i)
                )
    
    # Create activation tensor
    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.data().unsafe_offset(i).unsafe_store(val=Scalar[DType.float16](1.0))
    
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    
    # Warmup
    var warmup = matmul_quantized_q8k[QuantType.Q4_K_M](x, w_quant, dummy_scale)
    print("Warmup done")
    
    # Benchmark single-threaded
    var runs = 10
    var total_ms = Int64(0)
    for _ in range(runs):
        var start = now_ns()
        var result = matmul_quantized_q8k[QuantType.Q4_K_M](x, w_quant, dummy_scale)
        var end = now_ns()
        total_ms += Int64((end - start) // 1_000_000)
    var avg_ms = total_ms / Int64(runs)
    print("Single-threaded avg:", avg_ms, "ms")
    
    # Calculate GFLOPS
    # Each matmul: M * K * N operations
    # For K=4096, N=4096, M=1: 16.7M FLOPs
    var flops = 2 * M * K * N  # multiply-add counts as 2 FLOPs
    var gflops = Float64(flops) / 1e9
    var gflops_per_sec = gflops * 1000.0 / Float64(avg_ms)
    print("GFLOPS:", gflops_per_sec)
    
    # Benchmark threaded (if available)
    var total_ms_th = Int64(0)
    for _ in range(runs):
        var start = now_ns()
        var result = matmul_quantized_q8k_threaded[QuantType.Q4_K_M](x, w_quant, dummy_scale, 4)
        var end = now_ns()
        total_ms_th += Int64((end - start) // 1_000_000)
    var avg_ms_th = total_ms_th / Int64(runs)
    print("Threaded (4 threads) avg:", avg_ms_th, "ms")
    var gflops_per_sec_th = gflops * 1000.0 / Float64(avg_ms_th)
    print("GFLOPS (threaded):", gflops_per_sec_th)
    
    print("\n=== Performance Comparison ===")
    print("Single-threaded:", gflops_per_sec, "GFLOPS")
    print("Threaded:", gflops_per_sec_th, "GFLOPS")
    print("Speedup:", gflops_per_sec_th / gflops_per_sec, "x")
    
    block_template.unsafe_free()