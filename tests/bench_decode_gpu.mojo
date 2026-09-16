# Benchmark decode-optimized GPU matmul
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.quant_types import QuantType, block_bytes
from src.core.ops.quantized.qweight import QWeight
from src.core.thread_pool import now_ns
from src.core.ops.gpu.gpu_runtime import get_gpu_context
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.collections.optional import Optional
from max.gpu.host import DeviceContext

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
    var N = 14336  # FFN output dim

    var nb = K // QK_K  # number of blocks
    var bb = block_bytes(QuantType.Q4_K_M)

    print("=" * 60)
    print("Decode GPU Matmul Benchmark")
    print("=" * 60)
    print("Matrix size:", M, "x", K, "x", N)
    print("Total FLOPs:", Float64(2 * M * K * N) / 1e6, "M")

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

    # Create QWeight wrapper
    var w = QWeight()
    w.data = w_quant
    w.ggml_type = 12  # Q4_K
    w.quantized = True
    w.n_out = N
    w.n_in = K

    # Pre-upload weights to GPU (simulating model initialization)
    print("\nPre-uploading weights to GPU...")
    var gpu_ctx: Optional[DeviceContext] = None
    try:
        var ctx = get_gpu_context()
        gpu_ctx = Optional[DeviceContext](ctx)
        w.upload_to_gpu(ctx)
        print("Weights uploaded and cached")
    except:
        print("GPU not available, falling back to CPU-only benchmark")

    # Create activation tensor
    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.data().unsafe_offset(i).unsafe_store(val=Scalar[DType.float16](1.0))

    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = w.proj(x, dummy_scale, use_gpu=False)
        _ = w.proj(x, dummy_scale, use_gpu=True, gpu_ctx=gpu_ctx)

    # Benchmark CPU path
    print("\n=== CPU Path (baseline) ===")
    var runs = 10
    var total_ms_cpu = Int64(0)
    for _ in range(runs):
        var start = now_ns()
        _ = w.proj(x, dummy_scale, use_gpu=False)
        var end = now_ns()
        total_ms_cpu += Int64((end - start) // 1_000_000)
    var avg_ms_cpu = total_ms_cpu / Int64(runs)
    var gflops_cpu = Float64(2 * M * K * N) / 1e9 * 1000.0 / Float64(avg_ms_cpu)
    print("Average time:", avg_ms_cpu, "ms")
    print("Throughput:", gflops_cpu, "GFLOPS")

    # Benchmark GPU path
    print("\n=== GPU Path (decode-optimized with caching) ===")
    var total_ms_gpu = Int64(0)
    var gpu_ok = True
    for _ in range(runs):
        var start = now_ns()
        try:
            _ = w.proj(x, dummy_scale, use_gpu=True, gpu_ctx=gpu_ctx)
            var end = now_ns()
            total_ms_gpu += Int64((end - start) // 1_000_000)
        except:
            gpu_ok = False
            break

    if gpu_ok and total_ms_gpu > 0:
        var avg_ms_gpu = total_ms_gpu / Int64(runs)
        var gflops_gpu = Float64(2 * M * K * N) / 1e9 * 1000.0 / Float64(avg_ms_gpu)
        print("Average time:", avg_ms_gpu, "ms")
        print("Throughput:", gflops_gpu, "GFLOPS")

        # Speedup
        var speedup = Float64(avg_ms_cpu) / Float64(avg_ms_gpu)
        print("\n=== Comparison ===")
        print("Speedup:", speedup, "x")
        if speedup > 1.0:
            print("GPU is", speedup, "x faster than CPU")
        else:
            print("GPU is", 1.0/speedup, "x slower than CPU")
    else:
        print("GPU not available or error occurred")

    print("\n" + "=" * 60)

    block_template.unsafe_free()
