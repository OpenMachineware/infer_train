# tests/test_gpu_pipeline.mojo
#
# Correctness + performance test for the async GPU/CPU pipeline
# (src/core/ops/gpu/gpu_pipeline.mojo).
#
#   * Correctness: a chain of linear layers run through the pipeline must
#     match the same chain run through the synchronous per-op GPU entry point.
#   * Performance: the pipeline keeps weights on the device (uploaded once)
#     and keeps activations on the device across layers (no per-op DtoH+HtoD,
#     no per-op sync), so it should beat the synchronous per-op version, which
#     re-uploads the weights and syncs after every layer.
#
# Everything runs in main() because `List[Tensor]` is movable but not
# implicitly copyable in Mojo 1.0, so the weight lists cannot be passed or
# returned between helper functions.
#
# On a machine without a Metal GPU the pipeline is unavailable and the test
# reports SKIP.

from src.core.tensor import Tensor, tensor_zeros
from src.core.device import has_metal_gpu
from src.core.ops.gpu.gpu_pipeline import GpuPipeline
from src.core.ops.gpu.fused_gpu import fused_matmul_add_bias_gpu
from max.gpu.host import DeviceBuffer
from std.utils.static_tuple import StaticTuple
from std.time import monotonic

comptime M = 1
comptime K = 1536
comptime N = 1536
comptime L = 8


def main() raises:
    if not has_metal_gpu():
        print("SKIP: no Metal GPU")
        return

    # -- build weights + input -------------------------------------------
    var ws = List[Tensor[DType.float16, 2]]()
    var bs = List[Tensor[DType.float16, 1]]()
    for i in range(L):
        var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
        var b = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](N))
        for j in range(N * K):
            w.set(j, Scalar[DType.float16](Float32((j + i) % 7) * 0.01))
        for j in range(N):
            b.set(j, Scalar[DType.float16](Float32((j + i) % 5) * 0.01))
        ws.append(w)
        bs.append(b)
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for j in range(M * K):
        x.set(j, Scalar[DType.float16](Float32(j % 11) * 0.1))

    # -- correctness: pipeline must match the synchronous per-op path ----
    var cur = x
    for i in range(L):
        cur = fused_matmul_add_bias_gpu[DType.float16](cur, ws[i], bs[i])
    var sync_out = cur

    var pipe = GpuPipeline()
    var wbufs = List[DeviceBuffer[DType.float16]]()
    var bbufs = List[DeviceBuffer[DType.float16]]()
    for i in range(L):
        wbufs.append(pipe.to_device2[DType.float16](ws[i]))
        bbufs.append(pipe.to_device1[DType.float16](bs[i]))
    var cur2 = pipe.to_device2[DType.float16](x)
    for i in range(L):
        cur2 = pipe.linear[DType.float16](cur2, wbufs[i], bbufs[i], M, K, N)
    var pipe_out = pipe.to_host2[DType.float16](cur2, StaticTuple[Int, 2](M, N))

    var max_diff = Float32(0.0)
    for i in range(M * N):
        var d = abs(Float32(sync_out.get(i)) - Float32(pipe_out.get(i)))
        if d > max_diff:
            max_diff = d
    if max_diff > Float32(0.05):
        print("FAIL pipeline correctness, max_diff =", max_diff)
        raise Error("pipeline correctness failed")
    print("pipeline correctness OK (max_diff =", max_diff, ")")

    # -- performance ------------------------------------------------------
    comptime ITERS = 20
    # warmup (context creation + kernel compile + allocator warmup)
    var c = x
    for i in range(L):
        c = fused_matmul_add_bias_gpu[DType.float16](c, ws[i], bs[i])
    _ = c
    pipe.sync()

    # synchronous per-op: re-uploads the weights and syncs after every layer
    var t0 = monotonic()
    for _ in range(ITERS):
        var cc = x
        for i in range(L):
            cc = fused_matmul_add_bias_gpu[DType.float16](cc, ws[i], bs[i])
        _ = cc
    var t1 = monotonic()
    var sync_ns = (Int(t1) - Int(t0)) // ITERS

    # pipeline: weights already on the device, one sync per pass
    var t2 = monotonic()
    for _ in range(ITERS):
        var cc2 = pipe.to_device2[DType.float16](x)
        for i in range(L):
            cc2 = pipe.linear[DType.float16](cc2, wbufs[i], bbufs[i], M, K, N)
        _ = pipe.to_host2[DType.float16](cc2, StaticTuple[Int, 2](M, N))
    var t3 = monotonic()
    var pipe_ns = (Int(t3) - Int(t2)) // ITERS

    print("sync per-op   :", sync_ns / 1_000_000, "ms/pass")
    print("gpu pipeline  :", pipe_ns / 1_000_000, "ms/pass")
    print("speedup       :", Float64(Int(sync_ns)) / Float64(Int(pipe_ns)), "x")
    print("test_gpu_pipeline OK")
