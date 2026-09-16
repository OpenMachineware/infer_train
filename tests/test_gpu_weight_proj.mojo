# Test GPU weight-major matmul path

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.qweight import QWeight, qweight_from_fp16
from src.core.ops.gpu.matmul_gpu import matmul_weight_gpu
from src.core.ops.gpu.rms_norm_gpu import rms_norm_weight_gpu
from std.utils.static_tuple import StaticTuple
from std.math import sqrt


def test_matmul_weight_gpu() -> Bool:
    """Test weight-major matmul: y = x @ w^T where w is [N, K]."""
    var M = 2
    var K = 64
    var N = 32

    # Create input x [M, K]
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i) * 0.01))

    # Create weight w [N, K] (weight-major)
    var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    for i in range(N * K):
        w.set(i, Scalar[DType.float16](Float32(i) * 0.01))

    # GPU matmul
    var y_gpu = matmul_weight_gpu[DType.float16](x, w)

    # CPU reference: y = x @ w^T
    var y_cpu = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    for i in range(M):
        for j in range(N):
            var acc = Float32(0.0)
            for k in range(K):
                acc += Float32(x.get(i * K + k)) * Float32(w.get(j * K + k))
            y_cpu.set(i * N + j, Scalar[DType.float16](acc))

    # Compare
    var max_diff = Float32(0.0)
    for i in range(M * N):
        var diff = abs(Float32(y_gpu.get(i)) - Float32(y_cpu.get(i)))
        if diff > max_diff:
            max_diff = diff

    print("matmul_weight_gpu: M=", M, " K=", K, " N=", N, " max_diff=", max_diff)
    if max_diff > 1e-3:
        print("FAIL: max_diff too large")
        return False
    return True


def test_qweight_gpu() -> Bool:
    """Test QWeight.proj() with FP16 weights uses GPU."""
    var M = 2
    var K = 64
    var N = 32

    # Create input x [M, K]
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i) * 0.01))

    # Create weight w [N, K] and wrap as QWeight
    var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    for i in range(N * K):
        w.set(i, Scalar[DType.float16](Float32(i) * 0.01))

    var qw = qweight_from_fp16(w)

    # Use QWeight.proj with GPU (use_gpu=True by default)
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var y_gpu = qw.proj(x, dummy_scale, use_gpu=True)

    # CPU reference with use_gpu=False
    var y_cpu = qw.proj(x, dummy_scale, use_gpu=False)

    # Compare
    var max_diff = Float32(0.0)
    for i in range(M * N):
        var diff = abs(Float32(y_gpu.get(i)) - Float32(y_cpu.get(i)))
        if diff > max_diff:
            max_diff = diff

    print("QWeight.proj (GPU vs CPU): max_diff=", max_diff)
    if max_diff > 1e-3:
        print("FAIL: max_diff too large")
        return False
    return True


def test_rms_norm_weight_gpu() -> Bool:
    """Test RMS norm with weight on GPU."""
    var rows = 2
    var dim = 64

    # Create input x [rows, dim]
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](rows, dim))
    for i in range(rows * dim):
        x.set(i, Scalar[DType.float16](Float32(i) * 0.01))

    # Create weight w [dim]
    var w = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](dim))
    for i in range(dim):
        w.set(i, Scalar[DType.float16](Float32(i) * 0.01))

    # GPU
    var eps = Float32(1e-5)
    var y_gpu = rms_norm_weight_gpu[DType.float16](x, w, eps)

    # CPU reference
    from src.core.ops.cpu.rms_norm_cpu import rms_norm_weight_cpu
    var y_cpu = rms_norm_weight_cpu[DType.float16](x, w, eps)

    # Compare
    var max_diff = Float32(0.0)
    for i in range(rows * dim):
        var diff = abs(Float32(y_gpu.get(i)) - Float32(y_cpu.get(i)))
        if diff > max_diff:
            max_diff = diff

    print("rms_norm_weight_gpu: max_diff=", max_diff)
    if max_diff > 1e-3:
        print("FAIL: max_diff too large")
        return False
    return True


def main():
    var all_passed = True

    if not test_matmul_weight_gpu():
        all_passed = False

    if not test_qweight_gpu():
        all_passed = False

    if not test_rms_norm_weight_gpu():
        all_passed = False

    if all_passed:
        print("test_gpu_weight_proj OK")
    else:
        print("test_gpu_weight_proj FAILED")
