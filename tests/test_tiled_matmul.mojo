# Test tiled GPU matmul

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.gpu.matmul_tiled_gpu import matmul_weight_tiled_gpu
from src.core.ops.gpu.matmul_gpu import matmul_weight_gpu
from std.utils.static_tuple import StaticTuple


def test_tiled_vs_naive() -> Bool:
    """Test tiled matmul matches naive kernel."""
    var M = 4
    var K = 128
    var N = 4

    # Create input x [M, K]
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i) * 0.01))

    # Create weight w [N, K] (weight-major)
    var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    for i in range(N * K):
        w.set(i, Scalar[DType.float16](Float32(i) * 0.01))

    # GPU tiled matmul
    var y_tiled = matmul_weight_tiled_gpu[DType.float16](x, w)

    # GPU naive matmul
    var y_naive = matmul_weight_gpu[DType.float16](x, w)

    # Compare
    var max_diff = Float32(0.0)
    for i in range(M * N):
        var diff = abs(Float32(y_tiled.get(i)) - Float32(y_naive.get(i)))
        if diff > max_diff:
            max_diff = diff

    print("tiled vs naive: M=", M, " K=", K, " N=", N, " max_diff=", max_diff)
    if max_diff > 1e-2:
        print("FAIL: max_diff too large")
        return False
    return True


def main():
    var all_passed = True

    if not test_tiled_vs_naive():
        all_passed = False

    if all_passed:
        print("test_tiled_matmul OK")
    else:
        print("test_tiled_matmul FAILED")
