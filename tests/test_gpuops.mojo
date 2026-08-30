# tests/test_gpu_ops.mojo
#
# Numeric tests for the Metal GPU kernels in src/core/ops/gpu.  Each test
# compares the GPU op against a hand-computed reference (the same values the
# CPU unit tests use).  On a machine without a Metal GPU the ops fall back to
# the CPU kernels, so the tests still pass (they validate the numeric
# contract, not the device).

from src.core.tensor import Tensor, tensor_zeros
from src.core.device import has_metal_gpu
from src.core.ops.gpu.swiglu_gpu import (
    swiglu_gpu_dynamic,
    swiglu_gpu_backward,
)
from src.core.ops.gpu.embedding_gpu import embedding_gpu_dynamic
from src.core.ops.gpu.rope_gpu import (
    rope_gpu_dynamic,
    rope_gpu_backward_pos,
)
from src.core.ops.gpu.softmax_gpu import (
    softmax_gpu_dynamic,
    softmax_gpu_backward,
)
from src.core.ops.gpu.rms_norm_gpu import (
    rms_norm_gpu_dynamic,
    rms_norm_gpu_backward,
)
from src.core.ops.gpu.matmul_gpu import (
    matmul_gpu_dynamic,
    matmul_gpu_backward,
)
from src.core.ops.gpu.add_gpu import (
    add_gpu_dynamic,
    add_row_gpu,
    add_gpu_backward,
)
from std.utils.static_tuple import StaticTuple
from std.math import exp, cos, sin, sqrt


def check_close(
    got: Float32, expected: Float32, tol: Float32, label: String
) raises:
    if abs(got - expected) > tol:
        print("FAIL", label, "got", got, "expected", expected)
        raise Error("check failed: " + label)


def test_swiglu() raises:
    comptime R = 2
    comptime C = 5
    var gate = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    var up = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    for i in range(R * C):
        gate.set(i, Scalar[DType.float32](Float32(i - 4)))
        up.set(i, Scalar[DType.float32](Float32(i * 2)))
    var out = swiglu_gpu_dynamic[DType.float32](gate, up)
    for i in range(R * C):
        var g = Float32(i - 4)
        var u = Float32(i * 2)
        var silu = g / (Float32(1.0) + exp(-g))
        check_close(Float32(out.get(i)), silu * u, Float32(1e-4), "swiglu")

    # backward
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    for i in range(R * C):
        go.set(i, Scalar[DType.float32](Float32(1.0)))
    var saved = List[Tensor[DType.float32, 2]]()
    saved.append(gate)
    saved.append(up)
    var grads = swiglu_gpu_backward[DType.float32, R, C](go, saved)
    for i in range(R * C):
        var g = Float32(i - 4)
        var u = Float32(i * 2)
        var silu = g / (Float32(1.0) + exp(-g))
        var sig = Float32(1.0) / (Float32(1.0) + exp(-g))
        var dsilu = sig * (Float32(1.0) + g - silu)
        check_close(
            Float32(grads[1].get(i)), silu, Float32(1e-4), "swiglu bwd up"
        )
        check_close(
            Float32(grads[0].get(i)), u * dsilu, Float32(1e-4), "swiglu bwd gate"
        )


def test_embedding() raises:
    # table [vocab=3, hidden=2]: row t is the embedding of token t
    comptime V = 3
    comptime H = 2
    var table = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](V, H))
    for i in range(6):
        table.set(i, Scalar[DType.float16](Float32(i + 1)))
    var tokens = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](3))
    tokens.set(0, Scalar[DType.int32](2))
    tokens.set(1, Scalar[DType.int32](0))
    tokens.set(2, Scalar[DType.int32](1))
    var out = embedding_gpu_dynamic[DType.float16](tokens, table)
    check_f16(Float32(out.get(0)), 5.0, "emb[0,0]")
    check_f16(Float32(out.get(1)), 6.0, "emb[0,1]")
    check_f16(Float32(out.get(2)), 1.0, "emb[1,0]")
    check_f16(Float32(out.get(3)), 2.0, "emb[1,1]")
    check_f16(Float32(out.get(4)), 3.0, "emb[2,0]")
    check_f16(Float32(out.get(5)), 4.0, "emb[2,1]")


def check_f16(got: Float32, expected: Float32, label: String) raises:
    if abs(got - expected) > Float32(0.02):
        print("FAIL", label, "got", got, "expected", expected)
        raise Error("check failed: " + label)


def test_rope() raises:
    # [1 head, 2 tokens, dim 4]: pairs (0,2) and (1,3), theta = 1 -> angle = pos
    var x = tensor_zeros[DType.float32, 3](StaticTuple[Int, 3](1, 2, 4))
    x.set(0, Scalar[DType.float32](Float32(1.0)))
    x.set(1, Scalar[DType.float32](Float32(2.0)))
    x.set(2, Scalar[DType.float32](Float32(3.0)))
    x.set(3, Scalar[DType.float32](Float32(4.0)))
    x.set(4, Scalar[DType.float32](Float32(5.0)))
    x.set(5, Scalar[DType.float32](Float32(6.0)))
    x.set(6, Scalar[DType.float32](Float32(7.0)))
    x.set(7, Scalar[DType.float32](Float32(8.0)))

    # start_pos = 1: token0 at pos1 (angle 1), token1 at pos2 (angle 2)
    var y = rope_gpu_dynamic[DType.float32](x, 1, Float32(1.0))
    var c1 = cos(Float32(1.0))
    var s1 = sin(Float32(1.0))
    check_close(Float32(y.get(0)), 1.0 * c1 - 3.0 * s1, Float32(1e-5), "rope t0 d0")
    check_close(Float32(y.get(2)), 1.0 * s1 + 3.0 * c1, Float32(1e-5), "rope t0 d2")
    var c2 = cos(Float32(2.0))
    var s2 = sin(Float32(2.0))
    check_close(Float32(y.get(4)), 5.0 * c2 - 7.0 * s2, Float32(1e-5), "rope t1 d0")
    check_close(Float32(y.get(6)), 5.0 * s2 + 7.0 * c2, Float32(1e-5), "rope t1 d2")

    # backward: inverse rotation
    var g = tensor_zeros[DType.float32, 3](StaticTuple[Int, 3](1, 2, 4))
    for i in range(8):
        g.set(i, Scalar[DType.float32](Float32(i + 1)))
    var gy = rope_gpu_backward_pos[DType.float32](g, 1, Float32(1.0))
    # grad_x0 = g0*c + g1*s ; for t0: g0=1,g1=3 -> 1*c1+3*s1
    check_close(Float32(gy.get(0)), 1.0 * c1 + 3.0 * s1, Float32(1e-5), "rope bwd t0 d0")
    check_close(Float32(gy.get(2)), -1.0 * s1 + 3.0 * c1, Float32(1e-5), "rope bwd t0 d2")


def test_softmax() raises:
    comptime R = 3
    comptime C = 6
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    for i in range(R * C):
        x.set(i, Scalar[DType.float32](Float32(i % 5)))
    var out = softmax_gpu_dynamic[DType.float32](x)
    # row sums to 1
    for r in range(R):
        var s = Float32(0.0)
        for c in range(C):
            s += Float32(out.get(r * C + c))
        check_close(s, Float32(1.0), Float32(1e-5), "softmax rowsum")
    # check one element: row0 = [0,1,2,3,4,0]; max=4
    var e0 = exp(Float32(0.0) - Float32(4.0))
    var e1 = exp(Float32(1.0) - Float32(4.0))
    var e2 = exp(Float32(2.0) - Float32(4.0))
    var e3 = exp(Float32(3.0) - Float32(4.0))
    var e4 = exp(Float32(4.0) - Float32(4.0))
    var total = e0 + e1 + e2 + e3 + e4 + e0
    check_close(Float32(out.get(0)), e0 / total, Float32(1e-5), "softmax[0,0]")
    check_close(Float32(out.get(3)), e3 / total, Float32(1e-5), "softmax[0,3]")

    # backward
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    for i in range(R * C):
        go.set(i, Scalar[DType.float32](Float32(1.0)))
    var saved = List[Tensor[DType.float32, 2]]()
    saved.append(out)
    var grads = softmax_gpu_backward[DType.float32, C](go, saved)
    # grad_x[i,j] = p[i,j]*(go[i,j] - sum_k go[i,k]*p[i,k]) = p[i,j]*(1 - 1) = 0
    # since sum_k p[i,k] = 1 and go is all ones.
    for i in range(R * C):
        check_close(Float32(grads[0].get(i)), Float32(0.0), Float32(1e-5), "softmax bwd")


def test_rms_norm() raises:
    comptime R = 2
    comptime C = 8
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    for i in range(R * C):
        x.set(i, Scalar[DType.float32](Float32(i + 1)))
    var out = rms_norm_gpu_dynamic[DType.float32](x)
    # row0 = [1..8]; ss = 204; rms = sqrt(204/8 + eps) ~ sqrt(25.5)
    var ss = Float32(0.0)
    for j in range(C):
        ss += Float32(j + 1) * Float32(j + 1)
    var rms = sqrt(ss / Float32(C) + Float32(1e-5))
    for j in range(C):
        check_close(
            Float32(out.get(j)), Float32(j + 1) / rms, Float32(1e-4), "rmsnorm"
        )

    # backward
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    for i in range(R * C):
        go.set(i, Scalar[DType.float32](Float32(1.0)))
    var saved = List[Tensor[DType.float32, 2]]()
    saved.append(x)
    saved.append(out)
    var grads = rms_norm_gpu_backward[DType.float32, C](go, saved)
    # grad_x[j] = (1 - out[j]*s/C)/rms where s = sum_j go[j]*out[j] = sum out[j]
    var s = Float32(0.0)
    for j in range(C):
        s += Float32(out.get(j))
    for j in range(C):
        var expected = (Float32(1.0) - Float32(out.get(j)) * s / Float32(C)) / rms
        check_close(
            Float32(grads[0].get(j)), expected, Float32(1e-4), "rmsnorm bwd"
        )


def test_matmul() raises:
    # a [2,3] @ b [3,2]
    comptime M = 2
    comptime K = 3
    comptime N = 2
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, K))
    var b = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](K, N))
    for i in range(M * K):
        a.set(i, Scalar[DType.float32](Float32(i + 1)))
    for i in range(K * N):
        b.set(i, Scalar[DType.float32](Float32(i + 1)))
    var out = matmul_gpu_dynamic[DType.float32](a, b)
    # a = [[1,2,3],[4,5,6]], b = [[1,2],[3,4],[5,6]]
    # out[0,0] = 1*1+2*3+3*5 = 22 ; out[0,1] = 1*2+2*4+3*6 = 28
    # out[1,0] = 4*1+5*3+6*5 = 49 ; out[1,1] = 4*2+5*4+6*6 = 64
    check_close(Float32(out.get(0)), 22.0, Float32(1e-4), "mm[0,0]")
    check_close(Float32(out.get(1)), 28.0, Float32(1e-4), "mm[0,1]")
    check_close(Float32(out.get(2)), 49.0, Float32(1e-4), "mm[1,0]")
    check_close(Float32(out.get(3)), 64.0, Float32(1e-4), "mm[1,1]")

    # backward
    var go = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, N))
    for i in range(M * N):
        go.set(i, Scalar[DType.float32](Float32(1.0)))
    var saved = List[Tensor[DType.float32, 2]]()
    saved.append(a)
    saved.append(b)
    var grads = matmul_gpu_backward[DType.float32](go, saved)
    # grad_a[i,k] = sum_j go[i,j]*b[k,j] (go all ones):
    # grad_a[0,0] = b[0,0]+b[0,1] = 1+2 = 3
    # grad_a[0,1] = b[1,0]+b[1,1] = 3+4 = 7
    # grad_a[0,2] = b[2,0]+b[2,1] = 5+6 = 11
    check_close(Float32(grads[0].get(0)), 3.0, Float32(1e-4), "mm bwd a[0,0]")
    check_close(Float32(grads[0].get(1)), 7.0, Float32(1e-4), "mm bwd a[0,1]")
    check_close(Float32(grads[0].get(2)), 11.0, Float32(1e-4), "mm bwd a[0,2]")
    # grad_b[k,j] = sum_i a[i,k]*go[i,j]; grad_b is [K,N]=[3,2], index k*2+j
    # grad_b[0,0] = 1*1+4*1 = 5 ; grad_b[1,0] = 2*1+5*1 = 7 ; grad_b[2,0] = 3*1+6*1 = 9
    check_close(Float32(grads[1].get(0)), 5.0, Float32(1e-4), "mm bwd b[0,0]")
    check_close(Float32(grads[1].get(2)), 7.0, Float32(1e-4), "mm bwd b[1,0]")
    check_close(Float32(grads[1].get(4)), 9.0, Float32(1e-4), "mm bwd b[2,0]")


def test_add() raises:
    comptime R = 3
    comptime C = 4
    var a = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](R, C))
    var b = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](R, C))
    for i in range(R * C):
        a.set(i, Scalar[DType.float16](Float32(i)))
        b.set(i, Scalar[DType.float16](Float32(10 - i)))
    var out = add_gpu_dynamic[DType.float16](a, b)
    for i in range(R * C):
        check_f16(Float32(out.get(i)), 10.0, "add f16")

    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    var bias = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](C))
    for i in range(R * C):
        x.set(i, Scalar[DType.float32](Float32(i)))
    for j in range(C):
        bias.set(j, Scalar[DType.float32](Float32(100 + j)))
    var outr = add_row_gpu[DType.float32](x, bias)
    for i in range(R):
        for j in range(C):
            check_close(
                Float32(outr.get(i * C + j)),
                Float32(i * C + j) + Float32(100 + j),
                Float32(1e-5),
                "add row",
            )

    var g = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](R, C))
    for i in range(R * C):
        g.set(i, Scalar[DType.float32](Float32(i + 1)))
    var saved = List[Tensor[DType.float32, 2]]()
    saved.append(x)
    saved.append(x)
    var grads = add_gpu_backward[DType.float32, R, C](g, saved)
    for i in range(R * C):
        check_close(Float32(grads[0].get(i)), Float32(i + 1), Float32(1e-5), "add bwd a")
        check_close(Float32(grads[1].get(i)), Float32(i + 1), Float32(1e-5), "add bwd b")


def main() raises:
    print("metal:", has_metal_gpu())
    test_add()
    test_swiglu()
    test_embedding()
    test_rope()
    test_softmax()
    test_rms_norm()
    test_matmul()
    print("test_gpu_ops OK")
