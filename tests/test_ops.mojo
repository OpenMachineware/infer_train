# tests/test_ops.mojo
#
# Numeric unit tests for the M3 operators: embedding, rope (NEOX), add,
# SwiGLU, and multi-head attention (causal), plus an erased-interface check
# through the OpRegistry.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.embedding_cpu import embedding_cpu_dynamic
from src.core.ops.cpu.rope_cpu import rope_cpu_dynamic
from src.core.ops.cpu.add_cpu import add_cpu_dynamic, add_row_cpu
from src.core.ops.cpu.swiglu_cpu import swiglu_cpu_dynamic
from src.core.ops.attention.mha import multi_head_attention
from src.core.ops.base.op_registry import OpRegistry
from src.core.ops.base.op_interface import to_any, AnyTensor
from std.utils.static_tuple import StaticTuple
from std.math import cos, sin, exp, sqrt


def main():
    test_embedding()
    test_rope()
    test_add()
    test_swiglu()
    test_mha()
    test_registry()
    print("test_ops OK")


def test_embedding():
    # table [vocab=3, hidden=2]: row t is the embedding of token t
    # (GGUF token_embd is vocab-major on disk)
    var table = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](3, 2))
    for i in range(6):
        table.set(i, Scalar[DType.float16](Float32(i + 1)))
    var tokens = tensor_zeros[DType.int32, 1](StaticTuple[Int, 1](3))
    tokens.set(0, Scalar[DType.int32](2))
    tokens.set(1, Scalar[DType.int32](0))
    tokens.set(2, Scalar[DType.int32](1))
    var out = embedding_cpu_dynamic[DType.float16](tokens, table)
    # out[0] = row 2 = [5, 6]; out[1] = [1, 2]; out[2] = [3, 4]
    check_f16(Float32(out.get(0)), 5.0, "emb[0,0]")
    check_f16(Float32(out.get(1)), 6.0, "emb[0,1]")
    check_f16(Float32(out.get(2)), 1.0, "emb[1,0]")
    check_f16(Float32(out.get(3)), 2.0, "emb[1,1]")
    check_f16(Float32(out.get(4)), 3.0, "emb[2,0]")
    check_f16(Float32(out.get(5)), 4.0, "emb[2,1]")


def test_rope():
    # [1 head, 1 token, dim 4]: pairs (0,2) and (1,3), theta = 1 -> angle = pos
    var x = tensor_zeros[DType.float32, 3](StaticTuple[Int, 3](1, 1, 4))
    x.set(0, Scalar[DType.float32](Float32(1.0)))
    x.set(1, Scalar[DType.float32](Float32(2.0)))
    x.set(2, Scalar[DType.float32](Float32(3.0)))
    x.set(3, Scalar[DType.float32](Float32(4.0)))

    # pos 0: identity
    var y0 = rope_cpu_dynamic[DType.float32](x, 0, Float32(1.0))
    check_f32(Float32(y0.get(0)), 1.0, "rope pos0 d0")
    check_f32(Float32(y0.get(3)), 4.0, "rope pos0 d3")

    # pos 1, angle 1 rad
    var y1 = rope_cpu_dynamic[DType.float32](x, 1, Float32(1.0))
    var c = cos(Float32(1.0))
    var s = sin(Float32(1.0))
    check_f32(Float32(y1.get(0)), 1.0 * c - 3.0 * s, "rope pos1 d0")
    check_f32(Float32(y1.get(2)), 1.0 * s + 3.0 * c, "rope pos1 d2")
    check_f32(Float32(y1.get(1)), 2.0 * c - 4.0 * s, "rope pos1 d1")
    check_f32(Float32(y1.get(3)), 2.0 * s + 4.0 * c, "rope pos1 d3")


def test_add():
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    var b = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 3))
    for i in range(6):
        a.set(i, Scalar[DType.float32](Float32(i)))
        b.set(i, Scalar[DType.float32](Float32(10 - i)))
    var out = add_cpu_dynamic[DType.float32](a, b)
    for i in range(6):
        check_f32(Float32(out.get(i)), 10.0, "add[" + String(i) + "]")

    # row-broadcast bias
    var bias = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](3))
    bias.set(0, Scalar[DType.float32](Float32(1)))
    bias.set(1, Scalar[DType.float32](Float32(2)))
    bias.set(2, Scalar[DType.float32](Float32(3)))
    var biased = add_row_cpu[DType.float32](a, bias)
    check_f32(Float32(biased.get(0)), 1.0, "bias[0,0]")
    check_f32(Float32(biased.get(1)), 3.0, "bias[0,1]")
    check_f32(Float32(biased.get(3)), 4.0, "bias[1,0]")


def test_swiglu():
    var gate = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 5))
    var up = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, 5))
    var gvals = List[Float32]()
    for v in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        gvals.append(Float32(v))
    for i in range(5):
        gate.set(i, Scalar[DType.float32](gvals[i]))
        up.set(i, Scalar[DType.float32](Float32(i + 1)))
    var out = swiglu_cpu_dynamic[DType.float32](gate, up)
    for i in range(5):
        var g = gvals[i]
        var silu = g / (Float32(1.0) + exp(-g))
        var expect = silu * Float32(i + 1)
        check_f32(Float32(out.get(i)), expect, "swiglu[" + String(i) + "]")


def test_mha():
    # 1 head, T=2, dim=2. Query row 0 attends only position 0 (causal).
    var q = tensor_zeros[DType.float32, 3](StaticTuple[Int, 3](1, 2, 2))
    var k = tensor_zeros[DType.float32, 3](StaticTuple[Int, 3](1, 2, 2))
    var v = tensor_zeros[DType.float32, 3](StaticTuple[Int, 3](1, 2, 2))
    # q[0] = [1,0], q[1] = [0,1]
    q.set(0, Scalar[DType.float32](Float32(1)))
    q.set(3, Scalar[DType.float32](Float32(1)))
    # k[0] = [1,0], k[1] = [0,1]  -> scores only at matching positions
    k.set(0, Scalar[DType.float32](Float32(1)))
    k.set(3, Scalar[DType.float32](Float32(1)))
    # v[0] = [5,6], v[1] = [7,8]
    v.set(0, Scalar[DType.float32](Float32(5)))
    v.set(1, Scalar[DType.float32](Float32(6)))
    v.set(2, Scalar[DType.float32](Float32(7)))
    v.set(3, Scalar[DType.float32](Float32(8)))

    var out = multi_head_attention[DType.float32, 1, 2](q, k, v)
    # row 0 attends only pos 0 -> [5, 6]
    check_f32(Float32(out.get(0)), 5.0, "mha[0,0]")
    check_f32(Float32(out.get(1)), 6.0, "mha[0,1]")
    # row 1: score(1,0)=0, score(1,1)=1 -> softmax over [0, 1/sqrt(2)]
    var e0 = exp(Float32(0.0))
    var e1 = exp(Float32(1.0) / sqrt(Float32(2.0)))
    var p0 = e0 / (e0 + e1)
    var p1 = e1 / (e0 + e1)
    check_f32(Float32(out.get(2)), p0 * 5.0 + p1 * 7.0, "mha[1,0]")
    check_f32(Float32(out.get(3)), p0 * 6.0 + p1 * 8.0, "mha[1,1]")


def test_registry():
    var registry = OpRegistry()
    registry.register_default_ops()
    # every M3 op must be present and dispatchable
    var names = List[String]()
    for n in ["embedding", "rope", "add", "swiglu", "mha", "lm_head", "swiglu_ffn"]:
        names.append(n)
    for name in names:
        var op = registry.get(name, None)
        print("  registry:", name, "ok")

    # erased round-trip: add two small f16 tensors through the registry
    var a = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, 2))
    var b = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, 2))
    a.set(0, Scalar[DType.float16](Float32(1)))
    a.set(1, Scalar[DType.float16](Float32(2)))
    b.set(0, Scalar[DType.float16](Float32(3)))
    b.set(1, Scalar[DType.float16](Float32(4)))
    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float16, 2](a))
    inputs.append(to_any[DType.float16, 2](b))
    var op = registry.get("add", None)
    var results = op.forward(inputs)
    check_f16(Float32(results[0].data.unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)), 4.0, "registry add[0]")
    check_f16(Float32(results[0].data.unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=1)), 6.0, "registry add[1]")


def check_f32(actual: Float32, expected: Float32, label: String):
    var diff = actual - expected
    if diff < 0:
        diff = -diff
    if diff > Float32(1e-4):
        print("FAIL", label, "actual:", actual, "expected:", expected)
        abort()


def check_f16(actual: Float32, expected: Float32, label: String):
    var diff = actual - expected
    if diff < 0:
        diff = -diff
    if diff > Float32(1e-2):
        print("FAIL", label, "actual:", actual, "expected:", expected)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
