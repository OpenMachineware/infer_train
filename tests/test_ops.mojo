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
from src.core.ops.cpu.matmul_cpu import matmul_quantized_cpu
from src.core.ops.attention.mha import multi_head_attention
from src.core.ops.base.op_registry import OpRegistry
from src.core.ops.base.op_interface import to_any, AnyTensor
from src.core.ops.quantized.quant_types import (
    QuantType,
    ggml_type,
    num_bits,
    group_size,
    block_bytes,
    block_elems,
)
from std.utils.static_tuple import StaticTuple
from std.math import cos, sin, exp, sqrt
from std.memory.unsafe import bitcast


def main():
    test_embedding()
    test_rope()
    test_add()
    test_swiglu()
    test_mha()
    test_registry()
    test_matmul_quantized()
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


# -- matmul_quantized (M7: comptime-specialized GGUF block formats) ----------
#
# The reference values are derived from llama.cpp's `ggml-quants.c`
# dequantization formulas with hand-crafted quantized blocks:
#   * Q8_0:  value = d * qs[i]                    (34-byte blocks)
#   * Q4_K:  value = d * sc * q - dmin * m        (144-byte super-blocks)
# with scales chosen so that sc = 1 and m = 0 for every sub-block.  All
# test values are small integers or halves, so the FP32 reference sums are
# exact (no rounding) and the comparison is bit-exact.  The dequantizers
# themselves are validated against gguf-py-generated bins in
# `tests/test_dequant_m7.mojo`.


def write_f16(buf: Tensor[DType.uint8, 2], idx: Int, value: Float32):
    """Store an FP16 scalar as two little-endian bytes."""
    var u16 = bitcast[DType.uint16](Scalar[DType.float16](value))
    buf.set(idx, Scalar[DType.uint8](UInt8(u16 & 0xFF)))
    buf.set(idx + 1, Scalar[DType.uint8](UInt8((u16 >> 8) & 0xFF)))


def build_q4_k_row(buf: Tensor[DType.uint8, 2], base: Int, d: Float32):
    """One Q4_K super-block (144 bytes, 256 elements) with deq[k] = d * (k % 16).

    dmin = 0 and the 12 scale bytes are [1,1,1,1,0,0,0,0,1,1,1,1], which
    unpacks to (sc, m) = (1, 0) for all 8 sub-blocks per
    `get_scale_min_k4` in ggml-quants.c.
    """
    write_f16(buf, base, d)
    write_f16(buf, base + 2, Float32(0.0))
    var scales = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    for j in range(12):
        buf.set(base + 4 + j, Scalar[DType.uint8](UInt8(scales[j])))
    for p in range(4):
        for l in range(32):
            var lo = UInt8((64 * p + l) % 16)
            var hi = UInt8((64 * p + 32 + l) % 16)
            buf.set(
                base + 16 + p * 32 + l,
                Scalar[DType.uint8](lo | (hi << 4)),
            )


def build_q8_0_row(
    buf: Tensor[DType.uint8, 2], base: Int, d: Float32, descending: Bool
):
    """One Q8_0 block (34 bytes, 32 elements): qs = k-16 or 15-k."""
    write_f16(buf, base, d)
    for k in range(32):
        var q = 15 - k if descending else k - 16
        buf.set(base + 2 + k, Scalar[DType.uint8](UInt8(q & 0xFF)))


def test_quant_type_helpers():
    if num_bits(QuantType.Q4_K_M) != 4:
        print("FAIL: num_bits Q4_K_M")
        abort()
    if num_bits(QuantType.Q8_0) != 8:
        print("FAIL: num_bits Q8_0")
        abort()
    if num_bits(QuantType.Q6_K) != 6:
        print("FAIL: num_bits Q6_K")
        abort()
    if num_bits(QuantType.Q5_K) != 5:
        print("FAIL: num_bits Q5_K")
        abort()
    if num_bits(QuantType.Q2_K) != 2:
        print("FAIL: num_bits Q2_K")
        abort()
    if num_bits(QuantType.IQ4_XS) != 4:
        print("FAIL: num_bits IQ4_XS")
        abort()
    # every GGUF K format quantizes in 32-element sub-blocks
    for qt in [
        QuantType.Q4_K_M,
        QuantType.Q8_0,
        QuantType.Q6_K,
        QuantType.Q5_K,
        QuantType.Q2_K,
        QuantType.IQ4_XS,
    ]:
        if group_size(qt) != 32:
            print("FAIL: group_size must be 32 for all GGUF K formats")
            abort()
    if block_bytes(QuantType.Q4_K_M) != 144:
        print("FAIL: block_bytes Q4_K_M")
        abort()
    if block_bytes(QuantType.Q8_0) != 34:
        print("FAIL: block_bytes Q8_0")
        abort()
    if block_bytes(QuantType.Q6_K) != 210:
        print("FAIL: block_bytes Q6_K")
        abort()
    if block_bytes(QuantType.Q5_K) != 176:
        print("FAIL: block_bytes Q5_K")
        abort()
    if block_bytes(QuantType.Q2_K) != 56:
        print("FAIL: block_bytes Q2_K")
        abort()
    if block_bytes(QuantType.IQ4_XS) != 136:
        print("FAIL: block_bytes IQ4_XS")
        abort()
    if block_elems(QuantType.Q8_0) != 32:
        print("FAIL: block_elems Q8_0")
        abort()
    if block_elems(QuantType.Q4_K_M) != 256:
        print("FAIL: block_elems Q4_K_M")
        abort()
    if ggml_type(QuantType.Q4_K_M) != 12:
        print("FAIL: ggml_type Q4_K_M")
        abort()
    if ggml_type(QuantType.Q8_0) != 8:
        print("FAIL: ggml_type Q8_0")
        abort()
    if ggml_type(QuantType.Q6_K) != 14:
        print("FAIL: ggml_type Q6_K")
        abort()
    if ggml_type(QuantType.Q5_K) != 13:
        print("FAIL: ggml_type Q5_K")
        abort()
    if ggml_type(QuantType.Q2_K) != 11:
        print("FAIL: ggml_type Q2_K")
        abort()
    if ggml_type(QuantType.IQ4_XS) != 23:
        print("FAIL: ggml_type IQ4_XS")
        abort()
    print("  quant_type helpers ok")


def test_matmul_quantized_q8_0_f32():
    comptime M = 2
    comptime N = 2
    comptime K = 32
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, K))
    for i in range(M):
        for k in range(K):
            a.set(i * K + k, Scalar[DType.float32](Float32(k % 5)))
    var b_quant = tensor_zeros[DType.uint8, 2](StaticTuple[Int, 2](N, 34))
    # row 0: d = 0.5,  qs[k] = k - 16   -> deq[k] = 0.5 * (k - 16)
    # row 1: d = -0.25, qs[k] = 15 - k  -> deq[k] = -0.25 * (15 - k)
    build_q8_0_row(b_quant, 0, Float32(0.5), False)
    build_q8_0_row(b_quant, 34, Float32(-0.25), True)
    var scale = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
    var out = matmul_quantized_cpu[DType.float32, QuantType.Q8_0, 32](
        a, b_quant, scale
    )
    for i in range(M):
        for j in range(N):
            var expected = Float32(0.0)
            for k in range(K):
                var deq = (
                    Float32(0.5) * Float32(k - 16)
                    if j == 0
                    else Float32(-0.25) * Float32(15 - k)
                )
                expected += Float32(k % 5) * deq
            check_f32(
                Float32(out.get(i * N + j)),
                expected,
                "q8_0 f32 [" + String(i) + "," + String(j) + "]",
            )
    print("  q8_0 f32 ok")


def test_matmul_quantized_q4_k_f32():
    comptime M = 2
    comptime N = 2
    comptime K = 256
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, K))
    for i in range(M):
        for k in range(K):
            a.set(i * K + k, Scalar[DType.float32](Float32((k % 4) + i)))
    var b_quant = tensor_zeros[DType.uint8, 2](StaticTuple[Int, 2](N, 144))
    # row 0: d = 1.0 -> deq[k] = k % 16; row 1: d = 2.0 -> deq[k] = 2 * (k % 16)
    build_q4_k_row(b_quant, 0, Float32(1.0))
    build_q4_k_row(b_quant, 144, Float32(2.0))
    var scale = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
    var out = matmul_quantized_cpu[DType.float32, QuantType.Q4_K_M, 32](
        a, b_quant, scale
    )
    for i in range(M):
        for j in range(N):
            var d = Float32(1.0) if j == 0 else Float32(2.0)
            var expected = Float32(0.0)
            for k in range(K):
                expected += Float32((k % 4) + i) * d * Float32(k % 16)
            check_f32(
                Float32(out.get(i * N + j)),
                expected,
                "q4_k f32 [" + String(i) + "," + String(j) + "]",
            )
    print("  q4_k f32 ok")


def test_matmul_quantized_q8_0_f16():
    # Same blocks as the f32 Q8_0 test; the dequantized weights and the
    # result are stored in FP16 (the values are exact halves, so only the
    # final store may round).
    comptime M = 2
    comptime N = 2
    comptime K = 32
    var a = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M):
        for k in range(K):
            a.set(i * K + k, Scalar[DType.float16](Float32(k % 5)))
    var b_quant = tensor_zeros[DType.uint8, 2](StaticTuple[Int, 2](N, 34))
    build_q8_0_row(b_quant, 0, Float32(0.5), False)
    build_q8_0_row(b_quant, 34, Float32(-0.25), True)
    var scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var out = matmul_quantized_cpu[DType.float16, QuantType.Q8_0, 32](
        a, b_quant, scale
    )
    for i in range(M):
        for j in range(N):
            var expected = Float32(0.0)
            for k in range(K):
                var deq = (
                    Float32(0.5) * Float32(k - 16)
                    if j == 0
                    else Float32(-0.25) * Float32(15 - k)
                )
                expected += Float32(k % 5) * deq
            var expected16 = Float32(Scalar[DType.float16](expected))
            check_f32(
                Float32(out.get(i * N + j)),
                expected16,
                "q8_0 f16 [" + String(i) + "," + String(j) + "]",
            )
    print("  q8_0 f16 ok")


def test_matmul_quantized_registry():
    # Erased-interface round trip: the registered "matmul_quantized_cpu"
    # entry is comptime-specialized to Q4_K_M (group_size 32).
    var registry = OpRegistry()
    registry.register_default_ops()
    var op = registry.get("matmul_quantized_cpu", None)
    comptime M = 1
    comptime N = 2
    comptime K = 256
    var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, K))
    for k in range(K):
        a.set(k, Scalar[DType.float32](Float32(k % 4)))
    var b_quant = tensor_zeros[DType.uint8, 2](StaticTuple[Int, 2](N, 144))
    build_q4_k_row(b_quant, 0, Float32(1.0))
    build_q4_k_row(b_quant, 144, Float32(2.0))
    var scale = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
    var inputs = List[AnyTensor]()
    inputs.append(to_any[DType.float32, 2](a))
    inputs.append(to_any[DType.uint8, 2](b_quant))
    inputs.append(to_any[DType.float32, 1](scale))
    var results = op.forward(inputs)
    var data = results[0].data.unsafe_bitcast[Scalar[DType.float32]]()
    for j in range(N):
        var d = Float32(1.0) if j == 0 else Float32(2.0)
        var expected = Float32(0.0)
        for k in range(K):
            expected += Float32(k % 4) * d * Float32(k % 16)
        check_f32(
            Float32(data.unsafe_load[width=1](offset=j)),
            expected,
            "registry q4_k [0," + String(j) + "]",
        )
    print("  registry matmul_quantized_cpu ok")


def test_matmul_quantized():
    test_quant_type_helpers()
    test_matmul_quantized_q8_0_f32()
    test_matmul_quantized_q4_k_f32()
    test_matmul_quantized_q8_0_f16()
    test_matmul_quantized_registry()
    print("test_matmul_quantized OK")


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
