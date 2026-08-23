# core/ops/quantized/dequantize.mojo
#
# GGUF block-quantization dequantizers, all written in pure Mojo.
#
# The decode logic follows llama.cpp's `ggml-quants.c` (translated to Mojo):
#   * super-block size QK_K = 256 for the K formats
#   * Q4_K block  = 144 bytes: d(2) dmin(2) scales(12) qs(128)
#     value       = d * sc * q - dmin * m           (6-bit scale/min pairs)
#   * Q5_K block  = 176 bytes: d(2) dmin(2) scales(12) qh(32) qs(128)
#     value       = d * (sc * q - m)
#   * Q6_K block  = 210 bytes: ql(128) qh(64) scales(16) d(2)
#     value       = d * sc * q                      (q centered at 32)
#   * Q8_0 block  = 34 bytes:  d(2) qs(32)
#     value       = d * qs[i]                       (signed int8)
#   * IQ4_NL      = 18 bytes:  d(2) qs(16)
#     value       = d * kvalues_iq4nl[q]            (4-bit codebook)
#   * IQ4_XS      = 136 bytes: d(2) scales_h(2) scales_l(4) qs(128)
#     value       = d * (ls - 32) * kvalues_iq4nl[q]
#
# Output is always written into a pre-allocated `Tensor[DType.float16, 2]`.

from ...tensor import Tensor
from ...utils import unimplemented
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.unsafe import bitcast

comptime QK_K = 256
comptime GGML_F32 = 0
comptime GGML_F16 = 1
comptime GGML_Q4_0 = 2
comptime GGML_Q5_K = 13
comptime GGML_Q6_K = 14
comptime GGML_Q8_0 = 8
comptime GGML_Q4_K = 12
comptime GGML_IQ4_NL = 20
comptime GGML_IQ4_XS = 23
comptime GGML_NF4 = 30  # infer-train private NF4 (64-el blocks)


def _kvalues_iq4nl(q: Int) -> Int:
    """IQ4_NL nonlinear codebook: 4-bit index -> signed value."""
    if q == 0:
        return -127
    if q == 1:
        return -104
    if q == 2:
        return -83
    if q == 3:
        return -65
    if q == 4:
        return -49
    if q == 5:
        return -35
    if q == 6:
        return -22
    if q == 7:
        return -10
    if q == 8:
        return 1
    if q == 9:
        return 13
    if q == 10:
        return 25
    if q == 11:
        return 38
    if q == 12:
        return 53
    if q == 13:
        return 69
    if q == 14:
        return 89
    return 113


def _get_scale_min_k4(
    j: Int, scales: Pointer[UInt8, MutUntrackedOrigin]
) -> Tuple[Int, Int]:
    """Unpack the 6-bit scale and min for Q5_K sub-block `j`."""
    if j < 4:
        return (
            Int(scales.unsafe_load[width=1](offset=j)) & 63,
            Int(scales.unsafe_load[width=1](offset=j + 4)) & 63,
        )
    var d = (
        Int(scales.unsafe_load[width=1](offset=j + 4)) & 0xF
    ) | ((Int(scales.unsafe_load[width=1](offset=j - 4)) >> 6) << 4)
    var m = (
        Int(scales.unsafe_load[width=1](offset=j + 4)) >> 4
    ) | ((Int(scales.unsafe_load[width=1](offset=j)) >> 6) << 4)
    return (d, m)


def _dequantize_q5_k_block(
    block: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float16, 2],
    base: Int,
):
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var dmin = Float32(half.unsafe_load[width=1](offset=1))
    var scales = block.unsafe_offset(4)
    var qh = block.unsafe_offset(16)
    var qs = block.unsafe_offset(48)

    for sb in range(8):
        var (sc, m) = _get_scale_min_k4(sb, scales)
        var d_sb = d * Float32(sc)
        var m_sb = dmin * Float32(m)
        var bit = 1 << sb
        var ql_base = (sb // 2) * 32
        for pos in range(32):
            var ql_byte = Int(qs.unsafe_load[width=1](offset=ql_base + pos))
            var qh_byte = Int(qh.unsafe_load[width=1](offset=pos))
            var nibble: Int
            if sb % 2 == 0:
                nibble = ql_byte & 0xF
            else:
                nibble = (ql_byte >> 4) & 0xF
            var to_add = 16 if (qh_byte & bit) != 0 else 0
            var q = nibble + to_add
            var y = d_sb * Float32(q) - m_sb
            dst.set(base + sb * 32 + pos, Scalar[DType.float16](y))


def _dequantize_q6_k_block(
    block: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float16, 2],
    base: Int,
):
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=104))
    var ql = block.unsafe_offset(0)
    var qh = block.unsafe_offset(128)
    var sc = block.unsafe_offset(192)

    for n in range(2):
        var ql_off = 64 * n
        var qh_off = 32 * n
        var sc_off = 8 * n
        for l in range(32):
            var si = l // 16
            var ql_lo = Int(ql.unsafe_load[width=1](offset=ql_off + l))
            var ql_hi = Int(ql.unsafe_load[width=1](offset=ql_off + l + 32))
            var qh_b = Int(qh.unsafe_load[width=1](offset=qh_off + l))
            var q1 = ((ql_lo & 0xF) | ((qh_b & 3) << 4)) - 32
            var q2 = ((ql_hi & 0xF) | (((qh_b >> 2) & 3) << 4)) - 32
            var q3 = ((ql_lo >> 4) | (((qh_b >> 4) & 3) << 4)) - 32
            var q4 = ((ql_hi >> 4) | (((qh_b >> 6) & 3) << 4)) - 32

            var s0 = Float32(_read_i8(sc, sc_off + si))
            var s1 = Float32(_read_i8(sc, sc_off + si + 2))
            var s2 = Float32(_read_i8(sc, sc_off + si + 4))
            var s3 = Float32(_read_i8(sc, sc_off + si + 6))

            dst.set(
                base + n * 128 + l, Scalar[DType.float16](d * s0 * Float32(q1))
            )
            dst.set(
                base + n * 128 + l + 32,
                Scalar[DType.float16](d * s1 * Float32(q2)),
            )
            dst.set(
                base + n * 128 + l + 64,
                Scalar[DType.float16](d * s2 * Float32(q3)),
            )
            dst.set(
                base + n * 128 + l + 96,
                Scalar[DType.float16](d * s3 * Float32(q4)),
            )


def _read_i8(
    data: Pointer[UInt8, MutUntrackedOrigin], offset: Int
) -> Int8:
    var raw = data.unsafe_load[width=1](offset=offset)
    return bitcast[DType.int8](raw)


def _dequantize_q4_k_block(
    block: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float16, 2],
    base: Int,
):
    """Q4_K super-block: 8 sub-blocks of 32, 6-bit scale/min per sub-block."""
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var dmin = Float32(half.unsafe_load[width=1](offset=1))
    var scales = block.unsafe_offset(4)
    var qs = block.unsafe_offset(16)

    for sb in range(8):
        var (sc, m) = _get_scale_min_k4(sb, scales)
        var d_sb = d * Float32(sc)
        var m_sb = dmin * Float32(m)
        var ql_base = (sb // 2) * 32
        var ql_off = (sb % 2) * 32  # low nibble: +0, high nibble: +32
        for pos in range(32):
            var ql_byte = Int(qs.unsafe_load[width=1](offset=ql_base + pos))
            var q: Int
            if sb % 2 == 0:
                q = ql_byte & 0xF
            else:
                q = (ql_byte >> 4) & 0xF
            _ = ql_off
            var y = d_sb * Float32(q) - m_sb
            dst.set(base + sb * 32 + pos, Scalar[DType.float16](y))


def _dequantize_q8_0_block(
    block: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float16, 2],
    base: Int,
):
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var qs = block.unsafe_offset(2)
    for i in range(32):
        var q = Int(bitcast[DType.int8](qs.unsafe_load[width=1](offset=i)))
        dst.set(base + i, Scalar[DType.float16](d * Float32(q)))


def _dequantize_iq4_nl_block(
    block: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float16, 2],
    base: Int,
):
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var qs = block.unsafe_offset(2)
    for j in range(16):
        var b = Int(qs.unsafe_load[width=1](offset=j))
        dst.set(
            base + j,
            Scalar[DType.float16](d * Float32(_kvalues_iq4nl(b & 0xF))),
        )
        dst.set(
            base + 16 + j,
            Scalar[DType.float16](d * Float32(_kvalues_iq4nl(b >> 4))),
        )


def _dequantize_nf4_block(
    block: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float16, 2],
    base: Int,
):
    """NF4 (private type 30): d fp16 + 32x 4-bit codebook indices."""
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var qs = block.unsafe_offset(2)
    for j in range(32):
        var b = Int(qs.unsafe_load[width=1](offset=j))
        dst.set(
            base + j,
            Scalar[DType.float16](d * _nf4_value(b & 0xF)),
        )
        dst.set(
            base + 32 + j,
            Scalar[DType.float16](d * _nf4_value(b >> 4)),
        )


def _nf4_value(q: Int) -> Float32:
    if q == 0:
        return Float32(-1.0)
    if q == 1:
        return Float32(-0.6961928009986877)
    if q == 2:
        return Float32(-0.5250730514526367)
    if q == 3:
        return Float32(-0.39491748809814453)
    if q == 4:
        return Float32(-0.28444138169288635)
    if q == 5:
        return Float32(-0.18477343022823334)
    if q == 6:
        return Float32(-0.09105003625154495)
    if q == 7:
        return Float32(0.0)
    if q == 8:
        return Float32(0.07958029955625534)
    if q == 9:
        return Float32(0.16093020141124725)
    if q == 10:
        return Float32(0.24616029858589172)
    if q == 11:
        return Float32(0.33791524171829224)
    if q == 12:
        return Float32(0.44070982933044434)
    if q == 13:
        return Float32(0.5626170039176941)
    if q == 14:
        return Float32(0.7229568362236023)
    return Float32(1.0)


def _dequantize_iq4_xs_block(
    block: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float16, 2],
    base: Int,
):
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var sh = Int(
        block.unsafe_bitcast[Scalar[DType.uint16]]().unsafe_load[width=1](
            offset=1
        )
    )
    var scales_l = block.unsafe_offset(4)
    var qs = block.unsafe_offset(8)
    var q_ptr = qs
    for ib in range(8):
        var low = Int(
            scales_l.unsafe_load[width=1](offset=ib // 2)
        ) >> (4 * (ib % 2)) & 0xF
        var high = (sh >> (2 * ib)) & 3
        var ls = low | (high << 4)
        var dl = d * Float32(ls - 32)
        for j in range(16):
            var b = Int(q_ptr.unsafe_load[width=1](offset=j))
            dst.set(
                base + ib * 32 + j,
                Scalar[DType.float16](
                    dl * Float32(_kvalues_iq4nl(b & 0xF))
                ),
            )
            dst.set(
                base + ib * 32 + 16 + j,
                Scalar[DType.float16](
                    dl * Float32(_kvalues_iq4nl(b >> 4))
                ),
            )
        q_ptr = q_ptr.unsafe_offset(16)


def dequantize_into(
    ggml_type: Int,
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[DType.float16, 2],
    numel: Int,
):
    """Dequantize `numel` weights at `data + offset` into `dst` (fp16).

    `dst` must be pre-allocated with exactly `numel` elements.
    """
    if ggml_type == GGML_Q5_K:
        var nb = numel // QK_K
        for b in range(nb):
            _dequantize_q5_k_block(
                data.unsafe_offset(offset + b * 176), dst, b * QK_K
            )
    elif ggml_type == GGML_Q6_K:
        var nb = numel // QK_K
        for b in range(nb):
            _dequantize_q6_k_block(
                data.unsafe_offset(offset + b * 210), dst, b * QK_K
            )
    elif ggml_type == GGML_Q4_K:
        var nb = numel // QK_K
        for b in range(nb):
            _dequantize_q4_k_block(
                data.unsafe_offset(offset + b * 144), dst, b * QK_K
            )
    elif ggml_type == GGML_Q8_0:
        var nb = numel // 32
        for b in range(nb):
            _dequantize_q8_0_block(
                data.unsafe_offset(offset + b * 34), dst, b * 32
            )
    elif ggml_type == GGML_IQ4_NL:
        var nb = numel // 32
        for b in range(nb):
            _dequantize_iq4_nl_block(
                data.unsafe_offset(offset + b * 18), dst, b * 32
            )
    elif ggml_type == GGML_IQ4_XS:
        var nb = numel // QK_K
        for b in range(nb):
            _dequantize_iq4_xs_block(
                data.unsafe_offset(offset + b * 136), dst, b * QK_K
            )
    elif ggml_type == GGML_NF4:
        var nb = numel // 64
        for b in range(nb):
            _dequantize_nf4_block(
                data.unsafe_offset(offset + b * 34), dst, b * 64
            )
    elif ggml_type == GGML_F16:
        var src = data.unsafe_offset(offset).unsafe_bitcast[
            Scalar[DType.float16]
        ]()
        for i in range(numel):
            dst.set(i, src.unsafe_load[width=1](offset=i))
    elif ggml_type == GGML_F32:
        var src = data.unsafe_offset(offset).unsafe_bitcast[
            Scalar[DType.float32]
        ]()
        for i in range(numel):
            dst.set(
                i, Scalar[DType.float16](src.unsafe_load[width=1](offset=i))
            )
    else:
        unimplemented("dequantize_into: unsupported ggml type")
