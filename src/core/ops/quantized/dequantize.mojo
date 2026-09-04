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
#   * Q4_0 block  = 18 bytes:  d(2) qs(16)
#     value       = d * (q - 8)                     (unsigned 4-bit)
#   * IQ4_NL      = 18 bytes:  d(2) qs(16)
#     value       = d * kvalues_iq4nl[q]            (4-bit codebook)
#   * IQ4_XS      = 136 bytes: d(2) scales_h(2) scales_l(4) qs(128)
#     value       = d * (ls - 32) * kvalues_iq4nl[q]
#
# Output precision:
#   * `dequantize_into`     -> Tensor[DType.float16, 2]  (inference path;
#     the FP32 result is rounded to half precision on store)
#   * `dequantize_into_f32` -> Tensor[DType.float32, 2]  (bit-exact with
#     llama.cpp's `dequantize_row_*`; all arithmetic stays in FP32)
#
# The per-tensor quant type (`ggml_type`) is read from the GGUF metadata by
# the caller and passed in; the dispatch selects the dequantizer at runtime,
# so one call handles whatever quantization each tensor actually uses.
#
# All output tensors are pre-allocated by the caller (the MemoryPool owns
# the bulk buffers).
#
# Block mode (Q4-resident): `dequantize_blocks` / `dequantize_block` decode
# one or a few blocks (super-blocks) at a time into a caller-owned scratch
# buffer instead of the whole tensor.  The fused quantized matmul
# (`matmul_quantized_cpu` in `ops/cpu/matmul_cpu.mojo`) uses this to
# dequantize a block, accumulate it into the dot product, and discard it -
# the dequantized values never leave the kernel scope, so a quantized
# weight's resident footprint stays its on-disk (Q4) size.

from ...tensor import Tensor
from ...utils import unimplemented
from .quant_types import QuantType, block_elems
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.unsafe import bitcast
from std.utils.static_tuple import StaticTuple

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


# -- shared helpers -----------------------------------------------------------


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
    """Unpack the 6-bit scale and min for Q4_K/Q5_K sub-block `j`.

    Mirrors `get_scale_min_k4` in ggml-quants.c.
    """
    if j < 4:
        return (
            Int(scales.unsafe_load[width=1](offset=j)) & 63,
            Int(scales.unsafe_load[width=1](offset=j + 4)) & 63,
        )
    var d = (Int(scales.unsafe_load[width=1](offset=j + 4)) & 0xF) | (
        (Int(scales.unsafe_load[width=1](offset=j - 4)) >> 6) << 4
    )
    var m = (Int(scales.unsafe_load[width=1](offset=j + 4)) >> 4) | (
        (Int(scales.unsafe_load[width=1](offset=j)) >> 6) << 4
    )
    return (d, m)


def _read_i8(data: Pointer[UInt8, MutUntrackedOrigin], offset: Int) -> Int8:
    var raw = data.unsafe_load[width=1](offset=offset)
    return bitcast[DType.int8](raw)


def _nf4_value(q: Int) -> Float32:
    """NF4 (private type 30) 4-bit codebook."""
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


def _store_dequant[dtype: DType](dst: Tensor[dtype, 2], i: Int, y: Float32):
    """Store one FP32-computed dequantized value into `dst`.

    `comptime if` specializes on the destination dtype: for FP32 the value
    is stored unchanged, so the result is bit-exact with llama.cpp's
    `dequantize_row_*` output; for FP16 it is rounded to half precision.
    """
    comptime if dtype == DType.float32:
        dst.set(i, Scalar[dtype](y))
    else:
        dst.set(i, Scalar[dtype](y))


# -- block dequantizers -------------------------------------------------------
#
# Every block function is generic over the destination dtype so the same
# decode logic serves both the fp16 inference path and the fp32 path.


def _dequantize_q4_k_block[
    dtype: DType
](block: Pointer[UInt8, MutUntrackedOrigin], dst: Tensor[dtype, 2], base: Int,):
    """Q4_K super-block: the Q4_K_M workhorse (GGML type 12).

    Mirrors `dequantize_row_q4_K` in ggml-quants.c.  Sub-blocks come in
    pairs: 32 bytes of `qs` hold both sub-blocks of the pair (low nibble =
    even sub-block, high nibble = odd sub-block), and each value is
    `d * sc * q - dmin * m`, computed in FP32 from the 6-bit scale/min
    pairs.
    """
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var dmin = Float32(half.unsafe_load[width=1](offset=1))
    var scales = block.unsafe_offset(4)
    var qs = block.unsafe_offset(16)

    var q = qs
    for pair in range(4):
        var (sc0, m0) = _get_scale_min_k4(pair * 2, scales)
        var d0 = d * Float32(sc0)
        var m0v = dmin * Float32(m0)
        var (sc1, m1) = _get_scale_min_k4(pair * 2 + 1, scales)
        var d1 = d * Float32(sc1)
        var m1v = dmin * Float32(m1)
        for l in range(32):
            var b = Int(q.unsafe_load[width=1](offset=l))
            _store_dequant(
                dst, base + pair * 64 + l, d0 * Float32(b & 0xF) - m0v
            )
            _store_dequant(
                dst, base + pair * 64 + 32 + l, d1 * Float32(b >> 4) - m1v
            )
        q = q.unsafe_offset(32)


def _dequantize_q8_0_block[
    dtype: DType
](block: Pointer[UInt8, MutUntrackedOrigin], dst: Tensor[dtype, 2], base: Int,):
    """Q8_0 block: 32-element blocks, fp16 delta + int8 quants."""
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var qs = block.unsafe_offset(2)
    for i in range(32):
        var q = Int(bitcast[DType.int8](qs.unsafe_load[width=1](offset=i)))
        _store_dequant(dst, base + i, d * Float32(q))


def _dequantize_q4_0_block[
    dtype: DType
](block: Pointer[UInt8, MutUntrackedOrigin], dst: Tensor[dtype, 2], base: Int,):
    """Q4_0 block: 32-element blocks, fp16 delta + unsigned 4-bit quants.

    Mirrors `dequantize_row_q4_0` in ggml-quants.c: the block is
    `d(2 bytes fp16) + qs(16 bytes, 2 quants per byte)` and every value is
    `d * (q - 8)`, computed in FP32.
    """
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var qs = block.unsafe_offset(2)
    for j in range(16):
        var b = Int(qs.unsafe_load[width=1](offset=j))
        _store_dequant(dst, base + j, d * (Float32(b & 0xF) - Float32(8)))
        _store_dequant(dst, base + 16 + j, d * (Float32(b >> 4) - Float32(8)))


def _dequantize_q5_k_block[
    dtype: DType
](block: Pointer[UInt8, MutUntrackedOrigin], dst: Tensor[dtype, 2], base: Int,):
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
            _store_dequant(dst, base + sb * 32 + pos, d_sb * Float32(q) - m_sb)


def _dequantize_q6_k_block[
    dtype: DType
](block: Pointer[UInt8, MutUntrackedOrigin], dst: Tensor[dtype, 2], base: Int,):
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

            _store_dequant(dst, base + n * 128 + l, d * s0 * Float32(q1))
            _store_dequant(dst, base + n * 128 + l + 32, d * s1 * Float32(q2))
            _store_dequant(dst, base + n * 128 + l + 64, d * s2 * Float32(q3))
            _store_dequant(dst, base + n * 128 + l + 96, d * s3 * Float32(q4))


def _dequantize_iq4_nl_block[
    dtype: DType
](block: Pointer[UInt8, MutUntrackedOrigin], dst: Tensor[dtype, 2], base: Int,):
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var qs = block.unsafe_offset(2)
    for j in range(16):
        var b = Int(qs.unsafe_load[width=1](offset=j))
        _store_dequant(dst, base + j, d * Float32(_kvalues_iq4nl(b & 0xF)))
        _store_dequant(dst, base + 16 + j, d * Float32(_kvalues_iq4nl(b >> 4)))


def _dequantize_iq4_xs_block[
    dtype: DType
](block: Pointer[UInt8, MutUntrackedOrigin], dst: Tensor[dtype, 2], base: Int,):
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
        var low = (
            Int(scales_l.unsafe_load[width=1](offset=ib // 2)) >> (4 * (ib % 2))
            & 0xF
        )
        var high = (sh >> (2 * ib)) & 3
        var ls = low | (high << 4)
        var dl = d * Float32(ls - 32)
        for j in range(16):
            var b = Int(q_ptr.unsafe_load[width=1](offset=j))
            _store_dequant(
                dst, base + ib * 32 + j, dl * Float32(_kvalues_iq4nl(b & 0xF))
            )
            _store_dequant(
                dst,
                base + ib * 32 + 16 + j,
                dl * Float32(_kvalues_iq4nl(b >> 4)),
            )
        q_ptr = q_ptr.unsafe_offset(16)


def _dequantize_nf4_block[
    dtype: DType
](block: Pointer[UInt8, MutUntrackedOrigin], dst: Tensor[dtype, 2], base: Int,):
    """NF4 (private type 30): d fp16 + 32x 4-bit codebook indices."""
    var half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var qs = block.unsafe_offset(2)
    for j in range(32):
        var b = Int(qs.unsafe_load[width=1](offset=j))
        _store_dequant(dst, base + j, d * _nf4_value(b & 0xF))
        _store_dequant(dst, base + 32 + j, d * _nf4_value(b >> 4))


# -- dispatch -----------------------------------------------------------------


def _dequantize_dispatch[
    dtype: DType
](
    ggml_type: Int,
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[dtype, 2],
    numel: Int,
):
    """Runtime dispatch on the GGUF per-tensor quant type.

    `ggml_type` is the type field of the tensor's GGUF metadata, so the
    matching dequantizer is selected at runtime for whatever quantization
    the tensor actually uses.
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
    elif ggml_type == GGML_Q4_0:
        var nb = numel // 32
        for b in range(nb):
            _dequantize_q4_0_block(
                data.unsafe_offset(offset + b * 18), dst, b * 32
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
            dst.set(i, Scalar[dtype](src.unsafe_load[width=1](offset=i)))
    elif ggml_type == GGML_F32:
        var src = data.unsafe_offset(offset).unsafe_bitcast[
            Scalar[DType.float32]
        ]()
        for i in range(numel):
            dst.set(i, Scalar[dtype](src.unsafe_load[width=1](offset=i)))
    else:
        unimplemented("dequantize_into: unsupported ggml type")


# -- public entry points -------------------------------------------------------


def dequantize_into(
    ggml_type: Int,
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[DType.float16, 2],
    numel: Int,
):
    """Dequantize `numel` weights at `data + offset` into `dst` (fp16).

    `ggml_type` is the per-tensor quant type from the GGUF metadata; the
    dispatch selects the dequantizer at runtime.  `dst` must be
    pre-allocated with exactly `numel` elements.
    """
    _dequantize_dispatch[DType.float16](ggml_type, data, offset, dst, numel)


def dequantize_into_f32(
    ggml_type: Int,
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[DType.float32, 2],
    numel: Int,
):
    """Dequantize `numel` weights at `data + offset` into `dst` (fp32).

    Same runtime dispatch as `dequantize_into`, but the result is kept in
    FP32: every value is bit-exact with llama.cpp's `dequantize_row_*`
    (ggml-quants.c), since all arithmetic is done in FP32 in the same
    order as the C reference.
    """
    _dequantize_dispatch[DType.float32](ggml_type, data, offset, dst, numel)


def dequantize_q4_k_m_f32(
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[DType.float32, 2],
    numel: Int,
):
    """Q4_K_M -> FP32 dequantize kernel.

    `Q4_K_M` is the GGUF quantization scheme whose dominant tensor type is
    GGML Q4_K (type 12); this kernel dequantizes a Q4_K tensor's `numel`
    weights (a multiple of QK_K = 256) into an FP32 destination.  The
    result matches llama.cpp's `dequantize_row_q4_K` bit-for-bit.

    For mixed-type Q4_K_M models, use `dequantize_into_f32` with each
    tensor's `ggml_type` from the GGUF metadata.
    """
    var nb = numel // QK_K
    for b in range(nb):
        _dequantize_q4_k_block(
            data.unsafe_offset(offset + b * 144), dst, b * QK_K
        )


# -- per-format public entry points (comptime dispatch) ----------------------
#
# `dequantize_into` / `dequantize_into_f32` select the format at runtime
# from the GGUF metadata.  The per-format entry points below exist so
# comptime-specialized operators (e.g. `matmul_quantized_cpu` in
# `ops/cpu/matmul_cpu.mojo`) can pick the dequantizer at compile time with
# a `comptime if`.  Each is a thin wrapper over the same block kernels
# above, so the decode logic stays bit-exact with llama.cpp.


def dequantize_q4_K_M[
    dtype: DType
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[dtype, 2],
    numel: Int,
):
    """Q4_K (Q4_K_M scheme) -> `dtype`; bit-exact with `dequantize_row_q4_K`."""
    var nb = numel // QK_K
    for b in range(nb):
        _dequantize_q4_k_block(
            data.unsafe_offset(offset + b * 144), dst, b * QK_K
        )


def dequantize_q5_K[
    dtype: DType
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[dtype, 2],
    numel: Int,
):
    """Q5_K -> `dtype`; bit-exact with `dequantize_row_q5_K`."""
    var nb = numel // QK_K
    for b in range(nb):
        _dequantize_q5_k_block(
            data.unsafe_offset(offset + b * 176), dst, b * QK_K
        )


def dequantize_q6_K[
    dtype: DType
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[dtype, 2],
    numel: Int,
):
    """Q6_K -> `dtype`; bit-exact with `dequantize_row_q6_K`."""
    var nb = numel // QK_K
    for b in range(nb):
        _dequantize_q6_k_block(
            data.unsafe_offset(offset + b * 210), dst, b * QK_K
        )


def dequantize_q8_0[
    dtype: DType
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[dtype, 2],
    numel: Int,
):
    """Q8_0 -> `dtype`; bit-exact with `dequantize_row_q8_0`."""
    var nb = numel // 32
    for b in range(nb):
        _dequantize_q8_0_block(
            data.unsafe_offset(offset + b * 34), dst, b * 32
        )


def dequantize_iq4_xs[
    dtype: DType
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[dtype, 2],
    numel: Int,
):
    """IQ4_XS -> `dtype`; bit-exact with `dequantize_row_iq4_xs`."""
    var nb = numel // QK_K
    for b in range(nb):
        _dequantize_iq4_xs_block(
            data.unsafe_offset(offset + b * 136), dst, b * QK_K
        )


def dequantize_q4_0[
    dtype: DType
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Tensor[dtype, 2],
    numel: Int,
):
    """Q4_0 -> `dtype`; bit-exact with `dequantize_row_q4_0`."""
    var nb = numel // 32
    for b in range(nb):
        _dequantize_q4_0_block(data.unsafe_offset(offset + b * 18), dst, b * 32)


# -- block mode (Q4-resident, per-block dequantization) -------------------
#
# `dequantize_into` / `dequantize_into_f32` decode a whole tensor.  The
# block-mode entry points below decode exactly `n_blocks` consecutive
# blocks (super-blocks) into a caller-owned scratch buffer, so the fused
# quantized matmul can dequantize one block, fold it into the dot product,
# and discard it before the next block: the dequantized values never leave
# the kernel scope, and the weight's resident footprint stays its on-disk
# (Q4) size instead of doubling into fp16/fp32.
#
# The decode is the same bit-exact block kernels used by the whole-tensor
# path (a zero-copy `Tensor` view over the scratch pointer), so block mode
# and whole-tensor mode agree element for element.


def dequantize_blocks[
    dtype: DType,
    quant_type: QuantType,
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin],
    n_blocks: Int,
):
    """Dequantize `n_blocks` consecutive blocks of `quant_type` into `dst`.

    `data + offset` must point at the first block; `dst` must hold
    `n_blocks * block_elems(quant_type)` elements.  `quant_type` is a
    comptime parameter, so only the chosen format's decode survives
    compilation (the Q4_K_M / Q4_0 workhorses included).
    """
    comptime be = block_elems(quant_type)
    var view = Tensor[dtype, 2](
        StaticTuple[Int, 2](1, n_blocks * be), dst
    )
    comptime if quant_type == QuantType.Q4_K_M:
        for b in range(n_blocks):
            _dequantize_q4_k_block(
                data.unsafe_offset(offset + b * 144), view, b * be
            )
    elif quant_type == QuantType.Q4_0:
        for b in range(n_blocks):
            _dequantize_q4_0_block(
                data.unsafe_offset(offset + b * 18), view, b * be
            )
    elif quant_type == QuantType.Q8_0:
        for b in range(n_blocks):
            _dequantize_q8_0_block(
                data.unsafe_offset(offset + b * 34), view, b * be
            )
    elif quant_type == QuantType.Q6_K:
        for b in range(n_blocks):
            _dequantize_q6_k_block(
                data.unsafe_offset(offset + b * 210), view, b * be
            )
    elif quant_type == QuantType.Q5_K:
        for b in range(n_blocks):
            _dequantize_q5_k_block(
                data.unsafe_offset(offset + b * 176), view, b * be
            )
    elif quant_type == QuantType.IQ4_XS:
        for b in range(n_blocks):
            _dequantize_iq4_xs_block(
                data.unsafe_offset(offset + b * 136), view, b * be
            )
    elif quant_type == QuantType.Q2_K:
        unimplemented("dequantize_blocks: Q2_K dequantizer not implemented")
    else:
        unimplemented("dequantize_blocks: unknown quant type")


def dequantize_block[
    dtype: DType,
    quant_type: QuantType,
](
    data: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin],
):
    """Dequantize exactly one block (super-block) of `quant_type` into
    `dst` (`block_elems(quant_type)` elements).  The one-block form of
    `dequantize_blocks` - the inner-loop primitive of the fused quantized
    matmul."""
    dequantize_blocks[dtype, quant_type](data, offset, dst, 1)
