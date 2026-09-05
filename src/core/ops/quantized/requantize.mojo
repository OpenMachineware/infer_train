# core/ops/quantized/requantize.mojo
#
# M7: post-fine-tuning weight re-quantization (requantize.mojo).
#
# Converts fp16 (dequantized) weights back into GGUF-compatible block
# formats so a fine-tuned .mmdl checkpoint can be exported for inference:
#
#   Q4_K  (GGML type 12): llama.cpp `quantize_row_q4_K_ref` - 256-element
#          super-blocks, 6-bit scale/min pairs per 32-element sub-block.
#   Q8_0  (GGML type 8):  32-element blocks, fp16 delta + int8 quants.
#   NF4   (infer-train private type 30): 64-element blocks with the
#          bitsandbytes NF4 codebook + fp16 scale.
#
# Round-trip quality is format-limited (as with llama.cpp's reference
# quantizer); the dequantized inference path is identical to loading the
# equivalent GGUF, so throughput matches a native quantized load.

from ...tensor import Tensor
from ...utils import unimplemented
from std.math import sqrt, round

comptime QK_K = 256
comptime GGML_Q4_K = 12
comptime GGML_Q8_0 = 8
comptime GGML_NF4 = 30  # private extension (64-el blocks)


def _nf4_codebook(q: Int) -> Float32:
    """The bitsandbytes NF4 nonlinear codebook (16 levels)."""
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


struct QuantizedWeights(Movable):
    """One tensor's requantized payload + its GGUF type code."""

    var ggml_type: Int
    var data: List[UInt8]
    var block_bytes: Int  # bytes per block

    def __init__(out self, ggml_type: Int, block_bytes: Int):
        self.ggml_type = ggml_type
        self.data = List[UInt8]()
        self.block_bytes = block_bytes


def _f16_bytes(v: Float32, mut out: List[UInt8]):
    var u = _f16_to_u16(Scalar[DType.float16](v))
    out.append(UInt8(u & 0xFF))
    out.append(UInt8((u >> 8) & 0xFF))


def _f16_to_u16(v: Scalar[DType.float16]) -> Int:
    """Bit pattern of an fp16 scalar (little-endian u16)."""
    var x = Float32(v)
    if x != x:  # NaN
        return 0x7E00
    var sign = 0
    if x < 0:
        sign = 0x8000
        x = -x
    if x == Float32(0):
        return sign
    if x >= Float32(65504.0):
        return sign | 0x7BFF
    # normalize
    var exp = 0
    var frac = x
    while frac >= Float32(2.0):
        frac = frac / Float32(2.0)
        exp += 1
    while frac < Float32(1.0) and exp > -14:
        frac = frac * Float32(2.0)
        exp -= 1
    var mant = Int(round((frac - Float32(1.0)) * Float32(1024.0)))
    if exp > 15:
        return sign | 0x7BFF
    if exp < -14:
        return sign
    return sign | ((exp + 15) << 10) | mant


# -- Q8_0 ---------------------------------------------------------------------


def requantize_q8_0(
    src: Tensor[DType.float16, 1], numel: Int
) -> QuantizedWeights:
    """32-element blocks: fp16 delta + int8 quants (GGML Q8_0)."""
    var out = QuantizedWeights(GGML_Q8_0, 34)
    var nb = numel // 32
    for b in range(nb):
        var amax = Float32(0)
        for j in range(32):
            var v = Float32(src.get(b * 32 + j))
            if v < 0:
                v = -v
            if v > amax:
                amax = v
        var d = amax / Float32(127.0)
        _f16_bytes(d, out.data)
        var id = Float32(0)
        if d != Float32(0):
            id = Float32(1.0) / d
        for j in range(32):
            var q = Int(round(Float32(src.get(b * 32 + j)) * id))
            if q > 127:
                q = 127
            if q < -127:
                q = -127
            out.data.append(UInt8(q & 0xFF))
    return out^


# -- Q4_K (llama.cpp quantize_row_q4_K_ref) -----------------------------------


def _pack_scale_min(j: Int, ls: Int, lm: Int, mut scales: List[UInt8]):
    if j < 4:
        scales[j] = UInt8(ls)
        scales[j + 4] = UInt8(lm)
    else:
        scales[j + 4] = UInt8((ls & 0xF) | ((lm & 0xF) << 4))
        scales[j - 4] = UInt8(Int(scales[j - 4]) | ((ls >> 4) << 6))
        scales[j] = UInt8(Int(scales[j]) | ((lm >> 4) << 6))


def requantize_q4_k(
    src: Tensor[DType.float16, 1], numel: Int
) -> QuantizedWeights:
    """256-element super-blocks (GGML Q4_K), reference quantizer."""
    var out = QuantizedWeights(GGML_Q4_K, 144)
    var nb = numel // QK_K
    for i in range(nb):
        # per-32-element sub-block scales/mins (weighted by magnitude)
        var max_scale = Float32(0)
        var max_min = Float32(0)
        var sub_scales = List[Float32]()
        var sub_mins = List[Float32]()
        var sub_q = List[Int]()
        for j in range(8):
            var mn = Float32(src.get(i * QK_K + j * 32))
            var mx = mn
            for l in range(32):
                var v = Float32(src.get(i * QK_K + j * 32 + l))
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            if mn > Float32(0):
                mn = Float32(0)  # make_qkx2_quants: min is clamped at 0
            var rng = (mx - mn) / Float32(15.0)  # 4-bit sub-block scale
            if rng > max_scale:
                max_scale = rng
            var pos_min = -mn  # *the_min = -min (stored positive)
            if pos_min > max_min:
                max_min = pos_min
            sub_scales.append(rng)
            sub_mins.append(pos_min)
            sub_q.append(0)
        var inv_scale = Float32(0)
        if max_scale > Float32(0):
            inv_scale = Float32(63.0) / max_scale
        var inv_min = Float32(0)
        if max_min > Float32(0):
            inv_min = Float32(63.0) / max_min
        var scales = List[UInt8]()
        for _ in range(12):
            scales.append(UInt8(0))
        var ls_list = List[Int]()
        var lm_list = List[Int]()
        for j in range(8):
            var ls = Int(round(inv_scale * sub_scales[j]))
            var lm = Int(round(inv_min * sub_mins[j]))
            if ls > 63:
                ls = 63
            if lm > 63:
                lm = 63
            if ls < 0:
                ls = 0
            if lm < 0:
                lm = 0
            ls_list.append(ls)
            lm_list.append(lm)
            _pack_scale_min(j, ls, lm, scales)
        _f16_bytes(max_scale / Float32(63.0), out.data)
        _f16_bytes(max_min / Float32(63.0), out.data)
        for s in scales:
            out.data.append(s)
        # quants
        var L = List[Int]()
        for _ in range(QK_K):
            L.append(0)
        for j in range(8):
            var d = Float32(max_scale / Float32(63.0)) * Float32(ls_list[j])
            if d == Float32(0):
                continue
            var dm = Float32(max_min / Float32(63.0)) * Float32(lm_list[j])
            for l in range(32):
                var q = Int(
                    round((Float32(src.get(i * QK_K + j * 32 + l)) + dm) / d)
                )
                if q > 15:
                    q = 15
                if q < 0:
                    q = 0
                L[j * 32 + l] = q
        for j in range(4):
            for l in range(32):
                out.data.append(
                    UInt8(L[j * 64 + l] | (L[j * 64 + l + 32] << 4))
                )
    return out^


# -- NF4 (bitsandbytes codebook, 64-element blocks) ---------------------------


def requantize_nf4(
    src: Tensor[DType.float16, 1], numel: Int
) -> QuantizedWeights:
    """64-element blocks: fp16 scale + 4-bit NF4 codebook indices.

    Stored as the infer-train private GGML type 30 (llama.cpp cannot read
    NF4 weights; use Q4_K/Q8_0 for llama.cpp-compatible exports).
    """
    var out = QuantizedWeights(GGML_NF4, 34)
    var nb = numel // 64
    for b in range(nb):
        var amax = Float32(0)
        for j in range(64):
            var v = Float32(src.get(b * 64 + j))
            if v < 0:
                v = -v
            if v > amax:
                amax = v
        var d = amax  # codebook spans [-1, 1]
        _f16_bytes(d, out.data)
        var id = Float32(0)
        if d != Float32(0):
            id = Float32(1.0) / d
        for j in range(32):
            var q0 = _nearest_nf4(Float32(src.get(b * 64 + j)) * id)
            var q1 = _nearest_nf4(Float32(src.get(b * 64 + j + 32)) * id)
            out.data.append(UInt8(q0 | (q1 << 4)))
    return out^


def _nearest_nf4(v: Float32) -> Int:
    var best = 0
    var best_d = Float32(3.0e38)
    for q in range(16):
        var d = v - _nf4_codebook(q)
        if d < 0:
            d = -d
        if d < best_d:
            best_d = d
            best = q
    return best


# -- dispatcher ---------------------------------------------------------------


def requantize(
    src: Tensor[DType.float16, 1], numel: Int, format: String
) -> QuantizedWeights:
    """Quantize `numel` fp16 weights into the named format.

    Formats: "Q4_K_M" / "Q4_K" (GGML Q4_K), "Q8_0", "NF4".
    """
    if format == "Q8_0":
        return requantize_q8_0(src, numel)
    if format == "NF4":
        return requantize_nf4(src, numel)
    if format == "Q4_K_M" or format == "Q4_K":
        return requantize_q4_k(src, numel)
    unimplemented("requantize: unknown format " + format)
    return QuantizedWeights(GGML_Q8_0, 34)
