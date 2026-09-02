# core/ops/quantized/quant_types.mojo
#
# Compile-time quantization format tags for the GGUF block formats
# (llama.cpp `ggml-quants.c`) plus per-format helpers.
#
# These tags parameterize the generic quantized operators (e.g.
# `matmul_quantized_cpu` in `ops/cpu/matmul_cpu.mojo`): `quant_type` is a
# comptime parameter, so the dequantizer is selected at compile time and
# every unused format path is eliminated.
#
# Mojo 1.0 note: `enum` was removed; `QuantType` follows the same
# struct-with-`comptime`-constants pattern as `QuantFormat` in
# `core/quantization.mojo` (and `Device` in `core/device.mojo`), which is
# the pattern known to work with `comptime if` in this toolchain.

struct QuantType(Copyable, Equatable, Movable, ImplicitlyCopyable):
    """Quantization format tag (compile-time value).

    The values mirror the GGUF block formats in llama.cpp's
    `ggml-quants.c`; `ggml_type` maps each tag to the `ggml_type` field of
    a tensor's GGUF metadata.
    """

    var _tag: Int8

    def __init__(out self, tag: Int8):
        self._tag = tag

    # GGUF block formats (tag -> ggml_type)
    comptime Q4_K_M = QuantType(Int8(0))  # Q4_K super-block, ggml 12
    comptime Q8_0 = QuantType(Int8(1))  # 32-element blocks, ggml 8
    comptime Q6_K = QuantType(Int8(2))  # Q6_K super-block, ggml 14
    comptime Q5_K = QuantType(Int8(3))  # Q5_K super-block, ggml 13
    comptime Q2_K = QuantType(Int8(4))  # Q2_K super-block, ggml 11 (reserved)
    comptime IQ4_XS = QuantType(Int8(5))  # IQ4_XS super-block, ggml 23

    def __eq__(self, other: Self) -> Bool:
        return self._tag == other._tag

    def __ne__(self, other: Self) -> Bool:
        return self._tag != other._tag


def ggml_type(quant_type: QuantType) -> Int:
    """GGUF `ggml_type` metadata value for `quant_type` (comptime)."""
    if quant_type == QuantType.Q4_K_M:
        return 12
    if quant_type == QuantType.Q8_0:
        return 8
    if quant_type == QuantType.Q6_K:
        return 14
    if quant_type == QuantType.Q5_K:
        return 13
    if quant_type == QuantType.Q2_K:
        return 11
    if quant_type == QuantType.IQ4_XS:
        return 23
    return -1


def num_bits(quant_type: QuantType) -> Int:
    """Nominal bits per weight element for `quant_type` (comptime).

    K-quant super-blocks carry extra bits for scales/mins on top of the
    nominal width (e.g. Q4_K is ~4.5 bits/elem); this reports the nominal
    codebook width only.
    """
    if quant_type == QuantType.Q4_K_M:
        return 4
    if quant_type == QuantType.Q8_0:
        return 8
    if quant_type == QuantType.Q6_K:
        return 6
    if quant_type == QuantType.Q5_K:
        return 5
    if quant_type == QuantType.Q2_K:
        return 2
    if quant_type == QuantType.IQ4_XS:
        return 4
    return 0


def group_size(quant_type: QuantType) -> Int:
    """Number of elements sharing one scale for `quant_type` (comptime).

    Every GGUF K format quantizes in 32-element sub-blocks (for Q8_0 the
    sub-block is the whole 32-element block).  0 = not a group format.
    """
    if (
        quant_type == QuantType.Q4_K_M
        or quant_type == QuantType.Q8_0
        or quant_type == QuantType.Q6_K
        or quant_type == QuantType.Q5_K
        or quant_type == QuantType.Q2_K
        or quant_type == QuantType.IQ4_XS
    ):
        return 32
    return 0


def block_bytes(quant_type: QuantType) -> Int:
    """Bytes per quantized block (super-block) for `quant_type` (comptime).

    Block sizes from llama.cpp's `ggml-quants.c`: Q4_K 144, Q5_K 176,
    Q6_K 210, Q8_0 34, Q2_K 56, IQ4_XS 136.
    """
    if quant_type == QuantType.Q4_K_M:
        return 144
    if quant_type == QuantType.Q8_0:
        return 34
    if quant_type == QuantType.Q6_K:
        return 210
    if quant_type == QuantType.Q5_K:
        return 176
    if quant_type == QuantType.Q2_K:
        return 56
    if quant_type == QuantType.IQ4_XS:
        return 136
    return 0


def block_elems(quant_type: QuantType) -> Int:
    """Elements per quantized block (super-block) for `quant_type` (comptime).

    Q8_0 blocks hold 32 elements; every K-quant super-block holds 256.
    """
    if quant_type == QuantType.Q8_0:
        return 32
    return 256
