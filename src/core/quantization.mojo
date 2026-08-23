# core/quantization.mojo
#
# Quantization data structures and the mixed-scheme quantize/dequantize
# pipeline.
#
# "Mixed scheme" means the *strategy* is decided at compile time (format,
# granularity, group size, symmetric-vs-asymmetric are all comptime
# parameters) while the *numerical parameters* (scale and zero point) are
# runtime tensors loaded from the model, so every layer may carry different
# scales without recompiling.
#
# Mojo 1.0 notes:
#   * `enum` was removed; `QuantFormat`/`QuantGranularity` are register
#     passable structs with `comptime` constants (see device.mojo).
#   * `comptime if` replaces the deprecated `@parameter if` and removes the
#     zero-point branch at compile time when the scheme is symmetric.
#   * M1 quantizes "in place" in the source dtype (an affine transform)
#     rather than bit-packing int4/int8; the pack/unpack stage lands with the
#     weight loader milestone.

from .device import Device
from .tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple

# ---------------------------------------------------------------------------
# Compile-time strategy enums (struct-based, see device.mojo for the pattern)
# ---------------------------------------------------------------------------


struct QuantFormat(Copyable, Equatable, Movable, ImplicitlyCopyable):
    var _tag: Int8

    def __init__(out self, tag: Int8):
        self._tag = tag

    comptime Q4_0 = QuantFormat(Int8(0))
    comptime Q4_K = QuantFormat(Int8(1))
    comptime Q6_K = QuantFormat(Int8(2))
    comptime Q8_0 = QuantFormat(Int8(3))
    comptime NF4 = QuantFormat(Int8(4))
    comptime FP4 = QuantFormat(Int8(5))

    def __eq__(self, other: Self) -> Bool:
        return self._tag == other._tag

    def __ne__(self, other: Self) -> Bool:
        return self._tag != other._tag


struct QuantGranularity(Copyable, Equatable, Movable, ImplicitlyCopyable):
    var _tag: Int8

    def __init__(out self, tag: Int8):
        self._tag = tag

    comptime PerTensor = QuantGranularity(Int8(0))
    comptime PerRow = QuantGranularity(Int8(1))
    comptime PerGroup = QuantGranularity(Int8(2))

    def __eq__(self, other: Self) -> Bool:
        return self._tag == other._tag

    def __ne__(self, other: Self) -> Bool:
        return self._tag != other._tag


# ---------------------------------------------------------------------------
# Runtime quantization parameters.  Scale dtype is fixed to float32, the
# universal scale representation used by GGUF-style weight formats.
# ---------------------------------------------------------------------------


struct QuantizationInfo(Copyable, Movable, ImplicitlyCopyable):
    var format: QuantFormat
    var granularity: QuantGranularity
    var group_size: Int
    var is_symmetric: Bool
    var scale: Tensor[DType.float32, 1]
    var zero_point: Optional[Tensor[DType.float32, 1]]

    def __init__(
        out self,
        format: QuantFormat,
        granularity: QuantGranularity,
        group_size: Int,
        is_symmetric: Bool,
        scale: Tensor[DType.float32, 1],
        zero_point: Optional[Tensor[DType.float32, 1]],
    ):
        self.format = format
        self.granularity = granularity
        self.group_size = group_size
        self.is_symmetric = is_symmetric
        self.scale = scale
        self.zero_point = zero_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qmax_for[format: QuantFormat]() -> Int:
    """Largest positive representable integer for a format (comptime)."""
    var qmax: Int = 127
    comptime if (
        format == QuantFormat.Q4_0
        or format == QuantFormat.Q4_K
        or format == QuantFormat.NF4
        or format == QuantFormat.FP4
    ):
        qmax = 15
    elif format == QuantFormat.Q6_K:
        qmax = 31
    return qmax


def _scale_index[granularity: QuantGranularity, group_size: Int, rank: Int](
    flat_index: Int, numel: Int, rows: Int
) -> Int:
    """Map a flat element index to its scale/zero-point vector index."""
    var idx = 0
    comptime if granularity == QuantGranularity.PerRow:
        var cols = 1
        if rows > 0:
            cols = numel // rows
        if cols < 1:
            cols = 1
        idx = flat_index // cols
    elif granularity == QuantGranularity.PerGroup:
        if group_size > 0:
            idx = flat_index // group_size
    _ = rank
    return idx


def _scale_length[granularity: QuantGranularity, group_size: Int, rank: Int](
    numel: Int, rows: Int
) -> Int:
    var length = 1
    comptime if granularity == QuantGranularity.PerRow:
        length = rows
    elif granularity == QuantGranularity.PerGroup:
        if group_size > 0:
            length = (numel + group_size - 1) // group_size
    _ = rank
    return length


# ---------------------------------------------------------------------------
# Static quantization: caller supplies scale (and optional zero point).
# ---------------------------------------------------------------------------


def quantize_static[
    dtype: DType,
    rank: Int,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    tensor: Tensor[dtype, rank],
    scale: Tensor[dtype, 1],
    zero_point: Optional[Tensor[dtype, 1]] = None,
) -> Tensor[dtype, rank]:
    """Apply a known affine quantization: `q = (x - zp) / scale`.

    For symmetric schemes `zero_point` is ignored entirely (and is
    conventionally `None`); `comptime if` removes the subtraction at compile
    time.  `format` participates only through the caller-chosen scale, which
    is how per-layer mixed precision is expressed.
    """
    var out = tensor_zeros[dtype, rank](tensor.shape())
    var numel = tensor.numel()
    var rows = 1
    comptime if rank >= 2:
        rows = tensor.shape()[0]

    for i in range(numel):
        var scale_idx = _scale_index[granularity, group_size, rank](
            i, numel, rows
        )
        var s = scale.get(scale_idx)
        var x = tensor.get(i)
        comptime if is_symmetric:
            out.set(i, x / s)
        else:
            var zp = zero_point.value().get(scale_idx)
            out.set(i, (x - zp) / s)
    return out^


# ---------------------------------------------------------------------------
# Dynamic quantization: derive scale (and zero point) from the input.
# ---------------------------------------------------------------------------


def quantize_dynamic[
    dtype: DType,
    rank: Int,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    tensor: Tensor[dtype, rank]
) -> Tuple[Tensor[dtype, rank], QuantizationInfo]:
    """Compute scales from `tensor`, quantize, and return tensor + info.

    Scale length depends on `granularity`; `comptime if` selects the correct
    reduction strategy with no runtime branch.
    """
    var (scale, zero_point) = _dynamic_scale_zp[
        dtype, rank, format, granularity, group_size, is_symmetric
    ](tensor)

    var scale_len = scale.numel()
    var scale_dtype = tensor_zeros[dtype, 1](StaticTuple[Int, 1](scale_len))
    var zp_dtype: Optional[Tensor[dtype, 1]] = None
    for i in range(scale_len):
        scale_dtype.set(i, Scalar[dtype](Float32(scale.get(i))))
    comptime if not is_symmetric:
        var zp_d = tensor_zeros[dtype, 1](StaticTuple[Int, 1](scale_len))
        for i in range(scale_len):
            zp_d.set(i, Scalar[dtype](Float32(zero_point.value().get(i))))
        zp_dtype = zp_d^

    var quantized = quantize_static[
        dtype, rank, format, granularity, group_size, is_symmetric
    ](tensor, scale_dtype, zp_dtype)

    var info = QuantizationInfo(
        format, granularity, group_size, is_symmetric, scale, zero_point
    )
    return (quantized^, info^)


def _dynamic_scale_zp[
    dtype: DType,
    rank: Int,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    tensor: Tensor[dtype, rank]
) -> Tuple[Tensor[DType.float32, 1], Optional[Tensor[DType.float32, 1]]]:
    """Derive per-tensor / per-row / per-group scale and zero point."""
    var numel = tensor.numel()
    var rows = 1
    comptime if rank >= 2:
        rows = tensor.shape()[0]

    var qmax = _qmax_for[format]()
    var scale_len = _scale_length[granularity, group_size, rank](
        numel, rows
    )
    var scale = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](scale_len))
    var zero_point: Optional[Tensor[DType.float32, 1]] = None

    comptime if granularity == QuantGranularity.PerRow:
        var (mins, maxs) = _per_row_ranges[dtype, rank](tensor)
        var zp_tensor = tensor_zeros[DType.float32, 1](
            StaticTuple[Int, 1](scale_len)
        )
        for r in range(rows):
            _fill_scale_zp[
                is_symmetric
            ](mins.get(r), maxs.get(r), qmax, scale, zp_tensor, r)
        comptime if not is_symmetric:
            zero_point = zp_tensor^
    elif granularity == QuantGranularity.PerGroup:
        var zp_tensor = tensor_zeros[DType.float32, 1](
            StaticTuple[Int, 1](scale_len)
        )
        var g = 0
        var start = 0
        while start < numel:
            var end = start + group_size
            if end > numel:
                end = numel
            var mn = Float32(tensor.get(start))
            var mx = mn
            var i = start + 1
            while i < end:
                var v = Float32(tensor.get(i))
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
                i += 1
            _fill_scale_zp[is_symmetric](mn, mx, qmax, scale, zp_tensor, g)
            g += 1
            start = end
        comptime if not is_symmetric:
            zero_point = zp_tensor^
    else:
        var mn = Float32(0)
        var mx = Float32(0)
        if numel > 0:
            mn = Float32(tensor.get(0))
            mx = mn
            for i in range(1, numel):
                var v = Float32(tensor.get(i))
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        var zp_tensor = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
        _fill_scale_zp[is_symmetric](mn, mx, qmax, scale, zp_tensor, 0)
        comptime if not is_symmetric:
            zero_point = zp_tensor^

    return (scale^, zero_point)


def _fill_scale_zp[is_symmetric: Bool](
    mn: Float32,
    mx: Float32,
    qmax: Int,
    mut scale: Tensor[DType.float32, 1],
    mut zero_point: Tensor[DType.float32, 1],
    slot: Int,
):
    """Write scale (and zero point when asymmetric) for one block."""
    comptime if is_symmetric:
        var amax = mx if mx > -mn else -mn
        scale.set(slot, Scalar[DType.float32](amax / Float32(qmax)))
    else:
        var diff = mx - mn
        if diff == 0:
            diff = 1
        scale.set(
            slot, Scalar[DType.float32](diff / Float32(2 * qmax + 1))
        )
        zero_point.set(
            slot,
            Scalar[DType.float32](
                -Float32(qmax + 1)
                - mn * Float32(2 * qmax + 1) / diff
            ),
        )


def _per_row_ranges[dtype: DType, rank: Int](
    tensor: Tensor[dtype, rank]
) -> Tuple[Tensor[DType.float32, 1], Tensor[DType.float32, 1]]:
    """Return (min-per-row, max-per-row) for a rank-2 tensor."""
    var rows = tensor.shape()[0]
    var cols = tensor.shape()[1]
    var mins = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](rows))
    var maxs = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](rows))
    for r in range(rows):
        var base = r * cols
        var mn = Float32(tensor.get(base))
        var mx = mn
        for c in range(1, cols):
            var v = Float32(tensor.get(base + c))
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        mins.set(r, Scalar[DType.float32](mn))
        maxs.set(r, Scalar[DType.float32](mx))
    return (mins^, maxs^)


# ---------------------------------------------------------------------------
# Dequantization: invert the affine transform.
# ---------------------------------------------------------------------------


def dequantize[
    dtype: DType,
    rank: Int,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    quantized: Tensor[dtype, rank],
    scale: Tensor[dtype, 1],
    zero_point: Optional[Tensor[dtype, 1]] = None,
) -> Tensor[dtype, rank]:
    """Reconstruct floating point values from quantized ones.

    * Asymmetric: `value = scale * q + zero_point`
    * Symmetric:  `value = scale * q` (zero_point is None)

    `format`/`granularity`/`group_size`/`is_symmetric` are comptime; only
    `scale`/`zero_point` are runtime data loaded from the model.
    """
    var out = tensor_zeros[dtype, rank](quantized.shape())
    var numel = quantized.numel()
    var rows = 1
    comptime if rank >= 2:
        rows = quantized.shape()[0]

    for i in range(numel):
        var scale_idx = _scale_index[granularity, group_size, rank](
            i, numel, rows
        )
        var s = scale.get(scale_idx)
        var q = quantized.get(i)
        comptime if is_symmetric:
            out.set(i, s * q)
        else:
            var zp = zero_point.value().get(scale_idx)
            out.set(i, s * q + zp)
    return out^
