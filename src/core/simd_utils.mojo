# core/simd_utils.mojo
#
# M5: shared SIMD helpers for the CPU kernels.
#
# Every elementwise kernel reduces to the same three shapes - a vectorized
# main loop, a scalar tail, and (for f16) f32 accumulation - so they share
# these helpers instead of re-deriving the W/alignment bookkeeping.  The
# widths are comptime constants chosen per dtype (128-bit NEON/SSE):
#
#   f16: 8 lanes   f32: 4 lanes
#
# Mojo 1.0 note: SIMD lane inserts are miscompiled, so helpers that need
# per-lane work (silu, exp) read lanes via `v[i]` and accumulate scalarly
# while still using vector loads/stores where possible.
#
# M8: adaptive SIMD width.
#
# The legacy kernels above keep their fixed 128-bit widths (the default
# behavior is unchanged).  On top of that, this module provides:
#
#   * `get_optimal_simd_width(dim, is_power_of_two)` - a comptime-evaluable
#     heuristic that picks a SIMD bit width (64/128/256) for a row of
#     `dim` elements;
#   * `simd_lanes(bit_width, dtype)` - the bit-width -> lane-count map used
#     to turn the choice into a `SIMD[dtype, W]` / `unsafe_load[width=W]`
#     parameter;
#   * `AutotuneCache` + `autotune_width_f16/f32` - a runtime micro-benchmark
#     that times the three candidate widths on a scratch row and remembers
#     the winner per row length.
#
# The chosen width is always handed to the kernels as a *comptime*
# parameter: the autotuner produces a runtime Int (the bit width), and the
# `*_autotuned` dispatchers in the op files branch over it, each branch
# calling a compile-time instantiation with a literal lane width.  Mojo 1.0
# has no runtime codegen, so all candidate instantiations are compiled into
# the binary and the dispatch selects among them.

from .tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple
from std.time import perf_counter_ns

comptime W_F16 = 8
comptime W_F32 = 4

# M8: the SIMD bit widths the autotuner searches over.
comptime SIMD_WIDTH_64 = 64
comptime SIMD_WIDTH_128 = 128
comptime SIMD_WIDTH_256 = 256


def simd_width[dtype: DType]() -> Int:
    comptime if dtype == DType.float16:
        return W_F16
    else:
        return W_F32


def f32_accumulate[width: Int](v: SIMD[DType.float32, width]) -> Float32:
    """Horizontal sum of a f32 accumulation vector."""
    return Float32(v.reduce_add())


def load_f32[
    dtype: DType, width: Int
](t: Tensor[dtype, 2], offset: Int) -> SIMD[DType.float32, width]:
    """Widen a `width`-lane load of `t` to f32."""
    return (
        t.data().unsafe_load[width=width](offset=offset).cast[DType.float32]()
    )


def store_scalar[
    dtype: DType
](mut t: Tensor[dtype, 2], offset: Int, v: Float32):
    t.set(offset, Scalar[dtype](v))


def dot_product[
    dtype: DType
](
    a: Tensor[dtype, 2],
    b: Tensor[dtype, 2],
    a_base: Int,
    b_base: Int,
    k: Int,
) -> Float32:
    """Dot product of a[a_base : a_base+k] and b[b_base : b_base+k] in f32.

    This is the shared inner loop of every projection kernel; using it
    keeps the SIMD bookkeeping (main loop, tail, f32 accumulate) in one
    place.
    """
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var k_main = (k // W) * W
    var acc = SIMD[DType.float32, W](0)
    var i = 0
    while i < k_main:
        var av = (
            a.data()
            .unsafe_load[width=W](offset=a_base + i)
            .cast[DType.float32]()
        )
        var bv = (
            b.data()
            .unsafe_load[width=W](offset=b_base + i)
            .cast[DType.float32]()
        )
        acc = acc + av * bv
        i += W
    var total = Float32(acc.reduce_add())
    while i < k:
        total += Float32(a.get(a_base + i)) * Float32(b.get(b_base + i))
        i += 1
    return total


def elementwise_sum_squares[
    dtype: DType, width: Int = 0
](t: Tensor[dtype, 2], base: Int, n: Int) -> Float32:
    """sum of squares of t[base : base+n], computed in f32.

    `width` is the SIMD lane count (comptime); 0 selects the legacy
    per-dtype width (8 lanes f16 / 4 lanes f32).
    """
    comptime W = width if width > 0 else (
        W_F16 if dtype == DType.float16 else W_F32
    )
    var n_main = (n // W) * W
    var acc = SIMD[DType.float32, W](0)
    var i = 0
    while i < n_main:
        var v = (
            t.data().unsafe_load[width=W](offset=base + i).cast[DType.float32]()
        )
        acc = acc + v * v
        i += W
    var total = Float32(acc.reduce_add())
    while i < n:
        var v = Float32(t.get(base + i))
        total += v * v
        i += 1
    return total


# -- M8: adaptive SIMD width ---------------------------------------------------


def is_power_of_two(n: Int) -> Bool:
    """True when n > 0 and n & (n-1) == 0."""
    return n > 0 and (n & (n - 1)) == 0


def get_optimal_simd_width(dim: Int, is_power_of_two: Bool) -> Int:
    """Pick the SIMD bit width (64/128/256) for a row of `dim` elements.

    Evaluable in a comptime context (constexpr-style), so the result can
    drive a comptime kernel parameter:

        comptime bits = get_optimal_simd_width(K, is_power_of_two(K))
        comptime W = simd_lanes(bits, dtype)

    Rules:
      * powers of two: the widest candidate that fits the row.  A 256-bit
        vector holds 16 f16 / 8 f32 elements, so the fit bounds are
        `dim >= 16` -> 256, `dim >= 8` -> 128, else 64 (valid for both
        dtypes);
      * other dims: the widest candidate that divides the row exactly
        (no scalar tail), falling back to 64-bit.
    """
    if dim < 4:
        return SIMD_WIDTH_64
    if is_power_of_two:
        if dim >= 16:
            return SIMD_WIDTH_256
        if dim >= 8:
            return SIMD_WIDTH_128
        return SIMD_WIDTH_64
    if dim >= 16 and dim % 16 == 0:
        return SIMD_WIDTH_256
    if dim >= 8 and dim % 8 == 0:
        return SIMD_WIDTH_128
    return SIMD_WIDTH_64


def simd_lanes(bit_width: Int, dtype: DType) -> Int:
    """Map a SIMD bit width (64/128/256) to the lane count for `dtype`.

    Comptime-evaluable (a plain `if` on the arguments: the call is
    constant-folded when both are compile-time constants, and `comptime
    if` would reject the runtime-typed `dtype` parameter); the result is
    the `width` parameter of `SIMD[dtype, W]` / `unsafe_load[width=W]`
    in the kernels.
    """
    if dtype == DType.float16:
        if bit_width == SIMD_WIDTH_64:
            return 4
        if bit_width == SIMD_WIDTH_256:
            return 16
        return 8
    else:
        if bit_width == SIMD_WIDTH_64:
            return 2
        if bit_width == SIMD_WIDTH_256:
            return 8
        return 4


struct AutotuneResult(Movable):
    """The micro-benchmark outcome for one row length.

    `sink` is the accumulated benchmark value: the caller must use it
    (print it) or the compiler deletes the whole timed loop.
    """

    var dim: Int
    var best: Int  # winning bit width (64/128/256)
    var ns64: Int64
    var ns128: Int64
    var ns256: Int64
    var sink: Float32

    def __init__(
        out self,
        dim: Int,
        best: Int,
        ns64: Int64,
        ns128: Int64,
        ns256: Int64,
        sink: Float32,
    ):
        self.dim = dim
        self.best = best
        self.ns64 = ns64
        self.ns128 = ns128
        self.ns256 = ns256
        self.sink = sink


def _time_width[
    dtype: DType, width: Int
](t: Tensor[dtype, 2], n: Int, iters: Int) -> Tuple[Int64, Float32]:
    """Time `iters` passes of the canonical inner loop at `width` lanes.

    The read window rotates per iteration so the loop body depends on the
    iteration index and cannot be hoisted or constant-folded away.  The
    accumulated `sink` is returned (not discarded): a dead local is
    optimized out together with the whole loop, so the caller must keep
    the value observable.
    """
    var sink = Float32(0)
    var t0 = perf_counter_ns()
    for it in range(iters):
        var off = (it % 3) * (n // 4)
        sink += elementwise_sum_squares[dtype, width](t, off, n - off)
    var t1 = perf_counter_ns()
    return (Int64(t1 - t0), sink)


def autotune_width_f16(dim: Int, iters: Int = 300) -> AutotuneResult:
    """Benchmark the 64/128/256-bit f16 widths on a scratch row of `dim`
    elements and return the per-width timings plus the fastest width."""
    var t = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, dim))
    for i in range(dim):
        t.set(i, Scalar[DType.float16](Float32(i % 97) * Float32(0.01)))
    var r64 = _time_width[DType.float16, 4](t, dim, iters)
    var r128 = _time_width[DType.float16, 8](t, dim, iters)
    var r256 = _time_width[DType.float16, 16](t, dim, iters)
    var best = SIMD_WIDTH_128
    var best_ns = r128[0]
    if r64[0] < best_ns:
        best = SIMD_WIDTH_64
        best_ns = r64[0]
    if r256[0] < best_ns:
        best = SIMD_WIDTH_256
    return AutotuneResult(
        dim, best, r64[0], r128[0], r256[0], r64[1] + r128[1] + r256[1]
    )


def autotune_width_f32(dim: Int, iters: Int = 300) -> AutotuneResult:
    """The f32 counterpart of `autotune_width_f16` (2/4/8 lanes)."""
    var t = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](1, dim))
    for i in range(dim):
        t.set(i, Scalar[DType.float32](Float32(i % 97) * Float32(0.01)))
    var r64 = _time_width[DType.float32, 2](t, dim, iters)
    var r128 = _time_width[DType.float32, 4](t, dim, iters)
    var r256 = _time_width[DType.float32, 8](t, dim, iters)
    var best = SIMD_WIDTH_128
    var best_ns = r128[0]
    if r64[0] < best_ns:
        best = SIMD_WIDTH_64
        best_ns = r64[0]
    if r256[0] < best_ns:
        best = SIMD_WIDTH_256
    return AutotuneResult(
        dim, best, r64[0], r128[0], r256[0], r64[1] + r128[1] + r256[1]
    )


struct AutotuneCache(Movable):
    """Runtime cache of autotuned SIMD widths, keyed by row length.

    `autotune_f16` / `autotune_f32` benchmark once per dim and remember
    the winner; `get` returns the cached winner, or the comptime heuristic
    (`get_optimal_simd_width`) when the dim has not been benchmarked.
    The cache is a struct (no runtime globals): the Interpreter and the
    CLI each own an instance.
    """

    var widths: Dict[String, Int]  # "simd/<dim>" -> bit width

    def __init__(out self):
        self.widths = Dict[String, Int]()

    def _key(self, dim: Int) -> String:
        return "simd/" + String(dim)

    def get(self, dim: Int) -> Int:
        var w = self.widths.get(self._key(dim), 0)
        if w > 0:
            return w
        return get_optimal_simd_width(dim, is_power_of_two(dim))

    def autotune_f16(mut self, dim: Int, iters: Int = 300) -> Int:
        var w = self.widths.get(self._key(dim), 0)
        if w > 0:
            return w
        var r = autotune_width_f16(dim, iters)
        var best = r.best
        self.widths[self._key(dim)] = best
        return best

    def autotune_f32(mut self, dim: Int, iters: Int = 300) -> Int:
        var w = self.widths.get(self._key(dim), 0)
        if w > 0:
            return w
        var r = autotune_width_f32(dim, iters)
        var best = r.best
        self.widths[self._key(dim)] = best
        return best
