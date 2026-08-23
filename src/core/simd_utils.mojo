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

from .tensor import Tensor

comptime W_F16 = 8
comptime W_F32 = 4


def simd_width[dtype: DType]() -> Int:
    comptime if dtype == DType.float16:
        return W_F16
    else:
        return W_F32


def f32_accumulate[width: Int](v: SIMD[DType.float32, width]) -> Float32:
    """Horizontal sum of a f32 accumulation vector."""
    return Float32(v.reduce_add())


def load_f32[dtype: DType, width: Int](
    t: Tensor[dtype, 2], offset: Int
) -> SIMD[DType.float32, width]:
    """Widen a `width`-lane load of `t` to f32."""
    return t.data().unsafe_load[width=width](
        offset=offset
    ).cast[DType.float32]()


def store_scalar[dtype: DType](mut t: Tensor[dtype, 2], offset: Int, v: Float32):
    t.set(offset, Scalar[dtype](v))


def dot_product[dtype: DType](
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
        var av = a.data().unsafe_load[width=W](
            offset=a_base + i
        ).cast[DType.float32]()
        var bv = b.data().unsafe_load[width=W](
            offset=b_base + i
        ).cast[DType.float32]()
        acc = acc + av * bv
        i += W
    var total = Float32(acc.reduce_add())
    while i < k:
        total += Float32(a.get(a_base + i)) * Float32(b.get(b_base + i))
        i += 1
    return total


def elementwise_sum_squares[dtype: DType](
    t: Tensor[dtype, 2], base: Int, n: Int
) -> Float32:
    """sum of squares of t[base : base+n], computed in f32."""
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var n_main = (n // W) * W
    var acc = SIMD[DType.float32, W](0)
    var i = 0
    while i < n_main:
        var v = t.data().unsafe_load[width=W](
            offset=base + i
        ).cast[DType.float32]()
        acc = acc + v * v
        i += W
    var total = Float32(acc.reduce_add())
    while i < n:
        var v = Float32(t.get(base + i))
        total += v * v
        i += 1
    return total
