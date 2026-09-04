# tests/test_jit_cache.mojo
#
# M8: JIT shape specialization for the fused matmul+rmsnorm CPU op.
#
# Checks:
#   1. get_or_compile: a miss compiles + records the shape, a hit returns
#      the cached kernel; the stats (compiles/hits/misses/hit rate) track
#      the lookups;
#   2. the comptime-specialized kernel (shape/width/unroll/tile knobs)
#      matches the generic fused kernel;
#   3. run_fused_jit with specialize=False is exactly the generic path
#      (the default behavior); with specialize=True it matches the
#      generic kernel and records the model shape;
#   4. the legacy FFN cache contract (test_jit.mojo) is unchanged.

from src.core.jit.jit_cache import JitCache
from src.core.ops.cpu.matmul_cpu import (
    CompiledFusedKernel,
    compile_fused_kernel,
    fused_matmul_rms_norm_key,
)
from src.core.ops.fused.matmul_rms_norm import fused_matmul_rms_norm
from src.core.tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple


def _fill_f16(mut t: Tensor[DType.float16, 2], seed: Int, scale: Float32):
    for i in range(t.numel()):
        t.set(
            i,
            Scalar[DType.float16](
                Float32((i * 7 + seed * 13) % 101) / Float32(101.0) * scale
            ),
        )


def _max_diff(
    a: Tensor[DType.float16, 2], b: Tensor[DType.float16, 2]
) -> Float32:
    var mx = Float32(0)
    for i in range(a.numel()):
        var d = Float32(a.get(i)) - Float32(b.get(i))
        if d < Float32(0):
            d = -d
        if d > mx:
            mx = d
    return mx


def check(label: String, ok: Bool):
    if ok:
        print("[ok]   ", label)
    else:
        print("[FAIL] ", label)
        std.os.os.abort()


def main():
    print("== M8 JIT shape specialization tests ==")

    # 1. get_or_compile: miss -> compile, hit -> cached
    var cache = JitCache()
    var sig = fused_matmul_rms_norm_key(2, 256, 128)
    var k1 = cache.get_or_compile(sig, compile_fused_kernel, 2, 256, 128, 128)
    check(
        "miss compiles and records the shape",
        cache.compiles == 1
        and cache.misses == 1
        and cache.hits == 0
        and cache.has(sig),
    )
    var k2 = cache.get_or_compile(sig, compile_fused_kernel, 2, 256, 128, 128)
    check(
        "hit returns the cached kernel",
        cache.hits == 1
        and cache.compiles == 1
        and k1.m == k2.m
        and k1.n == k2.n
        and k1.k == k2.k
        and k1.width == k2.width,
    )

    # 2. the specialized kernel matches the generic fused op
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](2, 128))
    var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](256, 128))
    _fill_f16(x, 1, Float32(1.0))
    _fill_f16(w, 2, Float32(0.05))
    var legacy = fused_matmul_rms_norm[DType.float16](x, w)
    var got = k1.run(x, w, Float32(1e-5))
    check(
        "specialized (2,256,128) matches generic",
        _max_diff(legacy, got) < Float32(1e-2),
    )

    # all three SIMD widths of the same shape (the (2,256,128) entry
    # also exercises the UNROLL=2 / TILE knobs)
    var wbits = List[Int]()
    wbits.append(64)
    wbits.append(128)
    wbits.append(256)
    for wi in range(len(wbits)):
        var wb = wbits[wi]
        var kk = CompiledFusedKernel(
            fused_matmul_rms_norm_key(2, 256, 128), 2, 256, 128, wb
        )
        var gg = kk.run(x, w, Float32(1e-5))
        check(
            "width " + String(wb) + " matches generic",
            _max_diff(legacy, gg) < Float32(1e-2),
        )

    # 3. run_fused_jit: the model FFN shape (1, 8960, 1536)
    var x1 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, 1536))
    var w1 = tensor_zeros[DType.float16, 2](
        StaticTuple[Int, 2](8960, 1536)
    )
    _fill_f16(x1, 3, Float32(1.0))
    _fill_f16(w1, 4, Float32(0.02))
    var ref1 = fused_matmul_rms_norm[DType.float16](x1, w1)
    var off = cache.run_fused_jit(x1, w1, Float32(1e-5), False, 128)
    check("specialize off == generic (default path)", _max_diff(ref1, off) == Float32(0))
    var on = cache.run_fused_jit(x1, w1, Float32(1e-5), True, 128)
    check(
        "specialize on matches generic (model shape)",
        _max_diff(ref1, on) < Float32(2e-2),
    )
    check(
        "model shape recorded in the cache",
        cache.has(fused_matmul_rms_norm_key(1, 8960, 1536)),
    )
    var on2 = cache.run_fused_jit(x1, w1, Float32(1e-5), True, 256)
    check(
        "256-bit model-shape run matches generic",
        _max_diff(ref1, on2) < Float32(2e-2),
    )
    check("second model-shape run was a cache hit", cache.hits >= 2)

    # an off-table shape: miss -> generic fallback, recorded
    var x2 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](2, 64))
    var w2 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](128, 64))
    _fill_f16(x2, 5, Float32(1.0))
    _fill_f16(w2, 6, Float32(0.02))
    var ref2 = fused_matmul_rms_norm[DType.float16](x2, w2)
    var off2 = cache.run_fused_jit(x2, w2, Float32(1e-5), True, 128)
    check(
        "off-table shape falls back to the generic kernel",
        _max_diff(ref2, off2) == Float32(0)
        and cache.has(fused_matmul_rms_norm_key(2, 128, 64)),
    )

    # 4. stats + the legacy FFN contract
    print(
        "  stats: compiles", cache.compiles, "hits", cache.hits,
        "misses", cache.misses, "hit rate", cache.hit_rate(),
    )
    check(
        "hit rate in [0,1]",
        cache.hit_rate() >= Float64(0.0)
        and cache.hit_rate() <= Float64(1.0),
    )
    check(
        "legacy FFN key stays pre-registered",
        cache.has("ffn/1/8960/1536"),
    )

    print("== JIT shape specialization tests passed ==")
