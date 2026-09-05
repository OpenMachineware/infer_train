# tests/test_simd_utils.mojo
#
# M8: SIMD adaptive width.
#
# Checks:
#   1. the comptime width-selection heuristic (powers of two get the
#      widest fitting candidate; other dims get the widest divisor);
#   2. the bit-width -> lane-count map, evaluated at comptime;
#   3. the width-parameterized elementwise kernels (rms_norm / add /
#      swiglu) match the legacy default-width results at 64/128/256-bit;
#   4. the runtime autotuner returns a valid candidate and the cache
#      remembers the winner (heuristic fallback for un-benchmarked dims).

from src.core.simd_utils import (
    AutotuneCache,
    autotune_width_f16,
    autotune_width_f32,
    get_optimal_simd_width,
    is_power_of_two,
    simd_lanes,
)
from src.core.ops.cpu.rms_norm_cpu import (
    rms_norm_cpu_autotuned,
    rms_norm_cpu_dynamic,
)
from src.core.ops.cpu.add_cpu import add_cpu_autotuned, add_cpu_dynamic
from src.core.ops.cpu.swiglu_cpu import swiglu_cpu_autotuned, swiglu_cpu_dynamic
from src.core.tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple


def _fill_f16(mut t: Tensor[DType.float16, 2], seed: Int):
    for i in range(t.numel()):
        t.set(
            i,
            Scalar[DType.float16](
                Float32((i * 7 + seed * 13) % 101) / Float32(101.0)
            ),
        )


def _max_diff_f16(
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
    print("== M8 SIMD adaptive width tests ==")

    # 1. the comptime heuristic
    check(
        "is_power_of_two",
        is_power_of_two(1)
        and is_power_of_two(2)
        and is_power_of_two(1024)
        and not is_power_of_two(0)
        and not is_power_of_two(1536)
        and not is_power_of_two(1537),
    )
    check("pow2: 16 -> 256", get_optimal_simd_width(16, True) == 256)
    check("pow2: 8 -> 128", get_optimal_simd_width(8, True) == 128)
    check("pow2: 4 -> 64", get_optimal_simd_width(4, True) == 64)
    check("pow2: 2 -> 64", get_optimal_simd_width(2, True) == 64)
    check(
        "1536 -> 256",
        get_optimal_simd_width(1536, is_power_of_two(1536)) == 256,
    )
    check(
        "8960 -> 256",
        get_optimal_simd_width(8960, is_power_of_two(8960)) == 256,
    )
    check("24 -> 128", get_optimal_simd_width(24, False) == 128)
    check("12 -> 64", get_optimal_simd_width(12, False) == 64)
    check("6 -> 64", get_optimal_simd_width(6, False) == 64)
    check("3 -> 64", get_optimal_simd_width(3, False) == 64)

    # comptime evaluation of the heuristic + the lane map
    comptime cw = get_optimal_simd_width(1536, True)
    comptime lanes = simd_lanes(cw, DType.float16)
    comptime lanes32 = simd_lanes(cw, DType.float32)
    check(
        "comptime eval: 1536 -> 256-bit -> 16 f16 / 8 f32 lanes",
        cw == 256 and lanes == 16 and lanes32 == 8,
    )

    # 2. width-parameterized kernels match the legacy default width
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](3, 1536))
    _fill_f16(x, 1)
    var legacy = rms_norm_cpu_dynamic[DType.float16](x)
    check(
        "rms_norm: 64/128/256-bit match legacy",
        _max_diff_f16(legacy, rms_norm_cpu_autotuned[DType.float16](x, 64))
        < Float32(1e-3)
        and _max_diff_f16(legacy, rms_norm_cpu_autotuned[DType.float16](x, 128))
        < Float32(1e-3)
        and _max_diff_f16(legacy, rms_norm_cpu_autotuned[DType.float16](x, 256))
        < Float32(1e-3),
    )

    var a = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](2, 1000))
    var b = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](2, 1000))
    _fill_f16(a, 2)
    _fill_f16(b, 3)
    var aref = add_cpu_dynamic[DType.float16](a, b)
    check(
        "add: 64/128/256-bit match legacy",
        _max_diff_f16(aref, add_cpu_autotuned[DType.float16](a, b, 64))
        < Float32(1e-6)
        and _max_diff_f16(aref, add_cpu_autotuned[DType.float16](a, b, 128))
        < Float32(1e-6)
        and _max_diff_f16(aref, add_cpu_autotuned[DType.float16](a, b, 256))
        < Float32(1e-6),
    )

    var g = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](2, 1000))
    var u = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](2, 1000))
    _fill_f16(g, 4)
    _fill_f16(u, 5)
    var sref = swiglu_cpu_dynamic[DType.float16](g, u)
    check(
        "swiglu: 64/128/256-bit match legacy",
        _max_diff_f16(sref, swiglu_cpu_autotuned[DType.float16](g, u, 64))
        < Float32(1e-3)
        and _max_diff_f16(sref, swiglu_cpu_autotuned[DType.float16](g, u, 128))
        < Float32(1e-3)
        and _max_diff_f16(sref, swiglu_cpu_autotuned[DType.float16](g, u, 256))
        < Float32(1e-3),
    )

    # f32 widths (2/4/8 lanes), odd length to exercise the scalar tail
    var xf = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 999))
    var bf = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 999))
    for i in range(xf.numel()):
        xf.set(i, Scalar[DType.float32](Float32(i % 50) * Float32(0.01)))
        bf.set(i, Scalar[DType.float32](Float32(i % 37) * Float32(0.01)))
    var aref32 = add_cpu_dynamic[DType.float32](xf, bf)
    var maxd = Float32(0)
    var widths = List[Int]()
    widths.append(64)
    widths.append(128)
    widths.append(256)
    for wi in range(len(widths)):
        var got = add_cpu_autotuned[DType.float32](xf, bf, widths[wi])
        for i in range(aref32.numel()):
            var d = Float32(aref32.get(i)) - Float32(got.get(i))
            if d < Float32(0):
                d = -d
            if d > maxd:
                maxd = d
    check(
        "f32 add: 64/128/256-bit match legacy (odd len)", maxd < Float32(1e-6)
    )

    # 3. the runtime autotuner
    var r = autotune_width_f16(1536)
    check(
        "autotune f16 returns a candidate",
        r.best == 64 or r.best == 128 or r.best == 256,
    )
    # r.sink is printed so the compiler keeps the timed loops alive
    print(
        "  autotune f16 dim=1536: 64-bit",
        r.ns64,
        "ns, 128-bit",
        r.ns128,
        "ns, 256-bit",
        r.ns256,
        "ns -> best",
        r.best,
        "bit",
        "(sink",
        r.sink,
        ")",
    )
    var r32 = autotune_width_f32(1536)
    check(
        "autotune f32 returns a candidate",
        r32.best == 64 or r32.best == 128 or r32.best == 256,
    )
    print(
        "  autotune f32 dim=1536: 64-bit",
        r32.ns64,
        "ns, 128-bit",
        r32.ns128,
        "ns, 256-bit",
        r32.ns256,
        "ns -> best",
        r32.best,
        "bit (sink",
        r32.sink,
        ")",
    )

    var cache = AutotuneCache()
    var w1 = cache.autotune_f16(1536)
    var w2 = cache.autotune_f16(1536)
    check("cache remembers the winner", w1 == w2)
    check("cache get == autotuned width", cache.get(1536) == w1)
    check(
        "cache get falls back to the heuristic",
        cache.get(24) == get_optimal_simd_width(24, False),
    )

    print("== SIMD adaptive width tests passed ==")
