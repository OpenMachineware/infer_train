# tests/test_kv_cache_m7.mojo
#
# M7 KV-cache features: paged attention round-trip, sliding-window
# semantics, and context-length adaptation.

from src.core.ops.attention.kv_cache import KVCache, KVCacheLayer, kv_cache_append
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple


def main():
    # dense round-trip (regression)
    var dense = KVCacheLayer(2, 8, 4)
    var k1 = tensor_zeros[DType.float16, 3](StaticTuple[Int, 3](2, 1, 4))
    var v1 = tensor_zeros[DType.float16, 3](StaticTuple[Int, 3](2, 1, 4))
    for h in range(2):
        for d in range(4):
            k1.set(h * 4 + d, Scalar[DType.float16](Float32(h * 10 + d + 1)))
            v1.set(h * 4 + d, Scalar[DType.float16](Float32(h * 100 + d + 2)))
    print("A")
    kv_cache_append[DType.float16](dense, k1, v1, 3)
    print("B")
    check(dense.filled == 4, "dense filled")
    check(dense.get_k(1, 3, 2) == Float32(13.0), "dense k read")
    check(dense.get_v(0, 3, 0) == Float32(2.0), "dense v read")

    # paged round-trip: 2 kv heads, max_len 10, head_dim 4, page 4
    print("C")
    var paged = KVCacheLayer(2, 10, 4)
    paged.enable_paged(4, 2, 4)
    print("D")
    for pos in range(10):
        kv_cache_append[DType.float16](paged, k1, v1, pos)
    print("E")
    check(paged.filled == 10, "paged filled")
    check(paged.get_k(1, 0, 2) == Float32(13.0), "paged k page0")
    check(paged.get_k(1, 5, 2) == Float32(13.0), "paged k page1")
    check(paged.get_v(0, 9, 0) == Float32(2.0), "paged v last page")

    # sliding window: window 4 -> ring reuse; first_position = filled - 4
    print("F")
    var sw = KVCacheLayer(2, 8, 4)
    sw.set_window(4)
    for pos in range(8):
        var kk = tensor_zeros[DType.float16, 3](StaticTuple[Int, 3](2, 1, 4))
        for d in range(4):
            kk.set(d, Scalar[DType.float16](Float32(pos + d + 1)))
        var vv = tensor_zeros[DType.float16, 3](StaticTuple[Int, 3](2, 1, 4))
        kv_cache_append[DType.float16](sw, kk, vv, pos)
    check(sw.filled == 8, "sw filled")
    check(sw.first_position() == 4, "sw first position")
    check(sw.storage_pos(5) == 1, "sw ring mapping")
    # position 4 was written after position 0 -> slot 0 holds pos 4's data
    check(sw.get_k(0, 4, 0) == Float32(5.0), "sw ring value")

    # context-length adaptation to a memory budget
    print("G")
    var cache = KVCache(2, 2, 64, 4)
    var before = cache.kv_cache_bytes()
    var new_len = cache.adjust_capacity(before // 2)
    check(new_len < 64, "adapted context shrinks")
    check(cache.kv_cache_bytes() <= before // 2 + before // 16, "adapted bytes")
    print("test_kv_cache_m7 OK")


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
