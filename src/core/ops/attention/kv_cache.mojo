# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/attention/kv_cache.mojo
#
# KV cache for autoregressive decoding.
#
# M3: single-sequence, preallocated (unchanged API).
# M7 adds, on the same KVCacheLayer:
#   * Paged attention: the cache can be backed by fixed-size blocks
#     (page_size tokens per block) with a block table; allocation is
#     block-granular so the cache grows without reallocation and without
#     fragmentation between sequences.
#   * Sliding window: `window > 0` keeps only the last `window` positions
#     addressable (the attention mask skips older ones; the storage is
#     reused ring-buffer style when window < max_len).
#   * Context-length adaptation: `KVCache.adjust_capacity(...)` rebuilds
#     the cache to fit a memory budget (see `kv_cache_bytes`).
#
# Quantized storage (--kv-cache-type, off by default):
#   * `KVCacheType` selects the resident format: FP16 (default, unchanged
#     behavior), Q4_0 or Q8_0.
#   * In Q4/Q8 modes the fp16 `k`/`v` tensors stay empty and the cache
#     holds packed rows in `kq`/`vq` (uint8): 32-element blocks with an
#     fp16 scale - the llama.cpp Q4_0/Q8_0 layout, bit-exact with
#     ops/quantized/dequantize.mojo.  Rows are written through
#     `set_kv_row` (which quantizes) and read back through
#     `get_k_row`/`get_v_row` (which dequantize to fp16); the per-element
#     `set_kv`/`get_k`/`get_v` keep working (the getters dequantize a row
#     on demand).
#
# The MHA kernel resolves every read/write through `get_k`/`get_v`/
# `set_kv`/`set_kv_row`, so both dense and paged layouts are transparent
# to callers.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple

comptime DEFAULT_PAGE_SIZE = 64

# Quantized KV storage: 32-element blocks (llama.cpp Q4_0/Q8_0 layout,
# identical to ops/quantized/dequantize.mojo):
#   Q4_0 block = 18 bytes: d(2, fp16) + qs(16, 2 quants/byte);
#                     value = d * (q - 8),  q = unsigned 4-bit
#   Q8_0 block = 34 bytes: d(2, fp16) + qs(32, int8);
#                     value = d * q,        q = signed int8
comptime KV_QK = 32
comptime KV_Q4_BLOCK_BYTES = 18
comptime KV_Q8_BLOCK_BYTES = 34


# ---------------------------------------------------------------------------
# Storage type (Mojo 1.0: comptime-member enum, the QuantFormat pattern)
# ---------------------------------------------------------------------------


struct KVCacheType(Copyable, Equatable, ImplicitlyCopyable, Movable):
    """KV cache resident format: FP16 (default) / Q4_0 / Q8_0."""

    var _tag: Int8

    def __init__(out self, tag: Int8):
        self._tag = tag

    comptime FP16 = KVCacheType(Int8(0))
    comptime Q4_0 = KVCacheType(Int8(1))
    comptime Q8_0 = KVCacheType(Int8(2))

    def __eq__(self, other: Self) -> Bool:
        return self._tag == other._tag

    def __ne__(self, other: Self) -> Bool:
        return self._tag != other._tag


def kv_cache_type_from_str(s: String) -> KVCacheType:
    """Parse the --kv-cache-type flag value (fp16 / q4_0 / q8_0)."""
    if s == "fp16":
        return KVCacheType.FP16
    if s == "q4_0":
        return KVCacheType.Q4_0
    if s == "q8_0":
        return KVCacheType.Q8_0
    unimplemented("kv_cache_type_from_str: unknown type " + s)
    return KVCacheType.FP16


# ---------------------------------------------------------------------------
# Quantized row kernels (Q4_0 / Q8_0)
# ---------------------------------------------------------------------------
#
# One row is [head_dim] fp16 values = a sequence of full 32-element blocks;
# the last block may hold fewer values (the unused qs bytes stay zero and
# are never read back).  Quantization follows llama.cpp's
# quantize_row_ref_q4_0 / quantize_row_ref_q8_0 (nearest-int rounding):
#   Q4_0: d = amax / -7;   q = clamp(round(x / d) + 8, 0, 15)
#   Q8_0: d = amax / 127;  q = clamp(round(x / d), -127, 127)
# so the dequantized value is `d * (q - 8)` / `d * q`, matching the GGUF
# dequantizers bit-for-bit.


def _block_amax(src: Tensor[DType.float16, 1], start: Int, n: Int) -> Float32:
    """max |x| over src[start .. start+n)."""
    var amax = Float32(0)
    for j in range(n):
        var v = Float32(src.get(start + j))
        if v < Float32(0):
            v = -v
        if v > amax:
            amax = v
    return amax


def _store_f16(dst: Tensor[DType.uint8, 1], off: Int, value: Float32):
    """Write one fp16 (little-endian, 2 bytes) into a uint8 buffer."""
    var p = dst.data().unsafe_offset(off).unsafe_bitcast[
        Scalar[DType.float16]
    ]()
    p.unsafe_store(0, Scalar[DType.float16](value))


def _load_f16(src: Tensor[DType.uint8, 1], off: Int) -> Float32:
    """Read one fp16 (little-endian, 2 bytes) from a uint8 buffer."""
    var p = src.data().unsafe_offset(off).unsafe_bitcast[Scalar[DType.float16]]()
    return Float32(p.unsafe_load[width=1](offset=0))


def quantize_row_q4_0(
    src: Tensor[DType.float16, 1],
    dst: Tensor[DType.uint8, 1],
    dst_off: Int,
):
    """Quantize one [head_dim] fp16 row into Q4_0 blocks at dst+dst_off.

    `dst` must hold at least `(numel + 31) // 32 * 18` bytes from
    `dst_off`.  The scale is derived per 32-element block (d = amax / -7),
    the inverse of `_dequantize_q4_0_block` in ops/quantized/dequantize.
    """
    var n = src.numel()
    var start = 0
    var off = dst_off
    while start < n:
        var cnt = KV_QK
        if n - start < cnt:
            cnt = n - start
        var amax = _block_amax(src, start, cnt)
        var d = amax / Float32(-7.0)
        var id = Float32(0)
        if d != Float32(0):
            id = Float32(1.0) / d
        _store_f16(dst, off, d)
        var qs = dst.data().unsafe_offset(off + 2)
        for j in range(16):
            var lo = 0
            var hi = 0
            var i0 = 2 * j
            var i1 = 2 * j + 1
            if i0 < cnt:
                var q = Int(round(Float32(src.get(start + i0)) * id)) + 8
                if q > 15:
                    q = 15
                if q < 0:
                    q = 0
                lo = q
            if i1 < cnt:
                var q = Int(round(Float32(src.get(start + i1)) * id)) + 8
                if q > 15:
                    q = 15
                if q < 0:
                    q = 0
                hi = q
            qs.unsafe_store(j, UInt8(lo | (hi << 4)))
        off += KV_Q4_BLOCK_BYTES
        start += KV_QK


def dequantize_row_q4_0(
    src: Tensor[DType.uint8, 1],
    src_off: Int,
    dst: Tensor[DType.float16, 1],
):
    """Dequantize Q4_0 blocks at src+src_off into `dst` (fp16).

    Inverse of `quantize_row_q4_0`; the decode is the same as
    `_dequantize_q4_0_block` (value = d * (q - 8), FP32 arithmetic).
    """
    var n = dst.numel()
    var start = 0
    var off = src_off
    while start < n:
        var cnt = KV_QK
        if n - start < cnt:
            cnt = n - start
        var d = _load_f16(src, off)
        var qs = src.data().unsafe_offset(off + 2)
        for j in range(16):
            var b = Int(qs.unsafe_load[width=1](offset=j))
            var i0 = 2 * j
            var i1 = 2 * j + 1
            if i0 < cnt:
                dst.set(
                    start + i0,
                    Scalar[DType.float16](d * (Float32(b & 0xF) - Float32(8))),
                )
            if i1 < cnt:
                dst.set(
                    start + i1,
                    Scalar[DType.float16](d * (Float32(b >> 4) - Float32(8))),
                )
        off += KV_Q4_BLOCK_BYTES
        start += KV_QK


def quantize_row_q8_0(
    src: Tensor[DType.float16, 1],
    dst: Tensor[DType.uint8, 1],
    dst_off: Int,
):
    """Quantize one [head_dim] fp16 row into Q8_0 blocks at dst+dst_off.

    `dst` must hold at least `(numel + 31) // 32 * 34` bytes from
    `dst_off`.  The scale is derived per 32-element block (d = amax / 127),
    the inverse of `_dequantize_q8_0_block` in ops/quantized/dequantize.
    """
    var n = src.numel()
    var start = 0
    var off = dst_off
    while start < n:
        var cnt = KV_QK
        if n - start < cnt:
            cnt = n - start
        var amax = _block_amax(src, start, cnt)
        var d = amax / Float32(127.0)
        var id = Float32(0)
        if d != Float32(0):
            id = Float32(1.0) / d
        _store_f16(dst, off, d)
        var qs = dst.data().unsafe_offset(off + 2)
        for j in range(KV_QK):
            var q = 0
            if j < cnt:
                q = Int(round(Float32(src.get(start + j)) * id))
                if q > 127:
                    q = 127
                if q < -127:
                    q = -127
            qs.unsafe_store(j, UInt8(q & 0xFF))
        off += KV_Q8_BLOCK_BYTES
        start += KV_QK


def dequantize_row_q8_0(
    src: Tensor[DType.uint8, 1],
    src_off: Int,
    dst: Tensor[DType.float16, 1],
):
    """Dequantize Q8_0 blocks at src+src_off into `dst` (fp16).

    Inverse of `quantize_row_q8_0`; the decode is the same as
    `_dequantize_q8_0_block` (value = d * q, FP32 arithmetic).
    """
    var n = dst.numel()
    var start = 0
    var off = src_off
    while start < n:
        var cnt = KV_QK
        if n - start < cnt:
            cnt = n - start
        var d = _load_f16(src, off)
        var qs = src.data().unsafe_offset(off + 2)
        for j in range(KV_QK):
            if j < cnt:
                var q = Int(qs.unsafe_load[width=1](offset=j))
                if q > 127:
                    q -= 256
                dst.set(start + j, Scalar[DType.float16](d * Float32(q)))
        off += KV_Q8_BLOCK_BYTES
        start += KV_QK


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


struct KVCacheLayer(Copyable, Movable):
    """One layer's K/V cache: [n_kv_heads, max_len, head_dim] fp16 each.

    M7: when `page_size > 0` the storage is [n_blocks, n_kv_heads,
    page_size, head_dim] with an explicit block table - paged attention.
    `window > 0` enables the sliding window (attention only reads the last
    `window` positions).

    Quantized: when `kv_type` is Q4_0/Q8_0, `k`/`v` stay empty and the
    packed rows live in `kq`/`vq` (uint8, 32-element blocks with an fp16
    scale); `set_kv_row` quantizes on write, `get_k_row`/`get_v_row`
    dequantize to fp16 on read.
    """

    var k: Tensor[DType.float16, 3]
    var v: Tensor[DType.float16, 3]
    var kq: Tensor[DType.uint8, 1]  # packed K rows (Q4_0/Q8_0 modes)
    var vq: Tensor[DType.uint8, 1]  # packed V rows (Q4_0/Q8_0 modes)
    var max_len: Int
    var filled: Int  # number of valid positions (0..max_len)
    var page_size: Int  # 0 = dense; > 0 = paged
    var block_table: List[Int]  # position page -> block index (paged)
    var window: Int  # sliding window; 0 = unlimited
    var kv_type: KVCacheType  # FP16 (default) / Q4_0 / Q8_0
    var n_kv_heads: Int
    var head_dim: Int

    def __init__(
        out self,
        n_kv_heads: Int,
        max_len: Int,
        head_dim: Int,
        kv_type: KVCacheType = KVCacheType.FP16,
    ):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.kv_type = kv_type
        self.max_len = max_len
        self.filled = 0
        self.page_size = 0
        self.block_table = List[Int]()
        self.window = 0
        # NOTE: no self-method calls below (flow analysis: every field must
        # be initialized before `self` is used), so the layout is computed
        # locally from the constructor arguments.
        if kv_type != KVCacheType.FP16:
            var nb = (head_dim + KV_QK - 1) // KV_QK
            var bb: Int
            if kv_type == KVCacheType.Q4_0:
                bb = KV_Q4_BLOCK_BYTES
            else:
                bb = KV_Q8_BLOCK_BYTES
            var rb = nb * bb
            self.k = Tensor[DType.float16, 3](StaticTuple[Int, 3](0, 0, 0))
            self.v = Tensor[DType.float16, 3](StaticTuple[Int, 3](0, 0, 0))
            self.kq = tensor_zeros[DType.uint8, 1](
                StaticTuple[Int, 1](n_kv_heads * max_len * rb)
            )
            self.vq = tensor_zeros[DType.uint8, 1](
                StaticTuple[Int, 1](n_kv_heads * max_len * rb)
            )
        else:
            self.k = tensor_zeros[DType.float16, 3](
                StaticTuple[Int, 3](n_kv_heads, max_len, head_dim)
            )
            self.v = tensor_zeros[DType.float16, 3](
                StaticTuple[Int, 3](n_kv_heads, max_len, head_dim)
            )
            self.kq = Tensor[DType.uint8, 1](StaticTuple[Int, 1](0))
            self.vq = Tensor[DType.uint8, 1](StaticTuple[Int, 1](0))

    def __copyinit__(out self, existing: Self):
        """Deep copy (List is not implicitly copyable)."""
        self.k = existing.k
        self.v = existing.v
        self.kq = existing.kq
        self.vq = existing.vq
        self.max_len = existing.max_len
        self.filled = existing.filled
        self.page_size = existing.page_size
        self.block_table = List[Int]()
        for b in existing.block_table:
            self.block_table.append(b)
        self.window = existing.window
        self.kv_type = existing.kv_type
        self.n_kv_heads = existing.n_kv_heads
        self.head_dim = existing.head_dim

    def is_quantized(self) -> Bool:
        return self.kv_type != KVCacheType.FP16

    def row_bytes(self) -> Int:
        """Bytes per (head, position) row in the packed storage (0 for FP16)."""
        if self.kv_type == KVCacheType.FP16:
            return 0
        var nb = (self.head_dim + KV_QK - 1) // KV_QK
        if self.kv_type == KVCacheType.Q4_0:
            return nb * KV_Q4_BLOCK_BYTES
        return nb * KV_Q8_BLOCK_BYTES

    def enable_paged(mut self, page_size: Int, n_kv_heads: Int, head_dim: Int):
        """Switch this layer to paged storage (blocks of `page_size`)."""
        if page_size < 1:
            unimplemented("KVCacheLayer.enable_paged: bad page size")
        var n_blocks = (self.max_len + page_size - 1) // page_size
        self.page_size = page_size
        if self.is_quantized():
            var rb = self.row_bytes()
            self.kq = tensor_zeros[DType.uint8, 1](
                StaticTuple[Int, 1](n_blocks * n_kv_heads * page_size * rb)
            )
            self.vq = tensor_zeros[DType.uint8, 1](
                StaticTuple[Int, 1](n_blocks * n_kv_heads * page_size * rb)
            )
        else:
            self.k = tensor_zeros[DType.float16, 3](
                StaticTuple[Int, 3](n_blocks, n_kv_heads, page_size * head_dim)
            )
            self.v = tensor_zeros[DType.float16, 3](
                StaticTuple[Int, 3](n_blocks, n_kv_heads, page_size * head_dim)
            )
        self.block_table = List[Int]()
        for b in range(n_blocks):
            self.block_table.append(b)

    def set_window(mut self, window: Int):
        self.window = window

    def reset(mut self):
        self.filled = 0

    # -- storage accessors (dense or paged, transparent to the MHA) ----------

    def storage_pos(self, position: Int) -> Int:
        """Physical slot for a logical `position` (ring reuse under SWA)."""
        if self.window > 0 and self.window < self.max_len:
            return position % self.window
        if position < 0 or position >= self.max_len:
            unimplemented("KVCacheLayer: position out of range")
        return position

    def _quant_row_offset(self, head: Int, position: Int) -> Int:
        """Byte offset of the (head, position) row in the packed storage."""
        var rb = self.row_bytes()
        if self.page_size > 0:
            var page = position // self.page_size
            var off = position % self.page_size
            var block = self.block_table[page]
            return (block * self.n_kv_heads + head) * (self.page_size * rb) + off * rb
        var slot = self.storage_pos(position)
        return (head * self.max_len + slot) * rb

    def get_k(self, head: Int, position: Int, d: Int) -> Float32:
        if self.is_quantized():
            var row = tensor_zeros[DType.float16, 1](
                StaticTuple[Int, 1](self.head_dim)
            )
            self.get_k_row(head, position, row)
            return Float32(row.get(d))
        if self.page_size > 0:
            var page = position // self.page_size
            var off = position % self.page_size
            var block = self.block_table[page]
            return Float32(
                self.k.get(
                    (block * self.k.shape()[1] + head) * self.k.shape()[2]
                    + off * self.head_dim_of()
                    + d
                )
            )
        var slot = self.storage_pos(position)
        return Float32(
            self.k.get((head * self.max_len + slot) * self.k.shape()[2] + d)
        )

    def get_v(self, head: Int, position: Int, d: Int) -> Float32:
        if self.is_quantized():
            var row = tensor_zeros[DType.float16, 1](
                StaticTuple[Int, 1](self.head_dim)
            )
            self.get_v_row(head, position, row)
            return Float32(row.get(d))
        if self.page_size > 0:
            var page = position // self.page_size
            var off = position % self.page_size
            var block = self.block_table[page]
            return Float32(
                self.v.get(
                    (block * self.v.shape()[1] + head) * self.v.shape()[2]
                    + off * self.head_dim_of()
                    + d
                )
            )
        var slot = self.storage_pos(position)
        return Float32(
            self.v.get((head * self.max_len + slot) * self.v.shape()[2] + d)
        )

    def head_dim_of(self) -> Int:
        return self.head_dim

    def set_kv(
        mut self, head: Int, position: Int, d: Int, kv: Float32, vv: Float32
    ):
        if self.is_quantized():
            unimplemented(
                "KVCacheLayer.set_kv: quantized caches store whole rows - "
                + "use set_kv_row"
            )
        if self.page_size > 0:
            var page = position // self.page_size
            var off = position % self.page_size
            var block = self.block_table[page]
            self.k.set(
                (block * self.k.shape()[1] + head) * self.k.shape()[2]
                + off * self.head_dim_of()
                + d,
                Scalar[DType.float16](kv),
            )
            self.v.set(
                (block * self.v.shape()[1] + head) * self.v.shape()[2]
                + off * self.head_dim_of()
                + d,
                Scalar[DType.float16](vv),
            )
            return
        var slot = self.storage_pos(position)
        self.k.set(
            (head * self.max_len + slot) * self.k.shape()[2] + d,
            Scalar[DType.float16](kv),
        )
        self.v.set(
            (head * self.max_len + slot) * self.k.shape()[2] + d,
            Scalar[DType.float16](vv),
        )

    # -- row-level API (the quantized write/read path) ------------------------

    def set_kv_row(
        mut self,
        head: Int,
        position: Int,
        k_row: Tensor[DType.float16, 1],
        v_row: Tensor[DType.float16, 1],
    ):
        """Store one (head, position) K/V row; quantizes when the cache is.

        FP16 mode stores the row verbatim (the per-element `set_kv` path);
        Q4_0/Q8_0 modes quantize the row in place (per-32-element-block
        scale) into `kq`/`vq`.
        """
        if self.is_quantized():
            var off = self._quant_row_offset(head, position)
            if self.kv_type == KVCacheType.Q4_0:
                quantize_row_q4_0(k_row, self.kq, off)
                quantize_row_q4_0(v_row, self.vq, off)
            else:
                quantize_row_q8_0(k_row, self.kq, off)
                quantize_row_q8_0(v_row, self.vq, off)
            return
        for d in range(self.head_dim):
            self.set_kv(
                head,
                position,
                d,
                Float32(k_row.get(d)),
                Float32(v_row.get(d)),
            )

    def get_k_row(
        self, head: Int, position: Int, dst: Tensor[DType.float16, 1]
    ):
        """Read one K row into `dst` (dequantized to fp16 when quantized)."""
        if self.is_quantized():
            var off = self._quant_row_offset(head, position)
            if self.kv_type == KVCacheType.Q4_0:
                dequantize_row_q4_0(self.kq, off, dst)
            else:
                dequantize_row_q8_0(self.kq, off, dst)
            return
        if self.page_size > 0:
            var page = position // self.page_size
            var off = position % self.page_size
            var block = self.block_table[page]
            for d in range(self.head_dim):
                dst.set(
                    d,
                    self.k.get(
                        (block * self.k.shape()[1] + head) * self.k.shape()[2]
                        + off * self.head_dim
                        + d
                    ),
                )
            return
        var slot = self.storage_pos(position)
        for d in range(self.head_dim):
            dst.set(
                d,
                self.k.get((head * self.max_len + slot) * self.head_dim + d),
            )

    def get_v_row(
        self, head: Int, position: Int, dst: Tensor[DType.float16, 1]
    ):
        """Read one V row into `dst` (dequantized to fp16 when quantized)."""
        if self.is_quantized():
            var off = self._quant_row_offset(head, position)
            if self.kv_type == KVCacheType.Q4_0:
                dequantize_row_q4_0(self.vq, off, dst)
            else:
                dequantize_row_q8_0(self.vq, off, dst)
            return
        if self.page_size > 0:
            var page = position // self.page_size
            var off = position % self.page_size
            var block = self.block_table[page]
            for d in range(self.head_dim):
                dst.set(
                    d,
                    self.v.get(
                        (block * self.v.shape()[1] + head) * self.v.shape()[2]
                        + off * self.head_dim
                        + d
                    ),
                )
            return
        var slot = self.storage_pos(position)
        for d in range(self.head_dim):
            dst.set(
                d,
                self.v.get((head * self.max_len + slot) * self.head_dim + d),
            )

    def first_position(self) -> Int:
        """First position the sliding window still attends to."""
        if self.window <= 0:
            return 0
        var first = self.filled - self.window
        if first < 0:
            first = 0
        return first


# ---------------------------------------------------------------------------
# Cache (all layers)
# ---------------------------------------------------------------------------


struct KVCache(Movable):
    var layers: List[KVCacheLayer]

    def __init__(out self):
        self.layers = List[KVCacheLayer]()

    def __init__(
        out self,
        num_layers: Int,
        n_kv_heads: Int,
        max_len: Int,
        head_dim: Int,
        kv_type: KVCacheType = KVCacheType.FP16,
    ):
        self.layers = List[KVCacheLayer]()
        for _ in range(num_layers):
            self.layers.append(
                KVCacheLayer(n_kv_heads, max_len, head_dim, kv_type)
            )

    def num_layers(self) -> Int:
        return len(self.layers)

    def capacity(self) -> Int:
        # Hybrid (qwen35) recurrent layers hold no KV storage (max_len 0),
        # so the usable context length is the max over all layers - not
        # necessarily layer 0 (which is recurrent in qwen35).
        var cap = 0
        for i in range(len(self.layers)):
            if self.layers[i].max_len > cap:
                cap = self.layers[i].max_len
        return cap

    def filled(self) -> Int:
        if len(self.layers) > 0:
            return self.layers[0].filled
        return 0

    def reset(mut self):
        for i in range(len(self.layers)):
            self.layers[i].reset()

    def set_window(mut self, window: Int):
        for i in range(len(self.layers)):
            self.layers[i].set_window(window)

    def enable_paged(mut self, page_size: Int, n_kv_heads: Int, head_dim: Int):
        for i in range(len(self.layers)):
            self.layers[i].enable_paged(page_size, n_kv_heads, head_dim)

    def kv_cache_bytes(self) -> Int:
        """Total bytes held by this cache (fp16 K + V, or packed Q4/Q8 rows)."""
        var total = 0
        for i in range(len(self.layers)):
            if self.layers[i].is_quantized():
                total += self.layers[i].kq.numel()
                total += self.layers[i].vq.numel()
            else:
                total += self.layers[i].k.numel() * 2
                total += self.layers[i].v.numel() * 2
        return total

    def adjust_capacity(mut self, target_bytes: Int) -> Int:
        """Adapt the context length to a memory budget (M7 1.4).

        Rebuilds every layer with `max_len` such that the KV cache (in its
        resident format) fits in `target_bytes`; returns the new max_len.
        Filled state is dropped (the caller resets generation).
        """
        var n_layers = len(self.layers)
        if n_layers == 0 or target_bytes <= 0:
            return 0
        var n_kv = self.layers[0].n_kv_heads
        var head_dim = self.layers[0].head_dim
        var kv_type = self.layers[0].kv_type
        var per_pos: Int
        if kv_type == KVCacheType.FP16:
            per_pos = n_layers * n_kv * head_dim * 2 * 2  # K + V, fp16
        else:
            var nb = (head_dim + KV_QK - 1) // KV_QK
            var bb: Int
            if kv_type == KVCacheType.Q4_0:
                bb = KV_Q4_BLOCK_BYTES
            else:
                bb = KV_Q8_BLOCK_BYTES
            per_pos = n_layers * n_kv * 2 * nb * bb  # K + V, packed
        var max_len = target_bytes // per_pos
        if max_len < 1:
            max_len = 1
        var fresh = List[KVCacheLayer]()
        for _ in range(n_layers):
            fresh.append(KVCacheLayer(n_kv, max_len, head_dim, kv_type))
        self.layers = fresh^
        return max_len


def kv_cache_append[
    dtype: DType
](
    mut cache: KVCacheLayer,
    key: Tensor[dtype, 3],
    value: Tensor[dtype, 3],
    position: Int,
):
    """Copy the [n_kv_heads, 1, head_dim] key/value at `position`.

    (Exposed for API parity with the M2 placeholder; the MHA kernel writes
    the cache directly on its hot path.)  Quantized caches take the
    row-level path (`set_kv_row`), so the whole row is quantized at once.
    """
    var n_kv = key.shape()[0]
    var head_dim = key.shape()[2]
    if cache.is_quantized():
        var k_row = tensor_zeros[DType.float16, 1](
            StaticTuple[Int, 1](head_dim)
        )
        var v_row = tensor_zeros[DType.float16, 1](
            StaticTuple[Int, 1](head_dim)
        )
        for h in range(n_kv):
            for d in range(head_dim):
                k_row.set(
                    d, Scalar[DType.float16](Float32(key.get(h * head_dim + d)))
                )
                v_row.set(
                    d, Scalar[DType.float16](Float32(value.get(h * head_dim + d)))
                )
            cache.set_kv_row(h, position, k_row, v_row)
    else:
        for h in range(n_kv):
            for d in range(head_dim):
                cache.set_kv(
                    h,
                    position,
                    d,
                    Float32(key.get(h * head_dim + d)),
                    Float32(value.get(h * head_dim + d)),
                )
    if position + 1 > cache.filled:
        cache.filled = position + 1
