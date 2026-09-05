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
# The MHA kernel resolves every read/write through `get_k`/`get_v`/
# `set_kv`, so both dense and paged layouts are transparent to callers.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple

comptime DEFAULT_PAGE_SIZE = 64


struct KVCacheLayer(Copyable, Movable):
    """One layer's K/V cache: [n_kv_heads, max_len, head_dim] fp16 each.

    M7: when `page_size > 0` the storage is [n_blocks, n_kv_heads,
    page_size, head_dim] with an explicit block table - paged attention.
    `window > 0` enables the sliding window (attention only reads the last
    `window` positions).
    """

    var k: Tensor[DType.float16, 3]
    var v: Tensor[DType.float16, 3]
    var max_len: Int
    var filled: Int  # number of valid positions (0..max_len)
    var page_size: Int  # 0 = dense; > 0 = paged
    var block_table: List[Int]  # position page -> block index (paged)
    var window: Int  # sliding window; 0 = unlimited

    def __init__(out self, n_kv_heads: Int, max_len: Int, head_dim: Int):
        self.k = tensor_zeros[DType.float16, 3](
            StaticTuple[Int, 3](n_kv_heads, max_len, head_dim)
        )
        self.v = tensor_zeros[DType.float16, 3](
            StaticTuple[Int, 3](n_kv_heads, max_len, head_dim)
        )
        self.max_len = max_len
        self.filled = 0
        self.page_size = 0
        self.block_table = List[Int]()
        self.window = 0

    def __copyinit__(out self, existing: Self):
        """Deep copy (List is not implicitly copyable)."""
        self.k = existing.k
        self.v = existing.v
        self.max_len = existing.max_len
        self.filled = existing.filled
        self.page_size = existing.page_size
        self.block_table = List[Int]()
        for b in existing.block_table:
            self.block_table.append(b)
        self.window = existing.window

    def enable_paged(mut self, page_size: Int, n_kv_heads: Int, head_dim: Int):
        """Switch this layer to paged storage (blocks of `page_size`)."""
        if page_size < 1:
            unimplemented("KVCacheLayer.enable_paged: bad page size")
        var n_blocks = (self.max_len + page_size - 1) // page_size
        self.page_size = page_size
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

    def get_k(self, head: Int, position: Int, d: Int) -> Float32:
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
        if self.page_size > 0:
            return self.k.shape()[2] // self.page_size
        return self.k.shape()[2]

    def set_kv(
        mut self, head: Int, position: Int, d: Int, kv: Float32, vv: Float32
    ):
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

    def first_position(self) -> Int:
        """First position the sliding window still attends to."""
        if self.window <= 0:
            return 0
        var first = self.filled - self.window
        if first < 0:
            first = 0
        return first


struct KVCache(Movable):
    var layers: List[KVCacheLayer]

    def __init__(out self):
        self.layers = List[KVCacheLayer]()

    def __init__(
        out self, num_layers: Int, n_kv_heads: Int, max_len: Int, head_dim: Int
    ):
        self.layers = List[KVCacheLayer]()
        for _ in range(num_layers):
            self.layers.append(KVCacheLayer(n_kv_heads, max_len, head_dim))

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
        """Total bytes held by this cache (fp16 K + V)."""
        var total = 0
        for i in range(len(self.layers)):
            total += self.layers[i].k.numel() * 2
            total += self.layers[i].v.numel() * 2
        return total

    def adjust_capacity(mut self, target_bytes: Int) -> Int:
        """Adapt the context length to a memory budget (M7 1.4).

        Rebuilds every layer with `max_len` such that the fp16 KV cache
        fits in `target_bytes`; returns the new max_len.  Filled state is
        dropped (the caller resets generation).
        """
        var n_layers = len(self.layers)
        if n_layers == 0 or target_bytes <= 0:
            return 0
        var n_kv = self.layers[0].k.shape()[0]
        var head_dim = self.layers[0].k.shape()[2]
        var per_pos = n_layers * n_kv * head_dim * 2 * 2  # K + V, fp16
        var max_len = target_bytes // per_pos
        if max_len < 1:
            max_len = 1
        var fresh = List[KVCacheLayer]()
        for _ in range(n_layers):
            fresh.append(KVCacheLayer(n_kv, max_len, head_dim))
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
    the cache directly on its hot path.)
    """
    var n_kv = key.shape()[0]
    var head_dim = key.shape()[2]
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
