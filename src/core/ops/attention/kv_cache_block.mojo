# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/attention/kv_cache_block.mojo
#
# Block-major KV cache for improved decode performance.
# Layout: [n_blocks, n_kv_heads, block_size, head_dim]
#
# Design:
# - vLLM-style 16-token blocks for optimal cache locality
# - Dense mode: preallocated blocks
# - Paged mode: dynamic block allocation (future)
# - Unified access API: get_k_row / set_kv_row
#
# Why block-major?
# - Decode phase: traverse KV sequence, need sequential memory access
# - 16 tokens = 16 * head_dim floats, fits SIMD width (AVX-512/NEON)
# - Better cache utilization for long sequences

from ...tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple
from std.math import exp

# Block size: 16 tokens per block (vLLM standard)
# - AVX-512: 16 floats per vector
# - NEON: 8 floats per vector, 2 blocks = 16 floats
# - Cache line: 64 bytes = 16 * float32 = 32 * float16
comptime BLOCK_SIZE = 16


# ============================================================================
# Block-major KV Cache Layer (FP16)
# ============================================================================


struct BlockKVCacheLayer:
    """Block-major KV cache with 16-token blocks.

    Storage: [n_blocks, n_kv_heads, block_size, head_dim]
    - n_blocks: number of 16-token blocks
    - n_kv_heads: number of KV heads
    - block_size: 16 (comptime)
    - head_dim: dimension per head

    Benefits:
    - Better cache locality for decode (sequential KV traversal)
    - SIMD-friendly: 16 tokens per block matches vector width
    - Supports both dense (preallocated) and paged (dynamic) modes
    """

    var k: Tensor[DType.float16, 3]  # [n_blocks, n_kv_heads, block_size * head_dim]
    var v: Tensor[DType.float16, 3]
    var n_blocks: Int
    var n_kv_heads: Int
    var head_dim: Int
    var max_len: Int  # maximum sequence length
    var filled: Int  # number of filled positions
    var block_table: List[Int]  # position -> block mapping (for paged mode)
    var is_paged: Bool  # false = dense (fixed blocks), true = dynamic allocation

    def __init__(
        out self,
        n_kv_heads: Int,
        max_len: Int,
        head_dim: Int,
    ):
        """Initialize dense block cache (fixed allocation)."""
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.filled = 0
        self.is_paged = False
        self.block_table = List[Int]()

        # Calculate number of blocks
        # Round up to cover max_len tokens
        self.n_blocks = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Allocate storage: [n_blocks, n_kv_heads, block_size * head_dim]
        # Note: Use 3D tensor for simplicity (block_size * head_dim as last dim)
        # In memory: each block holds 16 consecutive positions for a head
        var block_dim = BLOCK_SIZE * head_dim
        self.k = tensor_zeros[DType.float16, 3](
            StaticTuple[Int, 3](self.n_blocks, n_kv_heads, block_dim)
        )
        self.v = tensor_zeros[DType.float16, 3](
            StaticTuple[Int, 3](self.n_blocks, n_kv_heads, block_dim)
        )

    def __copyinit__(out self, existing: Self):
        """Deep copy."""
        self.k = existing.k
        self.v = existing.v
        self.n_blocks = existing.n_blocks
        self.n_kv_heads = existing.n_kv_heads
        self.head_dim = existing.head_dim
        self.max_len = existing.max_len
        self.filled = existing.filled
        self.block_table = List[Int]()
        for b in existing.block_table:
            self.block_table.append(b)
        self.is_paged = existing.is_paged

    # -- Accessor methods --------------------------------------------------

    def set_kv_row(
        mut self,
        head: Int,
        position: Int,
        k_row: Tensor[DType.float16, 1],
        v_row: Tensor[DType.float16, 1],
    ):
        """Store one (head, position) K/V row.

        Maps position to block and offset, stores in block-major layout.
        """
        var block_id = position // BLOCK_SIZE
        var offset = position % BLOCK_SIZE

        # Bounds check
        if block_id >= self.n_blocks:
            return  # Silently ignore (should not happen in correct usage)

        # Store K row
        # Index: [block_id, head, offset * head_dim : (offset+1) * head_dim]
        var base = (block_id * self.n_kv_heads + head) * (BLOCK_SIZE * self.head_dim) + offset * self.head_dim
        for d in range(self.head_dim):
            self.k.set(base + d, k_row.get(d))

        # Store V row
        for d in range(self.head_dim):
            self.v.set(base + d, v_row.get(d))

        # Update filled count
        if position + 1 > self.filled:
            self.filled = position + 1

    def get_k_row(
        self, head: Int, position: Int, dst: Tensor[DType.float16, 1]
    ):
        """Read one K row into `dst`."""
        var block_id = position // BLOCK_SIZE
        var offset = position % BLOCK_SIZE

        if block_id >= self.n_blocks:
            return  # Return zeros (should not happen)

        var base = (block_id * self.n_kv_heads + head) * (BLOCK_SIZE * self.head_dim) + offset * self.head_dim
        var src_ptr = self.k.data().unsafe_offset(base)
        var dst_ptr = dst.data()

        # SIMD copy
        var d = 0
        while d + 8 <= self.head_dim:
            var v = src_ptr.unsafe_load[width=8](offset=d)
            dst_ptr.unsafe_store(d, v)
            d += 8
        while d < self.head_dim:
            dst_ptr.unsafe_store(d, src_ptr.unsafe_load(offset=d))
            d += 1

    def get_v_row(
        self, head: Int, position: Int, dst: Tensor[DType.float16, 1]
    ):
        """Read one V row into `dst`."""
        var block_id = position // BLOCK_SIZE
        var offset = position % BLOCK_SIZE

        if block_id >= self.n_blocks:
            return  # Return zeros

        var base = (block_id * self.n_kv_heads + head) * (BLOCK_SIZE * self.head_dim) + offset * self.head_dim
        var src_ptr = self.v.data().unsafe_offset(base)
        var dst_ptr = dst.data()

        # SIMD copy
        var d = 0
        while d + 8 <= self.head_dim:
            var v = src_ptr.unsafe_load[width=8](offset=d)
            dst_ptr.unsafe_store(d, v)
            d += 8
        while d < self.head_dim:
            dst_ptr.unsafe_store(d, src_ptr.unsafe_load(offset=d))
            d += 1

    def get_k(self, head: Int, position: Int, d: Int) -> Float32:
        """Get single K element (for compatibility, but slower)."""
        var row = Tensor[DType.float16, 1](StaticTuple[Int, 1](self.head_dim))
        self.get_k_row(head, position, row)
        return Float32(row.get(d))

    def get_v(self, head: Int, position: Int, d: Int) -> Float32:
        """Get single V element."""
        var row = Tensor[DType.float16, 1](StaticTuple[Int, 1](self.head_dim))
        self.get_v_row(head, position, row)
        return Float32(row.get(d))

    def reset(mut self):
        """Clear filled positions."""
        self.filled = 0

    def first_position(self) -> Int:
        """First valid position (for sliding window, future)."""
        return 0


# ============================================================================
# Helper functions
# ============================================================================


def block_kv_cache_bytes(
    n_kv_heads: Int,
    max_len: Int,
    head_dim: Int,
) -> Int:
    """Calculate memory needed for block cache (K + V, fp16)."""
    var n_blocks = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    var bytes_per_block = n_kv_heads * BLOCK_SIZE * head_dim * 2  # fp16 = 2 bytes
    return n_blocks * bytes_per_block * 2  # K + V
