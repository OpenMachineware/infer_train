# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/kv_cache_gpu.mojo
#
# GPU KV cache for autoregressive decoding.
# K/V cache resides on GPU, avoiding per-layer CPU↔GPU transfers.

from max.gpu.host import DeviceContext, DeviceBuffer
from std.utils.static_tuple import StaticTuple
from std.gpu import global_idx, grid_dim, block_dim
from std.memory import Pointer
from std.origin import MutAnyOrigin
from src.core.ops.gpu.gpu_runtime import grid1d

comptime BLOCK = 256


# -- GPU kernels for KV cache --------------------------------------------------


def _copy_to_cache_kernel_f16(
    src: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    offset: Int32,
    n: Int32,
):
    """Copy src[0:n] to dst[offset:offset+n]."""
    var n_i = Int(n)
    var offset_i = Int(offset)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        dst[unsafe_offset=offset_i + i] = src[unsafe_offset=i]
        i += stride


struct KVCacheLayerGPU:
    """One layer's K/V cache on GPU: [n_kv_heads, max_len, head_dim] fp16 each.

    The cache is persistent on GPU memory, updated in-place during decode.
    No CPU↔GPU transfers between layers.

    Args:
        ctx: GPU device context
        n_kv_heads: Number of KV heads
        max_len: Maximum sequence length
        head_dim: Dimension per head
    """

    var k_buf: DeviceBuffer[DType.float16]  # [n_kv_heads * max_len * head_dim]
    var v_buf: DeviceBuffer[DType.float16]  # [n_kv_heads * max_len * head_dim]
    var n_kv_heads: Int
    var max_len: Int
    var head_dim: Int
    var filled: Int  # Number of valid positions (0..max_len)

    def __init__(
        out self,
        ctx: DeviceContext,
        n_kv_heads: Int,
        max_len: Int,
        head_dim: Int,
    ) raises:
        """Initialize GPU KV cache with zeroed buffers."""
        self.n_kv_heads = n_kv_heads
        self.max_len = max_len
        self.head_dim = head_dim
        self.filled = 0

        # Allocate GPU buffers for K and V
        var total_elements = n_kv_heads * max_len * head_dim
        self.k_buf = ctx.enqueue_create_buffer[DType.float16](total_elements)
        self.v_buf = ctx.enqueue_create_buffer[DType.float16](total_elements)

    def reset(mut self):
        """Reset cache (clear filled counter, don't free GPU memory)."""
        self.filled = 0

    def update(
        mut self,
        ctx: DeviceContext,
        k_new: DeviceBuffer[DType.float16],  # [n_kv_heads * head_dim] - one position
        v_new: DeviceBuffer[DType.float16],  # [n_kv_heads * head_dim] - one position
        position: Int,
    ) raises:
        """Update cache with new K/V at given position.

        Copies the new K/V from the provided buffers into the cache.
        For decode: position = filled, k_new/v_new come from projection.

        Args:
            ctx: GPU device context
            k_new: New K values for all heads [n_kv_heads * head_dim]
            v_new: New V values for all heads [n_kv_heads * head_dim]
            position: Target position in cache (0-indexed)
        """
        if position < 0 or position >= self.max_len:
            raise "KVCacheLayerGPU: position out of range"

        # Calculate offset for this position
        var offset = position * self.n_kv_heads * self.head_dim
        var n_elements = self.n_kv_heads * self.head_dim

        # Copy new K into cache
        ctx.enqueue_function[_copy_to_cache_kernel_f16](
            k_new,
            self.k_buf,
            Int32(offset),
            Int32(n_elements),
            grid_dim=grid1d(n_elements, BLOCK),
            block_dim=BLOCK,
        )

        # Copy new V into cache
        ctx.enqueue_function[_copy_to_cache_kernel_f16](
            v_new,
            self.v_buf,
            Int32(offset),
            Int32(n_elements),
            grid_dim=grid1d(n_elements, BLOCK),
            block_dim=BLOCK,
        )

        # Update filled counter
        if position + 1 > self.filled:
            self.filled = position + 1

    def get_k_buffer(self) -> DeviceBuffer[DType.float16]:
        """Get the full K cache buffer [n_kv_heads, max_len, head_dim]."""
        return self.k_buf

    def get_v_buffer(self) -> DeviceBuffer[DType.float16]:
        """Get the full V cache buffer [n_kv_heads, max_len, head_dim]."""
        return self.v_buf

    def capacity(self) -> Int:
        """Maximum sequence length."""
        return self.max_len

    def filled_len(self) -> Int:
        """Number of valid positions."""
        return self.filled


struct KVCacheGPU:
    """All layers' K/V caches on GPU."""

    var layers: List[KVCacheLayerGPU]

    def __init__(out self):
        self.layers = List[KVCacheLayerGPU]()

    def __init__(
        out self,
        ctx: DeviceContext,
        num_layers: Int,
        n_kv_heads: Int,
        max_len: Int,
        head_dim: Int,
    ) raises:
        """Initialize GPU KV cache for all layers."""
        self.layers = List[KVCacheLayerGPU]()
        for _ in range(num_layers):
            self.layers.append(
                KVCacheLayerGPU(ctx, n_kv_heads, max_len, head_dim)
            )

    def num_layers(self) -> Int:
        return len(self.layers)

    def capacity(self) -> Int:
        if len(self.layers) == 0:
            return 0
        return self.layers[0].capacity()

    def filled(self) -> Int:
        if len(self.layers) > 0:
            return self.layers[0].filled_len()
        return 0

    def reset(mut self):
        for i in range(len(self.layers)):
            self.layers[i].reset()

    def kv_cache_bytes(self) -> Int:
        """Total GPU memory used (K + V in fp16)."""
        var total = 0
        for i in range(len(self.layers)):
            # Each layer: n_kv_heads * max_len * head_dim * 2 (K+V) * 2 bytes (fp16)
            total += self.layers[i].n_kv_heads * self.layers[i].max_len * self.layers[i].head_dim * 4
        return total
