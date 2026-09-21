# Test: Paged KV Cache
# Verify dynamic block allocation and access

from src.core.ops.attention.kv_cache import KVCacheLayer, KVCacheType
from src.core.tensor import Tensor, tensor_zeros
from std.utils import StaticTuple

comptime HEAD_DIM = 64
comptime N_KV_HEADS = 8
comptime PAGE_SIZE = 16  # 16 tokens per block

def test_paged_basic():
    """Test basic paged KV allocation and access."""
    print("=== Testing Paged KV Cache ===\n")

    # Create cache with paged mode
    var cache = KVCacheLayer(N_KV_HEADS, 0, HEAD_DIM, KVCacheType.FP16)  # max_len=0 for dynamic
    cache.enable_paged(PAGE_SIZE, N_KV_HEADS, HEAD_DIM)

    print("Initial state:")
    print("  n_blocks:", cache.n_blocks)
    print("  free_blocks:", len(cache.free_blocks))
    print("  block_table:", len(cache.block_table))
    print("  filled:", cache.filled)

    # Ensure capacity for 50 tokens (should allocate 4 blocks: 16*4=64)
    print("\nEnsuring capacity for 50 tokens...")
    cache.ensure_capacity(50)

    print("After ensure_capacity(50):")
    print("  n_blocks:", cache.n_blocks)
    print("  block_table:", len(cache.block_table))

    # Write some data
    print("\nWriting data to positions 0-49...")
    var k_row = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
    var v_row = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))

    for pos in range(50):
        for d in range(HEAD_DIM):
            k_row.set(d, Scalar[DType.float16](Float32((pos + d) % 100) / 100.0))
            v_row.set(d, Scalar[DType.float16](Float32((pos * d) % 100) / 100.0))

        for h in range(N_KV_HEADS):
            cache.set_kv_row(h, pos, k_row, v_row)

    cache.filled = 50

    # Read back and verify
    print("\nReading back and verifying...")
    var k_read = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
    var v_read = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))

    var errors = 0
    for pos in range(50):
        for h in range(N_KV_HEADS):
            cache.get_k_row(h, pos, k_read)
            cache.get_v_row(h, pos, v_read)

            for d in range(HEAD_DIM):
                var k_expected = Float32((pos + d) % 100) / 100.0
                var v_expected = Float32((pos * d) % 100) / 100.0
                var k_got = Float32(k_read.get(d))
                var v_got = Float32(v_read.get(d))

                if abs(k_got - k_expected) > 0.01 or abs(v_got - v_expected) > 0.01:
                    errors += 1

    print("Verification:", "PASSED" if errors == 0 else "FAILED")

    # Test dynamic growth: extend to 100 tokens
    print("\nExtending to 100 tokens...")
    cache.ensure_capacity(100)

    print("After ensure_capacity(100):")
    print("  n_blocks:", cache.n_blocks)
    print("  block_table:", len(cache.block_table))

    # Write more data
    print("Writing data to positions 50-99...")
    for pos in range(50, 100):
        for d in range(HEAD_DIM):
            k_row.set(d, Scalar[DType.float16](Float32((pos + d) % 100) / 100.0))
            v_row.set(d, Scalar[DType.float16](Float32((pos * d) % 100) / 100.0))

        for h in range(N_KV_HEADS):
            cache.set_kv_row(h, pos, k_row, v_row)

    cache.filled = 100

    # Verify all 100 tokens
    print("Verifying all 100 tokens...")
    errors = 0
    for pos in range(100):
        for h in range(N_KV_HEADS):
            cache.get_k_row(h, pos, k_read)
            cache.get_v_row(h, pos, v_read)

            for d in range(HEAD_DIM):
                var k_expected = Float32((pos + d) % 100) / 100.0
                var v_expected = Float32((pos * d) % 100) / 100.0
                var k_got = Float32(k_read.get(d))
                var v_got = Float32(v_read.get(d))

                if abs(k_got - k_expected) > 0.01 or abs(v_got - v_expected) > 0.01:
                    errors += 1

    print("Verification:", "PASSED" if errors == 0 else "FAILED")

    # Test reset
    print("\nTesting reset...")
    cache.reset()
    print("After reset:")
    print("  n_blocks:", cache.n_blocks)
    print("  free_blocks:", len(cache.free_blocks))
    print("  block_table:", len(cache.block_table))
    print("  filled:", cache.filled)

    print("\n=== All tests complete ===\n")


def main():
    test_paged_basic()
