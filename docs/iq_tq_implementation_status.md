# IQ/TQ Series Implementation Status

## Completed

### Q4_K Optimization ✅
- **File**: `src/quant/vec_dot/arm.rs`
- **Performance**: 84.8 GFLOPS (exceeds llama.cpp by 3-10%)
- **Key optimization**: `vpaddq_s16` for bsums, correct scale decode
- **Commits**: b24aee5, 7f817c7, aaae792

### IQ4_NL ✅
- **Block size**: 32 (pairs with Q8_0)
- **Lookup table**: `KVALUES_IQ4NL` (16 items)
- **Instruction**: `vqtbl1q_s8` (TBL)
- **Commit**: 7f817c7

### IQ4_XS ✅
- **Block size**: 256 (pairs with Q8_K)
- **Scale decode**: `scales_l` + `scales_h`, offset -32
- **Commit**: aaae792

## Remaining Work

### Priority Order
1. **IQ1_S/IQ1_M** - Most important (lowest bpw: 1.5625)
2. **IQ2_XXS/IQ2_XS/IQ2_S** - Medium bpw
3. **IQ3_XXS/IQ3_S** - Higher bpw
4. **TQ1_0/TQ2_0** - New formats

### Lookup Tables Needed

| Format | Table Name | Size | Location in llama.cpp |
|--------|-----------|------|----------------------|
| IQ1_S | `iq1s_grid` | 2048 × uint64 | `ggml-common.h:1135` |
| IQ1_M | Uses iq1s_grid + delta | - | `ggml-common.h` |
| IQ2_XXS | `iq2xxs_grid` | 512 × uint32 | `ggml-common.h` |
| IQ2_XS | `iq2xs_grid` | 512 × uint64 | `ggml-common.h` |
| IQ2_S | `iq2s_grid` | 768 × uint32 | `ggml-common.h` |
| IQ3_XXS | `iq3xxs_grid` | 512 × uint32 | `ggml-common.h` |
| IQ3_S | `iq3s_grid` | 512 × uint32 | `ggml-common.h` |
| TQ1_0 | None (polynomial decode) | - | `quants.c:1397` |
| TQ2_0 | None (bit decode) | - | `quants.c:1574` |

### Implementation Pattern

```rust
// 1. Extract lookup table from llama.cpp
static IQ1S_GRID: [u64; 2048] = [ /* values */ ];

// 2. Use TBL instruction for small tables (< 256 entries)
//    Or direct memory access for large tables

// 3. Scale decode varies per format:
//    - IQ4_XS: (scales_l[idx] | (scales_h << 4)) - 32
//    - IQ1_S: 2*((qh >> 12) & 7) + 1
//    - IQ1_M: merged scale calculation

// 4. Delta correction (IQ1_S/M/IQ3 series)
//    const DELTA: f32 = 0.125; // or 0.5 for IQ3
//    sum += delta * bsum_correction
```

## Key Files

- **Rust kernels**: `src/quant/vec_dot/arm.rs`
- **Type definitions**: `src/quant/types.rs`
- **llama.cpp reference**: `llama.cpp-0.4.1/ggml/src/ggml-cpu/arch/arm/quants.c`
- **Lookup tables**: `llama.cpp-0.4.1/ggml/src/ggml-common.h`

## Benchmark Setup

- **File**: `benches/vec_dot_neon_bench.rs`
- **Pattern**: Add generator function + benchmark function
- **Test size**: nb=256 (n=65536 for block=256, n=131072 for block=32)

## Methodology

1. Read llama.cpp implementation line by line
2. Understand decode logic (grid index, scale, delta)
3. Extract lookup table to Rust static
4. Implement NEON version using TBL/SDOT
5. Benchmark against llama.cpp
6. Verify correctness (result must match)

## Performance Target

All formats should **exceed llama.cpp** (following Q4_K pattern).
