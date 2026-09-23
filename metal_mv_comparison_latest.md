# Metal MV Performance Analysis (2026-09-23 Latest)

## Test Setup
- Platform: Apple M1 Max
- M = 1024 rows
- K = 256, 512, 1024, 2048, 4096
- Optimization: Split kernels with dynamic dispatch

## Latest Results (After Split Optimization)

### K=4096 Performance Comparison

| Format | Our GFLOPS | llama.cpp GFLOPS | Ratio | Status |
|--------|-----------|------------------|-------|--------|
| IQ4_NL | 29.19 | 28.71 | **102%** | ✓ |
| IQ1_S | 31.22 | 29.13 | **107%** | ✓ |
| IQ1_M | 27.45 | 29.13 | **94%** | ✗ |
| IQ2_XXS | 29.37 | 28.58 | **103%** | ✓ |
| IQ2_XS | 29.40 | 28.68 | **102%** | ✓ |
| IQ2_S | 28.99 | 28.35 | **102%** | ✓ |
| IQ3_XXS | 29.20 | 29.05 | **100%** | ✓ |
| IQ3_S | 29.36 | 29.02 | **101%** | ✓ |
| TQ2_0 | 32.09 | 29.21 | **110%** | ✓ |
| TQ1_0 | 27.69 | ~29.0 | **95%** | ✗ |

## Summary Statistics

### At K=4096 (largest size):
- **Exceed llama.cpp: 8/10 formats** (80%)
- **Below llama.cpp: 2/10 formats** (20%)
  - IQ1_M: 94% (6% slower)
  - TQ1_0: 95% (5% slower)

### Across All K Sizes (Total tests: 10 formats × 5 sizes = 50):

**Formats exceeding llama.cpp at ALL sizes:**
1. IQ1_S ✓ (all 5 sizes)
2. IQ2_XXS ✓ (all 5 sizes)
3. IQ2_XS ✓ (all 5 sizes)
4. IQ2_S ✓ (all 5 sizes)
5. IQ3_XXS ✓ (all 5 sizes)
6. IQ3_S ✓ (all 5 sizes)
7. TQ2_0 ✓ (all 5 sizes)

**Formats with SOME sizes below llama.cpp:**
1. IQ4_NL: Below at K=1024 (98%)
2. IQ1_M: Below at K=512 (95%), K=4096 (94%)
3. TQ1_0: Below at K=4096 (95%)

## Gap Analysis

### Closest to parity (within 5%):
- IQ4_NL: 102% (only 2% above)
- IQ1_M: 94% (6% below)
- TQ1_0: 95% (5% below)

### Strongest performance:
- TQ2_0: 110% (10% above llama.cpp)
- IQ1_S: 107% (7% above)

## Root Cause Analysis

**Why IQ1_M underperforms:**
- Complex merged scale calculation
- Multiple scale lookups per block
- Possible optimization: precompute merged scales

**Why TQ1_0 underperforms:**
- Similar to IQ1_M, uses complex scale extraction
- Possible optimization: improve qh bit extraction

**Why IQ4_NL is close but not dominant:**
- Already well-optimized in llama.cpp
- Uses threadgroup memory for kvalues table
- Limited optimization space

## Recommendations

1. **Priority 1: Fix IQ1_M and TQ1_0**
   - These are the only 2 formats below llama.cpp at K=4096
   - Need to optimize scale extraction logic
   - Target: >100% at all sizes

2. **Priority 2: Optimize split threshold**
   - Current threshold: nb32 < 32
   - May need format-specific thresholds
   - Test each format at boundary conditions

3. **Priority 3: Test thermal throttling**
   - M1 Max thermal limits may affect K=4096 results
   - Add cooling delays between benchmarks
   - Ensure consistent performance

## Files Modified

- `shaders/mv_iq_tq.metal`: Added split=1 kernels for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S
- `src/quant/vec_dot/gpu/metal.rs`: Added dynamic dispatch and split=1 pipelines
- `src/bin/metal_all_sizes_bench.rs`: Multi-size benchmark

## Next Steps

1. Run llama.cpp Metal benchmarks directly for comparison
2. Optimize IQ1_M merged scale calculation
3. Optimize TQ1_0 qh extraction
4. Test with thermal management
5. Verify correctness at all sizes
