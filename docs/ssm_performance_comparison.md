# SSM GPU Performance Comparison

## Summary (2026-10-15)

| Operator | Self-written | llama.cpp | Ratio | Status |
|----------|-------------|-----------|-------|--------|
| Conv1D (d_conv=4) | 87.36 us | 111.51 us | **0.78x** | ✅ FASTER |
| SSD (d_state=64) | 2818 GFLOPS | 8000-10000 GFLOPS | **0.35x** | ⚠️ Needs optimization |

## Conv1D Details

**Parameters:** d_inner=3328, n_tok=512, d_conv=4

**Key insight:** The apparent 3.6x gap when measuring single kernel dispatch was **synchronization overhead**, not kernel performance:
- Single kernel per CB: 400 us (87 us kernel + 313 us overhead)
- 10 kernels per CB: 87 us per kernel (amortized overhead)

**Implementation:**
- Metal template specialization with compile-time batch sizes (256, 128, 64, 32, 16, 8, 2)
- d_conv=4 hard-coded, full loop unrolling
- Threadgroup covers BATCH_SIZE tokens

## SSD Details

**Parameters:** d_state=64, head_dim=64, n_tok=512

**Gap cause:** Algorithm differences in SAM (State Accumulator Matrix) building:
- My NSG=4 SIMD groups may not be optimal
- Need to study llama.cpp's MMA patterns more carefully

## Lessons

1. **GPU benchmark must batch kernels** - Single dispatch shows sync overhead, not kernel speed
2. **Conv1D matches/exceeds llama.cpp** when measured correctly
3. **SSD optimization is next priority**

## Files

- Conv1D kernel: [src/ops/metal/ops/ssm_conv.metal](file:///Users/jia/Desktop/infer_train/src/ops/metal/ops/ssm_conv.metal)
- SSD kernel: [src/ops/metal/ops/ssd_optimized.metal](file:///Users/jia/Desktop/infer_train/src/ops/metal/ops/ssd_optimized.metal)
- Benchmark: [benches/conv1d_direct_compare.rs](file:///Users/jia/Desktop/infer_train/benches/conv1d_direct_compare.rs)
