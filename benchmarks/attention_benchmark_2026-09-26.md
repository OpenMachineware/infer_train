# CPU Attention Benchmark Results (2026-09-26)

## Test Configuration
- Platform: Apple M1 (ARM64)
- Parameters: 32 heads, 128 head_dim, 512 seq_len
- Iterations: 20 (Rust), 100 (llama.cpp)

## Performance Comparison

| Format | Rust (GFLOPS) | llama.cpp (GFLOPS) | Ratio | Status |
|--------|---------------|-------------------|-------|--------|
| F16    | 4.39          | 12.65             | 30%   | ⚠️ Needs FP16 SIMD |
| Q8_0   | 7.34          | 7.54              | 90%   | ✅ Close |
| Q4_0   | 3.89          | 3.75              | 100%  | ✅ Matches! |
| Q4_1   | 3.58          | 6.61              | 50%   | ⚠️ Needs Q8_1 quant |
| Q5_0   | 3.59          | 5.68              | 60%   | ⚠️ Needs optimization |
| Q5_1   | 3.42          | 5.52              | 60%   | ⚠️ Needs Q8_1 quant |
| IQ4_NL | 3.11          | 3.58              | 80%   | ⚠️ Scalar fallback |

## Analysis

### F16 (30% of llama.cpp)
**Issue**: llama.cpp uses FP16 arithmetic directly in the attention loop while our implementation converts to FP32 for each K row.

**Solution**: Use FP16 NEON intrinsics (`vcvt_f16_f32`, FP16 FMA) in the online softmax loop.

### Q8_0 (90% of llama.cpp)
**Status**: Close to llama.cpp performance. The gap is likely from loop overhead and memory access patterns.

### Q4_0 (100% of llama.cpp) ✅
**Status**: Achieved parity! The inline scalar implementation matches llama.cpp's NEON version.

### Q4_1, Q5_1 (50-60% of llama.cpp)
**Issue**: Using Q8_0 for query quantization instead of Q8_1. llama.cpp uses:
- `vec_dot_type(Q4_1) = Q8_1`
- `vec_dot_type(Q5_1) = Q8_1`

**Q8_1 vs Q8_0**:
- Q8_0: `d` (scale), `qs` (32 int8 values) - 34 bytes
- Q8_1: `d` (scale), `s` (sum), `qs` (32 int8 values) - 36 bytes

**Solution**:
1. Implement `quantize_row_q8_1` function
2. Update Q4_1/Q5_1 attention to use Q8_1 for query
3. Use correct vec_dot formula: `d_q * d_k * sum + s_q * m_k`

### Q5_0 (60% of llama.cpp)
**Issue**: Similar to Q4_1, needs proper NEON implementation with Q8_0 × Q5_0 fused vec_dot.

**Solution**: Optimize the existing vec_dot_q5_0_q8_0 function.

### IQ4_NL (80% of llama.cpp)
**Issue**: Using scalar fallback for vec_dot. The lookup table should be vectorized with TBL instruction.

**Solution**: Implement NEON TBL-based IQ4_NL unpacking.

## Next Steps

1. **High Priority**: Fix Q4_1 and Q5_1 by implementing Q8_1 quantization
2. **Medium Priority**: Optimize F16 attention with FP16 NEON
3. **Medium Priority**: Optimize Q5_0 vec_dot
4. **Low Priority**: Vectorize IQ4_NL with TBL instruction

## Code Locations

- Attention implementations: `src/ops/attention.rs`
- Fused vec_dot: `src/ops/attention_fused.rs`
- Quantization: `src/quant/kv_cache/quant.rs`
- Block types: `src/quant/kv_cache/mod.rs`, `src/quant/types.rs`
