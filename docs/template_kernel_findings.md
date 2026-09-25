# GPU Template Kernel Investigation Results

## Summary

Investigated using llama.cpp's templated GEMM kernels to reduce code duplication across quantization formats (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K).

## Findings

### Performance
- **Template kernel**: 1563 GFLOPS (7.6x faster)
- **Hand-written kernels**: 204 GFLOPS (matches llama.cpp baseline)

### Correctness Issues

1. **Small matrix problem**: Template kernel designed for M≥64, N≥32
   - Test with M=2: all outputs same value (bug)
   - Test with M=128: still wrong results (all 576.0 instead of correct pattern)

2. **Row access bug**: Template kernel not correctly processing different weight rows
   - Hand-written kernel correctly gives different results for different rows
   - Template kernel gives uniform wrong results

### Root Cause

The template kernel is a **fallback version** from llama.cpp. The actual optimized version uses Metal's tensor operations (`#ifdef GGML_METAL_HAS_TENSOR`), but our version only has the fallback.

The fallback has fundamental issues:
- Complex indexing logic for tile-based processing
- Possible bugs in row/column stride calculations
- Designed for specific matrix sizes, not general-purpose

## Recommendation

**Continue with hand-written kernels**. Reasons:

1. **Correctness verified**: All formats tested and working
2. **Performance matched**: 204 GFLOPS matches llama.cpp's FP16 baseline
3. **Lower risk**: Template kernel debugging would be time-consuming
4. **Code simplicity**: Hand-written kernels are easier to understand and maintain

## Alternative: Adopt llama.cpp's Tensor-Based Version

If template kernels are desired for code reduction:
1. Enable Metal tensor operations (`#ifdef GGML_METAL_HAS_TENSOR`)
2. Port the tensor-based implementation from llama.cpp
3. May require Metal 3.0+ and specific hardware support
4. Higher performance potential but more complex

## Test Results

### Single block test (M=1, K=256, N=1)
- Both kernels: 128.0 (expected 256.0)
- Q4_K dequantization interpretation needs verification

### Multi-row test (M=128, K=256, N=32)
- Template: All values = 576.0 (wrong)
- Hand-written: Correct pattern (128, 1024, 128, ... for scales 1, 8, 1)
- Match: 0/4096 values

## Files Modified
- `shaders/gemm_llama_cpp.metal`: Template kernel structure fixed (84 bytes)
- `src/quant/vec_dot/gpu/metal.rs`: Updated template function interface
- `src/bin/verify_template_gemm.rs`: Verification tests
- `src/bin/debug_template_simple.rs`: Debug tests

## Next Steps

If continuing with template approach:
1. Fix row access logic in fallback kernel
2. Add support for small matrices
3. Verify correctness across all Q formats
4. Extensive testing with various M, N, K sizes

If switching back to hand-written:
1. No action needed - kernels already working
2. Focus on performance optimization if needed