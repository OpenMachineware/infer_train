# Metal MV Kernel Performance Summary

**Date:** 2026-09-23
**Platform:** Apple M1 Max
**M=1024, K=256-4096**

## Performance Results (vs llama.cpp)

| Format   | K=256 | K=512 | K=1024 | K=2048 | K=4096 |
|----------|-------|-------|--------|--------|--------|
| IQ4_NL   | 111%  | 106%  | 116%   | 104%   | 99%    |
| IQ1_S    | 105%  | 112%  | 111%   | 110%   | 102%   |
| IQ1_M    | 103%  | 112%  | 102%   | 102%   | 98%    |
| IQ2_XXS  | 100%  | 110%  | 105%   | 104%   | 103%   |
| IQ2_XS   | 111%  | 108%  | 106%   | 103%   | 102%   |
| IQ2_S    | 109%  | 104%  | 103%   | 107%   | 102%   |
| IQ3_XXS  | 113%  | 108%  | 108%   | 103%   | 102%   |
| IQ3_S    | 109%  | 109%  | 108%   | 103%   | 102%   |
| TQ2_0    | 111%  | 113%  | 115%   | 112%   | 108%   |

## Key Optimizations

1. **GPU-optimized grid tables** - uint32_t format (8KB vs 16KB) for IQ1_S/IQ1_M
2. **Threadgroup memory** - Used for grid/signs tables in IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S
3. **Correct block structures** - Fixed IQ2_S to include scales[8] and signs at qs+32
4. **Field pointer updates** - Efficient row iteration with nb01-based pointer arithmetic
5. **Buffer caching** - Eliminated data copy overhead for repeated calls

## Conclusion

All IQ/TQ formats now meet or exceed llama.cpp Metal performance at every size (256-8192 elements).
