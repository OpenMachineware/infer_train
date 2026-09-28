# F16 Attention Optimization Findings

## Summary

After extensive investigation, we identified the root cause of the performance gap between our F16 attention and llama.cpp.

## Key Findings

1. **Pure C implementation matches llama.cpp**: When the attention code is compiled together with the benchmark (single compilation unit), we achieve 12+ GFLOPS - matching llama.cpp's 12-13 GFLOPS.

2. **Static library linking causes 3x slowdown**: When the attention code is compiled into a static library and linked separately, performance drops to 3-4 GFLOPS. This is because:
   - The compiler cannot inline the `static inline` functions across compilation units
   - Function calls add overhead
   - Optimization opportunities are lost

3. **Loop structure matters**: The llama.cpp-style loop structure is more compiler-friendly:
   ```c
   // llama.cpp style (faster)
   const int np = (n & ~31);
   for (int i = 0; i < np; i += 32) { ... }
   
   // vs our original (slower)
   for (; i + 32 <= n; i += 32) { ... }
   ```

4. **Softmax branch structure matters**: The simpler llama.cpp-style softmax handling is faster:
   ```c
   // llama.cpp style (faster)
   if (score > M) {
       float ms = expf(M - score);
       S *= ms;
       scale_accumulator(ms);
       M = score;
   }
   float vs = expf(score - M);
   
   // vs tracking old_max (slower)
   ```

## Solution Options

1. **Header-only implementation**: Move the implementation to a header file with `static inline` functions. This allows the compiler to inline at the call site.

2. **LTO (Link Time Optimization)**: Enable LTO in the build to allow cross-module optimization.

3. **Rust inline assembly**: Rewrite the hot loops directly in Rust using `std::arch::aarch64` intrinsics, avoiding the FFI boundary.

## Current Performance

| seq_len | Pure C (GFLOPS) | Via static lib (GFLOPS) | llama.cpp (GFLOPS) |
|---------|----------------|------------------------|-------------------|
| 512     | 12+            | 3-4                    | 15                |
| 1024    | 12+            | 3-4                    | 12                |
| 2048    | 8-9            | 3-4                    | 8-9               |
| 4096    | 5-6            | 3-4                    | 5-6               |

## Recommendations

1. Implement the attention kernel directly in Rust using NEON intrinsics
2. This avoids the FFI overhead and allows the Rust compiler to optimize
3. The algorithm has been validated - the pure C version matches llama.cpp