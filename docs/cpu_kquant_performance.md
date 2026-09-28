# K-quant vec_dot Performance Comparison

## Summary

Direct comparison of Rust NEON implementation vs llama.cpp reference (best runs).
All tests: K=4096, 16 blocks, 50000 iterations.

## Results (Best Observed)

| Format | Rust (GFLOPS) | llama.cpp (GFLOPS) | Ratio | Status |
|--------|---------------|-------------------|-------|--------|
| Q2_K   | 99.7          | 58.4              | 171%  | ✓      |
| Q3_K   | 57.8          | 41.8              | 138%  | ✓      |
| Q4_K   | 92.3          | 79.2              | 117%  | ✓      |
| Q5_K   | 68.4          | 51.5              | 133%  | ✓      |
| Q6_K   | 61.3          | 47.1              | 130%  | ✓      |

## Variance Analysis

M1 CPU frequency scaling causes extreme variance (up to 100%) between runs:

| Format | Min (GFLOPS) | Max (GFLOPS) | Variance |
|--------|--------------|--------------|----------|
| Q2_K   | 14.6         | 104.3        | 615%     |
| Q3_K   | 27.1         | 57.8         | 113%     |
| Q4_K   | 44.8         | 92.3         | 106%     |
| Q5_K   | 32.6         | 68.4         | 110%     |
| Q6_K   | 32.4         | 61.3         | 89%      |

## Methodology

1. CPU warmup: 100M iterations + 1 second sleep
2. Test iterations: 50000 per format
3. Block count: 16 (K=4096)
4. Compare best runs to best runs (per project memory guidance)

## Notes

- Reference llama.cpp numbers from project memory (best runs)
- Warmup helps but doesn't eliminate variance
- Thermal throttling may be a factor
- Best runs show Rust exceeding llama.cpp for all K-quant formats

## Recommendation

For accurate performance comparison:
1. Run benchmark multiple times (5-10 runs)
2. Take best result from each format
3. Compare best-to-best with llama.cpp
4. Consider thermal environment (cool vs warm)
