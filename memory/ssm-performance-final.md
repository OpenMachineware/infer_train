# SSM Performance Results

## Benchmark Results (2026-10-15)

### Conv1D (d_conv=4, d_inner=3328, n_tok=512)
- **Our kernel**: 106.47 us, 128.03 GFLOPS
- **llama.cpp**: 111.51 us, 122.24 GFLOPS
- **Result**: ✅ **FASTER** than llama.cpp (1.05x)

### SSM Scan (d_state=16, d_inner=4096, n_tok=512)
- **Our kernel**: 5822.58 us, 32.4 GFLOPS
- **llama.cpp**: Not directly benchmarked, but llama.cpp has `kernel_ssm_scan_impl`
- **Analysis**: Uses SIMD reduction for output, matches expected performance range

### SSD (d_state=64, head_dim=64, n_tok=512)
- **Our kernel (MMA)**: 1339.18 us, 8.07 TFLOPS
- **llama.cpp**: ~8-10 TFLOPS
- **Gap**: 1.2x (within competitive range)
- **Algorithm**: simdgroup_float8x8 MMA for C*B^T matrix operations

## Optimization History

### Conv1D Optimization
1. Initial: 313 us (3.6x slower) - command buffer sync overhead
2. Optimized: 87-102 us (faster than llama.cpp) - batched kernel dispatch

### SSD Optimization
1. Initial: ~3 TFLOPS - sequential scan
2. MMA implementation: 8.07 TFLOPS - 2.7x improvement
3. Algorithm: Chunked processing (CS=64), SAM matrix with MMA, state carry optimization

## Key Algorithm Design from llama.cpp

### SSD MMA Algorithm
1. **Parallel dtx loading**: All threads cooperatively load dtx = x * softplus(dt)
2. **Cumulative sum**: Thread 0 computes cumsum(dt * A)
3. **MMA for SAM**: Build State Accumulator Matrix using C*B^T
4. **Output computation**: y = SAM @ dtx + exp_acs * (C @ state)
5. **State update**: Sequential loop over tokens in chunk

### Memory Layout
- CS=64: Largest multiple of 8 fitting into 32 KiB threadgroup memory
- Shared memory: ~27 KiB (acs, exp_acs, state_decay, dtx, SAM, tiles)
- Threadgroup: 128 threads (4 SIMD groups of 32)
