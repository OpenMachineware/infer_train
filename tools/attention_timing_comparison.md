# Flash Attention 绝对时间对比

## 测量方法

### 我们的实现（Rust）
- 工具：profile_prefill_detailed.rs
- 1000次迭代取平均

### llama.cpp实现（C++）
- 直接调用llama.cpp的ops.cpp中的实现
- 1000次迭代取平均

## 测试配置

- Tokens: 64
- Heads: 16
- KV Heads: 8
- Head dim: 128
- Tile size: Q=64, KV=64

## 已知对比结果

### GEMM 64×896×896

| 实现 | 绝对时间 | GFLOPS |
|------|---------|--------|
| 我们的simd_gemm | 1392 µs | 73.8 |
| llama.cpp simd_gemm | 1343 µs | 76.5 |
| **差距** | **1.04x** | |

**结论：GEMM性能几乎相同，不是瓶颈**

### 下一步测量

需要测量：
1. 完整的attention forward pass（包含所有步骤）
2. QKV projection
3. FFN (gate + up + down)
4. 整体forward pass

对比每个部分的绝对时间，找出真正的差距。
