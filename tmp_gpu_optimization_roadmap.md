# GPU Decode Performance Optimization Roadmap

## Current Status (2026-10-23 Session 11)

**Performance** (Q4_K models, Qwen3-0.6B):
| Metric | We | llama.cpp | Gap |
|--------|------|-----------|------|
| Prefill | 2571 t/s | 920 t/s | **0.36x (faster!)** |
| Decode | 120-125 t/s | 184 t/s | **1.47-1.53x** |
| **Overall BW** | **20.2%** | **31.0%** | **11pp** |
| **Single Kernel** | **5.0%** | **?** | **Unknown** |

**Root Cause**: **GEMV kernel带宽利用率极低**

## Bandwidth Analysis Results

### Single Kernel Profiling
```
Kernel                    M      K   Time(μs)  BW(GB/s)   Util%
-----------------------------------------------------------------
Q projection           2048   1024     265.52      4.49    4.5%
K projection           1024   1024     271.46      2.20    2.2%
V projection           1024   1024     331.94      1.80    1.8%
Gate projection        3072   1024     288.62      6.19    6.2%
Up projection          3072   1024     276.49      6.46    6.5%
Down projection        1024   3072     309.75      5.77    5.8%
-----------------------------------------------------------------
AVERAGE                                   4.48    4.5%
```

### Dispatch Overhead Analysis
```
Separate command buffers (6 dispatches):  3.1% util
Single command buffer (batched):          5.0% util
Speedup from batching:                    1.63x
```

### Key Findings
1. **Kernel本身是主要问题**: 5% vs 31%的理论带宽
2. **Dispatch overhead次要**: 批量化只提升1.63x
3. **整体decode vs 单kernel**: 20% vs 5% = 多kernel并行执行有overlap

## Optimization Strategy

### 目标: 提升GEMV kernel带宽利用率从5%到更高

**调查方向:**
1. **对比llama.cpp的kernel实现**
   - 检查thread配置
   - 检查memory access pattern
   - 检查是否有cache优化

2. **分析我们的kernel瓶颈**
   - 是否有uncoalesced memory access?
   - 是否有thread divergence?
   - 是否有insufficient parallelism?

3. **测量llama.cpp的kernel带宽**
   - 用同样的方法测量llama.cpp
   - 对比数值差异

### Next Steps

1. **对比llama.cpp Q4_K kernel**
   - 检查llama.cpp源码
   - 找出关键差异

2. **尝试优化方向**
   - 增加thread count?
   - 改变memory access pattern?
   - 使用不同dispatch configuration?

## Progress Log

### Session 11
1. ✅ FusedGateUp优化: 20%提升 (101→120 t/s)
2. ✅ 带宽分析工具: 发现核心问题
3. ✅ Kernel-level profiling: 发现kernel本身是瓶颈 (5% util)
4. ✅ Dispatch overhead测试: 批量化提升1.63x

### Session 10
1. ✅ NWG parallelization for Flash Attention
2. ✅ Verified prefill performance

### Session 9
1. ✅ Vec registry + dtype cache
2. ✅ Dispatch fix

## Key Insights

1. **Decode是memory-bound**: 性能 = 带宽利用率 × 理论带宽
2. **GEMV占90%内存访问**: 优化GEMV = 优化整体
3. **Kernel带宽利用率极低**: 5% vs 31%理论
4. **Dispatch overhead次要**: 批量化只提升1.63x
5. **Kernel实现是关键**: 需要深入分析kernel代码
