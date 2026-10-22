# GPU Decode Performance Optimization Roadmap

## Current Status (2026-10-23 Session 11)

**Performance** (Q4_K models, Qwen3-0.6B):
| Metric | We | llama.cpp | Gap |
|--------|------|-----------|------|
| Prefill | 2571 t/s | 920 t/s | **0.36x (faster!)** |
| Decode | 120-125 t/s | 184 t/s | **1.47-1.53x** |
| **Bandwidth** | **20.2%** | **31.0%** | **11% points** |

**Root Cause**: 带宽利用率不足

## Bandwidth Analysis

### Memory Access Per Token (Qwen3-0.6B, 28 layers)
```
Per layer: 6.02 MB
Total: 168.48 MB/token

GEMV (权重读取): 90% of bandwidth
Flash Attention: <5% of bandwidth
KV Cache: <5% of bandwidth
```

### Bandwidth Utilization
```
理论峰值 (M1): 100 GB/s
我们 (120 t/s): 20.2 GB/s → 20.2%
llama.cpp (184 t/s): 31.0 GB/s → 31.0%

差距: 我们比llama.cpp少用11%的带宽
```

## Optimization Strategy

### 目标: 将带宽利用率从20%提升到31%

**Priority 1: GEMV kernel bandwidth optimization**
- GEMV占90%的内存访问
- 当前利用率低 → 优化GEMV = 优化整体

**调查方向:**
1. **实际测量每个kernel的带宽**
   - 在Metal层添加精确计时
   - 计算每个GEMV kernel的实际带宽
   - 对比我们 vs llama.cpp的GEMV带宽

2. **分析GEMV kernel的内存访问模式**
   - 是否有cache miss?
   - 是否有bank conflict?
   - 是否有coalescing问题?

3. **对比kernel实现细节**
   - thread配置
   - memory访问顺序
   - SIMD使用方式

### 测试计划

1. **创建kernel-level bandwidth profiler**
   - 测量单个GEMV kernel的时间
   - 计算实际带宽
   - 对比Q4_K GEMV在我们和llama.cpp的表现

2. **逐个kernel对比**
   - Q projection GEMV
   - K projection GEMV
   - Gate/Up/Down projection GEMV
   - 找出哪个kernel差距最大

3. **基于数据的优化**
   - 针对差距最大的kernel优化
   - 测量优化效果
   - 迭代

## Progress Log

### Session 11
1. ✅ FusedGateUp优化: 20%提升 (101→120 t/s)
2. ✅ 带宽分析工具: 发现核心问题
3. ⏳ 开始kernel-level带宽profiling

### Session 10
1. ✅ NWG parallelization for Flash Attention
2. ✅ Verified prefill performance

### Session 9
1. ✅ Vec registry + dtype cache
2. ✅ Dispatch fix

## Key Insights

1. **Decode是memory-bound**: 性能 = 带宽利用率 × 理论带宽
2. **GEMV占90%内存访问**: 优化GEMV = 优化整体
3. **11%带宽利用率差距**: 这是量化的优化目标
4. **需要kernel-level数据**: 猜测不如测量
