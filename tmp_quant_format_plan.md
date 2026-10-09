# 多SIMD组量化格式支持计划

## 目标
将multi-SIMD组kernel扩展到所有量化格式，并对比llama.cpp性能

## 当前进度

### F16 ✓
- [x] g1/g2/g4/g8 正确性验证
- [x] 性能测试: g8达到95% llama.cpp
- [x] 提交代码

### 待完成格式
- [ ] Q4_0
- [ ] Q4_1
- [ ] Q5_0
- [ ] Q5_1
- [ ] Q8_0
- [ ] BF16
- [ ] K-quant: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K

## 步骤

### Step 1: 添加量化格式的multi-group实例化
- 在flash_attn_quant.metal添加模板实例化
- 需要为每种格式添加g1/g2/g4/g8版本

### Step 2: 正确性验证
- 使用真实量化数据测试
- 对比CPU参考实现

### Step 3: 性能对比
- 与llama.cpp对比（量化格式应该更公平，因为不依赖F16 intrinsics）
- 使用alternating测试方法

## 关键发现
- F16格式: Rust无intrinsics导致5%性能损失
- 量化格式: 反量化是瓶颈，可能有更公平的对比

## 预期
量化格式性能应该更接近llama.cpp，因为：
1. 不依赖F16计算
2. 反量化可以优化
3. 内存带宽是主要瓶颈
