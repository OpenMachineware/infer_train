# 多SIMD组量化格式支持计划

## 目标
将multi-SIMD组kernel扩展到所有量化格式，并对比llama.cpp性能

## 当前进度

### F16 ✓
- [x] g1/g2/g4/g8 正确性验证
- [x] 性能测试: g8达到95% llama.cpp
- [x] 提交代码

### Q4_0 ✓
- [x] g1/g2/g4/g8 模板实例化
- [x] 性能对比完成：
  - seq_k ≤ 2048: 快7-48%
  - seq_k = 4096: 慢38% (需优化自适应策略)
- [x] 提交代码

### Q8_0 ✓
- [x] g1/g2/g4/g8 模板实例化
- [x] 性能对比完成：
  - seq_k ≤ 1024: 快7-46%
  - seq_k = 4096: 慢58% (需优化自适应策略)
- [x] 提交代码

### Q4_1/Q5_0/Q5_1/BF16 ✓
- [x] g1/g2/g4/g8 模板实例化
- [x] 正确性验证 (使用F16 K/V作为CPU参考, max_diff < 6e-3)
- [x] shmem修复: 512 + nsg*16 + nsg*512 bytes
- [ ] 性能对比 (llama.cpp benchmark工具需要修复)

### K-quant: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- [x] 单SIMD组kernel已实现 (flash_decode_attention_kquant)
- [ ] 多SIMD组kernel需要新模板 (type4x4 vs type4不兼容)
- [ ] 正确性验证
- [ ] 性能对比

## 关键发现

### 性能
1. **短序列优势** (seq_k ≤ 2048): 量化格式快7-48%
2. **长序列劣势** (seq_k = 4096): llama.cpp快38-58%
   - 原因：我们的自适应策略使用8个SIMD组，llama.cpp用2个
   - 解决方案：调整自适应策略，seq_k=4096应使用2-4组

### 技术
1. F16格式: Rust无intrinsics导致5%性能损失
2. 量化格式: shmem大小计算关键 - 512 + nsg*16 + nsg*512 bytes
3. K-quant: 使用type4x4反量化，与标准Q格式的type4不兼容

## 下一步

1. **完成Q4_1/Q5_0/Q5_1/BF16性能对比**
   - 修复llama.cpp benchmark工具 (缺少ggml-metal-impl.h)
   - 或使用内置GFLOPS测量

2. **K-quant验证**
   - 测试现有单SIMD组kernel的正确性
   - 如需要多SIMD组优化，创建type4x4模板变体

3. **优化自适应策略**
   - seq_k=4096应使用2-4个SIMD组，而不是8个
   - 分析llama.cpp的nsg选择策略
