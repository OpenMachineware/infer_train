# Flash Decode步骤对比分析

## 当前结果

**我们的实现**：
- Complete kernel: 103.23 GFLOPS
- QK Compute单独: 60.47 GFLOPS

## 下一步：创建llama.cpp对应的步骤测试

### 需要从llama.cpp提取的关键kernel：

1. **Query Load (fa_vec_common.metal)**
   - Lines 116-127: Load query to shared memory
   - 简单的global->shared memory copy

2. **QK Computation (fa_vec_common.metal)**
   - Lines 322-328: Direct F16 path
   - 关键：如何读取K cache，如何计算dot product

3. **Softmax (fa_vec_common.metal)**
   - Lines 345-425: Online softmax with reduction
   - 关键：reduction pattern，M/S update

4. **V Accumulation (fa_vec_common.metal)**
   - Lines 470-482: Direct F16 V read
   - Lines 500-545: Reduction and storage

### 计划：

1. 复制llama.cpp的代码片段
2. 创建独立的测试kernel
3. 测量每个步骤
4. 制作对比表格

### 关键差异点需要检查：

1. **Query Load**:
   - 是否使用了不同的load pattern？
   - 是否有预取优化？

2. **QK Compute**:
   - dot product的实现是否相同？
   - 是否使用了fma指令？
   - 是否有循环展开差异？

3. **Softmax**:
   - reduction pattern是否相同？
   - exp函数实现是否相同？

4. **V Accumulation**:
   - 内存访问模式是否相同？
   - 是否有store forwarding优化？
