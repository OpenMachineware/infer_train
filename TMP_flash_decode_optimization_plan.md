# Flash Decode 优化计划

## 目标

通过对比我们的实现与llama.cpp在**每个步骤**上的性能，找出差距最大的环节，有针对性地优化。

---

## 分析方法

### 1. 分解Flash Decode为具体步骤

Flash Decode的主要步骤：

```
1. Query加载      - 从global memory加载query到shared memory
2. K读取          - 从KV cache读取Key（可能需要解量化）
3. QK计算         - 计算Query和Key的点积
4. QK归约         - 在SIMD group内归约得到完整score
5. Score存储      - 将score存储到shared memory
6. Online Softmax - 更新M、S值，rescale输出
7. V读取          - 从KV cache读取Value（可能需要解量化）
8. V累加          - 使用softmax weight累加Value到输出
9. 输出归一化     - 最终归一化并写入global memory
```

### 2. 测量方法

#### 方法A: 微基准测试（推荐）
为每个步骤创建独立的kernel，只执行该步骤：
- 可以精确测量每个步骤的时间
- 可以对比我们vs llama.cpp的实现
- 需要创建多个测试kernel

#### 方法B: 指令计数分析
分析Metal assembly，统计每个步骤的指令数：
- 准确度更高
- 需要理解GPU microarchitecture
- 需要手动分析shader assembly

#### 方法C: 理论性能分析
计算每个步骤的理论FLOPS和带宽需求：
- GPU峰值性能：~10 TFLOPS (M系列GPU)
- 内存带宽：~400 GB/s
- 可以计算出理论最优时间

### 3. 对比基准

从之前的benchmark结果：
- 我们Multi-WG: 72.71 GFLOPS
- llama.cpp: ~90 GFLOPS
- 差距：约20%

**关键问题**：这20%的差距主要来自哪个步骤？

---

## 优化步骤

### Step 1: 建立测试框架

创建可以测量单个步骤的kernel：

```metal
// 1. 只测试Query加载
kernel void test_query_load(...);

// 2. 只测试K读取+解量化
kernel void test_k_read(...);

// 3. 只测试QK计算
kernel void test_qk_compute(...);

// 以此类推...
```

### Step 2: 测量我们的每个步骤

运行测试kernel，记录每个步骤的时间。

### Step 3: 实现llama.cpp的对应步骤

从llama.cpp代码中提取每个步骤的实现，创建对应的测试kernel。

### Step 4: 对比分析

制作对比表格：

| 步骤 | 我们 (µs) | llama.cpp (µs) | 差距 (%) | 优先级 |
|------|-----------|----------------|----------|--------|
| Query加载 | ? | ? | ? | ? |
| K读取 | ? | ? | ? | ? |
| QK计算 | ? | ? | ? | ? |
| ... | ? | ? | ? | ? |

### Step 5: 针对性优化

针对差距最大的步骤进行优化：
- 如果是内存访问：优化访问模式
- 如果是计算：优化指令使用
- 如果是同步：减少barrier

---

## 具体执行计划

### Phase 1: 基础测试框架 (当前)

- [x] 创建Multi-WG kernel
- [x] 测量Partial vs Finalize时间
- [x] 发现Partial占90%时间

### Phase 2: 细粒度步骤测量

1. **Query加载测试**
   - 创建只加载query到shared memory的kernel
   - 测量时间
   - 对比llama.cpp

2. **K循环测试**
   - 分离K读取、QK计算、Score存储
   - 分别测量

3. **V循环测试**
   - 分离V读取、V累加
   - 分别测量

### Phase 3: 对比与优化

1. 针对差距最大的步骤
2. 分析llama.cpp的实现差异
3. 应用相同的优化

### Phase 4: 自适应优化

1. 实现自适应NWG选择
2. 根据seq_k动态调整
3. 测试不同场景

---

## 预期结果

通过这种方法，我们应该能够：
1. 找出最耗时的步骤
2. 知道该步骤与llama.cpp的具体差距
3. 针对性地应用优化
4. 将性能差距从20%缩小到<10%

---

## 当前进展

### 最新测试结果 (2026-10-11)

**分步骤对比测试 (Our vs llama.cpp style)**:

| 步骤 | Our (µs) | llama (µs) | 差距 | 分析 |
|------|----------|------------|------|------|
| Query Load | 220.57 | 207.12 | **-6.5%** | 稍慢，可能内存访问模式 |
| QK Compute | 230.44 | 227.75 | -1.2% | 基本持平 |
| V Accum | 226.89 | 246.52 | **+8.0%** | 我们更快 ✅ |
| **Complete** | 261.42 | 251.07 | **-4.1%** | 整体差距小 |

**完整kernel性能**:
- Vectorized kernel (F16, seq_k=2048): **72.71 GFLOPS**
- Vectorized kernel (Q4_0, seq_k=2048): **68.21 GFLOPS**
- llama.cpp参考: **~90 GFLOPS**
- 当前差距: **约19%**

### 关键发现

1. **Query Load是主要差距**:
   - 我们慢6.5%
   - 可能是因为内存访问模式差异

2. **QK Compute基本持平**:
   - 差距仅1.2%
   - 核心计算效率相当

3. **V Accumulation我们更快**:
   - 快8%
   - 说明我们的V处理效率不错

4. **整体差距4.1% (简化kernel)**:
   - 在测试kernel中差距很小
   - 生产kernel差距19%可能来自其他因素

### 差距来源分析

**测试kernel vs 生产kernel差距**:
- 测试kernel: 4.1%差距 (261.42 vs 251.07 µs)
- 生产kernel: 19%差距 (72.71 vs ~90 GFLOPS)

**可能原因**:
1. 生产kernel有更多处理（边界检查、padding等）
2. 生产kernel的量化格式处理更复杂
3. 内存布局可能不同

### llama.cpp关键模式

从fa_vec_common.metal分析:

1. **Query加载** (lines 116-127):
   ```metal
   for (short i = tiisg; i < PK4; i += NW) {
       sq4[qq*PK4 + i] = (q4_t) q4[i];
   }
   ```
   - 使用stride循环（虽然HEAD_DIM=128时只有一次迭代）
   - q4_t类型（可能是half4，我们用float4）

2. **QK计算** (lines 322-328):
   ```metal
   const k4_t k_elem = pk4[cc*NE*NS10/4 + ii*NL];
   mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
   ```
   - 使用双循环展开 (Q x C/NE)
   - 索引计算: cc*NE*NS10/4 + ii*NL

3. **共享内存类型**:
   - llama.cpp: `threadgroup q4_t * sq4` (half4)
   - 我们: `threadgroup float4 s_query` (float4)
   - half4占用更少共享内存，可能影响缓存效率

---

## 下一步计划

### 立即行动

1. ~~**优化Query Load**~~: 尝试half4失败，性能下降32%
   - 原因：half→float转换开销太大
   - 结论：保持float4

2. ~~**替换barrier**~~: 用simdgroup_barrier替代threadgroup_barrier
   - 结果：性能没有显著变化
   - 结论：两者性能相近

3. **数据类型修正**: 修正benchmark中query数据类型
   - FP16→FP32，与kernel期望一致
   - 性能略有下降（内存带宽增加）

4. **关键发现**: 单独步骤测试显示我们更快
   - Query Load: +6.6%
   - QK Compute: +7.1%
   - V Accum: +2.4%
   - **但完整kernel慢16%**

5. **问题定位**: 完整kernel有额外overhead
   - 不是单个步骤的问题
   - 可能是循环结构或barrier使用
   - 需要进一步分析

### 性能状态

**当前性能** (修复数据类型后):
- F16: 48.92 GFLOPS (seq_k=2048)
- Q4_0: 44.22 GFLOPS (seq_k=2048)
- llama.cpp参考: ~90 GFLOPS
- **差距: 约45%**

**差距扩大的原因**:
- 修正数据类型后，Query从2KB→4KB
- 单独步骤测试都更快，说明算法正确
- 完整kernel有额外overhead需要分析
