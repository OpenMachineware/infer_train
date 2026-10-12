# Flash Decode Design Principles

## 核心设计思路（学习自llama.cpp）

### 1. 线程分工策略

**关键原理**：将SIMD组内的32个线程划分为二维网格，实现高效的并行处理。

```
线程布局 (NE=4):
  tx = tiisg % 8  (0-7)   - 负责head_dim维度
  ty = tiisg / 8  (0-3)   - 负责缓存位置维度

示意图：
         tx: 0  1  2  3  4  5  6  7
    ty=0: [  thread 0-7  ]  处理缓存位置0,4,8,12...
    ty=1: [ thread 8-15  ]  处理缓存位置1,5,9,13...
    ty=2: [ thread 16-23 ]  处理缓存位置2,6,10,14...
    ty=3: [ thread 24-31 ]  处理缓存位置3,7,11,15...
```

**为什么这样设计？**
- head_dim=128, 每个线程处理128/8=16个元素
- 每个线程同时处理4个缓存位置（NE=4）
- 指令级并行：GPU可以流水线执行4个位置的dot product

### 2. 缓存处理策略

**关键参数**：C=32（每次迭代处理32个缓存位置）

**原理**：
```
每个SIMD组：Q=1个query, C=32个缓存位置
每个线程：  NE=4个缓存位置
需要的线程：C/NE = 32/4 = 8组线程

线程分工：
  ty=0: 处理位置 0,  4,  8,  12 (4个位置)
  ty=1: 处理位置 1,  5,  9,  13
  ty=2: 处理位置 2,  6,  10, 14
  ty=3: 处理位置 3,  7,  11, 15
  ...

每个位置：
  tx=0-7: 处理head_dim的不同部分
  head_dim=128, 每个线程负责16个元素
```

**为什么是32？**
- 平衡循环开销和寄存器压力
- 经过大量测试验证的最优值
- 足够大以分摊循环开销
- 足够小以适应寄存器数量

### 3. 树形归约策略

**关键原理**：利用SIMD shuffle指令实现无同步的快速归约。

**传统方法的问题**：
```
传统归约：
  1. 每个线程写共享内存
  2. threadgroup_barrier()  <- 昂贵的同步
  3. 某个线程读取并归约
  4. 写回结果
  5. threadgroup_barrier()
```

**树形归约**：
```metal
// NE=4的归约示例
// 将32个线程分为4组，每组8个线程
// 每组内部归约，然后广播

if (NE <= 1) result += simd_shuffle_down(result, 16);
if (NE <= 2) result += simd_shuffle_down(result,  8);
if (NE <= 4) result += simd_shuffle_down(result,  4);
if (NE <= 8) result += simd_shuffle_down(result,  2);
if (NE <=16) result += simd_shuffle_down(result,  1);

// 广播到所有线程
result = simd_shuffle(result, group_leader);
```

**优势**：
- 纯寄存器操作，无需共享内存
- 无同步屏障
- 编译器可以充分优化

### 4. 内存访问模式

**关键原理**：优化内存访问以最大化带宽利用率。

**K/V访问模式**：
```metal
// 量化格式的访问
device const kd4_t * pk = ...;

// 线程tx访问的元素索引
const short i = ii*NL + tx;

// 反量化调用
deq_k_t4(pk + i/nl_k, i%nl_k, mk);
```

**为什么这样访问？**
- `i/nl_k`：确定访问哪个block
- `i%nl_k`：确定block内的哪个4元素组
- `tx`连续增加 → 内存访问连续 → 更好的缓存利用率

**NS10/NS20的作用**：
```metal
// 对于F16等非量化格式
pk4 = (device const k4_t *) (k + ic*args.nb11);
pk4 += ty*NS10/4 + tx;  // NS10是stride

// NS10 = NL相关
// 确保不同ty访问不同的缓存行
```

### 5. 共享内存布局

**关键参数**：
```
SMEM_Q = Q*NSG*PK        // Query缓存
SMEM_S = NSG*SH          // Softmax临时存储
SMEM_O = 2*NSG*Q*PV      // 输出累积

SH = 4*Q*C               // 每个SIMD组的softmax空间
```

**布局原理**：
```
内存地址：
[0, SMEM_Q)            - Query (所有SIMD组共享)
[SMEM_Q, SMEM_Q+SMEM_S) - Softmax (每个SIMD组独立)
[SMEM_Q+SMEM_S, end)   - Output (每个SIMD组独立)
```

**为什么独立？**
- 避免SIMD组间同步
- 每个SIMD组处理不同的KV切片
- 最后需要reduce（在单独的kernel中）

## 性能关键点总结

### 为什么llama.cpp快？

1. **循环次数少**：C=32，循环次数是seq_k/32
2. **指令并行**：NE=4，每个线程处理4个位置
3. **无同步开销**：树形归约避免barrier
4. **内存友好**：连续访问，缓存利用率高

### 我们之前为什么慢？

1. **循环次数多**：CHUNK=4，循环次数是seq_k/4
2. **无指令并行**：每个线程只处理1个位置
3. **同步开销大**：标量归约需要barrier
4. **内存访问差**：小chunk导致频繁内存访问

### 性能差距来源

```
循环次数：  32/4 = 8x
指令并行：  4/1 = 4x
同步开销：  ~1.5x
内存访问：  ~1.5x

总计：8 * 4 * 1.5 * 1.5 ≈ 3-4x
```

## 实现原则

基于以上理解，新的实现应该：

1. **采用相同的线程分工**：tx/ty二维布局
2. **使用相同的关键参数**：NE=4, C=32
3. **实现树形归约**：用simd_shuffle
4. **优化内存访问**：连续访问，利用缓存

**但代码必须自己写**：
- 变量命名自己决定
- 代码组织自己决定
- 具体实现细节可以不同
- 注释风格自己决定

## 下一步

1. 根据这些原理自己写新的kernel
2. 测试性能是否匹配
3. 如果慢了，分析原因并调整
