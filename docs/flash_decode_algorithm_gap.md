# Flash Decode算法差距分析

## 问题1：V Accumulation内存访问模式

### llama.cpp (正确)
```metal
// 预先调整指针，考虑ty和tx
device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);
pv4 += ty*NS20/4 + tx;  // 关键！

FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
        const v4_t v_elem = pv4[cc*NE*NS20/4 + ii*NL];  // 连续访问
        lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
    }
}
```

### 我们的实现 (错误)
```metal
// 每次都重新计算偏移
device const VBlock* v_block = ... + cache_idx * HEAD_DIM + head * HEAD_DIM;

for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
    const int elem_idx = j * NL + tx;  // 非连续！
    DEQUANT_V(v_block + elem_idx / DEQUANT_LANES_V, ...);
}
```

**差距**：
- llama.cpp: 指针预先调整，后续访问 `pv4[ii*NL]` 是连续的（步长=NL=8）
- 我们的: 每次都计算 `elem_idx = j*NL + tx`，导致跨步访问

## 问题2：本地累加器

### llama.cpp
```metal
o4_t lo[Q][DV4/NL];  // 本地寄存器数组
FOR_UNROLL (...) {
    lo[qq][ii] += ...;  // 累积到寄存器
}
// 最后才累加到全局输出
```

### 我们
```metal
float4 O[HEAD_DIM / (4 * NL)];  // 可能是寄存器
// 但每次cc循环都访问
for (int cc = 0; cc < C/NE; ++cc) {
    O[j] += ...;  // 每次都访问
}
```

**差距**：
- llama.cpp: V accumulation先累积到本地寄存器 `lo`，减少全局状态访问
- 我们: 没有这个优化

## 问题3：Output scaling优化

### llama.cpp
```metal
// O = diag(ms)*O
if ((DV4/NL % NW == 0) || ty == 0) {
    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
        so4[qq*DV4 + ii*NL] *= ms;
    }
}
```

### 我们
```metal
// 所有线程都执行
for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
    O[j] *= ms;
}
```

**差距**：
- llama.cpp: 条件判断避免某些ty重复执行
- 我们: 所有线程都执行，可能有冗余

## 内存访问连续性分析

### Q*K^T 阶段
- llama.cpp: `pk4[cc*NE*NS10/4 + ii*NL]`
  - 外层循环cc，步长 NE*NS10/4 = 4*32/4 = 32
  - 内层循环ii，步长 NL = 8
  - 对于固定的cc和ii，tx连续访问：tx=0,1,2,...,7访问连续内存

- 我们: `elem_idx = ii * NL + tx`
  - tx=0: elem_idx = 0, 8, 16, 24 (非连续！)
  - tx=1: elem_idx = 1, 9, 17, 25 (非连续！)

### V Accumulation 阶段
- llama.cpp: `pv4[cc*NE*NS20/4 + ii*NL]`
  - 预先调整指针后，tx连续访问
  - 步长设计让内存访问对齐

- 我们: `cache_idx = ic + tiisg`
  - tiisg = ty*NL + tx
  - 不同tiisg访问完全不同的cache item
  - 内存访问跨度很大

## 性能影响估算

假设内存延迟是主要瓶颈：
- 连续访问: 延迟~100 cycles
- 跨步访问（stride=8）: 延迟~200-400 cycles（取决于缓存）

对于seq_k=1024，每个head需要：
- 1024次Q*K^T
- 1024次V accumulation

如果每次访问多延迟200 cycles：
- 额外延迟 = 1024 * 200 * 32 heads = 6.5M cycles
- 在1GHz GPU上 = 6.5ms
- 对于44.50 GFLOPS的kernel（约377µs），这是巨大的开销

## 结论

主要差距在于：
1. **内存访问模式**：非连续访问导致缓存失效
2. **指针预调整**：llama.cpp预先调整指针避免每次计算偏移
3. **本地累加器**：减少全局状态访问
