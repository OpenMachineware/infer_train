# K-quant MV GPU性能对比 (M=4096, K=4096)

## 测试方法
- 同环境直接对比
- 每个测试运行10次取平均值
- 无预热tricks

## 结果

| Format | llama.cpp | Rust | Ratio | 状态 |
|--------|-----------|------|-------|------|
| Q2_K | 33.60 GFLOPS | 51.94 GFLOPS | 155% | ✅ |
| Q3_K | 32.58 GFLOPS | 50.16 GFLOPS | 154% | ✅ |
| Q4_K | 32.32 GFLOPS | 41.00 GFLOPS | 127% | ✅ |
| Q5_K | 31.48 GFLOPS | 37.53 GFLOPS | 119% | ✅ |
| Q6_K | 53.51 GFLOPS | 44.42 GFLOPS | 83% | ❌ |

## 测试文件

**llama.cpp**: `native/bench_template_mv.mm`
**Rust**: `src/bin/test_qXk_mv.rs`
