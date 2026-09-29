# K-quant GPU性能对比 - 完整测试 (M=4096, K=4096)

## 测试方法
- 同环境直接对比，无预热tricks
- 每个测试运行10次取平均值
- GEMM使用N=64 (batch size)

---

## MV模板 (Decode)

| Format | llama.cpp | Rust | Ratio | 状态 |
|--------|-----------|------|-------|------|
| Q2_K | 33.60 GFLOPS | 51.94 GFLOPS | 155% | ✅ |
| Q3_K | 32.58 GFLOPS | 50.16 GFLOPS | 154% | ✅ |
| Q4_K | 32.32 GFLOPS | 41.00 GFLOPS | 127% | ✅ |
| Q5_K | 31.48 GFLOPS | 37.53 GFLOPS | 119% | ✅ |
| Q6_K | 53.51 GFLOPS | 44.42 GFLOPS | 83% | ❌ |

---

## GEMM模板 (Prefill, N=64)

| Format | llama.cpp | Rust | Ratio | 状态 |
|--------|-----------|------|-------|------|
| Q2_K | 998.92 GFLOPS | 2258.73 GFLOPS | 226% | ✅ |
| Q3_K | 1001.48 GFLOPS | 3619.81 GFLOPS | 361% | ✅ |
| Q4_K | 970.75 GFLOPS | 2616.89 GFLOPS | 269% | ✅ |
| Q5_K | 1021.64 GFLOPS | 1982.61 GFLOPS | 194% | ✅ |
| Q6_K | 1028.98 GFLOPS | 2559.88 GFLOPS | 249% | ✅ |

---

## 总结

- **MV**: 4/5 通过，Q6_K需要调查
- **GEMM**: 5/5 全部通过，性能优秀（194-361%）

---

## 测试文件

**llama.cpp**:
- MV: `native/bench_template_mv.mm`
- GEMM: `native/bench_template_gemm.mm`

**Rust**:
- MV: `src/bin/test_qXk_mv.rs`
- GEMM: `src/bin/test_qXk_gemm.rs`
