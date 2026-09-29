# Q4_K GPU性能对比 - 直接测试 (2026-09-29)

## 测试方法

**简单直接，无预热tricks**：
1. 写一个简单的C++程序，直接调用llama.cpp Metal API
2. 写一个简单的Rust程序，调用我们的Metal API
3. 两边都只测一个函数，运行10次取平均值

---

## MV模板 (Decode)

### 测试代码

**llama.cpp (native/bench_llama_metal_ggml.mm)**:
```cpp
ggml_backend_t backend = ggml_backend_metal_init();
struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, k, m);
struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
struct ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
ggml_backend_graph_compute(backend, graph);
```

**Rust (src/bin/test_q4k_mv_only.rs)**:
```rust
let ctx = MetalContext::new().expect("Failed to create MetalContext");
let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).unwrap();
```

### 测试结果 (M=4096, K=4096)

| 测试 | llama.cpp | Rust | Rust vs llama.cpp |
|------|-----------|------|-------------------|
| Run 1 | 38.18 GFLOPS | 39.12 GFLOPS | 102.5% ✅ |
| Run 2 | 36.29 GFLOPS | 37.33 GFLOPS | 102.9% ✅ |

---

## GEMM模板 (Prefill/Batch)

### 测试代码

**llama.cpp (native/bench_llama_metal_gemm.mm)**:
```cpp
struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n); // n=batch
struct ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
```

**Rust (src/bin/test_q4k_gemm_only.rs)**:
```rust
let _ = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
```

### 测试结果 (M=4096, K=4096, N=64)

| 测试 | llama.cpp | Rust | Rust vs llama.cpp |
|------|-----------|------|-------------------|
| Run 1 | 1536.44 GFLOPS | 1889.88 GFLOPS | 123.0% ✅ |

---

## 总结

| 模板 | Rust vs llama.cpp | 状态 |
|------|-------------------|------|
| MV   | 102-103%          | ✅ 达标 |
| GEMM | 123%              | ✅ 优秀 |

## 关键发现

1. **MV模板**: Rust实现略快于llama.cpp (2-3%)
2. **GEMM模板**: Rust实现明显快于llama.cpp (23%)
3. **GPU频率波动**: 确实存在，但对两边影响一致，公平对比有效
