# 操作符测试框架

这个框架用于对比我们的Rust实现和llama.cpp的C++实现，既可以查看单个算子的输出，也可以进行性能测试。

## 目录结构

- `src/bin/op_test.rs` - Rust版本的测试程序
- `tests/op_test.cpp` - llama.cpp C++版本的测试程序
- `tests/Makefile` - C++编译脚本

## 使用方法

### Rust版本

```bash
# 看输出
cargo run --release --bin op_test -- --op rms_norm --verbose

# 测性能 (1000次迭代)
cargo run --release --bin op_test -- --op rms_norm --bench 1000

# 指定输入大小
cargo run --release --bin op_test -- --op rms_norm --size 2048 --verbose

# 改变随机种子
cargo run --release --bin op_test -- --op rms_norm --seed 12345 --verbose
```

### C++版本 (llama.cpp)

```bash
cd tests
make

# 看输出
./op_test_cpp --op rms_norm --verbose

# 测性能
./op_test_cpp --op rms_norm --bench 1000

# 使用Makefile的便捷命令
make test-rms
make test-q8k
make test-q5k
```

## 支持的算子

| 算子 | 描述 |
|------|------|
| `rms_norm` | RMS归一化 |
| `silu` | SiLU激活函数 |
| `quantize_q8_k` | Q8_K量化 (用于K-quant输入) |
| `vec_dot_q5_k` | Q5_K x Q8_K点积 |
| `vec_dot_q6_k` | Q6_K x Q8_K点积 |
| `vec_dot_q4_k` | Q4_K x Q8_K点积 |

## 对比测试流程

1. 使用相同的随机种子，确保输入一致
2. 先运行Rust版本看输出
3. 再运行C++版本看输出
4. 对比输出是否一致
5. 如果一致，可以运行性能测试对比速度

### 示例：对比Q8_K量化

```bash
# Rust版本
cargo run --release --bin op_test -- --op quantize_q8_k --verbose --seed 42

# C++版本
cd tests && ./op_test_cpp --op quantize_q8_k --verbose --seed 42
```

对比输出中的 `d` 和 `qs` 值是否一致。

## 添加新算子测试

1. 在 `op_test.rs` 中添加 `test_xxx()` 函数
2. 在 `match args.op.as_str()` 中添加对应分支
3. 在 `op_test.cpp` 中添加相应的测试函数
4. 测试完成后，如果是临时调试，可以删除测试代码
5. 如果是有价值的回归测试，保留在git中

## 调试技巧

### 使用真实模型权重测试

```rust
// 在op_test.rs中加载真实权重
let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
let gguf = GgufFile::load(model_path).unwrap();
let data = gguf.get_tensor_data("blk.0.ffn_gate.weight").unwrap();
// 使用真实数据进行测试
```

### 逐元素对比

```bash
# 导出Rust输出到文件
cargo run --release --bin op_test -- --op xxx --verbose > rust_output.txt

# 导出C++输出到文件
./op_test_cpp --op xxx --verbose > cpp_output.txt

# 对比
diff rust_output.txt cpp_output.txt
```
