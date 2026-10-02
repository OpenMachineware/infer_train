# Instruments Time Profiler 定位GPU MV性能差距

## 步骤1: 编译Rust测试

```bash
cd /Users/jia/Desktop/infer_train
cargo build --bin gpu_mv_instrumented --release
```

## 步骤2: 用Instruments分析Rust

```bash
# 打开Instruments GUI
open -a Instruments

# 或者命令行运行
instruments -t "Time Profiler" -D rust_profile.trace target/release/gpu_mv_instrumented
```

在Time Profiler中查看:
1. 按调用栈展开
2. 找到 `mv_q4_k_f32` 函数
3. 看时间花在哪些子调用:
   - `RefCell::borrow_mut` ?
   - `Option::unwrap` ?
   - `new_command_buffer` ?
   - `set_buffer` ?
   - `dispatch_thread_groups` ?
   - `wait_until_completed` ?

## 步骤3: 编译llama.cpp对比测试

```bash
cd /Users/jia/Desktop/infer_train/native

# 使用CMake编译llama.cpp
mkdir -p llama.cpp-0.5.0/build
cd llama.cpp-0.5.0/build
cmake .. -DGGML_METAL=ON
make -j8

# 编译测试程序
cd /Users/jia/Desktop/infer_train/native/tests
clang++ -O3 -I../llama.cpp-0.5.0/ggml/include \
    -L../llama.cpp-0.5.0/build/ggml/src \
    -lggml -lggml-metal -framework Metal -framework Foundation \
    -o llama_q4_k_test test_llama_metal_q4_k.cpp
```

## 步骤4: 用Instruments分析llama.cpp

```bash
# 运行10000次迭代
instruments -t "Time Profiler" -D llama_profile.trace ./llama_q4_k_test 4096 4096 10000
```

## 步骤5: 对比分析

打开两个trace文件，对比：
1. **GPU kernel执行时间** - 应该相同（相同shader）
2. **CPU开销** - 看 `ggml_backend_graph_compute` vs `mv_q4_k_f32` 的差异
3. **内存分配** - 看buffer创建/复用模式

## 预期发现

| 项目 | Rust | llama.cpp | 原因 |
|------|------|-----------|------|
| GPU kernel | 相同 | 相同 | 同一个Metal模板 |
| Buffer alloc | 每次检查size | 预分配 | RefCell开销 |
| Command buffer | 每次new | 可能复用 | 待验证 |
| set_buffer | 每次调用 | 可能批量 | 待验证 |

## 重点关注

1. **Call tree** - 展开看哪个函数占时间最多
2. **Invert call tree** - 按时间排序，看热点
3. **Symbolicate** - 确保符号表加载，能看到函数名
