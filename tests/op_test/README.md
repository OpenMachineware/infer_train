# Operator Test Framework

This framework compares our Rust implementation against llama.cpp's C++ implementation.
It supports both output inspection and performance benchmarking.

## Directory Structure

- `src/bin/op_test.rs` - Rust test program
- `tests/op_test/op_test.cpp` - llama.cpp C++ test program
- `tests/op_test/Makefile` - C++ build script

## Usage

### Rust Version

```bash
# View output
cargo run --release --bin op_test -- --op rms_norm --verbose

# Benchmark (1000 iterations)
cargo run --release --bin op_test -- --op rms_norm --bench 1000

# Specify input size
cargo run --release --bin op_test -- --op rms_norm --size 2048 --verbose

# Change random seed
cargo run --release --bin op_test -- --op rms_norm --seed 12345 --verbose
```

### C++ Version (llama.cpp)

```bash
cd tests/op_test
make

# View output
./op_test_cpp --op rms_norm --verbose

# Benchmark
./op_test_cpp --op rms_norm --bench 1000

# Using Makefile shortcuts
make test-rms
make test-q8k
make test-q5k
```

## Supported Operators

| Operator | Description |
|----------|-------------|
| `rms_norm` | RMS normalization |
| `silu` | SiLU activation function |
| `quantize_q8_k` | Q8_K quantization (input for K-quant) |
| `vec_dot_q5_k` | Q5_K x Q8_K dot product |
| `vec_dot_q6_k` | Q6_K x Q8_K dot product |
| `vec_dot_q4_k` | Q4_K x Q8_K dot product |

## Comparison Testing

1. Use the same random seed for consistent inputs
2. Run Rust version first to see output
3. Run C++ version to see output
4. Compare outputs for consistency
5. If consistent, run benchmarks to compare performance

### Example: Compare Q8_K Quantization

```bash
# Rust version
cargo run --release --bin op_test -- --op quantize_q8_k --verbose --seed 42

# C++ version
cd tests/op_test && ./op_test_cpp --op quantize_q8_k --verbose --seed 42
```

Compare the `d` and `qs` values in the output.

## Adding New Operator Tests

1. Add `test_xxx()` function in `op_test.rs`
2. Add corresponding branch in `match args.op.as_str()`
3. Add corresponding test function in `op_test.cpp`
4. After testing, delete temporary debugging code
5. Keep valuable regression tests in git

## Debugging Tips

### Test with Real Model Weights

```rust
// Load real weights in op_test.rs
let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
let gguf = GgufFile::load(model_path).unwrap();
let data = gguf.get_tensor_data("blk.0.ffn_gate.weight").unwrap();
// Test with real data
```

### Element-by-Element Comparison

```bash
# Export Rust output
cargo run --release --bin op_test -- --op xxx --verbose > rust_output.txt

# Export C++ output
./op_test_cpp --op xxx --verbose > cpp_output.txt

# Compare
diff rust_output.txt cpp_output.txt
```