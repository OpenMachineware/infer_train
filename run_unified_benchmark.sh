#!/bin/bash
# Unified GPU benchmark: llama.cpp vs our Rust implementation
# Ensures both run under identical GPU conditions

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp-0.5.0"
BUILD_DIR="$LLAMA_DIR/build-metal"
TEST_BACKEND_OPS="$BUILD_DIR/bin/test-backend-ops"

if [ ! -f "$TEST_BACKEND_OPS" ]; then
    echo "Error: test-backend-ops not found at $TEST_BACKEND_OPS"
    echo "Please build llama.cpp with Metal support first"
    exit 1
fi

echo "=== Unified GPU Benchmark ==="
echo "Testing llama.cpp Metal vs Rust Metal in same session"
echo ""

# Warmup both implementations to stabilize GPU frequency
echo "Step 1: GPU warmup (running both implementations multiple times)..."
echo ""

# Warmup llama.cpp
echo "  Warming up llama.cpp..."
for i in {1..3}; do
    "$TEST_BACKEND_OPS" perf -b MTL0 -o 'MUL_MAT.*q4_k' > /dev/null 2>&1 || true
    sleep 1
done

# Warmup Rust
echo "  Warming up Rust Metal..."
for i in {1..3}; do
    cargo run --release --bin bench_gpu_operators 2>&1 | grep -A5 "MV.*Q4_K.*4096" > /dev/null || true
    sleep 1
done

echo "  Warmup complete"
echo ""

# Now run actual benchmarks
echo "Step 2: Running benchmarks..."
echo ""

echo "=== llama.cpp Metal Performance ==="
"$TEST_BACKEND_OPS" perf -b MTL0 -o 'MUL_MAT.*q4_k' 2>&1 | head -100 || true
echo ""

echo "=== Rust Metal Performance (first run) ==="
cargo run --release --bin bench_gpu_operators 2>&1 | grep -A20 "MV.*Q4_K" || true
echo ""

echo "=== Benchmark Complete ==="
