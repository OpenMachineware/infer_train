#!/bin/bash
# Compare inference performance: llama.cpp vs infer_train

set -e

MODEL="${1:-/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf}"
PROMPT="${2:-Hello}"
N_TOKENS="${3:-50}"

echo "=== Model: $MODEL ==="
echo "=== Prompt: $PROMPT ==="
echo "=== N tokens: $N_TOKENS ==="

# llama.cpp benchmark
echo -e "\n=== llama.cpp (Metal) ==="
cd /Users/jia/Desktop/infer_train/llama.cpp-0.6.0/build/bin
./llama-cli -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" 2>&1 | grep -E "(Prompt:|Generation:|t/s)"

# llama-bench for detailed metrics
echo -e "\n=== llama-bench detailed ==="
./llama-bench -m "$MODEL" -p 512,1 -n 32,1 -r 3 2>&1 | grep -E "(model|pp|tg|t/s)"

# Our Rust implementation
echo -e "\n=== infer_train (Rust/Metal) ==="
cd /Users/jia/Desktop/infer_train
cargo run --release --features metal --bin test-gpu-inference "$MODEL" 2>&1 || echo "Binary not built or failed"

echo -e "\n=== Done ==="
