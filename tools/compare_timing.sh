#!/bin/bash
# Simple timing comparison between our code and llama.cpp

MODEL="$HOME/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf"

echo "=== TIMING COMPARISON ==="
echo "Model: $MODEL"
echo ""

# Our implementation
echo "--- Our Implementation ---"
echo "1. Prefill (single token):"
start=$(python3 -c "import time; print(int(time.time() * 1000000))")
cargo run --release --features metal --bin bench-decode -- "$MODEL" 2>&1 | grep "Prefill:"
end=$(python3 -c "import time; print(int(time.time() * 1000000))")
echo ""

echo "2. Decode (10 tokens):"
cargo run --release --features metal --bin bench-decode -- "$MODEL" 2>&1 | grep -E "Token|Average"
echo ""

# llama.cpp
echo "--- llama.cpp ---"
LLAMA_CLI="$HOME/Desktop/infer_train/llama.cpp-0.7.0/build/bin/llama-cli"
if [ -x "$LLAMA_CLI" ]; then
    echo "Running llama-cli..."
    echo "Testing decode with prompt 'Hello'..."
    /usr/bin/time -l "$LLAMA_CLI" -m "$MODEL" -p "Hello" -n 10 --temp 0 2>&1 | grep -E "eval time|tokens per"
else
    echo "llama-cli not found, skipping"
fi
