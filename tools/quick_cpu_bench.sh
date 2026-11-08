#!/bin/bash
# Quick CPU benchmark comparing our implementation vs llama.cpp

MODEL="$1"
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model.gguf>"
    exit 1
fi

echo "=== Testing Our Implementation ==="
echo "Running generate tool..."

# Time the execution
START=$(python3 -c "import time; print(time.time()*1000)")

# Run our tool with timeout to prevent hanging
timeout 60 ./target/release/generate "$MODEL" "The quick brown fox jumps over the lazy dog. This is a test of the CPU inference engine. We are measuring performance." 128 2>&1 | grep -E "Prefill|Decode|tokens" || echo "Generate timed out or failed"

END=$(python3 -c "import time; print(time.time()*1000)")

echo ""
echo "Total time: $(python3 -c "print(($END - $START)/1000)") seconds"
echo ""

echo "=== Testing llama.cpp ==="
~/Desktop/infer_train/llama.cpp-0.9.0/build_cpu/bin/llama-bench -m "$MODEL" -p 64 -n 128 -r 1 2>&1 | grep -E "pp64|tg128" || echo "llama-bench failed"
