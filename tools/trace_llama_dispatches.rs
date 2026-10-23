#!/bin/bash
# Trace llama.cpp Metal calls to count dispatches

cd /Users/jia/Desktop/infer_train/llama.cpp-0.7.0

# Use Metal validation to trace calls
METAL_DEVICE_WRAPPER_TYPE=1 ./build/bin/llama-cli -m ../Qwen3-0.6B-UD-Q4_K_XL.gguf -p "Hello" -n 1 2>&1 | grep -i "dispatch\|kernel" | wc -l