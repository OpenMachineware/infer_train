#!/bin/bash
# Compile llama.cpp Q4_K benchmark - link against pre-built dylib

set -e

LLAMA_DIR="../llama.cpp-0.5.0"
BUILD_DIR="$LLAMA_DIR/build-cpu"
GGML_SRC="$LLAMA_DIR/ggml/src"
GGML_CPU_SRC="$GGML_SRC/ggml-cpu"

# Use pre-built dylib
DYLIB_DIR="$BUILD_DIR/bin"

if [ ! -d "$DYLIB_DIR" ]; then
    echo "Error: dylib directory not found at $DYLIB_DIR"
    exit 1
fi

INCLUDES="-I$LLAMA_DIR/ggml/include -I$GGML_SRC -I$GGML_CPU_SRC -I$GGML_SRC/ggml-cpu/arch/arm"
CFLAGS="-O3 -mcpu=apple-m1 -DGGML_USE_NEON -DGGML_USE_DOTPROD"

echo "Compiling benchmark..."
clang $CFLAGS $INCLUDES \
    bench_llama_q4k.c \
    -o bench_llama_q4k \
    -L$DYLIB_DIR \
    -lggml-cpu \
    -Wl,-rpath,@executable_path/../$DYLIB_DIR \
    -lm

echo "Done! Run: ./bench_llama_q4k"
