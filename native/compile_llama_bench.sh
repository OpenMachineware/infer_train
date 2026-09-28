#!/bin/bash
# Compile llama.cpp vec_dot benchmark with all dependencies

set -e

LLAMA_DIR="../llama.cpp-0.5.0"
GGML_SRC="$LLAMA_DIR/ggml/src"
GGML_CPU_SRC="$GGML_SRC/ggml-cpu"
GGML_ARM_SRC="$GGML_CPU_SRC/arch/arm"

# Common includes
INCLUDES="-I$LLAMA_DIR/ggml/include -I$GGML_SRC -I$GGML_CPU_SRC"

# Compiler flags
CFLAGS="-O3 -mcpu=apple-m1 -DGGML_USE_NEON -DGGML_USE_DOTPROD"

echo "Compiling llama.cpp common sources..."

# Compile ggml-quants.c (contains generic implementations)
clang -c $CFLAGS $INCLUDES -DGGML_COMMON_IMPL_C \
    $GGML_SRC/ggml-quants.c \
    -o ggml_quants.o

# Compile ARM quantization source
clang -c $CFLAGS $INCLUDES -DGGML_COMMON_IMPL_C \
    $GGML_ARM_SRC/quants.c \
    -o llama_quants.o

echo "Compiling benchmark..."
clang $CFLAGS $INCLUDES \
    bench_llama_vec_dot.c \
    ggml_quants.o \
    llama_quants.o \
    -o bench_llama_vec_dot \
    -lm

echo "Done! Run: ./bench_llama_vec_dot"
