#!/bin/bash
# Build static library from llama.cpp vec_dot functions
# This allows direct comparison from Rust via FFI

set -e

LLAMA_DIR="../llama.cpp-0.5.0"
GGML_SRC="$LLAMA_DIR/ggml/src"
GGML_CPU_SRC="$GGML_SRC/ggml-cpu"
GGML_ARM_SRC="$GGML_CPU_SRC/arch/arm"

# Common includes
INCLUDES="-I$LLAMA_DIR/ggml/include -I$GGML_SRC -I$GGML_CPU_SRC -I$GGML_ARM_SRC"

# Compiler flags
CFLAGS="-O3 -mcpu=apple-m1 -DGGML_USE_NEON -DGGML_USE_DOTPROD"

echo "Compiling llama.cpp sources..."

# Compile ggml-quants.c
clang -c $CFLAGS $INCLUDES -DGGML_COMMON_IMPL_C \
    $GGML_SRC/ggml-quants.c \
    -o ggml_quants.o

# Compile ARM quantization source
clang -c $CFLAGS $INCLUDES -DGGML_COMMON_IMPL_C \
    $GGML_ARM_SRC/quants.c \
    -o llama_quants.o

# Compile vec.cpp (contains SILU, softmax, GELU tables)
clang -c $CFLAGS $INCLUDES -DGGML_COMMON_IMPL_C \
    -x c++ $GGML_CPU_SRC/vec.cpp \
    -o vec.o

# Compile wrapper
clang -c $CFLAGS $INCLUDES \
    llama_vec_dot_wrapper.c \
    -o llama_vec_dot_wrapper.o

# Create static library
ar rcs libllama_vecdot.a \
    ggml_quants.o \
    llama_quants.o \
    vec.o \
    llama_vec_dot_wrapper.o

echo "Done! Static library: libllama_vecdot.a"
