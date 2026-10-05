#!/bin/bash
# 编译 iq2_xxs_llama_dump.c 并链接 llama.cpp 源文件

set -e

LLAMA_DIR="/Users/jia/Desktop/infer_train/llama.cpp-0.5.0"
BUILD_DIR="$LLAMA_DIR/build-cpu"

# 编译必要的源文件
gcc -c -I"$LLAMA_DIR/ggml/src" -I"$LLAMA_DIR/ggml/src/ggml-cpu" -I"$LLAMA_DIR/ggml/include" -I"$BUILD_DIR/ggml/src" \
    "$LLAMA_DIR/ggml/src/ggml.c" \
    -o /tmp/ggml.o

gcc -c -I"$LLAMA_DIR/ggml/src" -I"$LLAMA_DIR/ggml/src/ggml-cpu" -I"$LLAMA_DIR/ggml/include" -I"$BUILD_DIR/ggml/src" \
    "$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.c" \
    -o /tmp/ggml-cpu.o

gcc -c -I"$LLAMA_DIR/ggml/src" -I"$LLAMA_DIR/ggml/src/ggml-cpu" -I"$LLAMA_DIR/ggml/include" -I"$BUILD_DIR/ggml/src" \
    "$LLAMA_DIR/ggml/src/ggml-quants.c" \
    -o /tmp/ggml-quants.o

# 编译测试程序
gcc -c -I"$LLAMA_DIR/ggml/src" -I"$LLAMA_DIR/ggml/src/ggml-cpu" -I"$LLAMA_DIR/ggml/include" -I"$BUILD_DIR/ggml/src" \
    "/Users/jia/Desktop/infer_train/tools/iq2_xxs_llama_dump.c" \
    -o /tmp/iq2_xxs_llama_dump.o

# 链接
gcc /tmp/iq2_xxs_llama_dump.o /tmp/ggml.o /tmp/ggml-cpu.o /tmp/ggml-quants.o \
    -lm -o /tmp/iq2_xxs_llama_dump

echo "编译成功：/tmp/iq2_xxs_llama_dump"